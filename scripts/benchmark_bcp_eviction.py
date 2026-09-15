#!/usr/bin/env python3
"""PLAN-CS-20260914-R2: paired, incremental BCP replay and serving validation.

Only this process's worker is instrumented. No tools from the trajectory are executed.
All generated data, logs, compilation caches and results belong outside the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIMIT = 131072


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def external(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Experiment outputs must be outside the repository")
    return path


def rolling_interface(messages, keep=12):
    tools = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
    drops = {str(event): [tools[n - keep]] for n, event in enumerate(tools) if n >= keep}
    return {"drop_message": drops, "reposition": [int(event) for event in drops]}


def select_cases(rows, count):
    # Input roots have explicit precedence; never choose a trial using benchmark performance.
    eligible = {}
    for row in rows:
        if row["full_tokens"] > LIMIT:
            eligible.setdefault(row["case_id"], row)
    selected = sorted(eligible.values(), key=lambda row: int(row["case_id"]))[:count]
    if len(selected) != count:
        raise ValueError(f"Need {count} distinct >128K cases; found {len(eligible)}")
    return selected


def common_turns(turns, max_tokens):
    result = []
    for turn in turns:
        if turn["full_tokens"] + max_tokens > LIMIT:
            break
        result.append(turn)
    return result


def prepare(args):
    from minisgl.benchmark.reposition_bcp import browsecomp_plus_tools
    from minisgl.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer

    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manager = TokenizeManager(
        AutoTokenizer.from_pretrained(args.model, local_files_only=True),
        radix_drop_key_mode="delta-marker",
    )
    tools = browsecomp_plus_tools()
    rows, provenance = [], []
    for source in args.source:
        for path in sorted(source.glob("shard*/trajectories.jsonl")):
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            provenance.append({"path": str(path), "sha256": sha})
            for line_number, line in enumerate(data.splitlines(), 1):
                row = json.loads(line)
                trajectory = row["trajectory"]
                ends = [
                    i for i, message in enumerate(trajectory) if message.get("role") == "assistant"
                ]
                if not ends:
                    continue
                tokens, _, _ = manager._render_harmony_message_drop(
                    trajectory[: ends[-1]], enable_thinking=None, tools=tools
                )
                rows.append(
                    {
                        "case_id": str(row["case_id"]),
                        "trajectory": trajectory,
                        "ends": ends,
                        "full_tokens": len(tokens),
                        "source": str(path),
                        "source_line": line_number,
                        "source_sha256": sha,
                    }
                )
            print(json.dumps({"scanned": str(path), "rows": len(rows)}), flush=True)
    selected = select_cases(rows, args.count)
    cases = []
    for row in selected:
        trajectory = row.pop("trajectory")
        turns = []
        for index, end in enumerate(row.pop("ends")):
            messages = trajectory[:end]
            tokens, owners, _ = manager._render_harmony_message_drop(
                messages, enable_thinking=None, tools=tools
            )
            schedule = rolling_interface(messages)
            dropped = {
                message for values in schedule["drop_message"].values() for message in values
            }
            turns.append(
                {
                    "turn": index,
                    "end": end,
                    "full_tokens": len(tokens),
                    "active_tokens_hint": sum(owner not in dropped for owner in owners),
                    "drop_events": len(schedule["reposition"]),
                    "messages_sha256": digest(messages),
                }
            )
        case_file = f"case-{row['case_id']}.json.gz"
        with gzip.open(output / case_file, "wt") as stream:
            json.dump(trajectory, stream, ensure_ascii=False)
        row.update(
            file=case_file,
            turns=turns,
            common_turn_count=len(common_turns(turns, args.max_tokens)),
            trajectory_sha256=digest(trajectory),
        )
        if not row["common_turn_count"]:
            raise ValueError(f"No native-length common turns in {row['case_id']}")
        cases.append(row)
        print(
            json.dumps(
                {
                    "prepared": row["case_id"],
                    "turns": len(turns),
                    "common_turns": row["common_turn_count"],
                    "full_tokens": row["full_tokens"],
                }
            ),
            flush=True,
        )
    write_json(
        output / "manifest.json",
        {
            "plan": "PLAN-CS-20260914-R2",
            "model": args.model,
            "max_tokens": args.max_tokens,
            "context_limit": LIMIT,
            "rolling_keep": 12,
            "tools": tools,
            "sources": provenance,
            "cases": cases,
        },
    )


def metric_values(metrics):
    count = metrics["generated_tokens"]
    start, first, last = (
        metrics[key]
        for key in ("request_received_ns", "first_token_generated_ns", "request_finished_ns")
    )
    if count < 1 or not 0 <= start <= first <= last:
        raise ValueError("Invalid server generation metrics")
    return {
        "ttft_s": (first - start) / 1e9,
        "tpot_s": (last - first) / 1e9 / (count - 1) if count > 1 else None,
    }


def pressure_turn_ends(trajectory):
    """First recorded assistant query after each of TR12, 13, 14 and 15."""
    result, tools = {}, 0
    for index, message in enumerate(trajectory):
        if message.get("role") == "tool":
            tools += 1
        elif message.get("role") == "assistant" and 12 <= tools <= 15:
            result.setdefault(tools, index)
    if set(result) != {12, 13, 14, 15}:
        raise ValueError("Need assistant queries immediately after TR12 through TR15")
    return [result[n] for n in range(12, 16)]


def prepare_pressure(args):
    from minisgl.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer

    source = json.loads((args.input / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manager = TokenizeManager(tokenizer, radix_drop_key_mode="delta-marker")
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for original in source["cases"]:
        with gzip.open(args.input / original["file"], "rt") as stream:
            trajectory = json.load(stream)
        if digest(trajectory) != original["trajectory_sha256"]:
            raise ValueError("Source trajectory hash mismatch")
        try:
            ends = pressure_turn_ends(trajectory)
        except ValueError:
            continue
        trajectory = trajectory[:ends[-1]]
        # Preserve original ordering and tool-call identifiers. Only document and
        # reasoning text are shortened; this is explicitly a derived short workload.
        for message in trajectory:
            fields = [("content", args.tool_tokens)] if message["role"] == "tool" else []
            if message["role"] == "assistant":
                fields += [("reasoning_content", 192), ("content", 192)]
            for field, cap in fields:
                value = message.get(field)
                if isinstance(value, str):
                    ids = tokenizer.encode(value, add_special_tokens=False)
                    message[field] = tokenizer.decode(ids[:cap])
        turns = []
        for tr, end in zip(range(12, 16), ends, strict=True):
            messages = trajectory[:end]
            tokens, _, _ = manager._render_harmony_message_drop(
                messages, enable_thinking=None, tools=source["tools"]
            )
            if len(tokens) + 64 > 32768:
                raise ValueError("Derived pressure context exceeds 32K")
            turns.append(dict(turn=tr, end=end, full_tokens=len(tokens),
                              messages_sha256=digest(messages)))
        case = dict(case_id=original["case_id"], file=original["file"], turns=turns,
                    trajectory_sha256=digest(trajectory),
                    source_trajectory_sha256=original["trajectory_sha256"],
                    source_full_tokens=original["full_tokens"])
        with gzip.open(output / case["file"], "wt") as stream:
            json.dump(trajectory, stream, ensure_ascii=False)
        cases.append(case)
        if len(cases) == 8:
            break
    if len(cases) != 8:
        raise ValueError("Need eight distinct source conversations")
    manifest = dict(plan="PLAN-CS-20260916-R2", pressure=True, fixed_output=True,
                    max_tokens=64, model=args.model, tools=source["tools"],
                    source_manifest_sha256=digest(source), tool_tokens=args.tool_tokens,
                    assistant_text_tokens=192, rolling_keep=12, cases=cases)
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"cases": [c["case_id"] for c in cases],
                      "lengths": [[t["full_tokens"] for t in c["turns"]] for c in cases]}))


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": values[min(len(values) - 1, int(0.95 * (len(values) - 1)))],
    }


def summarize(records, elapsed, expected):
    passed = [r for r in records if not r.get("error")]
    generated = sum(r["metrics"]["generated_tokens"] for r in passed)
    return {
        "expected_turns": expected,
        "completed_turns": len(passed),
        "failed_turns": sum(bool(r.get("error")) for r in records),
        "uncompleted_turns": expected - len(passed),
        "elapsed_s": elapsed,
        "generated_tokens": generated,
        "output_tokens_per_s": generated / elapsed,
        "turns_per_s": len(passed) / elapsed,
        "ttft_s": distribution([r["ttft_s"] for r in passed]),
        "tpot_s": distribution([r["tpot_s"] for r in passed]),
        "over_128k_turns": sum(r["metrics"]["prompt_tokens"] > LIMIT for r in passed),
        "over_128k_ttft_s": distribution(
            [r["ttft_s"] for r in passed if r["metrics"]["prompt_tokens"] > LIMIT]
        ),
        "over_128k_tpot_s": distribution(
            [r["tpot_s"] for r in passed if r["metrics"]["prompt_tokens"] > LIMIT]
        ),
        "length_capped_turns": sum(r["finish_reason"] == "length" for r in passed),
    }


def matrix_cells(stress_count=16):
    cells = []
    # Largest concurrency first fixes a conservative page capacity for each TP.
    for tp in (2, 4):
        if tp == 2:
            for eviction in ("ordinary", "drop-aware"):
                cells.append(
                    {
                        "tp": tp,
                        "concurrency": 8,
                        "count": stress_count,
                        "workload": "rolling",
                        "phase": "full",
                        "eviction": eviction,
                        "suite": f"stress{stress_count}",
                    }
                )
        for concurrency in (8, 4, 2, 1):
            for workload in ("no_drop", "rolling"):
                for eviction in ("ordinary", "drop-aware"):
                    cells.append(
                        {
                            "tp": tp,
                            "concurrency": concurrency,
                            "count": concurrency,
                            "workload": workload,
                            "phase": "common",
                            "eviction": eviction,
                            "suite": "scaling",
                        }
                    )
            for eviction in ("ordinary", "drop-aware"):
                cells.append(
                    {
                        "tp": tp,
                        "concurrency": concurrency,
                        "count": concurrency,
                        "workload": "rolling",
                        "phase": "full",
                        "eviction": eviction,
                        "suite": "scaling",
                    }
                )
    return cells


def cell_name(cell):
    return (
        f"{cell['suite']}-tp{cell['tp']}-c{cell['concurrency']}-"
        f"{cell['workload']}-{cell['phase']}-{cell['eviction']}"
    )


def install_observers():
    """CPU counters during serving; collective page audits only when the client is idle."""
    import torch
    import torch.distributed as dist
    from minisgl.engine.engine import Engine
    from minisgl.kvcache.radix_cache import RadixPrefixCache
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.prefill import PrefillAdder
    from minisgl.scheduler.cache import CacheManager
    from minisgl.core import Req

    root = Path(os.environ["MINISGL_BCP_OBSERVER"])
    census = Counter()
    batches = []
    pressure = Counter()
    committed = []
    live_locks = 0
    live_drop_calls = 0
    evicted_pages = 0
    seen = set()
    original_init, original_forward = Engine.__init__, Engine.forward_batch
    original_idle = Scheduler.run_when_idle
    original_lock, original_evict = RadixPrefixCache.lock_handle, RadixPrefixCache.evict
    original_allocate = PrefillAdder._try_allocate_one
    original_empty = CacheManager.match_empty_req
    original_release = Scheduler._release_occurrence_transients
    original_append = Req.append_host

    def append(req, token):
        original_append(req, token)
        committed.append((time.perf_counter_ns(), req.uid, token.tolist()))

    def allocate(self, *args, **kwargs):
        table_available = self.table_manager.available_size > 0
        result = original_allocate(self, *args, **kwargs)
        if result is None and table_available:
            pressure["capacity_waits"] += 1
        return result

    def empty(self, *args, **kwargs):
        pressure["empty_matches"] += 1
        return original_empty(self, *args, **kwargs)

    def release(self, req):
        mask = req.occurrence_birth_owned_mask
        before = int(mask.sum()) if mask is not None else 0
        result = original_release(self, req)
        after = int(mask.sum()) if mask is not None else 0
        pressure["early_birth_releases"] += before - after
        return result

    def nodes(tree):
        stack = list(tree.root_node.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        write_json(
            root / f"ready-{os.getpid()}.json",
            {"pid": os.getpid(), "num_pages": self.num_pages, "max_seq_len": self.max_seq_len},
        )

    def forward(self, batch, sampling):
        key = f"{batch.phase}|bs={batch.size}|graph={self.graph_runner.can_use_cuda_graph(batch)}"
        batches.append((time.perf_counter_ns(), key))
        return original_forward(self, batch, sampling)

    def lock(self, handle, unlock=False):
        nonlocal live_locks
        result = original_lock(self, handle, unlock)
        live_locks += -1 if unlock else 1
        return result

    def evict(self, size):
        nonlocal live_drop_calls, evicted_pages
        before = self.eviction_stats["drop_pages"]
        result = original_evict(self, size)
        evicted_pages += len(result)
        if live_locks > 0 and self.eviction_stats["drop_pages"] > before:
            live_drop_calls += 1
        return result

    def idle(self):
        nonlocal live_drop_calls, evicted_pages
        original_idle(self)
        group = self.engine.tp_cpu_group
        command = [None]
        if dist.get_rank(group) == 0:
            marker = root / "audit-command.json"
            if marker.exists():
                command[0] = json.loads(marker.read_text())
        dist.broadcast_object_list(command, src=0, group=group)
        if command[0] is None or command[0]["id"] in seen:
            return
        spec = command[0]
        seen.add(spec["id"])
        census.clear()
        census.update(
            key
            for timestamp, key in batches
            if spec.get("start_ns", 0) <= timestamp <= spec.get("end_ns", 2**63 - 1)
        )
        cache, tree = self.cache_manager, self.cache_manager.prefix_cache
        result = {
            "command": spec["id"],
            "pid": os.getpid(),
            "census": dict(census),
            "eviction": dict(tree.eviction_stats),
            "evicted_pages": evicted_pages,
            "live_lock_drop_calls": live_drop_calls,
            "live_locks": live_locks,
            "free": len(cache.free_slots),
            "available": cache.available_size,
            "total": cache.num_pages,
            "pressure": dict(pressure),
            "committed": [(uid, tokens) for stamp, uid, tokens in committed
                          if spec.get("start_ns", 0) <= stamp <= spec.get("end_ns", 2**63 - 1)],
        }
        try:
            assert live_locks == 0, f"Unbalanced live handles: {live_locks}"
            resident = []
            for node in nodes(tree):
                assert node.ref_count == 0 and node.path_ref_count == 0
                resident.append(node.value[node.value >= 0])
            free = cache.free_slots
            all_pages = torch.cat([free, *resident])
            assert bool(torch.all((all_pages >= 0) & (all_pages < cache.num_pages)))
            assert len(torch.unique(free)) == len(free)
            if resident:
                assert not bool(torch.isin(free, torch.cat(resident)).any())
            assert len(torch.unique(all_pages)) == cache.num_pages
            assert cache.available_size == cache.num_pages
            if spec.get("drain", True):
                pages = cache._allocate(cache.num_pages)
                assert len(torch.unique(pages)) == cache.num_pages
                assert not any(node.page_length for node in nodes(tree))
                cache.free_occurrence_pages(pages)
            result.update(passed=True, free_after_drain=len(cache.free_slots))
        except Exception as exc:
            result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        write_json(root / f"audit-{spec['id']}-{os.getpid()}.json", result)
        if spec.get("reset_counters"):
            census.clear()
            batches.clear()
            pressure.clear()
            committed.clear()
            tree.eviction_stats.update(leaf_pages=0, drop_pages=0, hole_fills=0)
            live_drop_calls = evicted_pages = 0

    Engine.__init__, Engine.forward_batch = init, forward
    RadixPrefixCache.lock_handle, RadixPrefixCache.evict = lock, evict
    Scheduler.run_when_idle = idle
    PrefillAdder._try_allocate_one = allocate
    CacheManager.match_empty_req = empty
    Scheduler._release_occurrence_transients = release
    Req.append_host = append


async def audit(client, url, root, tp, label, model, **window):
    write_json(
        root / "audit-command.json",
        {"id": label, "reset_counters": label.startswith("initial-"), **window},
    )
    wake = await client.post(
        url + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": f"Reply OK. Audit {label}."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        },
    )
    wake.raise_for_status()
    deadline = time.monotonic() + 180
    while len(list(root.glob(f"audit-{label}-*.json"))) != tp:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Not all {tp} ranks acknowledged audit {label}")
        await asyncio.sleep(0.2)
    results = [json.loads(p.read_text()) for p in sorted(root.glob(f"audit-{label}-*.json"))]
    if not all(row["passed"] for row in results):
        raise RuntimeError(f"Page audit failed: {results}")
    return results


def launch(args, cell, root, pages):
    source_repo = Path(getattr(args, "source_repo", None) or REPO).resolve()
    gpus = args.gpus.split(",")[: cell["tp"]]
    if len(gpus) != cell["tp"]:
        raise ValueError("Insufficient GPUs for TP")
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    usage = {i.strip(): int(mem) for i, mem in (line.split(",") for line in used.splitlines())}
    if any(usage[g] > 100 for g in gpus):
        raise RuntimeError(f"Selected GPUs busy: {usage}")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(gpus),
        PYTHONPATH=str(source_repo / "python"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        MINISGL_BCP_OBSERVER=str(root),
    )
    runtime = external(getattr(args, "runtime_root", None) or
                       (external(args.output) / f"runtime-tp{cell['tp']}"))
    for key in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH"):
        directory = runtime / key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    for key, suffix in (("CC", "gcc"), ("CXX", "g++")):
        compiler = Path(sys.executable).parent / f"x86_64-conda-linux-gnu-{suffix}"
        if compiler.exists():
            env[key] = str(compiler)
    if "CXX" in env:
        # nvcc does not honor CXX for its host compiler when invoked by TVM FFI.
        env["NVCC_PREPEND_FLAGS"] = f"-ccbin={env['CXX']}"
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        str(cell["tp"]),
        "--dtype",
        "bfloat16",
        "--disable-pynccl",
        "--memory-ratio",
        str(args.memory_ratio),
        "--max-running-requests",
        str(cell["concurrency"]),
        "--cuda-graph-max-bs",
        str(cell["concurrency"]),
        "--max-seq-len-override",
        str(getattr(args, "context_limit", LIMIT)),
        "--max-prefill-length",
        str(args.chunk),
        "--request-timeout",
        str(args.timeout),
        "--cache-type",
        "radix",
        "--page-size",
        "1",
        "--attention-backend",
        "fi",
        "--radix-drop-key-mode",
        "delta-marker",
        "--contextual-prefill-mode",
        "mask",
        "--reposition-execution-mode",
        "paged-occurrence",
        "--tool-call-parser",
        "gpt-oss",
        "--reasoning-parser",
        "gpt-oss",
    ]
    if cell["eviction"] == "drop-aware":
        argv.append("--drop-aware-eviction")
    if pages:
        argv.extend(["--num-pages", str(pages)])
    with (root / "server.log").open("xb") as stream:
        child = subprocess.Popen(
            argv,
            env=env,
            cwd=source_repo,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=stream,
            start_new_session=True,
        )
    write_json(
        root / "launch.json",
        {
            "argv": argv,
            "gpus": gpus,
            "pid": child.pid,
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_repo, text=True
            ).strip(),
            "cell": cell,
            "requested_pages": pages,
            "compiler_env": {key: env.get(key) for key in ("CC", "CXX", "NVCC_PREPEND_FLAGS")},
        },
    )
    return child


def stop(child):
    # Only this harness's private process group; workers may outlive an exited API parent.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def replay(args, cell, manifest, cases, root, client, url):
    semaphore = asyncio.Semaphore(cell["concurrency"])
    records = []
    expected = sum(len(case["selected_turns"]) for case in cases)
    # Buffered append preserves completed evidence after a worker/request failure.
    with gzip.open(root / "turns.jsonl.gz", "wt") as output:

        async def conversation(case):
            async with semaphore:
                for turn in case["selected_turns"]:
                    messages = case["trajectory"][: turn["end"]]
                    payload = {
                        "model": args.model,
                        "messages": messages,
                        "tools": manifest["tools"],
                        "max_tokens": manifest["max_tokens"],
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 17,
                        "stream": False,
                    }
                    if manifest.get("fixed_output"):
                        payload["ignore_eos"] = True
                    if cell["workload"] == "rolling":
                        payload.update(rolling_interface(messages))
                    row = {
                        "case_id": case["case_id"],
                        "turn": turn["turn"],
                        "messages_sha256": turn["messages_sha256"],
                        "full_tokens_prepared": turn["full_tokens"],
                        "request_start_s": time.perf_counter(),
                    }
                    response = None
                    try:
                        response = await client.post(url + "/v1/chat/completions", json=payload)
                        row["status_code"] = response.status_code
                        response.raise_for_status()
                        body = response.json()
                        row["response"] = body
                        metrics = body["server_metrics"]
                        row.update(
                            metrics=metrics,
                            **metric_values(metrics),
                            finish_reason=body["choices"][0]["finish_reason"],
                        )
                        if metrics["prompt_tokens"] != turn["full_tokens"]:
                            raise ValueError(
                                f"Full-token provenance mismatch: {metrics['prompt_tokens']} "
                                f"!= {turn['full_tokens']}"
                            )
                        if metrics["generated_tokens"] > manifest["max_tokens"]:
                            raise ValueError("Output cap exceeded")
                        # Existing overlap scheduling may reach its device limit
                        # one forward before the final host commit. Keep that
                        # baseline behavior and measure actual committed output.
                        if manifest.get("fixed_output") and (
                            row["finish_reason"] != "length"
                            or metrics["generated_tokens"] not in (
                                manifest["max_tokens"] - 1, manifest["max_tokens"]
                            )
                        ):
                            raise ValueError("Length-limited output not reached")
                        if cell["workload"] == "no_drop" and metrics["drop_skipped_tokens"]:
                            raise ValueError("no_drop unexpectedly reports skipped Drop tokens")
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        if response is not None and response.status_code != 200:
                            row["error_body"] = response.text[:8000]
                    row["request_end_s"] = time.perf_counter()
                    records.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                "cell": cell_name(cell),
                                "case": case["case_id"],
                                "turn": turn["turn"],
                                "error": row.get("error"),
                                "tokens": row.get("metrics", {}).get("generated_tokens"),
                                "completed": len(records),
                                "expected": expected,
                            }
                        ),
                        flush=True,
                    )
                    if row.get("error"):
                        break

        start = time.perf_counter_ns()
        await asyncio.gather(*(conversation(case) for case in cases))
        end = time.perf_counter_ns()
    return dict(summarize(records, (end - start) / 1e9, expected), start_ns=start, end_ns=end)


async def run_cell(args, cell, manifest, pages, session):
    import httpx

    root = external(args.output) / cell_name(cell)
    if (root / "summary.json").exists():
        return json.loads((root / "summary.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in manifest["cases"][: cell["count"]]:
        with gzip.open(args.input / case["file"], "rt") as stream:
            trajectory = json.load(stream)
        if digest(trajectory) != case["trajectory_sha256"]:
            raise ValueError("Trajectory manifest hash mismatch")
        turns = (
            common_turns(case["turns"], manifest["max_tokens"])
            if cell["phase"] == "common"
            else case["turns"]
        )
        if args.turn_limit:
            turns = turns[: args.turn_limit]
        cases.append(dict(case, trajectory=trajectory, selected_turns=turns))
    summary = {
        "cell": cell,
        "name": cell_name(cell),
        "status": "FAIL",
        "case_ids": [case["case_id"] for case in cases],
        "smoke": bool(args.turn_limit),
    }
    try:
        if session.get("child") is None:
            server_root = (
                external(args.output)
                / "servers"
                / (f"tp{cell['tp']}-{cell['eviction']}-{time.time_ns()}")
            )
            server_root.mkdir(parents=True)
            session["root"] = server_root
            session["child"] = launch(
                args, dict(cell, concurrency=session["max_concurrency"]), server_root, pages
            )
        child, server_root = session["child"], session["root"]
        write_json(
            root / "launch.json",
            {
                **json.loads((server_root / "launch.json").read_text()),
                "client_cell": cell,
                "server_directory": str(server_root),
            },
        )
        url = f"http://127.0.0.1:{args.port}"
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {child.returncode}")
                try:
                    response = await client.get(url + "/v1/models", timeout=5)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup deadline exceeded")
                await asyncio.sleep(2)
            ready = [json.loads(path.read_text()) for path in server_root.glob("ready-*.json")]
            if len(ready) != cell["tp"] or len({r["num_pages"] for r in ready}) != 1:
                raise RuntimeError(f"Inconsistent TP initialization: {ready}")
            summary["num_pages"] = ready[0]["num_pages"]
            summary["initial_audit"] = await audit(
                client, url, server_root, cell["tp"], "initial-" + cell_name(cell), args.model
            )
            if manifest.get("pressure"):
                # Cold C1 output control, then TR12 seeding; neither is throughput.
                control_root = root / "control"
                control_root.mkdir()
                control = dict(cases[0], selected_turns=cases[0]["selected_turns"][-1:])
                summary["control"] = await asyncio.wait_for(
                    replay(args, dict(cell, concurrency=1), manifest, [control],
                           control_root, client, url), args.cell_timeout)
                if summary["control"]["uncompleted_turns"]:
                    raise RuntimeError("C1 control failed")
                summary["control_audit"] = await audit(
                    client, url, server_root, cell["tp"],
                    "initial-seed-" + cell_name(cell), args.model,
                    start_ns=summary["control"]["start_ns"],
                    end_ns=summary["control"]["end_ns"])
                seed_root = root / "seed"
                seed_root.mkdir()
                seeds = [dict(c, selected_turns=c["selected_turns"][:1]) for c in cases]
                summary["seed"] = await asyncio.wait_for(
                    replay(args, cell, manifest, seeds, seed_root, client, url), args.cell_timeout)
                if summary["seed"]["uncompleted_turns"]:
                    raise RuntimeError("TR12 seed failed")
                await audit(client, url, server_root, cell["tp"],
                            "initial-measure-" + cell_name(cell), args.model, drain=False)
                cases = [dict(c, selected_turns=c["selected_turns"][1:]) for c in cases]
                summary.update(await asyncio.wait_for(
                    replay(args, cell, manifest, cases, root, client, url), args.cell_timeout))
            else:
                summary.update(await replay(args, cell, manifest, cases, root, client, url))
            summary["final_audit"] = await audit(
                client,
                url,
                server_root,
                cell["tp"],
                "final-" + cell_name(cell),
                args.model,
                start_ns=summary["start_ns"],
                end_ns=summary["end_ns"],
            )
            health = await client.post(
                url + "/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "After reclamation, say healthy."}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": False,
                },
            )
            health.raise_for_status()
            summary["post_drain_health"] = health.json()
            if not summary["uncompleted_turns"]:
                summary["status"] = "PASS"
            summary["drop_eviction_covered"] = any(
                rank["eviction"]["drop_pages"] > 0 for rank in summary["final_audit"]
            )
            summary["capacity_wait_covered"] = any(
                rank["pressure"].get("capacity_waits", 0) > 0
                for rank in summary["final_audit"]
            )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if summary["status"] != "PASS" and "final_audit" not in summary and session.get("child"):
            stop(session.pop("child"))
            await asyncio.sleep(5)
        write_json(root / "summary.json", summary)
    return summary


async def run(args):
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.input / "manifest.json").read_text())
    if args.model != manifest["model"]:
        raise ValueError("Use the model/tokenizer used to prepare the manifest")
    cells = matrix_cells(args.stress_count)
    if args.cell:
        names = set(args.cell)
        cells = [cell for cell in cells if cell_name(cell) in names]
        if len(cells) != len(names):
            raise ValueError("Unknown cell name; use list-cells")
    if len(manifest["cases"]) < max(cell["count"] for cell in cells):
        raise ValueError("Input manifest has fewer cases than the requested matrix")
    fingerprint = {
        "manifest_sha256": digest(manifest),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        **{
            key: getattr(args, key)
            for key in (
                "model", "gpus", "chunk", "memory_ratio", "pages", "turn_limit", "timeout",
                "stress_count",
            )
        },
    }
    run_path = output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Resume configuration differs; use a separate output directory")
    write_json(
        run_path,
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "fingerprint": fingerprint,
            "cells": cells,
        },
    )
    capacity_path = output / "capacity.json"
    capacity = json.loads(capacity_path.read_text()) if capacity_path.exists() else {}
    session, previous = {}, None
    try:
        # Stable grouping reuses model/tokenizer/graphs, with a verified empty cache per cell.
        for cell in sorted(cells, key=lambda c: (c["tp"], c["eviction"] != "ordinary")):
            key = (cell["tp"], cell["eviction"])
            if key != previous:
                if session.get("child"):
                    stop(session["child"])
                    await asyncio.sleep(5)
                session = {
                    "max_concurrency": max(
                        c["concurrency"] for c in cells if (c["tp"], c["eviction"]) == key
                    )
                }
                previous = key
            pages = args.pages or capacity.get(str(cell["tp"]))
            summary = await run_cell(args, cell, manifest, pages, session)
            if "num_pages" in summary:
                capacity.setdefault(str(cell["tp"]), summary["num_pages"])
                write_json(capacity_path, capacity)
            print(
                json.dumps(
                    {
                        "finished": cell_name(cell),
                        "status": summary["status"],
                        "tokens_per_s": summary.get("output_tokens_per_s"),
                        "error": summary.get("error"),
                    }
                ),
                flush=True,
            )
            report(args)
    finally:
        if session.get("child"):
            stop(session["child"])
    report(args)


async def run_pressure(args):
    """Six bounded cells with the same client and observer on both source trees."""
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.input / "manifest.json").read_text())
    if not manifest.get("pressure") or len(manifest["cases"]) != 8:
        raise ValueError("Use prepare-pressure's eight-conversation manifest")
    args.turn_limit = None
    args.context_limit = 32768
    results, session = [], {}
    write_json(output / "run.json", dict(
        manifest_sha256=digest(manifest), total_timeout_s=args.total_timeout,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    try:
        async with asyncio.timeout(args.total_timeout):
            for variant, source in (("before", args.before_repo), ("after", REPO)):
                args.source_repo = source
                for eviction in ("ordinary", "drop-aware"):
                    session = {"max_concurrency": 8}
                    workloads = ("no_drop", "rolling") if eviction == "ordinary" else ("rolling",)
                    for workload in workloads:
                        cell = dict(tp=2, concurrency=8, count=8, workload=workload,
                                    phase="full", eviction=eviction, suite="pressure-" + variant)
                        result = await run_cell(args, cell, manifest, args.pages, session)
                        results.append(result)
                        write_json(output / "results.json", results)
                        print(json.dumps({"cell": cell_name(cell), "status": result["status"],
                                          "tokens_per_s": result.get("output_tokens_per_s"),
                                          "error": result.get("error")}), flush=True)
                    if session.get("child"):
                        stop(session.pop("child"))
                        await asyncio.sleep(5)
    except TimeoutError:
        write_json(output / "deadline.json", {"status": "TOTAL_DEADLINE", "seconds": args.total_timeout})
    finally:
        if session.get("child"):
            stop(session.pop("child"))
    comparisons = []
    for workload, eviction in (("no_drop", "ordinary"), ("rolling", "ordinary"),
                               ("rolling", "drop-aware")):
        pair = [r for r in results if r["cell"]["workload"] == workload
                and r["cell"]["eviction"] == eviction]
        complete = len(pair) == 2 and all(r["status"] == "PASS" for r in pair)
        row = dict(workload=workload, eviction=eviction, complete=complete)
        if complete:
            row["throughput_ratio_after_before"] = pair[1]["output_tokens_per_s"] / pair[0]["output_tokens_per_s"]
            controls = [[token for _, tokens in r["control_audit"][0]["committed"]
                         for token in tokens] for r in pair]
            row["cold_c1_token_counts"] = [len(tokens) for tokens in controls]
            row["cold_c1_output_equal"] = controls[0] == controls[1] and len(controls[0]) in (63, 64)
        comparisons.append(row)
    write_json(output / "comparison.json", comparisons)


def report(args):
    output = external(args.output)
    summaries = [json.loads(p.read_text()) for p in sorted(output.glob("*/summary.json"))]
    rows = []
    for summary in summaries:
        rows.append(
            {
                **summary["cell"],
                "status": summary["status"],
                "smoke": summary["smoke"],
                "name": summary["name"],
                "ttft_mean_s": summary.get("ttft_s", {}).get("mean"),
                "tpot_mean_s": summary.get("tpot_s", {}).get("mean"),
                **{
                    key: summary.get(key)
                    for key in (
                        "output_tokens_per_s",
                        "elapsed_s",
                        "completed_turns",
                        "uncompleted_turns",
                        "over_128k_turns",
                        "num_pages",
                        "drop_eviction_covered",
                        "error",
                    )
                },
            }
        )
    write_json(output / "results.json", summaries)
    if rows:
        with (output / "results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pairs = []
    for row in rows:
        if row["eviction"] != "drop-aware":
            continue
        baseline = next(
            (r for r in rows if r["name"] == row["name"].replace("drop-aware", "ordinary")), None
        )
        valid = baseline and all(r["status"] == "PASS" and not r["smoke"] for r in (row, baseline))
        pairs.append(
            {
                "candidate": row["name"],
                "baseline": baseline["name"] if baseline else None,
                "valid_comparison": bool(valid),
                "throughput_ratio": row["output_tokens_per_s"] / baseline["output_tokens_per_s"]
                if valid
                else None,
            }
        )
    write_json(output / "comparisons.json", pairs)
    if not getattr(args, "plot", False):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for tp in (2, 4):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for workload in ("no_drop", "rolling"):
            for eviction in ("ordinary", "drop-aware"):
                series = sorted(
                    [
                        r
                        for r in rows
                        if r["tp"] == tp
                        and r["suite"] == "scaling"
                        and r["phase"] == "common"
                        and r["workload"] == workload
                        and r["eviction"] == eviction
                        and r["status"] == "PASS"
                        and not r["smoke"]
                    ],
                    key=lambda r: r["concurrency"],
                )
                for axis, field, title in zip(
                    axes,
                    ("ttft_mean_s", "tpot_mean_s", "output_tokens_per_s"),
                    ("TTFT (s)", "TPOT (s)", "Output tokens/s"),
                ):
                    axis.plot(
                        [r["concurrency"] for r in series],
                        [r[field] for r in series],
                        marker="o",
                        label=f"{workload} / {eviction}",
                    )
                    axis.set(xlabel="Client concurrency", ylabel=title, xticks=[1, 2, 4, 8])
                    axis.grid(alpha=0.3)
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"GPT-OSS-120B TP={tp}: identical native-length turns")
        fig.tight_layout()
        fig.savefig(output / f"scaling-tp{tp}.png", dpi=180)
        plt.close(fig)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server

        launch_server()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, action="append", required=True)
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--count", type=int, default=32)
    prepare_parser.add_argument("--max-tokens", type=int, default=4096)
    short_prepare = commands.add_parser("prepare-pressure")
    short_prepare.add_argument("--input", type=Path, required=True)
    short_prepare.add_argument("--output", type=Path, required=True)
    short_prepare.add_argument("--model", required=True)
    short_prepare.add_argument("--tool-tokens", type=int, default=512)
    pressure_parser = commands.add_parser("run-pressure")
    pressure_parser.add_argument("--input", type=Path, required=True)
    pressure_parser.add_argument("--output", type=Path, required=True)
    pressure_parser.add_argument("--before-repo", type=Path, required=True)
    pressure_parser.add_argument("--runtime-root", type=Path)
    pressure_parser.add_argument("--model", required=True)
    pressure_parser.add_argument("--gpus", default="2,3")
    pressure_parser.add_argument("--port", type=int, default=30916)
    pressure_parser.add_argument("--chunk", type=int, default=2048)
    pressure_parser.add_argument("--memory-ratio", type=float, default=0.9)
    pressure_parser.add_argument("--pages", type=int, default=49152)
    pressure_parser.add_argument("--timeout", type=int, default=300)
    pressure_parser.add_argument("--cell-timeout", type=int, default=300)
    pressure_parser.add_argument("--startup-timeout", type=int, default=600)
    pressure_parser.add_argument("--total-timeout", type=int, default=3600)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--gpus", default="0,1,2,3")
    run_parser.add_argument("--port", type=int, default=30914)
    run_parser.add_argument("--chunk", type=int, default=16384)
    run_parser.add_argument("--memory-ratio", type=float, default=0.9)
    run_parser.add_argument("--pages", type=int)
    run_parser.add_argument("--stress-count", type=int, choices=(16, 32), default=16)
    run_parser.add_argument("--cell", action="append")
    run_parser.add_argument(
        "--turn-limit", type=int, help="Smoke only; excluded from throughput comparisons"
    )
    run_parser.add_argument("--timeout", type=int, default=7200)
    run_parser.add_argument("--startup-timeout", type=int, default=1800)
    list_parser = commands.add_parser("list-cells")
    list_parser.add_argument("--stress-count", type=int, choices=(16, 32), default=16)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "prepare-pressure":
        prepare_pressure(args)
    elif args.command == "run-pressure":
        asyncio.run(run_pressure(args))
    elif args.command == "run":
        asyncio.run(run(args))
    elif args.command == "report":
        report(args)
    else:
        for cell in matrix_cells(args.stress_count):
            print(cell_name(cell))


if os.environ.get("MINISGL_BCP_OBSERVER"):
    install_observers()
if __name__ == "__main__":
    main()
