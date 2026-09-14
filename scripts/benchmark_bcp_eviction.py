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


def matrix_cells():
    cells = []
    # Largest concurrency first fixes a conservative page capacity for each TP.
    for tp in (2, 4):
        if tp == 2:
            for eviction in ("ordinary", "drop-aware"):
                cells.append(
                    {
                        "tp": tp,
                        "concurrency": 8,
                        "count": 32,
                        "workload": "rolling",
                        "phase": "full",
                        "eviction": eviction,
                        "suite": "stress32",
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

    root = Path(os.environ["MINISGL_BCP_OBSERVER"])
    census = Counter()
    batches = []
    live_locks = 0
    live_drop_calls = 0
    evicted_pages = 0
    seen = set()
    original_init, original_forward = Engine.__init__, Engine.forward_batch
    original_idle = Scheduler.run_when_idle
    original_lock, original_evict = RadixPrefixCache.lock_handle, RadixPrefixCache.evict

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
            tree.eviction_stats.update(leaf_pages=0, drop_pages=0, hole_fills=0)
            live_drop_calls = evicted_pages = 0

    Engine.__init__, Engine.forward_batch = init, forward
    RadixPrefixCache.lock_handle, RadixPrefixCache.evict = lock, evict
    Scheduler.run_when_idle = idle


async def audit(client, url, root, tp, label, model, **window):
    write_json(
        root / "audit-command.json", {"id": label, "reset_counters": label == "initial", **window}
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
        PYTHONPATH=str(REPO / "python"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        MINISGL_BCP_OBSERVER=str(root),
    )
    runtime = external(args.output) / f"runtime-tp{cell['tp']}"
    for key in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH"):
        directory = runtime / key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    for key, suffix in (("CC", "gcc"), ("CXX", "g++")):
        compiler = Path(sys.executable).parent / f"x86_64-conda-linux-gnu-{suffix}"
        if compiler.exists():
            env[key] = str(compiler)
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
        str(LIMIT),
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
            cwd=REPO,
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
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "cell": cell,
            "requested_pages": pages,
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


async def run_cell(args, cell, manifest, pages):
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
    child = None
    try:
        child = launch(args, cell, root, pages)
        url = f"http://127.0.0.1:{args.port}"
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {child.returncode}")
                try:
                    response = await client.get(url + "/health", timeout=5)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup deadline exceeded")
                await asyncio.sleep(2)
            ready = [json.loads(path.read_text()) for path in root.glob("ready-*.json")]
            if len(ready) != cell["tp"] or len({r["num_pages"] for r in ready}) != 1:
                raise RuntimeError(f"Inconsistent TP initialization: {ready}")
            summary["num_pages"] = ready[0]["num_pages"]
            summary["initial_audit"] = await audit(
                client, url, root, cell["tp"], "initial", args.model
            )
            summary.update(await replay(args, cell, manifest, cases, root, client, url))
            summary["final_audit"] = await audit(
                client,
                url,
                root,
                cell["tp"],
                "final",
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
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if child is not None:
            stop(child)
        write_json(root / "summary.json", summary)
    return summary


async def run(args):
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.input / "manifest.json").read_text())
    if args.model != manifest["model"]:
        raise ValueError("Use the model/tokenizer used to prepare the manifest")
    cells = matrix_cells()
    if args.cell:
        names = set(args.cell)
        cells = [cell for cell in cells if cell_name(cell) in names]
        if len(cells) != len(names):
            raise ValueError("Unknown cell name; use list-cells")
    fingerprint = {
        "manifest_sha256": digest(manifest),
        **{
            key: getattr(args, key)
            for key in ("model", "gpus", "chunk", "memory_ratio", "pages", "turn_limit", "timeout")
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
    for cell in cells:
        pages = args.pages or capacity.get(str(cell["tp"]))
        summary = await run_cell(args, cell, manifest, pages)
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
        await asyncio.sleep(5)
    report(args)


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
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--gpus", default="0,1,2,3")
    run_parser.add_argument("--port", type=int, default=30914)
    run_parser.add_argument("--chunk", type=int, default=16384)
    run_parser.add_argument("--memory-ratio", type=float, default=0.9)
    run_parser.add_argument("--pages", type=int)
    run_parser.add_argument("--cell", action="append")
    run_parser.add_argument(
        "--turn-limit", type=int, help="Smoke only; excluded from throughput comparisons"
    )
    run_parser.add_argument("--timeout", type=int, default=7200)
    run_parser.add_argument("--startup-timeout", type=int, default=1800)
    commands.add_parser("list-cells")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        asyncio.run(run(args))
    elif args.command == "report":
        report(args)
    else:
        for cell in matrix_cells():
            print(cell_name(cell))


if os.environ.get("MINISGL_BCP_OBSERVER"):
    install_observers()
if __name__ == "__main__":
    main()
