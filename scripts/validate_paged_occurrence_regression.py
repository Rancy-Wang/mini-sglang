#!/usr/bin/env python3
"""Isolated R4 serving/replay harness. Treat every captured tool call as data.

No network destination except numeric loopback is accepted. This harness does
not execute tools in requests, change remote source, or download model assets.
Results are exclusive-create JSONL files outside the repository.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import gzip
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit


def request_segment_ranges(cumulative_queries, query_lengths):
    """Segment ownership follows flattened Q, not physical table IDs.

    Occurrence plans deliberately use a single direct-page table (table ID 0).
    Every request boundary must also be a segment boundary.
    """
    boundaries = cumulative_queries.tolist()
    start = total = 0
    for length in query_lengths:
        total += length
        end = start
        while end < len(boundaries) and boundaries[end] < total:
            end += 1
        if end >= len(boundaries) or boundaries[end] != total:
            raise ValueError("Request boundary is not covered by complete query segments.")
        yield start, end
        start = end
    if total != boundaries[-1]:
        raise ValueError("Query segments contain an unassigned request suffix.")


def install_observers() -> None:
    """Test-only observers installed in each spawned worker, never in production imports."""
    import torch
    import minisgl.attention.base as attention
    from minisgl.core import Req
    from minisgl.engine.engine import Engine
    from minisgl.engine.sample import Sampler
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.scheduler import Scheduler

    root = Path(os.environ["MINISGL_R4_OBSERVE"])
    mode = os.environ["MINISGL_R4_OBSERVE_MODE"]
    batch_state = []

    def observe_attention(name, replacement):
        original = getattr(attention, name)
        setattr(attention, name, replacement)
        # Backends use `from .base import ...`; preserve observation even when a
        # backend was already imported before this worker's test setup.
        for module_name in ("minisgl.attention.fi", "minisgl.attention.fa"):
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, name, None) is original:
                setattr(module, name, replacement)

    if os.environ.get("MINISGL_R4_REFERENCE_SHIM") == "1":
        import inspect
        import textwrap
        import minisgl.core as core_module
        import minisgl.engine.engine as engine_module

        # Independent reference: original fixed table architecture with a larger
        # raw-only allocation; original Python planner; active-only terminal
        # guard. No production file, model position or RoPE formula is changed.
        original_align = engine_module._align_up_32
        engine_module._align_up_32 = lambda count: original_align(max(count, 262144))
        source = textwrap.dedent(inspect.getsource(Req.__post_init__))
        old = "torch.max(terminal_positions)"
        assert source.count(old) == 1 and "active_terminal_positions" not in source
        source = source.replace(old, "torch.max(terminal_positions[self.full_keep_mask.to(torch.bool)])")
        namespace = dict(vars(core_module))
        exec(compile(source, "<R4-independent-active-terminal-guard>", "exec"), namespace)
        Req.__post_init__ = namespace["__post_init__"]
        attention.try_build_occurrence_sliding_plan = lambda *args, **kwargs: None

    def emit(kind, **fields):
        with (root / f"observer-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, "time_ns": time.perf_counter_ns(),
                                     **fields}, separators=(",", ":")) + "\n")

    def digest(tensor):
        value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(value).hexdigest()

    original_forward = Engine.forward_batch

    @functools.wraps(original_forward)
    def forward(self, batch, args):
        batch_state[:] = [batch]
        graph = self.graph_runner.can_use_cuda_graph(batch)
        queries = [[req.uid, req.cached_len, req.device_len] for req in batch.reqs]
        started = time.perf_counter_ns()
        result = original_forward(self, batch, args)
        if mode == "exact":
            offset = 0
            for req, (_, start, end) in zip(batch.reqs, queries):
                # Hash every newly written KV in logical query order, not the
                # entire growing history on each decode token. Reposition writes
                # are observed separately below; reused pages were checked at
                # their original write in the same replay history.
                pages = batch.out_loc[offset:offset + end - start].to(torch.int64)
                offset += end - start
                layer_hashes = []
                for layer in range(self.kv_cache.num_layers):
                    layer_hashes.append([
                        digest(self.kv_cache.k_cache(layer).index_select(0, pages)),
                        digest(self.kv_cache.v_cache(layer).index_select(0, pages)),
                    ])
                emit("kv", uid=req.uid, cached_len=req.cached_len, pages=len(pages),
                     layers=layer_hashes, true_positions=digest(req.true_positions),
                     raw_positions=digest(req.raw_positions))
            if batch.occurrence_destination_pages is not None:
                pages = batch.occurrence_destination_pages.to(torch.int64)
                emit("reposition_kv", uids=[req.uid for req in batch.reqs],
                     positions=digest(batch.occurrence_position_pairs), pages=len(pages),
                     layers=[[digest(self.kv_cache.k_cache(layer).index_select(0, pages)),
                              digest(self.kv_cache.v_cache(layer).index_select(0, pages))]
                             for layer in range(self.kv_cache.num_layers)])
        emit("forward", queries=queries, size=batch.size, phase=batch.phase, graph=graph,
             host_ns=time.perf_counter_ns() - started,
             allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved())
        batch_state.clear()
        return result

    Engine.forward_batch = forward
    original_sample = Sampler.sample

    @functools.wraps(original_sample)
    def sample(self, logits, args):
        if mode == "exact" and batch_state:
            for index, req in enumerate(batch_state[0].reqs):
                emit("logits", uid=req.uid, cached_len=req.cached_len, sha256=digest(logits[index]))
        return original_sample(self, logits, args)

    Sampler.sample = sample
    original_append = Req.append_host

    @functools.wraps(original_append)
    def append(self, token):
        offset = len(self.input_ids)
        if mode == "exact":
            emit("token", uid=self.uid, token=token.tolist(),
                 position=int(self.true_positions[offset]), raw_position=int(self.raw_positions[offset]))
        return original_append(self, token)

    Req.append_host = append
    for name in ("build_context_attention_batch", "build_occurrence_attention_batch"):
        original = getattr(attention, name)

        def timed(reqs, *args, _original=original, _name=name, **kwargs):
            started = time.perf_counter_ns()
            result = _original(reqs, *args, **kwargs)
            emit(_name, uids=[req.uid for req in reqs], host_ns=time.perf_counter_ns() - started,
                 segments=result.num_segments, keys=len(result.key_positions))
            if mode == "exact":
                occurrence_base = 0
                ranges = request_segment_ranges(
                    result.cu_seqlens_q, [req.device_len - req.cached_len for req in reqs])
                for req, (segment_offset, end) in zip(reqs, ranges, strict=True):
                    first_key, last_key = (int(result.cu_seqlens_k[index])
                                           for index in (segment_offset, end))
                    emit("csr", uid=req.uid, planner=_name, sliding_window=kwargs.get("sliding_window"),
                         cached_len=req.cached_len, device_len=req.device_len,
                         query_sha256=digest(result.cu_seqlens_q[segment_offset:end + 1] -
                                             result.cu_seqlens_q[segment_offset]),
                         offsets_sha256=digest(result.cu_seqlens_k[segment_offset:end + 1] - first_key),
                         keys_sha256=digest(result.key_positions[first_key:last_key] - occurrence_base))
                    if _name == "build_occurrence_attention_batch":
                        occurrence_base += len(req.occurrence_raw_tokens)
            return result

        observe_attention(name, timed)

    original_allocate = CacheManager._allocate

    @functools.wraps(original_allocate)
    def allocate(self, count):
        started = time.perf_counter_ns()
        result = original_allocate(self, count)
        emit("allocate", pages=count, host_ns=time.perf_counter_ns() - started,
             free_pages=len(self.free_slots))
        return result

    CacheManager._allocate = allocate

    original_pages = attention.compile_context_page_tables

    @functools.wraps(original_pages)
    def page_tables(*args, **kwargs):
        started = time.perf_counter_ns()
        result = original_pages(*args, **kwargs)
        emit("page_tables", host_ns=time.perf_counter_ns() - started,
             layout=kwargs.get("output_layout", "both"))
        return result

    observe_attention("compile_context_page_tables", page_tables)
    original_reset_idle = Scheduler.run_when_idle
    reset_seen = set()

    @functools.wraps(original_reset_idle)
    def reset_idle(self):
        original_reset_idle(self)
        for marker in sorted(root.glob("reset-*.request")):
            if marker.name in reset_seen:
                continue
            cache = self.cache_manager
            assert cache.available_size == cache.num_pages, "Cannot reset a protected cache"
            pages = cache._allocate(cache.num_pages)
            assert not cache.prefix_cache._ordinary_slot_nodes
            cache.free_occurrence_pages(pages)
            reset_seen.add(marker.name)
            emit("cache_reset", marker=marker.name, free_pages=len(cache.free_slots))
            (root / f"{marker.stem}.ack-{os.getpid()}").touch(exist_ok=False)

    Scheduler.run_when_idle = reset_idle
    if mode != "pressure":
        return

    def nodes(cache):
        pending = list(cache.root_node.children.values())
        found = []
        while pending:
            node = pending.pop()
            found.append(node)
            pending.extend(node.children.values())
        return found

    original_commit = CacheManager.cache_req

    @functools.wraps(original_commit)
    def commit(self, req, *, finished):
        result = original_commit(self, req, finished=finished)
        if finished:
            emit("completed_tree", uid=req.uid, repos=req.radix_current_reposition,
                 nodes=[[node.uuid, node.ref_count, node.page_length,
                         digest(node._key)] for node in nodes(self.prefix_cache)])
        return result

    CacheManager.cache_req = commit
    original_idle = Scheduler.run_when_idle
    pressure_done = False

    @functools.wraps(original_idle)
    def idle(self):
        nonlocal pressure_done
        original_idle(self)
        if pressure_done or not (root / "pressure.request").exists():
            return
        pressure_done = True
        cache = self.cache_manager
        tree = cache.prefix_cache
        before = nodes(tree)
        assert before and all(node.ref_count == 0 for node in before)
        from minisgl.kvcache.radix_cache import RadixCacheHandle

        leaf = max((node for node in before if node.is_leaf()), key=lambda node: node.page_length)
        protected = RadixCacheHandle(leaf.length, leaf)
        tree.lock_handle(protected)
        eligible = {node.uuid for node in before if node.ref_count == 0}
        protected_ids = {node.uuid for node in before if node.ref_count > 0}
        snapshot = {str(slot): sorted(node.uuid for node in owners)
                    for slot, owners in tree._ordinary_slot_nodes.items()}
        emit("pressure_before", nodes=[[node.uuid, node.ref_count, node.page_length] for node in before],
             physical_owners=snapshot, free_pages=len(cache.free_slots), total_pages=cache.num_pages,
             dfs_leaves=[node.uuid for node in tree._collect_leave_nodes_for_evict()])
        held = None
        try:
            # Consume the genuine free list, then request every evictable physical
            # page through the existing allocator -> DFS -> heap eviction path.
            held = cache._allocate(cache.available_size)
            after_ids = {node.uuid for node in nodes(tree)}
            remaining = eligible & after_ids
            emit("pressure_protected", remaining_eligible=sorted(remaining),
                 protected_survive=protected_ids <= after_ids, free_pages=len(cache.free_slots),
                 retained_owners={str(slot): sorted(node.uuid for node in owners)
                                  for slot, owners in tree._ordinary_slot_nodes.items()})
            assert len(cache.free_slots) == 0
            assert protected_ids <= after_ids
            # Eviction stops once the requested physical capacity is reclaimed.
            # A zero-ref branch sharing ONLY protected pages may therefore stay:
            # deleting it cannot return memory until the live owner is unlocked.
            # Require every retained page to have a protected owner, then test
            # complete branch removal in the fully unlocked phase below.
            assert all(any(owner.ref_count > 0 for owner in owners)
                       for owners in tree._ordinary_slot_nodes.values())
        finally:
            if held is not None:
                cache.free_occurrence_pages(held)
            tree.lock_handle(protected, unlock=True)
        all_pages = cache._allocate(cache.num_pages)
        assert len(torch.unique(all_pages)) == cache.num_pages
        assert not tree._ordinary_slot_nodes
        assert not any(node.page_length for node in nodes(tree))
        assert len(cache.free_slots) == 0
        cache.free_occurrence_pages(all_pages)
        emit("pressure_complete", free_pages=len(cache.free_slots), total_pages=cache.num_pages,
             residual_nodes=[[node.uuid, node.page_length] for node in nodes(tree)])

    Scheduler.run_when_idle = idle


def loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or not ipaddress.ip_address(parsed.hostname).is_loopback
            or parsed.username or parsed.password):
        raise ValueError("Only numeric HTTP loopback destinations are permitted.")
    return value


def effective_interface(payload: dict, length: int) -> None:
    for parent, key in ((payload, "drop_message"),
                        (payload.get("drop_rule", {}), "drop_messages")):
        if isinstance(parent, dict) and isinstance(parent.get(key), dict):
            parent[key] = {event: ids for event, ids in parent[key].items()
                           if int(event) < length}
            if not parent[key]:
                parent.pop(key)
    if isinstance(payload.get("drop_rule"), dict):
        rule = payload["drop_rule"]
        if rule.get("type") == "message_drop" and "drop_messages" not in rule:
            payload.pop("drop_rule")
    if isinstance(payload.get("reposition"), list):
        payload["reposition"] = [event for event in payload["reposition"] if int(event) < length]
        if not payload["reposition"]:
            payload.pop("reposition")


def rolling_interface(messages: list[dict]) -> dict:
    """K=12 complete tool responses; event IDs are public zero-based message IDs."""
    tools = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
    drops = {str(event): [tools[n - 12]] for n, event in enumerate(tools) if n >= 12}
    return {"drop_message": drops, "reposition": list(map(int, drops))} if drops else {}


def prepare(args) -> None:
    """Read existing rollout usage once; tokenize only selected stress histories."""
    from transformers import AutoTokenizer
    from minisgl.benchmark.reposition_bcp import browsecomp_plus_tools, load_rollout_prompt_token_hints
    from minisgl.core import SamplingParams
    from minisgl.message import TokenizeMsg
    from minisgl.tokenizer.tokenize import TokenizeManager

    roots = sorted(args.source.glob(
        "rolling_tool_drop_k12_kv_full_document_100q_t1_tp8ret_c4_shard[0-3]"))
    assert len(roots) == 4
    hints = load_rollout_prompt_token_hints([root / "rollouts.jsonl" for root in roots])
    ranked = sorted(hints.items(), key=lambda item: (-item[1], item[0]))
    stress_ids = [case for case, _ in ranked[:8]]
    five_ids = ["781", "249", "806", "785", "827"]
    selected = set(stress_ids + five_ids)
    rows = {}
    for root in roots:
        with (root / "trajectories.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                case = str(row["case_id"])
                if case in selected:
                    assert case not in rows, f"Duplicate case {case}"
                    rows[case] = (row["trajectory"], str(root / "trajectories.jsonl"))
    assert set(rows) == selected
    args.output.mkdir(parents=True, exist_ok=False)
    for case in five_ids:
        trajectory, provenance = rows[case]
        ends = [i for i, message in enumerate(trajectory) if message.get("role") == "assistant"][:20]
        assert len(ends) == 20
        target = args.output / "five" / f"case-{case}"
        target.mkdir(parents=True)
        for turn, end in enumerate(ends, 1):
            messages = trajectory[:end]
            payload = {"model": args.model, "messages": messages, "tools": browsecomp_plus_tools(),
                       **rolling_interface(messages)}
            with gzip.open(target / f"turn-{turn:03d}.request.json.gz", "xt") as stream:
                json.dump(payload, stream, ensure_ascii=False)
        print(json.dumps({"case": case, "turns": len(ends), "provenance": provenance}), flush=True)
    manager = TokenizeManager(AutoTokenizer.from_pretrained(args.model, local_files_only=True),
                              radix_drop_key_mode="delta-marker")
    report = {"ranking_from_existing_rollout_usage": ranked, "selected": [],
              "five_ids": five_ids, "five_policy": "K12 drop and reposition at each new tool response",
              "stress_policy": "K12 drop; reposition after >=8192 new raw tokens at next tool end"}
    target = args.output / "stress"
    target.mkdir()
    for case in stress_ids:
        trajectory, provenance = rows[case]
        end = max(i for i, message in enumerate(trajectory) if message.get("role") == "assistant")
        messages = trajectory[:end]
        payload = {"model": args.model, "messages": messages, "tools": browsecomp_plus_tools(),
                   "max_tokens": 1, "ignore_eos": True, "temperature": 0, "stream": False,
                   **rolling_interface(messages)}
        payload.pop("reposition", None)

        def tokenize():
            return manager.tokenize([TokenizeMsg(
                uid=1, text=messages, sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages), drop_message=payload.get("drop_message"),
                reposition=payload.get("reposition"), tools=payload["tools"],
                use_context_mask=True)])[0]

        tokenize()  # Initialize production Harmony renderer state.
        _, owners, _ = manager._render_harmony_message_drop(
            messages, enable_thinking=None, tools=payload["tools"])
        ranges = manager._build_owner_position_ranges(owners)
        previous, repos, boundaries = 0, [], []
        for message_id, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            boundary = max(end for _, end in ranges[message_id])
            if boundary - previous >= 8192:
                repos.append(message_id)
                boundaries.append(boundary)
                previous = boundary
        payload["reposition"] = repos
        result = tokenize()
        item = {"case_id": case, "historical_prompt_token_hint": hints[case],
                "raw_tokens": result.prompt_tokens, "active_tokens": len(result.input_ids),
                "next_position": result.radix_next_position, "provenance": provenance,
                "reposition_message_ids": repos, "reposition_raw_insert_offsets": boundaries}
        report["selected"].append(item)
        (target / f"case_{case}.json").write_text(json.dumps(payload, ensure_ascii=False))
        print(json.dumps(item), flush=True)
    (target / "preflight.json").write_text(json.dumps(report, indent=2))


def start_server(args) -> None:
    gpus = [int(value) for value in args.gpus.split(",")]
    memory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    used = {int(row.split(",")[0]): int(row.split(",")[1]) for row in memory.splitlines()}
    if any(used[gpu] > 100 for gpu in gpus):
        raise RuntimeError(f"Selected GPUs are not free: {[(gpu, used[gpu]) for gpu in gpus]}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(args.repo.resolve() / "python"))
    if args.observe:
        env["MINISGL_R4_OBSERVE"] = str(root)
        env["MINISGL_R4_OBSERVE_MODE"] = args.observe
    if args.reference_shim:
        if args.observe != "exact":
            raise ValueError("Reference safety shim requires exact observation, never timing.")
        baseline_core = (args.repo / "python/minisgl/core.py").read_text()
        if "occurrence_external_storage" in baseline_core:
            raise ValueError("Reference safety shim requires the unchanged baseline source.")
        env["MINISGL_R4_REFERENCE_SHIM"] = "1"
    runtime = args.runtime_root.resolve() if args.runtime_root else root
    for key in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR"):
        path = runtime / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    binary = Path(sys.executable).parent
    for key, name in (("CC", "x86_64-conda-linux-gnu-gcc"),
                      ("CXX", "x86_64-conda-linux-gnu-g++")):
        if (binary / name).exists():
            env[key] = str(binary / name)
    argv = [sys.executable, "-m", "minisgl", "--model-path", args.model,
            "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", str(len(gpus)),
            "--dtype", "bfloat16", "--disable-pynccl", "--memory-ratio", str(args.memory_ratio),
            "--max-running-requests", "8", "--cuda-graph-max-bs", "8",
            "--max-seq-len-override", "131072", "--max-prefill-length", str(args.chunk),
            "--request-timeout", "3600", "--cache-type", "radix", "--page-size", "1",
            "--attention-backend", "fi", "--radix-drop-key-mode", "delta-marker",
            "--contextual-prefill-mode", "mask", "--reposition-execution-mode", "paged-occurrence",
            "--tool-call-parser", "gpt-oss", "--reasoning-parser", "gpt-oss"]
    if args.observe:
        argv[1:3] = [str(Path(__file__).resolve()), "worker"]
    with (root / "server.log").open("xb") as log:
        child = subprocess.Popen(argv, cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    identity = {"pid": child.pid, "argv": argv, "gpus": gpus,
                "start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split()[21],
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=args.repo, text=True).strip()}
    (root / "launch.json").write_text(json.dumps(identity, indent=2))
    print(json.dumps(identity), flush=True)


def stop_server(args) -> None:
    identity = json.loads((args.output / "launch.json").read_text())
    pid = identity["pid"]
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        print("Recorded process already exited.")
        return
    argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
    if (argv != identity["argv"] or proc.joinpath("stat").read_text().split()[21]
            != identity["start_ticks"] or os.getpgid(pid) != pid):
        raise RuntimeError("Process identity changed; refusing to send a signal.")
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM only to owned server {pid}; inspect worker cleanup before reuse.")


async def send(client, url, payload, label):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    record = {"label": label, "request_sha256": hashlib.sha256(encoded).hexdigest(),
              "message_count": len(payload["messages"]), "max_tokens": payload.get("max_tokens")}
    started = time.perf_counter_ns()
    try:
        response = await client.post(url, json=payload)
        record["status_code"] = response.status_code
        try:
            record["response"] = response.json()
        except ValueError:
            record["response_text"] = response.text
        metrics = record.get("response", {}).get("server_metrics")
        if metrics:
            record["ttft_ms"] = (metrics["first_token_generated_ns"] -
                                 metrics["request_received_ns"]) / 1e6
            if metrics["generated_tokens"] > 1:
                # End-to-end post-first-token cost, including terminal handling;
                # this is deliberately not labelled CUDA decode-kernel time.
                record["decode_e2e_ms_per_token"] = (
                    metrics["request_finished_ns"] - metrics["first_token_generated_ns"]
                ) / 1e6 / (metrics["generated_tokens"] - 1)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_ms"] = (time.perf_counter_ns() - started) / 1e6
    return record


async def replay(args) -> None:
    import httpx

    source_bytes = args.request.read_bytes()
    source = json.loads(source_bytes)
    messages = source["messages"]
    lengths = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    lengths.append(len(messages))
    if args.terminal_only:
        lengths = [len(messages)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as log:
        log.write(json.dumps({"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                              "target_count": len(lengths), "original_terminal": args.original_terminal}) + "\n")
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
            for turn, length in enumerate(lengths, 1):
                payload = copy.deepcopy(source)
                payload["messages"] = messages[:length]
                if not (args.original_terminal and length == len(messages)):
                    payload.update(max_tokens=1, ignore_eos=True)
                payload["stream"] = False
                if args.seed is not None:
                    payload["seed"] = args.seed
                payload.pop("stream_options", None)
                effective_interface(payload, length)
                record = await send(client, args.url, payload, turn)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps({key: value for key, value in record.items() if key != "response"}), flush=True)
                if record.get("status_code") != 200:
                    raise RuntimeError(f"Replay failed at turn {turn}: {record}")
            probe = {"model": source["model"], "messages": [{"role": "user", "content": "Say OK."}],
                     "max_tokens": 1, "temperature": 0, "stream": False}
            record = await send(client, args.url, probe, "post_replay_health")
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("status_code") != 200:
                raise RuntimeError("Post-replay inference health check failed.")


def wave_payload(payload, max_tokens):
    payload = dict(payload)
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("Stress generation budget must be positive.")
        payload.update(max_tokens=max_tokens, ignore_eos=True)
    return payload


async def wave(args) -> None:
    import httpx

    selected = json.loads((args.input / "preflight.json").read_text())["selected"][:args.bs]
    assert len(selected) == args.bs
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False,
                                 limits=httpx.Limits(max_connections=args.bs)) as client:
        results = await asyncio.gather(*[
            send(client, args.url, wave_payload(
                json.loads((args.input / f"case_{item['case_id']}.json").read_text()), args.max_tokens),
                 item["case_id"]) for item in selected
        ])
    with args.output.open("x") as log:
        json.dump({"http_concurrency": args.bs, "actual_gpu_batch": "requires server evidence",
                   "results": results}, log, ensure_ascii=False, indent=2)
    print(json.dumps([{key: value for key, value in row.items() if key != "response"}
                      for row in results]), flush=True)
    if any(row.get("status_code") != 200 for row in results):
        raise RuntimeError("At least one concurrent request failed.")


async def five(args) -> None:
    import httpx

    cases = ["781", "249", "806", "785", "827"]
    with args.output.open("x") as log:
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False,
                                     limits=httpx.Limits(max_connections=4)) as client:
            for group in (cases[:1], cases[1:]):
                for turn in range(1, 21):
                    payloads = []
                    for case in group:
                        with gzip.open(args.input / f"case-{case}/turn-{turn:03d}.request.json.gz", "rt") as stream:
                            payload = json.load(stream)
                        if args.mode == "no_drop":
                            for key in ("drop_rule", "drop_message", "reposition"):
                                payload.pop(key, None)
                        payload.update(max_tokens=args.max_tokens, temperature=0, top_p=1, top_k=-1,
                                       seed=17, stream=False)
                        payload.pop("stream_options", None)
                        payloads.append(payload)
                    records = await asyncio.gather(*[
                        send(client, args.url, payload, f"{case}:{turn}")
                        for case, payload in zip(group, payloads)
                    ])
                    for record in records:
                        record.update(mode=args.mode, http_concurrency=len(group))
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    print(json.dumps([{key: value for key, value in row.items() if key != "response"}
                                      for row in records]), flush=True)
                    if any(row.get("status_code") != 200 for row in records):
                        raise RuntimeError(f"Five-case replay failed at turn {turn}.")


async def clear_cache(args) -> None:
    """Idle-only test reset. The identical health prefix remains in every cache."""
    import httpx

    marker = args.server_root / f"reset-{time.time_ns()}.request"
    marker.touch(exist_ok=False)
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        probe = {
            "model": args.model, "messages": [{"role": "user", "content": "Say OK."}],
            "temperature": 0, "max_tokens": 1, "stream": False}
        wake = await send(client, args.url, probe, "cache_reset_wakeup")
        deadline = time.monotonic() + 30
        while len(acknowledgments := list(args.server_root.glob(f"{marker.stem}.ack-*"))) != args.tp:
            if wake.get("status_code") != 200 or time.monotonic() > deadline:
                raise RuntimeError(f"Incomplete cache reset: {wake}; acknowledgments={acknowledgments}")
            await asyncio.sleep(0.05)
        record = await send(client, args.url, probe, "cache_reset_health")
        if record.get("status_code") != 200:
            raise RuntimeError(f"Inference after cache reset failed: {record}")
    with args.output.open("x") as stream:
        json.dump({"health": record, "acknowledgments": [str(p) for p in acknowledgments]}, stream)
    print(json.dumps({"status": "cache_reset_complete", "tp": args.tp}), flush=True)


def paired_noninferiority(baseline, candidate):
    """One-sided Student-t interval over independent paired run log-ratios.

    Exactly five paired repetitions are preregistered. Turns within one replay
    are correlated observations, never counted as independent repetitions.
    """
    import math

    if len(baseline) != 5 or len(candidate) != 5 or min(baseline + candidate) <= 0:
        raise ValueError("The preregistered gate requires five positive paired run means.")
    ratios = [math.log(c / b) for b, c in zip(baseline, candidate, strict=True)]
    mean = statistics.mean(ratios)
    upper = math.exp(mean + 2.131846786 * statistics.stdev(ratios) / math.sqrt(5))
    return {"paired_ratio": math.exp(mean), "upper_one_sided_95": upper,
            "noninferior_2_percent": upper <= 1.02, "paired_runs": 5}


async def benchmark(args) -> None:
    """Five alternating paired runs, one server at a time on the same GPUs.

    Warm up both workload modes, reset only idle unprotected KV before every
    measured mode, and retain all measurements including outliers.
    """
    import httpx

    args.output.mkdir(parents=True, exist_ok=False)
    specification = {
        "repetitions": 5, "order": [(["baseline", "candidate"] if i % 2 == 0 else
                                     ["candidate", "baseline"]) for i in range(5)],
        "gpus": args.gpus, "port": args.port, "chunk": args.chunk,
        "modes": ["no_drop", "rolling_drop"], "cases": ["781", "249", "806", "785", "827"],
        "turns": 20, "max_tokens": 8, "seed": 17, "warmup": "both complete modes before measurements",
        "primary": "TTFT paired run arithmetic means, separately by mode and HTTP bs",
        "gate": "one-sided 95% t interval on 5 paired log ratios <=1.02; no outlier removal",
    }
    (args.output / "specification.json").write_text(json.dumps(specification, indent=2))
    for repetition, order in enumerate(specification["order"], 1):
        for branch in order:
            root = args.output / f"rep-{repetition:02d}-{branch}"
            launch = SimpleNamespace(
                repo=args.baseline if branch == "baseline" else args.repo,
                model=args.model, gpus=args.gpus, port=args.port, chunk=args.chunk,
                memory_ratio=args.memory_ratio, observe="timing", reference_shim=False,
                runtime_root=args.baseline_cache if branch == "baseline" else args.candidate_cache,
                output=root)
            started = time.perf_counter_ns()
            start_server(launch)
            try:
                deadline = time.monotonic() + 900
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    while True:
                        try:
                            ready = await client.get(f"http://127.0.0.1:{args.port}/v1/models")
                            if ready.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        identity = json.loads((root / "launch.json").read_text())
                        proc = Path(f"/proc/{identity['pid']}/stat")
                        if not proc.exists() or proc.read_text().split()[2] == "Z":
                            raise RuntimeError(f"Server exited during startup: {root}")
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"Server startup timed out: {root}")
                        await asyncio.sleep(2)
                (root / "initialization.json").write_text(json.dumps({
                    "startup_ns": time.perf_counter_ns() - started}))
                for phase in ("warmup", "measured"):
                    for mode in specification["modes"]:
                        common = dict(url=f"http://127.0.0.1:{args.port}/v1/chat/completions", timeout=1800)
                        await clear_cache(SimpleNamespace(
                            **common, server_root=root, model=args.model, tp=len(args.gpus.split(",")),
                            output=root / f"{phase}-{mode}-reset.json"))
                        await five(SimpleNamespace(
                            **common, input=args.input, mode=mode, max_tokens=8,
                            output=root / f"{phase}-{mode}.jsonl"))
                print(json.dumps({"finished": str(root)}), flush=True)
            finally:
                stop_server(launch)
                deadline = time.monotonic() + 60
                while True:
                    memory = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
                    ], text=True)
                    used = {int(row.split(",")[0]): int(row.split(",")[1]) for row in memory.splitlines()}
                    if all(used[int(gpu)] <= 100 for gpu in args.gpus.split(",")):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("Owned server did not release GPUs; refusing the next run.")
                    await asyncio.sleep(1)
    report = {"specification": specification, "strata": []}
    for mode in specification["modes"]:
        for bs in (1, 4):
            metrics = {}
            for metric in ("ttft_ms", "decode_e2e_ms_per_token"):
                branch_values, branch_means = {}, {}
                for branch in ("baseline", "candidate"):
                    values, means = [], []
                    for repetition in range(1, 6):
                        path = args.output / f"rep-{repetition:02d}-{branch}" / f"measured-{mode}.jsonl"
                        rows = [json.loads(line) for line in path.open()]
                        selected = [row[metric] for row in rows
                                    if row["http_concurrency"] == bs and metric in row]
                        assert selected and all(row["status_code"] == 200 for row in rows)
                        means.append(statistics.mean(selected))
                        values.extend(selected)
                    percentiles = statistics.quantiles(values, n=100, method="inclusive")
                    branch_values[branch] = {"n": len(values), "mean": statistics.mean(values),
                                             "p50": percentiles[49], "p90": percentiles[89],
                                             "p99": percentiles[98], "paired_run_means": means}
                    branch_means[branch] = means
                metrics[metric] = {**branch_values,
                                   **paired_noninferiority(branch_means["baseline"], branch_means["candidate"])}
            report["strata"].append({"mode": mode, "http_bs": bs, "metrics": metrics})
    report["ttft_noninferior"] = all(row["metrics"]["ttft_ms"]["noninferior_2_percent"]
                                       for row in report["strata"])
    report["remaining_review"] = "exact outputs, planner/mapping, tail and memory review are separate gates"
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server
        launch_server()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start")
    launch.add_argument("--repo", type=Path, required=True)
    launch.add_argument("--model", required=True)
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--port", type=int, required=True)
    launch.add_argument("--chunk", type=int, default=32768)
    launch.add_argument("--memory-ratio", type=float, default=0.90)
    launch.add_argument("--observe", choices=["timing", "exact", "pressure"])
    launch.add_argument("--reference-shim", action="store_true")
    launch.add_argument("--runtime-root", type=Path)
    stop = commands.add_parser("stop")
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--request", type=Path, required=True)
    replay_parser.add_argument("--original-terminal", action="store_true")
    replay_parser.add_argument("--terminal-only", action="store_true")
    replay_parser.add_argument("--seed", type=int)
    wave_parser = commands.add_parser("wave")
    wave_parser.add_argument("--input", type=Path, required=True)
    wave_parser.add_argument("--bs", type=int, choices=[4, 8], required=True)
    wave_parser.add_argument("--max-tokens", type=int,
                             help="Override generation budget to cover decode/graph replay.")
    five_parser = commands.add_parser("five")
    five_parser.add_argument("--input", type=Path, required=True)
    five_parser.add_argument("--mode", choices=["no_drop", "rolling_drop"], required=True)
    five_parser.add_argument("--max-tokens", type=int, default=8)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--model", required=True)
    clear_parser = commands.add_parser("clear")
    clear_parser.add_argument("--server-root", type=Path, required=True)
    clear_parser.add_argument("--model", required=True)
    clear_parser.add_argument("--tp", type=int, default=2)
    bench_parser = commands.add_parser("benchmark")
    for name in ("repo", "baseline", "input", "baseline-cache", "candidate-cache"):
        bench_parser.add_argument(f"--{name}", type=Path, required=True)
    bench_parser.add_argument("--model", required=True)
    bench_parser.add_argument("--gpus", required=True)
    bench_parser.add_argument("--port", type=int, required=True)
    bench_parser.add_argument("--chunk", type=int, default=32768)
    bench_parser.add_argument("--memory-ratio", type=float, default=0.90)
    for command in (launch, stop, replay_parser, wave_parser, five_parser, prepare_parser, clear_parser,
                    bench_parser):
        command.add_argument("--output", type=Path, required=True)
    for command in (replay_parser, wave_parser, five_parser, clear_parser):
        command.add_argument("--url", type=loopback_url, required=True)
        command.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.command == "start":
        start_server(args)
    elif args.command == "stop":
        stop_server(args)
    elif args.command == "replay":
        asyncio.run(replay(args))
    elif args.command == "wave":
        asyncio.run(wave(args))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "clear":
        asyncio.run(clear_cache(args))
    elif args.command == "benchmark":
        asyncio.run(benchmark(args))
    else:
        asyncio.run(five(args))


if os.environ.get("MINISGL_R4_OBSERVE"):
    install_observers()

if __name__ == "__main__":
    main()
