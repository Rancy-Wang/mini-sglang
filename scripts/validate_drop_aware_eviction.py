#!/usr/bin/env python3
"""Paired Drop-aware serving validation. Outputs must be outside the repository.

The worker instrumentation is test-only, identical for both revisions, and
records real engine batches. It never executes tool calls from request data.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time


def rolling_interface(messages, keep=12):
    tools = [i for i, message in enumerate(messages) if message["role"] == "tool"]
    drops = {str(event): [tools[n - keep]] for n, event in enumerate(tools) if n >= keep}
    return {"drop_message": drops, "reposition": [int(i) for i in drops]}


def workload_interface(messages, keep, workload):
    if workload == "no-drop":
        return {}
    schedule = rolling_interface(messages, keep)
    if workload == "rolling-drop":
        schedule.pop("reposition")
    return schedule


def make_messages(case, rounds, repetitions):
    messages = [{"role": "system", "content": "Read the supplied documents and answer briefly."},
                {"role": "user", "content": f"Independent research case {case}. Summarize the evidence."}]
    for turn in range(rounds):
        call = f"call_{case}_{turn}"
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": call,
                "type": "function", "function": {"name": "read_document",
                    "arguments": json.dumps({"document": f"{case}-{turn}"})}}]},
            {"role": "tool", "tool_call_id": call, "name": "read_document",
             "content": f"Document {case}-{turn}. " + (
                 f"Observation {turn}: the archive records dates, measurements, and comparisons. "
             ) * repetitions},
        ]
    return messages


TOOLS = [{"type": "function", "function": {"name": "read_document",
    "description": "Read an archived document.", "parameters": {"type": "object",
    "properties": {"document": {"type": "string"}}, "required": ["document"]}}}]


def write_control(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def prepare(args, root):
    from transformers import AutoTokenizer
    from minisgl.core import SamplingParams
    from minisgl.message import TokenizeMsg
    from minisgl.tokenizer.tokenize import TokenizeManager

    manager = TokenizeManager(AutoTokenizer.from_pretrained(args.model, local_files_only=True),
                              radix_drop_key_mode="delta-marker")
    manifest = []
    for case in range(args.requests):
        rounds = 34 + case % 6
        repetitions = max(1, min(220, args.min_full_tokens // (rounds * 16)))
        while True:
            messages = make_messages(case, rounds, repetitions)
            schedule = workload_interface(messages, args.rolling_keep, args.workload)
            result = manager.tokenize([TokenizeMsg(
                uid=case + 1, text=messages, sampling_params=SamplingParams(max_tokens=8),
                target_msg_id=len(messages), tools=TOOLS, use_context_mask=True,
                **schedule)])[0]
            if result.prompt_tokens >= args.min_full_tokens:
                break
            repetitions = int(repetitions * args.min_full_tokens / result.prompt_tokens) + 2
        payload = {"model": args.model, "messages": messages, "tools": TOOLS,
                   "max_tokens": 8, "temperature": 0, "top_p": 1, "seed": 17,
                   "ignore_eos": True, "stream": False, **schedule}
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        (root / f"case-{case:02d}.json").write_bytes(encoded)
        manifest.append({"case": case, "full_tokens": result.prompt_tokens,
                         "active_tokens": len(result.input_ids), "rounds": rounds,
                         "repetitions": repetitions, "sha256": hashlib.sha256(encoded).hexdigest()})
        print(json.dumps(manifest[-1]), flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def install_observers():
    """Cheap forward census; expensive page audits only in explicit audit mode."""
    import torch
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
    from minisgl.engine.sample import Sampler

    root = Path(os.environ["MINISGL_DAE_OBSERVER"])
    audit = os.environ.get("MINISGL_DAE_AUDIT") == "1"
    original_forward = Engine.forward_batch
    original_idle = Scheduler.run_when_idle
    original_schedule = PrefillManager.schedule_next_batch
    original_scheduler_forward = Scheduler._forward
    original_sample = Sampler.sample
    barrier_seen = set()
    drain_seen = set()
    pressure_seen = set()
    batch_state = []

    def control():
        path = root / "numerical.json"
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}

    def tree_nodes(tree):
        stack = list(tree.root_node.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def emit(kind, **fields):
        with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, **fields}) + "\n")

    @functools.wraps(original_schedule)
    def schedule(self, budget):
        # Queue a specified test wave before admitting its first prefill. This
        # is identical on both revisions and does not force actual GPU bs=8.
        marker = root / "wave.json"
        try:
            spec = json.loads(marker.read_text())
        except FileNotFoundError:
            spec = None
        if spec is not None:
            if spec["id"] not in barrier_seen:
                if len(self.pending_list) < spec["count"]:
                    return None
                barrier_seen.add(spec["id"])
        return original_schedule(self, budget)

    @functools.wraps(original_forward)
    def forward(self, batch, sampling):
        queries = [[r.uid, r.cached_len, r.device_len, r.prompt_tokens] for r in batch.reqs]
        graph = self.graph_runner.can_use_cuda_graph(batch)
        spec = control()
        batch_state[:] = [(batch, spec, queries)]
        result = original_forward(self, batch, sampling)
        if spec.get("capture"):
            directory = root / spec["label"]
            directory.mkdir(exist_ok=True)
            offset = 0
            for req, (_, start, end, _) in zip(batch.reqs, queries):
                plan = getattr(req, "drop_recovery_plan", None)
                if plan is not None and start == plan.start:
                    emit("recovery_plan", label=spec["label"], matched=plan.matched_length,
                         intervals=plan.intervals)
                pages = batch.out_loc[offset:offset + end - start].long()
                offset += end - start
                kv = torch.stack([torch.stack([
                    self.kv_cache.k_cache(layer).index_select(0, pages),
                    self.kv_cache.v_cache(layer).index_select(0, pages)])
                    for layer in range(self.kv_cache.num_layers)]).cpu()
                torch.save({"kv": kv, "start": start, "end": end,
                            "raw": req.raw_positions[start:end].clone()},
                           directory / f"kv-{os.getpid()}-{start}-{int(req.raw_positions[start])}.pt")
        batch_state.clear()
        emit("forward", queries=queries, size=batch.size, phase=str(batch.phase), graph=graph)
        return result

    @functools.wraps(original_sample)
    def sample(self, logits, sampling):
        if batch_state and batch_state[0][1].get("capture"):
            batch, spec, queries = batch_state[0]
            directory = root / spec["label"]
            directory.mkdir(exist_ok=True)
            for index, (_, start, end, _) in enumerate(queries):
                if isinstance(batch.reqs[index], ChunkedReq):
                    continue
                torch.save(logits[index].detach().cpu(),
                           directory / f"logits-{os.getpid()}-{end}.pt")
        return original_sample(self, logits, sampling)

    @functools.wraps(original_scheduler_forward)
    def pressure(self, forward_input):
        spec = control()
        if spec.get("pressure") and spec["label"] not in pressure_seen:
            batch = forward_input.batch
            if "batch_size" in spec:
                assert batch.size == spec["batch_size"]
            if any(getattr(r.cache_handle, "skip_ranges", ()) for r in batch.reqs):
                pressure_seen.add(spec["label"])
                cache = self.cache_manager
                tree = cache.prefix_cache
                critical = [batch.out_loc]
                for value in (batch.occurrence_source_pages, batch.occurrence_destination_pages):
                    if value is not None:
                        critical.append(value)
                critical = torch.cat(critical).long()
                assert bool(torch.all(critical >= 0))
                protected = {n.uuid for n in tree_nodes(tree) if n.path_ref_count}
                drop_before = tree.eviction_stats["drop_pages"]
                held = cache._allocate(cache.available_size)
                try:
                    assert not bool(torch.isin(critical, held.long()).any())
                    assert len(cache.free_slots) == cache.available_size == 0
                    assert protected <= {n.uuid for n in tree_nodes(tree)}
                    assert tree.eviction_stats["drop_pages"] > 0
                    if spec.get("require_new_drop"):
                        assert tree.eviction_stats["drop_pages"] > drop_before
                    # Poison reclaimed KV so stale page IDs cannot accidentally
                    # pass a numerical check by retaining their old contents.
                    for layer in range(self.engine.kv_cache.num_layers):
                        self.engine.kv_cache.k_cache(layer)[held.long()] = float("nan")
                        self.engine.kv_cache.v_cache(layer)[held.long()] = float("nan")
                    emit("pressure", eviction=dict(tree.eviction_stats), released=len(held),
                         protected_nodes=len(protected), batch_size=batch.size)
                finally:
                    cache.free_occurrence_pages(held)
        return original_scheduler_forward(self, forward_input)

    @functools.wraps(original_idle)
    def idle(self):
        original_idle(self)
        cache = self.cache_manager
        emit("idle", free=len(cache.free_slots), total=cache.num_pages,
             evictable=cache.prefix_cache.evictable_size,
             protected=cache.prefix_cache.protected_size,
             eviction=getattr(cache.prefix_cache, "eviction_stats", {}))
        if audit:
            resident = []
            for node in tree_nodes(cache.prefix_cache):
                assert node.ref_count == 0 and getattr(node, "path_ref_count", 0) == 0
                values = node.value[node.value >= 0]
                resident.append(values)
            all_pages = torch.cat([cache.free_slots, *resident])
            assert len(torch.unique(all_pages)) == cache.num_pages
        # Idle mutations use one TP broadcast, avoiding per-rank file races.
        import torch.distributed as dist
        command = [None]
        group = self.engine.tp_cpu_group
        if dist.get_rank(group) == 0:
            command[0] = [p.name for p in sorted(root.glob("drain-*.request"))]
        dist.broadcast_object_list(command, src=0, group=group)
        for name in command[0]:
            if name in drain_seen:
                continue
            assert cache.available_size == cache.num_pages
            assert all(n.ref_count == getattr(n, "path_ref_count", 0) == 0
                       for n in tree_nodes(cache.prefix_cache))
            pages = cache._allocate(cache.num_pages)
            assert len(torch.unique(pages)) == cache.num_pages
            assert not any(n.page_length for n in tree_nodes(cache.prefix_cache))
            cache.free_occurrence_pages(pages)
            emit("drained", free=len(cache.free_slots), total=cache.num_pages)
            drain_seen.add(name)
            (root / f"{name}.ack-{os.getpid()}").touch()

    Engine.forward_batch = forward
    Scheduler.run_when_idle = idle
    PrefillManager.schedule_next_batch = schedule
    Scheduler._forward = pressure
    Sampler.sample = sample


def launch(args, repo, root, candidate):
    root.mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo / "python"),
               MINISGL_DAE_OBSERVER=str(root))
    if args.audit:
        env["MINISGL_DAE_AUDIT"] = "1"
    runtime = args.output / "runtime"
    runtime.mkdir(exist_ok=True)
    for name in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH"):
        directory = runtime / name.lower()
        directory.mkdir(exist_ok=True)
        env[name] = str(directory)
    binary = Path(sys.executable).parent
    for key, name in (("CC", "x86_64-conda-linux-gnu-gcc"), ("CXX", "x86_64-conda-linux-gnu-g++")):
        if (binary / name).exists():
            env[key] = str(binary / name)
    argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--model-path", args.model,
            "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", str(len(args.gpus.split(','))),
            "--dtype", "bfloat16", "--disable-pynccl", "--memory-ratio", str(args.memory_ratio),
            "--max-running-requests", "8", "--cuda-graph-max-bs", "8",
            "--max-seq-len-override", "131072", "--max-prefill-length", str(args.chunk),
            "--request-timeout", "7200", "--cache-type", "radix", "--page-size", "1",
            "--attention-backend", "fi", "--radix-drop-key-mode", "delta-marker",
            "--contextual-prefill-mode", "mask", "--reposition-execution-mode", args.mode,
            "--tool-call-parser", args.tool_call_parser, "--reasoning-parser", args.reasoning_parser]
    if candidate:
        argv.append("--drop-aware-eviction")
    if args.pages:
        argv.extend(["--num-pages", str(args.pages)])
    with (root / "server.log").open("xb") as log:
        child = subprocess.Popen(argv, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    (root / "launch.json").write_text(json.dumps({"pid": child.pid, "argv": argv,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}, indent=2))
    return child


async def drain_and_probe(client, url, root, args):
    from validate_paged_occurrence_regression import send

    write_control(root / "numerical.json", {})
    (root / "drain-final.request").touch()
    probe = {"model": args.model, "messages": [{"role": "user", "content":
             "Reply with one word: ready."}], "max_tokens": 8, "temperature": 0,
             "ignore_eos": True, "stream": False}
    wake = await send(client, url + "/v1/chat/completions", probe, "drain-wakeup")
    if wake.get("status_code") != 200:
        raise RuntimeError(f"Drain wakeup failed: {wake}")
    deadline = time.monotonic() + 120
    while len(list(root.glob("drain-final.request.ack-*"))) != len(args.gpus.split(',')):
        if time.monotonic() > deadline:
            raise RuntimeError("Not every TP rank drained all pages")
        await asyncio.sleep(.2)
    probe["messages"][0]["content"] = "After memory reclamation, reply with one word: healthy."
    health = await send(client, url + "/v1/chat/completions", probe, "post-drain-health")
    if health.get("status_code") != 200:
        raise RuntimeError(f"Post-drain inference failed: {health}")
    (root / "health.json").write_text(json.dumps(health, indent=2))


def compare_numerical(root, label):
    import torch

    records = []
    for reference in sorted((root / "reference").glob("logits-*.pt")):
        current = root / label / reference.name
        if not current.exists():
            continue  # Intermediate reference chunks need not be recomputed.
        before, after = torch.load(reference, weights_only=True).float(), torch.load(current, weights_only=True).float()
        delta = after - before
        rel = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(before).clamp_min(1e-12))
        maximum = float(delta.abs().max())
        top = torch.topk(before, 2).values
        margin = float(top[0] - top[1])
        same = int(before.argmax()) == int(after.argmax())
        assert torch.isfinite(after).all() and rel <= .01 and maximum <= .1, (reference.name, rel, maximum)
        assert margin <= .2 or same, (reference.name, margin, "greedy token mismatch")
        records.append({"file": reference.name, "relative_l2": rel, "max_abs": maximum,
                        "margin": margin, "same_argmax": same})
    if not records:
        raise RuntimeError("No corresponding logits were compared")
    kv_records = []
    by_pid = {}
    for reference in sorted((root / "reference").glob("kv-*.pt")):
        by_pid.setdefault(reference.name.split('-')[1], []).append(
            torch.load(reference, weights_only=True))
    reference_kv = {}
    for pid, chunks in by_pid.items():
        raw = torch.cat([row["raw"] for row in chunks])
        order = torch.argsort(raw)
        reference_kv[pid] = (raw[order], torch.cat([row["kv"] for row in chunks], dim=2)[:, :, order])
    for current in sorted((root / label).glob("kv-*.pt")):
        after = torch.load(current, weights_only=True)
        pid = current.name.split('-')[1]
        raw, before = reference_kv[pid]
        lookup = torch.searchsorted(raw, after["raw"]).long()
        assert torch.equal(raw[lookup], after["raw"]), (current.name, "KV positions not fully compared")
        torch.testing.assert_close(after["kv"].float(), before[:, :, lookup].float(), atol=.02, rtol=.02)
        kv_records.append({"file": current.name, "tokens": len(after["raw"])})
    return {"logits": records, "kv": kv_records}


async def run_recovery(args):
    """One model lifetime: reference, Drop under a live lease, repair, reuse, drain."""
    import httpx
    from validate_paged_occurrence_regression import send

    if not args.pages:
        args.pages = 32768
    root = args.output / "recovery"
    child = launch(args, args.baseline.resolve() if args.baseline else args.repo,
                   root, args.baseline is None)
    url = f"http://127.0.0.1:{args.port}"
    responses = {}
    try:
        async with httpx.AsyncClient(timeout=7200, trust_env=False) as client:
            deadline = time.monotonic() + 1200
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Server exited: {root / 'server.log'}")
                try:
                    if (await client.get(url + "/v1/models", timeout=2)).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup timed out")
                await asyncio.sleep(2)
            messages = make_messages(910, 15, 6)
            # The reference already contains Reposition after TR13. TR14 adds
            # the second Delta; TR15 can release TR2 under that matched path.
            # Returning to TR13 needs TR2 and its earlier birth dependencies.
            steps = [
                ("reference", 13, True, False), ("noise", 13, True, False),
                ("prime", 14, False, False), ("pressure", 15, False, True),
                ("repair", 13, True, False), ("reuse", 13, True, False),
            ]
            for label, rounds, capture, pressure in (steps[:2] if args.baseline else steps):
                selected = messages[:2 + 2 * rounds]
                payload = {"model": args.model, "messages": selected, "tools": TOOLS,
                           "max_tokens": 8, "temperature": 0, "top_p": 1,
                           "seed": 17, "ignore_eos": True, "stream": False,
                           **rolling_interface(selected, 12)}
                if args.recovery_tool_choice is not None:
                    payload["tool_choice"] = args.recovery_tool_choice
                if args.recovery_single_reposition:
                    # Directed recovery control, deliberately not canonical
                    # Rolling Drop: later Drops keep the first Reposition's
                    # positions so a compatible resident suffix can be reused.
                    payload["reposition"] = payload["reposition"][:1]
                write_control(root / "numerical.json", {"label": label, "capture": capture,
                                                       "pressure": pressure})
                row = await send(client, url + "/v1/chat/completions", payload, label)
                responses[label] = row
                (root / "responses.json").write_text(json.dumps(responses, indent=2))
                if row.get("status_code") != 200:
                    raise RuntimeError(f"Numerical request failed: {label}: {row}")
                print(json.dumps({"recovery_step": label, "status": row["status_code"]}), flush=True)
                if label == "noise":
                    (root / "noise.json").write_text(json.dumps(compare_numerical(root, label), indent=2))
            if args.baseline:
                return
            events = [json.loads(line) for path in root.glob("events-*.jsonl")
                      for line in path.read_text().splitlines()]
            if len([e for e in events if e["kind"] == "pressure"]) != len(args.gpus.split(',')):
                raise RuntimeError("The requested live-lock internal eviction was not triggered")
            if args.mode == "paged-occurrence" and not any(
                e["kind"] == "recovery_plan" and e["label"] == "repair"
                and any(a < e["matched"] for a, _ in e["intervals"]) for e in events
            ):
                raise RuntimeError("No matched interior hole was recomputed")
            report = {label: compare_numerical(root, label) for label in ("repair", "reuse")}
            (root / "numerical.json").unlink()
            (root / "comparison.json").write_text(json.dumps(report, indent=2))
            await drain_and_probe(client, url, root, args)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


async def run_server(args, repo, root, candidate, manifest):
    import httpx
    from validate_paged_occurrence_regression import send

    child = launch(args, repo, root, candidate)
    url = f"http://127.0.0.1:{args.port}"
    records = []
    try:
        async with httpx.AsyncClient(timeout=7200, trust_env=False) as client:
            deadline = time.monotonic() + 1200
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Server exited: {root / 'server.log'}")
                try:
                    if (await client.get(url + "/v1/models", timeout=2)).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup timed out")
                await asyncio.sleep(2)
            for offset in range(0, args.requests, args.concurrency):
                payloads = [json.loads((args.input / f"case-{i:02d}.json").read_text())
                            for i in range(offset, min(args.requests, offset + args.concurrency))]
                # Prepare the immediately preceding conversation states outside timing.
                for index, payload in enumerate(payloads):
                    if args.stress_pressure:
                        seed = copy.deepcopy(payload)
                        seed["messages"] = seed["messages"][:2 + 2 * args.rolling_keep]
                        seed.update(workload_interface(seed["messages"], args.rolling_keep, args.workload))
                        seed["max_tokens"] = 1
                        row = await send(client, url + "/v1/chat/completions", seed,
                                         f"seed-{offset + index}")
                        if row.get("status_code") != 200:
                            raise RuntimeError(f"Historical seed failed: {row}")
                    warm = copy.deepcopy(payload)
                    warm["messages"] = warm["messages"][:-2]
                    warm.update(workload_interface(warm["messages"], args.rolling_keep, args.workload))
                    warm["max_tokens"] = 1
                    row = await send(client, url + "/v1/chat/completions", warm, f"prepare-{offset + index}")
                    if row.get("status_code") != 200:
                        raise RuntimeError(f"Preparation failed: {row}")
                write_control(root / "wave.json", {"id": offset, "count": len(payloads)})
                if args.stress_pressure:
                    write_control(root / "numerical.json", {"label": f"wave-{offset}",
                                  "pressure": True, "require_new_drop": True,
                                  "batch_size": len(payloads)})
                started = time.perf_counter()
                rows = await asyncio.gather(*[send(client, url + "/v1/chat/completions", payload,
                    f"measured-{offset + index}") for index, payload in enumerate(payloads)])
                elapsed = time.perf_counter() - started
                (root / "wave.json").unlink()
                if args.stress_pressure:
                    (root / "numerical.json").unlink()
                if any(row.get("status_code") != 200 for row in rows):
                    raise RuntimeError(f"Wave failed: {rows}")
                records.append({"offset": offset, "elapsed_s": elapsed, "requests": rows})
                (root / "measurements.json").write_text(json.dumps(records, indent=2))
                print(json.dumps({"root": str(root), "wave": offset, "elapsed_s": elapsed}), flush=True)
            await drain_and_probe(client, url, root, args)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
    events = [json.loads(line) for path in root.glob("events-*.jsonl") for line in path.read_text().splitlines()]
    bs8 = [e for e in events if e["kind"] == "forward" and e["size"] == 8
           and all(q[2] - q[1] > 1 and q[3] >= args.min_full_tokens for q in e["queries"])]
    graph_events = [e for e in events if e["kind"] == "forward" and e["graph"]
                    and all(q[3] >= args.min_full_tokens for q in e["queries"])]
    graph = bool(graph_events)
    if args.stress_pressure:
        expected = (args.requests + args.concurrency - 1) // args.concurrency
        pressures = [e for e in events if e["kind"] == "pressure"]
        if len(pressures) != expected * len(args.gpus.split(',')):
            raise RuntimeError("Not every stress wave evicted dropped pages under live locks")
    graph_bs8 = sum(e["size"] == 8 for e in graph_events) // len(args.gpus.split(','))
    elapsed = sum(r["elapsed_s"] for r in records)
    summary = {"elapsed_s": elapsed, "requests_per_second": args.requests / elapsed,
               "full_tokens_per_second": sum(r["full_tokens"] for r in manifest) / elapsed,
               "observed_bs8_prefill": len(bs8) // len(args.gpus.split(',')),
               "observed_bs8_decode": graph_bs8, "graph_replay": graph}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.require_forward_bs8 and summary["observed_bs8_prefill"] < 3:
        raise RuntimeError(f"Insufficient actual bs=8 prefill: {summary}")
    if args.require_forward_bs8 and not graph_bs8:
        raise RuntimeError(f"No actual bs=8 decode graph replay: {summary}")
    if not graph:
        raise RuntimeError("No normal decode CUDA graph replay observed")
    return summary


async def run(args):
    args.output = args.output.resolve()
    args.repo = args.repo.resolve()
    if args.repo == args.output or args.repo in args.output.parents:
        raise ValueError("Experimental outputs must be outside the repository")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.suite == "recovery":
        await run_recovery(args)
        return
    if args.input is None:
        args.input = args.output / "inputs"
        args.input.mkdir()
        manifest = prepare(args, args.input)
    else:
        manifest = json.loads((args.input / "manifest.json").read_text())
    if len(manifest) != args.requests or any(r["full_tokens"] < args.min_full_tokens for r in manifest):
        raise ValueError("Input manifest does not meet full-token/request requirements")
    results = []
    for rep in range(args.repetitions):
        for candidate in ([False, True] if rep % 2 == 0 else [True, False]):
            if not candidate and args.baseline is None:
                continue
            if candidate and args.baseline_only:
                continue
            repo = args.repo if candidate else args.baseline.resolve()
            root = args.output / f"rep-{rep}-{'candidate' if candidate else 'baseline'}"
            summary = await run_server(args, repo, root, candidate, manifest)
            results.append({"rep": rep, "candidate": candidate, **summary})
            (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    if args.baseline and not args.baseline_only:
        gains = [next(r["requests_per_second"] for r in results if r["rep"] == rep and r["candidate"])
                 / next(r["requests_per_second"] for r in results if r["rep"] == rep and not r["candidate"])
                 for rep in range(args.repetitions)]
        report = {"paired_throughput_ratios": gains, "median_ratio": statistics.median(gains),
                  "repeatable_gain": len(gains) >= 3 and all(g > 1 for g in gains)}
        if args.regression_limit is not None:
            report["regression_limit"] = args.regression_limit
            report["regression_pass"] = len(gains) >= 3 and all(
                g >= 1 - args.regression_limit for g in gains)
        (args.output / "throughput.json").write_text(json.dumps(report, indent=2))
        if not report.get("regression_pass", report["repeatable_gain"]):
            raise RuntimeError(f"Required repeatable throughput improvement not established: {report}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server
        launch_server()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["stress", "recovery"], default="stress")
    parser.add_argument("--recovery-single-reposition", action="store_true",
                        help="Directed recovery control: later Drops preserve source positions.")
    parser.add_argument("--recovery-tool-choice", choices=["auto", "none"])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tool-call-parser", default="gpt-oss")
    parser.add_argument("--reasoning-parser", default="gpt-oss")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, default=30714)
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--memory-ratio", type=float, default=0.30)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--min-full-tokens", type=int, default=131073)
    parser.add_argument("--rolling-keep", type=int, default=12)
    parser.add_argument("--workload", choices=["rolling-reposition", "rolling-drop", "no-drop"],
                        default="rolling-reposition")
    parser.add_argument("--regression-limit", type=float,
                        help="For a regression control, allow at most this throughput loss per pair.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--require-forward-bs8", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--stress-pressure", action="store_true",
                        help="Candidate-only seeded-history live-lock reclamation, not a throughput comparison.")
    parser.add_argument("--mode", choices=["paged-occurrence", "staged"], default="paged-occurrence")
    args = parser.parse_args()
    if args.stress_pressure and (args.baseline or args.suite != "stress"
                                 or args.workload != "rolling-reposition"):
        parser.error("--stress-pressure requires candidate-only rolling-reposition stress")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    used = {int(line.split(',')[0]): int(line.split(',')[1]) for line in memory.splitlines()}
    if any(used[int(g)] > 100 for g in args.gpus.split(',')):
        raise RuntimeError("Selected GPUs are in use")
    asyncio.run(run(args))


if os.environ.get("MINISGL_DAE_OBSERVER"):
    install_observers()
if __name__ == "__main__":
    main()
