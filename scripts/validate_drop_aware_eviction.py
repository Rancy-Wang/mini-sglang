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


def prepare(args, root):
    from transformers import AutoTokenizer
    from minisgl.core import SamplingParams
    from minisgl.message import TokenizeMsg
    from minisgl.tokenizer.tokenize import TokenizeManager

    manager = TokenizeManager(AutoTokenizer.from_pretrained(args.model, local_files_only=True),
                              radix_drop_key_mode="delta-marker")
    manifest = []
    for case in range(args.requests):
        rounds, repetitions = 34 + case % 6, 220
        while True:
            messages = make_messages(case, rounds, repetitions)
            schedule = rolling_interface(messages, args.rolling_keep)
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
    from minisgl.scheduler.prefill import PrefillManager

    root = Path(os.environ["MINISGL_DAE_OBSERVER"])
    audit = os.environ.get("MINISGL_DAE_AUDIT") == "1"
    original_forward = Engine.forward_batch
    original_idle = Scheduler.run_when_idle
    original_schedule = PrefillManager.schedule_next_batch
    barrier_seen = set()

    def emit(kind, **fields):
        with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, **fields}) + "\n")

    @functools.wraps(original_schedule)
    def schedule(self, budget):
        # Queue a specified test wave before admitting its first prefill. This
        # is identical on both revisions and does not force actual GPU bs=8.
        marker = root / "wave.json"
        if marker.exists():
            spec = json.loads(marker.read_text())
            if spec["id"] not in barrier_seen:
                if len(self.pending_list) < spec["count"]:
                    return None
                barrier_seen.add(spec["id"])
        return original_schedule(self, budget)

    @functools.wraps(original_forward)
    def forward(self, batch, sampling):
        queries = [[r.uid, r.cached_len, r.device_len, r.prompt_tokens] for r in batch.reqs]
        graph = self.graph_runner.can_use_cuda_graph(batch)
        result = original_forward(self, batch, sampling)
        emit("forward", queries=queries, size=batch.size, phase=str(batch.phase), graph=graph)
        return result

    @functools.wraps(original_idle)
    def idle(self):
        original_idle(self)
        cache = self.cache_manager
        emit("idle", free=len(cache.free_slots), total=cache.num_pages,
             evictable=cache.prefix_cache.evictable_size,
             protected=cache.prefix_cache.protected_size,
             eviction=getattr(cache.prefix_cache, "eviction_stats", {}))
        if audit:
            stack = list(cache.prefix_cache.root_node.children.values())
            resident = []
            while stack:
                node = stack.pop()
                assert node.ref_count == 0 and getattr(node, "path_ref_count", 0) == 0
                values = node.value[node.value >= 0]
                resident.append(values)
                stack.extend(node.children.values())
            all_pages = torch.cat([cache.free_slots, *resident])
            assert len(torch.unique(all_pages)) == cache.num_pages

    Engine.forward_batch = forward
    Scheduler.run_when_idle = idle
    PrefillManager.schedule_next_batch = schedule


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
            "--tool-call-parser", "gpt-oss", "--reasoning-parser", "gpt-oss"]
    if candidate:
        argv.append("--drop-aware-eviction")
    with (root / "server.log").open("xb") as log:
        child = subprocess.Popen(argv, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    (root / "launch.json").write_text(json.dumps({"pid": child.pid, "argv": argv,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}, indent=2))
    return child


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
                    warm = copy.deepcopy(payload)
                    warm["messages"] = warm["messages"][:-2]
                    warm.update(rolling_interface(warm["messages"], args.rolling_keep))
                    warm["max_tokens"] = 1
                    row = await send(client, url + "/v1/chat/completions", warm, f"prepare-{offset + index}")
                    if row.get("status_code") != 200:
                        raise RuntimeError(f"Preparation failed: {row}")
                (root / "wave.json").write_text(json.dumps({"id": offset, "count": len(payloads)}))
                started = time.perf_counter()
                rows = await asyncio.gather(*[send(client, url + "/v1/chat/completions", payload,
                    f"measured-{offset + index}") for index, payload in enumerate(payloads)])
                elapsed = time.perf_counter() - started
                (root / "wave.json").unlink()
                if any(row.get("status_code") != 200 for row in rows):
                    raise RuntimeError(f"Wave failed: {rows}")
                records.append({"offset": offset, "elapsed_s": elapsed, "requests": rows})
                (root / "measurements.json").write_text(json.dumps(records, indent=2))
                print(json.dumps({"root": str(root), "wave": offset, "elapsed_s": elapsed}), flush=True)
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
    graph = any(e["kind"] == "forward" and e["graph"] for e in events)
    elapsed = sum(r["elapsed_s"] for r in records)
    summary = {"elapsed_s": elapsed, "requests_per_second": args.requests / elapsed,
               "full_tokens_per_second": sum(r["full_tokens"] for r in manifest) / elapsed,
               "observed_bs8_prefill": len(bs8) // len(args.gpus.split(',')), "graph_replay": graph}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.require_forward_bs8 and summary["observed_bs8_prefill"] < 3:
        raise RuntimeError(f"Insufficient actual bs=8 prefill: {summary}")
    if not graph:
        raise RuntimeError("No normal decode CUDA graph replay observed")
    return summary


async def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.repo = args.repo.resolve()
    if args.repo == args.output or args.repo in args.output.parents:
        raise ValueError("Experimental outputs must be outside the repository")
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
            repo = args.repo if candidate else args.baseline.resolve()
            root = args.output / f"rep-{rep}-{'candidate' if candidate else 'baseline'}"
            summary = await run_server(args, repo, root, candidate, manifest)
            results.append({"rep": rep, "candidate": candidate, **summary})
            (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    if args.baseline:
        gains = [next(r["requests_per_second"] for r in results if r["rep"] == rep and r["candidate"])
                 / next(r["requests_per_second"] for r in results if r["rep"] == rep and not r["candidate"])
                 for rep in range(args.repetitions)]
        report = {"paired_throughput_ratios": gains, "median_ratio": statistics.median(gains),
                  "repeatable_gain": len(gains) >= 3 and all(g > 1 for g in gains)}
        (args.output / "throughput.json").write_text(json.dumps(report, indent=2))
        if not report["repeatable_gain"]:
            raise RuntimeError(f"Required repeatable throughput improvement not established: {report}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server
        launch_server()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["stress"], default="stress")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, default=30714)
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--memory-ratio", type=float, default=0.30)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--min-full-tokens", type=int, default=131073)
    parser.add_argument("--rolling-keep", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--require-forward-bs8", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--mode", choices=["paged-occurrence", "staged"], default="paged-occurrence")
    args = parser.parse_args()
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
