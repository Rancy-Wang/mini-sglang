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
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit


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
    for key in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE",
                "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR"):
        path = root / key.lower()
        path.mkdir()
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
        record["response"] = response.json()
        metrics = record["response"].get("server_metrics")
        if metrics:
            record["ttft_ms"] = (metrics["first_token_generated_ns"] -
                                 metrics["request_received_ns"]) / 1e6
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


async def wave(args) -> None:
    import httpx

    selected = json.loads((args.input / "preflight.json").read_text())["selected"][:args.bs]
    assert len(selected) == args.bs
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False,
                                 limits=httpx.Limits(max_connections=args.bs)) as client:
        results = await asyncio.gather(*[
            send(client, args.url, json.loads((args.input / f"case_{item['case_id']}.json").read_text()),
                 item["case_id"]) for item in selected
        ])
    with args.output.open("x") as log:
        json.dump({"http_concurrency": args.bs, "actual_gpu_batch": "requires server evidence",
                   "results": results}, log, ensure_ascii=False, indent=2)
    print(json.dumps([{key: value for key, value in row.items() if key != "response"}
                      for row in results]), flush=True)
    if any(row.get("status_code") != 200 for row in results):
        raise RuntimeError("At least one concurrent request failed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start")
    launch.add_argument("--repo", type=Path, required=True)
    launch.add_argument("--model", required=True)
    launch.add_argument("--gpus", required=True)
    launch.add_argument("--port", type=int, required=True)
    launch.add_argument("--chunk", type=int, default=32768)
    launch.add_argument("--memory-ratio", type=float, default=0.90)
    stop = commands.add_parser("stop")
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--request", type=Path, required=True)
    replay_parser.add_argument("--original-terminal", action="store_true")
    replay_parser.add_argument("--terminal-only", action="store_true")
    wave_parser = commands.add_parser("wave")
    wave_parser.add_argument("--input", type=Path, required=True)
    wave_parser.add_argument("--bs", type=int, choices=[4, 8], required=True)
    for command in (launch, stop, replay_parser, wave_parser):
        command.add_argument("--output", type=Path, required=True)
    for command in (replay_parser, wave_parser):
        command.add_argument("--url", type=loopback_url, required=True)
        command.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.command == "start":
        start_server(args)
    elif args.command == "stop":
        stop_server(args)
    elif args.command == "replay":
        asyncio.run(replay(args))
    else:
        asyncio.run(wave(args))


if __name__ == "__main__":
    main()
