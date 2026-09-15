#!/usr/bin/env python3
"""Launch isolated TP2 services and the approved ten-cell throughput matrix.

The benchmark client is system independent. This optional launcher is mini-sglang
specific. Only its private child process groups are terminated. Resume skips
validated cells on the same source HEAD and input hash.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests/benchmark"))
from test_throughput import DEFAULT_INPUT, digest, write_json


def install_observer():
    """No per-token callbacks: capacity and page lifecycle checks only at idle."""
    import torch
    import torch.distributed as dist
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    root = Path(os.environ["MINISGL_THROUGHPUT_OBSERVER"])
    old_init, old_idle = Engine.__init__, Scheduler.run_when_idle
    seen = set()
    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        write_json(root / f"ready-{os.getpid()}.json", {"num_pages": self.num_pages})
    def idle(self):
        old_idle(self)
        group = self.engine.tp_cpu_group
        command = [None]
        path = root / "audit-command.json"
        if dist.get_rank(group) == 0 and path.exists():
            command[0] = json.loads(path.read_text())
        dist.broadcast_object_list(command, src=0, group=group)
        if command[0] is None or command[0]["id"] in seen:
            return
        label = command[0]["id"]
        seen.add(label)
        cache, tree = self.cache_manager, self.cache_manager.prefix_cache
        result = {"id": label, "num_pages": cache.num_pages, "eviction": dict(tree.eviction_stats)}
        try:
            stack = list(tree.root_node.children.values())
            resident = []
            while stack:
                node = stack.pop()
                assert node.ref_count == 0 and node.path_ref_count == 0
                resident.append(node.value[node.value >= 0])
                stack.extend(node.children.values())
            free = cache.free_slots
            pages = torch.cat([free, *resident])
            assert len(torch.unique(pages)) == cache.num_pages
            assert bool(torch.all((pages >= 0) & (pages < cache.num_pages)))
            assert len(torch.unique(free)) == len(free)
            if resident:
                assert not bool(torch.isin(free, torch.cat(resident)).any())
            assert cache.available_size == cache.num_pages
            allocated = cache._allocate(cache.num_pages)
            assert len(torch.unique(allocated)) == cache.num_pages
            cache.free_occurrence_pages(allocated)
            result.update(passed=True, free_after=len(cache.free_slots))
            tree.eviction_stats.update(leaf_pages=0, drop_pages=0, hole_fills=0)
        except Exception as exc:
            result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        write_json(root / f"audit-{label}-{os.getpid()}.json", result)
    Engine.__init__, Scheduler.run_when_idle = init, idle


def launch(args, group, gpus, port, root):
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    usage = dict((i.strip(), int(v)) for i, v in (line.split(",") for line in used.splitlines()))
    if any(usage[i] > 100 for i in gpus.split(",")):
        raise RuntimeError(f"Selected GPUs occupied: {usage}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONPATH=str(REPO / "python"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MINISGL_THROUGHPUT_OBSERVER=str(root))
    for key in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR"):
        directory = Path(args.output) / "runtime" / group / key.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    for key, suffix in (("CC", "gcc"), ("CXX", "g++")):
        compiler = Path(sys.executable).parent / f"x86_64-conda-linux-gnu-{suffix}"
        if compiler.exists():
            env[key] = str(compiler)
    if "CXX" in env:
        env["NVCC_PREPEND_FLAGS"] = f"-ccbin={env['CXX']}"
    argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--model-path", args.model,
            "--host", "127.0.0.1", "--port", str(port), "--tp-size", "2", "--dtype", "bfloat16",
            "--disable-pynccl", "--memory-ratio", "0.9", "--max-running-requests", "16",
            "--cuda-graph-max-bs", "16", "--max-seq-len-override", "131072",
            "--max-prefill-length", "16384", "--request-timeout", "7200", "--cache-type", "radix",
            "--page-size", "1", "--attention-backend", "fi", "--radix-drop-key-mode", "delta-marker",
            "--contextual-prefill-mode", "mask", "--reposition-execution-mode", "paged-occurrence",
            "--tool-call-parser", "gpt-oss", "--reasoning-parser", "gpt-oss"]
    if group == "drop-aware":
        argv.append("--drop-aware-eviction")
    if args.pages:
        argv.extend(["--num-pages", str(args.pages)])
    with (root / "server.log").open("x") as stream:
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream, stderr=stream,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(root / "launch.json", dict(argv=argv, pid=child.pid, gpus=gpus, head=args.head,
                                         env={k: env[k] for k in env if k.endswith("_DIR") or k in ("CUDA_VISIBLE_DEVICES", "CC", "CXX")}))
    return child


def stop(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


async def audit(session, url, root, label, model):
    write_json(root / "audit-command.json", {"id": label})
    async with session.post(url + "/v1/chat/completions", json=dict(
        model=model, messages=[{"role": "user", "content": "Reply OK."}], max_tokens=1)) as response:
        response.raise_for_status()
        await response.read()
    deadline = time.monotonic() + 300
    while len(list(root.glob(f"audit-{label}-*.json"))) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Idle page audit {label} timed out")
        await asyncio.sleep(1)
    rows = [json.loads(p.read_text()) for p in root.glob(f"audit-{label}-*.json")]
    if not all(r["passed"] for r in rows):
        raise RuntimeError(f"Page integrity failed: {rows}")
    return rows


async def group_run(args, group, gpus, port):
    import aiohttp
    cells = [(1,4,True),(2,4,False),(5,8,True),(6,8,False),(9,16,True),(10,16,False)] if group == "drop-aware" else [(3,4,True),(4,4,False),(7,8,False),(8,8,True)]
    if args.smoke:
        cells = [(0,2,True),(-1,2,True)]
    group_root = Path(args.output) / group
    group_root.mkdir(parents=True, exist_ok=True)
    root = group_root / ("server-" + str(time.time_ns()))
    root.mkdir()
    child = launch(args, group, gpus, port, root)
    url = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as session:
            deadline = time.monotonic() + 1800
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"Server exited: {root / 'server.log'}")
                try:
                    async with session.get(url + "/v1/models", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 1800s")
                await asyncio.sleep(5)
            ready = [json.loads(p.read_text()) for p in root.glob("ready-*.json")]
            if len(ready) != 2 or len({r["num_pages"] for r in ready}) != 1:
                raise RuntimeError(f"Invalid capacity evidence: {ready}")
            write_json(group_root / "capacity.json", ready[0])
            for number, concurrency, drop in cells:
                current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
                if current_head != args.head:
                    raise RuntimeError(f"Repository changed during matrix: {args.head} -> {current_head}")
                cell_root = group_root / (f"{number:02d}_C{concurrency}_" + ("drop" if drop else "no_drop"))
                cell_root.mkdir(exist_ok=True)
                status_path = cell_root / "status.json"
                if status_path.exists():
                    old = json.loads(status_path.read_text())
                    if old.get("state") == "completed" and old.get("head") == args.head and old.get("input_hash") == args.input_hash:
                        continue
                label = f"{number}-{time.time_ns()}"
                await audit(session, url, root, "initial-" + label, args.model)
                status = dict(cell=number, concurrency=concurrency, drop=drop, eviction=group,
                              state="running", head=args.head, input_hash=args.input_hash,
                              gpus=gpus, pages=ready[0]["num_pages"], started_at=time.time())
                write_json(status_path, status)
                argv = [sys.executable, str(REPO / "tests/benchmark/test_throughput.py"), "run",
                        "--host", "127.0.0.1", "--port", str(port), "--model", args.model,
                        "--requests-path", args.requests_path, "--output", str(cell_root),
                        "--concurrency", str(concurrency), "--num-requests", str(concurrency * 3),
                        "--drop" if drop else "--no-drop"]
                if args.smoke:
                    argv.extend(["--max-token-len", "16", "--num-requests", "4" if number == 0 else "2"])
                    argv.extend(["--smoke-max-turns", "2"] if number == 0 else ["--smoke-long-last"])
                with (cell_root / f"client-{time.time_ns()}.log").open("w") as stream:
                    proc = await asyncio.create_subprocess_exec(*argv, stdout=stream, stderr=stream)
                    code = await proc.wait()
                latest = json.loads((cell_root / "latest.json").read_text()) if (cell_root / "latest.json").exists() else {}
                status.update(finished_at=time.time(), exit_code=code, result=latest.get("result"))
                if code:
                    status["state"] = "failed"
                    write_json(status_path, status)
                    raise RuntimeError(f"Cell {number} failed: {cell_root}")
                status["audit"] = await audit(session, url, root, "final-" + label, args.model)
                status["state"] = "completed"
                write_json(status_path, status)
                print(json.dumps(status), flush=True)
    finally:
        await asyncio.to_thread(stop, child)


async def run(args):
    args.head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    args.input_hash = digest(json.loads(Path(args.requests_path).read_text()))
    root = Path(args.output).resolve()
    if REPO == root or REPO in root.parents:
        raise ValueError("Output must be outside repository")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "matrix.json", dict(head=args.head, input_hash=args.input_hash,
                                         model=args.model, smoke=args.smoke, rounds=3))
    jobs = [group_run(args, "drop-aware", "0,1", args.port)]
    if not args.smoke:
        jobs.append(group_run(args, "ordinary", "2,3", args.port + 1))
    outcomes = await asyncio.gather(*jobs, return_exceptions=True)
    errors = [str(x) for x in outcomes if isinstance(x, BaseException)]
    write_json(root / "matrix_status.json", {"state": "failed" if errors else "completed", "errors": errors})
    if errors:
        raise SystemExit("; ".join(errors))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server
        launch_server()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests-path", default=DEFAULT_INPUT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=30924)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--smoke", action="store_true")
    asyncio.run(run(parser.parse_args()))


if os.environ.get("MINISGL_THROUGHPUT_OBSERVER"):
    install_observer()
if __name__ == "__main__":
    main()
