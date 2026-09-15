"""Bounded BCP TP2 TTFT experiment, PLAN-CS-20260915-R1.

One nested cohort per concurrency. Recorded messages are data, never executed.
All experiment artifacts must live outside the Git repository.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "python")]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def external(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Experiment artifacts must be outside the repository")
    return path


def bounded_turns(value):
    value = int(value)
    if not 1 <= value <= 12:
        raise ValueError("Every conversation must contain 1..12 turns")
    return value


def rolling_interface(messages, keep=8):
    tools = [i for i, message in enumerate(messages) if message.get("role") == "tool"]
    drops = {str(event): [tools[n-keep]] for n, event in enumerate(tools) if n >= keep}
    return {"drop_message": drops, "reposition": [int(k) for k in drops]} if drops else {}


def choose_cases(rows, turns):
    """No synthetic padding or late-turn seed; first T assistant queries only."""
    eligible = {}
    for row in rows:
        trajectory = row.get("trajectory", [])
        ends = [i for i, msg in enumerate(trajectory) if msg.get("role") == "assistant"]
        if len(ends) < turns:
            continue
        if not rolling_interface(trajectory[:ends[turns-1]]):
            continue
        case_id = str(row["case_id"])
        eligible.setdefault(case_id, dict(row, case_id=case_id, ends=ends[:turns]))
    selected = sorted(eligible.values(), key=lambda row: row["case_id"])[:8]
    if len(selected) != 8:
        raise ValueError(f"Need 8 distinct cases with active K8 Drop by turn {turns}; got {len(selected)}")
    return selected


def prepare(args):
    from minisgl.benchmark.reposition_bcp import browsecomp_plus_tools
    from minisgl.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer

    root = external(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in args.source:
        data = path.read_bytes()
        source_sha = hashlib.sha256(data).hexdigest()
        for number, line in enumerate(data.splitlines(), 1):
            row = json.loads(line)
            rows.append(dict(row, source=str(path), source_line=number, source_sha256=source_sha))
    selected = choose_cases(rows, args.turns)
    manager = TokenizeManager(AutoTokenizer.from_pretrained(args.model, local_files_only=True),
                              radix_drop_key_mode="delta-marker")
    tools = browsecomp_plus_tools()
    cases = []
    for case in selected:
        records = []
        for turn, end in enumerate(case["ends"]):
            messages = case["trajectory"][:end]
            tokens, _, _ = manager._render_harmony_message_drop(messages, enable_thinking=None, tools=tools)
            if len(tokens) + 32 > 131072:
                raise ValueError(f"Case {case['case_id']} turn {turn} exceeds context limit")
            records.append(dict(turn=turn, end=end, full_tokens=len(tokens),
                                messages_sha256=digest(messages),
                                tool_responses=sum(m.get("role") == "tool" for m in messages),
                                rolling=rolling_interface(messages)))
        artifact = dict(case_id=case["case_id"], trajectory=case["trajectory"][:case["ends"][-1]],
                        turns=records, source=case["source"], source_line=case["source_line"],
                        source_sha256=case["source_sha256"])
        filename = f"case-{len(cases):02d}.json.gz"
        with gzip.open(root / filename, "wt") as stream:
            json.dump(artifact, stream, ensure_ascii=False)
        cases.append(dict(case_id=case["case_id"], file=filename, turns=records))
        print(json.dumps(cases[-1]), flush=True)
    write_json(root / "manifest.json", dict(plan="PLAN-CS-20260915-R1", keep=8,
               turns=args.turns, max_tokens=32, cases=cases, tools=tools, model=args.model,
               cohort_policy="one nested group of C cases, first C of the fixed eight"))


def metrics_values(metrics):
    ttft = (metrics["first_token_generated_ns"]-metrics["request_received_ns"])/1e6
    count = metrics["generated_tokens"]
    tpot = ((metrics["request_finished_ns"]-metrics["first_token_generated_ns"])/1e6/(count-1)
            if count > 1 else None)
    return dict(ttft_ms=ttft, tpot_ms=tpot)


def correlate(requests, events):
    """Exact monotonic-clock partition, no summation of overlapping TP ranks."""
    components, output_tokens = [], {}
    tokenizer = {(e.get("cell"), e["uid"]): e for e in events if e["kind"] == "tokenizer"}
    arrivals = defaultdict(list)
    batches = defaultdict(list)
    releases = {}
    for event in events:
        if event["kind"] == "arrival":
            arrivals[(event["cell"], event["uid"])].append(event)
        elif event["kind"] == "gate_release":
            releases[(event["cell"], event["turn"], event["pid"])] = event["end_ns"]
        elif event["kind"] == "batch":
            for offset, uid in enumerate(event["uids"]):
                batches[(event["cell"], uid, event["pid"])].append((event, offset))
    for row in requests:
        if "uid" not in row:
            continue
        cell, uid = row["cell"], row["uid"]
        tok = tokenizer.get((cell, uid))
        arrival_rows = arrivals.get((cell, uid), [])
        if tok is None or not arrival_rows:
            continue
        # PID-labelled representative, plus retain both ranks in raw evidence.
        arrival = min(arrival_rows, key=lambda e:e["pid"])
        pid = arrival["pid"]
        sequence = sorted(batches.get((cell, uid, pid), []), key=lambda e:e[0]["start_ns"])
        prefill = [e for e, _ in sequence if e["phase"] == "prefill"]
        if not prefill:
            continue
        metrics = row["response"]["server_metrics"]
        points = [metrics["request_received_ns"], tok["start_ns"], tok["end_ns"],
                  arrival["time_ns"], prefill[0]["start_ns"], metrics["first_token_generated_ns"]]
        names = ["frontend_queue", "tokenizer", "tokenizer_to_scheduler", "scheduler_to_forward", "forward_to_first_token"]
        part = dict(zip(names, [(b-a)/1e6 for a,b in zip(points, points[1:])]))
        release = releases.get((cell, row["turn"], pid), arrival["time_ns"])
        part["barrier_wait"] = (release-arrival["time_ns"])/1e6
        part["schedule_after_barrier"] = part["scheduler_to_forward"]-part["barrier_wait"]
        part["sum_ms"] = sum(part[name] for name in names)
        part["ttft_ms"] = row["ttft_ms"]
        part["partition_valid"] = all(a<=b for a,b in zip(points, points[1:])) and abs(part["sum_ms"]-row["ttft_ms"])<1e-6
        components.append(dict(cell=cell, uid=uid, turn=row["turn"], case_id=row["case_id"], pid=pid, **part))
        output_tokens[(row["mode"], row["concurrency"], row["workload"], row["case_id"], row["turn"])] = [
            event["tokens"][offset] for event,offset in sequence if "tokens" in event]
    comparison = []
    for key, tokens in output_tokens.items():
        if key[0] != "detail":
            continue
        baseline = output_tokens.get(("baseline", *key[1:]))
        comparison.append(dict(concurrency=key[1], workload=key[2], case_id=key[3], turn=key[4],
                               baseline_tokens=baseline, detail_tokens=tokens,
                               equal=baseline is not None and tokens==baseline))
    return components, comparison


def ttft_function_totals(requests, events):
    """Inclusive intervals clipped to each wave's TTFT, never sum TP ranks.

    A parent can cross first_token_generated_ns. Its full self/CPU time cannot
    safely be prorated, so retain those only for contained events and flag clips.
    Decode preparation overlapping the first sample belongs to this window too.
    """
    windows = {}
    for row in requests:
        if "uid" not in row:
            continue
        metrics = row["response"]["server_metrics"]
        key = (row["cell"], row["turn"])
        start, end = metrics["request_received_ns"], metrics["first_token_generated_ns"]
        old = windows.get(key, (start, end))
        windows[key] = (min(old[0], start), max(old[1], end))
    totals = defaultdict(lambda: defaultdict(float))
    for event in events:
        if event["kind"] != "function":
            continue
        window = windows.get((event.get("cell"), event.get("turn")))
        if window is None:
            continue
        overlap = min(window[1], event["end_ns"])-max(window[0], event["start_ns"])
        if overlap <= 0:
            continue
        key = (event["cell"], event["turn"], event["pid"], event["name"], event["phase"])
        total = totals[key]
        total["calls"] += 1
        total["inclusive_overlap_ms"] += overlap/1e6
        contained = window[0] <= event["start_ns"] and event["end_ns"] <= window[1]
        total["clipped_calls"] += not contained
        if contained:
            for metric in ("wall_ns", "cpu_ns", "self_ns", "self_cpu_ns"):
                total["contained_"+metric.replace("_ns", "_ms")] += event[metric]/1e6
    return [dict(cell=k[0], turn=k[1], pid=k[2], name=k[3], phase=k[4], **value)
            for k,value in totals.items()]


def launch(args, root):
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    used = dict(tuple(map(int, line.split(","))) for line in usage.splitlines())
    if any(used[gpu] > 100 for gpu in (0, 1)):
        raise RuntimeError(f"GPU 0,1 not idle; refusing launch: {used}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", PYTHONPATH=f"{REPO}:{REPO / 'python'}",
               MINISGL_TTFT_PROFILE_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    runtime = external(args.compile_cache) if args.compile_cache else root / "runtime"
    runtime.mkdir(exist_ok=bool(args.compile_cache))
    for key, name in {"TORCH_EXTENSIONS_DIR": "torch", "TRITON_CACHE_DIR": "triton",
                      "TVM_FFI_CACHE_DIR": "tvm", "CUDA_CACHE_PATH": "cuda"}.items():
        (runtime/name).mkdir(exist_ok=bool(args.compile_cache))
        env[key] = str(runtime/name)
    prefix = Path(sys.executable).parents[1]
    env["PATH"] = str(prefix/"bin") + ":" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix/"lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    for key, compiler in {"CC": "gcc", "CXX": "g++", "NVCC_CCBIN": "gcc"}.items():
        candidate = prefix/"bin"/f"x86_64-conda-linux-gnu-{compiler}"
        if candidate.exists():
            env[key] = str(candidate)
    if "CC" in env:
        env["NVCC_PREPEND_FLAGS"] = "-ccbin=" + env["CC"]
    argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--model-path", args.model,
            "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", "2",
            # Keep room for the frozen cohort's 100k-token MoE activation/workspace.
            # This value is fixed across every C and paired workload in the run.
            "--dtype", "bfloat16", "--disable-pynccl", "--memory-ratio", "0.75",
            "--max-running-requests", "8", "--cuda-graph-max-bs", "8",
            "--max-seq-len-override", "131072", "--max-prefill-length", str(args.chunk),
            "--request-timeout", "1800", "--cache-type", "radix", "--page-size", "1",
            "--attention-backend", "fi", "--radix-drop-key-mode", "delta-marker",
            "--contextual-prefill-mode", "mask", "--reposition-execution-mode", "paged-occurrence",
            "--tool-call-parser", "gpt-oss", "--reasoning-parser", "gpt-oss"]
    write_json(root / "launch.json", dict(argv=argv, env={k:v for k,v in env.items() if k in
               {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH",
                "CC", "CXX", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "CPATH",
                "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH"}},
               head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               gpu_preflight=usage))
    with (root/"server.log").open("w") as log:
        return subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


async def run(args):
    import httpx
    root = external(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    bounded_turns(manifest["turns"])
    if len(manifest["cases"]) != 8 or manifest["keep"] != 8:
        raise ValueError("Expected the approved eight-case K8 manifest")
    cases = []
    for case in manifest["cases"]:
        with gzip.open(args.manifest.parent/case["file"], "rt") as stream:
            cases.append(json.load(stream))
    write_json(root/"manifest.json", manifest)
    write_json(root/"control.json", dict(cell="startup", turn=-1, concurrency=1, detail=False))
    process = launch(args, root)
    url = f"http://127.0.0.1:{args.port}"
    try:
        async with httpx.AsyncClient(timeout=1800, trust_env=False,
                                     limits=httpx.Limits(max_connections=16)) as client:
            deadline = time.monotonic()+1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; inspect server.log")
                try:
                    response = await client.get(url+"/v1/models", timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup timeout")
                await asyncio.sleep(2)
            # Same model/tokenizer instance throughout. Each cell flushes idle KV,
            # retains compile caches/CUDA graphs, and uses exactly one nested cohort.
            for mode in args.modes:
                for concurrency in args.concurrency:
                    for workload in ("no_drop", "rolling"):
                        cell = f"{mode}-c{concurrency}-{workload}"
                        cell_dir = root/cell
                        cell_dir.mkdir()
                        with (cell_dir/"requests.jsonl").open("w") as output:
                            for turn in range(manifest["turns"]):
                                write_json(root/"control.json", dict(cell=cell, turn=turn,
                                           concurrency=concurrency, detail=mode=="detail", gpu_detail=mode in ("detail","gpu"),
                                           barrier=mode!="natural", nvtx=False))
                                async def request(case):
                                    spec = case["turns"][turn]
                                    messages = case["trajectory"][:spec["end"]]
                                    if digest(messages) != spec["messages_sha256"]:
                                        raise ValueError("Frozen message provenance mismatch")
                                    payload = dict(model=args.model, messages=messages, tools=manifest["tools"],
                                                   max_tokens=32, temperature=0, top_p=1, seed=17, stream=False)
                                    if workload == "rolling":
                                        payload.update(rolling_interface(messages))
                                    start = time.perf_counter_ns()
                                    response = await client.post(url+"/v1/chat/completions", json=payload)
                                    try:
                                        body = response.json()
                                    except ValueError:
                                        body = {"non_json_error": response.text}
                                    row = dict(cell=cell, concurrency=concurrency, mode=mode,
                                               workload=workload, case_id=case["case_id"], turn=turn,
                                               source=spec, request_sha256=digest(payload),
                                               client_start_ns=start, client_end_ns=time.perf_counter_ns(),
                                               status=response.status_code, response=body)
                                    if response.status_code != 200:
                                        output.write(json.dumps(row)+"\n"); output.flush()
                                        response.raise_for_status()
                                    metrics = body["server_metrics"]
                                    row.update(metrics_values(metrics))
                                    row["uid"] = int(body["id"].split("-")[-1])
                                    if metrics["prompt_tokens"] != spec["full_tokens"]:
                                        raise ValueError(f"Prompt provenance mismatch: {row}")
                                    if metrics["generated_tokens"] > 32:
                                        raise ValueError("Generated-token limit exceeded")
                                    output.write(json.dumps(row, ensure_ascii=False)+"\n"); output.flush()
                                    print(json.dumps({k:row[k] for k in
                                          ("cell", "turn", "case_id", "uid", "ttft_ms", "tpot_ms")}), flush=True)
                                await asyncio.gather(*(request(case) for case in cases[:concurrency]))
                                # Allow out-of-request buffered observer flush to finish.
                                await asyncio.sleep(0.2)
            write_json(root/"completed.json", dict(completed=True, time_ns=time.perf_counter_ns()))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    report(argparse.Namespace(output=root))


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = external(args.output)
    requests = [json.loads(line) for path in sorted(root.glob("*/requests.jsonl"))
                for line in path.read_text().splitlines()]
    events = [json.loads(line) for path in sorted(root.glob("events-*.jsonl"))
              for line in path.read_text().splitlines()]
    by_cell = defaultdict(list)
    for row in requests:
        by_cell[row["cell"]].append(row)
    components, output_comparison = correlate(requests, events)
    write_json(root/"components.json", components)
    write_json(root/"profile_output_comparison.json", output_comparison)
    write_json(root/"ttft_functions.json", ttft_function_totals(requests, events))
    summary, functions = [], []
    for cell, rows in by_cell.items():
        batches = [e for e in events if e.get("cell") == cell and e["kind"] == "batch"
                   and e["phase"] == "prefill"]
        first_batches = {}
        for b in sorted(batches, key=lambda e:e["start_ns"]):
            first_batches.setdefault((b["pid"], b["turn"]), b)
        requested_c = rows[0]["concurrency"]
        summary.append(dict(cell=cell, requests=len(rows),
                       mean_ttft_ms=statistics.mean(r["ttft_ms"] for r in rows),
                       late_ttft_ms=statistics.mean(r["ttft_ms"] for r in rows if r["turn"] >= 9),
                       first_prefill_batch_sizes={str(k):b["size"] for k,b in first_batches.items()},
                       actual_batch_gate=len(first_batches)==2*len({r["turn"] for r in rows})
                       and all(b["size"]==requested_c for b in first_batches.values())))
        # Totals by rank, not summed across TP. Inclusive vs exclusive are separate.
        totals = defaultdict(lambda: defaultdict(float))
        for event in events:
            if event.get("cell") != cell or event["kind"] != "function":
                continue
            key = (event["pid"], event["turn"], event["name"], event["phase"])
            totals[key]["calls"] += 1
            for metric in ("wall_ns", "cpu_ns", "self_ns", "self_cpu_ns"):
                totals[key][metric] += event[metric]
        functions.extend(dict(cell=cell, pid=k[0], turn=k[1], name=k[2], phase=k[3], **v)
                         for k,v in totals.items())
    write_json(root/"summary.json", summary)
    write_json(root/"functions.json", functions)
    for mode in sorted({r["mode"] for r in requests}):
        for concurrency in sorted({r["concurrency"] for r in requests}):
            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
            for workload, label in (("no_drop", "No drop"), ("rolling", "Rolling drop + reposition (K=8)")):
                rows = [r for r in requests if r["mode"]==mode and r["concurrency"]==concurrency and r["workload"]==workload]
                for axis, metric in zip(axes, ("ttft_ms", "tpot_ms")):
                    turns = sorted({r["turn"] for r in rows})
                    values = [statistics.mean(r[metric] for r in rows if r["turn"]==t and r[metric] is not None) for t in turns]
                    axis.plot(turns, values, "o-" if workload=="no_drop" else "s-", label=label)
                    axis.set_ylabel(metric.replace("_ms", " (ms)").upper()); axis.grid(alpha=.2)
                    if workload == "rolling":
                        effective = [r["turn"] for r in rows if r["source"]["rolling"]]
                        if effective:
                            axis.axvline(min(effective), color="gray", linestyle="--")
                            axis.axvspan(min(effective), max(turns), color="teal", alpha=.04)
            axes[0].legend(); axes[1].set_xlabel("Turn (zero-based); all measured turns retained")
            fig.suptitle(f"GPT-OSS-120B TP=2 | concurrency={concurrency} | {mode} | one cohort")
            fig.tight_layout(); fig.savefig(root/f"{mode}-c{concurrency}.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, action="append", required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--turns", type=bounded_turns, default=12)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--port", type=int, default=30915)
    run_parser.add_argument("--chunk", type=int, default=65536)
    run_parser.add_argument("--compile-cache", type=Path, help="Reuse an idle previous experiment's compile caches")
    run_parser.add_argument("--concurrency", type=int, choices=(1,2,4,8), nargs="+", default=[1,2,4,8])
    run_parser.add_argument("--modes", choices=("baseline", "detail", "natural", "gpu"), nargs="+", default=["baseline", "detail"])
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        report(args)


if os.environ.get("MINISGL_TTFT_PROFILE_ROOT"):
    from scripts.bcp_ttft_profile_hooks import install
    install()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.argv.pop(1)
        from minisgl.server.launch import launch_server
        launch_server()
    else:
        main()
