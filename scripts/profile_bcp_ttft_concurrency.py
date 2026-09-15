"""Bounded BCP TP2 TTFT experiment, PLAN-CS-20260915-R1.

One nested cohort per concurrency. Recorded messages are data, never executed.
All experiment artifacts must live outside the Git repository.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
import os
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("MINISGL_SOURCE_REPO", Path(__file__).resolve().parents[1])).resolve()
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


def mode_control(mode, turn):
    return dict(detail=mode == "detail", gpu_detail=mode in ("detail", "gpu"),
                barrier=mode not in ("natural", "overlap"), nvtx=False,
                fixed_split=mode in ("fixed", "fixed-overlap"),
                overlap=mode in ("overlap", "fixed-overlap") and turn in (9, 10, 11))


def overlap_partition(engine_end, forward_end, collect_start, sync_start, sync_end, recorded):
    points = [engine_end, forward_end, collect_start, sync_start, sync_end, recorded]
    if points != sorted(points):
        raise ValueError("Non-monotonic overlap interval; do not invent attribution")
    names = ["post_engine", "control_gap", "collect_before_sync", "copy_wait", "record_gap"]
    return dict(zip(names, (b-a for a,b in zip(points, points[1:]))))


def nsys_rows(path):
    """Read an exported Nsight database without inventing device timestamps.

    CUDA correlation IDs are process-local. NVTX and CUDA records already share
    the Nsight time axis; the clock marks align our perf_counter request metrics.
    """
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = {r[0]:r[1] for r in db.execute("SELECT id,value FROM StringIds")}
        ranges, clocks = [], []
        for raw in db.execute("SELECT * FROM NVTX_EVENTS"):
            r = dict(raw)
            text = r.get("text") or strings.get(r.get("textId"), "")
            try:
                label = json.loads(text)
            except (ValueError, TypeError):
                continue
            if not isinstance(label, dict):
                continue
            if "r3_clock" in label:
                clocks.append(dict(start=r["start"], **label))
            if "r3" in label and r["end"] is not None:
                ranges.append(dict(start=r["start"], end=r["end"],
                                   pid=(r["globalTid"] >> 24) & 0xffffff,
                                   tid=r["globalTid"] & 0xffffff, **label))
        apis = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for raw in db.execute("SELECT * FROM "+table):
                r = dict(raw)
                r.update(name=strings[r["nameId"]], pid=(r["globalTid"] >> 24) & 0xffffff,
                         tid=r["globalTid"] & 0xffffff)
                apis.append(r)
        correlation = {(r["pid"],r["correlationId"]):r for r in apis}
        gpu = []
        for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table not in tables:
                continue
            for raw in db.execute("SELECT * FROM "+table):
                r = dict(raw)
                r.update(kind=kind, pid=(r["globalPid"] >> 24) & 0xffffff)
                if kind == "kernel":
                    r["name"] = strings.get(r.get("demangledName"), "unknown kernel")
                else:
                    r["name"] = kind+":"+str(r.get("copyKind", ""))
                api = correlation.get((r["pid"],r.get("correlationId")))
                if api:
                    r.update(api_start=api["start"], api_end=api["end"], api_name=api["name"])
                    owners = [s for s in ranges if s["pid"] == api["pid"] and s["tid"] == api["tid"]
                              and s["start"] <= api["start"] <= s["end"]]
                    if owners:
                        owner = min(owners, key=lambda s:s["end"]-s["start"])
                        r.update(owner=owner["r3"], uids=owner["uids"], source=owner.get("source"))
                gpu.append(r)
        return dict(ranges=ranges, clocks=clocks, apis=apis, gpu=gpu)


def trace_clock_offset(clocks, host_events):
    """A mark occurred inside [before_ns, after_ns]; retain alignment error."""
    host = {(e["pid"], e["before_ns"]):e for e in host_events if e["kind"] == "trace_clock"}
    matches = [(c, host[(c["pid"],c["r3_clock"])]) for c in clocks
               if (c["pid"],c["r3_clock"]) in host]
    if not matches:
        raise ValueError("No bracketed host/Nsight clock marker; cannot align request metrics")
    lower = max(c["start"]-e["after_ns"] for c,e in matches)
    upper = min(c["start"]-e["before_ns"] for c,e in matches)
    if lower > upper:
        raise ValueError("Clock brackets disagree across ranks; alignment requires investigation")
    sample = matches[0][1]
    return dict(offset_ns=(lower+upper)//2, uncertainty_ns=(upper-lower+1)//2,
                cell=sample["cell"], turn=sample["turn"], marks=len(matches))


def execution_union_ns(rows, lo, hi):
    """Clipped execution union on one GPU, never sum overlapping streams/ranks."""
    end, total = lo, 0
    for row in sorted(rows, key=lambda r:r["start"]):
        start, stop = max(lo, row["start"]), min(hi, row["end"])
        if stop > max(start, end):
            total += stop-max(start, end)
            end = stop
    return total


def blocked_copy_evidence(trace):
    """Attribute each rank's longest compact operator to its own host thread/API."""
    result = []
    for pid in sorted({r["pid"] for r in trace["ranges"]}):
        ops = [r for r in trace["ranges"] if r["pid"] == pid and r["r3"].startswith("compact.op.")]
        if not ops:
            continue
        op = max(ops, key=lambda r:r["end"]-r["start"])
        apis = [a for a in trace["apis"] if a["pid"] == pid and a["tid"] == op["tid"]
                and op["start"] <= a["start"] and a["end"] <= op["end"]]
        if not apis:
            continue
        api = max(apis, key=lambda a:a["end"]-a["start"])
        related = [g for g in trace["gpu"] if g["pid"] == pid
                   and g.get("correlationId") == api["correlationId"]]
        devices = sorted({g["deviceId"] for g in related})
        execution = []
        for device in devices:
            gpu = [g for g in trace["gpu"] if g["deviceId"] == device
                   and g["end"] > api["start"] and g["start"] < api["end"]]
            kernels = defaultdict(list)
            for g in gpu:
                if g["kind"] == "kernel":
                    kernels[g["name"]].append(g)
            totals = [dict(name=name, count=len(gs), execution_union_ns=
                           execution_union_ns(gs, api["start"], api["end"])) for name,gs in kernels.items()]
            execution.append(dict(device=device, active_union_ns=execution_union_ns(gpu, api["start"], api["end"]),
                top_kernels=sorted(totals,key=lambda r:r["execution_union_ns"],reverse=True)[:10]))
        result.append(dict(pid=pid, operator=op, api=api, correlated_gpu=related, execution=execution))
    return result


def overlap_report(args):
    root = external(args.output)
    events = [json.loads(line) for path in root.glob("events-*.jsonl")
              for line in path.read_text().splitlines()]
    requests = [json.loads(line) for path in root.glob("*/requests.jsonl")
                for line in path.read_text().splitlines()]
    ranks = {e["pid"]:e["rank"] for e in events if e["kind"] == "communication_config"}
    summaries = []
    for path in sorted(root.glob("overlap*.sqlite")):
        trace = nsys_rows(path)
        clock = trace_clock_offset(trace["clocks"], events)
        rows = [r for r in requests if r["cell"] == clock["cell"] and r["turn"] == clock["turn"]]
        if len(rows) != 8 or len({g["deviceId"] for g in trace["gpu"]}) != 2:
            raise ValueError("Missing requests or one TP GPU in trace")
        req_by_uid = {r["uid"]:r for r in rows}
        host = [e for e in events if e.get("cell") == clock["cell"] and e.get("turn") == clock["turn"]]
        batches = sorted([e for e in host if e["kind"] == "batch" and e["phase"] == "prefill"],
                         key=lambda e:(e["pid"],e["start_ns"]))
        spans = [e for e in host if e["kind"] == "overlap_span"]
        partitions = []
        for pid in sorted({b["pid"] for b in batches}):
            bs = [b for b in batches if b["pid"] == pid]
            for previous, following in zip(bs, bs[1:]):
                collect = next((s for s in spans if s["pid"] == pid and s["name"] == "result.collect"
                                and s["uids"] == previous["uids"] and s["start_ns"] > previous["end_ns"]), None)
                forward = next((s for s in spans if s["pid"] == pid and s["name"] == "scheduler.forward"
                                and s["uids"] == following["uids"] and s["start_ns"] <= following["start_ns"]
                                and s["end_ns"] >= following["end_ns"]), None)
                if not collect or not forward or collect["start_ns"] < forward["end_ns"]:
                    continue  # Different actual scheduling: never force a P1/P2 story.
                sync = next((s for s in spans if s["pid"] == pid and s["name"] == "event.synchronize"
                             and s.get("sample_copy") and s["uids"] == previous["uids"]
                             and collect["start_ns"] <= s["start_ns"] <= collect["end_ns"]), None)
                if sync is None:
                    raise ValueError("Missing identified sample-copy synchronization")
                # Only rank 0's generation timestamp is returned by the HTTP API.
                recorded = req_by_uid[previous["uids"][0]]["response"]["server_metrics"]["first_token_generated_ns"]
                endpoints = [following["end_ns"],forward["end_ns"],collect["start_ns"],sync["start_ns"],sync["end_ns"]]
                rank0_clock = ranks.get(pid) == 0
                if rank0_clock and not sync["end_ns"] <= recorded <= collect["end_ns"]:
                    raise ValueError("Rank 0 token timestamp is outside its collection interval")
                parts = overlap_partition(*endpoints, recorded) if rank0_clock else None
                lo, hi = following["end_ns"]+clock["offset_ns"], sync["end_ns"]+clock["offset_ns"]
                api_waits = sorted([dict(name=a["name"], start=a["start"], end=a["end"],
                                        overlap_ns=max(0,min(hi,a["end"])-max(lo,a["start"])))
                                    for a in trace["apis"] if a["pid"] == pid
                                    and a["start"] < hi and a["end"] > lo],
                                   key=lambda a:a["overlap_ns"],reverse=True)[:20]
                engine_range = next((r for r in trace["ranges"] if r["pid"] == pid
                                     and r["r3"] == "engine.forward" and r["uids"] == previous["uids"]), None)
                copies = [] if engine_range is None else [g for g in trace["gpu"] if g["pid"] == pid
                         and g["kind"] == "memcpy" and g.get("copyKind") == 2
                         and g.get("bytes") == 4*len(previous["uids"])
                         and engine_range["start"] <= g.get("api_start",-1) <= engine_range["end"]]
                partitions.append(dict(pid=pid, previous=previous["uids"], following=following["uids"],
                    engine_end_ns=following["end_ns"], forward_end_ns=forward["end_ns"],
                    collect_start_ns=collect["start_ns"], sync_start_ns=sync["start_ns"],
                    sync_end_ns=sync["end_ns"], recorded_ns=recorded if rank0_clock else None,
                    partition_ns=parts, collection_end_ns=collect["end_ns"], top_cuda_api=api_waits,
                    sample_d2h_candidates=copies,
                    sample_ready_host_ns=copies[0]["end"]-clock["offset_ns"] if len(copies) == 1 else None))
        summary = dict(file=path.name, clock=clock, partitions=partitions,
                       gpu_rows=len(trace["gpu"]), api_rows=len(trace["apis"]),
                       blocked_copy=blocked_copy_evidence(trace),
                       prepared_index_uploads=[g for g in trace["gpu"]
                           if g["kind"] == "memcpy" and g.get("owner") == "compact.indices.pack"])
        write_json(root/(path.stem+"-parsed.json"), dict(summary=summary, **trace))
        summaries.append(summary)
        plot_overlap_trace(root, path.stem, trace, clock, rows)
    if not summaries:
        raise ValueError("No exported overlap*.sqlite traces")
    write_json(root/"overlap-summary.json", summaries)
    plot_overlap_comparison(root, summaries, requests)


def plot_overlap_comparison(root, summaries, requests):
    """Compare actual adjacent batches only; no fabricated GPU forward rectangles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    selected = [s for s in summaries if any(p["partition_ns"] for p in s["partitions"])]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 1, figsize=(16, 4.4*len(selected)), squeeze=False)
    colors = dict(engine="#457fb1", sync="#9a76b5", compact="#eda746", blocked="#d9594c",
                  p1="#54a8b0", p2="#448763", unknown="#bcbfc4")
    for ax, summary in zip(axes[:,0], selected):
        trace = json.loads((root/(Path(summary["file"]).stem+"-parsed.json")).read_text())
        clock = summary["clock"]
        rows = [r for r in requests if r["cell"] == clock["cell"] and r["turn"] == clock["turn"]]
        base = min(r["response"]["server_metrics"]["request_received_ns"] for r in rows)+clock["offset_ns"]
        sec = lambda ns:(ns-base)/1e9
        p = next(p for p in summary["partitions"] if p["partition_ns"])
        pids = sorted({p["pid"] for p in summary["partitions"]})
        for rank,pid in enumerate(pids):
            y = rank*2
            for name,color in (("engine.forward",colors["engine"]), ("compact",colors["compact"]),
                               ("event.synchronize",colors["sync"])):
                bars = [(sec(r["start"]),(r["end"]-r["start"])/1e9) for r in trace["ranges"]
                        if r["pid"] == pid and r["r3"] == name]
                ax.broken_barh(bars,(y-.3,.6),facecolors=color)
            for evidence in summary["blocked_copy"]:
                if evidence["pid"] == pid and evidence["api"]["end"]-evidence["api"]["start"] > 1e9:
                    a = evidence["api"]
                    ax.broken_barh([(sec(a["start"]),(a["end"]-a["start"])/1e9)],(y-.3,.6),facecolors=colors["blocked"])
                    ax.text(sec((a["start"]+a["end"])//2),y,
                            f"cudaMemcpyAsync host blocked: {(a['end']-a['start'])/1e9:.3f} s",ha="center",va="center",color="white",fontsize=10)
            for label in ("unknown","p1","p2"):
                def group(g):
                    if g.get("owner") == "engine.forward":
                        if g.get("uids") == p["previous"]: return "p1"
                        if g.get("uids") == p["following"]: return "p2"
                    return "unknown"
                bars = [(sec(g["start"]),(g["end"]-g["start"])/1e9) for g in trace["gpu"]
                        if g["pid"] == pid and group(g) == label]
                ax.broken_barh(bars,(y+.7,.6),facecolors=colors[label])
        ready = p["sample_ready_host_ns"]
        recorded = sec(p["recorded_ns"]+clock["offset_ns"])
        ax.axvline(recorded,color="#b62637",lw=1.5)
        suffix = "P1 D2H unidentified"
        if ready is not None:
            ready_s = sec(ready+clock["offset_ns"])
            ax.axvline(ready_s,color="#263238",lw=1,ls="--")
            suffix = f"P1 D2H ready {ready_s:.3f} s | CPU records P1 {recorded:.3f} s | ready-to-record {(p['recorded_ns']-ready)/1e6:.2f} ms"
        ax.set_title(f"{clock['cell']} / turn {clock['turn']} / physical batches {len(p['previous'])}+{len(p['following'])}\n{suffix}",loc="left",fontsize=12)
        ax.set_yticks(range(len(pids)*2),[label for rank in range(len(pids)) for label in (f"CPU rank {rank}",f"GPU rank {rank}")])
        ax.set_ylim(len(pids)*2-.4,-.6)
        ax.set_xlim(0,max(sec(r["response"]["server_metrics"]["first_token_generated_ns"]+clock["offset_ns"]) for r in rows)+1)
        ax.grid(axis="x",alpha=.2); ax.set_xlabel("Seconds since first HTTP receipt (one aligned CPU / GPU axis)")
    common_end = max(ax.get_xlim()[1] for ax in axes[:,0])
    for ax in axes[:,0]:
        ax.set_xlim(0, common_end)
    handles = [Patch(color=colors[k],label=v) for k,v in (("engine","CPU Engine (enqueue + any waits)"),
        ("sync","CPU event wait"),("compact","CPU compact"),("blocked","CPU blocking CUDA API"),
        ("p1","Actual GPU P1 work"),("p2","Actual GPU P2 work"),("unknown","Other / unassigned GPU work"))]
    fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=9,bbox_to_anchor=(.5,.018))
    fig.text(.01,.004,"Dashed: P1 token reaches CPU memory. Red: CPU records P1. GPU lanes merge streams visually; raw stream plots remain available. Rank 1 P1 attribution may be absent at capture start.",fontsize=9)
    fig.tight_layout(rect=(0,.105,1,1)); fig.savefig(root/"p1-p2-causal-comparison.png",dpi=160); plt.close(fig)


def plot_overlap_trace(root, stem, trace, clock, requests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    base = min(r["response"]["server_metrics"]["request_received_ns"] for r in requests)+clock["offset_ns"]
    sec = lambda ns:(ns-base)/1e9
    first = {r["uid"]:r["response"]["server_metrics"]["first_token_generated_ns"]+clock["offset_ns"] for r in requests}
    finish = max(r["response"]["server_metrics"]["request_finished_ns"] for r in requests)+clock["offset_ns"]
    pids = sorted({r["pid"] for r in trace["ranges"] if r["r3"] == "scheduler.forward"})
    colors = {"engine.forward":"#327da8", "compact":"#df9a37", "result.collect":"#a64962"}
    groups = [(pid,g["deviceId"],g["streamId"]) for pid in pids for g in trace["gpu"] if g["pid"] == pid]
    groups = sorted(set(groups))
    lanes = [("cpu",pid,None) for pid in pids]+[("gpu",pid,(dev,stream)) for pid,dev,stream in groups]
    for zoom in (False, True):
        fig, ax = plt.subplots(figsize=(17,max(6,1.0*len(lanes))))
        for y,(kind,pid,key) in enumerate(lanes):
            if kind == "cpu":
                for name,color in colors.items():
                    for r in trace["ranges"]:
                        if r["pid"] == pid and r["r3"] == name:
                            ax.broken_barh([(sec(r["start"]),(r["end"]-r["start"])/1e9)],(y-.3,.6),facecolors=color)
                for r in trace["ranges"]:
                    if r["pid"] == pid and r["r3"].startswith("compact.op") and r["end"]-r["start"] > 5e7:
                        ax.text(sec(r["start"]),y-.36,r["r3"].removeprefix("compact.op.")+"\n"+str(r.get("source","")).split("/")[-1],fontsize=8,va="top")
            else:
                dev,stream = key
                for color,condition in (("#409878",lambda g:g["kind"] == "kernel"),
                                        ("#9a70b0",lambda g:g["kind"] != "kernel")):
                    bars = [(sec(g["start"]),(g["end"]-g["start"])/1e9) for g in trace["gpu"]
                            if g["pid"] == pid and g["deviceId"] == dev and g["streamId"] == stream and condition(g)]
                    ax.broken_barh(bars,(y-.3,.6),facecolors=color)
        for uid,t in first.items():
            ax.axvline(sec(t),color="#bd3546",alpha=.25,lw=.7)
        ax.set_yticks(range(len(lanes)), [f"CPU PID {pid}" if kind == "cpu" else f"GPU {key[0]} / stream {key[1]}" for kind,pid,key in lanes])
        ax.invert_yaxis(); ax.grid(axis="x",alpha=.2)
        ax.set_xlabel("Seconds since first HTTP receipt (measured CPU / CUDA aligned timeline)")
        ax.set_title(f"{clock['cell']} | Turn {clock['turn']} | "+("longest compact interval" if zoom else "whole wave"))
        ax.legend(handles=[Patch(color=c,label=n) for n,c in colors.items()]+
                  [Patch(color="#409878",label="GPU kernel"),Patch(color="#9a70b0",label="GPU memory operation")],loc="upper right",fontsize=9)
        if zoom:
            candidates = [r for r in trace["ranges"] if r["r3"] == "compact"]
            if not candidates:
                candidates = [r for r in trace["ranges"] if r["r3"] == "engine.forward"]
            longest = max(candidates,key=lambda r:r["end"]-r["start"])
            ax.set_xlim(max(0,sec(longest["start"])-.3),sec(longest["end"])+.3)
        else:
            ax.set_xlim(0,sec(finish)+.2)
        fig.text(.01,.005,f"Clock alignment uncertainty <= {clock['uncertainty_ns']/1000:.1f} us. Red lines: CPU first-token records. GPU bars are execution, not host enqueue.",fontsize=10)
        fig.tight_layout(rect=(0,.025,1,1)); fig.savefig(root/f"{stem}-{'zoom' if zoom else 'timeline'}.png",dpi=160); plt.close(fig)


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
        raw_tokens = [event["tokens"][offset] for event,offset in sequence if "tokens" in event]
        # Overlap may execute an unused lookahead token after the request limit.
        # Compare the generated sequence, retaining every raw launch in events.
        generated = metrics.get("generated_tokens", len(raw_tokens))
        output_tokens[(row["mode"], row["concurrency"], row["workload"], row["case_id"], row["turn"])] = (
            raw_tokens[:generated] if len(raw_tokens) >= generated else None)
    comparison = []
    for key, tokens in output_tokens.items():
        if key[0] not in ("detail", "overlap"):
            continue
        reference_mode = "natural" if key[0] == "overlap" else "baseline"
        baseline = output_tokens.get((reference_mode, *key[1:]))
        comparison.append(dict(concurrency=key[1], workload=key[2], case_id=key[3], turn=key[4],
                               reference_mode=reference_mode, measured_mode=key[0],
                               baseline_tokens=baseline, detail_tokens=tokens,
                               equal=baseline is not None and tokens is not None and tokens==baseline))
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


def audit_prefill_batches(requests, events):
    """Require one complete physical prefill per TP rank, not just HTTP concurrency."""
    waves = defaultdict(list)
    for row in requests:
        if "uid" in row:
            waves[(row["cell"], row["turn"])].append(row)
    result = []
    for (cell, turn), rows in waves.items():
        expected = rows[0]["concurrency"]
        uids = sorted(r["uid"] for r in rows)
        batches = [e for e in events if e.get("cell") == cell and e.get("turn") == turn
                   and e["kind"] == "batch" and e["phase"] == "prefill"]
        by_rank = defaultdict(list)
        for batch in batches:
            by_rank[batch["pid"]].append(batch)
        passed = len(uids) == expected and len(set(uids)) == expected and len(by_rank) == 2
        passed &= all(len(bs) == 1 and sorted(bs[0]["uids"]) == uids
                      and bs[0]["size"] == expected for bs in by_rank.values())
        result.append(dict(cell=cell, turn=turn, expected=expected, uids=uids,
                           physical_batches={str(pid): [dict(size=b["size"], uids=b["uids"],
                               extend=b.get("extend"), cached=b.get("cached")) for b in bs]
                               for pid, bs in by_rank.items()}, passed=passed))
    return result


def launch(args, root):
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    used = dict(tuple(map(int, line.split(","))) for line in usage.splitlines())
    gpus = getattr(args, "gpus", [0, 1])
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(used[gpu] > 100 for gpu in gpus):
        raise RuntimeError(f"Selected TP2 GPUs not idle/distinct; refusing launch: {gpus}, {used}")
    source = getattr(args, "source_repo", None) or REPO
    source = source.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
               MINISGL_SOURCE_REPO=str(source), PYTHONPATH=f"{source}:{source / 'python'}",
               MINISGL_TTFT_PROFILE_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               MINISGL_TTFT_DISTRIBUTED_TIMEOUT="600",
               TORCH_NCCL_TRACE_BUFFER_SIZE="4096", TORCH_NCCL_DUMP_ON_TIMEOUT="1")
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
    if any(mode in ("overlap", "fixed-overlap") for mode in args.modes):
        if not args.nsys:
            raise ValueError("The overlap mode requires --nsys for CPU/GPU aligned tracing")
        argv = [str(args.nsys), "profile", "--sample=none", "--cpuctxsw=none",
                "--trace=cuda,nvtx,osrt", "--capture-range=cudaProfilerApi",
                "--capture-range-end=repeat:6", "--kill=none", "--cuda-graph-trace=graph",
                "--output="+str(root/"overlap")] + argv
    write_json(root / "launch.json", dict(argv=argv, env={k:v for k,v in env.items() if k in
               {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH",
                "CC", "CXX", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "CPATH",
                "MINISGL_TTFT_DISTRIBUTED_TIMEOUT", "TORCH_NCCL_TRACE_BUFFER_SIZE",
                "TORCH_NCCL_DUMP_ON_TIMEOUT",
                "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH"}},
               head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
               source_repo=str(source), short_validation=getattr(args, "short_validation", False),
               gpu_preflight=usage))
    with (root/"server.log").open("w") as log:
        return subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def short_validation_turns(cases):
    """A real TR8 seed followed by TR9/TR10; never renumber history/events."""
    indices = []
    for count in (8, 9, 10):
        matches = [{r["turn"] for r in case["turns"] if r["tool_responses"] == count}
                   for case in cases]
        common = set.intersection(*matches)
        if not common:
            raise ValueError(f"No common recorded query with exactly {count} tool responses")
        indices.append(min(common))
    if indices != sorted(set(indices)):
        raise ValueError("Short validation must preserve chronological request order")
    return indices


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
    short = getattr(args, "short_validation", False)
    if short and (args.modes != ["fixed"] or args.concurrency != [8]):
        raise ValueError("Short validation permits only fixed C8, one group per workload")
    selected_turns = short_validation_turns(cases) if short else list(range(manifest["turns"]))
    write_json(root/"selection.json", dict(turns=selected_turns, seed=selected_turns[0] if short else None,
               short_validation=short, expected_requests=len(selected_turns)*8*2 if short else None))
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
            inference_start = time.monotonic()
            for mode in args.modes:
                for concurrency in args.concurrency:
                    if mode in ("fixed", "fixed-overlap") and concurrency != 8:
                        continue
                    for workload in ("no_drop", "rolling"):
                        cell = f"{mode}-c{concurrency}-{workload}"
                        cell_dir = root/cell
                        cell_dir.mkdir()
                        with (cell_dir/"requests.jsonl").open("w") as output:
                            for turn in selected_turns:
                                seed_wave = short and turn == selected_turns[0]
                                control = mode_control(mode, turn)
                                if seed_wave:
                                    control["fixed_split"] = False
                                write_json(root/"control.json", dict(cell=cell, turn=turn,
                                           concurrency=1 if seed_wave else concurrency,
                                           cohort=[dict(case_id=c["case_id"], tokens=c["turns"][turn]["full_tokens"])
                                                   for c in sorted(cases[:concurrency], key=lambda c:
                                                       ["210","215","229","236","226","223","231","233"].index(str(c["case_id"])))],
                                           **control))
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
                                remaining = args.inference_budget_seconds - (time.monotonic() - inference_start)
                                if remaining <= 0:
                                    raise TimeoutError("Approved inference budget exhausted; no automatic retry")
                                async def wave():
                                    if seed_wave:
                                        # Cold prefixes exceed the 1+7 token budget.
                                        # Compute real KV serially, identical across
                                        # versions; measured waves remain strict 1+7.
                                        for case in cases[:concurrency]:
                                            await request(case)
                                    else:
                                        await asyncio.gather(*(request(case) for case in cases[:concurrency]))
                                await asyncio.wait_for(wave(), timeout=remaining)
                                # Allow out-of-request buffered observer flush to finish.
                                await asyncio.sleep(0.2)
            write_json(root/"completed.json", dict(completed=True, time_ns=time.perf_counter_ns(),
                       inference_seconds=time.monotonic()-inference_start))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    if not args.skip_report:
        report(argparse.Namespace(output=root))


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = external(args.output)
    requests = [json.loads(line) for path in sorted(root.glob("*/requests.jsonl"))
                for line in path.read_text().splitlines()]
    failed = [r for r in requests if "ttft_ms" not in r]
    write_json(root/"failed_requests.json", failed)
    requests = [r for r in requests if "ttft_ms" in r]
    events = [json.loads(line) for path in sorted(root.glob("events-*.jsonl"))
              for line in path.read_text().splitlines()]
    by_cell = defaultdict(list)
    for row in requests:
        by_cell[row["cell"]].append(row)
    components, output_comparison = correlate(requests, events)
    write_json(root/"components.json", components)
    write_json(root/"profile_output_comparison.json", output_comparison)
    write_json(root/"ttft_functions.json", ttft_function_totals(requests, events))
    write_json(root/"batch_audit.json", audit_prefill_batches(requests, events))
    gpu_totals = defaultdict(lambda: defaultdict(float))
    for e in events:
        if e["kind"] != "gpu_range" or "gpu_stream_ms" not in e:
            continue
        total = gpu_totals[(e["cell"], e["turn"], e["pid"], e["name"])]
        total["calls"] += 1
        total["gpu_stream_ms"] += e["gpu_stream_ms"]
    write_json(root/"gpu_functions.json", [dict(cell=k[0], turn=k[1], pid=k[2], name=k[3], **v)
                                          for k, v in gpu_totals.items()])
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
                       late_ttft_ms=(statistics.mean(r["ttft_ms"] for r in rows if r["turn"] >= 9)
                                     if any(r["turn"] >= 9 for r in rows) else None),
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
                    values = []
                    for t in turns:
                        measured = [r[metric] for r in rows if r["turn"]==t and r[metric] is not None]
                        values.append(statistics.mean(measured) if measured else float("nan"))
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


def committed_outputs(requests, events):
    """Use Req.append_host commits, never infer output by truncating GPU launches."""
    commits = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event["kind"] == "committed_token":
            commits[(event["cell"], event["uid"])][event["pid"]].extend(event["tokens"])
    result = {}
    for row in requests:
        if "uid" not in row:
            continue
        key = (row["mode"], row["concurrency"], row["workload"], str(row["case_id"]), row["turn"])
        ranks = list(commits[(row["cell"], row["uid"])].values())
        expected = row["response"]["server_metrics"]["generated_tokens"]
        valid = len(ranks) == 2 and all(len(tokens) == expected for tokens in ranks) and ranks[0] == ranks[1]
        result[key] = dict(tokens=ranks[0] if valid else None, commit_valid=valid,
                           rank_tokens=ranks, choices=row["response"]["choices"],
                           usage=row["response"]["usage"], request_sha256=row["request_sha256"])
    return result


def canonical_choices(choices):
    """GPT-OSS response_parser._tool_call generates random IDs, not model output.

    Preserve every other field, and retain raw choices separately in the report.
    """
    value = copy.deepcopy(choices)
    for choice in value:
        for call in choice.get("message", {}).get("tool_calls") or []:
            if re.fullmatch(r"call_[0-9a-f]{24}", str(call.get("id", ""))):
                call["id"] = "<server-generated-tool-call-id>"
    return value


def compare_committed(before_requests, before_events, after_requests, after_events):
    before = committed_outputs(before_requests, before_events)
    after = committed_outputs(after_requests, after_events)
    comparisons = []
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key), after.get(key)
        present = left is not None and right is not None
        flags = {name+"_equal": bool(present and left[name] == right[name])
                 for name in ("tokens", "choices", "usage", "request_sha256")}
        raw_choices_equal = flags["choices_equal"]
        flags["choices_equal"] = bool(present and
            canonical_choices(left["choices"]) == canonical_choices(right["choices"]))
        valid = bool(present and left["commit_valid"] and right["commit_valid"])
        comparisons.append(dict(mode=key[0], concurrency=key[1], workload=key[2],
            case_id=key[3], turn=key[4], before=left, after=right,
            commits_valid=valid, raw_choices_equal=raw_choices_equal,
            **flags, passed=valid and all(flags.values())))
    return comparisons


def physical_batch_signatures(requests, events):
    identities = {(r["cell"], r["uid"]): str(r["case_id"]) for r in requests if "uid" in r}
    signatures = defaultdict(lambda: defaultdict(list))
    for event in sorted((e for e in events if e["kind"] == "batch"), key=lambda e:e["start_ns"]):
        if all((event["cell"], uid) in identities for uid in event["uids"]):
            signature = dict(phase=event["phase"], graph=event["graph"],
                cases=[identities[(event["cell"], uid)] for uid in event["uids"]],
                extend=event["extend"], cached=event["cached"])
            signatures[(event["cell"],event["turn"])][event["pid"]].append(signature)
    return {key: list(ranks.values()) for key,ranks in signatures.items()}


def compare_versions(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roots = [external(args.before), external(args.after)]
    output = external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = []
    for root in roots:
        requests = [json.loads(line) for path in sorted(root.glob("*/requests.jsonl"))
                    for line in path.read_text().splitlines()]
        for row in requests:
            if "server_metrics" in row.get("response", {}):
                metrics = row["response"]["server_metrics"]
                row["e2e_ms"] = (metrics["request_finished_ns"]-metrics["request_received_ns"])/1e6
        events = [json.loads(line) for path in sorted(root.glob("events-*.jsonl"))
                  for line in path.read_text().splitlines()]
        data.append((requests, events))
    comparisons = compare_committed(*data[0], *data[1])
    completed = all((root/"completed.json").exists() for root in roots)
    fixed = [row for row in comparisons if row["mode"] == "fixed"]
    write_json(output/"output-comparison.json", comparisons)
    batch_signatures = [physical_batch_signatures(*item) for item in data]
    batch_comparisons = []
    for key in sorted(batch_signatures[0].keys() | batch_signatures[1].keys()):
        left, right = [item.get(key, []) for item in batch_signatures]
        equal = len(left) == len(right) == 2 and left[0] == left[1] == right[0] == right[1]
        batch_comparisons.append(dict(cell=key[0], turn=key[1], equal=equal, before=left, after=right))
    write_json(output/"batch-comparison.json", batch_comparisons)
    summary = []
    keys = sorted({(r["mode"], r["concurrency"], r["workload"]) for r in data[0][0]})
    for mode, concurrency, workload in keys:
        for late in (False, True):
            row = dict(mode=mode, concurrency=concurrency, workload=workload, late=late)
            for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
                values = [[r[metric] for r in requests if r["mode"] == mode and
                           r["concurrency"] == concurrency and r["workload"] == workload and
                           (not late or r["turn"] >= 9) and r.get(metric) is not None]
                          for requests, _ in data]
                for name, group in zip(("before", "after"), values):
                    row[name+"_"+metric] = statistics.mean(group) if group else None
                row[metric+"_change_pct"] = (100*(statistics.mean(values[1])/statistics.mean(values[0])-1)
                                             if all(values) else None)
            summary.append(row)
    write_json(output/"performance.json", summary)
    write_json(output/"verification.json", dict(
        requests=len(comparisons), completed=completed,
        full_output_pass=completed and bool(comparisons) and all(r["passed"] for r in comparisons),
        fixed_output_pass=completed and bool(fixed) and all(r["passed"] for r in fixed),
        raw_choices_pass=completed and bool(comparisons) and all(r["raw_choices_equal"] for r in comparisons),
        output_policy="Exact committed tokens, choices and usage; choices exclude only server-generated call_<24 hex> tool IDs. Raw responses and equality are retained.",
        note="One cohort per cell. Raw timing changes are not statistical confidence or a universal no-regression proof.",
        heads=[json.loads((root/"launch.json").read_text())["head"] for root in roots]))
    for mode, concurrency in sorted({key[:2] for key in keys}):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        for version, (requests, _) in zip(("Before", "After"), data):
            for workload, color in (("no_drop", "tab:blue"), ("rolling", "tab:orange")):
                rows = [r for r in requests if (r["mode"],r["concurrency"],r["workload"]) == (mode,concurrency,workload)]
                turns = sorted({r["turn"] for r in rows})
                for ax, metric in zip(axes, ("ttft_ms", "tpot_ms")):
                    values = [statistics.mean(r[metric] for r in rows if r["turn"] == turn and r.get(metric) is not None) for turn in turns]
                    ax.plot(turns, values, "o--" if version == "Before" else "s-", color=color,
                            label=f"{version} | {workload}")
                    ax.set_ylabel(metric); ax.grid(alpha=.2)
        axes[0].legend(); axes[1].set_xlabel("Turn (zero-based); all peaks retained")
        fig.suptitle(f"R4 | GPT-OSS-120B TP2 | {mode} C{concurrency} | one cohort")
        fig.tight_layout(); fig.savefig(output/f"{mode}-c{concurrency}.png", dpi=150); plt.close(fig)


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
    run_parser.add_argument("--modes", choices=("baseline", "detail", "natural", "gpu", "overlap", "fixed", "fixed-overlap"), nargs="+", default=["baseline", "detail"])
    run_parser.add_argument("--nsys", type=Path, help="Existing Nsight Systems executable; no installation")
    run_parser.add_argument("--skip-report", action="store_true", help="Export/plot captured artifacts locally")
    run_parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    run_parser.add_argument("--source-repo", type=Path,
                            help="Read-only Git worktree for baseline production code; same driver")
    run_parser.add_argument("--short-validation", action="store_true",
                            help="Only TR8 seed and TR9/TR10 measurements, fixed C8")
    run_parser.add_argument("--inference-budget-seconds", type=float, default=float("inf"))
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--output", type=Path, required=True)
    overlap_parser = sub.add_parser("overlap-report")
    overlap_parser.add_argument("--output", type=Path, required=True)
    comparison_parser = sub.add_parser("compare")
    comparison_parser.add_argument("--before", type=Path, required=True)
    comparison_parser.add_argument("--after", type=Path, required=True)
    comparison_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        asyncio.run(run(args))
    elif args.command == "overlap-report":
        overlap_report(args)
    elif args.command == "compare":
        compare_versions(args)
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
