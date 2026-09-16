#!/usr/bin/env python3
"""Fixed-concurrency, whole-trajectory HTTP benchmark (no live tool execution).

Metric formulas follow SGLang bench_serving.py, Apache-2.0, revision below.
Preparation uses the target tokenizer; the timed client only uses HTTP/SSE.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from throughput_compute import compute_summary

REFERENCE = "03ea13a54557de52da5faab2c422da07c3727407"
DEFAULT_INPUT = "/share/public/wangruoxi/local/throughput_bcp/manifest.json"
LIMIT = 128 * 1024
THRESHOLD = 96 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def rolling_schedule(messages, owners, full_tokens, threshold=THRESHOLD, keep=12):
    """Owners come from one complete chat-template render, never concatenation.

    Drop at the end of TR_n; compact at a message end whose *current* position
    length reaches threshold. Only a new dropped span can reduce that length.
    The unowned generation suffix is appended after the last message boundary.
    """
    counts = Counter(owners)
    ends = {}
    for offset, owner in enumerate(owners):
        if owner >= 0:
            ends[owner] = offset + 1
    tools, drops, repositions, checks = [], {}, [], []
    removed, compacted = 0, 0
    for i, message in enumerate(messages):
        if message.get("role") == "tool":
            tools.append(i)
            if len(tools) > keep:
                old = tools[-keep - 1]
                drops[str(i)] = [old]
                removed += counts[old]
        if i not in ends:
            raise ValueError(f"Message {i} has no token ownership")
        position = ends[i] - compacted
        if position >= threshold and removed > compacted:
            repositions.append(i)
            checks.append({"message_id": i, "before": position,
                           "after": ends[i] - removed, "raw_end": ends[i]})
            compacted = removed
    return {"drop_message": drops, "reposition": repositions,
            "position_tokens": full_tokens - compacted,
            "active_tokens": full_tokens - removed, "reposition_checks": checks}


def prepare(args):
    from minisgl.benchmark.reposition_bcp import browsecomp_plus_tools
    from minisgl.core import SamplingParams
    from minisgl.message.tokenizer import TokenizeMsg
    from minisgl.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manager = TokenizeManager(AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True),
                              radix_drop_key_mode="delta-marker")
    tools = browsecomp_plus_tools()
    candidates, sources = [], []
    for source in args.source:
        for path in sorted(Path(source).glob("shard*/trajectories.jsonl")):
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            sources.append({"path": str(path), "sha256": sha})
            for line_number, line in enumerate(raw.splitlines(), 1):
                row = json.loads(line)
                key = str(row["case_id"])
                trajectory = row["trajectory"]
                ends = [i for i, m in enumerate(trajectory) if m.get("role") == "assistant"]
                if not ends:
                    continue
                ids, owners, _ = manager._render_harmony_message_drop(
                    trajectory[:ends[-1]], enable_thinking=None, tools=tools)
                schedule = rolling_schedule(trajectory[:ends[-1]], owners, len(ids))
                if schedule["position_tokens"] >= LIMIT or any(
                    x["before"] > LIMIT for x in schedule["reposition_checks"]
                ):
                    print(json.dumps({"excluded": key, "reason": "position_limit_before_legal_boundary",
                                      "source": str(path), "line": line_number}), flush=True)
                    continue
                candidates.append(dict(case_id=key, trajectory=trajectory, ends=ends,
                                       full_tokens=len(ids), source=str(path),
                                       source_line=line_number, source_sha256=sha))
            print(json.dumps({"scanned": str(path), "trajectories": len(candidates)}), flush=True)
    groups = [{}, {}]
    for row in candidates:
        groups[0 if row["full_tokens"] > LIMIT else 1].setdefault(row["case_id"], row)
    # A case may have multiple real trials. Choose the first qualifying trial in
    # its class, and reserve long-only IDs first so short alternatives stay usable.
    longs = sorted(groups[0].values(), key=lambda r: (r["case_id"] in groups[1], int(r["case_id"])))[:40]
    long_ids = {r["case_id"] for r in longs}
    shorts = sorted((r for k, r in groups[1].items() if k not in long_ids), key=lambda r: int(r["case_id"]))[:40]
    if len(longs) != 40 or len(shorts) != 40:
        raise ValueError(f"Need 40 long + 40 short DISTINCT cases, qualifying classes={list(map(len, groups))}, disjoint selection={[len(longs), len(shorts)]}")
    longs.sort(key=lambda r: int(r["case_id"]))
    selected = [row for pair in zip(longs, shorts) for row in pair]
    cases = []
    for row in selected:
        trajectory, ends = row.pop("trajectory"), row.pop("ends")
        turns = []
        for turn, end in enumerate(ends):
            messages = trajectory[:end]
            ids, owners, _ = manager._render_harmony_message_drop(
                messages, enable_thinking=None, tools=tools)
            schedule = rolling_schedule(messages, owners, len(ids))
            # Independent compiler check: catches template ownership and boundary drift.
            result = manager.tokenize([TokenizeMsg(
                uid=turn, text=messages, sampling_params=SamplingParams(max_tokens=1),
                tools=tools, drop_message={int(k): v for k, v in schedule["drop_message"].items()},
                reposition=schedule["reposition"] or None)])[0]
            actual = result.radix_next_position if result.reposition_layout is not None else len(ids)
            if actual != schedule["position_tokens"]:
                raise ValueError(f"Position mismatch {row['case_id']}:{turn}: {actual} != {schedule}")
            if actual >= LIMIT or any(x["before"] > LIMIT for x in schedule["reposition_checks"]):
                raise ValueError(f"Rolling schedule exceeds model position capacity: {row['case_id']}:{turn}")
            turns.append(dict(turn=turn, end=end, full_tokens=len(ids),
                              messages_sha256=digest(messages), **schedule))
        filename = f"case-{row['case_id']}.json.gz"
        with gzip.open(root / filename, "wt") as stream:
            json.dump(trajectory, stream, ensure_ascii=False)
        row.update(file=filename, turns=turns, trajectory_sha256=digest(trajectory),
                   long=row["full_tokens"] > LIMIT)
        cases.append(row)
        print(json.dumps({"prepared": row["case_id"], "turns": len(turns),
                          "long": row["long"], "full_tokens": row["full_tokens"]}), flush=True)
        write_json(root / "preparation_progress.json", {"completed": len(cases), "total": 80})
    manifest = dict(schema=1, tokenizer=args.tokenizer, context_limit=LIMIT, rolling_keep=12,
                    reposition_threshold=THRESHOLD, sources=sources, tools=tools, cases=cases)
    write_json(root / "manifest.json", manifest)
    write_json(root / "preparation_report.json", dict(distinct=80, long=40, short=40,
               manifest_sha256=digest(manifest), compiler_verified_turns=sum(len(c["turns"]) for c in cases)))


def calculate_metrics(records, duration):
    """SGLang's complete numeric schema; records already have retokenized lengths.

    Each record is one HTTP turn. Failed outputs have output_lens=0. A no-success
    run is explicitly invalid; unlike upstream we return null E2E instead of
    throwing in percentile([]). Streaming peaks count text events, as upstream.
    """
    if duration <= 0:
        raise ValueError("Metric duration must be positive")
    successful = [r for r in records if r["success"]]
    output_lens = [r["output_len"] if r["success"] else 0 for r in records]
    total_input = sum(r["prompt_len"] for r in successful)
    total_output = sum(output_lens)
    retokenized = sum(r["retokenized_len"] for r in successful)
    metrics = dict(completed=len(successful), total_input=total_input,
                   total_input_text=total_input, total_input_vision=0,
                   total_output=total_output, total_output_retokenized=retokenized,
                   request_throughput=len(successful) / duration,
                   input_throughput=total_input / duration,
                   output_throughput=total_output / duration,
                   output_throughput_retokenized=retokenized / duration,
                   total_throughput=(total_input + total_output) / duration,
                   total_throughput_retokenized=(total_input + retokenized) / duration)
    values = dict(ttft=[r["ttft"] for r in successful],
                  tpot=[(r["latency"] - r["ttft"]) / (r["output_len"] - 1)
                        for r in successful if r["output_len"] > 1],
                  itl=[v for r in successful for v in r["itl"]],
                  e2e_latency=[r["latency"] for r in successful])
    for name, series in values.items():
        arr = series or [0]
        funcs = {"mean": np.mean, "median": np.median, "std": np.std,
                 **{f"p{p}": lambda v, p=p: np.percentile(v, p) for p in (90, 95, 99)}}
        for stat, function in funcs.items():
            metrics[f"{stat}_{name}_ms"] = (None if not series and name == "e2e_latency"
                                             else float(function(arr) * 1000))
    metrics["max_itl_ms"] = max(values["itl"] or [0]) * 1000
    metrics["concurrency"] = sum(values["e2e_latency"]) / duration
    peak_tokens, peak_requests = 0., 0
    if successful:
        start = min(r["start_time"] for r in successful)
        end = max(r["start_time"] + r["latency"] for r in successful)
        size = math.ceil(end - start) + 1
        tokens, requests = np.zeros(size), np.zeros(size)
        for r in successful:
            event = r["start_time"] + r["ttft"]
            for delta in [0, *r["itl"]]:
                event += delta
                bucket = int(event - start)
                if 0 <= bucket < size:
                    tokens[bucket] += 1
            lo = int(r["start_time"] - start)
            hi = min(int(r["start_time"] + r["latency"] - start) + 1, size)
            requests[lo:hi] += 1
        peak_tokens, peak_requests = float(tokens.max()), int(requests.max())
    metrics.update(max_output_tokens_per_s=peak_tokens, max_concurrent_requests=peak_requests)
    return metrics, output_lens


async def sse_events(content):
    """aiohttp readline preserves split TCP frames; support multi-line SSE data."""
    data = []
    async for line in content:
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


async def request(session, url, payload, clock=time.perf_counter):
    start = clock()
    result = dict(start_time=start, end_time=start, latency=0., ttft=0., itl=[],
                  generated_text="", text_chunks=[], output_len=payload["max_tokens"],
                  prompt_len=0, success=False, strict_success=False, cancelled=False,
                  done=False, usage=None, finish_reason=None, error=None, events=[])
    last_text, text_parts = None, []
    try:
        async with session.post(url, json=payload) as response:
            result["http_status"] = response.status
            if response.status != 200:
                result["error"] = await response.text()
            else:
                async for data in sse_events(response.content):
                    now = clock()
                    result["latency"] = now - start
                    if data == "[DONE]":
                        result["done"] = True
                        continue
                    event = json.loads(data)
                    result["events"].append({"time": now, "data": event})
                    if event.get("error"):
                        result["error"] = event["error"]
                    if event.get("usage"):
                        result["usage"] = event["usage"]
                        result["output_len"] = event["usage"].get("completion_tokens", result["output_len"])
                        result["prompt_len"] = event["usage"].get("prompt_tokens", 0)
                    for choice in event.get("choices") or []:
                        result["finish_reason"] = choice.get("finish_reason") or result["finish_reason"]
                        delta = choice.get("delta") or {}
                        text = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
                        if text:
                            if last_text is None:
                                result["ttft"] = now - start
                            else:
                                result["itl"].append(now - last_text)
                                result["text_chunks"].append(text)
                            last_text = now
                            text_parts.append(text)
                # Deliberate parity with SGLang's OAI-chat adapter: clean HTTP 200 EOF.
                result["success"] = True
    except asyncio.CancelledError:
        result.update(cancelled=True, error="measurement_cutoff_or_client_cancel")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["generated_text"] = "".join(text_parts)
    result["end_time"] = clock()
    usage = result["usage"]
    result["strict_success"] = bool(result["success"] and result["done"] and not result["error"]
                                     and result["finish_reason"] in ("stop", "length", "tool_calls")
                                     and usage is not None and "completion_tokens" in usage
                                     and "prompt_tokens" in usage)
    return result


def build_payload(case, turn, tools, args):
    capacity = turn["position_tokens"] if args.drop else turn["full_tokens"]
    remaining = args.context_limit - capacity
    if remaining <= 0:
        return None
    payload = dict(model=args.model, messages=case["trajectory"][:turn["end"]], tools=tools,
                   stream=True, stream_options={"include_usage": True}, temperature=0,
                   max_tokens=min(args.max_token_len, remaining), n=1)
    if args.drop:
        payload["drop_message"] = turn["drop_message"]
        if turn["reposition"]:
            payload["reposition"] = turn["reposition"]
    return payload


class Scheduler:
    """Single event-loop ownership makes selection and first-pass cutoff atomic."""
    def __init__(self, cases, concurrency, execute, on_event=lambda e: None):
        self.cases, self.concurrency, self.execute, self.on_event = cases, concurrency, execute, on_event
        self.pending = deque(cases)
        self.active, self.completed = set(), []
        self.instances, self.round_ends = [], []
        self.stop = asyncio.Event()
        self.cutoff = None
        self.replay_cursor = 0
        self.workers = []

    def choose(self):
        if self.stop.is_set():
            return None
        if self.pending:
            return self.pending.popleft(), False
        for _ in range(len(self.completed)):
            item = self.completed[self.replay_cursor % len(self.completed)]
            self.replay_cursor += 1
            case = item["case"]
            if case["case_id"] not in self.active:
                return case, True
        raise RuntimeError("No distinct completed task available for tail fill")

    async def worker(self, slot):
        while (selected := self.choose()) is not None:
            case, filler = selected
            key = case["case_id"]
            assert key not in self.active
            self.active.add(key)
            instance = dict(instance=len(self.instances), case_id=key, filler=filler,
                            slot=slot, start_time=time.perf_counter(), long=case["long"])
            self.instances.append(instance)
            self.on_event(dict(kind="task_start", **instance))
            try:
                status = await self.execute(case, instance)
            except asyncio.CancelledError:
                status = "cancelled_at_cutoff"
            except Exception as exc:
                status = f"client_error: {type(exc).__name__}: {exc}"
            end = time.perf_counter()
            instance.update(end_time=end, status=status)
            self.active.remove(key)
            if not filler and not self.stop.is_set():
                self.completed.append({"case": case, "instance": instance})
                if len(self.completed) % self.concurrency == 0:
                    self.round_ends.append(end)
                    self.on_event(dict(kind="round_end", round=len(self.round_ends), time=end,
                                       completed=len(self.completed)))
                if len(self.completed) == len(self.cases):
                    self.cutoff = end
                    self.stop.set()
                    for worker in self.workers:
                        if worker is not asyncio.current_task():
                            worker.cancel()
            self.on_event(dict(kind="task_end", **instance))

    async def run(self):
        self.start = time.perf_counter()
        self.workers = [asyncio.create_task(self.worker(i)) for i in range(self.concurrency)]
        await asyncio.gather(*self.workers)
        return self


def summarize(records, start, end):
    included = [r for r in records if start < r["end_time"] <= end]
    metrics, lens = calculate_metrics(included, end - start)
    strict = [dict(r, success=r["strict_success"]) for r in included]
    fillers = [r for r in included if r["filler"]]
    return dict(start_time=start, end_time=end, duration_s=end - start, metrics=metrics,
                sglang_logical_metrics=metrics, compute_metrics=compute_summary(included, end - start),
                first_pass_compute=compute_summary([r for r in included if not r["filler"]], end - start),
                filler_compute=compute_summary(fillers, end - start),
                output_lens=lens, strict_metrics=calculate_metrics(strict, end - start)[0],
                first_pass_metrics=calculate_metrics([r for r in included if not r["filler"]], end - start)[0],
                filler_metrics=calculate_metrics(fillers, end - start)[0], http_turns=len(included),
                crossing_window_turns=sum(r["start_time"] < start for r in included),
                failed_turns=sum(not r["success"] for r in included),
                strict_failed_turns=sum(not r["strict_success"] for r in included),
                cached_tokens=sum((r["usage"] or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                  for r in included if r["success"]))


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer

    manifest_path = Path(args.requests_path)
    manifest = json.loads(manifest_path.read_text())
    args.num_requests = args.num_requests or min(3 * args.concurrency, 80)
    if not (1 <= args.concurrency <= args.num_requests <= 80) or args.num_requests % 2:
        raise ValueError("Require 1 <= concurrency <= num_requests <= 80; num_requests must be even")
    if args.max_token_len <= 0:
        raise ValueError("max-token-len must be positive")
    cases = manifest["cases"][:args.num_requests]
    if len(cases) != args.num_requests or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Not enough distinct input tasks")
    if sum(c["long"] for c in cases) * 2 != len(cases):
        raise ValueError("Input prefix must contain exactly half long tasks")
    for case in cases:
        with gzip.open(manifest_path.parent / case["file"], "rt") as stream:
            case["trajectory"] = json.load(stream)
        if digest(case["trajectory"]) != case["trajectory_sha256"]:
            raise ValueError(f"Input hash mismatch: {case['case_id']}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or manifest["tokenizer"], local_files_only=True)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_{'drop' if args.drop else 'no_drop'}_C{args.concurrency}"
    result_path = root / f"{name}.json"
    journal_path = root / f"{name}.events.jsonl"
    started_wall = datetime.now().astimezone().isoformat()
    records = []
    writer = ThreadPoolExecutor(max_workers=1)
    journal = journal_path.open("x")
    def emit(event):
        snapshot = [dict(r) for r in records] if event["kind"] == "round_end" else None
        def append():
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            if event["kind"] == "task_end":
                write_json(root / f"{name}.progress.json", event)
            if snapshot is not None:
                for r in snapshot:
                    r["retokenized_len"] = len(tokenizer.encode(r["generated_text"], add_special_tokens=False))
                index = event["round"] - 1
                previous = scheduler.start if index == 0 else scheduler.round_ends[index - 1]
                report = summarize(snapshot, previous, event["time"])
                report.update(round=event["round"], cumulative=summarize(snapshot, scheduler.start, event["time"]))
                write_json(root / f"{name}.round-{event['round']}.json", report)
        writer.submit(append)
    base = args.host if "://" in args.host else "http://" + args.host
    url = base.rstrip("/") + (f":{args.port}" if args.port else "") + "/" + args.post.lstrip("/")
    headers = {"Authorization": "Bearer " + os.environ[args.api_key_env]} if args.api_key_env else {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout), headers=headers,
                                    connector=aiohttp.TCPConnector(limit=args.concurrency)) as session:
        async def execute(case, instance):
            turns = case["turns"][:args.smoke_max_turns] if args.smoke_max_turns else case["turns"]
            if args.smoke_long_last:
                turns = [case["turns"][-1]]
            for turn in turns:
                payload = build_payload(case, turn, manifest["tools"], args)
                if payload is None:
                    return "context_limit_reached" if not args.drop else "unexpected_drop_position_limit"
                r = await request(session, url, payload)
                r.update(case_id=case["case_id"], instance=instance["instance"], filler=instance["filler"],
                         turn=turn["turn"], expected_prompt_len=turn["full_tokens"],
                         position_tokens=turn["position_tokens"] if args.drop else turn["full_tokens"],
                         requested_max_tokens=payload["max_tokens"],
                         drop_events=len(payload.get("drop_message", {})),
                         reposition=payload.get("reposition", []))
                # SGLang uses DatasetRow.prompt_len, independent of final usage.
                r["prompt_len"] = turn["full_tokens"]
                r["usage_prompt_matches"] = bool(r["usage"] is not None and
                    r["usage"].get("prompt_tokens") == turn["full_tokens"])
                # Cache savings may be zero on a valid cold-cache Drop request.
                # An extension field acknowledges the protocol, not every mask bit.
                details = (r["usage"] or {}).get("prompt_tokens_details", {})
                r["drop_acknowledged"] = (True if not r["drop_events"] or
                    "drop_skipped_tokens" in details or "repos_tokens" in details else None)
                records.append(r)
                emit(dict(kind="http_turn", **r))
                if r["cancelled"]:
                    return "cancelled_at_cutoff"
                if not r["strict_success"]:
                    return "http_or_stream_failure"
            return "all_turns_completed"
        scheduler = Scheduler(cases, args.concurrency, execute, emit)
        await scheduler.run()
    writer.shutdown(wait=True)
    journal.close()
    # Retokenization and bulky result serialization are outside the measurement.
    for r in records:
        r["retokenized_len"] = len(tokenizer.encode(r["generated_text"], add_special_tokens=False))
    cutoff = scheduler.cutoff
    for instance in scheduler.instances:
        own = [r for r in records if r["instance"] == instance["instance"] and r["end_time"] <= cutoff]
        instance.update(lifetime_s=instance["end_time"] - instance["start_time"],
                        completed_turns=len(own), successful_turns=sum(r["success"] for r in own),
                        total_input=sum(r["prompt_len"] for r in own if r["success"]),
                        total_output=sum(r["output_len"] for r in own if r["success"]))
    rounds, previous = [], scheduler.start
    for index, end in enumerate(scheduler.round_ends):
        cohort = [x["instance"] for x in scheduler.completed[index * args.concurrency:(index + 1) * args.concurrency]]
        window = summarize(records, previous, end)
        window.update(round=index + 1, completed_tasks=cohort,
                      cumulative=summarize(records, scheduler.start, end))
        if rounds:
            window["throughput_change_fraction"] = {
                key: (window["metrics"][key] / rounds[-1]["metrics"][key] - 1
                      if rounds[-1]["metrics"][key] else None)
                for key in ("input_throughput", "output_throughput", "total_throughput")}
        rounds.append(window)
        previous = end
    overall = summarize(records, scheduler.start, cutoff)
    http_busy = sum(max(0, min(r["end_time"], cutoff) - max(r["start_time"], scheduler.start)) for r in records)
    overall.update(actual_http_concurrency=http_busy / (cutoff - scheduler.start),
                   idle_slot_seconds=args.concurrency * (cutoff - scheduler.start) - http_busy)
    failures = [x["instance"] for x in scheduler.completed if x["instance"]["status"] not in
                ("all_turns_completed", "context_limit_reached")]
    drop_turns = [r for r in records if r["end_time"] <= cutoff and r["drop_events"]]
    drop_protocol_acknowledged = not drop_turns or any(r["drop_acknowledged"] for r in drop_turns)
    result = dict(schema=1, reference_sglang=REFERENCE, started_at=started_wall,
                  args=vars(args), manifest_sha256=digest(json.loads(manifest_path.read_text())),
                  url=url, valid=not failures and overall["strict_failed_turns"] == 0
                    and all(r["usage_prompt_matches"] for r in records if r["end_time"] <= cutoff)
                    and drop_protocol_acknowledged,
                  drop_protocol_acknowledged=drop_protocol_acknowledged,
                  smoke=bool(args.smoke_max_turns or args.smoke_long_last),
                  overall=overall, rounds=rounds, tasks=scheduler.instances,
                  completion_order=[x["instance"]["case_id"] for x in scheduler.completed],
                  cutoff=cutoff, excluded_at_cutoff=[r for r in records if r["end_time"] > cutoff],
                  turns=[r for r in records if r["end_time"] <= cutoff], journal=str(journal_path),
                  notes=["compute_metrics uses actual server forward tokens; missing telemetry is null",
                         "metrics/sglang_logical_metrics are SGLang logical compatibility only",
                         "ITL and peak output count nonempty text SSE events, not exact token timestamps",
                         "round metrics assign whole HTTP turns to their completion window",
                         "compatibility metrics accept clean HTTP200 EOF; strict metrics require usage/DONE/normal finish"])
    write_json(result_path, result)
    write_json(root / "latest.json", {"result": str(result_path), "valid": result["valid"]})
    print(json.dumps({"result": str(result_path), "valid": result["valid"], "compute_metrics": overall["compute_metrics"],
                      "sglang_logical_metrics": overall["metrics"]}), flush=True)
    if not result["valid"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", action="append", required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output", required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--concurrency", type=int, default=8)
    bench.add_argument("--num-requests", type=int)
    bench.add_argument("--max-token-len", "--max-length", type=int, default=10000)
    bench.add_argument("--host", default="127.0.0.1")
    bench.add_argument("--port", type=int, default=8000)
    bench.add_argument("--post", default="/v1/chat/completions")
    bench.add_argument("--model", default="gpt-oss-120b")
    bench.add_argument("--tokenizer")
    bench.add_argument("--drop", action=argparse.BooleanOptionalAction, default=False)
    bench.add_argument("--requests-path", default=DEFAULT_INPUT)
    bench.add_argument("--output", required=True)
    bench.add_argument("--context-limit", type=int, default=LIMIT)
    bench.add_argument("--timeout", type=float, default=7200)
    bench.add_argument("--api-key-env")
    bench.add_argument("--smoke-max-turns", type=int)
    bench.add_argument("--smoke-long-last", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
