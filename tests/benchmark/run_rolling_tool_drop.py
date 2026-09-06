"""Replay captured chat requests or assistant prefixes without importing an engine.

Prepare (read-only source -> compact, portable input bundle)::

    python tests/benchmark/run_rolling_tool_drop.py --source SNAPSHOT \
        --tools-json tools.json --prepare-only --output INPUTS

Replay against an already running server (httpx required)::

    python tests/benchmark/run_rolling_tool_drop.py --source INPUTS/inputs.json \
        --base-url http://127.0.0.1:31012 --model MODEL --test-type rolling_drop \
        --cases 231 798 784 800 --output RUN

Restart the server between modes/repetitions to isolate caches. Four case workers
keep each case's turns sequential; the actual server batch can be smaller than 4.
Generated outputs are measured, not inserted into subsequent historical prompts.
All durations are seconds. Client TPOT uses usage tokens, never SSE chunk counts.
Server timings use its own monotonic clock and sampled (including stop) tokens.
Physical retries are replayed only when present in accessible captures.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def load_inputs(source, tools_path=None):
    """Fail on inaccessible captures; use an explicitly selected snapshot instead."""
    source = Path(source)
    if source.is_file() and source.name != "request.json.gz":
        bundle = read_json(source)
        if bundle.get("schema") == "rolling-tool-replay-v1":
            return bundle
    paths = [source] if source.is_file() else sorted(source.glob("**/request.json.gz"))
    cases = {}
    sources = []
    if paths:
        for path in paths:
            match = re.search(
                r"case_([^/]+)/trial_(\d+)/.*?/(\d+)_agent_(\d+)_retry_(\d+)/", str(path)
            )
            if not match:
                raise ValueError(f"Cannot identify case/turn/retry from {path}")
            case_id, trial, physical, logical, retry = match.groups()
            key = f"{case_id}/trial_{int(trial)}"
            value = read_json(path)
            payload = value.get("request", value)
            if not isinstance(payload.get("messages"), list):
                raise ValueError(f"Not a chat messages request: {path}")
            case = cases.setdefault(key, {"case_id": case_id, "trial": int(trial), "turns": []})
            case["turns"].append(
                {
                    "logical_turn": int(logical),
                    "physical": int(physical),
                    "retry": int(retry),
                    "request": payload,
                    "source": str(path),
                }
            )
            sources.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        for case in cases.values():
            case["turns"].sort(key=lambda turn: turn["physical"])
        provenance = "raw_capture"
    else:
        if tools_path is None:
            raise ValueError("Trajectory reconstruction requires --tools-json (saved tool schemas)")
        tools = read_json(Path(tools_path))
        if not isinstance(tools, list) or not tools:
            raise ValueError("--tools-json must contain a nonempty OpenAI tools array")
        sources.append(
            {
                "path": str(tools_path),
                "sha256": hashlib.sha256(Path(tools_path).read_bytes()).hexdigest(),
            }
        )
        for path in sorted(source.glob("**/trajectories.jsonl")):
            run_path = path.with_name("run.json")
            generation = read_json(run_path)["configuration"]["benchmark_config"]["generation"]
            for item in (path, run_path):
                sources.append(
                    {"path": str(item), "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
                )
            rollout_path = path.with_name("rollouts.jsonl")
            historical = {}
            if rollout_path.exists():
                sources.append(
                    {
                        "path": str(rollout_path),
                        "sha256": hashlib.sha256(rollout_path.read_bytes()).hexdigest(),
                    }
                )
                with rollout_path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        historical[(str(row["case_id"]), row["trial"])] = row.get(
                            "metadata", {}
                        ).get("model_calls", [])
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    case_id, trial = str(row["case_id"]), row["trial"]
                    key = f"{case_id}/trial_{trial}"
                    if key in cases:
                        raise ValueError(f"Duplicate case/trial: {key}")
                    trajectory = row["trajectory"]
                    calls = historical.get((case_id, trial), [])
                    boundaries = [
                        i for i, msg in enumerate(trajectory) if msg["role"] == "assistant"
                    ]
                    if calls and len(calls) != len(boundaries):
                        raise ValueError(f"Assistant/logical-call count mismatch: {key}")
                    cases[key] = {
                        "case_id": case_id,
                        "trial": trial,
                        "trajectory": trajectory,
                        "generation": generation,
                        "tools": tools,
                        "source": str(path),
                        "turns": [
                            {
                                "logical_turn": n,
                                "physical": None,
                                "retry": None,
                                "prefix_end": end,
                                "historical": calls[n] if calls else None,
                            }
                            for n, end in enumerate(boundaries)
                        ],
                    }
        provenance = "trajectory_reconstructed"
    if not cases:
        raise ValueError(f"No requests or trajectories found under {source}")
    return {
        "schema": "rolling-tool-replay-v1",
        "provenance": provenance,
        "sources": sources,
        "cases": list(cases.values()),
    }


def make_request(case, turn, mode, model, max_tokens):
    if "request" in turn:
        body = copy.deepcopy(turn["request"])
    else:
        body = {k: v for k, v in case["generation"].items() if v is not None}
        body.update(
            messages=copy.deepcopy(case["trajectory"][: turn["prefix_end"]]),
            tools=copy.deepcopy(case["tools"]),
        )
    # Always remove capture-specific context controls, including nested extra_body.
    extra = body.pop("extra_body", {}) or {}
    body.update(extra)
    for key in ("drop_message", "drop_rule", "reposition"):
        body.pop(key, None)
    tool_ids = [i for i, msg in enumerate(body["messages"]) if msg["role"] == "tool"]
    if mode == "rolling_drop" and len(tool_ids) > 12:
        body["drop_message"] = {
            str(tool_ids[i]): [tool_ids[i - 12]] for i in range(12, len(tool_ids))
        }
    elif mode != "no_drop" and mode != "rolling_drop":
        raise ValueError(f"Unknown test type: {mode}")
    body.update(
        model=model,
        max_tokens=max_tokens,
        max_completion_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        n=1,
    )
    return body


def timings(start, first, end, tokens):
    return {
        "ttft_s": None if first is None else first - start,
        "e2e_s": end - start,
        "tpot_s": (end - first) / (tokens - 1)
        if first is not None and isinstance(tokens, int) and tokens > 1
        else None,
        "decode_s": None if first is None else end - first,
        "decode_intervals": max(tokens - 1, 0) if isinstance(tokens, int) else None,
    }


def server_timings(metrics):
    if not metrics:
        return None
    try:
        start, first, end = (
            metrics[key]
            for key in ("request_received_ns", "first_token_generated_ns", "request_finished_ns")
        )
        tokens = metrics["generated_tokens"]
        if not 0 <= start <= first <= end or tokens < 1:
            return None
        # Subtract integers before converting to avoid losing clock precision.
        return timings(0, (first - start) / 1e9, (end - start) / 1e9, tokens)
    except (KeyError, TypeError):
        return None


async def send_request(client, url, body):
    start = time.perf_counter()
    first = None
    usage, metrics, finish = {}, None, None
    done = False
    deltas = []
    error = None
    status = None
    try:
        async with client.stream("POST", url, json=body) as response:
            status = response.status_code
            if status != 200:
                raise RuntimeError(
                    f"HTTP {status}: {(await response.aread()).decode(errors='replace')[:4000]}"
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("server_metrics"):
                    metrics = chunk["server_metrics"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    meaningful = any(
                        delta.get(key) for key in ("content", "reasoning_content")
                    ) or any(
                        call.get("function", {}).get("name")
                        or call.get("function", {}).get("arguments")
                        for call in delta.get("tool_calls", [])
                    )
                    if meaningful:
                        if first is None:
                            first = time.perf_counter()
                        deltas.append(delta)
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
        if not done or finish is None:
            raise RuntimeError("Incomplete SSE response: missing [DONE] or finish_reason")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {
        "ok": error is None,
        "error": error,
        "http_status": status,
        "usage": usage,
        "finish_reason": finish,
        "server_metrics": metrics,
        "server": server_timings(metrics),
        "client": timings(start, first, end, usage.get("completion_tokens")),
        "deltas": deltas,
    }


def summarize(rows, wall_s):
    def group_stats(group):
        good = [row for row in group if row["ok"]]
        result = {"attempts": len(group), "successes": len(good), "errors": len(group) - len(good)}
        for clock in ("client", "server"):
            samples = [row[clock] for row in good if row.get(clock)]
            stats = {}
            for key in ("ttft_s", "tpot_s", "e2e_s"):
                values = [sample[key] for sample in samples if sample[key] is not None]
                units = {
                    (row["case_id"], row["trial"], row["logical_turn"])
                    for row in good
                    if row.get(clock) and row[clock][key] is not None
                }
                stats[key] = {
                    "count": len(values),
                    "sum": sum(values) if values else None,
                    "mean": sum(values) / len(values) if values else None,
                    "case_turn_count": len(units),
                    "mean_case_turn_sum": sum(values) / len(units) if units else None,
                }
            eligible = [sample for sample in samples if sample["tpot_s"] is not None]
            intervals = sum(sample["decode_intervals"] for sample in eligible)
            stats["weighted_tpot_s"] = (
                sum(sample["decode_s"] for sample in eligible) / intervals if intervals else None
            )
            stats["decode_s_sum"] = sum(sample["decode_s"] for sample in eligible)
            result[clock] = stats
        return result

    output = {
        "global_stats": group_stats(rows),
        "wall_s": wall_s,
        "successful_requests_per_s": sum(row["ok"] for row in rows) / wall_s if wall_s else None,
        "completion_tokens_per_s": sum(
            row.get("usage", {}).get("completion_tokens", 0) for row in rows if row["ok"]
        )
        / wall_s
        if wall_s
        else None,
        "failed_client_e2e_s_sum": sum(row["client"]["e2e_s"] for row in rows if not row["ok"]),
    }
    for name, keys in (
        ("per_case", ("case_id", "trial")),
        ("per_turn", ("logical_turn",)),
        ("per_case_turn", ("case_id", "trial", "logical_turn")),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        output[name] = [dict(zip(keys, key), **group_stats(group)) for key, group in groups.items()]
    return output


async def replay(cases, execute, save):
    """Fixed four case slots, no concurrent turns from a single case."""
    queue = asyncio.Queue()
    for case in cases:
        queue.put_nowait(case)

    async def worker(slot):
        while not queue.empty():
            case = queue.get_nowait()
            for turn in case["turns"]:
                row = await execute(case, turn)
                row.update(
                    case_id=case["case_id"],
                    trial=case["trial"],
                    slot=slot,
                    logical_turn=turn["logical_turn"],
                    physical=turn["physical"],
                    retry=turn["retry"],
                )
                save(row)

    await asyncio.gather(*(worker(slot) for slot in range(4)))


async def run(args, bundle, cases):
    import httpx

    url = args.base_url.rstrip("/")
    url += "/chat/completions" if url.endswith("/v1") else "/v1/chat/completions"
    rows = []
    start = time.perf_counter()
    with (args.output / "turns.jsonl").open("x") as stream:

        def save(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            # Large generated deltas remain on disk, not in aggregate memory.
            rows.append({key: value for key, value in row.items() if key != "deltas"})
            print(
                json.dumps({key: row[key] for key in ("case_id", "logical_turn", "ok", "client")}),
                flush=True,
            )

        async with httpx.AsyncClient(
            timeout=args.timeout, trust_env=False, limits=httpx.Limits(max_connections=4)
        ) as client:

            async def execute(case, turn):
                body = make_request(case, turn, args.test_type, args.model, args.max_tokens)
                row = await send_request(client, url, body)
                row.update(
                    request_sha256=digest(body),
                    messages_sha256=digest(body["messages"]),
                    tool_responses=sum(msg["role"] == "tool" for msg in body["messages"]),
                    drop_events=len(body.get("drop_message", {})),
                    historical=turn.get("historical"),
                    provenance=bundle["provenance"],
                )
                return row

            await replay(cases, execute, save)
    summary = summarize(rows, time.perf_counter() - start)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary["global_stats"]["errors"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing output is never overwritten",
    )
    parser.add_argument(
        "--tools-json",
        type=Path,
        help="Tool schemas for reconstructed input, with provenance recorded",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--test-type", choices=("no_drop", "rolling_drop"))
    parser.add_argument("--cases", nargs="+", help="Case IDs in scheduling order; default all")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--run-label", default="", help="External launch/HEAD/repetition identifier"
    )
    args = parser.parse_args()
    if not args.prepare_only and not all((args.base_url, args.model, args.test_type)):
        parser.error("Replay requires --base-url, --model and --test-type")
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    bundle = load_inputs(args.source, args.tools_json)
    cases = bundle["cases"]
    if args.cases:
        missing = set(args.cases) - {case["case_id"] for case in cases}
        if missing or len(set(args.cases)) != len(args.cases):
            parser.error(f"Missing or duplicate --cases: {args.cases}, missing={sorted(missing)}")
        cases = sorted(
            (case for case in cases if case["case_id"] in args.cases),
            key=lambda case: args.cases.index(case["case_id"]),
        )
    if not args.prepare_only and len(cases) < 4:
        parser.error("This benchmark requires at least four distinct case trajectories (bs=4)")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "argv": sys.argv,
        "input_sha256": digest(bundle),
        "provenance": bundle["provenance"],
        "sources": bundle["sources"],
        "case_order": [case["case_id"] for case in cases],
        "turns": sum(len(case["turns"]) for case in cases),
        "concurrency": 4,
        "max_tokens": args.max_tokens,
        "test_type": args.test_type,
        "created_unix_s": time.time(),
        "run_label": args.run_label,
        "notes": "Historical prompts; natural generation; client and server clocks separate; failed timings excluded from success aggregates",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.prepare_only:
        bundle = dict(bundle, cases=cases)
        (args.output / "inputs.json").write_text(json.dumps(bundle, ensure_ascii=False) + "\n")
        print(json.dumps(metadata, indent=2))
        return 0
    return 1 if asyncio.run(run(args, bundle, cases)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
