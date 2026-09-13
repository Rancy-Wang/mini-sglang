"""Compare saved paired replays without rerunning inference or hiding slow turns.

The only ignored generated-message field is the server-generated tool-call UUID.
Usage changes are reported separately; text, reasoning, argument strings, finish
reasons, token IDs and input fingerprints must match exactly. Timing summaries
are measurements, not an automatic waiver of the performance acceptance gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path


def canonical_message(message: dict) -> dict:
    result = copy.deepcopy(message)
    for call in result.get("tool_calls") or []:
        # response_parser._tool_call generates call_<uuid4> independently of
        # the model. Do not normalize tool names, argument JSON or whitespace.
        call.pop("id", None)
    return result


def load_replay(path: Path) -> dict:
    complete = json.loads((path / "complete.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    turns = [json.loads(line) for line in (path / "turns.jsonl").read_text().splitlines()]
    expected = list(range(manifest["uid_range"][0], manifest["uid_range"][1] + 1))
    if (complete != {"status": "complete", "records": len(expected)}
            or [row["uid"] for row in turns] != expected):
        raise ValueError(f"Incomplete or reordered replay: {path}")
    tokens = {}
    for file in (path / "observer").glob("tokens-*.json"):
        for key, value in json.loads(file.read_text()).items():
            uid = int(key)
            if uid in tokens or not value:
                raise ValueError(f"Duplicate or empty token capture: {path}, UID {uid}")
            tokens[uid] = value
    inputs = {}
    for file in (path / "observer").glob("inputs-*.jsonl"):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            inputs.setdefault(row["uid"], []).append((row["warmup"], row["tensors"]))
    # TokenizerServer.run startup probes are not model requests. Preserve and
    # compare them separately; never discard unknown or missing request UIDs.
    probes = {uid: inputs.pop(uid) for uid in (-1, -2) if uid in inputs}
    if sorted(tokens) != expected or sorted(inputs) != expected:
        raise ValueError(f"Missing token IDs or input fingerprints: {path}")
    if any(len(tokens[row["uid"]]) != row["server_metrics"]["generated_tokens"] for row in turns):
        raise ValueError(f"Captured token IDs do not cover the committed output stream: {path}")
    return dict(manifest=manifest, turns=turns, tokens=tokens, inputs=inputs, probes=probes)


def compare_pair(baseline: dict, candidate: dict) -> dict:
    if baseline.get("probes", {}) != candidate.get("probes", {}):
        raise ValueError("Tokenizer startup probe fingerprints differ")
    for key in ("argv", "mode", "qid", "uid_range", "max_tokens", "gpus",
                "trajectory_sha256", "tools_sha256", "rolling_k"):
        if baseline["manifest"][key] != candidate["manifest"][key]:
            raise ValueError(f"Paired configurations differ: {key}")
    rows = []
    for old, new in zip(baseline["turns"], candidate["turns"], strict=True):
        uid = old["uid"]
        old_response, new_response = old["canonical_response"], new["canonical_response"]
        row = {
            "uid": uid,
            "input_equal": old["request_sha256"] == new["request_sha256"]
                           and baseline["inputs"][uid] == candidate["inputs"][uid],
            "tokens_equal": baseline["tokens"][uid] == candidate["tokens"][uid],
            "message_equal": canonical_message(old_response["message"])
                             == canonical_message(new_response["message"]),
            "finish_equal": old_response["finish_reason"] == new_response["finish_reason"],
            "token_count_equal": all(old_response["usage"][key] == new_response["usage"][key]
                                     for key in ("prompt_tokens", "completion_tokens")),
            "prompt_count_equal": old_response["usage"]["prompt_tokens"]
                                  == new_response["usage"]["prompt_tokens"],
            "candidate_completion_usage_exact": new_response["usage"]["completion_tokens"]
                                                == len(candidate["tokens"][uid]),
            "usage_equal": old_response["usage"] == new_response["usage"],
        }
        row["output_pass"] = all(row[key] for key in (
            "input_equal", "tokens_equal", "message_equal", "finish_equal", "prompt_count_equal"
        ))
        for metric in ("server_ttft_ms", "server_tpot_ms"):
            row[metric] = {"baseline": old[metric], "candidate": new[metric],
                           "delta": new[metric] - old[metric]}
        rows.append(row)
    return {"baseline_head": baseline["manifest"]["head"],
            "candidate_head": candidate["manifest"]["head"],
            "output_pass": all(row["output_pass"] for row in rows), "turns": rows}


def summarize(pairs: list[dict]) -> dict:
    result = {}
    for group, begin, end in (("all", 0, 30), ("post_reposition", 13, 30), ("late", 24, 30)):
        metrics = {}
        for metric in ("server_ttft_ms", "server_tpot_ms"):
            trial_means = []
            for pair in pairs:
                values = [row[metric] for row in pair["turns"] if begin <= row["uid"] <= end]
                if not values:
                    continue
                trial_means.append({side: statistics.mean(value[side] for value in values)
                                    for side in ("baseline", "candidate")})
            if trial_means:
                old = statistics.mean(row["baseline"] for row in trial_means)
                new = statistics.mean(row["candidate"] for row in trial_means)
                metrics[metric] = {"baseline_ms": old, "candidate_ms": new,
                                   "delta_ms": new - old, "change_percent": 100 * (new / old - 1),
                                   "paired_trial_means": trial_means}
        result[group] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=("repos", "no_drop"),
                        default=["repos", "no_drop"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-accurate-usage", action="store_true",
                        help="Also require completion usage to equal captured committed tokens.")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("At least one complete paired replay is required.")
    result = {}
    for mode in args.modes:
        pairs = [compare_pair(
            load_replay(args.root / f"baseline_{mode}_r{repeat}"),
            load_replay(args.root / f"candidate_{mode}_r{repeat}"),
        ) for repeat in range(1, args.repeats + 1)]
        result[mode] = {"output_pass": all(pair["output_pass"] for pair in pairs),
                        "completion_usage_pass": all(row["candidate_completion_usage_exact"]
                                                     for pair in pairs for row in pair["turns"]),
                        "timing": summarize(pairs), "pairs": pairs}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not all(mode["output_pass"] for mode in result.values()):
        raise SystemExit(1)
    if args.require_accurate_usage and not all(mode["completion_usage_pass"]
                                             for mode in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
