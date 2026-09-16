"""Forward-token accounting. Unknown telemetry stays unknown, never logical throughput."""
from __future__ import annotations

from collections import Counter

LEGACY_HEAD = "f089bf66ebf3200dbb4b96e99ad69765da474af2"


def terminal_metrics(record):
    for event in reversed(record.get("events", [])):
        metrics = event.get("data", {}).get("server_metrics")
        if metrics is not None:
            return metrics
    return record.get("server_metrics") or {}


def mask_extend(prompt, matched, spans):
    """Replay mask-free eligibility for final-dead [start,end,expiry] spans.

    Legacy page-size=1/no-hole matches cover [0,matched). Surviving queries
    must not see any final-dead key; otherwise the planner uses the full stream.
    Ranges are disjoint, sorted, and contain only effective drops.
    """
    def next_kept(position):
        for lo, hi, _ in spans:
            if lo <= position < hi:
                position = hi
        return position
    compact = all(next_kept(max(matched, lo)) >= min(expiry, prompt)
                  for lo, _, expiry in spans)
    if not compact:
        return prompt - matched, "full_mask"
    uncached_dead = sum(max(0, hi - max(lo, matched)) for lo, hi, _ in spans)
    return prompt - matched - uncached_dead, "compact_mask_free"


def legacy_proof(status, evidence, manifest_hash):
    if not status or status.get("state") != "completed" or status.get("head") != LEGACY_HEAD:
        return False
    audits = status.get("audit", [])
    return (status.get("input_hash") == manifest_hash
            and evidence.get("manifest_sha256") == manifest_hash
            and evidence.get("engine_head") == LEGACY_HEAD
            and evidence.get("profile") == "mask-paged-occurrence-page1-no-holes"
            and len(audits) == 2 and all(a.get("passed") and
                a.get("eviction", {}).get("drop_pages") == 0 and
                a.get("eviction", {}).get("hole_fills") == 0 for a in audits))


def turn_compute(record, *, legacy=False, evidence=None):
    """Exact forward counters first; narrowly verified legacy reconstruction second.

    Request eligibility follows strict successful HTTP turns. A failed/aborted
    turn's partial work is excluded, matching calculate_metrics' success filter.
    """
    if not record.get("strict_success"):
        return dict(included=False, source="failed_or_aborted")
    metrics = terminal_metrics(record)
    usage = record.get("usage") or {}
    generated = metrics.get("generated_tokens", usage.get("completion_tokens"))
    prefill, decode = (metrics.get(k) for k in ("prefill_compute_tokens", "decode_compute_tokens"))
    if prefill is not None and decode is not None and metrics.get("context_stage_count", 0) <= 1:
        if min(prefill, decode) < 0:
            raise ValueError("Negative forward counters")
        return dict(included=True, source="server_forward_counters", prefill_tokens=prefill,
                    decode_tokens=decode, decode_lower=decode, decode_upper=decode,
                    generated_tokens=generated)
    if not legacy:
        return dict(included=True, source="missing_verified_forward_counters",
                    prefill_tokens=None, decode_tokens=None, generated_tokens=generated)
    prompt = usage.get("prompt_tokens")
    if prompt != record["prompt_len"] or generated != usage.get("completion_tokens"):
        raise ValueError("Legacy prompt/generated telemetry mismatch")
    details = usage.get("prompt_tokens_details", {})
    excluded = {key: int(details.get(key, 0)) for key in
                ("cached_tokens", "drop_skipped_tokens", "repos_tokens")}
    matched = sum(excluded.values())
    if min(excluded.values()) < 0 or not 0 <= matched < prompt:
        raise ValueError("Invalid resident prefix accounting")
    path = "ordinary"
    prefill = prompt - matched
    if record.get("reposition"):
        path = "paged_occurrence_no_holes"
    elif record.get("drop_events"):
        key = f"{record['case_id']}:{record['turn']}"
        plan = (evidence or {}).get("turns", {}).get(key)
        if plan is None or plan["prompt_tokens"] != prompt:
            return dict(included=True, source="missing_drop_boundary_evidence", prefill_tokens=None,
                        decode_tokens=None, generated_tokens=generated)
        prefill, path = mask_extend(prompt, matched, plan["dead_spans"])
    if metrics.get("context_stage_count", 0) > 1:
        raise ValueError("Legacy staged work cannot be reconstructed from final usage")
    # The first sampled token comes from Prefill. In overlap mode one additional
    # Decode may run before EOS/stop is observed. The old response has no record
    # of whether Prefill priority prevented that speculative batch.
    lower = max(generated - 1, 0)
    upper = lower if generated >= record["requested_max_tokens"] else generated
    return dict(included=True, source="verified_legacy_usage", path=path,
                prefill_tokens=prefill, decode_tokens=lower if lower == upper else None,
                decode_lower=lower, decode_upper=upper, generated_tokens=generated,
                excluded_reuse=excluded)


def compute_summary(records, duration):
    if duration <= 0:
        raise ValueError("duration must be positive")
    rows = [r.get("compute") or turn_compute(r) for r in records]
    rows = [r for r in rows if r["included"]]
    prefill_known = all(r.get("prefill_tokens") is not None for r in rows)
    decode_known = all(r.get("decode_tokens") is not None for r in rows)
    bounds_known = all(r.get("decode_lower") is not None for r in rows)
    prefill = sum(r["prefill_tokens"] for r in rows) if prefill_known else None
    decode = sum(r["decode_tokens"] for r in rows) if decode_known else None
    low = sum(r["decode_lower"] for r in rows) if bounds_known else None
    high = sum(r["decode_upper"] for r in rows) if bounds_known else None
    rate = lambda value: None if value is None else value / duration
    total = None if prefill is None or decode is None else prefill + decode
    excluded = Counter()
    for r in rows:
        excluded.update(r.get("excluded_reuse", {}))
    return dict(scope="strict_successful_turns_by_completion_window; excludes aborted partial work and graph padding",
                duration_s=duration, completed_turns=len(rows),
                exact=prefill_known and decode_known,
                unknown_prefill_turns=sum(r.get("prefill_tokens") is None for r in rows),
                unknown_decode_turns=sum(r.get("decode_tokens") is None for r in rows),
                sources=dict(Counter(r["source"] for r in rows)), excluded_reuse=dict(excluded),
                prefill_tokens=prefill, decode_tokens=decode, all_tokens=total,
                prefill_throughput=rate(prefill), decode_throughput=rate(decode), all_throughput=rate(total),
                decode_tokens_lower=low, decode_tokens_upper=high,
                decode_throughput_lower=rate(low), decode_throughput_upper=rate(high),
                all_throughput_lower=rate(prefill + low) if prefill is not None and low is not None else None,
                all_throughput_upper=rate(prefill + high) if prefill is not None and high is not None else None,
                generated_output_tokens=sum(r.get("generated_tokens") or 0 for r in rows),
                generated_output_throughput=sum(r.get("generated_tokens") or 0 for r in rows) / duration)


def recalculate_document(document, status, evidence):
    verified = legacy_proof(status, evidence, document["manifest_sha256"])
    if verified and (document["args"]["concurrency"] != status.get("concurrency") or
                     document["args"]["drop"] != status.get("drop")):
        raise ValueError("Audit does not belong to this measurement setting")
    records = []
    for record in document["turns"]:
        records.append(dict(record, compute=turn_compute(record, legacy=verified, evidence=evidence)))
    def window(original):
        start, end = original["start_time"], original["end_time"]
        own = [r for r in records if start < r["end_time"] <= end]
        result = dict(start_time=start, end_time=end,
                      compute_metrics=compute_summary(own, end-start),
                      sglang_logical_metrics=original["metrics"],
                      first_pass_compute=compute_summary([r for r in own if not r["filler"]], end-start),
                      filler_compute=compute_summary([r for r in own if r["filler"]], end-start))
        if "round" in original:
            result["round"] = original["round"]
        if "cumulative" in original:
            result["cumulative"] = window(original["cumulative"])
        return result
    return dict(schema=2, legacy_reconstruction_verified=verified,
                manifest_sha256=document["manifest_sha256"], engine_head=(status or {}).get("head"),
                overall=window(document["overall"]), rounds=[window(r) for r in document["rounds"]],
                turns=[{k: r.get(k) for k in ("case_id", "turn", "instance", "filler", "compute")} for r in records],
                excluded_at_cutoff=len(document.get("excluded_at_cutoff", [])),
                notes=["No inference rerun; raw responses remain unchanged.",
                       "Legacy Decode/All bounds reflect unrecorded overlap work; null is not zero.",
                       "Rates use benchmark wall time, not GPU busy time or FLOPS.",
                       "Whole turns are attributed to their completion window, not exact GPU-time slices."])


def prepare_evidence(args):
    """CPU-only tokenization of stored trajectories; never submit model requests."""
    import gzip
    import json
    import subprocess
    from pathlib import Path
    import numpy as np
    from minisgl.core import SamplingParams
    from minisgl.message.tokenizer import TokenizeMsg
    from minisgl.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer
    from test_throughput import digest, write_json

    import inspect
    if Path(inspect.getfile(TokenizeManager)).resolve().is_relative_to(Path(args.engine_repo).resolve()) is False:
        raise ValueError("Tokenizer must come from the recorded engine checkout")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.engine_repo, text=True).strip()
    launch = json.loads(Path(args.launch).read_text())
    argv = launch["argv"]
    required = {"--page-size": "1", "--contextual-prefill-mode": "mask",
                "--reposition-execution-mode": "paged-occurrence", "--radix-drop-key-mode": "delta-marker"}
    if head != LEGACY_HEAD or launch["head"] != head or any(
        flag not in argv or argv[argv.index(flag)+1] != value for flag, value in required.items()
    ):
        raise ValueError("Legacy reconstruction needs the audited engine/configuration")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    manager = TokenizeManager(AutoTokenizer.from_pretrained(manifest["tokenizer"], local_files_only=True),
                              radix_drop_key_mode="delta-marker")
    result = dict(profile="mask-paged-occurrence-page1-no-holes", engine_head=head,
                  manifest_sha256=digest(manifest), launch=launch, turns={})
    for case in manifest["cases"]:
        with gzip.open(manifest_path.parent / case["file"], "rt") as stream:
            trajectory = json.load(stream)
        if digest(trajectory) != case["trajectory_sha256"]:
            raise ValueError("Source trajectory changed")
        for turn in case["turns"]:
            if not turn["drop_message"] or turn["reposition"]:
                continue
            t = manager.tokenize([TokenizeMsg(uid=0, text=trajectory[:turn["end"]],
                sampling_params=SamplingParams(max_tokens=1), tools=manifest["tools"],
                drop_message={int(k): v for k, v in turn["drop_message"].items()})])[0]
            keep = t.full_keep_mask.numpy().astype(bool)
            expiry = t.full_token_visible_until.numpy()
            if len(keep) != turn["full_tokens"] or int(keep.sum()) != turn["active_tokens"]:
                raise ValueError("Production tokenization differs from recorded source")
            spans = []
            for position in np.flatnonzero(~keep):
                p, e = int(position), int(expiry[position])
                if spans and spans[-1][1] == p and spans[-1][2] == e:
                    spans[-1][1] += 1
                else:
                    spans.append([p, p+1, e])
            result["turns"][f"{case['case_id']}:{turn['turn']}"] = dict(
                prompt_tokens=len(keep), active_tokens=int(keep.sum()), dead_spans=spans)
        print(json.dumps({"evidence_case": case["case_id"], "turns": len(result["turns"])}), flush=True)
    write_json(args.output, result)


def recalculate_file(result_path, status_path=None, evidence_path=None, output_path=None):
    import hashlib
    import json
    import subprocess
    from pathlib import Path
    from test_throughput import write_json

    path = Path(result_path)
    raw = path.read_bytes()
    document = json.loads(raw)
    status = json.loads(Path(status_path).read_text()) if status_path else None
    evidence = json.loads(Path(evidence_path).read_text()) if evidence_path else {}
    if status and Path(status.get("result", "")).name != path.name:
        raise ValueError("Audit result filename does not match raw response file")
    report = recalculate_document(document, status, evidence)
    report.update(source_result=str(path.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                  analysis_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[2], text=True).strip())
    output = Path(output_path) if output_path else path.with_suffix(".compute.json")
    if output.resolve() == path.resolve():
        raise ValueError("Never overwrite raw measurements")
    write_json(output, report)
    print(json.dumps({"result": str(output), "compute_metrics": report["overall"]["compute_metrics"]}))
    return output


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("evidence")
    for name in ("manifest", "engine-repo", "launch", "output"):
        prepare.add_argument("--" + name, required=True)
    calc = commands.add_parser("recalculate")
    calc.add_argument("--result", required=True)
    calc.add_argument("--status")
    calc.add_argument("--evidence")
    calc.add_argument("--output")
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--root", required=True)
    matrix.add_argument("--evidence", required=True)
    args = parser.parse_args()
    if args.command == "evidence":
        prepare_evidence(args)
    elif args.command == "recalculate":
        recalculate_file(args.result, args.status, args.evidence, args.output)
    else:
        for status_path in sorted(Path(args.root).glob("*/*/status.json")):
            status = json.loads(status_path.read_text())
            if status.get("state") == "completed" and status.get("result"):
                result = Path(status["result"])
                target = result.with_suffix(".compute.json")
                if not target.exists():
                    recalculate_file(result, status_path, args.evidence, target)


if __name__ == "__main__":
    main()
