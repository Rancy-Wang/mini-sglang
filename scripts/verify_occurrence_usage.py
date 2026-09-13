"""Independent page-lineage oracle for diagnostic occurrence execution traces.

No production usage masks/counters are used to calculate the expected counts.
Each trace describes one completed request, with initial Radix pages and ordered
chunks containing actual birth writes, transform arguments and attention reads.
Tracing is deliberately offline: it must not run in performance measurements.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def audit_request(trace: dict) -> dict:
    # page -> (raw token, current position, initial-hit ancestry, rotated here)
    pages: dict[int, tuple[int, int, bool, bool]] = {}
    initial: set[int] = set()
    for row in trace["initial_pages"]:
        page, raw, pos = (int(row[name]) for name in ("page", "raw", "position"))
        if page in pages or raw in initial:
            raise ValueError("Initial cache identities must be unique.")
        pages[page] = (raw, pos, True, False)
        initial.add(raw)
    used: set[int] = set()
    rotated: set[int] = set()
    visible_candidates: set[int] = set()
    expiry = trace["visible_until"]
    for chunk in trace["chunks"]:
        for row in chunk["birth_writes"]:
            page, raw, pos = (int(row[name]) for name in ("page", "raw", "position"))
            pages[page] = (raw, pos, False, False)
        # This order is the actual kernel invocation order, not raw-ID sorting.
        for row in chunk["transforms"]:
            source, dest, old, new = (int(row[name]) for name in
                                      ("source", "destination", "old", "new"))
            raw, position, inherited, changed = pages[source]
            if position != old:
                raise ValueError(f"Transform source position mismatch: {position} != {old}")
            if source == dest:
                raise ValueError("Occurrence transform overwrites its source page.")
            pages[dest] = (raw, new, inherited, changed or old != new)
        reads = list(chunk.get("attention_reads", []))
        for segment in chunk.get("attention_segments", []):
            start, end = segment["query_start"], segment["query_end"]
            selected = segment["pages"]
            query_count = end - start
            if query_count <= 0 or len(selected) < query_count:
                raise ValueError("Invalid causal attention segment.")
            raw_ids = [pages[page][0] for page in selected]
            if raw_ids[-query_count:] != list(range(start, end)):
                raise ValueError("Attention segment query tail is not in raw order.")
            prefix = raw_ids[:-query_count]
            if set(prefix) != {raw for raw in range(start) if start < expiry[raw]}:
                raise ValueError("Attention segment prefix visibility mismatch.")
            if any(expiry[raw] < end for raw in raw_ids):
                raise ValueError("An attention segment crosses a visibility boundary.")
            # With an ordered causal tail and constant prefix visibility, the last
            # query's keys are exactly the union of every query's reads. This
            # proves all queries without an O(query_count * prefix_length) trace.
            reads.append({"query_raw": end - 1, "pages": selected})
        for read in reads:
            query = int(read["query_raw"])
            actual_raw: set[int] = set()
            for page in read["pages"]:
                raw, _pos, inherited, changed = pages[int(page)]
                if raw > query or query >= expiry[raw]:
                    raise ValueError(f"Invisible or noncausal page read: raw={raw}, query={query}")
                if raw in actual_raw:
                    raise ValueError("A query reads two occurrences of the same raw token.")
                actual_raw.add(raw)
                if inherited:
                    used.add(raw)
                    if changed:
                        rotated.add(raw)
            # The diagnostic stream records full attention, not sliding-only reads.
            expected = {raw for raw in range(query + 1) if query < expiry[raw]}
            if actual_raw != expected:
                raise ValueError(f"Full attention visibility mismatch at query {query}")
            visible_candidates.update(initial & expected)
    if used != visible_candidates:
        raise ValueError("An initially matched visible token was recomputed instead of reused.")
    if not rotated <= used <= initial:
        raise ValueError("Invalid cache ancestry partition.")
    details = {
        "cached_tokens": len(used - rotated),
        "repos_tokens": len(rotated),
        "drop_skipped_tokens": len(initial - used),
    }
    return {"uid": trace["uid"], "expected": details,
            "used_raw": sorted(used), "rotated_raw": sorted(rotated),
            "skipped_raw": sorted(initial - used)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    responses = {row["uid"]: row for row in
                 map(json.loads, args.responses.read_text().splitlines())}
    reports = []
    for path in sorted(args.trace_dir.glob("request-*.json*")):
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                trace = json.load(stream)
        else:
            trace = json.loads(path.read_text())
        report = audit_request(trace)
        response = responses[report["uid"]]
        usage = response.get("canonical_response", response)["usage"]
        details = usage.get("prompt_tokens_details", {})
        report["reported"] = {name: int(details.get(name, 0)) for name in report["expected"]}
        report["passed"] = report["reported"] == report["expected"]
        reports.append(report)
    if not reports:
        raise ValueError("No request traces found; an empty audit cannot pass.")
    result = {"passed": all(row["passed"] for row in reports), "requests": reports}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
