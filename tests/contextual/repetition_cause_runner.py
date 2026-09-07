"""Classify the R9 independent-runtime repetition intervention.

This analyser intentionally consumes frozen JSON artifacts instead of loading a
model.  It only emits a causal verdict when cache cloning leaves the failure
unchanged and recomputing every survivor KV turns the failure off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _select(
    payload: dict[str, Any],
    case: str,
    *,
    clone: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    matches = [
        row
        for row in payload["results"]
        if row["case"]["name"] == case
        and bool(row.get("clone_survivors")) == clone
        and bool(row.get("rebuild_survivors")) == rebuild
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {case!r} result for clone={clone}, rebuild={rebuild}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _longest_run(tokens: list[int]) -> int:
    longest = current = 0
    previous = None
    for token in tokens:
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def diagnose_reference(
    payload: dict[str, Any],
    retention_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload.get("status") != "completed":
        raise ValueError("Reference artifact is not complete.")
    no_drop = _select(payload, "original_no_drop")
    drop = _select(payload, "original_drop_first_user")
    copied = _select(payload, "original_drop_first_user", clone=True)
    rebuilt = _select(payload, "original_drop_first_user", rebuild=True)
    twice_drop = _select(payload, "two_turn_drop_first_user")
    twice_copied = _select(payload, "two_turn_drop_first_user", clone=True)
    twice_rebuilt = _select(payload, "two_turn_drop_first_user", rebuild=True)
    middle_drop = _select(payload, "two_turn_drop_second_user")
    middle_rebuilt = _select(payload, "two_turn_drop_second_user", rebuild=True)
    system_drop = _select(payload, "system_kept_drop_first_user")
    system_rebuilt = _select(payload, "system_kept_drop_first_user", rebuild=True)
    orphan = _select(payload, "orphaned_assistant_no_drop")
    question_only = _select(payload, "question_only_no_drop")

    checks = {
        "same_canonical_prompt": drop["canonical_sha256"] == no_drop["canonical_sha256"],
        "drop_repeats": _longest_run(drop["tokens"]) >= 8,
        "fresh_allocation_preserves_failure": copied["tokens"] == drop["tokens"],
        "survivor_rebuild_restores_baseline": rebuilt["tokens"] == no_drop["tokens"],
        "longer_drop_repeats": _longest_run(twice_drop["tokens"]) >= 8,
        "longer_fresh_allocation_preserves_failure": (
            twice_copied["tokens"] == twice_drop["tokens"]
        ),
        "longer_survivor_rebuild_restores_baseline": (
            twice_rebuilt["tokens"] == no_drop["tokens"]
        ),
        "middle_drop_control_unchanged": middle_drop["tokens"] == middle_rebuilt["tokens"],
        "kept_system_control_unchanged": system_drop["tokens"] == system_rebuilt["tokens"],
        "contiguous_orphan_control_is_normal": orphan["tokens"] == no_drop["tokens"],
        "question_only_control_does_not_repeat": _longest_run(question_only["tokens"]) < 8,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"R9 causal gates failed: {failed}")

    retention = None
    if retention_payload is not None:
        if (
            retention_payload["fingerprint"]["model_manifest"]
            != payload["fingerprint"]["model_manifest"]
        ):
            raise ValueError("Retention and rebuild artifacts use different model files.")
        rows = retention_payload.get("minimal_retention_interventions", [])
        singleton_rows = {
            tuple(row["retained_removed_raw"]): _longest_run(row["tokens"])
            for row in rows
            if len(row["retained_removed_raw"]) == 1
        }
        retention = {
            "tested_interventions": len(rows),
            "singletons": [
                {
                    "retained_removed_raw": list(raw),
                    "longest_identical_token_run": run,
                    "mechanical_repetition": run >= 5,
                }
                for raw, run in sorted(singleton_rows.items())
            ],
        }

    return {
        "verdict": "stale_survivor_kv_after_drop",
        "cause": (
            "Deleted token rows are removed, but later survivor K/V rows were computed while "
            "those tokens were still visible. Copying those rows cannot erase the indirect "
            "dependency; recomputing all survivor rows does."
        ),
        "checks": checks,
        "fingerprint": payload["fingerprint"],
        "primary_tokens": {
            "no_drop": no_drop["tokens"],
            "drop": drop["tokens"],
            "copied": copied["tokens"],
            "rebuilt": rebuilt["tokens"],
        },
        "retention": retention,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--retention-reference")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnosis output: {output}")
    payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    retention_payload = (
        json.loads(Path(args.retention_reference).read_text(encoding="utf-8"))
        if args.retention_reference
        else None
    )
    diagnosis = diagnose_reference(payload, retention_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": diagnosis["verdict"], "output": str(output)}))


if __name__ == "__main__":
    main()
