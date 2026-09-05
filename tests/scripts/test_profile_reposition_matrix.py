from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.profile_reposition_matrix import (
    PROFILE_CONFIG_ENV,
    EventProfiler,
    ProfileTarget,
    create_profile_bootstrap,
    detect_slowdown,
    summarize_profile,
    targets_for_framework,
)


def _profile_probe(profiler: EventProfiler) -> None:
    frame = sys._getframe()
    profiler.callback(frame, "call", None)
    sum(range(100))
    profiler.callback(frame, "return", None)


def test_event_profiler_records_only_exact_mapped_function(tmp_path: Path) -> None:
    output = tmp_path / "profile-{pid}.jsonl"
    profiler = EventProfiler(
        [
            ProfileTarget(
                "scheduler",
                "tests/scripts/test_profile_reposition_matrix.py",
                "_profile_probe",
            )
        ],
        output,
        nvtx=False,
    )

    _profile_probe(profiler)
    profiler.flush()

    rows = [
        json.loads(line)
        for line in (tmp_path / f"profile-{__import__('os').getpid()}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["stage"] == "scheduler"
    assert rows[0]["calls"] == 1
    assert rows[0]["total_ns"] > 0


def test_bootstrap_is_task_local_and_covers_both_framework_maps(tmp_path: Path) -> None:
    env = create_profile_bootstrap(tmp_path / "profile", framework="minisgl", nvtx=False)

    assert Path(env[PROFILE_CONFIG_ENV]).is_file()
    assert (tmp_path / "profile" / "bootstrap" / "sitecustomize.py").is_file()
    assert "bootstrap" in env["PYTHONPATH"]
    assert {target.stage for target in targets_for_framework("minisgl")} >= {
        "tokenize",
        "scheduler",
        "prefill_extend",
        "decode",
        "radix_match",
        "free_and_cache",
        "evict",
    }
    assert {target.stage for target in targets_for_framework("sglang")} >= {
        "tokenize",
        "scheduler",
        "prefill_extend",
        "decode",
        "radix_match",
        "free_and_cache",
        "evict",
    }


def test_slowdown_requires_median_threshold_and_two_of_three_repetitions() -> None:
    slow = detect_slowdown([120, 122, 121], [100, 100, 100])
    noisy = detect_slowdown([120, 100, 100], [100, 100, 100])

    assert slow["profile_required"] is True
    assert slow["repeated_2_of_3"] is True
    assert slow["severe_2x"] is False
    assert noisy["profile_required"] is False


def test_profile_summary_aggregates_process_files(tmp_path: Path) -> None:
    first = tmp_path / "profile-1.jsonl"
    second = tmp_path / "profile-2.jsonl"
    first.write_text(
        json.dumps(
            {
                "stage": "decode",
                "filename_suffix": "model.py",
                "qualname": "decode",
                "calls": 2,
                "total_ns": 2_000_000,
            }
        )
        + "\n"
    )
    second.write_text(
        json.dumps(
            {
                "stage": "decode",
                "filename_suffix": "model.py",
                "qualname": "decode",
                "calls": 3,
                "total_ns": 4_000_000,
            }
        )
        + "\n"
    )

    summary = summarize_profile([first, second])

    assert summary["stages"]["decode"] == {
        "calls": 5,
        "total_ns": 6_000_000,
        "total_ms": 6.0,
        "mean_ms": 1.2,
    }
