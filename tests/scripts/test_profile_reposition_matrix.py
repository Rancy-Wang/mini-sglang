from __future__ import annotations

import json
import sys
import time
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


def _automatic_profile_probe() -> None:
    sum(range(100))


def _profile_child(profiler: EventProfiler) -> None:
    frame = sys._getframe()
    profiler.callback(frame, "call", None)
    sum(range(100))
    profiler.callback(frame, "return", None)


def _profile_parent(profiler: EventProfiler) -> None:
    frame = sys._getframe()
    profiler.callback(frame, "call", None)
    _profile_child(profiler)
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
    assert rows[0]["self_ns"] > 0
    assert rows[0]["samples_ns"] == [rows[0]["total_ns"]]


def test_event_profiler_ignores_calls_until_explicit_activation(tmp_path: Path) -> None:
    profiler = EventProfiler(
        [
            ProfileTarget(
                "scheduler",
                "tests/scripts/test_profile_reposition_matrix.py",
                "_profile_probe",
            )
        ],
        tmp_path / "profile-{pid}.jsonl",
        nvtx=False,
        activation_file=tmp_path / "ready.marker",
    )

    _profile_probe(profiler)
    profiler.activate()
    _profile_probe(profiler)
    profiler.flush()

    row = json.loads(next(tmp_path.glob("profile-*.jsonl")).read_text())
    assert row["calls"] == 1
    assert row["activated_ns"] is not None


def test_event_profiler_marker_activation_writes_periodic_snapshot(tmp_path: Path) -> None:
    activation_file = tmp_path / "ready.marker"
    profiler = EventProfiler(
        [
            ProfileTarget(
                "scheduler",
                "tests/scripts/test_profile_reposition_matrix.py",
                "_automatic_profile_probe",
            )
        ],
        tmp_path / "profile-{pid}.jsonl",
        nvtx=False,
        activation_file=activation_file,
        snapshot_interval_s=0.01,
    )
    profiler.install()
    try:
        _automatic_profile_probe()
        activation_file.touch()
        deadline = time.monotonic() + 1
        while not profiler._enabled and time.monotonic() < deadline:
            time.sleep(0.01)
        assert profiler._enabled
        _automatic_profile_probe()
        output = tmp_path / f"profile-{__import__('os').getpid()}.jsonl"
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert output.exists()
    finally:
        profiler.flush()

    row = json.loads(output.read_text())
    assert row["calls"] == 1


def test_event_profiler_separates_nested_inclusive_and_self_time(tmp_path: Path) -> None:
    profiler = EventProfiler(
        [
            ProfileTarget(
                "integrity",
                "tests/scripts/test_profile_reposition_matrix.py",
                name,
            )
            for name in ("_profile_parent", "_profile_child")
        ],
        tmp_path / "profile-{pid}.jsonl",
        nvtx=False,
    )

    _profile_parent(profiler)
    profiler.flush()

    summary = summarize_profile(list(tmp_path.glob("profile-*.jsonl")))
    functions = {row["qualname"]: row for row in summary["functions"]}
    assert functions["_profile_parent"]["total_ns"] > functions["_profile_parent"]["self_ns"]
    assert functions["_profile_parent"]["sample_count"] == 1
    assert functions["_profile_child"]["total_ns"] >= functions["_profile_child"]["self_ns"]
    assert summary["stages"]["integrity"]["p95_ms"] > 0


def test_bootstrap_is_task_local_and_covers_both_framework_maps(tmp_path: Path) -> None:
    activation_file = tmp_path / "ready.marker"
    env = create_profile_bootstrap(
        tmp_path / "profile",
        framework="minisgl",
        nvtx=False,
        activation_file=activation_file,
        snapshot_interval_s=0.25,
    )

    assert Path(env[PROFILE_CONFIG_ENV]).is_file()
    assert (tmp_path / "profile" / "bootstrap" / "sitecustomize.py").is_file()
    assert "bootstrap" in env["PYTHONPATH"]
    config = json.loads(Path(env[PROFILE_CONFIG_ENV]).read_text())
    assert config["activation_file"] == str(activation_file.resolve())
    assert config["snapshot_interval_s"] == 0.25
    assert {target.stage for target in targets_for_framework("minisgl")} >= {
        "tokenize",
        "reposition_sequence",
        "serialization",
        "scheduler",
        "scheduler_loop",
        "scheduler_prepare",
        "scheduler_forward",
        "scheduler_result",
        "scheduler_receive",
        "scheduler_reply",
        "scheduler_host",
        "scheduler_metrics",
        "scheduler_ipc",
        "prefill_extend",
        "decode",
        "radix_match",
        "free_and_cache",
        "evict",
    }
    minisgl_names = {target.qualname for target in targets_for_framework("minisgl")}
    assert "CacheManager.match_occurrence_req" in minisgl_names
    assert "CacheManager._match_req" in minisgl_names
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

    assert summary["stages"]["decode"]["calls"] == 5
    assert summary["stages"]["decode"]["total_ns"] == 6_000_000
    assert summary["stages"]["decode"]["self_ns"] == 6_000_000
    assert summary["stages"]["decode"]["total_ms"] == 6.0
    assert summary["stages"]["decode"]["mean_ms"] == 1.2
    assert summary["stages"]["decode"]["sample_count"] == 0
