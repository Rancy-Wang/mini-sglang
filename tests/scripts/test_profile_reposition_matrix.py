from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

from scripts.compare_occurrence_replays import canonical_message, compare_pair, load_replay
from scripts.profile_reposition_matrix import (
    PROFILE_CONFIG_ENV,
    EventProfiler,
    ProfileTarget,
    create_profile_bootstrap,
    detect_slowdown,
    summarize_profile,
    targets_for_framework,
)


def test_replay_comparator_only_ignores_tool_call_uuid():
    message = {"content": "same", "reasoning_content": "same reasoning",
               "tool_calls": [{"id": "call_a", "function": {"name": "f", "arguments": "{}"}}]}
    changed = copy.deepcopy(message)
    changed["tool_calls"][0]["id"] = "call_b"
    assert canonical_message(changed) == canonical_message(message)
    assert message["tool_calls"][0]["id"] == "call_a"
    changed["tool_calls"][0]["function"]["arguments"] = "{ }"
    assert canonical_message(changed) != canonical_message(message)


def test_replay_comparator_rejects_token_difference_even_with_equal_text():
    manifest = {key: None for key in (
        "argv", "mode", "qid", "uid_range", "max_tokens", "gpus",
        "trajectory_sha256", "tools_sha256", "rolling_k", "head"
    )}
    replay = {"manifest": manifest, "inputs": {0: "fingerprint"}, "tokens": {0: [1, 2]},
              "turns": [{"uid": 0, "request_sha256": "request", "server_ttft_ms": 1,
                         "server_tpot_ms": 1, "canonical_response": {
                             "message": {"content": "same text"}, "finish_reason": "length",
                             "usage": {"prompt_tokens": 4, "completion_tokens": 2}}}]}
    candidate = copy.deepcopy(replay)
    assert compare_pair(replay, candidate)["output_pass"]
    candidate["tokens"][0] = [1, 3]
    report = compare_pair(replay, candidate)
    assert not report["output_pass"]
    assert report["turns"][0]["message_equal"]

    candidate = copy.deepcopy(replay)
    replay["turns"][0]["canonical_response"]["usage"]["completion_tokens"] = 3
    report = compare_pair(replay, candidate)
    assert report["output_pass"]
    assert not report["turns"][0]["token_count_equal"]
    assert report["turns"][0]["candidate_completion_usage_exact"]
    candidate["turns"][0]["canonical_response"]["usage"]["completion_tokens"] = 3
    assert not compare_pair(replay, candidate)["turns"][0]["candidate_completion_usage_exact"]


def test_replay_loader_preserves_probes_and_rejects_missing_requests(tmp_path):
    (tmp_path / "complete.json").write_text(json.dumps({"status": "complete", "records": 1}))
    (tmp_path / "manifest.json").write_text(json.dumps({"uid_range": [0, 0]}))
    (tmp_path / "turns.jsonl").write_text(json.dumps({
        "uid": 0, "server_metrics": {"generated_tokens": 2}
    }) + "\n")
    observer = tmp_path / "observer"
    observer.mkdir()
    (observer / "tokens-0.json").write_text(json.dumps({"0": [1, 2]}))
    rows = [{"uid": uid, "warmup": False, "tensors": {"input_ids": "hash"}}
            for uid in (-1, -2, 0)]
    inputs = observer / "inputs-1.jsonl"
    inputs.write_text("\n".join(json.dumps(row) for row in rows))
    replay = load_replay(tmp_path)
    assert sorted(replay["inputs"]) == [0]
    assert sorted(replay["probes"]) == [-2, -1]
    for invalid in (rows[:2], rows + [dict(rows[0], uid=-3)]):
        inputs.write_text("\n".join(json.dumps(row) for row in invalid))
        try:
            load_replay(tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing requests and unknown UIDs must fail")


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
