from __future__ import annotations

import pytest
from pydantic import ValidationError

from minisgl.benchmark.contextualize.manifest import (
    MESSAGE_SHAPES,
    CaptureRecord,
    CaseMetadata,
    ManifestCase,
    classify_assistant_message,
    coverage_matrix,
    dump_jsonl,
    load_full_trajectories,
    load_manifest,
    request_hash,
)


def _message(shape: str):
    message = {"role": "assistant"}
    if "reasoning" in shape:
        message["reasoning_content"] = "reason"
    if "content" in shape:
        message["content"] = "answer"
    if "toolcall" in shape:
        message["tool_calls"] = [{"type": "function", "function": {"name": "lookup"}}]
    return message


def _case(case_id: str, shape: str, *, method: str = "full", summary_triggered=None):
    request = {
        "model": "model-a",
        "messages": [_message(shape), {"role": "user", "content": "next"}],
    }
    return ManifestCase(
        case_id=case_id,
        request=request,
        request_sha256=request_hash(request),
        metadata=CaseMetadata(
            source="tau3",
            method=method,
            model="model-a",
            summary_triggered=summary_triggered,
            target_message_index=0,
        ),
    )


def _trajectory_turn(task_id: int, turn_id: int, messages):
    request = {
        "model": f"model-{task_id}",
        "messages": messages,
        "stream": True,
    }
    return CaptureRecord(
        capture_id=f"task-{task_id}-turn-{turn_id}",
        captured_at_ns=turn_id,
        request=request,
        request_sha256=request_hash(request),
    )


def test_classifier_covers_all_seven_nonempty_combinations():
    assert {classify_assistant_message(_message(shape)) for shape in MESSAGE_SHAPES} == set(
        MESSAGE_SHAPES
    )
    assert classify_assistant_message({"role": "assistant", "content": ""}) is None
    assert classify_assistant_message({"role": "user", "content": "answer"}) is None


def test_manifest_hash_is_verified():
    case = _case("case-1", "content")
    raw = case.model_dump(mode="json")
    raw["request"]["messages"][0]["content"] = "tampered"

    with pytest.raises(ValidationError, match="request hash mismatch"):
        ManifestCase.model_validate(raw)


def test_summary_requires_a_recorded_trigger_and_dropkv_is_deferred():
    summary = _case("summary", "content", method="summary", summary_triggered=False)
    dropkv = _case("dropkv", "content", method="summary_drop_kv", summary_triggered=True)

    with pytest.raises(ValueError, match="summary_triggered=true"):
        summary.ensure_correctness_scope()
    with pytest.raises(ValueError, match="DropKV correctness is deferred"):
        dropkv.ensure_correctness_scope()


def test_correctness_rejects_drop_payload():
    correctness = _case("incorrect-full", "content")
    correctness.request["drop_message"] = {"2": [0]}
    correctness.request_sha256 = request_hash(correctness.request)

    with pytest.raises(ValueError, match="contains a Drop payload"):
        correctness.ensure_correctness_scope()


@pytest.mark.parametrize(
    ("method", "summary_triggered", "drop_requested"),
    [
        ("full", False, False),
        ("drop_kv", False, True),
        ("summary", True, False),
        ("summary_drop_kv", True, True),
    ],
)
def test_performance_accepts_full_summary_cross_drop_matrix(
    method, summary_triggered, drop_requested
):
    performance = _case(
        method,
        "content",
        method=method,
        summary_triggered=summary_triggered,
    )
    if drop_requested:
        performance.request["drop_message"] = {"2": [0]}
        performance.request_sha256 = request_hash(performance.request)

    performance.ensure_performance_scope()


def test_performance_rejects_unknown_summary_state_and_inconsistent_method_labels():
    unknown_summary = _case("unknown", "content", method="full")
    mislabeled_summary = _case(
        "bad-summary",
        "content",
        method="summary",
        summary_triggered=False,
    )
    missing_drop = _case(
        "missing-drop",
        "content",
        method="summary_drop_kv",
        summary_triggered=True,
    )

    with pytest.raises(ValueError, match="explicitly record"):
        unknown_summary.ensure_performance_scope()
    with pytest.raises(ValueError, match="conflicts with summary_triggered"):
        mislabeled_summary.ensure_performance_scope()
    with pytest.raises(ValueError, match="conflicts with drop_requested"):
        missing_drop.ensure_performance_scope()


def test_coverage_reports_missing_cells_instead_of_claiming_full_matrix():
    cases = [
        _case("full", "content"),
        _case("summary", "reasoning+content", method="summary", summary_triggered=True),
    ]

    report = coverage_matrix(cases, models=["model-a"])

    assert report["expected_cells"] == 14
    assert report["covered_cells"] == 2
    assert len(report["missing_cells"]) == 12


def test_manifest_jsonl_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    cases = [_case("case-1", "reasoning+toolcall")]

    dump_jsonl(path, cases)

    assert load_manifest(path) == cases


def test_load_full_trajectories_recurses_sorts_and_validates_history(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    for task_id, path in ((2, tmp_path / "z.jsonl"), (1, nested / "a.jsonl")):
        messages = [{"role": "user", "content": f"question-{task_id}"}]
        turns = [_trajectory_turn(task_id, 1, list(messages))]
        messages.extend(
            [
                {
                    "role": "assistant",
                    "reasoning_content": "reason-1",
                    "content": "answer-1",
                },
                {"role": "user", "content": "next-1"},
            ]
        )
        turns.append(_trajectory_turn(task_id, 2, list(messages)))
        messages.extend(
            [
                {
                    "role": "assistant",
                    "reasoning_content": "reason-2",
                    "content": "answer-2",
                },
                {"role": "user", "content": "next-2"},
            ]
        )
        turns.append(_trajectory_turn(task_id, 3, list(messages)))
        dump_jsonl(path, turns)

    tasks = load_full_trajectories(tmp_path, max_turns=3, min_tasks=2)

    assert [task.task_id for task in tasks] == ["nested/a", "z"]
    assert all(len(task.turns) == 3 for task in tasks)


def test_load_full_trajectories_rejects_short_drop_and_non_prefix_inputs(tmp_path):
    short = tmp_path / "short"
    short.mkdir()
    dump_jsonl(
        short / "task.jsonl",
        [_trajectory_turn(1, 1, [{"role": "user", "content": "question"}])],
    )
    with pytest.raises(ValueError, match="at least 2"):
        load_full_trajectories(short, max_turns=2, min_tasks=1)

    with_drop = tmp_path / "with-drop"
    with_drop.mkdir()
    record = _trajectory_turn(1, 1, [{"role": "user", "content": "question"}])
    record.request["drop_rule"] = {"type": "thinking_drop"}
    record.request_sha256 = request_hash(record.request)
    dump_jsonl(with_drop / "task.jsonl", [record])
    with pytest.raises(ValueError, match="contains a Drop payload"):
        load_full_trajectories(with_drop, max_turns=1, min_tasks=1)

    non_prefix = tmp_path / "non-prefix"
    non_prefix.mkdir()
    dump_jsonl(
        non_prefix / "task.jsonl",
        [
            _trajectory_turn(1, 1, [{"role": "user", "content": "question"}]),
            _trajectory_turn(1, 2, [{"role": "user", "content": "summary"}]),
        ],
    )
    with pytest.raises(ValueError, match="not a full-history extension"):
        load_full_trajectories(non_prefix, max_turns=2, min_tasks=1)
