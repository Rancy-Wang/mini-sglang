from __future__ import annotations

import pytest
from pydantic import ValidationError

from minisgl.benchmark.contextualize.manifest import (
    MESSAGE_SHAPES,
    CaseMetadata,
    ManifestCase,
    classify_assistant_message,
    coverage_matrix,
    dump_jsonl,
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
