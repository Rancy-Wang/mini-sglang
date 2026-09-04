from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from minisgl.benchmark.reposition_bcp import (
    CapturedRequest,
    ReplayTask,
    apply_rolling_interface,
    attribute_issues,
    audit_parsed_response,
    build_rolling_plan,
    derive_turn_groups,
    discover_captured_requests,
    load_rollout_prompt_token_hints,
    load_trajectory_replay_tasks,
    materialize_fixed_request_cases,
    render_text_trajectory,
    select_replay_tasks,
    select_task_set,
)


def _messages(turns: int, *, tools_per_turn: int = 2) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "protected"},
        {"role": "user", "content": "task"},
    ]
    for turn in range(1, turns + 1):
        call_ids = [f"call-{turn}-{index}" for index in range(tools_per_turn)]
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                    for call_id in call_ids
                ],
            }
        )
        messages.extend(
            {"role": "tool", "tool_call_id": call_id, "content": f"result {call_id}"}
            for call_id in call_ids
        )
    return messages


def _capture(case_id: str, physical: int, retry: int = 0) -> CapturedRequest:
    return CapturedRequest(
        case_id=case_id,
        physical_id=physical,
        logical_id=physical,
        retry_id=retry,
        path=Path(f"/{case_id}/{physical}/{retry}"),
        request={"messages": _messages(physical)},
    )


def test_turn_groups_keep_parallel_tool_responses_atomic() -> None:
    protected, turns = derive_turn_groups(_messages(2))

    assert protected == (0, 1)
    assert [turn.message_ids for turn in turns] == [(2, 3, 4), (5, 6, 7)]
    assert [turn.trigger_message_id for turn in turns] == [4, 7]


def test_sixth_and_later_turns_build_cumulative_drop_and_reposition() -> None:
    plan6 = build_rolling_plan(_messages(5))
    assert plan6.incoming_turn == 6
    assert plan6.drop_messages == {"16": [2, 3, 4]}
    assert plan6.reposition == (16,)

    plan7 = build_rolling_plan(_messages(6))
    assert plan7.drop_messages == {"16": [2, 3, 4], "19": [5, 6, 7]}
    assert plan7.reposition == (16, 19)
    dropped = {message_id for ids in plan7.drop_messages.values() for message_id in ids}
    assert not dropped.intersection(plan7.protected_prefix_ids)


def test_interface_modes_remove_old_controls_and_never_use_marker_ids() -> None:
    original = {
        "messages": _messages(5),
        "drop_message": {"old": "value"},
        "reposition": [1],
    }
    full, _ = apply_rolling_interface(original, mode="full")
    rolling, _ = apply_rolling_interface(original, mode="rolling")
    drop_only, _ = apply_rolling_interface(original, mode="rolling-drop-only")

    assert "drop_message" not in full and "drop_rule" not in full and "reposition" not in full
    assert rolling["drop_rule"]["drop_messages"] == {"16": [2, 3, 4]}
    assert rolling["reposition"] == [16]
    assert "reposition" not in drop_only
    assert "marker_id" not in json.dumps(rolling)
    assert original["reposition"] == [1]


def test_tool_response_must_belong_to_immediately_preceding_assistant() -> None:
    messages = _messages(1)
    messages[3]["tool_call_id"] = "not-declared"
    with pytest.raises(ValueError, match="not declared"):
        derive_turn_groups(messages)


def test_discovery_reads_gzip_and_selection_deduplicates_retries(tmp_path: Path) -> None:
    root = tmp_path / "shard0"
    for physical in range(6):
        retries = (0, 1) if physical == 2 else (0,)
        for retry in retries:
            path = (
                root
                / "episodes"
                / "case_231"
                / "trial_000"
                / "attempts"
                / "attempt_001"
                / "backend_calls"
                / f"{physical:06d}_agent_{physical:03d}_retry_{retry:02d}"
                / "request.json.gz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump({"request": {"messages": _messages(physical)}}, stream)

    captures = discover_captured_requests(root)
    tasks = select_replay_tasks(captures, limit=1, preferred_case_ids=["231"])

    assert len(captures) == 7
    assert [item.physical_id for item in tasks[0].requests] == list(range(6))
    assert tasks[0].requests[2].retry_id == 1


def test_audit_and_attribution_separate_model_symptoms_from_system_evidence() -> None:
    request = {
        "messages": _messages(1),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    parsed = {
        "message": {
            "role": "assistant",
            "content": "answer�",
            "tool_calls": [
                {"type": "function", "function": {"name": "lookup", "arguments": "bad"}}
            ],
        },
        "finish_reason": "tool_calls",
    }
    audit = audit_parsed_response(parsed, request=request)

    assert "content:replacement" in audit["issues"]
    assert "tool_calls:arguments_not_json_object" in audit["issues"]
    attributed = attribute_issues(audit["issues"], ["content:replacement"])
    assert {item["issue"]: item["attribution"] for item in attributed} == {
        "content:replacement": "model_or_benchmark",
        "tool_calls:arguments_not_json_object": "reposition_system_suspect",
    }


def test_text_trajectory_preserves_full_reasoning_content_and_tool_calls() -> None:
    records = [
        {
            "case_id": "231",
            "mode": "rolling",
            "turn": 6,
            "response": {
                "message": {
                    "reasoning_content": "think completely",
                    "content": "answer completely",
                    "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
                }
            },
            "audit": {"issues": []},
        }
    ]

    rendered = render_text_trajectory(records)

    assert "think completely" in rendered
    assert "answer completely" in rendered
    assert '"name": "lookup"' in rendered
    assert "[issues] none" in rendered


def test_readable_trajectory_fallback_reconstructs_request_prefixes_and_length_hints(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectories.jsonl"
    rollout_path = tmp_path / "rollouts.jsonl"
    trajectory_path.write_text(json.dumps({"case_id": "231", "trajectory": _messages(6)}) + "\n")
    rollout_path.write_text(
        json.dumps(
            {
                "case_id": "231",
                "metadata": {
                    "model_calls": [
                        {"usage": {"prompt_tokens": 64_123}},
                        {"usage": {"prompt_tokens": 100_456}},
                    ]
                },
            }
        )
        + "\n"
    )

    hints = load_rollout_prompt_token_hints([rollout_path])
    tasks = load_trajectory_replay_tasks([trajectory_path], prompt_token_hints=hints)

    assert hints == {"231": 100_456}
    assert len(tasks[0].requests) == 6
    assert tasks[0].requests[0].request["messages"] == _messages(6)[:2]
    assert tasks[0].requests[-1].provenance == "trajectory_reconstructed"
    assert tasks[0].prompt_tokens_hint == 100_456


def test_task_selection_adds_32k_64k_and_100k_coverage() -> None:
    tasks = [
        ReplayTask(
            str(index), tuple(_capture(str(index), turn) for turn in range(6)), prompt_tokens
        )
        for index, prompt_tokens in enumerate((10_000, 35_000, 70_000, 110_000), 1)
    ]

    selected = select_task_set(tasks, limit=4, preferred_case_ids=["1"])

    assert {task.prompt_tokens_hint for task in selected} == {10_000, 35_000, 70_000, 110_000}


def test_fixed_cases_cover_six_shapes_stream_modes_and_three_replays(tmp_path: Path) -> None:
    task = ReplayTask(
        "231",
        tuple(_capture("231", turn) for turn in range(9)),
        prompt_tokens_hint=110_000,
    )
    paths = materialize_fixed_request_cases([task], tmp_path / "fixed", max_tokens=32)

    assert len(paths) == 12
    assert all(len(path.read_text().splitlines()) == 3 for path in paths)
    overlap = next(
        path for path in paths if path.name == "overlapping_newly_effective__stream.jsonl"
    )
    overlap_request = json.loads(overlap.read_text().splitlines()[0])
    events = list(overlap_request["drop_rule"]["drop_messages"].values())
    assert set(events[0]).issubset(events[1])
    deferred = next(
        path for path in paths if path.name == "drops_before_later_reposition__nonstream.jsonl"
    )
    deferred_request = json.loads(deferred.read_text().splitlines()[0])
    assert len(deferred_request["drop_rule"]["drop_messages"]) > len(deferred_request["reposition"])
