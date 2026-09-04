from __future__ import annotations

import json

from minisgl.benchmark.reposition_trajectory import (
    DEFAULT_DROP_RULE,
    DEFAULT_REPOSITION,
    TrajectoryCase,
    apply_consistency_checks,
    build_control_request,
    build_stress_request,
    canonical_tool_calls,
    classify_issues,
    inspect_message,
    inspect_metrics,
    inspect_text,
    load_trajectory_case,
    parse_sse,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _assistant(query: str = "next query") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should search once.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": json.dumps({"query": query})},
            }
        ],
    }


def _case() -> TrajectoryCase:
    return TrajectoryCase(
        capture_id="capture-2",
        request={
            "model": "source-model",
            "messages": [
                {"role": "user", "content": "research question"},
                _assistant("first query"),
                {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            ],
            "tools": TOOLS,
            "enable_thinking": True,
        },
        expected_assistant=_assistant(),
    )


def test_load_trajectory_case_recovers_next_assistant(tmp_path) -> None:
    requests = [
        {
            "capture_id": "capture-1",
            "captured_at_ns": 1,
            "request": {"messages": [{"role": "user", "content": "research question"}]},
        },
        {
            "capture_id": "capture-2",
            "captured_at_ns": 2,
            "request": _case().request,
        },
    ]
    successor_messages = _case().request["messages"] + [
        _case().expected_assistant,
        {"role": "tool", "tool_call_id": "call-1", "content": "next result"},
    ]
    captures = requests + [
        {
            "capture_id": "capture-3",
            "captured_at_ns": 3,
            "request": {"messages": successor_messages},
        }
    ]
    trajectory = tmp_path / "task.jsonl"
    capture_input = tmp_path / "requests.jsonl"
    trajectory.write_text("\n".join(json.dumps(row) for row in requests) + "\n")
    capture_input.write_text("\n".join(json.dumps(row) for row in captures) + "\n")

    loaded = load_trajectory_case(trajectory, capture_input, turn_id=2)

    assert loaded.capture_id == "capture-2"
    assert loaded.request == _case().request
    assert canonical_tool_calls(loaded.expected_assistant) == [
        {"name": "search", "arguments": {"query": "next query"}}
    ]


def test_stress_request_has_exact_drop_and_reposition_interfaces() -> None:
    stress = build_stress_request(
        _case(), archive_a_entries=2, archive_b_entries=3, model="served-model"
    )

    assert stress["drop_rule"] == DEFAULT_DROP_RULE
    assert stress["reposition"] == DEFAULT_REPOSITION
    assert stress["messages"][1]["content"].splitlines()[0].startswith("ARCHIVE-A item=00000000")
    assert stress["messages"][3]["content"].splitlines()[-1].startswith("ARCHIVE-B item=00000002")
    assert stress["messages"][5:] == _case().request["messages"]
    assert stress["temperature"] == 0.0
    assert stress["seed"] == 17
    assert stress["model"] == "served-model"

    control = build_control_request(_case())
    assert "drop_rule" not in control
    assert "reposition" not in control


def test_parse_sse_preserves_reasoning_content_and_tool_arguments() -> None:
    events = [
        {"choices": [{"delta": {"reasoning_content": "think "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '"football"}'}}]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "server_metrics": {"tokenize_invocations": 1},
        },
        {"choices": [], "usage": {"prompt_tokens": 10}},
    ]
    raw = "\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\ndata: [DONE]\n"

    parsed = parse_sse(raw)

    assert parsed["message"]["reasoning_content"] == "think "
    assert parsed["message"]["content"] == "answer"
    assert canonical_tool_calls(parsed["message"]) == [
        {"name": "search", "arguments": {"query": "football"}}
    ]
    assert parsed["finish_reason"] == "tool_calls"
    assert parsed["usage"] == {"prompt_tokens": 10}


def test_inspection_detects_repetition_weird_characters_and_bad_tool_json() -> None:
    block = "a deliberately long pathological repeated generation block "
    issues, report = inspect_text(block * 3 + "\ufffd\x00", field="content")
    assert "content:consecutive_repetition" in issues
    assert "content:replacement" in issues
    assert "content:nul" in issues
    assert report["consecutive_repetition"] is True

    message = _assistant()
    message["tool_calls"][0]["function"]["arguments"] = "{not-json"
    issues, _ = inspect_message(message, tools=TOOLS, expected=_assistant())
    assert "tool_calls:arguments_not_json_object" in issues


def test_inspection_accepts_valid_oracle_tool_call_and_metrics() -> None:
    issues, report = inspect_message(_assistant(), tools=TOOLS, expected=_assistant())
    assert issues == []
    assert report["oracle_exact_match"] is True

    metrics = {
        "request_received_ns": 1,
        "first_token_generated_ns": 2,
        "request_finished_ns": 3,
        "prompt_tokens": 100,
        "active_prompt_tokens": 20,
        "generated_tokens": 5,
        "completion_tokens": 5,
        "tokenize_invocations": 1,
        "context_stage_count": 3,
        "reposition_transition_count": 2,
        "reposition_h2d_bytes": 40,
        "reposition_d2h_bytes": 0,
    }
    assert inspect_metrics(metrics, stress=True, cold=True) == []


def test_attribution_uses_baseline_stream_warm_and_concurrency_cells() -> None:
    records = [
        {"cell": "baseline_c1", "issues": ["content:replacement"]},
        {"cell": "stress_cold_c1", "issues": ["content:replacement"]},
        {"cell": "stress_warm_stream_c1", "issues": ["tool_calls:count_mismatch"]},
        {"cell": "stress_warm_c4", "issues": ["content:consecutive_repetition"]},
    ]

    classified = {item["issue"]: item["attribution"] for item in classify_issues(records)}

    assert classified["content:replacement"] == "model_or_benchmark"
    assert classified["tool_calls:count_mismatch"] == "streaming_parser_or_system"
    assert classified["content:consecutive_repetition"] == "scheduler_cache_or_concurrency_system"


def test_consistency_check_compares_parsed_stress_outputs_without_call_ids() -> None:
    cold = {
        "cell": "stress_cold_c1",
        "response": {"ok": True, "message": _assistant(), "finish_reason": "tool_calls"},
        "issues": [],
    }
    same = {
        "cell": "stress_warm_stream_c1",
        "response": {"ok": True, "message": _assistant(), "finish_reason": "tool_calls"},
        "issues": [],
    }
    same["response"]["message"]["tool_calls"][0]["id"] = "different-generated-id"
    changed = {
        "cell": "stress_warm_c4",
        "response": {
            "ok": True,
            "message": _assistant("different query"),
            "finish_reason": "tool_calls",
        },
        "issues": [],
    }

    apply_consistency_checks([cold, same, changed])

    assert same["issues"] == []
    assert changed["issues"] == ["consistency:stress_output_mismatch"]
