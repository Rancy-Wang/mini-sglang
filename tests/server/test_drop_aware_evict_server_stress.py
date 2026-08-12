from __future__ import annotations

import copy
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest


DEFAULT_URL = os.environ.get("MINISGL_DEFAULT_URL")
DROP_EVICT_URL = os.environ.get("MINISGL_DROP_EVICT_URL")

pytestmark = pytest.mark.skipif(
    not DEFAULT_URL or not DROP_EVICT_URL,
    reason="requires MINISGL_DEFAULT_URL and MINISGL_DROP_EVICT_URL",
)


def _long_block(label: str, repeat: int) -> str:
    sentence = (
        f"{label}: telemetry remains ordered; preserve identifiers, timestamps, "
        "tool observations, and the final operational constraint. "
    )
    return sentence * repeat


def _payload(case_idx: int, generation: int, *, stream: bool) -> dict[str, Any]:
    repeat = int(os.environ.get("MINISGL_EVICT_STRESS_REPEAT", "180"))
    shared_observation = _long_block("shared observation", repeat)
    branch_observation = _long_block(
        f"branch {case_idx} generation {generation}", repeat // 2
    )
    tool_call_id = f"telemetry-{case_idx}-{generation}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise operations assistant. Use the supplied record and "
                "return a plain-text diagnosis without inventing measurements."
            ),
        },
        {
            "role": "user",
            "content": "Inspect this shared historical record. " + shared_observation,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "read_telemetry",
                        "arguments": json.dumps({"case": case_idx}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                {
                    "status": "stable",
                    "case": case_idx,
                    "generation": generation,
                    "observation": shared_observation,
                }
            ),
        },
        {
            "role": "assistant",
            "content": "The tool record is internally consistent and reports stable order.",
        },
        {
            "role": "user",
            "content": "Compare the branch-specific evidence. " + branch_observation,
        },
        {
            "role": "assistant",
            "content": (
                f"Branch {case_idx} generation {generation} keeps its identifiers and "
                "contains no ordering conflict."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Give the final concise diagnosis for case {case_idx}, generation "
                f"{generation}, and include both integers."
            ),
        },
    ]
    second_drop = [2, 3] if case_idx % 2 == 0 else [4]
    return {
        "model": os.environ.get("MINISGL_STRESS_MODEL", "gpt-oss-120b"),
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_telemetry",
                    "description": "Read the supplied deterministic telemetry record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"case": {"type": "integer"}},
                        "required": ["case"],
                    },
                },
            }
        ],
        "tool_choice": "none",
        "drop_message": {"4": [1], "7": second_drop},
        "reasoning_effort": "low",
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": int(os.environ.get("MINISGL_EVICT_STRESS_OUTPUT", "48")),
        "stream": stream,
    }


def _normalize_tool_calls(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _post(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.environ.get("MINISGL_EVICT_STRESS_TIMEOUT", "900"))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not payload["stream"]:
            body = json.loads(response.read().decode("utf-8"))
            choice = body["choices"][0]
            message = choice["message"]
            result = {
                "content": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or "",
                "tool_calls": _normalize_tool_calls(message.get("tool_calls")),
                "finish_reason": choice.get("finish_reason"),
            }
        else:
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls = None
            finish_reason = None
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise AssertionError(f"stream error: {chunk['error']}")
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content_parts.append(delta.get("content", ""))
                reasoning_parts.append(delta.get("reasoning_content", ""))
                if delta.get("tool_calls") is not None:
                    tool_calls = delta["tool_calls"]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
            result = {
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
                "tool_calls": _normalize_tool_calls(tool_calls),
                "finish_reason": finish_reason,
            }

    visible = result["content"] + result["reasoning_content"]
    assert visible or result["tool_calls"], result
    assert "\x00" not in visible
    assert "\ufffd" not in visible
    assert result["finish_reason"] in {"stop", "length", "tool_calls"}
    return result


def _run_both(payloads: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    assert DEFAULT_URL is not None and DROP_EVICT_URL is not None
    workers = int(os.environ.get("MINISGL_EVICT_STRESS_WORKERS", "8"))

    def run_one(url: str) -> list[dict[str, Any]]:
        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            futures = {
                executor.submit(_post, url, copy.deepcopy(payload)): idx
                for idx, payload in enumerate(payloads)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [results[idx] for idx in range(len(payloads))]

    # Give each server the same request order and concurrency topology. Mixing both
    # URLs in one executor makes the first server receive a larger initial batch,
    # which can change greedy ties even when the cache semantics are identical.
    default = run_one(DEFAULT_URL)
    drop = run_one(DROP_EVICT_URL)
    return default, drop


def _assert_diagnoses_case(result: dict[str, Any], case_idx: int, generation: int) -> None:
    content = " ".join(result["content"].lower().split())
    assert f"case {case_idx}" in content
    assert f"generation {generation}" in content
    assert result["tool_calls"] is None
    assert result["finish_reason"] == "stop"


def _answer_signature(result: dict[str, Any]) -> tuple[Any, Any, Any]:
    return result["content"], result["tool_calls"], result["finish_reason"]


def _assert_same_answers(left: list[dict], right: list[dict]) -> None:
    assert [_answer_signature(result) for result in left] == [
        _answer_signature(result) for result in right
    ]


def _assert_phase(results: list[dict], generation: int) -> None:
    for case_idx, result in enumerate(results):
        _assert_diagnoses_case(result, case_idx, generation)


def test_high_utilization_multi_request_drop_evict_preserves_answers_across_reuse_and_churn():
    case_count = int(os.environ.get("MINISGL_EVICT_STRESS_CASES", "8"))
    churn_rounds = int(os.environ.get("MINISGL_EVICT_STRESS_CHURN_ROUNDS", "3"))
    assert case_count >= 6
    assert churn_rounds >= 2

    # Before either cache is populated, both strategies must produce a complete,
    # semantically constrained answer. Separate GPUs can break greedy ties with
    # different but valid wording, so byte stability is checked per strategy below.
    probe = _payload(97, 911, stream=False)
    assert DEFAULT_URL is not None and DROP_EVICT_URL is not None
    probe_default = _post(DEFAULT_URL, copy.deepcopy(probe))
    probe_drop = _post(DROP_EVICT_URL, copy.deepcopy(probe))
    _assert_diagnoses_case(probe_default, 97, 911)
    _assert_diagnoses_case(probe_drop, 97, 911)

    cold_payloads = [
        _payload(case_idx, 0, stream=case_idx % 2 == 0)
        for case_idx in range(case_count)
    ]
    cold_default, cold_drop = _run_both(cold_payloads)
    _assert_phase(cold_default, 0)
    _assert_phase(cold_drop, 0)

    reuse_default, reuse_drop = _run_both(cold_payloads)
    _assert_same_answers(reuse_default, cold_default)
    _assert_same_answers(reuse_drop, cold_drop)

    opposite_transport = [copy.deepcopy(payload) for payload in cold_payloads]
    for payload in opposite_transport:
        payload["stream"] = not payload["stream"]
    opposite_default, opposite_drop = _run_both(opposite_transport)
    _assert_same_answers(opposite_default, cold_default)
    _assert_same_answers(opposite_drop, cold_drop)

    for generation in range(1, churn_rounds + 1):
        churn_payloads = [
            _payload(case_idx, generation, stream=(case_idx + generation) % 2 == 0)
            for case_idx in range(case_count)
        ]
        churn_default, churn_drop = _run_both(churn_payloads)
        _assert_phase(churn_default, generation)
        _assert_phase(churn_drop, generation)

    replay_default, replay_drop = _run_both(cold_payloads)
    _assert_same_answers(replay_default, cold_default)
    _assert_same_answers(replay_drop, cold_drop)
