from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

DEFAULT_DROP_RULE = {
    "type": "message_drop",
    "drop_messages": {"2": [1], "4": [3]},
}
DEFAULT_REPOSITION = [2, 4]


@dataclass(frozen=True)
class TrajectoryCase:
    capture_id: str
    request: dict[str, Any]
    expected_assistant: dict[str, Any]


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _request_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    request = record.get("request")
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError("Capture record has no valid request.messages")
    return messages


def load_trajectory_case(
    trajectory_file: Path,
    capture_input: Path,
    *,
    turn_id: int,
) -> TrajectoryCase:
    """Load one captured request and recover its next assistant turn as an oracle."""

    trajectory = _json_lines(trajectory_file)
    if turn_id < 1 or turn_id > len(trajectory):
        raise ValueError(f"turn_id must be in [1, {len(trajectory)}]")
    selected = trajectory[turn_id - 1]
    selected_request = selected.get("request")
    if not isinstance(selected_request, dict):
        raise ValueError("Selected trajectory row has no request object")
    selected_messages = _request_messages(selected)

    successors: list[tuple[int, int, dict[str, Any]]] = []
    for candidate in _json_lines(capture_input):
        candidate_messages = _request_messages(candidate)
        if len(candidate_messages) <= len(selected_messages):
            continue
        if candidate_messages[: len(selected_messages)] != selected_messages:
            continue
        timestamp = int(candidate.get("captured_at_ns", 0))
        successors.append((len(candidate_messages), timestamp, candidate))
    if not successors:
        raise ValueError("No later captured request extends the selected trajectory")

    successor = min(successors, key=lambda item: (item[0], item[1]))[2]
    added_messages = _request_messages(successor)[len(selected_messages) :]
    expected = next(
        (message for message in added_messages if message.get("role") == "assistant"),
        None,
    )
    if expected is None:
        raise ValueError("The next captured request contains no added assistant message")
    return TrajectoryCase(
        capture_id=str(selected.get("capture_id", "")),
        request=copy.deepcopy(selected_request),
        expected_assistant=copy.deepcopy(expected),
    )


def make_archive(cycle: str, entries: int) -> str:
    if entries < 1:
        raise ValueError("archive entries must be positive")
    lines = []
    for index in range(entries):
        check = (index * 2654435761 + (17 if cycle == "A" else 29)) & 0xFFFFFFFF
        lines.append(
            f"ARCHIVE-{cycle} item={index:08d} check={check:08x} "
            "synthetic historical evidence; disposable after its acknowledgement."
        )
    return "\n".join(lines)


def _common_sampling(request: dict[str, Any], *, max_tokens: int, model: str | None) -> None:
    request.update(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 17,
            "max_tokens": max_tokens,
        }
    )
    if model is not None:
        request["model"] = model


def build_control_request(
    case: TrajectoryCase,
    *,
    max_tokens: int = 2048,
    model: str | None = None,
) -> dict[str, Any]:
    request = copy.deepcopy(case.request)
    request.pop("drop_message", None)
    request.pop("drop_rule", None)
    request.pop("reposition", None)
    request["stream"] = False
    _common_sampling(request, max_tokens=max_tokens, model=model)
    return request


def build_stress_request(
    case: TrajectoryCase,
    *,
    archive_a_entries: int,
    archive_b_entries: int,
    max_tokens: int = 2048,
    model: str | None = None,
) -> dict[str, Any]:
    request = copy.deepcopy(case.request)
    original = request.get("messages")
    if not isinstance(original, list):
        raise ValueError("request.messages must be a list")
    prefix = [
        {
            "role": "system",
            "content": (
                "ARCHIVE-A and ARCHIVE-B below are synthetic disposable context. "
                "Acknowledge each archive without quoting it. After both archives, follow the "
                "recorded research trajectory and call the provided tool as needed."
            ),
        },
        {"role": "user", "content": make_archive("A", archive_a_entries)},
        {"role": "assistant", "content": "ARCHIVE-A acknowledged."},
        {"role": "user", "content": make_archive("B", archive_b_entries)},
        {"role": "assistant", "content": "ARCHIVE-B acknowledged."},
    ]
    request["messages"] = prefix + copy.deepcopy(original)
    request["drop_rule"] = copy.deepcopy(DEFAULT_DROP_RULE)
    request["reposition"] = list(DEFAULT_REPOSITION)
    request["stream"] = False
    _common_sampling(request, max_tokens=max_tokens, model=model)
    return request


def _merge_tool_call_delta(tool_calls: list[dict[str, Any]], delta: dict[str, Any]) -> None:
    index = int(delta.get("index", len(tool_calls)))
    while len(tool_calls) <= index:
        tool_calls.append({"type": "function", "function": {"name": "", "arguments": ""}})
    current = tool_calls[index]
    for key in ("id", "type"):
        if delta.get(key) is not None:
            current[key] = delta[key]
    function = delta.get("function")
    if isinstance(function, dict):
        target = current.setdefault("function", {})
        for key in ("name", "arguments"):
            if function.get(key) is not None:
                target[key] = str(target.get(key, "")) + str(function[key])


def parse_sse(raw_text: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    finish_reason: str | None = None
    server_metrics: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for line in raw_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("SSE data is not a JSON object")
        events.append(event)
        if isinstance(event.get("server_metrics"), dict):
            server_metrics = event["server_metrics"]
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning.append(delta["reasoning_content"])
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    _merge_tool_call_delta(tool_calls, call)
    message["content"] = "".join(content)
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "message": message,
        "finish_reason": finish_reason,
        "server_metrics": server_metrics,
        "usage": usage,
        "events": events,
    }


def parse_nonstream(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Response has no choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Response has no choices[0].message")
    return {
        "message": message,
        "finish_reason": choice.get("finish_reason"),
        "server_metrics": payload.get("server_metrics"),
        "usage": payload.get("usage"),
    }


def _suspicious_characters(text: str) -> dict[str, int]:
    counts = {
        "replacement": text.count("\ufffd"),
        "nul": text.count("\x00"),
        "invalid_control": 0,
        "surrogate": 0,
        "private_use": 0,
        "bidi_or_zero_width": 0,
    }
    suspicious_format = {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
    for char in text:
        category = unicodedata.category(char)
        if category == "Cc" and char not in "\n\r\t":
            counts["invalid_control"] += 1
        elif category == "Cs":
            counts["surrogate"] += 1
        elif category == "Co":
            counts["private_use"] += 1
        if char in suspicious_format:
            counts["bidi_or_zero_width"] += 1
    return counts


def _has_consecutive_repetition(text: str) -> bool:
    normalized = " ".join(text.split())
    words = normalized.split()
    for width in range(4, min(64, len(words) // 3) + 1):
        for start in range(len(words) - width * 3 + 1):
            block = words[start : start + width]
            if (
                block
                == words[start + width : start + width * 2]
                == words[start + width * 2 : start + width * 3]
            ):
                return True
    for width in (48, 96, 192, 384):
        if len(normalized) < width * 3:
            continue
        for start in range(0, len(normalized) - width * 3 + 1, max(1, width // 4)):
            block = normalized[start : start + width]
            if (
                block
                == normalized[start + width : start + width * 2]
                == normalized[start + width * 2 : start + width * 3]
            ):
                return True
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 40]
    return any(
        lines[index] == lines[index + 1] == lines[index + 2] for index in range(len(lines) - 2)
    )


def inspect_text(text: Any, *, field: str) -> tuple[list[str], dict[str, Any]]:
    if text is None:
        text = ""
    if not isinstance(text, str):
        return [f"{field}:not_string"], {"length": 0, "character_counts": {}}
    counts = _suspicious_characters(text)
    issues = [f"{field}:{name}" for name, count in counts.items() if count]
    repeated = _has_consecutive_repetition(text)
    if repeated:
        issues.append(f"{field}:consecutive_repetition")
    return issues, {
        "length": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "character_counts": counts,
        "consecutive_repetition": repeated,
    }


def _schema_type_matches(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    issues: list[str] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and not _schema_type_matches(value, schema_type):
        return [f"{path}:expected_{schema_type}"]
    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    issues.append(f"{path}.{key}:required")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child in properties.items():
                if key in value and isinstance(child, dict):
                    issues.extend(_validate_schema(value[key], child, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            issues.extend(_validate_schema(item, schema["items"], f"{path}[{index}]"))
    return issues


def _tool_schemas(tools: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(tools, list):
        return result
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            parameters = function.get("parameters")
            result[name] = parameters if isinstance(parameters, dict) else {}
    return result


def canonical_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return canonical
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            canonical.append({"name": None, "arguments": None})
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        canonical.append({"name": function.get("name"), "arguments": arguments})
    return canonical


def inspect_message(
    message: dict[str, Any],
    *,
    tools: Any,
    expected: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    text_report: dict[str, Any] = {}
    for field in ("reasoning_content", "content"):
        field_issues, report = inspect_text(message.get(field), field=field)
        issues.extend(field_issues)
        text_report[field] = report

    schemas = _tool_schemas(tools)
    actual_calls = canonical_tool_calls(message)
    expected_calls = canonical_tool_calls(expected)
    if len(actual_calls) != len(expected_calls):
        issues.append("tool_calls:count_mismatch")
    signatures: list[str] = []
    tool_report: list[dict[str, Any]] = []
    for call in actual_calls:
        name = call.get("name")
        arguments = call.get("arguments")
        call_issues: list[str] = []
        if not isinstance(name, str) or name not in schemas:
            call_issues.append("unknown_tool")
        if not isinstance(arguments, dict):
            call_issues.append("arguments_not_json_object")
        elif isinstance(name, str) and name in schemas:
            call_issues.extend(_validate_schema(arguments, schemas[name]))
        signature = json.dumps(call, sort_keys=True, ensure_ascii=False)
        signatures.append(signature)
        tool_report.append({"call": call, "issues": call_issues})
        issues.extend(f"tool_calls:{issue}" for issue in call_issues)
    if len(signatures) != len(set(signatures)):
        issues.append("tool_calls:duplicate")
    if expected_calls and [call.get("name") for call in actual_calls] != [
        call.get("name") for call in expected_calls
    ]:
        issues.append("tool_calls:oracle_name_mismatch")
    return sorted(set(issues)), {
        "text": text_report,
        "tool_calls": tool_report,
        "canonical_tool_calls": actual_calls,
        "oracle_tool_calls": expected_calls,
        "oracle_exact_match": actual_calls == expected_calls,
    }


METRIC_FIELDS = {
    "request_received_ns",
    "first_token_generated_ns",
    "request_finished_ns",
    "prompt_tokens",
    "active_prompt_tokens",
    "generated_tokens",
    "completion_tokens",
    "tokenize_invocations",
    "context_stage_count",
    "reposition_transition_count",
    "reposition_h2d_bytes",
    "reposition_d2h_bytes",
}


def inspect_metrics(metrics: Any, *, stress: bool, cold: bool) -> list[str]:
    if not isinstance(metrics, dict):
        return ["system:missing_server_metrics"]
    issues = [f"system:missing_metric:{key}" for key in sorted(METRIC_FIELDS - metrics.keys())]
    if issues:
        return issues
    if not (
        0
        <= int(metrics["request_received_ns"])
        <= int(metrics["first_token_generated_ns"])
        <= int(metrics["request_finished_ns"])
    ):
        issues.append("system:nonmonotonic_metrics")
    if int(metrics["tokenize_invocations"]) != 1:
        issues.append("system:tokenize_invocations_not_one")
    if int(metrics["reposition_d2h_bytes"]) != 0:
        issues.append("system:reposition_d2h_nonzero")
    if stress and int(metrics["context_stage_count"]) < 2:
        issues.append("system:insufficient_context_stages")
    if stress and cold and int(metrics["reposition_transition_count"]) < 1:
        issues.append("system:cold_retry_transition_missing")
    if stress and cold and int(metrics["reposition_h2d_bytes"]) < 1:
        issues.append("system:cold_retry_h2d_missing")
    return sorted(set(issues))


def classify_issues(records: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    by_cell: dict[str, set[str]] = {}
    for record in records:
        by_cell.setdefault(str(record.get("cell")), set()).update(record.get("issues", []))
    baseline = by_cell.get("baseline_c1", set())
    classifications: list[dict[str, str]] = []
    all_issues = sorted(set().union(*by_cell.values()) if by_cell else set())
    for issue in all_issues:
        cells = {cell for cell, issues in by_cell.items() if issue in issues}
        if issue.startswith("system:") or issue.startswith("transport:"):
            cause = "system"
        elif issue in baseline:
            cause = "model_or_benchmark"
        elif cells and cells <= {"stress_warm_stream_c1"}:
            cause = "streaming_parser_or_system"
        elif cells and cells <= {"stress_warm_c4"}:
            cause = "scheduler_cache_or_concurrency_system"
        elif "stress_cold_c1" not in cells and cells:
            cause = "warm_radix_retry_cache_or_system"
        else:
            cause = "indeterminate"
        classifications.append(
            {"issue": issue, "cells": ",".join(sorted(cells)), "attribution": cause}
        )
    return classifications


def apply_consistency_checks(records: Sequence[dict[str, Any]]) -> None:
    """Annotate deterministic stress outputs that diverge from the cold reference."""

    reference = next(
        (
            record
            for record in records
            if record.get("cell") == "stress_cold_c1" and record.get("response", {}).get("ok")
        ),
        None,
    )
    if reference is None:
        return

    def comparable(record: dict[str, Any]) -> dict[str, Any] | None:
        message = record.get("response", {}).get("message")
        if not isinstance(message, dict):
            return None
        return {
            "reasoning_content": message.get("reasoning_content") or "",
            "content": message.get("content") or "",
            "tool_calls": canonical_tool_calls(message),
            "finish_reason": record.get("response", {}).get("finish_reason"),
        }

    expected = comparable(reference)
    for record in records:
        if record is reference or not str(record.get("cell", "")).startswith("stress_"):
            continue
        if record.get("response", {}).get("ok") and comparable(record) != expected:
            record.setdefault("issues", []).append("consistency:stress_output_mismatch")
            record["issues"] = sorted(set(record["issues"]))


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


async def post_request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    request: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    raw_bytes = b""
    status_code: int | None = None
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        if request.get("stream") is True:
            async with client.stream(
                "POST", _chat_url(base_url), json=request, headers=headers
            ) as response:
                status_code = response.status_code
                raw_bytes = b"".join([chunk async for chunk in response.aiter_bytes()])
        else:
            response = await client.post(_chat_url(base_url), json=request, headers=headers)
            status_code = response.status_code
            raw_bytes = response.content
        finished_ns = time.time_ns()
        raw_text = raw_bytes.decode("utf-8", errors="strict")
        if status_code != 200:
            raise RuntimeError(f"HTTP {status_code}: {raw_text[:1000]}")
        if request.get("stream") is True:
            parsed = parse_sse(raw_text)
        else:
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise ValueError("Non-stream response is not a JSON object")
            parsed = parse_nonstream(payload)
        return {
            "ok": True,
            "status_code": status_code,
            "client_started_ns": started_ns,
            "client_finished_ns": finished_ns,
            "raw_response_text": raw_text,
            **parsed,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": status_code,
            "client_started_ns": started_ns,
            "client_finished_ns": time.time_ns(),
            "raw_response_hex": raw_bytes.hex(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "issues": ["transport:request_or_parse_failure"],
        }


def _request_digest(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def run_cell(
    client: httpx.AsyncClient,
    *,
    cell: str,
    concurrency: int,
    stream: bool,
    request: dict[str, Any],
    base_url: str,
    api_key: str,
    expected: dict[str, Any],
    cold: bool,
) -> list[dict[str, Any]]:
    bodies = []
    for _ in range(concurrency):
        body = copy.deepcopy(request)
        body["stream"] = stream
        if stream:
            body["stream_options"] = {"include_usage": True}
        else:
            body.pop("stream_options", None)
        bodies.append(body)
    responses = await asyncio.gather(
        *(post_request(client, base_url=base_url, request=body, api_key=api_key) for body in bodies)
    )
    records: list[dict[str, Any]] = []
    for index, (body, response) in enumerate(zip(bodies, responses)):
        issues = list(response.pop("issues", []))
        inspection: dict[str, Any] = {}
        if response.get("ok"):
            message = response.get("message")
            if isinstance(message, dict):
                message_issues, inspection = inspect_message(
                    message,
                    tools=body.get("tools"),
                    expected=expected,
                )
                issues.extend(message_issues)
            else:
                issues.append("transport:missing_parsed_message")
            issues.extend(
                inspect_metrics(
                    response.get("server_metrics"),
                    stress=cell != "baseline_c1",
                    cold=cold,
                )
            )
        records.append(
            {
                "cell": cell,
                "sample_index": index,
                "concurrency": concurrency,
                "stream": stream,
                "cold": cold,
                "request_sha256": _request_digest(body),
                "request": body,
                "response": response,
                "inspection": inspection,
                "issues": sorted(set(issues)),
            }
        )
    return records


def _tokenize_request(manager: Any, request: dict[str, Any]) -> Any:
    from minisgl.core import SamplingParams
    from minisgl.message import TokenizeMsg

    sampling = SamplingParams(
        temperature=float(request.get("temperature", 0.0)),
        top_k=int(request.get("top_k", -1)),
        top_p=float(request.get("top_p", 1.0)),
        max_tokens=int(request.get("max_tokens", 2048)),
        seed=int(request.get("seed", 17)),
    )
    return manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=request["messages"],
                sampling_params=sampling,
                target_msg_id=len(request["messages"]),
                drop_rule=request.get("drop_rule"),
                reposition=request.get("reposition"),
                enable_thinking=request.get("enable_thinking"),
                reasoning_effort=request.get("reasoning_effort"),
                tools=request.get("tools"),
                tool_choice=request.get("tool_choice"),
                use_context_mask=True,
            )
        ]
    )[0]


def _raw_cycles(result: Any) -> tuple[int, int]:
    boundaries = result.reposition_raw_boundaries
    values = boundaries.tolist() if boundaries is not None else []
    if len(values) != 2:
        raise ValueError(f"Expected two effective Reposition boundaries, got {values}")
    first = int(values[0]) + 1
    second = int(values[1]) - int(values[0])
    return first, second


def _calibrate_one(
    manager: Any,
    case: TrajectoryCase,
    *,
    target: int,
    cycle: int,
    fixed_a: int,
    fixed_b: int,
    max_tokens: int,
    model: str | None,
) -> int:
    low, high = 1, 4096

    def measure(entries: int) -> int:
        request = build_stress_request(
            case,
            archive_a_entries=entries if cycle == 0 else fixed_a,
            archive_b_entries=entries if cycle == 1 else fixed_b,
            max_tokens=max_tokens,
            model=model,
        )
        return _raw_cycles(_tokenize_request(manager, request))[cycle]

    while measure(high) < target:
        high *= 2
        if high > 131072:
            raise ValueError("Archive calibration exceeded 131072 entries")
    closest = high
    closest_distance = abs(measure(high) - target)
    while low <= high:
        middle = (low + high) // 2
        value = measure(middle)
        distance = abs(value - target)
        if distance < closest_distance:
            closest, closest_distance = middle, distance
        if value < target:
            low = middle + 1
        elif value > target:
            high = middle - 1
        else:
            return middle
    return closest


def calibrate_request(
    case: TrajectoryCase,
    *,
    model_path: str,
    model: str | None,
    max_context: int,
    target_ratio: float,
    tolerance: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from minisgl.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    manager = TokenizeManager(tokenizer, radix_drop_key_mode="delta-marker")
    target = round(max_context * target_ratio)
    archive_a = _calibrate_one(
        manager,
        case,
        target=target,
        cycle=0,
        fixed_a=1,
        fixed_b=1,
        max_tokens=max_tokens,
        model=model,
    )
    archive_b = _calibrate_one(
        manager,
        case,
        target=target,
        cycle=1,
        fixed_a=archive_a,
        fixed_b=1,
        max_tokens=max_tokens,
        model=model,
    )
    request = build_stress_request(
        case,
        archive_a_entries=archive_a,
        archive_b_entries=archive_b,
        max_tokens=max_tokens,
        model=model,
    )
    result = _tokenize_request(manager, request)
    cycle_a, cycle_b = _raw_cycles(result)
    if abs(cycle_a - target) > tolerance or abs(cycle_b - target) > tolerance:
        raise ValueError(f"Calibration missed target {target}±{tolerance}: {cycle_a}, {cycle_b}")
    ignored = (result.message_meta or {}).get("ignored_reposition_boundaries", [])
    if ignored:
        raise ValueError(f"Reposition boundaries were ignored: {ignored}")
    if result.drop_effective_event_count != 2:
        raise ValueError(
            f"Expected two effective Drop events, got {result.drop_effective_event_count}"
        )
    full_len = (
        len(result.full_input_ids) if result.full_input_ids is not None else result.prompt_tokens
    )
    if len(result.input_ids) + max_tokens > max_context:
        raise ValueError(
            "Final active prompt and generation budget exceed the configured context window: "
            f"{len(result.input_ids)} + {max_tokens} > {max_context}"
        )
    summary = {
        "model_path": model_path,
        "target_tokens": target,
        "tolerance": tolerance,
        "archive_a_entries": archive_a,
        "archive_b_entries": archive_b,
        "archive_a_sha256": hashlib.sha256(request["messages"][1]["content"].encode()).hexdigest(),
        "archive_b_sha256": hashlib.sha256(request["messages"][3]["content"].encode()).hexdigest(),
        "raw_cycle_tokens": [cycle_a, cycle_b],
        "raw_prompt_tokens": full_len,
        "active_prompt_tokens": len(result.input_ids),
        "effective_reposition_boundaries": result.reposition_raw_boundaries.tolist(),
        "ignored_reposition_boundaries": ignored,
        "drop_effective_event_count": result.drop_effective_event_count,
    }
    return request, summary


def _compact_request(request: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(request)
    messages = compact.get("messages", [])
    for index in (1, 3):
        if index < len(messages) and isinstance(messages[index].get("content"), str):
            content = messages[index]["content"]
            messages[index]["content"] = (
                f"<omitted archive: chars={len(content)} "
                f"sha256={hashlib.sha256(content.encode()).hexdigest()}>"
            )
    return compact


def _write_artifacts(
    output_dir: Path,
    *,
    case: TrajectoryCase,
    preflight: dict[str, Any],
    records: Sequence[dict[str, Any]],
    started_ns: int,
    finished_ns: int,
    server_log: str | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "trajectory.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")

    classifications = classify_issues(records)
    server_log_report: dict[str, Any] | None = None
    if server_log is not None:
        source = Path(server_log)
        if source.is_file():
            log_bytes = source.read_bytes()
            destination = output_dir / "server_log_window.txt"
            if source.resolve() != destination.resolve():
                destination.write_bytes(log_bytes)
            server_log_report = {
                "source": str(source),
                "artifact": str(destination),
                "bytes": len(log_bytes),
                "sha256": hashlib.sha256(log_bytes).hexdigest(),
            }
    samples = []
    for record in records:
        response = record["response"]
        samples.append(
            {
                "cell": record["cell"],
                "sample_index": record["sample_index"],
                "ok": response.get("ok"),
                "status_code": response.get("status_code"),
                "finish_reason": response.get("finish_reason"),
                "server_metrics": response.get("server_metrics"),
                "usage": response.get("usage"),
                "issues": record["issues"],
                "inspection": record["inspection"],
                "request_sha256": record["request_sha256"],
            }
        )
    result = {
        "schema_version": 1,
        "capture_id": case.capture_id,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "server_log": server_log_report,
        "preflight": preflight,
        "expected_assistant": case.expected_assistant,
        "samples": samples,
        "classifications": classifications,
        "passed": all(not sample["issues"] for sample in samples),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    transcript: list[str] = [
        "Reposition trajectory",
        "=====================",
        "",
        "Preflight",
        json.dumps(preflight, ensure_ascii=False, indent=2),
        "",
        "Expected assistant",
        json.dumps(case.expected_assistant, ensure_ascii=False, indent=2),
    ]
    for record in records:
        response = record["response"]
        transcript.extend(
            [
                "",
                f"Cell {record['cell']} sample {record['sample_index']}",
                "-" * 72,
                "Request (archives replaced by hash):",
                json.dumps(_compact_request(record["request"]), ensure_ascii=False, indent=2),
                "Parsed assistant:",
                json.dumps(response.get("message"), ensure_ascii=False, indent=2),
                "Issues:",
                json.dumps(record["issues"], ensure_ascii=False, indent=2),
                "Metrics:",
                json.dumps(response.get("server_metrics"), ensure_ascii=False, indent=2),
            ]
        )
    (output_dir / "trajectory.txt").write_text("\n".join(transcript) + "\n", encoding="utf-8")

    analysis = [
        "# Multi-Drop + Multi-Reposition analysis",
        "",
        f"Overall pass: **{result['passed']}**",
        "",
        "## Preflight",
        "",
        "```json",
        json.dumps(preflight, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Samples",
        "",
    ]
    for sample in samples:
        analysis.append(
            f"- `{sample['cell']}[{sample['sample_index']}]`: "
            f"issues={json.dumps(sample['issues'], ensure_ascii=False)}, "
            f"finish={sample['finish_reason']}"
        )
    analysis.extend(["", "## Attribution", ""])
    if classifications:
        for item in classifications:
            analysis.append(f"- `{item['issue']}` in `{item['cells']}`: **{item['attribution']}**")
    else:
        analysis.append("- No anomaly was detected.")
    (output_dir / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    case = load_trajectory_case(
        Path(args.trajectory_file),
        Path(args.capture_input),
        turn_id=args.turn_id,
    )
    stress, preflight = calibrate_request(
        case,
        model_path=args.model_path,
        model=args.model,
        max_context=args.max_context,
        target_ratio=args.target_ratio,
        tolerance=args.tolerance,
        max_tokens=args.max_tokens,
    )
    control = build_control_request(case, max_tokens=args.max_tokens, model=args.model)
    records: list[dict[str, Any]] = []
    started_ns = time.time_ns()
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        cells = (
            ("baseline_c1", 1, False, control, False),
            ("stress_cold_c1", 1, False, stress, True),
            ("stress_warm_stream_c1", 1, True, stress, False),
            ("stress_warm_c4", 4, False, stress, False),
        )
        for cell, concurrency, stream, request, cold in cells:
            records.extend(
                await run_cell(
                    client,
                    cell=cell,
                    concurrency=concurrency,
                    stream=stream,
                    request=request,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    expected=case.expected_assistant,
                    cold=cold,
                )
            )
    apply_consistency_checks(records)
    return _write_artifacts(
        Path(args.output_dir),
        case=case,
        preflight=preflight,
        records=records,
        started_ns=started_ns,
        finished_ns=time.time_ns(),
        server_log=args.server_log,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and preserve a multi-Drop + multi-Reposition text trajectory."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model")
    parser.add_argument("--capture-input", required=True)
    parser.add_argument("--trajectory-file", required=True)
    parser.add_argument("--turn-id", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-log")
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--tolerance", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
