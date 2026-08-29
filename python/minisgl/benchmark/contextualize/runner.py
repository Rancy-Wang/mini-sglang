from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import httpx

from .capture_proxy import main as capture_main
from .manifest import (
    CaptureRecord,
    CaseMetadata,
    ManifestCase,
    MatchConfig,
    OracleResult,
    TrajectoryTask,
    coverage_matrix,
    dump_jsonl,
    load_capture_records,
    load_full_trajectories,
    load_manifest,
    request_hash,
)


@dataclass(frozen=True)
class ChatResult:
    message: Dict[str, Any]
    finish_reason: str | None
    server_metrics: Dict[str, int] | None
    client_started_ns: int
    client_finished_ns: int
    usage: Dict[str, Any] | None = None


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").split("\n")).strip()


def _canonical_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip()


def _canonical_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return result
    for call in tool_calls:
        if not isinstance(call, dict):
            result.append({"invalid": call})
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            result.append({"type": call.get("type"), "function": function})
            continue
        result.append(
            {
                "type": call.get("type", "function"),
                "function": {
                    "name": function.get("name"),
                    "arguments": _canonical_arguments(function.get("arguments")),
                },
            }
        )
    return result


def compare_messages(
    reference: Dict[str, Any],
    target: Dict[str, Any],
    config: MatchConfig,
) -> List[str]:
    failures: List[str] = []
    reference_content = _normalize_text(reference.get("content"))
    target_content = _normalize_text(target.get("content"))
    reference_reasoning = _normalize_text(reference.get("reasoning_content"))
    target_reasoning = _normalize_text(target.get("reasoning_content"))

    if config.mode == "exact":
        if reference_content != target_content:
            failures.append("content_exact_mismatch")
        if config.compare_reasoning and reference_reasoning != target_reasoning:
            failures.append("reasoning_exact_mismatch")
    elif config.mode == "prefix":
        assert config.prefix_chars is not None
        if not target_content.startswith(reference_content[: config.prefix_chars]):
            failures.append("content_prefix_mismatch")
        if config.compare_reasoning and not target_reasoning.startswith(
            reference_reasoning[: config.prefix_chars]
        ):
            failures.append("reasoning_prefix_mismatch")
    else:
        searchable = "\n".join((target_reasoning, target_content))
        missing = [keyword for keyword in config.keywords if keyword not in searchable]
        if missing:
            failures.append("missing_keywords:" + json.dumps(missing, ensure_ascii=False))

    reference_tools = _canonical_tool_calls(reference)
    target_tools = _canonical_tool_calls(target)
    if config.compare_tool_calls and reference_tools != target_tools:
        failures.append("tool_calls_structural_mismatch")
    return failures


def _merge_tool_call_delta(tool_calls: List[Dict[str, Any]], delta: Dict[str, Any]) -> None:
    index = int(delta.get("index", len(tool_calls)))
    while len(tool_calls) <= index:
        tool_calls.append({"type": "function", "function": {"name": "", "arguments": ""}})
    current = tool_calls[index]
    if delta.get("type") is not None:
        current["type"] = delta["type"]
    if delta.get("id") is not None:
        current["id"] = delta["id"]
    function = delta.get("function")
    if isinstance(function, dict):
        target_function = current.setdefault("function", {})
        if function.get("name") is not None:
            target_function["name"] = str(target_function.get("name", "")) + str(function["name"])
        if function.get("arguments") is not None:
            target_function["arguments"] = str(target_function.get("arguments", "")) + str(
                function["arguments"]
            )


def _parse_nonstream(payload: Dict[str, Any], started_ns: int, finished_ns: int) -> ChatResult:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Chat response has no choices[0].")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Chat response has no choices[0].message object.")
    metrics = payload.get("server_metrics")
    usage = payload.get("usage")
    return ChatResult(
        message=message,
        finish_reason=choice.get("finish_reason"),
        server_metrics=metrics if isinstance(metrics, dict) else None,
        client_started_ns=started_ns,
        client_finished_ns=finished_ns,
        usage=usage if isinstance(usage, dict) else None,
    )


async def _post_streaming(
    client: httpx.AsyncClient,
    url: str,
    request_body: Dict[str, Any],
    headers: Dict[str, str],
    started_ns: int,
) -> ChatResult:
    message: Dict[str, Any] = {"role": "assistant"}
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    finish_reason: str | None = None
    server_metrics: Dict[str, int] | None = None
    usage: Dict[str, Any] | None = None
    async with client.stream("POST", url, json=request_body, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("server_metrics"), dict):
                server_metrics = event["server_metrics"]
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_parts.append(delta["reasoning_content"])
            if isinstance(delta.get("tool_calls"), list):
                for tool_delta in delta["tool_calls"]:
                    if isinstance(tool_delta, dict):
                        _merge_tool_call_delta(tool_calls, tool_delta)

    if content_parts:
        message["content"] = "".join(content_parts)
    else:
        message["content"] = ""
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return ChatResult(
        message=message,
        finish_reason=finish_reason,
        server_metrics=server_metrics,
        client_started_ns=started_ns,
        client_finished_ns=time.perf_counter_ns(),
        usage=usage,
    )


async def post_chat(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    request_body: Dict[str, Any],
    api_key: str,
    model_override: str | None = None,
    force_non_stream: bool = False,
) -> ChatResult:
    body = copy.deepcopy(request_body)
    if model_override is not None:
        body["model"] = model_override
    if force_non_stream:
        body["stream"] = False
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started_ns = time.perf_counter_ns()
    if body.get("stream") is True:
        return await _post_streaming(client, _chat_url(base_url), body, headers, started_ns)
    response = await client.post(_chat_url(base_url), json=body, headers=headers)
    finished_ns = time.perf_counter_ns()
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Chat response JSON must be an object.")
    return _parse_nonstream(payload, started_ns, finished_ns)


def prepare_manifest(
    captures: Sequence[CaptureRecord],
    *,
    source: str,
    method: str,
    model: str | None,
    summary_triggered: bool | None,
) -> List[ManifestCase]:
    cases: List[ManifestCase] = []
    for index, capture in enumerate(captures):
        request_model = model or capture.request.get("model")
        if not isinstance(request_model, str) or not request_model:
            raise ValueError(f"Capture {capture.capture_id!r} has no model; pass --model.")
        cases.append(
            ManifestCase(
                case_id=f"{source}-{index:06d}-{capture.capture_id[:8]}",
                request=capture.request,
                request_sha256=capture.request_sha256,
                metadata=CaseMetadata(
                    source=source,
                    method=method,
                    model=request_model,
                    summary_triggered=summary_triggered,
                    tags={"capture_id": capture.capture_id},
                ),
            )
        )
    return cases


async def record_oracles(
    cases: Sequence[ManifestCase],
    *,
    base_url: str,
    api_key: str,
    model_override: str | None,
    timeout_seconds: float,
) -> List[ManifestCase]:
    recorded: List[ManifestCase] = []
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for case in cases:
            case.ensure_correctness_scope()
            result = await post_chat(
                client,
                base_url=base_url,
                request_body=case.request,
                api_key=api_key,
                model_override=model_override,
                force_non_stream=True,
            )
            recorded.append(
                case.model_copy(
                    update={
                        "oracle": OracleResult(
                            message=result.message,
                            finish_reason=result.finish_reason,
                        )
                    }
                )
            )
    return recorded


async def verify_cases(
    cases: Sequence[ManifestCase],
    *,
    base_url: str,
    api_key: str,
    model_override: str | None,
    timeout_seconds: float,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for case in cases:
            case.ensure_correctness_scope()
            if case.oracle is None:
                raise ValueError(f"Case {case.case_id!r} has no oracle.")
            try:
                target = await post_chat(
                    client,
                    base_url=base_url,
                    request_body=case.request,
                    api_key=api_key,
                    model_override=model_override,
                    force_non_stream=True,
                )
                failures = compare_messages(case.oracle.message, target.message, case.matcher)
                if case.oracle.finish_reason != target.finish_reason:
                    failures.append("finish_reason_mismatch")
                results.append(
                    {
                        "case_id": case.case_id,
                        "passed": not failures,
                        "failures": failures,
                        "target": {
                            "message": target.message,
                            "finish_reason": target.finish_reason,
                        },
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "case_id": case.case_id,
                        "passed": False,
                        "failures": [f"request_error:{type(exc).__name__}:{exc}"],
                    }
                )
    passed = sum(bool(result["passed"]) for result in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "cases": results,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _derive_server_sample(metrics: Dict[str, int]) -> Dict[str, Any]:
    required = {
        "request_received_ns",
        "first_token_generated_ns",
        "request_finished_ns",
        "prompt_tokens",
        "active_prompt_tokens",
        "generated_tokens",
        "completion_tokens",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise ValueError(f"server_metrics is missing fields: {missing}")
    received = int(metrics["request_received_ns"])
    first = int(metrics["first_token_generated_ns"])
    finished = int(metrics["request_finished_ns"])
    generated = int(metrics["generated_tokens"])
    completion = int(metrics["completion_tokens"])
    prompt = int(metrics["prompt_tokens"])
    active_prompt = int(metrics["active_prompt_tokens"])
    if not 0 <= received <= first <= finished:
        raise ValueError("server_metrics timestamps are not monotonic.")
    if generated <= 0 or not 0 <= completion <= generated:
        raise ValueError("server_metrics token counts are invalid.")
    if not 0 <= active_prompt <= prompt:
        raise ValueError("server_metrics prompt token counts are invalid.")
    dropped_prompt = prompt - active_prompt
    return {
        "ttft_ms": (first - received) / 1e6,
        "e2e_ms": (finished - received) / 1e6,
        "tpot_ms": (finished - first) / (generated - 1) / 1e6 if generated > 1 else None,
        "prompt_tokens": prompt,
        "active_prompt_tokens": active_prompt,
        "dropped_prompt_tokens": dropped_prompt,
        "prompt_retention_ratio": active_prompt / prompt if prompt else None,
        "drop_effective": dropped_prompt > 0,
        "generated_tokens": generated,
        "completion_tokens": completion,
    }


async def benchmark_cases(
    cases: Sequence[ManifestCase],
    *,
    base_url: str,
    api_key: str,
    model_override: str | None,
    concurrencies: Sequence[int],
    num_requests: int | None,
    warmup_requests: int,
    timeout_seconds: float,
    force_non_stream: bool,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("Performance manifest is empty.")
    for case in cases:
        case.ensure_performance_scope()
    methods = {case.metadata.method for case in cases}
    if len(methods) != 1:
        raise ValueError(
            "One performance run must contain exactly one method so its throughput and latency "
            "statistics remain comparable."
        )
    method = next(iter(methods))
    summary_triggered = cases[0].metadata.summary_triggered
    drop_requested = cases[0].has_drop_payload()
    if any(concurrency <= 0 for concurrency in concurrencies):
        raise ValueError("Concurrency values must be positive.")
    request_count = num_requests or len(cases)
    if request_count <= 0:
        raise ValueError("num_requests must be positive.")
    workload = [cases[index % len(cases)] for index in range(request_count)]
    report_groups: List[Dict[str, Any]] = []

    limits = httpx.Limits(
        max_connections=max(concurrencies),
        max_keepalive_connections=max(concurrencies),
    )
    async with httpx.AsyncClient(timeout=timeout_seconds, limits=limits) as client:
        for index in range(warmup_requests):
            case = cases[index % len(cases)]
            await post_chat(
                client,
                base_url=base_url,
                request_body=case.request,
                api_key=api_key,
                model_override=model_override,
                force_non_stream=force_non_stream,
            )

        for concurrency in concurrencies:
            semaphore = asyncio.Semaphore(concurrency)

            async def run_one(case: ManifestCase) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await post_chat(
                            client,
                            base_url=base_url,
                            request_body=case.request,
                            api_key=api_key,
                            model_override=model_override,
                            force_non_stream=force_non_stream,
                        )
                        if result.server_metrics is None:
                            raise ValueError("Target response did not return server_metrics.")
                        sample = _derive_server_sample(result.server_metrics)
                        return {
                            "case_id": case.case_id,
                            "method": case.metadata.method,
                            "summary_triggered": case.metadata.summary_triggered,
                            "drop_requested": case.has_drop_payload(),
                            "passed": True,
                            "client_started_ns": result.client_started_ns,
                            "client_finished_ns": result.client_finished_ns,
                            "client_e2e_ms": (
                                result.client_finished_ns - result.client_started_ns
                            )
                            / 1e6,
                            **sample,
                        }
                    except Exception as exc:
                        return {
                            "case_id": case.case_id,
                            "passed": False,
                            "error": f"{type(exc).__name__}:{exc}",
                        }

            group_started_ns = time.perf_counter_ns()
            raw = await asyncio.gather(*(run_one(case) for case in workload))
            group_finished_ns = time.perf_counter_ns()
            successful = [sample for sample in raw if sample["passed"]]
            duration_seconds = (group_finished_ns - group_started_ns) / 1e9
            completion_tokens = sum(sample["completion_tokens"] for sample in successful)
            generated_tokens = sum(sample["generated_tokens"] for sample in successful)
            ttft = [sample["ttft_ms"] for sample in successful]
            e2e = [sample["e2e_ms"] for sample in successful]
            client_e2e = [sample["client_e2e_ms"] for sample in successful]
            tpot = [sample["tpot_ms"] for sample in successful if sample["tpot_ms"] is not None]
            prompt_tokens = [sample["prompt_tokens"] for sample in successful]
            active_prompt_tokens = [sample["active_prompt_tokens"] for sample in successful]
            dropped_prompt_tokens = [sample["dropped_prompt_tokens"] for sample in successful]
            retention_ratios = [
                sample["prompt_retention_ratio"]
                for sample in successful
                if sample["prompt_retention_ratio"] is not None
            ]
            report_groups.append(
                {
                    "method": method,
                    "summary_triggered": summary_triggered,
                    "drop_requested": drop_requested,
                    "concurrency": concurrency,
                    "requested": len(raw),
                    "succeeded": len(successful),
                    "failed": len(raw) - len(successful),
                    "duration_seconds": duration_seconds,
                    "request_throughput_per_second": len(successful) / duration_seconds,
                    "completion_token_throughput_per_second": completion_tokens / duration_seconds,
                    "generated_token_throughput_per_second": generated_tokens / duration_seconds,
                    "ttft_ms": _distribution(ttft),
                    "tpot_ms": _distribution(tpot),
                    "e2e_ms": _distribution(e2e),
                    "client_e2e_ms": _distribution(client_e2e),
                    "prompt_tokens": _distribution(prompt_tokens),
                    "active_prompt_tokens": _distribution(active_prompt_tokens),
                    "dropped_prompt_tokens": _distribution(dropped_prompt_tokens),
                    "prompt_retention_ratio": _distribution(retention_ratios),
                    "drop_effective_requests": sum(
                        bool(sample["drop_effective"]) for sample in successful
                    ),
                    "unique_cases": len({sample["case_id"] for sample in raw}),
                    "raw": raw,
                }
            )
    return {
        "metric_source": "server_metrics",
        "method": method,
        "summary_triggered": summary_triggered,
        "drop_requested": drop_requested,
        "warmup_requests": warmup_requests,
        "groups": report_groups,
    }


TrajectoryVariant = Literal["no_drop", "thinking_drop"]


def _normalize_usage(usage: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    def count(payload: Dict[str, Any], key: str) -> int:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"usage.{key} must be a non-negative integer.")
        return value

    prompt_tokens = count(usage, "prompt_tokens")
    completion_tokens = count(usage, "completion_tokens")
    total_tokens = count(usage, "total_tokens")
    if total_tokens != prompt_tokens + completion_tokens:
        raise ValueError("usage.total_tokens must equal prompt_tokens + completion_tokens.")

    raw_details = usage.get("prompt_tokens_details")
    details_reported = isinstance(raw_details, dict)
    if raw_details is not None and not details_reported:
        raise ValueError("usage.prompt_tokens_details must be an object when present.")
    details = raw_details if details_reported else {}
    cached_tokens = details.get("cached_tokens", 0)
    drop_skipped_tokens = details.get("drop_skipped_tokens", 0)
    for key, value in (
        ("cached_tokens", cached_tokens),
        ("drop_skipped_tokens", drop_skipped_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"usage.prompt_tokens_details.{key} must be a non-negative integer.")
    if cached_tokens + drop_skipped_tokens > prompt_tokens:
        raise ValueError("usage cached_tokens + drop_skipped_tokens must not exceed prompt_tokens.")
    return (
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens,
                "drop_skipped_tokens": drop_skipped_tokens,
            },
        },
        details_reported,
    )


def _thinking_drop_applicable(request: Dict[str, Any]) -> bool:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Trajectory request requires a messages list.")

    found = False
    for message_id, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{message_id}] must be an object.")
        if str(message.get("role", "")).lower() != "assistant":
            continue

        structured = message.get("reasoning_content")
        if structured is not None and not isinstance(structured, str):
            raise ValueError(f"messages[{message_id}].reasoning_content must be a string or null.")
        structured = structured or None

        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError(
                f"messages[{message_id}].content must be a string or null for thinking_drop."
            )
        has_tag = "<think>" in content or "</think>" in content
        inline: str | None = None
        if has_tag:
            if not content.startswith("<think>"):
                raise ValueError(
                    f"messages[{message_id}] has a non-leading or malformed <think> block."
                )
            close = content.find("</think>", len("<think>"))
            if close < 0:
                raise ValueError(f"messages[{message_id}] has an unclosed <think> block.")
            inline = content[len("<think>") : close]
            remainder = content[close + len("</think>") :]
            if any(tag in inline or tag in remainder for tag in ("<think>", "</think>")):
                raise ValueError(f"messages[{message_id}] has nested or multiple <think> blocks.")
        if structured is not None and inline is not None:
            raise ValueError(
                f"messages[{message_id}] cannot provide both reasoning_content and a leading "
                "<think> block."
            )
        found = found or bool(structured if structured is not None else inline)
    return found


def _build_trajectory_request(
    record: CaptureRecord,
    *,
    variant: TrajectoryVariant,
    turn_id: int,
    model_override: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    force_non_stream: bool,
) -> tuple[Dict[str, Any], bool]:
    request = copy.deepcopy(record.request)
    request.pop("drop_rule", None)
    request.pop("drop_message", None)
    request.pop("max_completion_tokens", None)
    if model_override is not None:
        request["model"] = model_override
    request.update(
        {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": 1,
            "ignore_eos": True,
            "stop": None,
        }
    )
    if force_non_stream:
        request["stream"] = False
    elif request.get("stream") is True:
        stream_options = request.get("stream_options")
        stream_options = copy.deepcopy(stream_options) if isinstance(stream_options, dict) else {}
        stream_options["include_usage"] = True
        request["stream_options"] = stream_options

    thinking_drop_applicable = False
    if variant == "thinking_drop" and turn_id > 1:
        thinking_drop_applicable = _thinking_drop_applicable(request)
        if not thinking_drop_applicable:
            raise ValueError(
                f"Turn {turn_id} has no historical assistant reasoning for thinking_drop."
            )
        request["drop_rule"] = {"type": "thinking_drop"}
    return request, thinking_drop_applicable


def _aggregate_trajectory_turn(turn_id: int, raw: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    responses = [sample for sample in raw if sample.get("request_succeeded") is True]
    valid = [sample for sample in responses if sample.get("fixed_length_ok") is True]

    def values(key: str) -> List[float]:
        return [float(sample[key]) for sample in valid if sample.get(key) is not None]

    def usage_values(key: str, *, detail: bool = False) -> List[float]:
        result: List[float] = []
        for sample in valid:
            usage = sample["usage"]
            payload = usage["prompt_tokens_details"] if detail else usage
            result.append(float(payload[key]))
        return result

    return {
        "turn_id": turn_id,
        "requested": len(raw),
        "succeeded": len(responses),
        "failed": len(raw) - len(responses),
        "fixed_length_failed": sum(sample.get("fixed_length_ok") is False for sample in responses),
        "sample_count": len(valid),
        "usage_details_reported": sum(
            sample.get("usage_details_reported") is True for sample in responses
        ),
        "drop_effective_requests": sum(
            sample.get("drop_effective") is True for sample in responses
        ),
        "performance": {
            "ttft_ms": _distribution(values("ttft_ms")),
            "tpot_ms": _distribution(values("tpot_ms")),
            "e2e_ms": _distribution(values("e2e_ms")),
            "client_e2e_ms": _distribution(values("client_e2e_ms")),
        },
        "usage": {
            "prompt_tokens": _distribution(usage_values("prompt_tokens")),
            "completion_tokens": _distribution(usage_values("completion_tokens")),
            "total_tokens": _distribution(usage_values("total_tokens")),
            "cached_tokens": _distribution(usage_values("cached_tokens", detail=True)),
            "drop_skipped_tokens": _distribution(usage_values("drop_skipped_tokens", detail=True)),
        },
    }


async def benchmark_trajectories(
    tasks: Sequence[TrajectoryTask],
    *,
    trajectory_dir: str | Path,
    base_url: str,
    api_key: str,
    model_override: str | None,
    concurrencies: Sequence[int],
    variants: Sequence[TrajectoryVariant],
    max_turns: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_seconds: float,
    force_non_stream: bool,
) -> Dict[str, Any]:
    if not tasks:
        raise ValueError("Trajectory workload is empty.")
    if max_turns <= 0 or max_tokens <= 0:
        raise ValueError("max_turns and max_tokens must be positive.")
    if not concurrencies or any(value <= 0 for value in concurrencies):
        raise ValueError("Concurrency values must be positive.")
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("Concurrency values must be unique.")
    if not variants or any(value not in {"no_drop", "thinking_drop"} for value in variants):
        raise ValueError("Variants must contain no_drop and/or thinking_drop.")
    if len(set(variants)) != len(variants):
        raise ValueError("Variants must be unique.")
    if len(tasks) < max(concurrencies):
        raise ValueError("Task count must be at least the maximum concurrency.")
    if any(len(task.turns) < max_turns for task in tasks):
        raise ValueError("Every trajectory must contain at least max_turns turns.")
    if temperature < 0 or not 0 < top_p <= 1 or (top_k != -1 and top_k <= 0):
        raise ValueError("Sampling requires temperature >= 0, 0 < top_p <= 1, and top_k=-1 or >0.")

    if "thinking_drop" in variants:
        for task in tasks:
            for turn_id, record in enumerate(task.turns[1:max_turns], start=2):
                try:
                    applicable = _thinking_drop_applicable(record.request)
                except ValueError as exc:
                    raise ValueError(
                        f"Trajectory {task.task_id!r} turn {turn_id} is invalid for "
                        f"thinking_drop: {exc}"
                    ) from exc
                if not applicable:
                    raise ValueError(
                        f"Trajectory {task.task_id!r} turn {turn_id} has no historical "
                        "assistant reasoning for thinking_drop."
                    )

    cells: List[Dict[str, Any]] = []
    execution_order: List[Dict[str, Any]] = []
    limits = httpx.Limits(
        max_connections=max(concurrencies),
        max_keepalive_connections=max(concurrencies),
    )
    async with httpx.AsyncClient(timeout=timeout_seconds, limits=limits) as client:
        for concurrency in concurrencies:
            for variant in variants:
                execution_order.append({"concurrency": concurrency, "variant": variant})
                semaphore = asyncio.Semaphore(concurrency)

                async def run_task(task: TrajectoryTask) -> List[Dict[str, Any]]:
                    async with semaphore:
                        task_results: List[Dict[str, Any]] = []
                        for turn_id, record in enumerate(task.turns[:max_turns], start=1):
                            request, applicable = _build_trajectory_request(
                                record,
                                variant=variant,
                                turn_id=turn_id,
                                model_override=model_override,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                force_non_stream=force_non_stream,
                            )
                            effective_hash = request_hash(request)
                            drop_requested = request.get("drop_rule") is not None
                            try:
                                result = await post_chat(
                                    client,
                                    base_url=base_url,
                                    request_body=request,
                                    api_key=api_key,
                                    force_non_stream=False,
                                )
                                if result.server_metrics is None:
                                    raise ValueError(
                                        "Target response did not return server_metrics."
                                    )
                                if result.usage is None:
                                    raise ValueError("Target response did not return usage.")
                                sample = _derive_server_sample(result.server_metrics)
                                usage, details_reported = _normalize_usage(result.usage)
                                actual_completion_tokens = usage["completion_tokens"]
                                fixed_length_ok = actual_completion_tokens == max_tokens
                                task_results.append(
                                    {
                                        "task_id": task.task_id,
                                        "turn_id": turn_id,
                                        "capture_id": record.capture_id,
                                        "source_request_sha256": record.request_sha256,
                                        "effective_request_sha256": effective_hash,
                                        "variant": variant,
                                        "concurrency": concurrency,
                                        "thinking_drop_applicable": applicable,
                                        "drop_requested": drop_requested,
                                        "request_succeeded": True,
                                        "passed": fixed_length_ok,
                                        "error": (
                                            None
                                            if fixed_length_ok
                                            else "completion_tokens_did_not_reach_max_tokens"
                                        ),
                                        "finish_reason": result.finish_reason,
                                        "client_started_ns": result.client_started_ns,
                                        "client_finished_ns": result.client_finished_ns,
                                        "client_e2e_ms": (
                                            result.client_finished_ns - result.client_started_ns
                                        )
                                        / 1e6,
                                        "server_metrics": result.server_metrics,
                                        "usage": usage,
                                        "usage_details_reported": details_reported,
                                        "expected_completion_tokens": max_tokens,
                                        "actual_completion_tokens": actual_completion_tokens,
                                        "actual_generated_tokens": sample["generated_tokens"],
                                        "fixed_length_ok": fixed_length_ok,
                                        **sample,
                                        "drop_effective": bool(
                                            drop_requested and sample["drop_effective"]
                                        ),
                                    }
                                )
                            except Exception as exc:
                                task_results.append(
                                    {
                                        "task_id": task.task_id,
                                        "turn_id": turn_id,
                                        "capture_id": record.capture_id,
                                        "source_request_sha256": record.request_sha256,
                                        "effective_request_sha256": effective_hash,
                                        "variant": variant,
                                        "concurrency": concurrency,
                                        "thinking_drop_applicable": applicable,
                                        "drop_requested": drop_requested,
                                        "request_succeeded": False,
                                        "passed": False,
                                        "error": f"{type(exc).__name__}:{exc}",
                                        "expected_completion_tokens": max_tokens,
                                        "fixed_length_ok": None,
                                    }
                                )
                        return task_results

                cell_started_ns = time.perf_counter_ns()
                task_results = await asyncio.gather(*(run_task(task) for task in tasks))
                cell_finished_ns = time.perf_counter_ns()
                raw = [sample for result in task_results for sample in result]
                responses = [sample for sample in raw if sample.get("request_succeeded") is True]
                duration_seconds = (cell_finished_ns - cell_started_ns) / 1e9
                cells.append(
                    {
                        "variant": variant,
                        "concurrency": concurrency,
                        "requested": len(raw),
                        "succeeded": len(responses),
                        "failed": len(raw) - len(responses),
                        "fixed_length_failed": sum(
                            sample.get("fixed_length_ok") is False for sample in responses
                        ),
                        "duration_seconds": duration_seconds,
                        "request_throughput_per_second": len(responses) / duration_seconds,
                        "completion_token_throughput_per_second": sum(
                            sample["usage"]["completion_tokens"] for sample in responses
                        )
                        / duration_seconds,
                        "generated_token_throughput_per_second": sum(
                            sample["generated_tokens"] for sample in responses
                        )
                        / duration_seconds,
                        "drop_effective_requests": sum(
                            sample.get("drop_effective") is True for sample in responses
                        ),
                        "turns": [
                            _aggregate_trajectory_turn(
                                turn_id,
                                [sample for sample in raw if sample["turn_id"] == turn_id],
                            )
                            for turn_id in range(1, max_turns + 1)
                        ],
                        "requests": raw,
                    }
                )

    return {
        "schema_version": "trajectory-performance-v1",
        "metric_source": "server_metrics_and_usage",
        "input": {
            "trajectory_dir": str(trajectory_dir),
            "task_count": len(tasks),
            "max_turns": max_turns,
            "sampling": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "ignore_eos": True,
                "stop": None,
            },
        },
        "execution": {
            "concurrencies": list(concurrencies),
            "variants": list(variants),
            "cache_isolation": (
                "shared_endpoint" if len(concurrencies) * len(variants) > 1 else "not_reset"
            ),
            "execution_order": execution_order,
        },
        "cells": cells,
    }


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model-override")
    parser.add_argument("--timeout", type=float, default=3600.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contextualize request-level and serving tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Run the exact-request capture proxy.")
    capture.add_argument("--upstream-base-url", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--host", default="127.0.0.1")
    capture.add_argument("--port", type=int, default=18000)
    capture.add_argument("--timeout", type=float, default=3600.0)

    prepare = subparsers.add_parser("prepare", help="Convert captures into a frozen manifest.")
    prepare.add_argument("--capture", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source", required=True)
    prepare.add_argument(
        "--method",
        choices=["full", "drop_kv", "summary", "summary_drop_kv"],
        required=True,
    )
    prepare.add_argument("--model")
    summary_group = prepare.add_mutually_exclusive_group()
    summary_group.add_argument("--summary-triggered", action="store_true")
    summary_group.add_argument("--summary-not-triggered", action="store_true")

    coverage = subparsers.add_parser("coverage", help="Report the 4 x 2 x 7 matrix coverage.")
    coverage.add_argument("--manifest", required=True)
    coverage.add_argument("--models", nargs="+", required=True)
    coverage.add_argument("--output")

    oracle = subparsers.add_parser("record-oracle", help="Record stable-SGLang reference output.")
    oracle.add_argument("--manifest", required=True)
    oracle.add_argument("--output", required=True)
    _add_connection_args(oracle)

    verify = subparsers.add_parser("verify", help="Replay and compare against frozen oracles.")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--output", required=True)
    _add_connection_args(verify)

    bench = subparsers.add_parser("bench", help="Benchmark frozen requests using server metrics.")
    bench.add_argument("--manifest", required=True)
    bench.add_argument("--output", required=True)
    bench.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    bench.add_argument("--num-requests", type=int)
    bench.add_argument("--warmup-requests", type=int, default=1)
    bench.add_argument("--preserve-stream", action="store_true")
    _add_connection_args(bench)

    trajectory_bench = subparsers.add_parser(
        "bench-trajectories",
        help="Benchmark ordered full-history task trajectories by turn.",
    )
    trajectory_bench.add_argument("--trajectory-dir", required=True)
    trajectory_bench.add_argument("--output", required=True)
    trajectory_bench.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    trajectory_bench.add_argument(
        "--variants",
        nargs="+",
        choices=["no_drop", "thinking_drop"],
        default=["no_drop", "thinking_drop"],
    )
    trajectory_bench.add_argument("--max-turns", type=int, default=10)
    trajectory_bench.add_argument("--max-tokens", type=int, default=2048)
    trajectory_bench.add_argument("--temperature", type=float, default=0.0)
    trajectory_bench.add_argument("--top-p", type=float, default=1.0)
    trajectory_bench.add_argument("--top-k", type=int, default=-1)
    trajectory_bench.add_argument("--preserve-stream", action="store_true")
    _add_connection_args(trajectory_bench)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capture":
        capture_main(
            [
                "--upstream-base-url",
                args.upstream_base_url,
                "--output",
                args.output,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--timeout",
                str(args.timeout),
            ]
        )
        return
    if args.command == "prepare":
        summary_triggered = (
            True if args.summary_triggered else False if args.summary_not_triggered else None
        )
        cases = prepare_manifest(
            load_capture_records(args.capture),
            source=args.source,
            method=args.method,
            model=args.model,
            summary_triggered=summary_triggered,
        )
        dump_jsonl(args.output, cases)
        return
    if args.command == "coverage":
        report = coverage_matrix(load_manifest(args.manifest), models=args.models)
        if args.output:
            _write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "record-oracle":
        recorded = asyncio.run(
            record_oracles(
                load_manifest(args.manifest),
                base_url=args.base_url,
                api_key=args.api_key,
                model_override=args.model_override,
                timeout_seconds=args.timeout,
            )
        )
        dump_jsonl(args.output, recorded)
        return
    if args.command == "verify":
        report = asyncio.run(
            verify_cases(
                load_manifest(args.manifest),
                base_url=args.base_url,
                api_key=args.api_key,
                model_override=args.model_override,
                timeout_seconds=args.timeout,
            )
        )
        _write_json(args.output, report)
        return
    if args.command == "bench":
        report = asyncio.run(
            benchmark_cases(
                load_manifest(args.manifest),
                base_url=args.base_url,
                api_key=args.api_key,
                model_override=args.model_override,
                concurrencies=args.concurrency,
                num_requests=args.num_requests,
                warmup_requests=args.warmup_requests,
                timeout_seconds=args.timeout,
                force_non_stream=not args.preserve_stream,
            )
        )
        _write_json(args.output, report)
        return
    if args.command == "bench-trajectories":
        tasks = load_full_trajectories(
            args.trajectory_dir,
            max_turns=args.max_turns,
            min_tasks=max(args.concurrency),
        )
        report = asyncio.run(
            benchmark_trajectories(
                tasks,
                trajectory_dir=args.trajectory_dir,
                base_url=args.base_url,
                api_key=args.api_key,
                model_override=args.model_override,
                concurrencies=args.concurrency,
                variants=args.variants,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                timeout_seconds=args.timeout,
                force_non_stream=not args.preserve_stream,
            )
        )
        _write_json(args.output, report)
        return
    raise AssertionError(f"Unhandled command: {args.command}")
