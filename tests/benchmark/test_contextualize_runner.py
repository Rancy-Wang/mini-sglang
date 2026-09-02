from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from minisgl.benchmark.contextualize.manifest import (
    CaptureRecord,
    CaseMetadata,
    ManifestCase,
    MatchConfig,
    TrajectoryTask,
    request_hash,
)
from minisgl.benchmark.contextualize.runner import (
    ChatResult,
    _build_trajectory_request,
    _derive_server_sample,
    _normalize_usage,
    _percentile,
    benchmark_cases,
    benchmark_trajectories,
    compare_messages,
    post_chat,
    prepare_manifest,
)


def _request(*, with_drop=True):
    request = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "question"}],
        "stream": False,
    }
    if with_drop:
        request["drop_message"] = {"2": [0]}
    return request


def _case(
    case_id="case-1",
    *,
    method="summary_drop_kv",
    summary_triggered=True,
    with_drop=True,
):
    request = _request(with_drop=with_drop)
    return ManifestCase(
        case_id=case_id,
        request=request,
        request_sha256=request_hash(request),
        metadata=CaseMetadata(
            source="browsecomp-plus",
            method=method,
            model="model-a",
            summary_triggered=summary_triggered,
        ),
    )


def _server_metrics(offset=0):
    return {
        "request_received_ns": 100 + offset,
        "first_token_generated_ns": 200 + offset,
        "request_finished_ns": 500 + offset,
        "prompt_tokens": 20,
        "active_prompt_tokens": 12,
        "generated_tokens": 4,
        "completion_tokens": 3,
        "tokenize_invocations": 1,
        "context_stage_count": 0,
        "radix_compile_ns": 0,
        "radix_match_ns": 0,
        "retry_plan_ns": 0,
        "reposition_transition_count": 0,
        "reposition_h2d_bytes": 0,
        "reposition_d2h_bytes": 0,
    }


def _trajectory_task(task_id, turns=3):
    messages = [{"role": "user", "content": f"question-{task_id}"}]
    records = []
    for turn_id in range(1, turns + 1):
        request = {
            "model": f"model-{task_id}",
            "messages": list(messages),
            "stream": True,
            "seed": turn_id,
        }
        records.append(
            CaptureRecord(
                capture_id=f"task-{task_id}-turn-{turn_id}",
                captured_at_ns=turn_id,
                request=request,
                request_sha256=request_hash(request),
            )
        )
        messages.extend(
            [
                {
                    "role": "assistant",
                    "reasoning_content": f"reason-{turn_id}",
                    "content": f"answer-{turn_id}",
                },
                {"role": "user", "content": f"next-{turn_id}"},
            ]
        )
    return TrajectoryTask(
        task_id=f"task-{task_id}",
        source_path=f"task-{task_id}.jsonl",
        turns=records,
    )


def test_matchers_support_exact_prefix_keywords_and_structured_tool_calls():
    reference = {
        "content": "alpha beta gamma",
        "reasoning_content": "reason",
        "tool_calls": [
            {"id": "ref", "type": "function", "function": {"name": "f", "arguments": '{"x":1}'}}
        ],
    }
    target = {
        "content": "alpha beta gamma",
        "reasoning_content": "reason",
        "tool_calls": [
            {"id": "target", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
        ],
    }

    assert compare_messages(reference, target, MatchConfig()) == []
    target["content"] = "alpha beta changed"
    assert (
        compare_messages(
            reference,
            target,
            MatchConfig(mode="prefix", prefix_chars=10),
        )
        == []
    )
    assert (
        compare_messages(
            reference,
            target,
            MatchConfig(mode="keywords", keywords=["alpha", "reason"]),
        )
        == []
    )


def test_server_metric_derivation_and_linear_percentile():
    sample = _derive_server_sample(_server_metrics())

    assert sample["ttft_ms"] == 0.0001
    assert sample["e2e_ms"] == 0.0004
    assert sample["tpot_ms"] == 0.0001
    assert sample["dropped_prompt_tokens"] == 8
    assert sample["prompt_retention_ratio"] == 0.6
    assert sample["drop_effective"] is True
    assert sample["tokenize_invocations"] == 1
    assert sample["reposition_d2h_bytes"] == 0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_prepare_manifest_preserves_exact_request_and_hash():
    request = _request()
    capture = CaptureRecord(
        capture_id="abcdef012345",
        captured_at_ns=123,
        request=request,
        request_sha256=request_hash(request),
    )

    cases = prepare_manifest(
        [capture],
        source="tau3",
        method="full",
        model=None,
        summary_triggered=None,
    )

    assert cases[0].request == request
    assert cases[0].request_sha256 == request_hash(request)
    assert cases[0].metadata.tags["capture_id"] == capture.capture_id


def test_post_chat_reads_nonstream_server_metrics():
    async def run():
        def handler(request: httpx.Request):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                    "server_metrics": _server_metrics(),
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await post_chat(
                client,
                base_url="http://test/v1",
                request_body=_request(),
                api_key="dummy",
            )

    result = asyncio.run(run())

    assert result.message["content"] == "answer"
    assert result.server_metrics == _server_metrics()
    assert result.usage == {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}


def test_post_chat_reads_streaming_usage_chunk_with_empty_choices():
    async def run():
        def handler(request: httpx.Request):
            body = json.loads(request.content)
            assert body["stream_options"] == {"include_usage": True}
            events = [
                {
                    "choices": [
                        {
                            "delta": {"content": "answer"},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                    "server_metrics": _server_metrics(),
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "total_tokens": 23,
                        "prompt_tokens_details": {
                            "cached_tokens": 10,
                            "drop_skipped_tokens": 2,
                        },
                    },
                },
            ]
            content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            content += "data: [DONE]\n\n"
            return httpx.Response(200, content=content)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await post_chat(
                client,
                base_url="http://test/v1",
                request_body={
                    **_request(),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                api_key="dummy",
            )

    result = asyncio.run(run())

    assert result.message["content"] == "answer"
    assert result.finish_reason == "length"
    assert result.server_metrics == _server_metrics()
    assert result.usage["prompt_tokens_details"]["drop_skipped_tokens"] == 2


def test_trajectory_request_sampling_and_thinking_drop_are_explicit():
    task = _trajectory_task(1)

    first, first_applicable = _build_trajectory_request(
        task.turns[0],
        variant="thinking_drop",
        turn_id=1,
        model_override="override",
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        force_non_stream=True,
    )
    second, second_applicable = _build_trajectory_request(
        task.turns[1],
        variant="thinking_drop",
        turn_id=2,
        model_override=None,
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        force_non_stream=False,
    )

    assert first_applicable is False
    assert "drop_rule" not in first
    assert first["model"] == "override"
    assert first["max_tokens"] == 2048
    assert first["ignore_eos"] is True
    assert first["stop"] is None
    assert first["stream"] is False
    assert second_applicable is True
    assert second["drop_rule"] == {"type": "thinking_drop"}
    assert second["stream_options"] == {"include_usage": True}


def test_normalize_usage_fills_unreported_cache_details_with_zero():
    normalized, reported = _normalize_usage(
        {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    )

    assert reported is False
    assert normalized["prompt_tokens_details"] == {
        "cached_tokens": 0,
        "drop_skipped_tokens": 0,
    }


def test_benchmark_uses_server_metrics_for_concurrency_summaries(monkeypatch):
    async def fake_post_chat(*args, **kwargs):
        now = 1_000
        metrics = _server_metrics()
        if "drop_message" not in kwargs["request_body"]:
            metrics["active_prompt_tokens"] = metrics["prompt_tokens"]
        return ChatResult(
            message={"role": "assistant", "content": "answer"},
            finish_reason="stop",
            server_metrics=metrics,
            client_started_ns=now,
            client_finished_ns=now + 500,
        )

    monkeypatch.setattr(
        "minisgl.benchmark.contextualize.runner.post_chat",
        fake_post_chat,
    )

    report = asyncio.run(
        benchmark_cases(
            [_case()],
            base_url="http://test",
            api_key="dummy",
            model_override=None,
            concurrencies=[1, 4, 8],
            num_requests=8,
            warmup_requests=1,
            timeout_seconds=1,
            force_non_stream=True,
        )
    )

    assert [group["concurrency"] for group in report["groups"]] == [1, 4, 8]
    assert all(group["succeeded"] == 8 for group in report["groups"])
    assert all(group["completion_token_throughput_per_second"] > 0 for group in report["groups"])
    assert all(group["drop_requested"] is True for group in report["groups"])
    assert all(group["drop_effective_requests"] == 8 for group in report["groups"])
    assert all(group["reposition_thresholds_passed"] for group in report["groups"])


def test_benchmark_accepts_no_summary_no_drop_baseline(monkeypatch):
    async def fake_post_chat(*args, **kwargs):
        metrics = _server_metrics()
        metrics["active_prompt_tokens"] = metrics["prompt_tokens"]
        return ChatResult(
            message={"role": "assistant", "content": "answer"},
            finish_reason="stop",
            server_metrics=metrics,
            client_started_ns=1_000,
            client_finished_ns=1_500,
        )

    monkeypatch.setattr(
        "minisgl.benchmark.contextualize.runner.post_chat",
        fake_post_chat,
    )
    report = asyncio.run(
        benchmark_cases(
            [_case(method="full", summary_triggered=False, with_drop=False)],
            base_url="http://test",
            api_key="dummy",
            model_override=None,
            concurrencies=[1],
            num_requests=1,
            warmup_requests=0,
            timeout_seconds=1,
            force_non_stream=True,
        )
    )

    assert report["method"] == "full"
    assert report["summary_triggered"] is False
    assert report["drop_requested"] is False
    assert report["groups"][0]["drop_effective_requests"] == 0


def test_benchmark_rejects_mixed_methods():
    cases = [
        _case(method="full", summary_triggered=False, with_drop=False),
        _case("drop", method="drop_kv", summary_triggered=False, with_drop=True),
    ]

    with pytest.raises(ValueError, match="exactly one method"):
        asyncio.run(
            benchmark_cases(
                cases,
                base_url="http://test",
                api_key="dummy",
                model_override=None,
                concurrencies=[1],
                num_requests=1,
                warmup_requests=0,
                timeout_seconds=1,
                force_non_stream=True,
            )
        )


def test_trajectory_benchmark_serializes_turns_and_aggregates_each_cell(monkeypatch):
    active_models = set()
    max_active = 0
    turns_by_model = {f"model-{task_id}": [] for task_id in range(4)}

    async def fake_post_chat(*args, **kwargs):
        nonlocal max_active
        request = kwargs["request_body"]
        model = request["model"]
        turn_id = request["seed"]
        assert model not in active_models
        active_models.add(model)
        max_active = max(max_active, len(active_models))
        turns_by_model[model].append(turn_id)
        await asyncio.sleep(0.001)
        active_models.remove(model)

        drop_requested = request.get("drop_rule") == {"type": "thinking_drop"}
        metrics = _server_metrics()
        metrics.update(
            prompt_tokens=100,
            active_prompt_tokens=80 if drop_requested else 100,
            generated_tokens=3,
            completion_tokens=3,
        )
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "total_tokens": 104,
            "prompt_tokens_details": {
                "cached_tokens": 50,
                "drop_skipped_tokens": 10 if drop_requested else 0,
            },
        }
        return ChatResult(
            message={"role": "assistant", "content": "answer"},
            finish_reason="length",
            server_metrics=metrics,
            client_started_ns=1_000,
            client_finished_ns=2_000,
            usage=usage,
        )

    monkeypatch.setattr(
        "minisgl.benchmark.contextualize.runner.post_chat",
        fake_post_chat,
    )
    report = asyncio.run(
        benchmark_trajectories(
            [_trajectory_task(task_id) for task_id in range(4)],
            trajectory_dir="trajectories",
            base_url="http://test",
            api_key="dummy",
            model_override=None,
            concurrencies=[1, 2, 4],
            variants=["no_drop", "thinking_drop"],
            max_turns=3,
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            timeout_seconds=1,
            force_non_stream=True,
        )
    )

    assert max_active == 4
    assert len(report["cells"]) == 6
    assert report["execution"]["cache_isolation"] == "shared_endpoint"
    assert all(cell["requested"] == 12 for cell in report["cells"])
    assert all(cell["failed"] == 0 for cell in report["cells"])
    assert all(cell["fixed_length_failed"] == 0 for cell in report["cells"])
    assert all(cell["completion_token_throughput_per_second"] > 0 for cell in report["cells"])
    for turns in turns_by_model.values():
        assert turns == [1, 2, 3] * 6

    thinking = next(
        cell
        for cell in report["cells"]
        if cell["variant"] == "thinking_drop" and cell["concurrency"] == 4
    )
    assert thinking["turns"][0]["drop_effective_requests"] == 0
    assert thinking["turns"][1]["drop_effective_requests"] == 4
    assert thinking["turns"][1]["usage"]["prompt_tokens"]["mean"] == 100.0
    assert thinking["turns"][1]["usage"]["cached_tokens"]["mean"] == 50.0
    assert thinking["turns"][1]["usage"]["drop_skipped_tokens"]["mean"] == 10.0
    assert thinking["requests"][0]["thinking_drop_applicable"] is False
    assert thinking["requests"][1]["thinking_drop_applicable"] is True
    assert thinking["requests"][1]["actual_completion_tokens"] == 4
    assert thinking["requests"][1]["actual_generated_tokens"] == 3


def test_trajectory_benchmark_retains_fixed_length_failure(monkeypatch):
    async def fake_post_chat(*args, **kwargs):
        metrics = _server_metrics()
        metrics.update(
            prompt_tokens=10,
            active_prompt_tokens=10,
            generated_tokens=3,
            completion_tokens=3,
        )
        return ChatResult(
            message={"role": "assistant", "content": "short"},
            finish_reason="length",
            server_metrics=metrics,
            client_started_ns=1_000,
            client_finished_ns=2_000,
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        )

    monkeypatch.setattr(
        "minisgl.benchmark.contextualize.runner.post_chat",
        fake_post_chat,
    )
    report = asyncio.run(
        benchmark_trajectories(
            [_trajectory_task(0, turns=1)],
            trajectory_dir="trajectories",
            base_url="http://test",
            api_key="dummy",
            model_override=None,
            concurrencies=[1],
            variants=["no_drop"],
            max_turns=1,
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            timeout_seconds=1,
            force_non_stream=True,
        )
    )

    cell = report["cells"][0]
    assert cell["succeeded"] == 1
    assert cell["fixed_length_failed"] == 1
    assert cell["turns"][0]["sample_count"] == 0
    assert cell["requests"][0]["passed"] is False
    assert cell["requests"][0]["actual_completion_tokens"] == 3
    assert cell["requests"][0]["actual_generated_tokens"] == 3
