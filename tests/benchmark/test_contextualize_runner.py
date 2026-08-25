from __future__ import annotations

import asyncio

import httpx
import pytest

from minisgl.benchmark.contextualize.manifest import (
    CaptureRecord,
    CaseMetadata,
    ManifestCase,
    MatchConfig,
    request_hash,
)
from minisgl.benchmark.contextualize.runner import (
    ChatResult,
    _derive_server_sample,
    _percentile,
    benchmark_cases,
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
    }


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
    assert compare_messages(
        reference,
        target,
        MatchConfig(mode="prefix", prefix_chars=10),
    ) == []
    assert compare_messages(
        reference,
        target,
        MatchConfig(mode="keywords", keywords=["alpha", "reason"]),
    ) == []


def test_server_metric_derivation_and_linear_percentile():
    sample = _derive_server_sample(_server_metrics())

    assert sample["ttft_ms"] == 0.0001
    assert sample["e2e_ms"] == 0.0004
    assert sample["tpot_ms"] == 0.0001
    assert sample["dropped_prompt_tokens"] == 8
    assert sample["prompt_retention_ratio"] == 0.6
    assert sample["drop_effective"] is True
    assert _percentile([1.0, 2.0, 3.0, 4.0], 95) == 3.8499999999999996


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
