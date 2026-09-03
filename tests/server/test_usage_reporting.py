from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from minisgl.message import UserReply
from minisgl.message.metrics import RequestMetricsState
from minisgl.server.api_server import (
    CacheUsageReport,
    FrontendManager,
    OpenAICompletionRequest,
    _build_usage,
)


def test_usage_matches_sglang_shape_with_drop_details() -> None:
    usage = _build_usage(
        prompt_tokens=82,
        completion_tokens=7,
        cached_tokens=50,
        drop_skipped_tokens=31,
    )

    assert usage == {
        "prompt_tokens": 82,
        "completion_tokens": 7,
        "total_tokens": 89,
        "prompt_tokens_details": {
            "cached_tokens": 50,
            "drop_skipped_tokens": 31,
        },
    }


def test_usage_omits_empty_prompt_details_and_checks_bounds() -> None:
    usage = _build_usage(
        prompt_tokens=5,
        completion_tokens=2,
        cached_tokens=0,
    )
    assert "prompt_tokens_details" not in usage
    with pytest.raises(ValueError, match="prompt_tokens"):
        _build_usage(
            prompt_tokens=5,
            completion_tokens=0,
            cached_tokens=4,
            drop_skipped_tokens=2,
        )


def _stream_chunks(*, include_usage: bool) -> list[dict | str]:
    manager = FrontendManager(
        config=SimpleNamespace(
            model_path="Qwen3-1.7B",
            tool_call_parser="auto",
            reasoning_parser="auto",
        ),
        send_tokenizer=None,
        recv_tokenizer=None,
    )

    async def wait_for_ack(_uid):
        yield UserReply(
            uid=3,
            incremental_output="answer",
            finished=True,
            finish_reason="stop",
            cached_tokens=4,
            prompt_tokens=10,
            completion_tokens=2,
        )

    manager.wait_for_ack = wait_for_ack

    async def collect():
        result = []
        async for raw in manager.stream_chat_completions(
            3,
            model="Qwen3-1.7B",
            include_usage=include_usage,
            cache_report=CacheUsageReport(cached_tokens=4, drop_skipped_tokens=5),
        ):
            line = raw.decode().strip()
            payload = line.removeprefix("data: ")
            result.append(payload if payload == "[DONE]" else json.loads(payload))
        return result

    return asyncio.run(collect())


def test_stream_usage_is_a_final_empty_choices_chunk() -> None:
    chunks = _stream_chunks(include_usage=True)

    assert chunks[-1] == "[DONE]"
    assert chunks[-2]["choices"] == []
    assert chunks[-2]["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 4,
        "drop_skipped_tokens": 5,
    }
    finish = chunks[-3]
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert "cached_tokens" not in finish
    assert "cache_hit_ratio" not in finish


def test_stream_without_include_usage_has_no_usage_chunk() -> None:
    chunks = _stream_chunks(include_usage=False)

    assert chunks[-1] == "[DONE]"
    assert not any(isinstance(chunk, dict) and "usage" in chunk for chunk in chunks)


def test_request_accepts_sglang_stream_options() -> None:
    req = OpenAICompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert req.stream_options is not None and req.stream_options.include_usage


def test_server_metrics_report_template_dispatch_ipc_and_retry_transfer_separately() -> None:
    state = RequestMetricsState(
        request_received_ns=10,
        prompt_tokens=20,
        active_prompt_tokens=12,
        tokenize_invocations=1,
        chat_template_invocations=1,
        context_stage_count=3,
        reposition_ipc_tensor_bytes=4096,
    )
    state.observe_reposition(
        radix_match_ns=30,
        retry_plan_ns=40,
        transition_count=5,
        h2d_bytes=100,
        d2h_bytes=0,
    )
    state.observe_token(20, visible=True)

    metrics = state.finish(25).as_api_dict()

    assert metrics["tokenize_invocations"] == 1
    assert metrics["chat_template_invocations"] == 1
    assert metrics["context_stage_count"] == 3
    assert metrics["reposition_ipc_tensor_bytes"] == 4096
    assert metrics["reposition_h2d_bytes"] == 100
    assert metrics["reposition_d2h_bytes"] == 0
