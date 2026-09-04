from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import minisgl.server.api_server as api_server
import pytest
from minisgl.message import (
    BaseTokenizerMsg,
    DetokenizeMsg,
    RequestMetricsState,
    ServerMetrics,
    UserReply,
)
from minisgl.server.api_server import FrontendManager, OpenAICompletionRequest


def _metrics() -> ServerMetrics:
    return ServerMetrics(
        request_received_ns=100,
        first_token_generated_ns=250,
        request_finished_ns=550,
        prompt_tokens=12,
        active_prompt_tokens=8,
        generated_tokens=4,
        completion_tokens=3,
        tokenize_invocations=1,
        context_stage_count=3,
        radix_compile_ns=10,
        radix_match_ns=20,
        retry_plan_ns=5,
        reposition_transition_count=7,
        reposition_h2d_bytes=84,
        reposition_d2h_bytes=0,
    )


def test_request_metrics_state_counts_generated_and_visible_tokens():
    state = RequestMetricsState(
        request_received_ns=100,
        prompt_tokens=12,
        active_prompt_tokens=8,
    )
    state.observe_token(250, visible=True)
    state.observe_token(400, visible=True)
    state.observe_token(550, visible=False)

    metrics = state.finish(550)

    assert metrics.first_token_generated_ns == 250
    assert metrics.generated_tokens == 3
    assert metrics.completion_tokens == 2
    assert metrics.as_api_dict()["active_prompt_tokens"] == 8


def test_request_metrics_state_rejects_nonmonotonic_token_timestamps():
    state = RequestMetricsState(
        request_received_ns=100,
        prompt_tokens=12,
        active_prompt_tokens=8,
    )
    state.observe_token(400, visible=True)

    with pytest.raises(ValueError, match="monotonic"):
        state.observe_token(399, visible=True)


def test_request_metrics_state_exposes_cumulative_reposition_counters():
    state = RequestMetricsState(
        request_received_ns=100,
        prompt_tokens=12,
        active_prompt_tokens=8,
        context_stage_count=3,
        radix_compile_ns=10,
    )
    state.observe_reposition(
        radix_match_ns=20,
        retry_plan_ns=5,
        transition_count=7,
        h2d_bytes=84,
    )
    state.observe_token(250, visible=True)

    metrics = state.finish(250)

    assert metrics.context_stage_count == 3
    assert metrics.radix_compile_ns == 10
    assert metrics.radix_match_ns == 20
    assert metrics.retry_plan_ns == 5
    assert metrics.reposition_transition_count == 7
    assert metrics.reposition_h2d_bytes == 84
    assert metrics.reposition_d2h_bytes == 0


def test_server_metrics_round_trip_through_tokenizer_message_serialization():
    original = DetokenizeMsg(
        uid=7,
        next_token=42,
        finished=True,
        finish_reason="stop",
        server_metrics=_metrics(),
    )

    restored = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(original))

    assert isinstance(restored, DetokenizeMsg)
    assert restored.server_metrics == _metrics()


def test_stream_terminal_chunk_exposes_server_metrics_without_a_second_usage_result():
    manager = object.__new__(FrontendManager)
    manager.config = SimpleNamespace(
        model_path="test-model",
        tool_call_parser=None,
        reasoning_parser=None,
    )

    async def wait_for_ack(uid):
        yield UserReply(uid=uid, incremental_output="answer", finished=False)
        yield UserReply(
            uid=uid,
            incremental_output="",
            finished=True,
            finish_reason="stop",
            server_metrics=_metrics(),
        )

    manager.wait_for_ack = wait_for_ack

    async def collect():
        return [chunk async for chunk in manager.stream_chat_completions(7)]

    chunks = asyncio.run(collect())
    payloads = [
        json.loads(chunk.decode().removeprefix("data: "))
        for chunk in chunks
        if chunk.startswith(b"data: {")
    ]

    assert all("server_metrics" not in payload for payload in payloads[:-1])
    assert payloads[-1]["server_metrics"] == _metrics().as_api_dict()
    assert "usage" not in payloads[-1]


def test_nonstream_response_propagates_request_start_and_real_usage(monkeypatch):
    class FakeState:
        config = SimpleNamespace(
            radix_drop_key_mode="delta-marker",
            model_path="test-model",
            tool_call_parser=None,
            reasoning_parser=None,
        )

        def new_user(self):
            return 9

        async def send_one(self, msg):
            self.msg = msg

        async def wait_for_ack(self, uid):
            yield UserReply(
                uid=uid,
                incremental_output="answer",
                finished=True,
                finish_reason="stop",
                prompt_tokens=12,
                completion_tokens=4,
                server_metrics=_metrics(),
            )

    state = FakeState()
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    request = OpenAICompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "question"}],
        max_tokens=1,
    )

    response = asyncio.run(api_server.v1_completions(request, SimpleNamespace()))

    assert isinstance(state.msg.request_received_ns, int)
    assert state.msg.request_received_ns > 0
    assert response["server_metrics"] == _metrics().as_api_dict()
    assert response["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
