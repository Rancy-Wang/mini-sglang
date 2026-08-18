from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import minisgl.server.api_server as api_server
from minisgl.message import UserReply
from minisgl.server.api_server import FrontendManager, OpenAICompletionRequest


async def _collect_stream(manager):
    return [chunk async for chunk in manager.stream_chat_completions(7)]


def test_streaming_response_exposes_ratio_only_on_terminal_finish_chunk():
    manager = object.__new__(FrontendManager)
    manager.config = SimpleNamespace(model_path="Qwen/Qwen3-1.7B")

    async def wait_for_ack(uid):
        yield UserReply(uid=uid, incremental_output="hello", finished=False)
        yield UserReply(
            uid=uid,
            incremental_output="",
            finished=True,
            finish_reason="stop",
            cache_hit_ratio=0.625,
        )

    manager.wait_for_ack = wait_for_ack
    chunks = asyncio.run(_collect_stream(manager))
    payloads = [
        json.loads(chunk.decode().removeprefix("data: "))
        for chunk in chunks
        if chunk.startswith(b"data: {")
    ]

    assert all("cache_hit_ratio" not in payload for payload in payloads[:-1])
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["cache_hit_ratio"] == 0.625


def test_nonstream_openai_response_exposes_final_generation_ratio(monkeypatch):
    class FakeState:
        config = SimpleNamespace(
            radix_drop_key_mode="delta-marker",
            model_path="Qwen/Qwen3-1.7B",
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
                cache_hit_ratio=0.75,
            )

    state = FakeState()
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    request = OpenAICompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "question"}],
        max_tokens=1,
    )
    response = asyncio.run(api_server.v1_completions(request, SimpleNamespace()))

    assert response["cache_hit_ratio"] == 0.75
    assert response["choices"][0]["message"]["content"] == "answer"


def test_incomplete_generation_reply_has_no_ratio_by_default():
    reply = UserReply(uid=1, incremental_output="partial", finished=False)
    assert reply.cache_hit_ratio is None
