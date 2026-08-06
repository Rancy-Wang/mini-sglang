import asyncio
import json
from types import SimpleNamespace

from minisgl.message import UserReply
from minisgl.server import api_server


def _harmony_marker(name: str) -> str:
    return "".join(("<", "|", name, "|", ">"))


def test_harmony_output_separates_reasoning_final_and_tool_calls():
    raw = "".join(
        (
            _harmony_marker("channel"),
            "analysis",
            _harmony_marker("message"),
            "analysis sample",
            _harmony_marker("end"),
            _harmony_marker("start"),
            "assistant to=functions.lookup",
            _harmony_marker("channel"),
            "commentary ",
            _harmony_marker("constrain"),
            "json",
            _harmony_marker("message"),
            '{"q":"weather"}',
            _harmony_marker("call"),
            _harmony_marker("start"),
            "assistant",
            _harmony_marker("channel"),
            "final",
            _harmony_marker("message"),
            "public answer",
            _harmony_marker("end"),
        )
    )

    parsed = api_server._parse_harmony_output(raw)

    assert parsed.reasoning_content == "analysis sample"
    assert parsed.content == "public answer"
    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0]["function"] == {
        "name": "lookup",
        "arguments": '{"q":"weather"}',
    }


def test_gpt_oss_stream_buffers_harmony_control_tokens(monkeypatch):
    manager = object.__new__(api_server.FrontendManager)
    manager.config = SimpleNamespace(model_path="/models/gpt-oss-20b")
    raw = "".join(
        (
            _harmony_marker("channel"),
            "analysis",
            _harmony_marker("message"),
            "reasoning sample",
            _harmony_marker("end"),
            _harmony_marker("start"),
            "assistant",
            _harmony_marker("channel"),
            "final",
            _harmony_marker("message"),
            "answer",
            _harmony_marker("end"),
        )
    )

    async def fake_wait_for_ack(uid):
        yield UserReply(uid=uid, incremental_output=raw[:30], finished=False)
        yield UserReply(
            uid=uid,
            incremental_output=raw[30:],
            finished=True,
            finish_reason="stop",
        )

    manager.wait_for_ack = fake_wait_for_ack
    monkeypatch.setattr(api_server, "_is_gpt_oss_model", lambda _: True)

    async def collect():
        return [
            chunk
            async for chunk in manager.stream_chat_completions(
                7,
                tools=None,
                tool_choice="none",
            )
        ]

    chunks = asyncio.run(collect())
    first = json.loads(chunks[0].decode().removeprefix("data: "))
    delta = first["choices"][0]["delta"]

    assert delta == {
        "role": "assistant",
        "reasoning_content": "reasoning sample",
        "content": "answer",
    }
    assert all(_harmony_marker("start") not in chunk.decode() for chunk in chunks)
