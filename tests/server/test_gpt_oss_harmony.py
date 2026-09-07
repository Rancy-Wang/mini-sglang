from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from minisgl.message import UserReply
from minisgl.server.api_server import FrontendManager
from openai_harmony import HarmonyEncodingName, load_harmony_encoding


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search documents",
            "parameters": {"type": "object"},
        },
    }
]


def test_streaming_api_uses_token_ids_and_never_leaks_role_recipient_header() -> None:
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    completion = (
        " to=functions.search<|channel|>commentary json"
        '<|message|>{"query":"api"}<|call|>'
    )
    token_ids = [
        int(token)
        for token in encoding.encode(completion, allowed_special="all")
    ]

    manager = object.__new__(FrontendManager)
    manager.config = SimpleNamespace(
        model_path="openai/gpt-oss-20b",
        tool_call_parser="auto",
        reasoning_parser="auto",
    )

    async def wait_for_ack(uid):
        for index, token_id in enumerate(token_ids):
            yield UserReply(
                uid=uid,
                incremental_output=encoding.decode([token_id]),
                incremental_token_ids=[token_id],
                finished=index == len(token_ids) - 1,
                finish_reason="stop" if index == len(token_ids) - 1 else None,
            )

    manager.wait_for_ack = wait_for_ack

    async def collect():
        return [
            chunk
            async for chunk in manager.stream_chat_completions(
                7,
                tools=TOOLS,
                tool_choice="auto",
                model="openai/gpt-oss-20b",
            )
        ]

    chunks = asyncio.run(collect())
    payloads = [
        json.loads(chunk.decode().removeprefix("data: "))
        for chunk in chunks
        if chunk.startswith(b"data: {")
    ]
    deltas = [choice["delta"] for payload in payloads for choice in payload["choices"]]
    calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
    visible_content = "".join(delta.get("content", "") for delta in deltas)

    assert visible_content == ""
    assert "to=functions" not in json.dumps(payloads)
    assert len(calls) == 1
    assert calls[0]["function"] == {
        "name": "search",
        "arguments": '{"query":"api"}',
    }
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
