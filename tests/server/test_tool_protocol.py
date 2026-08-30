from __future__ import annotations

import json

from minisgl.message import DetokenizeMsg
from minisgl.server.response_parser import (
    ChatResponseParser,
    infer_reasoning_parser,
    infer_tool_call_parser,
)
from minisgl.scheduler.prefill import _calculate_cache_reuse_ratio
from minisgl.tokenizer.detokenize import DetokenizeManager
from minisgl.tokenizer.tokenize import TokenizeManager


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search documents",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _parser(
    model: str,
    *,
    thinking: bool = False,
    stream_reasoning: bool = True,
    tools=TOOLS,
) -> ChatResponseParser:
    return ChatResponseParser(
        model_path=model,
        tools=tools,
        tool_call_parser="auto",
        reasoning_parser="auto",
        enable_thinking=thinking,
        separate_reasoning=True,
        stream_reasoning=stream_reasoning,
    )


def test_supported_family_parser_inference() -> None:
    assert infer_tool_call_parser("Qwen/Qwen3-1.7B") == "qwen"
    assert infer_tool_call_parser("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") == "qwen"
    assert infer_tool_call_parser("deepseek-ai/DeepSeek-R1-Distill-Llama-8B") == "llama3"
    assert infer_tool_call_parser("openai/gpt-oss-20b") == "gpt-oss"
    assert infer_reasoning_parser("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") == "deepseek-r1"
    assert infer_reasoning_parser("Qwen/Qwen3-1.7B") == "qwen3"


def test_qwen_native_tool_call_and_no_generic_json_guessing() -> None:
    parser = _parser("Qwen/Qwen3-1.7B")
    parsed = parser.parse_full(
        'checking\n<tool_call>\n{"name":"search","arguments":{"query":"kv cache"}}\n'
        "</tool_call>"
    )
    assert parsed.content == "checking"
    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0]["function"] == {
        "name": "search",
        "arguments": '{"query":"kv cache"}',
    }

    ordinary = _parser("Qwen/Qwen3-1.7B").parse_full('{"name":"search","arguments":{}}')
    assert ordinary.content == '{"name":"search","arguments":{}}'
    assert ordinary.tool_calls is None


def test_qwen_stream_handles_split_markers_and_emits_one_call() -> None:
    parser = _parser("Qwen/Qwen3-1.7B")
    pieces = [
        parser.feed("before <tool_ca"),
        parser.feed('ll>\n{"name":"search","arguments":{"query":"split"}}\n</tool_'),
        parser.feed("call>"),
        parser.finish(),
    ]
    assert "".join(piece.content for piece in pieces) == "before "
    calls = [call for piece in pieces for call in piece.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"


def test_stream_reasoning_false_buffers_until_reasoning_boundary() -> None:
    parser = _parser("Qwen/Qwen3-1.7B", thinking=True, stream_reasoning=False)
    pieces = [parser.feed("hidden "), parser.feed("plan</think>answer")]
    assert "".join(piece.reasoning_content for piece in pieces) == "hidden plan"
    assert "".join(piece.content for piece in pieces) == "answer"

    tool_parser = _parser("Qwen/Qwen3-1.7B", thinking=True, stream_reasoning=False)
    chunks = [
        tool_parser.feed("choose search<tool_ca"),
        tool_parser.feed('ll>\n{"name":"search","arguments":{"query":"buffered"}}\n</tool_call>'),
    ]
    assert "".join(chunk.reasoning_content for chunk in chunks) == "choose search"
    calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == '{"query":"buffered"}'


def test_qwen_and_deepseek_reasoning_are_separated_before_tools() -> None:
    qwen = _parser("Qwen/Qwen3-1.7B", thinking=True).parse_full(
        "plan</think><tool_call>\n"
        '{"name":"search","arguments":{"query":"answer"}}\n</tool_call>'
    )
    assert qwen.reasoning_content == "plan"
    assert qwen.tool_calls is not None

    deepseek = _parser("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").parse_full(
        "hidden reasoning</think>visible"
    )
    assert deepseek.reasoning_content == "hidden reasoning"
    assert deepseek.content == "visible"


def test_qwen3_coder_and_llama_native_formats() -> None:
    coder = _parser("Qwen/Qwen3-Coder-30B").parse_full(
        "<tool_call><function=search><parameter=query>hello</parameter>"
        "</function></tool_call>"
    )
    assert coder.tool_calls is not None
    assert json.loads(coder.tool_calls[0]["function"]["arguments"]) == {"query": "hello"}

    llama = _parser("deepseek-ai/DeepSeek-R1-Distill-Llama-8B").parse_full(
        '<|python_tag|>{"name":"search","arguments":{"query":"llama"}}'
    )
    assert llama.tool_calls is not None
    assert llama.tool_calls[0]["function"]["name"] == "search"

    stream = _parser("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    chunks = [
        stream.feed("brief thought<|python_"),
        stream.feed('tag|>{"name":"search","arguments":{"query":"stream"}}'),
        stream.finish(),
    ]
    assert "".join(chunk.reasoning_content for chunk in chunks) == "brief thought"
    calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == '{"query":"stream"}'


def test_harmony_stream_separates_reasoning_content_and_tool_call() -> None:
    parser = _parser("openai/gpt-oss-20b")
    chunks = [
        parser.feed("<|start|>assistant<|channel|>analysis<|message|>think"),
        parser.feed("ing<|end|><|start|>assistant<|channel|>commentary to=functions.search"),
        parser.feed('<|constrain|>json<|message|>{"query":"oss"}<|call|>'),
        parser.finish(),
    ]
    assert "".join(chunk.reasoning_content for chunk in chunks) == "thinking"
    calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == '{"query":"oss"}'


def _harmony_ids(text: str) -> list[int]:
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return [int(token) for token in encoding.encode(text, allowed_special="all")]


def _token_stream_result(
    text: str,
    *,
    chunk_size: int,
    tools=TOOLS,
) -> tuple[str, str, list[dict]]:
    parser = _parser("openai/gpt-oss-20b", tools=tools)
    token_ids = _harmony_ids(text)
    pieces = [
        parser.feed("", token_ids=token_ids[start : start + chunk_size])
        for start in range(0, len(token_ids), chunk_size)
    ]
    pieces.append(parser.finish())
    return (
        "".join(piece.content for piece in pieces),
        "".join(piece.reasoning_content for piece in pieces),
        [call for piece in pieces for call in piece.tool_calls],
    )


def test_harmony_token_stream_accepts_both_recipient_positions_for_every_partition() -> None:
    variants = (
        ' to=functions.search<|channel|>commentary json<|message|>{"query":"分块"}<|call|>',
        '<|channel|>commentary to=functions.search json<|message|>{"query":"分块"}<|call|>',
    )
    for text in variants:
        token_count = len(_harmony_ids(text))
        for chunk_size in range(1, token_count + 1):
            content, reasoning, calls = _token_stream_result(
                text,
                chunk_size=chunk_size,
            )
            assert content == ""
            assert reasoning == ""
            assert len(calls) == 1
            assert calls[0]["function"] == {
                "name": "search",
                "arguments": '{"query":"分块"}',
            }


def test_harmony_stream_and_full_token_parsing_are_semantically_identical() -> None:
    text = (
        "<|channel|>analysis<|message|>think<|end|>"
        "<|start|>assistant to=functions.search<|channel|>commentary json"
        '<|message|>{"query":"same"}<|call|>'
    )
    content, reasoning, calls = _token_stream_result(text, chunk_size=1)
    parsed = _parser("openai/gpt-oss-20b").parse_full(
        "",
        token_ids=_harmony_ids(text),
    )

    assert content == parsed.content == ""
    assert reasoning == parsed.reasoning_content == "think"
    assert parsed.tool_calls is not None
    assert [call["function"] for call in calls] == [
        call["function"] for call in parsed.tool_calls
    ]


def test_harmony_preserves_dotted_tool_names_and_raw_invalid_arguments() -> None:
    dotted_tools = [
        {
            "type": "function",
            "function": {
                "name": "math.sum",
                "description": "Sum values",
                "parameters": {"type": "object"},
            },
        }
    ]
    dotted = (
        " to=functions.math.sum<|channel|>commentary json"
        '<|message|>{"values":[1,2]}<|call|>'
    )
    _, _, dotted_calls = _token_stream_result(
        dotted,
        chunk_size=1,
        tools=dotted_tools,
    )
    assert dotted_calls[0]["function"] == {
        "name": "math.sum",
        "arguments": '{"values":[1,2]}',
    }

    invalid = " to=functions.search<|channel|>commentary json<|message|>{bad<|call|>"
    _, _, invalid_calls = _token_stream_result(invalid, chunk_size=1)
    assert invalid_calls[0]["function"] == {
        "name": "search",
        "arguments": "{bad",
    }


def test_harmony_process_eos_recovers_a_complete_unterminated_tool_body() -> None:
    text = (
        " to=functions.search<|channel|>commentary json"
        '<|message|>{"query":"eos"}'
    )
    content, reasoning, calls = _token_stream_result(text, chunk_size=1)
    assert content == reasoning == ""
    assert calls[0]["function"] == {
        "name": "search",
        "arguments": '{"query":"eos"}',
    }


def test_harmony_stream_accepts_analysis_channel_tool_call() -> None:
    parser = _parser("openai/gpt-oss-20b")
    chunks = [
        parser.feed(
            "<|start|>assistant<|channel|>analysis to=functions.search code"
            '<|message|>{"query":"analysis tool"}'
        ),
        parser.finish(),
    ]
    calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"
    assert calls[0]["function"]["arguments"] == '{"query":"analysis tool"}'


def test_harmony_full_falls_back_for_reordered_recipient_header() -> None:
    parsed = _parser("openai/gpt-oss-20b").parse_full(
        "<|start|>assistant<|channel|>commentary json to=functions.search"
        '<|message|>{"query":"reordered"}<|call|>'
    )
    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0]["function"]["name"] == "search"
    assert parsed.tool_calls[0]["function"]["arguments"] == '{"query":"reordered"}'


def test_harmony_stream_reasoning_false_buffers_until_channel_end() -> None:
    parser = _parser("openai/gpt-oss-20b", stream_reasoning=False)
    chunks = [
        parser.feed("<|start|>assistant<|channel|>analysis<|message|>think"),
        parser.feed("ing<|end|><|start|>assistant<|channel|>final<|message|>done"),
        parser.finish(),
    ]
    assert chunks[0].reasoning_content == ""
    assert "".join(chunk.reasoning_content for chunk in chunks) == "thinking"
    assert "".join(chunk.content for chunk in chunks) == "done"


class _FakeTokenizer:
    name_or_path = "fake"
    eos_token_id = 0

    def batch_decode(self, batches):
        return ["".join({1: "lastword"}.get(token, "") for token in batch) for batch in batches]


def test_terminal_detokenize_flushes_last_word() -> None:
    manager = DetokenizeManager(_FakeTokenizer())
    output = manager.detokenize([DetokenizeMsg(uid=1, next_token=1, finished=True)])
    assert output == ["lastword"]


def test_internal_cache_reuse_ratio_uses_selected_prefix() -> None:
    reuse_ratio = _calculate_cache_reuse_ratio(
        cached_len=5,
        matchable_prefix_len=5,
    )
    assert reuse_ratio == 1.0


def test_template_tools_follow_sglang_wrapper_then_flat_fallback() -> None:
    class _TemplateTokenizer:
        name_or_path = "Qwen3"

    manager = TokenizeManager(_TemplateTokenizer())
    assert manager._effective_template_tools(TOOLS) == TOOLS
    assert manager._flatten_tools(TOOLS) == [TOOLS[0]["function"]]
    messages, offset = manager._build_template_messages(
        [{"role": "user", "content": "hello"}],
        safe_mode=False,
    )
    assert messages == [{"role": "user", "content": "hello"}]
    assert offset == 0

    class _BareOnlyTokenizer:
        name_or_path = "Qwen3"

        def __init__(self) -> None:
            self.tool_shapes = []

        def apply_chat_template(self, messages, **kwargs):
            tools = kwargs["tools"]
            self.tool_shapes.append("wrapper" if "function" in tools[0] else "bare")
            if self.tool_shapes[-1] == "wrapper":
                raise ValueError("bare tools required")
            return [1, 2]

    tokenizer = _BareOnlyTokenizer()
    manager = TokenizeManager(tokenizer)
    assert manager._apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=None,
        tools=TOOLS,
    ) == [1, 2]
    assert tokenizer.tool_shapes == ["wrapper", "bare"]
    assert manager._effective_template_tools(TOOLS) == [TOOLS[0]["function"]]
