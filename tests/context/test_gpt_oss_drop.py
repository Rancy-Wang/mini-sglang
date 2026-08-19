import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.message.tokenizer import get_gpt_oss_terminal_stop_token_ids
from minisgl.tokenizer.tokenize import TokenizeManager
from openai_harmony import HarmonyEncodingName, load_harmony_encoding


class GptOssTokenizerStub:
    name_or_path = "openai/gpt-oss-20b"
    is_fast = True
    special_tokens_map = {}
    eos_token_id = 200002

    def encode(self, text, **kwargs):
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS).encode(text)


class CountingHarmonyEncoding:
    def __init__(self):
        self.inner = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.completion_renders = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def render_conversation_for_completion(self, *args, **kwargs):
        self.completion_renders += 1
        return self.inner.render_conversation_for_completion(*args, **kwargs)


def test_harmony_analysis_boundary_is_not_a_terminal_generation_stop():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    message_end = "".join(("<", "|", "end", "|", ">"))
    continuation_id = next(
        int(token_id)
        for token_id in encoding.stop_tokens()
        if encoding.decode([int(token_id)]) == message_end
    )
    terminal_ids = set(get_gpt_oss_terminal_stop_token_ids())

    assert continuation_id not in terminal_ids
    assert terminal_ids == set(map(int, encoding.stop_tokens())) - {continuation_id}


def test_harmony_agent_history_drop_preserves_system_tools_and_absolute_positions():
    messages = [
        {"role": "system", "content": "SYSTEM_KEEP"},
        {"role": "user", "content": "SECRET_USER"},
        {
            "role": "assistant",
            "reasoning_content": "PRIVATE_REASON",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"secret"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "lookup",
            "content": "SECRET_TOOL_RESULT",
        },
        {"role": "user", "content": "PUBLIC_NEXT"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    manager = TokenizeManager(GptOssTokenizerStub())
    result = manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=8),
                target_msg_id=len(messages),
                drop_message={4: [1, 2, 3]},
                reasoning_effort="high",
                tools=tools,
                tool_choice="auto",
            )
        ]
    )[0]

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    active_text = encoding.decode(result.input_ids.tolist())
    full_text = encoding.decode(result.full_input_ids.tolist())

    assert "Reasoning: high" in active_text
    assert "SYSTEM_KEEP" in active_text
    assert "type lookup" in active_text
    assert "PUBLIC_NEXT" in active_text
    assert "SECRET_USER" not in active_text
    assert "PRIVATE_REASON" not in active_text
    assert "SECRET_TOOL_RESULT" not in active_text
    assert all(
        marker in full_text
        for marker in ("SECRET_USER", "PRIVATE_REASON", "SECRET_TOOL_RESULT")
    )

    assert result.drop_event_positions.numel() == 1
    event_position = int(result.drop_event_positions[0])
    dropped = ~result.full_keep_mask.to(torch.bool)
    sentinel = torch.iinfo(torch.int32).max
    assert torch.all(result.full_token_visible_until[dropped] == event_position)
    assert torch.all(result.full_token_visible_until[~dropped] == sentinel)
    assert torch.equal(
        result.true_positions,
        torch.arange(len(result.full_input_ids), dtype=torch.int32)[~dropped],
    )


def test_harmony_message_drop_renders_once_and_splits_merged_instructions():
    messages = [
        {"role": "system", "content": "KEEP_SYSTEM_INSTRUCTION"},
        {"role": "developer", "content": "DROP_DEVELOPER_INSTRUCTION"},
        {"role": "user", "content": "CONTINUE"},
    ]
    manager = TokenizeManager(GptOssTokenizerStub())
    encoding = CountingHarmonyEncoding()
    manager._harmony_encoding = encoding
    result = manager.tokenize(
        [
            TokenizeMsg(
                uid=3,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_message={2: [1]},
            )
        ]
    )[0]
    active = encoding.decode(result.input_ids.tolist())
    full = encoding.decode(result.full_input_ids.tolist())

    assert encoding.completion_renders == 1
    assert "KEEP_SYSTEM_INSTRUCTION" in active
    assert "DROP_DEVELOPER_INSTRUCTION" not in active
    assert "DROP_DEVELOPER_INSTRUCTION" in full
    assert "CONTINUE" in active


def test_harmony_thinking_drop_removes_only_analysis_content_tokens():
    messages = [
        {"role": "user", "content": "Compute 15 + 27."},
        {
            "role": "assistant",
            "reasoning_content": "PRIVATE_HARMONY_REASONING",
            "content": "The answer is 42.",
        },
        {"role": "user", "content": "Multiply it by 3."},
    ]
    result = TokenizeManager(GptOssTokenizerStub()).tokenize(
        [
            TokenizeMsg(
                uid=2,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_rule={"type": "thinking_drop"},
            )
        ]
    )[0]
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    full = encoding.decode(result.full_input_ids.tolist())
    active = encoding.decode(result.input_ids.tolist())

    assert "PRIVATE_HARMONY_REASONING" in full
    assert "PRIVATE_HARMONY_REASONING" not in active
    assert "The answer is 42." in active
    # Harmony protocol/channel delimiters are owner metadata, not reasoning content.
    assert "analysis" in active
    assert result.drop_event_positions.numel() == 1
