import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager
from openai_harmony import HarmonyEncodingName, load_harmony_encoding


class GptOssTokenizerStub:
    name_or_path = "openai/gpt-oss-20b"
    is_fast = True
    special_tokens_map = {}
    eos_token_id = 200002

    def encode(self, text, **kwargs):
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS).encode(text)


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
    assert all(marker in full_text for marker in ("SECRET_USER", "PRIVATE_REASON", "SECRET_TOOL_RESULT"))

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
