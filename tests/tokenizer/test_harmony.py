from datetime import date

import pytest

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager


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


class _GptOssTokenizer:
    name_or_path = "openai/gpt-oss-20b"


def _render(messages):
    manager = TokenizeManager(_GptOssTokenizer())
    ids = manager._render_harmony_tokens(
        messages,
        add_generation_prompt=True,
        enable_thinking=None,
        tools=TOOLS,
    )
    return manager._get_harmony_encoding().decode_utf8(ids)


def test_harmony_history_matches_current_vllm_tool_headers():
    rendered = _render(
        [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "I will search.",
                "reasoning": "reasoning",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
    )

    assert f"Current date: {date.today().isoformat()}" in rendered
    commentary = "<|channel|>commentary<|message|>I will search."
    analysis = "<|channel|>analysis<|message|>reasoning"
    assert rendered.index(commentary) < rendered.index(analysis)
    assert (
        "<|start|>assistant to=functions.search<|channel|>commentary json<|message|>"
        '{"query":"x"}<|call|>'
    ) in rendered
    assert (
        "<|start|>functions.search to=assistant<|channel|>commentary<|message|>"
        "result<|end|>"
    ) in rendered


def test_harmony_accepts_legacy_reasoning_content_alias():
    rendered = _render(
        [{"role": "assistant", "content": None, "reasoning_content": "legacy reasoning"}]
    )
    assert "<|channel|>analysis<|message|>legacy reasoning" in rendered


def test_harmony_rejects_conflicting_reasoning_aliases():
    with pytest.raises(ValueError, match="must match"):
        _render(
            [
                {
                    "role": "assistant",
                    "reasoning": "new",
                    "reasoning_content": "legacy",
                }
            ]
        )


def test_harmony_accepts_developer_instructions():
    rendered = _render(
        [
            {"role": "developer", "content": "Use the supplied search tool."},
            {"role": "user", "content": "question"},
        ]
    )
    assert "Use the supplied search tool." in rendered


def test_harmony_partial_text_drop_uses_exact_provenance():
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should search.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "alpha beta gamma"},
        {"role": "assistant", "content": "", "reasoning_content": "continue"},
        {"role": "user", "content": "continue the investigation"},
    ]
    drop_messages = [
        {"role": message["role"], "content": None} for message in messages
    ]
    drop_messages[2]["content"] = "alpha beta"
    manager = TokenizeManager(_GptOssTokenizer())

    result = manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                drop_rule={"type": "text_drop", "drop_messages": drop_messages},
                tools=TOOLS,
            )
        ]
    )[0]

    assert result.message_meta["drop_rule_type"] == "text_drop"
    assert result.drop_position_ranges is not None
    assert result.drop_position_ranges.numel() > 0
    assert result.full_keep_mask is not None
    assert result.full_keep_mask.eq(0).any()
    assert len(result.input_ids) < result.prompt_tokens

    # A strict substring that cuts through a single token is valid, but that
    # boundary-crossing token must remain reusable instead of being dropped.
    drop_messages[2]["content"] = "bet"
    partial = manager.tokenize(
        [
            TokenizeMsg(
                uid=2,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                drop_rule={"type": "text_drop", "drop_messages": drop_messages},
                tools=TOOLS,
            )
        ]
    )[0]
    assert partial.drop_position_ranges is None
    assert len(partial.input_ids) == partial.prompt_tokens
