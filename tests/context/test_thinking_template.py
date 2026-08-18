from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.drop_rules import ThinkingDropRule, parse_drop_rule
from minisgl.tokenizer.thinking_template import prepare_thinking_template
from minisgl.tokenizer.tokenize import TokenizeManager
from transformers import AutoTokenizer


QWEN3_17B = Path("/share/public/public_models/Qwen3-1.7B")


class StrippingTemplateTokenizer:
    name_or_path = "guarded-qwen-test"
    special_tokens_map = {}
    chat_template = (
        "{% set ns = namespace(last_query_index=messages|length - 1) %}"
        "{% for message in messages %}"
        "{% if message.role == 'assistant' %}"
        "{% set reasoning_content = message.reasoning_content|default('') %}"
        "{% if loop.index0 > ns.last_query_index %}"
        "{% if reasoning_content %}<think>{{ reasoning_content }}</think>{% endif %}"
        "{{ message.content }}"
        "{% else %}{{ message.content }}{% endif %}"
        "{% else %}{{ message.content }}{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}GEN{% endif %}"
    )

    def get_chat_template(self, tools=None):
        return self.chat_template

    def apply_chat_template(self, messages, *, chat_template=None, **kwargs):
        template = Environment().from_string(chat_template or self.chat_template)
        return template.render(messages=messages, **kwargs)


def test_thinking_rule_accepts_structured_or_one_leading_block():
    structured = parse_drop_rule(
        {"type": "thinking_drop"},
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "reasoning_content": "private", "content": "final"},
        ],
    )
    inline = parse_drop_rule(
        {"type": "thinking_drop"},
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>private</think>final"},
        ],
    )
    assert isinstance(structured, ThinkingDropRule)
    assert structured.thinking_by_message == {1: "private"}
    assert inline.thinking_by_message == {1: "private"}


def test_thinking_rule_rejects_conflicting_or_ambiguous_sources():
    with pytest.raises(ValueError, match="cannot provide both"):
        parse_drop_rule(
            {"type": "thinking_drop"},
            [
                {
                    "role": "assistant",
                    "reasoning_content": "one",
                    "content": "<think>two</think>final",
                }
            ],
        )
    with pytest.raises(ValueError, match="cannot provide both"):
        parse_drop_rule(
            {"type": "thinking_drop"},
            [
                {
                    "role": "assistant",
                    "reasoning_content": "same",
                    "content": "<think>same</think>final",
                }
            ],
        )
    with pytest.raises(ValueError, match="nested or multiple"):
        parse_drop_rule(
            {"type": "thinking_drop"},
            [{"role": "assistant", "content": "<think>one</think><think>two</think>"}],
        )


def test_qwen_guard_adapter_is_request_local_and_preserves_nonthinking_output():
    tokenizer = StrippingTemplateTokenizer()
    before = tokenizer.chat_template
    plan = prepare_thinking_template(tokenizer, tools=None)

    assert plan.capability == "qwen_guard_adapter"
    assert plan.chat_template is not None
    assert plan.template_kwargs == {"preserve_thinking_history": True}
    assert tokenizer.chat_template == before


def test_unknown_stripping_template_fails_closed():
    tokenizer = StrippingTemplateTokenizer()
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    with pytest.raises(ValueError, match="thinking_history_not_preservable"):
        prepare_thinking_template(tokenizer, tools=None)


@pytest.mark.skipif(not QWEN3_17B.exists(), reason="Qwen3-1.7B tokenizer is unavailable")
def test_qwen_thinking_drop_retains_full_reasoning_but_drops_its_active_kv():
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_17B, local_files_only=True)
    manager = TokenizeManager(tokenizer)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compute 15 + 27 and reason privately."},
        {
            "role": "assistant",
            "reasoning_content": "15 plus 27 equals 42. I should verify the arithmetic.",
            "content": "The answer is 42.",
        },
        {"role": "user", "content": "Now multiply the answer by 3."},
    ]
    result = manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_rule={"type": "thinking_drop"},
            )
        ]
    )[0]

    full = tokenizer.decode(result.full_input_ids.tolist())
    active = tokenizer.decode(result.input_ids.tolist())
    assert "15 plus 27 equals 42" in full
    assert "15 plus 27 equals 42" not in active
    assert "The answer is 42." in active
    assert result.message_meta["thinking_template_capability"] in {
        "native",
        "qwen_guard_adapter",
    }
    assert result.drop_event_positions.numel() == 1
