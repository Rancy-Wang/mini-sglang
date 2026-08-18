from __future__ import annotations

import pytest

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.drop_rules import (
    MessageDropRule,
    TextDropRule,
    parse_drop_rule,
)
from minisgl.tokenizer.template_provenance import TemplateTokenProvenance
from minisgl.tokenizer.tokenize import TokenizeManager


class CharacterChatTokenizer:
    is_fast = True
    special_tokens_map = {}
    name_or_path = "character-test"

    def get_chat_template(self, tools=None):
        return (
            "{% for message in messages %}"
            "<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
            "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
        )

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, chat_template=None, **kwargs
    ):
        text = "".join(
            f"<{message['role']}>{message.get('content') or ''}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def _tokenize(messages, drop_rule):
    return TokenizeManager(CharacterChatTokenizer()).tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_rule=drop_rule,
            )
        ]
    )[0]


def test_public_rule_names_and_message_drop_messages_field():
    messages = [{"role": "user", "content": "hello"}]
    rule = parse_drop_rule(
        {"type": "message_drop", "drop_messages": {"0": [0]}}, messages
    )

    assert isinstance(rule, MessageDropRule)
    assert rule.drop_messages == {0: (0,)}
    assert "schedule" not in rule.to_wire()


def test_text_drop_is_order_aligned_and_defaults_to_earliest_overlap():
    messages = [
        {"role": "system", "content": "keep"},
        {"role": "user", "content": "ababa"},
    ]
    rule = parse_drop_rule(
        {
            "type": "text_drop",
            "drop_messages": [
                {"role": "system", "content": None},
                {"role": "user", "content": "aba"},
            ],
        },
        messages,
    )

    assert isinstance(rule, TextDropRule)
    assert rule.selections[1].spans == ((0, 3),)


def test_text_drop_occurrence_is_all_or_nothing_for_segment_lists():
    messages = [{"role": "user", "content": "one two one two"}]
    payload = {
        "type": "text_drop",
        "drop_messages": [
            {"role": "user", "content": ["one", "two"], "occurrence": [2]}
        ],
    }
    with pytest.raises(ValueError, match="one value per content segment"):
        parse_drop_rule(payload, messages)


@pytest.mark.parametrize(
    "content",
    [None, "", [], [""]],
)
def test_text_drop_explicit_noop_forms(content):
    messages = [{"role": "user", "content": "source"}]
    rule = parse_drop_rule(
        {"type": "text_drop", "drop_messages": [{"role": "user", "content": content}]},
        messages,
    )
    assert isinstance(rule, TextDropRule)
    assert rule.selections == (None,)


def test_text_drop_rejects_role_drift_and_non_subsets_without_echoing_text():
    messages = [{"role": "user", "content": "private-source"}]
    with pytest.raises(ValueError, match="role must match"):
        parse_drop_rule(
            {
                "type": "text_drop",
                "drop_messages": [{"role": "assistant", "content": "private"}],
            },
            messages,
        )
    with pytest.raises(ValueError) as error:
        parse_drop_rule(
            {
                "type": "text_drop",
                "drop_messages": [{"role": "user", "content": "not-present-secret"}],
            },
            messages,
        )
    assert "not-present-secret" not in str(error.value)


def test_full_content_text_drop_promotes_to_exact_message_owner_ranges():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]
    message_result = _tokenize(
        messages,
        {"type": "message_drop", "drop_messages": {"3": [1, 2]}},
    )
    text_result = _tokenize(
        messages,
        {
            "type": "text_drop",
            "drop_messages": [
                {"role": "system", "content": None},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": None},
            ],
        },
    )

    assert message_result.drop_event_positions.tolist() == text_result.drop_event_positions.tolist()
    assert message_result.drop_range_offsets.tolist() == text_result.drop_range_offsets.tolist()
    assert message_result.drop_position_ranges.tolist() == text_result.drop_position_ranges.tolist()
    assert message_result.full_keep_mask.tolist() == text_result.full_keep_mask.tolist()
    assert message_result.true_positions.tolist() == text_result.true_positions.tolist()


def test_partial_text_drop_uses_requested_occurrence_only():
    messages = [{"role": "user", "content": "aba--aba"}]
    result = _tokenize(
        messages,
        {
            "type": "text_drop",
            "drop_messages": [
                {"role": "user", "content": "aba", "occurrence": 2}
            ],
        },
    )
    full_text = "".join(chr(token) for token in result.full_input_ids.tolist())
    dropped_text = "".join(
        full_text[index]
        for index, keep in enumerate(result.full_keep_mask.tolist())
        if not keep
    )
    assert dropped_text == "aba"


def test_partial_selector_never_drops_a_boundary_crossing_token():
    provenance = TemplateTokenProvenance(
        input_ids=[1],
        owners=[0],
        offsets=[(0, len("assistant"))],
        rendered_text="assistant",
        char_owners=[0] * len("assistant"),
        cross_owner_tokens=0,
    )
    with pytest.raises(ValueError, match="contains no complete token"):
        TokenizeManager._token_ranges_for_char_spans(
            provenance,
            owner=0,
            spans=[(0, len("assist"))],
            field="text_drop content",
        )
