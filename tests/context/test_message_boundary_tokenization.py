from __future__ import annotations

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.template_provenance import build_template_token_provenance
from minisgl.tokenizer.tokenize import TokenizeManager


class ChunkedBoundaryTokenizer:
    is_fast = True
    special_tokens_map = {}

    def __init__(self):
        self.encode_calls = 0

    def get_chat_template(self, tools=None):
        return (
            "{% for message in messages %}"
            "<im_start>{{ message.role }}\n{{ message.content }}<im_end>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<im_start>assistant\n{% endif %}"
        )

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        **kwargs,
    ):
        text = "".join(
            f'<im_start>{message["role"]}\n{message["content"]}<im_end>\n'
            for message in messages
        )
        if add_generation_prompt:
            text += "<im_start>assistant\n"
        return self(text)["input_ids"] if tokenize else text

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        self.encode_calls += 1
        offsets = [
            (start, min(start + 3, len(text))) for start in range(0, len(text), 3)
        ]
        result = {
            "input_ids": [
                sum(text[start:end].encode("utf-8")) + 1 for start, end in offsets
            ]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def _message_drop(tokenizer, messages):
    return TokenizeManager(tokenizer).tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_message={len(messages) - 1: [1]},
            )
        ]
    )[0]


@pytest.mark.parametrize("message_count", [8, 32, 128])
def test_message_drop_uses_one_canonical_tokenizer_call(message_count):
    tokenizer = ChunkedBoundaryTokenizer()
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"M{index}"}
        for index in range(message_count)
    ]
    canonical_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    expected_ids = tokenizer(canonical_text, add_special_tokens=False)["input_ids"]
    tokenizer.encode_calls = 0

    result = _message_drop(tokenizer, messages)

    assert tokenizer.encode_calls == 1
    assert result.full_input_ids.tolist() == expected_ids
    assert result.message_meta["unstable_rounds"] == 0


def test_cross_message_token_is_owned_by_the_previous_message():
    tokenizer = ChunkedBoundaryTokenizer()
    messages = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    no_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    with_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    provenance = build_template_token_provenance(
        tokenizer,
        messages,
        canonical_text=with_gen,
        canonical_no_generation_text=no_gen,
        expected_input_ids=None,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=None,
    )

    assert provenance.cross_owner_tokens > 0
    for owner, (start, end) in zip(provenance.owners, provenance.offsets):
        char_owners = provenance.char_owners[start:end]
        if char_owners and len(set(char_owners)) > 1:
            assert owner == char_owners[0]


def test_render_only_boundaries_do_not_change_tokens_or_absolute_positions():
    tokenizer = ChunkedBoundaryTokenizer()
    messages = [
        {"role": "system", "content": "keep"},
        {"role": "user", "content": "drop-me"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "continue"},
    ]
    result = _message_drop(tokenizer, messages)
    dropped = ~result.full_keep_mask.to(torch.bool)

    assert result.true_positions.tolist() == [
        position for position, is_dropped in enumerate(dropped.tolist()) if not is_dropped
    ]
    assert result.input_ids.tolist() == result.full_input_ids[~dropped].tolist()
    assert max(result.true_positions.tolist()) < len(result.full_input_ids)
