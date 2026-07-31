from __future__ import annotations

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import BaseBackendMsg, TokenizeMsg, UserMsg
from minisgl.tokenizer.tokenize import TokenizeManager


class RepeatedTokenChatTokenizer:
    is_fast = True
    special_tokens_map = {}

    def get_chat_template(self, tools=None):
        return (
            "{% for message in messages %}"
            "{% if message.content != '<empty>' %}{{ message.content }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}G{% endif %}"
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
            message["content"] for message in messages if message.get("content") != "<empty>"
        )
        if add_generation_prompt:
            text += "G"
        if not tokenize:
            return text
        return self(text)["input_ids"]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [9 if char == "G" else 7 for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(position, position + 1) for position in range(len(text))]
        return result

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def _tokenize(messages, drop_message=None, *, mode="delta-marker"):
    manager = TokenizeManager(RepeatedTokenChatTokenizer(), radix_drop_key_mode=mode)
    return manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_message=drop_message,
            )
        ]
    )[0]


def test_backend_wire_contains_positions_but_not_drop_message():
    fields = UserMsg.__dataclass_fields__
    assert "drop_message" not in fields
    assert "drop_event_positions" in fields
    assert "drop_range_offsets" in fields
    assert "drop_position_ranges" in fields


def test_message_drop_compiles_to_exact_absolute_position_ranges():
    messages = [{"role": "user", "content": "X"} for _ in range(9)]
    result = _tokenize(messages, {4: [2, 3], 6: [2, 3]})

    assert result.drop_event_positions.tolist() == [5]
    assert result.drop_range_offsets.tolist() == [0, 1]
    assert result.drop_position_ranges.tolist() == [2, 4]
    assert result.full_keep_mask.tolist() == [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    assert result.true_positions.tolist() == [0, 1, 4, 5, 6, 7, 8, 9]
    assert result.input_ids.tolist() == [7, 7, 7, 7, 7, 7, 7, 9]

    sentinel = torch.iinfo(torch.int32).max
    assert result.full_token_visible_until.tolist() == [
        sentinel,
        sentinel,
        5,
        5,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
    ]


def test_noncontiguous_message_ownership_does_not_shift_positions():
    owners = [0, 1, 2, 2, 9, 2, 3]
    query_epoch = [0, 1, 2, 2, 3, 3, 4]
    owner_ranges = TokenizeManager._build_owner_position_ranges(owners)
    plan = TokenizeManager._build_position_drop_plan({3: [2]}, query_epoch, owner_ranges)

    assert owner_ranges[2] == [(2, 4), (5, 6)]
    assert plan.event_positions.tolist() == [6]
    assert plan.range_offsets.tolist() == [0, 2]
    assert plan.position_ranges.tolist() == [2, 4, 5, 6]

    keep = torch.ones(len(owners), dtype=torch.bool)
    for start, end in plan.position_ranges.view(-1, 2).tolist():
        keep[start:end] = False
    assert torch.arange(len(owners), dtype=torch.int32)[keep].tolist() == [0, 1, 4, 6]


def test_more_than_128_drop_events_have_no_round_limit():
    count = 140
    owners = list(range(count + 1))
    query_epoch = list(range(count + 1))
    drop_message = {event: [event - 1] for event in range(1, count)}
    plan = TokenizeManager._build_position_drop_plan(
        drop_message,
        query_epoch,
        TokenizeManager._build_owner_position_ranges(owners),
    )

    assert len(plan.event_positions) == count - 1
    assert len(plan.position_ranges) == 2 * (count - 1)
    assert plan.event_positions[-1].item() == count
    assert plan.position_ranges[-2:].tolist() == [count - 2, count - 1]


def test_position_drop_backend_wire_round_trips_only_one_dimensional_tensors():
    result = _tokenize(
        [{"role": "user", "content": "X"} for _ in range(5)],
        {3: [1, 2]},
    )
    msg = UserMsg(
        uid=1,
        input_ids=result.input_ids,
        true_positions=result.true_positions,
        radix_input_ids=result.radix_input_ids,
        radix_match_ids=result.radix_match_ids,
        sampling_params=SamplingParams(max_tokens=1),
        drop_event_positions=result.drop_event_positions,
        drop_range_offsets=result.drop_range_offsets,
        drop_position_ranges=result.drop_position_ranges,
    )

    decoded = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(decoded, UserMsg)
    assert decoded.drop_position_ranges.ndim == 1
    assert torch.equal(decoded.drop_event_positions, msg.drop_event_positions)
    assert torch.equal(decoded.drop_range_offsets, msg.drop_range_offsets)
    assert torch.equal(decoded.drop_position_ranges, msg.drop_position_ranges)


def test_no_drop_keeps_linear_token_stream_and_has_no_position_metadata():
    messages = [{"role": "user", "content": "X"} for _ in range(4)]
    result = _tokenize(messages)

    assert result.drop_event_positions is None
    assert result.drop_position_ranges is None
    assert result.full_token_visible_until is None
    assert torch.equal(result.input_ids.to(torch.int64), result.radix_match_ids)


@pytest.mark.parametrize("mode", ["bitmask", "symbol"])
def test_legacy_radix_modes_reject_drop_instead_of_using_message_semantics(mode):
    messages = [{"role": "user", "content": "X"} for _ in range(2)]
    with pytest.raises(ValueError, match="Token-position Drop requires"):
        _tokenize(messages, {1: [0]}, mode=mode)
