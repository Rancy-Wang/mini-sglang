from __future__ import annotations

from bisect import bisect_right

import pytest
import torch

from minisgl.core import build_context_visibility_mask_reference
from minisgl.kernel.context_mask import (
    build_context_visibility_mask,
    build_context_visibility_mask_packed,
)


def _message_reference(owners, query_epoch, visible_until):
    length = len(owners)
    result = torch.zeros((length, length), dtype=torch.bool)
    for query_position in range(length):
        for key_position in range(length):
            result[query_position, key_position] = (
                key_position <= query_position
                and query_epoch[query_position] <= visible_until[owners[key_position]]
            )
    return result


def _compile_token_visible_until(owners, query_epoch, message_visible_until):
    sentinel = torch.iinfo(torch.int32).max
    result = torch.full((len(owners),), sentinel, dtype=torch.int32)
    for key_position, owner in enumerate(owners):
        event_message = message_visible_until[owner]
        if event_message < sentinel:
            result[key_position] = bisect_right(query_epoch, event_message)
    return result


def test_token_position_mask_is_exactly_equivalent_to_message_mask():
    owners = [0, 0, 1, 1, 2, 2, 3, 3, 4]
    query_epoch = [0, 0, 1, 1, 2, 2, 3, 4, 5]
    sentinel = torch.iinfo(torch.int32).max
    message_visible_until = [2, sentinel, 3, sentinel, sentinel, sentinel]
    token_visible_until = _compile_token_visible_until(
        owners, query_epoch, message_visible_until
    )

    expected = _message_reference(owners, query_epoch, message_visible_until)
    actual = build_context_visibility_mask_reference(token_visible_until)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_packed_and_unpacked_masks_match_token_position_oracle():
    visible_until = torch.tensor(
        [
            torch.iinfo(torch.int32).max,
            4,
            4,
            torch.iinfo(torch.int32).max,
            7,
            7,
            torch.iinfo(torch.int32).max,
            torch.iinfo(torch.int32).max,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    query_start = 2
    query_length = 6
    key_length = 8
    oracle = build_context_visibility_mask_reference(
        visible_until.cpu(),
        query_positions=torch.arange(query_start, query_start + query_length),
        key_positions=torch.arange(key_length),
    ).reshape(-1)

    unpacked = build_context_visibility_mask(
        visible_until,
        query_start=query_start,
        query_length=query_length,
        key_length=key_length,
    ).cpu()
    assert torch.equal(unpacked.to(torch.bool), oracle)

    packed = build_context_visibility_mask_packed(
        visible_until,
        query_start=query_start,
        query_length=query_length,
        key_length=key_length,
    ).cpu()
    decoded = torch.tensor(
        [
            bool((int(packed[bit // 8].item()) >> (bit % 8)) & 1)
            for bit in range(query_length * key_length)
        ],
        dtype=torch.bool,
    )
    assert torch.equal(decoded, oracle)
