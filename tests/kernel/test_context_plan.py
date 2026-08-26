from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.core import SamplingParams
from minisgl.kernel.context_plan import first_mask_free_conflict_event
from minisgl.scheduler.prefill import _mask_free_context_reason_reference
from minisgl.scheduler.utils import PendingReq


def _wire(
    *,
    active_positions: list[int],
    events: list[tuple[int, list[tuple[int, int]]]],
    active_cached_len: int,
    effective_event_count: int | None = None,
) -> int | None:
    offsets = [0]
    flat_ranges: list[int] = []
    for _, ranges in events:
        for start, end in ranges:
            flat_ranges.extend((start, end))
        offsets.append(len(flat_ranges) // 2)
    return first_mask_free_conflict_event(
        torch.tensor(active_positions, dtype=torch.int32),
        torch.tensor([event for event, _ in events], dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(flat_ranges, dtype=torch.int32),
        active_cached_len=active_cached_len,
        effective_event_count=(
            len(events) if effective_event_count is None else effective_event_count
        ),
    )


def test_sparse_context_plan_boundaries_and_future_events() -> None:
    events = [(4, [(1, 3)]), (8, [(5, 6)])]

    assert (
        _wire(
            active_positions=[0, 3, 4, 5, 6, 7, 8],
            events=events,
            active_cached_len=2,
            effective_event_count=1,
        )
        is None
    )
    assert (
        _wire(
            active_positions=[0, 3, 4, 5, 6, 7, 8],
            events=events,
            active_cached_len=1,
            effective_event_count=1,
        )
        == 0
    )
    assert (
        _wire(
            active_positions=[0, 3, 4, 5, 6, 7, 8],
            events=events,
            active_cached_len=2,
            effective_event_count=2,
        )
        == 1
    )


def test_one_event_only_needs_its_earliest_drop_start() -> None:
    assert (
        _wire(
            active_positions=[0, 4, 7, 8],
            events=[(8, [(1, 2), (5, 6)])],
            active_cached_len=1,
        )
        == 0
    )


def test_sparse_kernel_matches_full_visibility_reference() -> None:
    rng = random.Random(20260826)
    never = torch.iinfo(torch.int32).max
    for _ in range(250):
        full_len = 128
        event_positions = sorted(rng.sample(range(16, full_len), rng.randint(1, 6)))
        available_starts = list(range(1, 15))
        rng.shuffle(available_starts)
        ranges_by_event: list[list[tuple[int, int]]] = []
        visible_until = torch.full((full_len,), never, dtype=torch.int32)
        keep_mask = torch.ones(full_len, dtype=torch.int32)
        flat_ranges: list[int] = []
        offsets = [0]
        for event_position in event_positions:
            starts = sorted(available_starts.pop() for _ in range(rng.randint(1, 2)))
            ranges = [(start, start + 1) for start in starts]
            ranges_by_event.append(ranges)
            for start, end in ranges:
                visible_until[start:end] = event_position
                keep_mask[start:end] = 0
                flat_ranges.extend((start, end))
            offsets.append(len(flat_ranges) // 2)

        active_positions = torch.nonzero(keep_mask, as_tuple=False).view(-1).to(torch.int32)
        active_cached_len = rng.randrange(len(active_positions))
        full_ids = torch.arange(full_len, dtype=torch.int32)
        req = PendingReq(
            uid=1,
            input_ids=full_ids[active_positions.to(torch.int64)],
            true_positions=active_positions,
            radix_input_ids=full_ids[active_positions.to(torch.int64)].to(torch.int64),
            radix_match_ids=full_ids.to(torch.int64),
            sampling_params=SamplingParams(max_tokens=1),
            prompt_tokens=full_len,
            full_input_ids=full_ids,
            full_token_visible_until=visible_until,
            full_keep_mask=keep_mask,
            use_context_mask=True,
        )
        reference = _mask_free_context_reason_reference(
            req,
            active_cached_len=active_cached_len,
            has_sliding_window=False,
        )
        conflict = first_mask_free_conflict_event(
            active_positions,
            torch.tensor(event_positions, dtype=torch.int32),
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(flat_ranges, dtype=torch.int32),
            active_cached_len=active_cached_len,
            effective_event_count=len(event_positions),
        )
        assert (conflict is None) == (reference is None)


def test_sparse_context_plan_rejects_invalid_wire() -> None:
    with pytest.raises(Exception):
        first_mask_free_conflict_event(
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1, 3], dtype=torch.int32),
            active_cached_len=1,
            effective_event_count=1,
        )
