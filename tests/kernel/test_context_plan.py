from __future__ import annotations

import itertools
import random
from bisect import bisect_left

import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.core import SamplingParams
from minisgl.kernel.context_plan import (
    first_mask_free_conflict_event,
    try_build_context_full_plan,
    try_build_context_sliding_plan,
    try_build_occurrence_capacity_index,
    try_build_occurrence_sliding_plan,
)
from minisgl.scheduler.prefill import PrefillAdder, _mask_free_context_reason_reference
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
        visible_until = torch.full((full_len,), never, dtype=torch.int32)
        keep_mask = torch.ones(full_len, dtype=torch.int32)
        flat_ranges: list[int] = []
        offsets = [0]
        for event_position in event_positions:
            starts = sorted(available_starts.pop() for _ in range(rng.randint(1, 2)))
            ranges: list[tuple[int, int]] = []
            for start in starts:
                if ranges and start <= ranges[-1][1]:
                    ranges[-1] = (ranges[-1][0], start + 1)
                else:
                    ranges.append((start, start + 1))
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
            raw_positions=active_positions,
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


def test_context_sliding_plan_matches_direct_reference() -> None:
    rng = random.Random(20260910)
    never = torch.iinfo(torch.int32).max
    for _ in range(100):
        full_token_count = rng.randint(8, 128)
        kept_raw = [raw_position for raw_position in range(full_token_count) if rng.random() >= 0.2]
        if len(kept_raw) < 2:
            kept_raw = [0, full_token_count - 1]
        raw = torch.tensor(kept_raw, dtype=torch.int32)
        key_length = len(raw)
        true_positions = torch.tensor(
            list(itertools.accumulate(rng.randint(1, 4) for _ in range(key_length))),
            dtype=torch.int32,
        )
        visible_until = torch.full((full_token_count,), never, dtype=torch.int32)
        for raw_position in range(full_token_count - 1):
            if rng.random() < 0.2:
                visible_until[raw_position] = rng.randint(raw_position + 1, full_token_count)
        query_start = rng.randrange(key_length)
        query_count = key_length - query_start
        window_left = rng.randint(0, 12)

        result = try_build_context_sliding_plan(
            visible_until,
            raw,
            true_positions,
            query_start=query_start,
            query_length=query_count,
            sliding_window=window_left,
        )

        assert result is not None
        offsets, keys = result
        expected_offsets = [0]
        expected_keys = []
        for query in range(query_start, key_length):
            threshold = int(true_positions[query]) - window_left
            row = [
                key
                for key in range(query)
                if int(visible_until[int(raw[key])]) > int(raw[query])
                and int(true_positions[key]) >= threshold
            ]
            row.append(query)
            expected_keys.extend(row)
            expected_offsets.append(len(expected_keys))
        assert offsets.tolist() == expected_offsets
        assert keys.tolist() == expected_keys


def test_context_full_plan_matches_direct_reference() -> None:
    rng = random.Random(20260912)
    never = torch.iinfo(torch.int32).max
    for _ in range(100):
        full_token_count = rng.randint(8, 128)
        kept_raw = [raw_position for raw_position in range(full_token_count) if rng.random() >= 0.2]
        if len(kept_raw) < 2:
            kept_raw = [0, full_token_count - 1]
        raw = torch.tensor(kept_raw, dtype=torch.int32)
        key_length = len(raw)
        visible_until = torch.full((full_token_count,), never, dtype=torch.int32)
        for raw_position in range(full_token_count - 1):
            if rng.random() < 0.2:
                visible_until[raw_position] = rng.randint(raw_position + 1, full_token_count)
        query_start = rng.randrange(key_length)
        query_count = key_length - query_start

        result = try_build_context_full_plan(
            visible_until,
            raw,
            query_start=query_start,
            query_length=query_count,
        )

        assert result is not None
        query_lengths, offsets, keys = result
        boundaries = {query_start, key_length}
        query_raw = kept_raw[query_start:key_length]
        for expiry in {int(visible_until[position]) for position in kept_raw[:key_length]}:
            local_boundary = bisect_left(query_raw, expiry)
            if 0 < local_boundary < query_count:
                boundaries.add(query_start + local_boundary)
        ordered = sorted(boundaries)
        expected_query_lengths = []
        expected_offsets = [0]
        expected_keys = []
        for start, end in zip(ordered, ordered[1:]):
            prefix = [
                key for key in range(start) if int(visible_until[kept_raw[key]]) > kept_raw[start]
            ]
            row = prefix + list(range(start, end))
            expected_query_lengths.append(end - start)
            expected_keys.extend(row)
            expected_offsets.append(len(expected_keys))
        assert query_lengths.tolist() == expected_query_lengths
        assert offsets.tolist() == expected_offsets
        assert keys.tolist() == expected_keys


def test_occurrence_sliding_plan_matches_direct_reference() -> None:
    rng = random.Random(20260911)
    for _ in range(100):
        key_length = rng.randint(8, 128)
        raw = torch.arange(key_length, dtype=torch.int32)
        positions = torch.tensor(
            list(itertools.accumulate(rng.randint(1, 4) for _ in range(key_length))),
            dtype=torch.int32,
        )
        query_start = rng.randrange(key_length)
        query_end = rng.randint(query_start + 1, key_length)
        window_left = rng.randint(0, 12)
        occurrence_base = rng.randint(0, 1000)
        result = try_build_occurrence_sliding_plan(
            raw,
            positions,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([key_length], dtype=torch.int32),
            torch.tensor([0, key_length], dtype=torch.int32),
            raw,
            positions[:query_end],
            cached_len=query_start,
            device_len=query_end,
            initial_cached_len=query_start,
            sliding_window=window_left,
            occurrence_base=occurrence_base,
        )

        assert result is not None
        offsets, keys, cached_positions = result
        expected_offsets = [0]
        expected_keys = []
        expected_cached = set()
        for query in range(query_start, query_end):
            threshold = int(positions[query]) - window_left
            row = [key for key in range(query + 1) if int(positions[key]) >= threshold]
            expected_keys.extend(occurrence_base + key for key in row)
            expected_offsets.append(len(expected_keys))
            expected_cached.update(key for key in row if key < query_start)
        assert offsets.tolist() == expected_offsets
        assert keys.tolist() == expected_keys
        assert cached_positions.tolist() == sorted(expected_cached)


def test_occurrence_sliding_plan_accepts_window_starting_at_cached_len() -> None:
    raw = torch.arange(4, dtype=torch.int32)
    positions = raw.clone()

    result = try_build_occurrence_sliding_plan(
        raw,
        positions,
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        raw,
        positions,
        cached_len=2,
        device_len=4,
        initial_cached_len=2,
        sliding_window=2,
        occurrence_base=7,
    )

    assert result is not None
    offsets, keys, cached_positions = result
    assert offsets.tolist() == [0, 3, 6]
    assert keys.tolist() == [7, 8, 9, 8, 9, 10]
    assert cached_positions.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("chunk_start", "terminal_owned", "source_positions"),
    [
        (0, [False, False, False, False, False], []),
        (2, [True, False, False, False, False], []),
        (2, [False, False, False, False, False], [-1, 0]),
    ],
)
def test_occurrence_capacity_index_matches_reference_for_every_endpoint(
    chunk_start: int,
    terminal_owned: list[bool],
    source_positions: list[int],
) -> None:
    occurrence_raw = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.int32)
    occurrence_positions = torch.tensor([0, 1, 2, 3, 4, -1, 0, 1], dtype=torch.int32)
    birth = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    terminal = torch.tensor([5, 6, 7, 3, 4], dtype=torch.int32)
    segment_starts = torch.tensor([0, 2], dtype=torch.int32)
    segment_ends = torch.tensor([2, 5], dtype=torch.int32)
    segment_offsets = torch.tensor([0, 2, 7], dtype=torch.int32)
    segment_keys = torch.tensor([0, 1, 5, 6, 2, 3, 4], dtype=torch.int32)
    owned = torch.tensor(terminal_owned, dtype=torch.bool)
    source = torch.tensor(source_positions, dtype=torch.int32)
    final_keep = torch.tensor([False, True, True, False, True], dtype=torch.bool)
    req = PendingReq(
        uid=7,
        input_ids=torch.arange(5, dtype=torch.int32),
        true_positions=torch.arange(5, dtype=torch.int32),
        raw_positions=torch.arange(5, dtype=torch.int32),
        radix_input_ids=torch.arange(5, dtype=torch.int32),
        radix_match_ids=torch.arange(5, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
        prompt_tokens=5,
        full_keep_mask=final_keep.to(torch.int32),
        occurrence_raw_tokens=occurrence_raw,
        occurrence_positions=occurrence_positions,
        occurrence_birth_indices=birth,
        occurrence_terminal_indices=terminal,
        occurrence_segment_query_starts=segment_starts,
        occurrence_segment_query_ends=segment_ends,
        occurrence_segment_key_offsets=segment_offsets,
        occurrence_segment_key_indices=segment_keys,
    )

    result = try_build_occurrence_capacity_index(
        occurrence_raw,
        occurrence_positions,
        birth,
        terminal,
        segment_starts,
        segment_ends,
        segment_offsets,
        segment_keys,
        owned,
        final_keep,
        source,
        chunk_start=chunk_start,
        max_chunk_end=5,
        output_len=req.output_len,
    )

    assert result is not None
    first_required_end, current, persistent, future = result
    for endpoint in range(chunk_start + 1, 6):
        expected_required, expected_current, expected_persistent, expected_future = (
            PrefillAdder._occurrence_capacity_for_chunk(
                req,
                start=chunk_start,
                end=endpoint,
                terminal_owned=owned,
                initial_source_positions=source,
            )
        )
        actual_required = (
            torch.nonzero(
                first_required_end <= endpoint,
                as_tuple=False,
            )
            .view(-1)
            .to(torch.int64)
        )
        index = endpoint - chunk_start - 1
        assert torch.equal(actual_required, expected_required)
        assert int(current[index]) == expected_current
        assert int(persistent[index]) == expected_persistent
        assert int(future[index]) == expected_future
