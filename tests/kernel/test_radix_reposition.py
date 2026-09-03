from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.kernel.radix import (
    fast_compare_radix_records,
    fast_compare_retry_radix_records,
    fast_compare_retry_radix_records_plan,
    radix_record_compare_backend,
    radix_record_edge_equal,
    radix_record_edge_hash,
)
from minisgl.kernel.radix_reposition import (
    DELTA_KIND,
    REPOSITION_KIND,
    TOKEN_KIND,
    RadixRepositionInput,
    compile_radix_reposition_layout,
    compile_radix_reposition_layout_batch,
)


def _compile(
    token_ids: list[int],
    drops: list[tuple[int, list[tuple[int, int]]]],
    repositions: list[int],
):
    flat_ranges: list[int] = []
    range_offsets = [0]
    for _, ranges in drops:
        for start, end in ranges:
            flat_ranges.extend((start, end))
        range_offsets.append(len(flat_ranges) // 2)
    return compile_radix_reposition_layout(
        torch.tensor(token_ids, dtype=torch.int64),
        torch.tensor([offset for offset, _ in drops], dtype=torch.int32),
        torch.tensor(range_offsets, dtype=torch.int32),
        torch.tensor(flat_ranges, dtype=torch.int32),
        torch.tensor(repositions, dtype=torch.int32),
        torch.tensor([boundary + 1 for boundary in repositions], dtype=torch.int32),
    )


def _reference(
    token_ids: list[int],
    drops: list[tuple[int, list[tuple[int, int]]]],
    repositions: list[int],
):
    drop_by_offset = {offset: ranges for offset, ranges in drops}
    reposition_by_offset = {
        boundary + 1: (idx, boundary) for idx, boundary in enumerate(repositions)
    }
    keep = [False] * len(token_ids)
    positions = [-1] * len(token_ids)
    birth_positions = [-1] * len(token_ids)
    birth_stages = [-1] * len(token_ids)
    repos_info = [-1] * len(token_ids)
    materialized_stage = [-1] * len(token_ids)
    transition_offsets = [0]
    transition_raw_tokens: list[int] = []
    transition_old_positions: list[int] = []
    transition_new_positions: list[int] = []
    active: list[int] = []
    effective = [False] * len(repositions)
    effective_stages = [-1] * len(repositions)
    ignored = [False] * len(repositions)
    current_reposition = -1
    next_position = 0
    stage = 0

    for insertion in range(len(token_ids) + 1):
        drop = drop_by_offset.get(insertion)
        if drop is not None:
            dropped = {token for start, end in drop for token in range(start, end)}
            active = [token for token in active if token not in dropped]
            for token in dropped:
                keep[token] = False

        reposition = reposition_by_offset.get(insertion)
        if reposition is not None:
            reposition_idx, boundary = reposition
            if not active:
                raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
            changed = [token for rank, token in enumerate(active) if positions[token] != rank]
            if not changed:
                ignored[reposition_idx] = True
            else:
                stage += 1
                effective[reposition_idx] = True
                effective_stages[reposition_idx] = stage
                for rank, token in enumerate(active):
                    if positions[token] == rank:
                        continue
                    transition_raw_tokens.append(token)
                    transition_old_positions.append(positions[token])
                    transition_new_positions.append(rank)
                    positions[token] = rank
                    repos_info[token] = boundary
                    materialized_stage[token] = stage
                transition_offsets.append(len(transition_raw_tokens))
                current_reposition = boundary
                next_position = len(active)

        if insertion == len(token_ids):
            continue
        keep[insertion] = True
        positions[insertion] = next_position
        birth_positions[insertion] = next_position
        birth_stages[insertion] = stage
        repos_info[insertion] = current_reposition
        materialized_stage[insertion] = stage
        active.append(insertion)
        next_position += 1

    records: list[list[int]] = []
    key_to_token: list[int] = []
    token_to_key: list[int] = [-1] * len(token_ids)
    virtual: list[bool] = []
    drop_event_to_key: list[int] = [-1] * len(drops)
    drop_index = 0
    reposition_index = 0
    for insertion in range(len(token_ids) + 1):
        if drop_index < len(drops) and drops[drop_index][0] == insertion:
            drop_event_to_key[drop_index] = len(records)
            for start, end in drops[drop_index][1]:
                records.append([DELTA_KIND, -start - 1, -end - 1, -1])
                key_to_token.append(-1)
                virtual.append(True)
            drop_index += 1
        if reposition_index < len(repositions) and repositions[reposition_index] + 1 == insertion:
            if effective[reposition_index]:
                records.append([REPOSITION_KIND, repositions[reposition_index], -1, -1])
                key_to_token.append(-1)
                virtual.append(True)
            reposition_index += 1
        if insertion == len(token_ids):
            continue
        token_to_key[insertion] = len(records)
        records.append(
            [TOKEN_KIND, token_ids[insertion], repos_info[insertion], positions[insertion]]
        )
        key_to_token.append(insertion)
        virtual.append(False)

    return {
        "records": records,
        "virtual_mask": virtual,
        "key_to_token": key_to_token,
        "token_to_key": token_to_key,
        "positions": positions,
        "repos_info": repos_info,
        "keep_mask": keep,
        "materialized_stage": materialized_stage,
        "birth_positions": birth_positions,
        "birth_stages": birth_stages,
        "transition_offsets": transition_offsets,
        "transition_raw_tokens": transition_raw_tokens,
        "transition_old_positions": transition_old_positions,
        "transition_new_positions": transition_new_positions,
        "effective_reposition_stages": effective_stages,
        "drop_event_to_key": drop_event_to_key,
        "effective_repositions": effective,
        "ignored_repositions": ignored,
        "next_position": next_position,
        "current_reposition": current_reposition,
    }


def _assert_layout_matches_reference(layout, expected) -> None:
    tensor_fields = (
        "records",
        "virtual_mask",
        "key_to_token",
        "token_to_key",
        "positions",
        "repos_info",
        "keep_mask",
        "materialized_stage",
        "birth_positions",
        "birth_stages",
        "transition_offsets",
        "transition_raw_tokens",
        "transition_old_positions",
        "transition_new_positions",
        "effective_reposition_stages",
        "drop_event_to_key",
        "effective_repositions",
        "ignored_repositions",
    )
    for field in tensor_fields:
        assert getattr(layout, field).tolist() == expected[field], field
    assert layout.next_position == expected["next_position"]
    assert layout.current_reposition == expected["current_reposition"]


def test_plan_counterexample_preserves_early_reposition_record() -> None:
    layout = _compile(
        [0, 1, 2, 3],
        [(3, [(1, 2)]), (4, [(0, 1)])],
        [2, 3],
    )
    assert layout.records.tolist() == [
        [TOKEN_KIND, 0, -1, 0],
        [TOKEN_KIND, 1, -1, 1],
        [TOKEN_KIND, 2, 3, 0],
        [DELTA_KIND, -2, -3, -1],
        [REPOSITION_KIND, 2, -1, -1],
        [TOKEN_KIND, 3, 3, 1],
        [DELTA_KIND, -1, -2, -1],
        [REPOSITION_KIND, 3, -1, -1],
    ]


def test_same_boundary_applies_drop_before_reposition() -> None:
    layout = _compile(
        list(range(6)),
        [(6, [(0, 2)])],
        [5],
    )
    assert layout.records[-2:].tolist() == [
        [DELTA_KIND, -1, -3, -1],
        [REPOSITION_KIND, 5, -1, -1],
    ]
    assert layout.positions.tolist() == [0, 1, 0, 1, 2, 3]
    assert layout.repos_info.tolist() == [-1, -1, 5, 5, 5, 5]
    assert layout.birth_positions.tolist() == [0, 1, 2, 3, 4, 5]
    assert layout.birth_stages.tolist() == [0, 0, 0, 0, 0, 0]
    assert layout.transition_offsets.tolist() == [0, 4]
    assert layout.transition_raw_tokens.tolist() == [2, 3, 4, 5]
    assert layout.transition_old_positions.tolist() == [2, 3, 4, 5]
    assert layout.transition_new_positions.tolist() == [0, 1, 2, 3]


def test_noop_reposition_is_ignored_without_metadata_change() -> None:
    layout = _compile([10, 11], [], [1])
    assert layout.records.tolist() == [
        [TOKEN_KIND, 10, -1, 0],
        [TOKEN_KIND, 11, -1, 1],
    ]
    assert layout.effective_repositions.tolist() == [False]
    assert layout.effective_reposition_stages.tolist() == [-1]
    assert layout.transition_offsets.tolist() == [0]
    assert layout.ignored_repositions.tolist() == [True]
    assert layout.current_reposition == -1


def test_reposition_rejects_empty_active_set() -> None:
    with pytest.raises(ValueError, match="has no active tokens"):
        _compile([10, 11], [(2, [(0, 2)])], [1])


def test_cpu_compiler_matches_independent_random_state_machine() -> None:
    rng = random.Random(20260902)
    for case in range(80):
        token_count = rng.randint(8, 96)
        event_offsets = sorted(rng.sample(range(2, token_count + 1), rng.randint(1, 6)))
        active: list[int] = []
        drops: list[tuple[int, list[tuple[int, int]]]] = []
        repositions: list[int] = []
        offset_set = set(event_offsets)
        for insertion in range(token_count + 1):
            if insertion in offset_set and len(active) > 1 and rng.random() < 0.85:
                drop_count = rng.randint(1, min(3, len(active) - 1))
                dropped = sorted(rng.sample(active, drop_count))
                dropped_set = set(dropped)
                active = [token for token in active if token not in dropped_set]
                ranges: list[tuple[int, int]] = []
                for token in dropped:
                    if ranges and ranges[-1][1] == token:
                        ranges[-1] = (ranges[-1][0], token + 1)
                    else:
                        ranges.append((token, token + 1))
                drops.append((insertion, ranges))
            if insertion in offset_set and active and rng.random() < 0.8:
                repositions.append(insertion - 1)
            if insertion < token_count:
                active.append(insertion)

        token_ids = [rng.randrange(0, 200_000) for _ in range(token_count)]
        layout = _compile(token_ids, drops, repositions)
        _assert_layout_matches_reference(
            layout,
            _reference(token_ids, drops, repositions),
        )


def test_structured_comparators_use_exact_and_retry_semantics() -> None:
    cached = torch.tensor(
        [
            [TOKEN_KIND, 10, 2, 4],
            [TOKEN_KIND, 11, 2, 5],
            [DELTA_KIND, -1, -3, -1],
            [REPOSITION_KIND, 8, -1, -1],
        ],
        dtype=torch.int32,
    )
    target = cached.clone()
    target[0, 2:] = torch.tensor([9, 0], dtype=torch.int32)
    target[1, 2:] = torch.tensor([9, 1], dtype=torch.int32)
    assert fast_compare_radix_records(cached, target) == 0
    assert fast_compare_retry_radix_records(cached, target) == len(cached)

    target[2, 2] = -4
    assert fast_compare_retry_radix_records(cached, target) == 2
    target[2, 2] = -3
    target[3, 1] = 9
    assert fast_compare_retry_radix_records(cached, target) == 3


def test_delta_child_edge_hashes_and_compares_the_complete_range_block() -> None:
    first = torch.tensor(
        [
            [DELTA_KIND, -1, -2, -1],
            [DELTA_KIND, -4, -5, -1],
            [TOKEN_KIND, 10, -1, 0],
        ],
        dtype=torch.int32,
    )
    same_edge = first.clone()
    same_edge[2, 1] = 99
    different_edge = first.clone()
    different_edge[1, 1:3] = torch.tensor([-3, -4], dtype=torch.int32)

    assert radix_record_edge_equal(first, same_edge)
    assert radix_record_edge_hash(first) == radix_record_edge_hash(same_edge)
    assert not radix_record_edge_equal(first, different_edge)
    assert radix_record_edge_hash(first) != radix_record_edge_hash(different_edge)


def test_retry_comparator_emits_only_changed_token_pages() -> None:
    cached = torch.tensor(
        [
            [TOKEN_KIND, 10, -1, 0],
            [DELTA_KIND, -1, -2, -1],
            [TOKEN_KIND, 11, 2, 3],
            [TOKEN_KIND, 12, 2, 4],
        ],
        dtype=torch.int32,
    )
    target = cached.clone()
    target[2, 2:] = torch.tensor([8, 1], dtype=torch.int32)
    target[3, 2:] = torch.tensor([8, 4], dtype=torch.int32)
    matched, plan = fast_compare_retry_radix_records_plan(
        cached,
        target,
        torch.tensor([0, -1, 1, 2], dtype=torch.int64),
        torch.tensor([3, -1, 7, 9], dtype=torch.int64),
    )

    assert matched == len(cached)
    assert plan.tolist() == [[1, 7, 3, 1]]
    assert radix_record_compare_backend() in {"portable", "neon", "avx2", "avx512"}


def test_bounded_batch_compiler_preserves_request_order() -> None:
    inputs = tuple(
        RadixRepositionInput(
            token_ids=torch.tensor([base, base + 1, base + 2], dtype=torch.int64),
            drop_insert_offsets=torch.tensor([3], dtype=torch.int32),
            drop_range_offsets=torch.tensor([0, 1], dtype=torch.int32),
            drop_ranges=torch.tensor([0, 1], dtype=torch.int32),
            reposition_raw_boundaries=torch.tensor([2], dtype=torch.int32),
            reposition_insert_offsets=torch.tensor([3], dtype=torch.int32),
        )
        for base in (10, 20, 30, 40)
    )

    layouts = compile_radix_reposition_layout_batch(inputs, max_workers=2)

    assert [layout.records[0, 1].item() for layout in layouts] == [10, 20, 30, 40]
    assert all(layout.next_position == 2 for layout in layouts)
    assert all(layout.transition_offsets.tolist() == [0, 2] for layout in layouts)
    assert all(layout.compile_ns > 0 for layout in layouts)


def test_compiler_rejects_token_id_narrowing() -> None:
    with pytest.raises(ValueError, match="int32"):
        _compile([1 << 31], [], [])
