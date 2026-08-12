from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.radix_delta import (
    DeltaMarkerRegistry,
    DeltaRadixLayout,
    inject_delta_markers,
)


@pytest.fixture(autouse=True)
def _unit_page_context():
    previous_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = core.Context(page_size=1)
    try:
        yield
    finally:
        core._GLOBAL_CTX = previous_ctx


def _wire(event: int, ranges: list[tuple[int, int]]):
    return (
        torch.tensor([event], dtype=torch.int32),
        torch.tensor([0, len(ranges)], dtype=torch.int32),
        torch.tensor(ranges, dtype=torch.int32).reshape(-1),
    )


def _layout(
    full_ids: torch.Tensor,
    registry: DeltaMarkerRegistry,
    *,
    event: int,
    ranges: list[tuple[int, int]],
) -> tuple[DeltaRadixLayout, torch.Tensor]:
    layout = inject_delta_markers(full_ids, *_wire(event, ranges), registry)
    assert layout is not None
    keep_mask = torch.ones(len(full_ids), dtype=torch.bool)
    for start, end in ranges:
        keep_mask[start:end] = False
    return layout, keep_mask


def _key_indices(layout: DeltaRadixLayout, real_values: torch.Tensor) -> torch.Tensor:
    values = torch.full((len(layout.keys),), -1, dtype=torch.int32)
    values[~layout.virtual_mask] = real_values
    return values


def _real_values(layout: DeltaRadixLayout, key_values: torch.Tensor) -> torch.Tensor:
    return key_values[~layout.virtual_mask]


def _commit_layout(
    cache: RadixPrefixCache,
    layout: DeltaRadixLayout,
    keep_mask: torch.Tensor,
    real_values: torch.Tensor,
):
    return cache.commit_drop_prefix(
        layout.keys,
        _key_indices(layout, real_values),
        layout.virtual_mask,
        layout.key_to_token,
        keep_mask,
    )


def _drop_cache(num_pages: int = 256) -> tuple[CacheManager, DeltaMarkerRegistry]:
    page_table = torch.full((32, 256), -1, dtype=torch.int32)
    manager = CacheManager(
        num_pages,
        1,
        page_table,
        type="radix",
        drop_aware_eviction=True,
    )
    registry = DeltaMarkerRegistry()
    manager.bind_delta_marker_registry(registry)
    return manager, registry


def test_default_mode_keeps_original_protected_leaf_lru_behavior():
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    manager = CacheManager(16, 1, page_table, type="radix")
    assert not manager.drop_aware_eviction

    token_ids = torch.arange(100, 108, dtype=torch.int64)
    slots = manager._allocate(len(token_ids))
    inserted = manager.prefix_cache.insert_prefix(token_ids, slots)
    manager.lock(inserted.handle)
    assert manager.prefix_cache.size_info.evictable_size == 0
    assert manager.prefix_cache.size_info.protected_size == len(token_ids)

    manager.unlock(inserted.handle)
    released = manager.prefix_cache.evict(len(token_ids))
    manager._free(released)
    assert manager.prefix_cache.match_prefix(token_ids).cuda_handle.cached_len == 0
    manager.check_integrity()


def test_half_open_drop_cut_splits_one_node_and_only_reclaims_common_drop_block():
    manager, registry = _drop_cache(32)
    cache = manager.prefix_cache
    assert isinstance(cache, RadixPrefixCache)
    full_ids = torch.arange(100, 116, dtype=torch.int64)

    first, first_keep = _layout(
        full_ids, registry, event=12, ranges=[(2, 8)]
    )
    first_slots = manager._allocate(16)
    first_result = _commit_layout(cache, first, first_keep, first_slots)

    second, second_keep = _layout(
        full_ids, registry, event=12, ranges=[(2, 5)]
    )
    matched_prefix = cache.match_prefix(second.keys, second.virtual_mask).cuda_handle
    assert matched_prefix.cached_len == 12
    matched_prefix_values = matched_prefix.get_matched_indices()
    matched_prefix_real = matched_prefix_values[
        ~matched_prefix.get_matched_virtual_mask()
    ]
    second_suffix = manager._allocate(4)
    second_values = torch.cat([matched_prefix_real, second_suffix])
    second_result = _commit_layout(cache, second, second_keep, second_values)

    first_real = _real_values(first, first_result.canonical_indices)
    second_real = _real_values(second, second_result.canonical_indices)
    assert torch.equal(first_real[:12], second_real[:12])
    dropped_owner = cache._slot_owner[int(first_real[2].item())]
    kept_owner = cache._slot_owner[int(first_real[5].item())]
    assert dropped_owner.page_length == 3
    assert dropped_owner.kv_need_leaf_count == 0
    assert kept_owner.page_length == 3
    assert kept_owner.kv_need_leaf_count == 1

    released = cache.evict(3)
    assert set(released.tolist()) == set(first_real[2:5].tolist())
    assert not dropped_owner.resident
    assert torch.all(dropped_owner.value == -1)
    assert kept_owner.resident
    cache.check_integrity()

    manager._free(released)
    registry.release_request_refs(first.marker_ids)
    registry.release_request_refs(second.marker_ids)
    manager.check_integrity()


@dataclass(frozen=True)
class _CommittedBranch:
    layout: DeltaRadixLayout
    keep_mask: torch.Tensor
    handle: RadixCacheHandle
    real_values: torch.Tensor


def _commit_shared_branch(
    manager: CacheManager,
    registry: DeltaMarkerRegistry,
    full_ids: torch.Tensor,
    branch_idx: int,
) -> _CommittedBranch:
    cache = manager.prefix_cache
    assert isinstance(cache, RadixPrefixCache)
    layout, keep_mask = _layout(
        full_ids,
        registry,
        event=48,
        ranges=[(8, 24), (32 + branch_idx, 33 + branch_idx)],
    )
    matched = cache.match_prefix(layout.keys, layout.virtual_mask).cuda_handle
    if matched.cached_len == 0:
        matched_real = torch.empty(0, dtype=torch.int32)
    else:
        matched_values = matched.get_matched_indices()
        matched_real = matched_values[~matched.get_matched_virtual_mask()]
    assert len(matched_real) in (0, 48)
    suffix = manager._allocate(len(full_ids) - len(matched_real))
    result = _commit_layout(
        cache,
        layout,
        keep_mask,
        torch.cat([matched_real, suffix]),
    )
    return _CommittedBranch(
        layout=layout,
        keep_mask=keep_mask,
        handle=result.handle,
        real_values=_real_values(layout, result.canonical_indices),
    )


def test_all_slots_pressure_prefers_referenced_drop_block_preserves_tree_and_refills():
    manager, registry = _drop_cache(256)
    cache = manager.prefix_cache
    assert isinstance(cache, RadixPrefixCache)
    full_ids = torch.arange(1000, 1064, dtype=torch.int64)
    branches = [
        _commit_shared_branch(manager, registry, full_ids, branch_idx)
        for branch_idx in range(12)
    ]

    assert len(manager.free_slots) == 16
    common_slots = branches[0].real_values[8:24].clone()
    for branch in branches[1:]:
        assert torch.equal(branch.real_values[:48], branches[0].real_values[:48])

    locked_handles: list[RadixCacheHandle] = []
    for branch in branches:
        pinned = branch.real_values[branch.keep_mask]
        handle = cache.with_pinned_slots(branch.handle, pinned)
        manager.lock(handle)
        locked_handles.append(handle)

    common_owner = cache._slot_owner[int(common_slots[0].item())]
    assert common_owner.page_length == 16
    assert common_owner.ref_count == len(branches)
    assert common_owner.kv_need_leaf_count == 0
    assert common_owner.kv_pin_count == 0

    # First consume every remaining free slot, then force a second allocation. The
    # only high-priority block is the referenced token range dropped by every leaf.
    transient = manager._allocate(len(manager.free_slots))
    assert len(manager.free_slots) == 0
    recycled = manager._allocate(len(common_slots))
    assert set(recycled.tolist()) == set(common_slots.tolist())
    assert not common_owner.resident
    assert common_owner.ref_count == len(branches)

    for branch in branches:
        rematched = cache.match_prefix(
            branch.layout.keys, branch.layout.virtual_mask
        ).cuda_handle
        assert rematched.cached_len == len(branch.layout.keys)
        rematched_real = _real_values(
            branch.layout, rematched.get_matched_indices()
        )
        assert torch.all(rematched_real[8:24] == -1)
        assert torch.equal(rematched_real[:8], branch.real_values[:8])
        assert torch.equal(rematched_real[24:], branch.real_values[24:])

    manager._free(torch.cat([transient, recycled]))
    for handle in locked_handles:
        manager.unlock(handle)
    manager.check_integrity()

    # A no-Drop request matches through the structural hole, Prefills from the
    # first missing kept token, and atomically repopulates the preserved nodes.
    plain_virtual = torch.zeros(len(full_ids), dtype=torch.bool)
    plain_key_to_token = torch.arange(len(full_ids), dtype=torch.int64)
    plain_keep = torch.ones(len(full_ids), dtype=torch.bool)
    structural = cache.match_prefix(full_ids, plain_virtual).cuda_handle
    assert structural.cached_len == 48
    structural_values = structural.get_matched_indices()
    missing = torch.nonzero(structural_values < 0, as_tuple=False).view(-1)
    assert missing.tolist() == list(range(8, 24))
    fresh = manager._allocate(len(missing) + 16)
    refill_values = structural_values.clone()
    refill_values[missing] = fresh[: len(missing)]
    refill_values = torch.cat([refill_values, fresh[len(missing) :]])
    refill = cache.commit_drop_prefix(
        full_ids,
        refill_values,
        plain_virtual,
        plain_key_to_token,
        plain_keep,
    )
    canonical = refill.canonical_indices
    assert torch.equal(canonical[8:24], fresh[:16])
    assert cache._slot_owner[int(canonical[8].item())].kv_need_leaf_count == 1

    # First writer remains canonical when another completion races to cache the
    # same fully resident path.
    ignored_candidates = torch.arange(10000, 10064, dtype=torch.int32)
    repeated = cache.commit_drop_prefix(
        full_ids,
        ignored_candidates,
        plain_virtual,
        plain_key_to_token,
        plain_keep,
    )
    assert torch.equal(repeated.canonical_indices, canonical)
    manager.check_integrity()

    for branch in branches:
        registry.release_request_refs(branch.layout.marker_ids)
    manager.check_integrity()


def test_full_slot_churn_repeatedly_preserves_slot_partition_and_radix_integrity():
    manager, registry = _drop_cache(256)
    cache = manager.prefix_cache
    assert isinstance(cache, RadixPrefixCache)
    full_ids = torch.arange(2000, 2064, dtype=torch.int64)
    branches = [
        _commit_shared_branch(manager, registry, full_ids, branch_idx)
        for branch_idx in range(12)
    ]
    assert manager.prefix_cache.size_info.total_size == 240

    for cycle in range(12):
        branch = branches[cycle % len(branches)]
        matched = cache.match_prefix(branch.layout.keys, branch.layout.virtual_mask).cuda_handle
        matched_key_values = matched.get_matched_indices()
        matched_values = matched_key_values[~matched.get_matched_virtual_mask()]
        kept = matched_values[branch.keep_mask[: len(matched_values)]]
        first_kept_hole = torch.nonzero(kept < 0, as_tuple=False).view(-1)
        safe_len = int(first_kept_hole[0].item()) if len(first_kept_hole) else len(kept)
        handle = cache.with_pinned_slots(matched, kept[:safe_len])
        manager.lock(handle)

        free_before = len(manager.free_slots)
        transient = manager._allocate(free_before)
        assert len(manager.free_slots) == 0
        forced = manager._allocate(1)
        assert len(forced) == 1
        manager._free(torch.cat([transient, forced]))
        manager.unlock(handle)
        manager.check_integrity()

        rematched = cache.match_prefix(
            branch.layout.keys, branch.layout.virtual_mask
        ).cuda_handle
        assert rematched.cached_len > 0
        assert torch.equal(
            rematched.get_matched_indices()[rematched.get_matched_virtual_mask()],
            torch.full(
                (int(torch.count_nonzero(branch.layout.virtual_mask).item()),),
                -1,
                dtype=torch.int32,
            ),
        )

    for branch in branches:
        registry.release_request_refs(branch.layout.marker_ids)
    manager.check_integrity()


def test_finished_drop_commit_keeps_generated_tokens_beyond_prompt_mask():
    manager, registry = _drop_cache(32)
    cache = manager.prefix_cache
    assert isinstance(cache, RadixPrefixCache)
    prompt_ids = torch.arange(3000, 3006, dtype=torch.int64)
    layout, prompt_keep = _layout(
        prompt_ids, registry, event=4, ranges=[(1, 3)]
    )
    generated_ids = torch.tensor([4000, 4001], dtype=torch.int64)
    keys = torch.cat([layout.keys, generated_ids])
    virtual_mask = torch.cat(
        [layout.virtual_mask, torch.zeros(2, dtype=torch.bool)]
    )
    key_to_token = torch.cat(
        [layout.key_to_token, torch.tensor([6, 7], dtype=torch.int64)]
    )
    token_to_key = torch.cat(
        [
            layout.token_to_key,
            torch.tensor([len(layout.keys), len(layout.keys) + 1], dtype=torch.int64),
        ]
    )
    active_positions = torch.tensor([0, 3, 4, 5, 6, 7], dtype=torch.int32)
    active_slots = manager._allocate(len(active_positions))
    manager.page_table[0, : len(active_positions)] = active_slots
    old_handle = cache.match_prefix(keys, virtual_mask).cuda_handle
    manager.lock(old_handle)
    req = SimpleNamespace(
        radix_key_virtual_mask=virtual_mask,
        radix_key_to_token=key_to_token,
        radix_token_to_key=token_to_key,
        radix_match_ids=keys,
        radix_commit_key_len=None,
        cache_handle=old_handle,
        table_idx=0,
        cached_len=len(active_positions),
        input_ids=torch.cat([prompt_ids[prompt_keep], generated_ids]),
        true_positions=active_positions,
        initial_active_cached_len=0,
        initial_full_match_indices=torch.empty(0, dtype=torch.int32),
        full_keep_mask=prompt_keep.to(dtype=torch.int32),
        prefix_keep_mask=None,
    )

    manager._cache_finished_drop_aware_delta_req(req)

    committed = cache.match_prefix(keys, virtual_mask).cuda_handle
    real_values = committed.get_matched_indices()[~virtual_mask]
    assert torch.all(real_values[1:3] == -1)
    assert torch.all(real_values[torch.tensor([0, 3, 4, 5, 6, 7])] >= 0)
    registry.release_request_refs(layout.marker_ids)
    manager.check_integrity()


def test_finished_commit_discards_stale_dropped_slots_after_reassignment():
    manager, registry = _drop_cache(24)
    cache = manager.prefix_cache
    assert isinstance(cache, RadixPrefixCache)
    prompt_ids = torch.arange(5000, 5008, dtype=torch.int64)
    layout, keep_mask = _layout(
        prompt_ids, registry, event=6, ranges=[(2, 4)]
    )
    initial_slots = manager._allocate(len(prompt_ids))
    committed = _commit_layout(cache, layout, keep_mask, initial_slots)
    initial_values = _real_values(layout, committed.canonical_indices)
    handle = cache.with_pinned_slots(committed.handle, initial_values[keep_mask])
    manager.lock(handle)

    transient = manager._allocate(len(manager.free_slots))
    reassigned = manager._allocate(2)
    assert set(reassigned.tolist()) == set(initial_values[2:4].tolist())
    other_ids = torch.tensor([9000, 9001], dtype=torch.int64)
    other_mask = torch.zeros(2, dtype=torch.bool)
    other_key_to_token = torch.arange(2, dtype=torch.int64)
    other_keep = torch.ones(2, dtype=torch.bool)
    cache.commit_drop_prefix(
        other_ids,
        reassigned,
        other_mask,
        other_key_to_token,
        other_keep,
    )

    active_positions = torch.nonzero(keep_mask, as_tuple=False).view(-1).to(torch.int32)
    active_values = initial_values[keep_mask]
    manager.page_table[0, : len(active_values)] = active_values
    req = SimpleNamespace(
        radix_key_virtual_mask=layout.virtual_mask,
        radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key,
        radix_match_ids=layout.keys,
        radix_commit_key_len=None,
        cache_handle=handle,
        table_idx=0,
        cached_len=len(active_values),
        input_ids=prompt_ids[keep_mask],
        true_positions=active_positions,
        initial_active_cached_len=len(active_values),
        initial_full_match_indices=initial_values.clone(),
        full_keep_mask=keep_mask.to(dtype=torch.int32),
        prefix_keep_mask=None,
    )

    manager._cache_finished_drop_aware_delta_req(req)

    rematched = cache.match_prefix(layout.keys, layout.virtual_mask).cuda_handle
    rematched_values = _real_values(layout, rematched.get_matched_indices())
    assert torch.all(rematched_values[2:4] == -1)
    assert all(cache._slot_owner[int(slot)] is not None for slot in reassigned.tolist())
    manager._free(transient)
    registry.release_request_refs(layout.marker_ids)
    manager.check_integrity()
