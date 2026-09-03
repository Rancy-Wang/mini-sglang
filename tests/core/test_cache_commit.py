from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.radix_delta import DeltaMarkerRegistry, inject_delta_markers


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(*, rows: int = 4, num_pages: int = 32) -> CacheManager:
    core.set_global_ctx(core.Context(page_size=1))
    page_table = torch.full((rows, num_pages), -1, dtype=torch.int32)
    return CacheManager(num_pages, 1, page_table, type="radix")


def _root_handle(cache_manager: CacheManager):
    return cache_manager.prefix_cache.match_prefix(torch.empty(0, dtype=torch.int64)).cuda_handle


def _delta_layout(cache_manager: CacheManager, full_ids: torch.Tensor):
    registry = DeltaMarkerRegistry()
    cache_manager.bind_delta_marker_registry(registry)
    layout = inject_delta_markers(
        full_ids,
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32),
        registry,
    )
    assert layout is not None
    return layout


def _delta_req(
    cache_manager: CacheManager,
    *,
    row: int,
    input_ids: torch.Tensor,
    true_positions: torch.Tensor,
    layout,
):
    return SimpleNamespace(
        input_ids=input_ids,
        true_positions=true_positions,
        raw_positions=true_positions,
        radix_input_ids=input_ids.to(torch.int64),
        radix_match_ids=layout.keys,
        radix_key_virtual_mask=layout.virtual_mask,
        radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key,
        radix_commit_key_len=None,
        initial_full_match_indices=torch.empty(0, dtype=torch.int32),
        initial_active_cached_len=0,
        cached_len=len(input_ids),
        table_idx=row,
        cache_handle=_root_handle(cache_manager),
        retry_transformed_mask=None,
        inactive_cached_positions=None,
        inactive_cached_pages=None,
    )


def test_unfinished_linear_commit_repoints_duplicate_pages_before_free():
    cache_manager = _make_cache_manager(num_pages=16)
    allocated = cache_manager._allocate(8)
    cache_manager.page_table[0, :4] = allocated[:4]
    cache_manager.page_table[1, :4] = allocated[4:]
    input_ids = torch.tensor([10, 11, 12, 13, 99], dtype=torch.int64)
    old_handle_a = _root_handle(cache_manager)
    old_handle_b = _root_handle(cache_manager)

    def make_req(row: int, handle):
        return SimpleNamespace(
            radix_input_ids=input_ids,
            radix_match_ids=input_ids,
            radix_key_virtual_mask=None,
            use_context_mask=False,
            cached_len=4,
            table_idx=row,
            cache_handle=handle,
        )

    req_a = make_req(0, old_handle_a)
    req_b = make_req(1, old_handle_b)
    cache_manager.cache_req(req_a, finished=False)
    cache_manager.cache_req(req_b, finished=False)

    canonical = req_b.cache_handle.get_matched_indices()
    assert torch.equal(cache_manager.page_table[1, :4], canonical)
    assert set(allocated[4:].tolist()).issubset(set(cache_manager.free_slots.tolist()))
    assert not (
        set(cache_manager.page_table[1, :4].tolist()) & set(cache_manager.free_slots.tolist())
    )


def test_staged_delta_commit_keeps_inserted_prefix_and_frees_pages_after_drop_hole():
    cache_manager = _make_cache_manager(num_pages=16)
    full_ids = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    layout = _delta_layout(cache_manager, full_ids)
    allocated = cache_manager._allocate(4)
    cache_manager.page_table[0, :4] = allocated
    req = _delta_req(
        cache_manager,
        row=0,
        input_ids=full_ids[torch.tensor([0, 1, 4, 5])].to(torch.int32),
        true_positions=torch.tensor([0, 1, 4, 5], dtype=torch.int32),
        layout=layout,
    )

    cache_manager.cache_req(req, finished=True)

    matched = cache_manager.prefix_cache.match_prefix(full_ids[:2]).cuda_handle
    assert torch.equal(matched.get_matched_indices(), allocated[:2])
    assert set(allocated[2:].tolist()).issubset(set(cache_manager.free_slots.tolist()))
    assert not (set(allocated[:2].tolist()) & set(cache_manager.free_slots.tolist()))


def test_mask_delta_concurrent_commit_keeps_canonical_and_frees_duplicate_pages():
    cache_manager = _make_cache_manager(num_pages=24)
    full_ids = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
    layout = _delta_layout(cache_manager, full_ids)
    allocated = cache_manager._allocate(12)
    cache_manager.page_table[0, :6] = allocated[:6]
    cache_manager.page_table[1, :6] = allocated[6:]
    positions = torch.arange(6, dtype=torch.int32)
    req_a = _delta_req(
        cache_manager,
        row=0,
        input_ids=full_ids.to(torch.int32),
        true_positions=positions,
        layout=layout,
    )
    req_b = _delta_req(
        cache_manager,
        row=1,
        input_ids=full_ids.to(torch.int32),
        true_positions=positions,
        layout=layout,
    )

    cache_manager.cache_req(req_a, finished=True)
    cache_manager.cache_req(req_b, finished=True)

    matched = cache_manager.prefix_cache.match_prefix(layout.keys, layout.virtual_mask).cuda_handle
    canonical = matched.get_matched_indices()
    real_mask = ~matched.get_matched_virtual_mask().to(device=canonical.device)
    assert torch.equal(canonical[real_mask], allocated[:6])
    assert set(allocated[6:].tolist()).issubset(set(cache_manager.free_slots.tolist()))
    assert not (set(allocated[:6].tolist()) & set(cache_manager.free_slots.tolist()))
