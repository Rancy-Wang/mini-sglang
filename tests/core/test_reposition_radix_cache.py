from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi")

import minisgl.core as core
from minisgl.kernel.radix_reposition import DELTA_KIND, REPOSITION_KIND, TOKEN_KIND
from minisgl.kvcache.radix_cache import RadixPrefixCache
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    yield
    core._GLOBAL_CTX = old_ctx


def _cache() -> RadixPrefixCache:
    return RadixPrefixCache(torch.device("cpu"))


def _records(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device="cpu")


def _mask(length: int) -> torch.Tensor:
    return torch.zeros(length, dtype=torch.bool, device="cpu")


def test_retry_stops_before_target_reposition_missing_from_source() -> None:
    cache = _cache()
    source = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, -1, 1],
        ]
    )
    cache.insert_prefix(source, torch.tensor([0, 1], dtype=torch.int32), _mask(2))
    target = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [REPOSITION_KIND, 0, -1, -1],
            [TOKEN_KIND, 11, 0, 1],
        ]
    )
    target_mask = torch.tensor([False, True, False], dtype=torch.bool)

    exact = cache.match_prefix(target, target_mask).cuda_handle
    retry = cache.match_retry_prefix(target, target_mask, exact)

    assert exact.cached_len == 1
    assert retry.cached_len == 1


def test_retry_greedily_selects_largest_reachable_source_branch() -> None:
    cache = _cache()
    short = _records(
        [
            [TOKEN_KIND, 10, 1, 1],
            [TOKEN_KIND, 11, 1, 2],
        ]
    )
    long = _records(
        [
            [TOKEN_KIND, 10, 2, 2],
            [TOKEN_KIND, 11, 2, 3],
            [TOKEN_KIND, 12, 2, 4],
            [TOKEN_KIND, 13, 2, 5],
        ]
    )
    cache.insert_prefix(short, torch.tensor([0, 1], dtype=torch.int32), _mask(2))
    cache.insert_prefix(long, torch.tensor([2, 3, 4, 5], dtype=torch.int32), _mask(4))
    target = _records(
        [
            [TOKEN_KIND, 10, 7, 0],
            [TOKEN_KIND, 11, 7, 1],
            [TOKEN_KIND, 12, 7, 2],
            [TOKEN_KIND, 99, 7, 3],
        ]
    )

    exact = cache.match_prefix(target, _mask(4)).cuda_handle
    retry = cache.match_retry_prefix(target, _mask(4), exact)

    assert exact.cached_len == 0
    assert retry.cached_len == 3
    assert retry.get_matched_indices()[:3].tolist() == [2, 3, 4]


def test_structured_exact_index_survives_edge_split() -> None:
    cache = _cache()
    first = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, -1, 1],
            [TOKEN_KIND, 12, -1, 2],
        ]
    )
    second = first.clone()
    second[2, 1] = 99

    cache.insert_prefix(first, torch.tensor([0, 1, 2], dtype=torch.int32), _mask(3))
    cache.insert_prefix(second, torch.tensor([0, 1, 3], dtype=torch.int32), _mask(3))

    assert cache.match_prefix(first, _mask(3)).cuda_handle.cached_len == 3
    assert cache.match_prefix(second, _mask(3)).cuda_handle.cached_len == 3
    assert cache.root_node.max_reachable_depth == 3
    cache.check_integrity()


def test_ordinary_radix_index_survives_edge_split() -> None:
    cache = _cache()
    first = torch.tensor([10, 11, 12], dtype=torch.int32)
    second = torch.tensor([10, 11, 99], dtype=torch.int32)

    cache.insert_prefix(first, torch.tensor([0, 1, 2], dtype=torch.int32))
    cache.insert_prefix(second, torch.tensor([0, 1, 3], dtype=torch.int32))

    assert cache.match_prefix(first).cuda_handle.cached_len == 3
    assert cache.match_prefix(second).cuda_handle.cached_len == 3
    assert cache.root_node.max_reachable_depth == 3
    cache.check_integrity()


def test_shared_retry_pages_are_counted_and_normally_evicted_once() -> None:
    cache = _cache()
    source = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, 2, 1],
            [TOKEN_KIND, 12, -1, 2],
            [TOKEN_KIND, 13, 2, 3],
        ]
    )
    target = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, 8, 0],
            [TOKEN_KIND, 12, -1, 2],
            [TOKEN_KIND, 13, 8, 1],
        ]
    )
    source_result = cache.insert_prefix(
        source,
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        _mask(4),
    )
    cache.insert_prefix(
        target,
        torch.tensor([0, 4, 2, 5], dtype=torch.int32),
        _mask(4),
    )

    assert cache.size_info.evictable_size == 6
    assert cache.size_info.protected_size == 0
    cache.check_integrity()

    cache.lock_handle(source_result.handle)
    assert cache.size_info.evictable_size == 2
    assert cache.size_info.protected_size == 4
    first = cache.evict(2)
    assert set(first.tolist()) == {4, 5}
    cache.check_integrity()

    cache.lock_handle(source_result.handle, unlock=True)
    second = cache.evict(4)
    assert set(second.tolist()) == {0, 1, 2, 3}
    assert cache.size_info.total_size == 0
    cache.check_integrity()


def test_retry_position_plan_keeps_changed_pages_that_are_dropped_later() -> None:
    page_table = torch.zeros((1, 8), dtype=torch.int32, device="cpu")
    manager = CacheManager(8, 1, page_table, "radix")
    source = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, -1, 1],
            [DELTA_KIND, -7, -1, -1],
            [TOKEN_KIND, 12, -1, 2],
            [DELTA_KIND, -8, -1, -1],
            [TOKEN_KIND, 13, -1, 3],
        ]
    )
    source_virtual = torch.tensor([False, False, True, False, True, False], dtype=torch.bool)
    manager.prefix_cache.insert_prefix(
        source,
        torch.tensor([0, 1, -1, 2, -1, 3], dtype=torch.int32),
        source_virtual,
    )

    target = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, 1, 0],
            [DELTA_KIND, -7, -1, -1],
            [REPOSITION_KIND, 1, -1, -1],
            [TOKEN_KIND, 12, 1, 1],
            [DELTA_KIND, -8, -1, -1],
            [TOKEN_KIND, 13, 1, 2],
        ]
    )
    req = SimpleNamespace(
        input_len=2,
        radix_match_ids=target,
        radix_token_to_key=torch.tensor([0, 1, 4, 6], dtype=torch.int64),
        radix_key_to_token=torch.tensor([0, 1, -1, -1, 2, -1, 3], dtype=torch.int64),
        radix_key_virtual_mask=torch.tensor(
            [False, False, True, True, False, True, False], dtype=torch.bool
        ),
        radix_commit_key_len=None,
        raw_positions=torch.tensor([2, 3], dtype=torch.int32),
        prefix_keep_mask=torch.tensor([0, 0, 1], dtype=torch.int32),
    )

    match = manager.match_req(req)

    assert match is not None
    assert match.full_cached_len == 2
    assert match.active_cached_len == 0
    assert match.retry_plan.tolist() == [[1, 1, 1, 0]]


def test_concurrent_commit_adopts_inactive_retry_page_and_frees_duplicate() -> None:
    page_table = torch.full((2, 8), -1, dtype=torch.int32, device="cpu")
    manager = CacheManager(16, 1, page_table, "radix")
    source_pages = manager._allocate(3)
    source = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, -1, 1],
            [TOKEN_KIND, 12, -1, 2],
            [DELTA_KIND, -7, -1, -1],
        ]
    )
    source_virtual = torch.tensor([False, False, False, True], dtype=torch.bool)
    manager.prefix_cache.insert_prefix(
        source,
        torch.cat([source_pages, torch.tensor([-1], dtype=torch.int32)]),
        source_virtual,
    )

    target = _records(
        [
            [TOKEN_KIND, 10, -1, 0],
            [TOKEN_KIND, 11, 2, 0],
            [TOKEN_KIND, 12, 2, 1],
            [DELTA_KIND, -7, -1, -1],
            [REPOSITION_KIND, 2, -1, -1],
            [TOKEN_KIND, 13, 2, 2],
            [DELTA_KIND, -8, -1, -1],
        ]
    )
    target_virtual = torch.tensor([False, False, False, True, True, False, True], dtype=torch.bool)
    token_to_key = torch.tensor([0, 1, 2, 5], dtype=torch.int64)
    key_to_token = torch.tensor([0, 1, 2, -1, -1, 3, -1], dtype=torch.int64)
    pending = SimpleNamespace(
        input_len=2,
        radix_match_ids=target,
        radix_token_to_key=token_to_key,
        radix_key_to_token=key_to_token,
        radix_key_virtual_mask=target_virtual,
        radix_commit_key_len=None,
        raw_positions=torch.tensor([2, 3], dtype=torch.int32),
        prefix_keep_mask=torch.tensor([0, 0, 1], dtype=torch.int32),
    )

    matches = [manager.match_req(pending), manager.match_req(pending)]
    assert all(match is not None for match in matches)
    for match in matches:
        assert match is not None
        assert match.full_cached_len == 3
        assert match.active_full_positions.device.type == "cpu"
        assert match.active_full_positions.tolist() == [2]
        assert match.retry_plan.tolist() == [
            [1, 1, 1, 0],
            [2, 2, 2, 1],
        ]
        manager.lock(match.handle)

    allocated = manager._allocate(6).view(2, 3)

    def make_req(row: int):
        match = matches[row]
        assert match is not None
        retry_inactive, retry_active, computed = allocated[row]
        page_table[row, :2] = torch.stack([retry_active, computed])
        initial_full = match.full_match_indices.clone()
        initial_full[1] = retry_inactive
        initial_full[2] = retry_active
        return SimpleNamespace(
            input_ids=torch.tensor([12, 13], dtype=torch.int32),
            true_positions=torch.tensor([1, 2], dtype=torch.int32),
            raw_positions=torch.tensor([2, 3], dtype=torch.int32),
            radix_input_ids=target[token_to_key[[2, 3]]],
            radix_match_ids=target,
            radix_key_virtual_mask=target_virtual,
            radix_key_to_token=key_to_token,
            radix_token_to_key=token_to_key,
            radix_commit_key_len=None,
            initial_full_match_indices=initial_full,
            initial_active_cached_len=1,
            cached_len=2,
            table_idx=row,
            cache_handle=match.handle,
            retry_transformed_mask=torch.tensor([True], dtype=torch.bool),
            inactive_cached_positions=torch.tensor([1], dtype=torch.int64),
            inactive_cached_pages=retry_inactive.view(1),
            use_context_mask=False,
        )

    req_a, req_b = make_req(0), make_req(1)
    manager.cache_req(req_a, finished=True)

    target_handle = manager.prefix_cache.match_prefix(target, target_virtual).cuda_handle
    target_real = target_handle.get_matched_indices()[~target_virtual]
    assert target_real.tolist() == [
        int(source_pages[0]),
        int(allocated[0, 0]),
        int(allocated[0, 1]),
        int(allocated[0, 2]),
    ]
    assert not set(allocated[0].tolist()) & set(manager.free_slots.tolist())

    manager.cache_req(req_b, finished=True)

    assert set(allocated[1].tolist()).issubset(set(manager.free_slots.tolist()))
    assert not set(allocated[0].tolist()) & set(manager.free_slots.tolist())
    manager.check_integrity()


def test_finished_candidate_requires_canonical_slot_not_only_insert_range() -> None:
    page_table = torch.full((1, 4), -1, dtype=torch.int32, device="cpu")
    manager = CacheManager(4, 1, page_table, "radix")
    candidate, canonical = manager._allocate(2)
    key = _records([[TOKEN_KIND, 10, -1, 0]])
    insert_result = manager.prefix_cache.insert_prefix(key, canonical.view(1), _mask(1))
    req = SimpleNamespace(
        initial_active_cached_len=0,
        retry_transformed_mask=None,
        inactive_cached_positions=None,
        inactive_cached_pages=None,
        radix_token_to_key=None,
    )

    manager._free_finished_candidates(
        req,
        candidate.view(1),
        torch.tensor([0], dtype=torch.int64),
        insert_result,
    )

    assert int(candidate) in manager.free_slots.tolist()
    assert int(canonical) not in manager.free_slots.tolist()
    manager.check_integrity()
