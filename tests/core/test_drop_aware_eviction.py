from __future__ import annotations

import pytest
import torch

pytest.importorskip("tvm_ffi")

import minisgl.core as core
from minisgl.kvcache.radix_cache import RadixPrefixCache


@pytest.fixture(autouse=True)
def context():
    previous = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    yield
    core._GLOBAL_CTX = previous


def cache(shared=False):
    return RadixPrefixCache(torch.device("cpu"), track_shared_page_owners=shared,
                            drop_aware_eviction=True)


def insert(c, tail=20, pages=None):
    # A = tokens 10..13; Delta drops A[1:3]; B = token tail.
    keys = torch.tensor([[0, 10, -1, 0], [0, 11, -1, 1], [0, 12, -1, 2],
                         [0, 13, -1, 3], [1, -2, -4, -1], [0, tail, -1, 4]],
                        dtype=torch.int32)
    values = torch.tensor([0, 1, 2, 3, -1, 4] if pages is None else pages,
                          dtype=torch.int32)
    return keys, c.insert_prefix(keys, values, keys[:, 0] != 0).handle


def nodes(handle):
    result = []
    node = handle.node
    while not node.is_root():
        result.append(node)
        node = node.parent
    return result[::-1]


@pytest.mark.parametrize("shared", [False, True])
def test_locked_middle_evict_fill_and_unlock_after_split(shared):
    c = cache(shared)
    keys, handle = insert(c)
    c.configure_drop_lock(handle, torch.tensor([True, False, False, True, True]))
    c.lock_handle(handle)
    assert all(n.path_ref_count == 1 for n in nodes(handle))
    assert [(n.ref_count, n.page_length) for n in nodes(handle)] == [
        (1, 1), (0, 2), (1, 1), (1, 0), (1, 1)]
    assert c.evict(2).tolist() == [1, 2]
    assert handle.get_matched_indices().tolist() == [0, -1, -1, 3, -1, 4]
    assert handle.full_token_len == 5 and handle.physical_cached_len == 3
    # Another request splits the released interval while the original lease lives.
    other = c.match_prefix(keys[:2], keys[:2, 0] != 0).cuda_handle
    c.lock_handle(other)
    c.insert_prefix(keys, torch.tensor([0, 7, 8, 3, -1, 4], dtype=torch.int32),
                    keys[:, 0] != 0)
    assert handle.get_matched_indices().tolist() == [0, 7, 8, 3, -1, 4]
    assert c.size_info.total_size == 5
    c.lock_handle(other, unlock=True)
    c.lock_handle(handle, unlock=True)
    assert all(n.ref_count == n.path_ref_count == 0 for n in nodes(handle))
    assert set(c.evict(5).tolist()) == {0, 7, 8, 3, 4}
    assert c.size_info.total_size == 0


def test_leaf_cascade_has_priority_over_dropped_middle():
    c = cache()
    _, handle = insert(c)
    c.configure_drop_lock(handle, torch.tensor([True, False, False, True, True]))
    c.lock_handle(handle)
    cold = torch.tensor([[0, 91, -1, 0], [0, 92, -1, 1]], dtype=torch.int32)
    c.insert_prefix(cold, torch.tensor([8, 9], dtype=torch.int32))
    c.match_prefix(cold[:1])  # Split, so eviction must discover the new leaf parent.
    assert set(c.evict(2).tolist()) == {8, 9}
    assert c.eviction_stats["drop_pages"] == 0
    assert set(c.evict(2).tolist()) == {1, 2}
    assert c.eviction_stats["leaf_pages"] == 2
    c.lock_handle(handle, unlock=True)


def test_no_matched_delta_cannot_reduce_parent_ref():
    c = cache()
    keys, _ = insert(c)
    handle = c.match_prefix(keys[:4], keys[:4, 0] != 0).cuda_handle
    c.configure_drop_lock(handle, torch.zeros(4, dtype=torch.bool))
    c.lock_handle(handle)
    assert handle.skip_ranges == []
    assert all(n.ref_count == n.path_ref_count == 1 for n in nodes(handle))
    c.lock_handle(handle, unlock=True)


def test_another_active_reference_prevents_middle_eviction():
    c = cache()
    keys, dropped = insert(c)
    active = c.match_prefix(keys[:4], keys[:4, 0] != 0).cuda_handle
    c.lock_handle(active)
    c.configure_drop_lock(dropped, torch.tensor([True, False, False, True, True]))
    c.lock_handle(dropped)
    assert c.evictable_size == 0
    c.lock_handle(active, unlock=True)
    assert c.evictable_size == 2
    c.lock_handle(dropped, unlock=True)


def test_hole_fill_uses_canonical_winner_without_overwrite():
    c = cache()
    keys, h = insert(c, pages=[0, -1, -1, 3, -1, 4])
    c.insert_prefix(keys, torch.tensor([0, 7, 8, 3, -1, 4], dtype=torch.int32), keys[:, 0] != 0)
    c.insert_prefix(keys, torch.tensor([0, 9, 10, 3, -1, 4], dtype=torch.int32), keys[:, 0] != 0)
    assert h.get_matched_indices().tolist() == [0, 7, 8, 3, -1, 4]
    assert c.eviction_stats["hole_fills"] == 2
    assert c.size_info.total_size == 5


def test_candidate_storage_is_bounded_under_repeated_matches():
    c = cache()
    keys, _ = insert(c)
    for _ in range(200):
        c.match_prefix(keys, keys[:, 0] != 0)
    assert len(c._leaf_candidates) + len(c._drop_candidates) <= 2 * len(c._candidates) + 64
