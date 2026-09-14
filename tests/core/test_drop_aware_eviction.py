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


def test_cacheback_does_not_resurrect_evicted_snapshot_after_page_reuse():
    from test_reposition_generated_cacheback import _final_message
    from minisgl.core import Req
    from minisgl.scheduler.cache import CacheManager

    message = _final_message()
    table = torch.full((1, 16), -1, dtype=torch.int32)
    manager = CacheManager(6, 1, table, "radix", drop_aware_eviction=True)
    key_end = int(message.radix_token_to_key[-1])
    virtual = message.radix_key_virtual_mask[:key_end]
    original = manager._allocate(5)
    values = torch.full((key_end,), -1, dtype=torch.int32)
    values[~virtual] = original
    handle = manager.prefix_cache.insert_prefix(
        message.radix_match_ids[:key_end], values, virtual).handle
    manager.prefix_cache.configure_drop_lock(handle, message.prefix_keep_mask[:5].to(torch.bool))
    manager.lock(handle)
    req = Req(
        input_ids=message.input_ids, true_positions=message.true_positions,
        raw_positions=message.raw_positions, radix_input_ids=message.radix_input_ids,
        radix_match_ids=message.radix_match_ids, initial_full_match_indices=original.clone(),
        initial_active_cached_len=3, true_seq_len=int(message.radix_next_position),
        table_idx=0, cached_len=3, output_len=1, uid=message.uid,
        sampling_params=message.sampling_params, cache_handle=handle,
        prompt_tokens=message.prompt_tokens, prefix_keep_mask=message.prefix_keep_mask,
        radix_key_virtual_mask=message.radix_key_virtual_mask,
        radix_key_to_token=message.radix_key_to_token,
        radix_token_to_key=message.radix_token_to_key,
        radix_positions=message.radix_positions, radix_repos_info=message.radix_repos_info,
        radix_next_position=message.radix_next_position,
        radix_current_reposition=message.radix_current_reposition,
    )
    table[0, :3] = original[2:]
    manager._free(manager.prefix_cache.evict(2))
    reused = manager._allocate(3)
    table[0, 3] = reused[0]
    assert set(reused[1:].tolist()) == set(original[:2].tolist())
    req.complete_one()
    req.append_host(torch.tensor([100], dtype=torch.int32))
    manager.cache_req(req, finished=True)
    matched = manager.prefix_cache.match_prefix(
        message.radix_match_ids[:key_end], virtual).cuda_handle
    assert matched.get_matched_indices()[~virtual][:2].tolist() == [-1, -1]
    manager._free(reused[1:])
    manager.check_integrity()
    manager._free(manager.prefix_cache.evict(manager.prefix_cache.evictable_size))
    assert sorted(manager.free_slots.tolist()) == list(range(6))
