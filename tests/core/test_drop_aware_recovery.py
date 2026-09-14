import torch

from minisgl.scheduler.drop_recovery import plan_recovery


def test_recovery_matches_visibility_dependency_oracle():
    import random

    rng = random.Random(1701)
    for _ in range(256):
        length = rng.randrange(1, 18)
        matched = rng.randrange(length)
        resident = [bool(rng.randrange(2)) for _ in range(matched)]
        rewind = [bool(rng.randrange(2)) for _ in range(matched)]
        incompatible = [bool(rng.randrange(2)) for _ in range(matched)]
        expiry = [rng.randrange(raw + 1, length + 2) for raw in range(length)]
        needed = set(range(matched, length))
        # Explicit query-to-key edges provide an independent small-graph oracle.
        for query in range(length - 1, -1, -1):
            if query not in needed:
                continue
            for raw in range(min(query, matched)):
                if expiry[raw] > query and (
                    not resident[raw] or incompatible[raw]
                    or (query < matched and rewind[raw])
                ):
                    needed.add(raw)
        plan = plan_recovery(torch.tensor(resident, dtype=torch.bool),
                             torch.tensor(expiry), length,
                             torch.tensor(rewind, dtype=torch.bool),
                             torch.tensor(incompatible, dtype=torch.bool))
        assert {raw for a, b in plan.intervals for raw in range(a, b)} == needed
        assert plan.required_prefix.tolist() == [
            raw in needed or any(raw < query < expiry[raw] for query in needed)
            for raw in range(matched)]
        assert plan.reusable_prefix.tolist() == [
            present and raw not in needed for raw, present in enumerate(resident)]


def test_recursive_dependencies_reuse_resident_suffix():
    plan = plan_recovery(
        torch.tensor([True, False, True, False, True, True]),
        torch.tensor([10, 4, 10, 10, 10, 10, 10, 10]), 8,
    )
    assert plan.intervals == ((1, 2), (3, 4), (6, 8))
    assert plan.required_prefix.tolist() == [True] * 6
    assert plan.next_interval(2) == (3, 4)


def test_dropped_holes_do_not_trigger_repair():
    plan = plan_recovery(torch.tensor([True, False, False, True]),
                         torch.tensor([10, 4, 4, 10, 10]), 5)
    assert plan.intervals == ((4, 5),)
    assert plan.required_prefix.tolist() == [True, False, False, True]


def test_dependency_drop_at_query_boundary_is_invisible():
    plan = plan_recovery(torch.tensor([False, True, False]),
                         torch.tensor([2, 10, 10, 10]), 4)
    assert plan.intervals == ((2, 4),)
    assert plan.required_prefix.tolist() == [False, True, True]


def test_historical_repair_restores_changed_source_versions_only_before_holes():
    resident = torch.tensor([True, False, True, False, True, True])
    changed = torch.tensor([False, False, True, False, True, True])
    expiry = torch.full((8,), 9)
    plan = plan_recovery(resident, expiry, 8, changed)
    assert plan.intervals == ((1, 4), (6, 8))
    assert plan.reusable_prefix.tolist() == [True, False, False, False, True, True]
    assert plan_recovery(torch.ones(6, dtype=torch.bool), expiry, 8, changed).intervals == ((6, 8),)


def test_future_repair_reserves_repositioned_terminal_copy():
    from test_reposition_chunk_capacity import _pending
    from minisgl.scheduler.drop_recovery import build_drop_capacity_index

    req = _pending(702)
    resident = torch.tensor([True, False, True, False, True, True, True])
    req.drop_recovery_plan = plan_recovery(resident, req.full_token_visible_until, 8)
    source = req.radix_positions[:7]
    owned = torch.zeros(8, dtype=torch.bool)
    _, _, _, future = build_drop_capacity_index(req, owned, source, 7, 1, 2)
    # After repairing raw 1, raw 3 still needs both its birth and terminal
    # versions; raw 7 needs one page, and generation needs one page.
    assert future.tolist() == [4]


def test_retry_version_repair_keeps_position_compatible_suffix():
    resident = torch.tensor([True, False, True, True, True, True])
    changed = torch.tensor([False, False, True, True, True, True])
    incompatible = torch.tensor([False, False, False, False, True, False])
    plan = plan_recovery(resident, torch.full((8,), 9), 8, changed, incompatible)
    assert plan.intervals == ((1, 5), (6, 8))
    assert plan.reusable_prefix.tolist() == [True, False, False, False, False, True]


def test_unused_holes_do_not_turn_retry_into_recovery():
    from test_reposition_chunk_capacity import _manager, _pending
    import minisgl.core as core

    previous = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    try:
        _, cache, _, _ = _manager(64, drop_aware=True)
        pending = _pending(703)
        keys = pending.radix_match_ids[:7].clone()
        keys[:, 3] += 1  # All source positions differ from the requested version.
        values = torch.tensor([-1, 0, -1, 1, 2, 3, 4], dtype=torch.int32)
        values[values >= 0] = cache._allocate(5)
        handle = cache.prefix_cache.insert_prefix(keys, values).handle
        cache._derive_active_match(pending, handle, values)
        assert pending.drop_recovery_plan.intervals == ((7, 8),)
        assert pending.drop_recovery_plan.required_prefix.tolist() == [False, True, False, True, True, True, True]
    finally:
        core._GLOBAL_CTX = previous


def test_occurrence_repair_skips_resident_suffix_and_drains(monkeypatch):
    from test_reposition_chunk_capacity import (
        _manager, _pending, _complete_intermediate_chunk,
    )
    import minisgl.core as core
    from minisgl.scheduler.prefill import ChunkedReq
    from minisgl.scheduler.scheduler import Scheduler
    from minisgl.attention.base import build_occurrence_attention_batch
    from types import SimpleNamespace

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    previous = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    core.get_global_ctx().attn_backend = SimpleNamespace(supports_multi_context_mask_prefill=True)
    try:
        manager, cache, table, _ = _manager(64, drop_aware=True)
        pending = _pending(701)
        pending.context_post_prefill_keep_mask = pending.full_keep_mask
        pages = cache._allocate(5)
        values = torch.full((7,), -1, dtype=torch.int32)
        values[torch.tensor([0, 2, 4, 5, 6])] = pages
        cache.prefix_cache.insert_prefix(pending.radix_match_ids[:7], values,
                                         pending.radix_key_virtual_mask[:7])
        manager.pending_list.append(pending)
        intervals = []
        while manager.pending_list:
            batch = manager.schedule_next_batch(8)
            assert batch is not None
            req = batch.reqs[0]
            intervals.append((req.cached_len, req.device_len))
            metadata = build_occurrence_attention_batch(batch.reqs)
            assert metadata is not None
            assert torch.all(metadata.direct_pages[metadata.key_positions.long()] >= 0)
            sliding = build_occurrence_attention_batch(batch.reqs, sliding_window=128)
            assert torch.all(sliding.direct_pages[sliding.key_positions.long()] >= 0)
            # All pages selected by attention must be resident.
            assert torch.all(req.occurrence_pages[req.occurrence_birth_indices[
                req.cached_len:req.device_len].long()] >= 0)
            if isinstance(req, ChunkedReq):
                _complete_intermediate_chunk(manager, req)
            else:
                req.complete_one()
                scheduler = object.__new__(Scheduler)
                scheduler.cache_manager = cache
                scheduler.table_manager = table
                scheduler._release_occurrence_transients(req)
                scheduler._compact_context_after_prefill(req)
                assert torch.all(table.page_table[req.table_idx, :req.cached_len] >= 0)
                cache.cache_req(req, finished=True)
                table.free(req.table_idx)
        assert intervals == [(1, 4), (7, 8)]
        cache._free(cache.prefix_cache.evict(cache.prefix_cache.evictable_size))
        assert len(cache.free_slots) == cache.num_pages
    finally:
        core._GLOBAL_CTX = previous


def test_completed_recovery_chunk_releases_expired_borrowed_pages():
    from types import SimpleNamespace
    import minisgl.core as core
    from minisgl.scheduler.drop_recovery import RecoveryPlan
    from minisgl.scheduler.scheduler import Scheduler
    from test_reposition_chunk_capacity import _manager

    previous = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    try:
        manager, cache, table, _ = _manager(16, drop_aware=True)
        keys = torch.tensor([[0, 10, -1, 0], [0, 11, -1, 1], [0, 12, -1, 2],
                             [0, 13, -1, 3], [1, -2, -4, -1], [0, 14, -1, 4]],
                            dtype=torch.int32)
        pages = cache._allocate(5)
        values = torch.tensor([pages[0], pages[1], pages[2], -1, -1, pages[4]],
                              dtype=torch.int32)
        handle = cache.prefix_cache.insert_prefix(keys, values, keys[:, 0] != 0).handle
        cache.prefix_cache.configure_drop_lock(handle, torch.ones(5, dtype=torch.bool))
        cache.lock(handle)
        slot = table.allocate()
        table.page_table[slot, :5] = pages
        chunk = SimpleNamespace(
            uid=704, reposition_execution_mode="paged-occurrence", occurrence_inflight=True,
            occurrence_transient_pages=torch.empty(0, dtype=torch.int32),
            drop_recovery_plan=RecoveryPlan(((3, 4), (5, 6)), torch.ones(5, dtype=torch.bool),
                                           5, torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool)),
            drop_recovery_query_start=3, cached_len=4, occurrence_birth_indices=torch.arange(6),
            occurrence_positions=torch.arange(6), occurrence_initial_source_positions=torch.arange(5),
            full_token_visible_until=torch.tensor([9, 4, 4, 9, 9, 9]),
            occurrence_birth_owned_mask=torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool),
            occurrence_birth_pages=torch.cat([pages, torch.tensor([-1], dtype=torch.int32)]),
            cache_handle=handle, initial_full_match_indices=values[keys[:, 0] == 0].clone(),
            table_idx=slot, inactive_cached_pages=None,
            occurrence_terminal_owned_mask=torch.zeros(6, dtype=torch.bool),
        )
        assert cache.prefix_cache.evictable_size == 0
        manager.complete_chunk(chunk)
        assert cache.prefix_cache.evictable_size == 2
        assert handle.skip_ranges == [(1, 3)]
        assert chunk.initial_full_match_indices[1:3].tolist() == [-1, -1]
        assert chunk.occurrence_birth_pages[1:3].tolist() == [-1, -1]
        assert table.occurrence_pages(slot)[1:3].tolist() == [-1, -1]
        cache._free(cache.prefix_cache.evict(2))
        # Cancellation after the early release must only free the remaining lease.
        scheduler = object.__new__(Scheduler)
        scheduler.cache_manager, scheduler.table_manager = cache, table
        scheduler._close_context_sequence = lambda uid: None
        scheduler._free_aborted_occurrence_resources(chunk)
        cache._free(cache.prefix_cache.evict(cache.prefix_cache.evictable_size))
        assert len(cache.free_slots) == cache.num_pages
    finally:
        core._GLOBAL_CTX = previous
