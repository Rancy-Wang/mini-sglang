import torch

from minisgl.scheduler.drop_recovery import plan_recovery


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
