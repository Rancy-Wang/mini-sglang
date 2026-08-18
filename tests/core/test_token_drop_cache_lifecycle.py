from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.message import UserMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import _calculate_cache_ratios
from minisgl.scheduler.radix_delta import DeltaMarkerRegistry, inject_delta_markers
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.utils import PendingReq


def test_token_position_branch_caches_matches_evicts_and_releases_registry():
    previous_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = core.Context(page_size=1)
    try:
        page_table = torch.full((1, 32), -1, dtype=torch.int32)
        cache = CacheManager(32, 1, page_table, type="radix")
        registry = DeltaMarkerRegistry()
        cache.bind_delta_marker_registry(registry)

        full_ids = torch.arange(100, 108, dtype=torch.int64)
        no_drop_slots = torch.arange(8, dtype=torch.int32)
        cache.prefix_cache.insert_prefix(full_ids, no_drop_slots)
        cache.free_slots = cache.free_slots[8:]

        layout = inject_delta_markers(
            full_ids,
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([2, 4], dtype=torch.int32),
            registry,
        )
        assert layout is not None
        keep_mask = torch.tensor([1, 1, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
        true_positions = torch.tensor([0, 1, 4, 5, 6, 7], dtype=torch.int32)
        active_ids = full_ids[true_positions.to(torch.int64)].to(torch.int32)
        pending = PendingReq(
            uid=1,
            input_ids=active_ids,
            true_positions=true_positions,
            radix_input_ids=active_ids.to(torch.int64),
            radix_match_ids=layout.keys,
            sampling_params=SamplingParams(max_tokens=1),
            prefix_keep_mask=keep_mask[:-1],
            radix_key_virtual_mask=layout.virtual_mask,
            radix_key_to_token=layout.key_to_token,
            radix_token_to_key=layout.token_to_key,
            radix_marker_ids=layout.marker_ids,
        )
        matched = cache.match_req(pending)
        assert matched is not None
        assert matched.full_cached_len == 5
        assert matched.active_match_indices.tolist() == [0, 1, 4]
        full_prefix_len, active_prefix_len = cache.matchable_prefix_lens(pending)
        assert (full_prefix_len, active_prefix_len) == (7, 5)
        assert _calculate_cache_ratios(
            matched.active_cached_len,
            full_prefix_len,
            active_prefix_len,
        ) == pytest.approx((3 / 7, 3 / 5))

        cache.lock(matched.handle)
        page_table[0].fill_(-1)
        page_table[0, : matched.active_cached_len] = matched.active_match_indices
        req = Req(
            input_ids=active_ids,
            true_positions=true_positions,
            radix_input_ids=active_ids.to(torch.int64),
            radix_match_ids=layout.keys,
            initial_full_match_indices=matched.full_match_indices.clone(),
            initial_active_cached_len=matched.active_cached_len,
            true_seq_len=8,
            table_idx=0,
            cached_len=matched.active_cached_len,
            output_len=1,
            uid=1,
            sampling_params=SamplingParams(max_tokens=1),
            cache_handle=matched.handle,
            prefix_keep_mask=keep_mask[:-1],
            is_warmup=True,
            radix_key_virtual_mask=layout.virtual_mask,
            radix_key_to_token=layout.key_to_token,
            radix_token_to_key=layout.token_to_key,
            radix_marker_ids=layout.marker_ids,
        )
        cache.allocate_paged([req])
        req.complete_one()
        cache.cache_req(req, finished=True)

        rematched = cache.match_req(pending)
        assert rematched is not None
        assert rematched.active_cached_len == len(active_ids) - 1
        assert _calculate_cache_ratios(
            rematched.active_cached_len,
            full_prefix_len,
            active_prefix_len,
        ) == pytest.approx((5 / 7, 1.0))
        assert registry.tree_ref_count == 1
        registry.release_request_refs(layout.marker_ids)
        cache.check_integrity()

        cache._free(cache.prefix_cache.evict(cache.prefix_cache.evictable_size))
        assert registry.size == 0
        cache.check_integrity()
    finally:
        core._GLOBAL_CTX = previous_ctx


def test_cache_ratios_preserve_no_drop_full_hit_and_reject_invalid_lengths():
    assert _calculate_cache_ratios(7, 7, 7) == (1.0, 1.0)
    assert _calculate_cache_ratios(0, 0, 0) == (1.0, 1.0)

    with pytest.raises(ValueError, match="cached <= active_matchable <= full_matchable"):
        _calculate_cache_ratios(4, 7, 3)


def test_warmup_ack_uses_internal_active_cache_reuse_ratio():
    class WarmupReq:
        is_warmup = True
        can_decode = False
        uid = 7
        cache_hit_ratio = 5 / 7
        cache_reuse_ratio = 1.0

    req = WarmupReq()
    replies = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = SimpleNamespace(lazy_free_region=lambda: nullcontext())
    scheduler.decode_manager = SimpleNamespace(remove_req=lambda actual: actual is req)
    scheduler.finished_reqs = set()
    scheduler._free_req_resources = lambda actual: actual is req
    scheduler.send_result = replies.extend
    copy_done = SimpleNamespace(synchronize=lambda: None)
    batch = SimpleNamespace(reqs=[req], is_prefill=True)

    scheduler._process_last_data(
        (SimpleNamespace(batch=batch), (None, torch.tensor([0]), copy_done))
    )

    assert len(replies) == 1
    assert replies[0].hit_ratio == 1.0


def test_scheduler_releases_markers_when_commit_boundary_is_invalid(monkeypatch):
    monkeypatch.setattr(
        Scheduler._process_one_msg.__globals__["logger"],
        "debug_rank0",
        lambda *_args, **_kwargs: None,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=32)
    scheduler.radix_symbol_registry = None
    scheduler.delta_marker_registry = DeltaMarkerRegistry()
    scheduler.prefill_manager = SimpleNamespace(
        add_one_req=lambda _msg: pytest.fail("invalid request reached PrefillManager")
    )

    full_ids = torch.arange(100, 104, dtype=torch.int64)
    msg = UserMsg(
        uid=1,
        input_ids=full_ids.to(torch.int32),
        true_positions=torch.arange(4, dtype=torch.int32),
        radix_input_ids=full_ids,
        radix_match_ids=full_ids.clone(),
        sampling_params=SamplingParams(max_tokens=1),
        drop_event_positions=torch.tensor([4], dtype=torch.int32),
        drop_range_offsets=torch.tensor([0, 1], dtype=torch.int32),
        drop_position_ranges=torch.tensor([1, 3], dtype=torch.int32),
        radix_commit_token_len=5,
    )

    with pytest.raises(ValueError, match="outside token length"):
        scheduler._process_one_msg(msg)

    assert msg.radix_marker_ids is None
    assert scheduler.delta_marker_registry.request_ref_count == 0
    assert scheduler.delta_marker_registry.size == 0
