from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.scheduler.cache import CacheManager, ContextMatchResult, FullMatchResult
from minisgl.scheduler.prefill import PrefillAdder, PrefillManager, _mask_free_context_reason
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _pending_req() -> PendingReq:
    full_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    keep_mask = torch.tensor([1, 0, 0, 1, 1], dtype=torch.int32)
    active_positions = torch.tensor([0, 3, 4], dtype=torch.int32)
    return PendingReq(
        uid=7,
        input_ids=full_ids[active_positions.to(torch.int64)],
        true_positions=active_positions,
        radix_input_ids=full_ids[active_positions.to(torch.int64)].to(torch.int64),
        radix_match_ids=full_ids.to(torch.int64),
        sampling_params=SamplingParams(max_tokens=1),
        prompt_tokens=len(full_ids),
        is_warmup=True,
        prefix_keep_mask=keep_mask[:-1],
        full_input_ids=full_ids,
        full_token_visible_until=torch.tensor(
            [torch.iinfo(torch.int32).max, 4, 4, torch.iinfo(torch.int32).max,
             torch.iinfo(torch.int32).max],
            dtype=torch.int32,
        ),
        full_keep_mask=keep_mask,
        drop_event_positions=torch.tensor([4], dtype=torch.int32),
        drop_range_offsets=torch.tensor([0, 1], dtype=torch.int32),
        drop_position_ranges=torch.tensor([1, 3], dtype=torch.int32),
        drop_effective_event_count=1,
        use_context_mask=True,
    )


def test_mask_free_context_accepts_drop_before_first_uncached_active_token() -> None:
    req = _pending_req()

    assert (
        _mask_free_context_reason(
            req,
            active_cached_len=2,
            has_sliding_window=False,
        )
        is None
    )


def test_mask_free_context_rejects_token_that_still_needs_dropped_context() -> None:
    req = _pending_req()

    assert _mask_free_context_reason(
        req,
        active_cached_len=1,
        has_sliding_window=False,
    ) == "visibility_changes_within_extend"


def test_mask_free_context_rejects_sliding_window_models() -> None:
    req = _pending_req()

    assert _mask_free_context_reason(
        req,
        active_cached_len=2,
        has_sliding_window=True,
    ) == "sliding_window_requires_absolute_key_selection"


def test_drop_aware_full_match_derives_active_pages_across_dropped_holes() -> None:
    core.set_global_ctx(core.Context(page_size=1))
    cache_manager = CacheManager(
        num_pages=32,
        page_size=1,
        page_table=torch.full((2, 32), -1, dtype=torch.int32),
        type="radix",
        drop_aware_eviction=True,
    )
    handle = cache_manager.prefix_cache.match_prefix(
        torch.empty(0, dtype=torch.int64)
    ).cuda_handle
    full_match = FullMatchResult(
        handle=handle,
        full_match_indices=torch.tensor([20, -1, -1, 23], dtype=torch.int32),
        full_cached_len=4,
        safe_match_indices=torch.tensor([20], dtype=torch.int32),
        safe_cached_len=1,
    )
    req = SimpleNamespace(
        prefix_keep_mask=torch.tensor([1, 0, 0, 1], dtype=torch.int32)
    )

    active = cache_manager.derive_active_match(req, full_match)

    assert active.active_cached_len == 2
    assert active.active_match_indices.tolist() == [20, 23]
    assert active.handle.pinned_slots == (20, 23)


class _PlanCache:
    def __init__(self, active_cached_len: int):
        self.handle = object()
        self.full_match = FullMatchResult(
            handle=self.handle,
            full_match_indices=torch.tensor([20, 21, 22, 23], dtype=torch.int32),
            full_cached_len=4,
            safe_match_indices=torch.tensor([20, 21, 22, 23], dtype=torch.int32),
            safe_cached_len=4,
        )
        active_indices = torch.tensor([20, 23], dtype=torch.int32)[:active_cached_len]
        self.active_match = ContextMatchResult(
            handle=self.handle,
            full_match_indices=self.full_match.full_match_indices,
            full_cached_len=4,
            active_match_indices=active_indices,
            active_cached_len=active_cached_len,
        )

    def match_full_req(self, req):
        return self.full_match

    def derive_active_match(self, req, full_match):
        assert full_match is self.full_match
        return self.active_match

    @property
    def available_size(self):
        return 128

    def matchable_prefix_lens(self, req):
        return 4, 2

    def lock(self, handle):
        assert handle is self.handle

    def unlock(self, handle):
        assert handle is self.handle


def test_context_prefill_plan_executes_direct_or_mask_from_same_full_lookup() -> None:
    req = _pending_req()
    direct_cache = _PlanCache(active_cached_len=2)
    direct_adder = PrefillAdder(
        token_budget=16,
        reserved_size=0,
        cache_manager=direct_cache,
        table_manager=SimpleNamespace(),
    )

    direct = direct_adder.plan_context_prefill(req)

    assert direct is not None
    assert not direct.use_context_mask
    assert direct.cached_indices.tolist() == [20, 23]
    assert direct.true_positions.tolist() == [0, 3, 4]
    assert direct.drop_skipped_tokens == 2

    mask_cache = _PlanCache(active_cached_len=1)
    mask_adder = PrefillAdder(
        token_budget=16,
        reserved_size=0,
        cache_manager=mask_cache,
        table_manager=SimpleNamespace(),
    )

    fallback = mask_adder.plan_context_prefill(_pending_req())

    assert fallback is not None
    assert fallback.use_context_mask
    assert fallback.cached_len == 4
    assert fallback.true_positions.tolist() == [0, 1, 2, 3, 4]
    assert fallback.reason == "visibility_changes_within_extend"
    assert fallback.drop_skipped_tokens == 0


def test_context_prefill_fast_path_can_be_disabled_for_ablation() -> None:
    adder = PrefillAdder(
        token_budget=16,
        reserved_size=0,
        cache_manager=_PlanCache(active_cached_len=2),
        table_manager=SimpleNamespace(),
        enable_mask_free_context_prefill=False,
    )

    plan = adder.plan_context_prefill(_pending_req())

    assert plan is not None
    assert plan.use_context_mask
    assert plan.reason == "mask_free_disabled"
    assert plan.drop_skipped_tokens == 0


def test_two_eligible_context_warmups_batch_as_ordinary_extend(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    cache = _PlanCache(active_cached_len=2)
    table = SimpleNamespace(
        available_size=4,
        token_pool=torch.zeros((4, 16), dtype=torch.int32),
        page_table=torch.full((4, 16), -1, dtype=torch.int32),
    )
    allocated_rows = iter(range(4))
    table.allocate = lambda: next(allocated_rows)
    table.free = lambda _idx: None
    manager = PrefillManager(
        cache_manager=cache,
        table_manager=table,
        decode_manager=SimpleNamespace(inflight_tokens=0),
    )
    req_a = _pending_req()
    req_b = _pending_req()
    req_b.uid = 8
    manager.pending_list.extend((req_a, req_b))

    batch = manager.schedule_next_batch(prefill_budget=16)

    assert batch is not None
    assert len(batch.reqs) == 2
    assert all(not req.use_context_mask for req in batch.reqs)
    assert all(req.full_input_ids is None for req in batch.reqs)
    assert [req.drop_skipped_tokens for req in batch.reqs] == [2, 2]
    assert [req.true_positions.tolist() for req in batch.reqs] == [[0, 3, 4], [0, 3, 4]]
