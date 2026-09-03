from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi")

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.kernel.radix_reposition import compile_radix_reposition_layout
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder, PrefillManager
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    yield
    core._GLOBAL_CTX = old_ctx


class _RecordingKVCache:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def retry_reposition(self, source, destination, transitions, rope_cache) -> None:
        self.calls.append((source.clone(), destination.clone(), transitions.clone()))


def _visible_until(
    token_count: int,
    event_positions: torch.Tensor,
    range_offsets: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    result = torch.full((token_count,), token_count + 1, dtype=torch.int32)
    for event, position in enumerate(event_positions.tolist()):
        for range_index in range(int(range_offsets[event]), int(range_offsets[event + 1])):
            start = int(ranges[2 * range_index])
            end = int(ranges[2 * range_index + 1])
            result[start:end] = position
    return result


def _pending(*, interior_drop: bool = False) -> PendingReq:
    token_ids = torch.arange(6, dtype=torch.int32)
    drops = torch.tensor([3, 4 if interior_drop else 5], dtype=torch.int32)
    drop_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    ranges = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
    repositions = torch.tensor([1, 2, 4], dtype=torch.int32)
    layout = compile_radix_reposition_layout(
        token_ids,
        drops,
        drop_offsets,
        ranges,
        torch.tensor([-101, -102], dtype=torch.int32),
        repositions,
        repositions + 1,
    )
    active_raw = torch.nonzero(layout.keep_mask, as_tuple=False).view(-1)
    return PendingReq(
        uid=7,
        input_ids=token_ids[active_raw],
        true_positions=layout.positions[active_raw],
        raw_positions=active_raw.to(torch.int32),
        radix_input_ids=layout.records[layout.token_to_key[active_raw]],
        radix_match_ids=layout.records,
        sampling_params=SamplingParams(max_tokens=1),
        prompt_tokens=len(token_ids),
        prefix_keep_mask=layout.keep_mask[:-1].to(torch.int32),
        full_input_ids=token_ids,
        full_token_visible_until=_visible_until(
            len(token_ids), drops, drop_offsets, ranges
        ),
        full_keep_mask=layout.keep_mask.to(torch.int32),
        drop_event_positions=drops,
        drop_range_offsets=drop_offsets,
        drop_position_ranges=ranges,
        drop_effective_event_count=2,
        radix_key_virtual_mask=layout.virtual_mask,
        radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key,
        radix_marker_ids=(-101, -102),
        radix_positions=layout.positions,
        radix_repos_info=layout.repos_info,
        radix_materialized_stage=layout.materialized_stage,
        reposition_raw_boundaries=repositions,
        reposition_input_ids=token_ids,
        reposition_birth_positions=layout.birth_positions,
        reposition_birth_stages=layout.birth_stages,
        reposition_transition_offsets=layout.transition_offsets,
        reposition_transition_raw_tokens=layout.transition_raw_tokens,
        reposition_transition_old_positions=layout.transition_old_positions,
        reposition_transition_new_positions=layout.transition_new_positions,
        reposition_effective_stages=layout.effective_reposition_stages,
        reposition_insert_offsets=repositions + 1,
        radix_next_position=layout.next_position,
        radix_current_reposition=layout.current_reposition,
        context_stage_count=len(layout.transition_offsets),
    )


def _pending_noop_reposition_after_tail_drop() -> PendingReq:
    token_ids = torch.arange(4, dtype=torch.int32)
    drops = torch.tensor([3], dtype=torch.int32)
    ranges = torch.tensor([2, 3], dtype=torch.int32)
    drop_offsets = torch.tensor([0, 1], dtype=torch.int32)
    repositions = torch.tensor([2], dtype=torch.int32)
    layout = compile_radix_reposition_layout(
        token_ids,
        drops,
        drop_offsets,
        ranges,
        torch.tensor([-201], dtype=torch.int32),
        repositions,
        repositions + 1,
    )
    active_raw = torch.nonzero(layout.keep_mask, as_tuple=False).view(-1)
    assert layout.transition_offsets.tolist() == [0]
    return PendingReq(
        uid=8,
        input_ids=token_ids[active_raw],
        true_positions=layout.positions[active_raw],
        raw_positions=active_raw.to(torch.int32),
        radix_input_ids=layout.records[layout.token_to_key[active_raw]],
        radix_match_ids=layout.records,
        sampling_params=SamplingParams(max_tokens=1),
        prompt_tokens=len(token_ids),
        prefix_keep_mask=layout.keep_mask[:-1].to(torch.int32),
        full_input_ids=token_ids,
        full_token_visible_until=_visible_until(
            len(token_ids), drops, drop_offsets, ranges
        ),
        full_keep_mask=layout.keep_mask.to(torch.int32),
        drop_event_positions=drops,
        drop_range_offsets=drop_offsets,
        drop_position_ranges=ranges,
        drop_effective_event_count=1,
        use_context_mask=True,
        radix_key_virtual_mask=layout.virtual_mask,
        radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key,
        radix_marker_ids=(-201,),
        radix_positions=layout.positions,
        radix_repos_info=layout.repos_info,
        radix_materialized_stage=layout.materialized_stage,
        reposition_raw_boundaries=repositions,
        reposition_insert_offsets=repositions + 1,
        reposition_input_ids=token_ids,
        reposition_birth_positions=layout.birth_positions,
        reposition_birth_stages=layout.birth_stages,
        reposition_transition_offsets=layout.transition_offsets,
        reposition_transition_raw_tokens=layout.transition_raw_tokens,
        reposition_transition_old_positions=layout.transition_old_positions,
        reposition_transition_new_positions=layout.transition_new_positions,
        reposition_effective_stages=layout.effective_reposition_stages,
        radix_next_position=layout.next_position,
        radix_current_reposition=layout.current_reposition,
        context_stage_count=len(layout.transition_offsets),
    )


def _manager() -> tuple[PrefillManager, CacheManager, TableManager, _RecordingKVCache]:
    page_table = torch.full((2, 64), -1, dtype=torch.int32)
    cache = CacheManager(64, 1, page_table, "radix")
    table = TableManager(2, page_table)
    kv_cache = _RecordingKVCache()
    manager = PrefillManager(
        cache_manager=cache,
        table_manager=table,
        decode_manager=SimpleNamespace(inflight_tokens=0),
        kv_cache=kv_cache,
        retry_rope_cache=torch.zeros((64, 2), dtype=torch.float32),
    )
    return manager, cache, table, kv_cache


def _materialize_batch(
    manager: PrefillManager,
    cache: CacheManager,
    *,
    prefill_budget: int = 64,
    lazy_free: bool = False,
):
    batch = manager.schedule_next_batch(prefill_budget=prefill_budget)
    assert batch is not None and len(batch.reqs) == 1
    req = batch.reqs[0]
    cache.allocate_paged(batch.reqs)
    req.complete_one()
    if isinstance(req, ChunkedReq):
        if lazy_free:
            with cache.lazy_free_region():
                manager.complete_chunk(req)
        else:
            manager.complete_chunk(req)
    return req


def test_cold_reposition_prefill_materializes_each_stage_before_final_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, kv_cache = _manager()
    pending = _pending()
    assert pending.reposition_effective_stages.tolist() == [-1, 1, 2]
    manager.pending_list.append(pending)

    first = _materialize_batch(manager, cache)
    assert isinstance(first, ChunkedReq)
    assert first.true_positions[: first.cached_len].tolist() == [0, 1, 2]
    assert pending.staged_actual_stage == 1
    assert pending.staged_active_raw.tolist() == [0, 2]

    second = _materialize_batch(manager, cache)
    assert isinstance(second, ChunkedReq)
    assert second.true_positions[: second.cached_len].tolist() == [0, 1, 2, 3]
    assert pending.staged_actual_stage == 2
    assert pending.staged_active_raw.tolist() == [2, 3, 4]

    final = _materialize_batch(manager, cache)
    assert not isinstance(final, ChunkedReq)
    assert final.true_positions[: final.cached_len].tolist() == [0, 1, 2, 3]
    assert final.radix_actual_materialized_stage == 2
    assert final.staged_full_page_indices is not None
    assert final.staged_full_page_indices[-1].item() == -1
    assert [call[2].tolist() for call in kv_cache.calls] == [
        [[2, 1]],
        [[1, 0], [2, 1], [3, 2]],
    ]

    final.append_host(torch.tensor([99], dtype=torch.int32))
    cache.cache_req(final, finished=True)
    assert bool(torch.all(final.staged_full_page_indices >= 0).item())
    table.free(final.table_idx)
    cache.check_integrity()


def test_aborted_inflight_staged_chunk_releases_after_completion(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, _ = _manager()
    pending = _pending()
    manager.pending_list.append(pending)

    batch = manager.schedule_next_batch(prefill_budget=64)
    assert batch is not None
    chunk = batch.reqs[0]
    assert isinstance(chunk, ChunkedReq)
    cache.allocate_paged(batch.reqs)
    chunk.complete_one()

    aborted = manager.abort_req(pending.uid)
    assert aborted is pending
    assert id(chunk) in manager.aborted_staged_chunks
    assert table.available_size == 1

    manager.complete_chunk(chunk)

    assert manager.aborted_staged_chunks == {}
    assert table.available_size == 2
    assert len(cache.free_slots) == cache.num_pages
    cache.check_integrity()


def test_noop_reposition_keeps_drop_on_one_pass_mask_planner(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, kv_cache = _manager()
    pending = _pending_noop_reposition_after_tail_drop()
    manager.pending_list.append(pending)

    final = _materialize_batch(manager, cache)
    assert not isinstance(final, ChunkedReq)
    assert final.raw_positions[: final.cached_len].tolist() == [0, 1, 3]
    assert final.true_positions[: final.cached_len].tolist() == [0, 1, 3]
    assert not final.staged_reposition
    assert not final.use_context_mask
    assert kv_cache.calls == []

    cache.cache_req(final, finished=True)
    assert final.staged_full_page_indices is None
    table.free(final.table_idx)
    cache.check_integrity()


def test_drop_inside_reposition_epoch_uses_mask_without_extra_stage(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, _ = _manager()
    pending = _pending(interior_drop=True)
    manager.pending_list.append(pending)

    first = _materialize_batch(manager, cache)
    assert isinstance(first, ChunkedReq)
    assert not first.use_context_mask
    assert pending.staged_actual_stage == 1
    assert pending.staged_segment_end == 5

    masked = _materialize_batch(manager, cache)
    assert isinstance(masked, ChunkedReq)
    assert masked.use_context_mask
    assert masked.raw_positions[: masked.cached_len].tolist() == [0, 2, 3, 4]
    assert pending.staged_actual_stage == 2
    assert pending.staged_active_raw.tolist() == [2, 3, 4]

    final = _materialize_batch(manager, cache)
    assert not isinstance(final, ChunkedReq)
    assert not final.use_context_mask
    final.append_host(torch.tensor([99], dtype=torch.int32))
    cache.cache_req(final, finished=True)
    table.free(final.table_idx)
    cache.check_integrity()


def _seed_raw_prefix(
    cache: CacheManager,
    pending: PendingReq,
    raw_cursor: int,
) -> torch.Tensor:
    assert pending.radix_token_to_key is not None
    assert pending.radix_key_virtual_mask is not None
    assert pending.radix_match_ids is not None
    key_len = int(pending.radix_token_to_key[raw_cursor])
    virtual_mask = pending.radix_key_virtual_mask[:key_len]
    pages = cache._allocate(raw_cursor)
    key_pages = torch.full((key_len,), -1, dtype=torch.int32)
    key_pages[~virtual_mask] = pages
    cache.prefix_cache.insert_prefix(
        pending.radix_match_ids[:key_len],
        key_pages,
        virtual_mask,
    )
    return pages


def test_partial_radix_seed_reuses_pages_and_counts_drop_skipped(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, kv_cache = _manager()
    pending = _pending()
    source_pages = _seed_raw_prefix(cache, pending, raw_cursor=3)
    manager.pending_list.append(pending)

    first = _materialize_batch(manager, cache)
    assert isinstance(first, ChunkedReq)
    assert first.radix_cached_tokens == 3
    assert first.usage_cached_tokens == 2
    assert first.drop_skipped_tokens == 1
    assert first.cached_len == 4
    assert first.initial_active_cached_len == 0
    assert len(kv_cache.calls) >= 1
    assert set(source_pages.tolist()).isdisjoint(set(cache.free_slots.tolist()))

    final = _materialize_batch(manager, cache)
    assert not isinstance(final, ChunkedReq)
    assert final.initial_active_cached_len == 0
    # One seed transform (16 B metadata), one two-token seed index vector
    # (16 B), two incremental raw indices (16 B), three R transitions
    # (36 B), and the three-token post-transition active index vector (24 B).
    assert final.reposition_h2d_bytes == 108
    assert final.reposition_d2h_bytes == 0
    final.append_host(torch.tensor([99], dtype=torch.int32))
    cache.cache_req(final, finished=True)
    assert set(source_pages.tolist()).isdisjoint(set(cache.free_slots.tolist()))
    table.free(final.table_idx)
    cache.check_integrity()


def test_deferred_staged_allocation_restores_request_and_rematches(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, _, _ = _manager()
    pending = _pending()
    _seed_raw_prefix(cache, pending, raw_cursor=3)
    original_stream = (
        pending.input_ids.clone(),
        pending.true_positions.clone(),
        pending.raw_positions.clone(),
        pending.radix_input_ids.clone(),
        pending.use_context_mask,
    )
    manager.pending_list.append(pending)

    original_allocate = PrefillAdder._allocate_staged
    allocation_attempts = 0
    match_attempts = 0
    original_match = cache.match_req

    def defer_once(adder, req, match):
        nonlocal allocation_attempts
        allocation_attempts += 1
        if allocation_attempts == 1:
            return None
        return original_allocate(adder, req, match)

    def count_match(req):
        nonlocal match_attempts
        match_attempts += 1
        return original_match(req)

    monkeypatch.setattr(PrefillAdder, "_allocate_staged", defer_once)
    monkeypatch.setattr(cache, "match_req", count_match)

    assert manager.schedule_next_batch(prefill_budget=64) is None
    assert not pending.staged_reposition
    assert pending.staged_cache_handle is None
    assert pending.staged_table_idx is None
    for actual, expected in zip(
        (
            pending.input_ids,
            pending.true_positions,
            pending.raw_positions,
            pending.radix_input_ids,
        ),
        original_stream[:4],
        strict=True,
    ):
        assert torch.equal(actual, expected)
    assert pending.use_context_mask is original_stream[4]

    first = _materialize_batch(manager, cache)
    assert isinstance(first, ChunkedReq)
    assert allocation_attempts == 2
    assert match_attempts == 2
    manager.pending_list.remove(pending)
    manager.release_staged(pending)
    cache.check_integrity()


def test_transition_failure_rolls_back_owned_pages_without_leak(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, _, kv_cache = _manager()
    pending = _pending()
    manager.pending_list.append(pending)

    batch = manager.schedule_next_batch(prefill_budget=64)
    assert batch is not None
    chunk = batch.reqs[0]
    assert isinstance(chunk, ChunkedReq)
    cache.allocate_paged(batch.reqs)
    chunk.complete_one()
    free_before_transition = len(cache.free_slots)

    def fail_retry(*_args) -> None:
        raise RuntimeError("injected Retry RoPE failure")

    monkeypatch.setattr(kv_cache, "retry_reposition", fail_retry)
    with pytest.raises(RuntimeError, match="injected Retry RoPE failure"):
        manager.complete_chunk(chunk)

    assert pending.staged_raw_cursor == 0
    assert pending.staged_actual_stage == 0
    assert len(cache.free_slots) == free_before_transition
    assert pending.staged_owned_page_mask is not None
    assert int(torch.count_nonzero(pending.staged_owned_page_mask)) == chunk.cached_len
    manager.pending_list.remove(pending)
    manager.release_staged(pending)
    cache.check_integrity()


def test_multichunk_staged_overlap_stop_preserves_every_page(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, _ = _manager()
    pending = _pending()
    pending.sampling_params.max_tokens = 12
    manager.pending_list.append(pending)

    while True:
        req = _materialize_batch(
            manager,
            cache,
            prefill_budget=2,
            lazy_free=True,
        )
        if not isinstance(req, ChunkedReq):
            break

    for token_id in (90, 91):
        cache.allocate_paged([req])
        req.complete_one()
        req.append_host(torch.tensor([token_id], dtype=torch.int32))
    assert req.can_decode
    generated_raw = req.raw_positions[len(req.input_ids) - 2 : len(req.input_ids)]
    assert generated_raw.tolist() == [6, 7]
    assert req.radix_positions[-2:].tolist() == [4, 5]
    assert req.radix_match_ids[-2:, 3].tolist() == [4, 5]

    with cache.lazy_free_region():
        cache.cache_req(req, finished=True)
    table.free(req.table_idx)
    cache.check_integrity()
