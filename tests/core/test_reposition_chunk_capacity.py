from __future__ import annotations

from types import SimpleNamespace

import minisgl.core as core
import minisgl.kernel.radix as radix_kernel
import pytest
import torch
from minisgl.attention.base import build_occurrence_attention_batch
from minisgl.core import SamplingParams
from minisgl.kernel.radix_reposition import RadixRepositionLayout
from minisgl.message import AbortBackendMsg, RequestRejectMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import (
    ChunkedReq,
    OccurrenceInputError,
    PrefillAdder,
    PrefillManager,
    RepositionCapacityError,
)
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq
from minisgl.tokenizer.reposition_occurrence import compile_reposition_occurrence_plan


@pytest.fixture(autouse=True)
def reset_global_ctx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

    def matching_rows(left: torch.Tensor, right: torch.Tensor) -> int:
        limit = min(len(left), len(right))
        if limit == 0:
            return 0
        equal = torch.all(left[:limit] == right[:limit], dim=1)
        mismatch = torch.nonzero(~equal, as_tuple=False).view(-1)
        return limit if len(mismatch) == 0 else int(mismatch[0])

    monkeypatch.setattr(
        radix_kernel,
        "radix_record_edge_hash",
        lambda records: hash(tuple(int(value) for value in records[0].tolist())),
    )
    monkeypatch.setattr(
        radix_kernel,
        "radix_record_edge_equal",
        lambda left, right: bool(torch.equal(left[0], right[0])),
    )
    monkeypatch.setattr(
        radix_kernel,
        "radix_record_retry_token",
        lambda records: int(records[0, 1]),
    )
    monkeypatch.setattr(radix_kernel, "fast_compare_radix_records", matching_rows)
    monkeypatch.setattr(radix_kernel, "fast_compare_retry_radix_records", matching_rows)
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    core.get_global_ctx().attn_backend = SimpleNamespace(supports_multi_context_mask_prefill=True)
    yield
    core._GLOBAL_CTX = old_ctx


class _RecordingKVCache:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def retry_reposition(
        self,
        source_pages: torch.Tensor,
        destination_pages: torch.Tensor,
        position_pairs: torch.Tensor,
        _rope_cache: torch.Tensor,
    ) -> None:
        self.calls.append(
            (
                source_pages.clone(),
                destination_pages.clone(),
                position_pairs.clone(),
            )
        )


def _layout() -> RadixRepositionLayout:
    token_count = 8
    positions = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5], dtype=torch.int32)
    records = torch.column_stack(
        (
            torch.zeros(token_count, dtype=torch.int32),
            torch.arange(100, 100 + token_count, dtype=torch.int32),
            torch.zeros(token_count, dtype=torch.int32),
            positions,
        )
    )
    return RadixRepositionLayout(
        drop_insert_offsets=torch.tensor([4, 7], dtype=torch.int32),
        drop_range_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
        drop_ranges=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        records=records,
        virtual_mask=torch.zeros(token_count, dtype=torch.bool),
        key_to_token=torch.arange(token_count, dtype=torch.int64),
        token_to_key=torch.arange(token_count, dtype=torch.int64),
        positions=positions,
        repos_info=torch.zeros(token_count, dtype=torch.int32),
        keep_mask=torch.tensor([False, True, False, True, True, True, True, True]),
        materialized_stage=torch.zeros(token_count, dtype=torch.int32),
        birth_positions=torch.tensor([0, 1, 2, 3, 3, 4, 5, 5], dtype=torch.int32),
        birth_stages=torch.tensor([0, 0, 0, 0, 1, 1, 1, 2], dtype=torch.int32),
        transition_offsets=torch.tensor([0, 3, 7], dtype=torch.int32),
        transition_raw_tokens=torch.tensor([1, 2, 3, 3, 4, 5, 6], dtype=torch.int32),
        transition_old_positions=torch.tensor([1, 2, 3, 2, 3, 4, 5], dtype=torch.int32),
        transition_new_positions=torch.tensor([0, 1, 2, 1, 2, 3, 4], dtype=torch.int32),
        effective_reposition_stages=torch.tensor([1, 2], dtype=torch.int32),
        drop_event_to_key=torch.tensor([-1, -1], dtype=torch.int64),
        effective_repositions=torch.tensor([True, True]),
        ignored_repositions=torch.tensor([False, False]),
        next_position=6,
        current_reposition=1,
        compile_ns=0,
    )


def _pending(uid: int) -> PendingReq:
    layout = _layout()
    visible_until = torch.tensor([4, 9, 7, 9, 9, 9, 9, 9], dtype=torch.int32)
    plan = compile_reposition_occurrence_plan(layout, visible_until)
    input_ids = layout.records[:, 1].clone()
    raw_positions = torch.arange(len(input_ids), dtype=torch.int32)
    return PendingReq(
        uid=uid,
        input_ids=input_ids,
        true_positions=layout.birth_positions,
        raw_positions=raw_positions,
        radix_input_ids=layout.records,
        radix_match_ids=layout.records,
        sampling_params=SamplingParams(max_tokens=1),
        prompt_tokens=len(input_ids),
        is_warmup=True,
        prefix_keep_mask=torch.ones(len(input_ids), dtype=torch.int32),
        full_input_ids=input_ids,
        full_token_visible_until=visible_until,
        full_keep_mask=layout.keep_mask.to(torch.int32),
        use_context_mask=True,
        context_compact_stream=False,
        radix_key_virtual_mask=layout.virtual_mask,
        radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key,
        radix_positions=layout.positions,
        radix_repos_info=layout.repos_info,
        radix_next_position=layout.next_position,
        reposition_execution_mode="paged-occurrence",
        occurrence_raw_tokens=plan.occurrence_raw_tokens,
        occurrence_positions=plan.occurrence_positions,
        occurrence_birth_indices=plan.birth_occurrences,
        occurrence_terminal_indices=plan.terminal_occurrences,
        occurrence_segment_query_starts=plan.segment_query_starts,
        occurrence_segment_query_ends=plan.segment_query_ends,
        occurrence_segment_key_offsets=plan.segment_key_offsets,
        occurrence_segment_key_indices=plan.segment_key_occurrences,
    )


def _manager(num_pages: int, table_count: int = 2, table_width: int = 32):
    page_table = torch.full((table_count, table_width), -1, dtype=torch.int32)
    cache = CacheManager(num_pages, 1, page_table, "radix")
    table = TableManager(table_count, page_table)
    kv_cache = _RecordingKVCache()
    manager = PrefillManager(
        cache_manager=cache,
        table_manager=table,
        decode_manager=SimpleNamespace(inflight_tokens=0),
        kv_cache=kv_cache,
        retry_rope_cache=torch.empty((32, 2), dtype=torch.float32),
    )
    return manager, cache, table, kv_cache


def _protect_prefix(cache: CacheManager, keys: torch.Tensor):
    pages = cache._allocate(len(keys))
    inserted = cache.prefix_cache.insert_prefix(keys, pages)
    cache.lock(inserted.handle)
    return inserted.handle


def _complete_intermediate_chunk(manager: PrefillManager, req: ChunkedReq) -> None:
    req.occurrence_inflight = True
    req.complete_one()
    manager.complete_chunk(req)


def _free_occurrence_request(req, cache: CacheManager, table: TableManager) -> None:
    if req.occurrence_transient_pages is not None:
        cache.free_occurrence_pages(req.occurrence_transient_pages)
    owned = req.occurrence_terminal_owned_mask
    assert owned is not None
    owned_indices = torch.nonzero(owned, as_tuple=False).view(-1)
    if len(owned_indices) > 0:
        cache.free_occurrence_pages(table.occurrence_pages(req.table_idx)[owned_indices].clone())
    cache.unlock(req.cache_handle)
    table.free(req.table_idx)


@pytest.mark.parametrize("budget", [2, 8])
def test_overflow_raw_storage_compacts_to_fixed_decode_table(budget: int) -> None:
    manager, cache, table, _ = _manager(32, table_count=1, table_width=8)
    pending = _pending(201)
    pending.context_post_prefill_keep_mask = pending.full_keep_mask
    manager.pending_list.append(pending)
    pointer = table.page_table.data_ptr()
    while True:
        batch = manager.schedule_next_batch(prefill_budget=budget)
        assert batch is not None
        req = batch.reqs[0]
        assert req.occurrence_external_storage
        build_occurrence_attention_batch([req])
        if not isinstance(req, ChunkedReq):
            break
        _complete_intermediate_chunk(manager, req)
    keep = pending.full_keep_mask.to(torch.bool)
    pages = table.occurrence_pages(req.table_idx)[keep].clone()
    table.occurrence_tokens(req.table_idx)[8] = 999
    req.complete_one()
    scheduler = object.__new__(Scheduler)
    scheduler.table_manager = table
    scheduler._compact_context_after_prefill(req)
    assert table.page_table.data_ptr() == pointer
    assert not table.has_occurrence_storage(req.table_idx)
    assert not req.occurrence_external_storage
    assert torch.equal(table.page_table[req.table_idx, :6], pages)
    assert torch.equal(table.token_pool[req.table_idx, :6], pending.input_ids[keep])
    assert table.token_pool[req.table_idx, 6] == 999
    assert req.true_positions.tolist() == [0, 1, 2, 3, 4, 5, 6]
    _free_occurrence_request(req, cache, table)
    if req.inactive_cached_pages is not None:
        cache.free_occurrence_pages(req.inactive_cached_pages)
    assert cache.available_size == cache.num_pages


def test_occurrence_construction_failure_rolls_back_allocated_pages(monkeypatch) -> None:
    manager, cache, table, _ = _manager(32, table_count=1, table_width=8)
    manager.pending_list.append(_pending(202))

    def fail_construction(*args, **kwargs):
        raise ValueError("synthetic request construction failure")

    monkeypatch.setattr(PrefillAdder, "_construct_req", fail_construction)
    with pytest.raises(OccurrenceInputError, match="synthetic request construction"):
        manager.schedule_next_batch(prefill_budget=8)
    assert cache.available_size == cache.num_pages
    assert table.available_size == 1
    assert not table.has_occurrence_storage(0)
    assert cache.prefix_cache.size_info.protected_size == 0


def test_paged_occurrence_chunks_until_the_full_prompt_is_covered() -> None:
    # The unchunked plan has 15 occurrence pages, while the actual peak
    # working set for this plan is 10 pages (retained terminal pages plus one
    # chunk's temporary occurrences).
    manager, cache, table, kv_cache = _manager(num_pages=10, table_count=1)
    manager.pending_list.append(_pending(uid=101))
    query_ranges: list[tuple[int, int]] = []
    saw_layer_transform = False

    while True:
        batch = manager.schedule_next_batch(prefill_budget=8)
        assert batch is not None
        assert len(batch.reqs) == 1
        req = batch.reqs[0]
        query_ranges.append((req.cached_len, req.device_len))
        assert req.occurrence_transform_source_pages is not None
        saw_layer_transform |= len(req.occurrence_transform_source_pages) > 0
        metadata = build_occurrence_attention_batch([req])
        assert metadata.num_queries == req.extend_len
        assert bool(torch.all(metadata.direct_pages[metadata.key_positions] >= 0).item())
        req.record_context_cache_usage(metadata.cached_tokens[0], metadata.cached_positions[0])
        if not isinstance(req, ChunkedReq):
            break
        transient_count = len(req.occurrence_transient_pages)
        available_before = cache.available_size
        _complete_intermediate_chunk(manager, req)
        assert cache.available_size >= available_before + transient_count
        owned_mask = req.occurrence_terminal_owned_mask
        owned_pages = table.page_table[req.table_idx, : len(owned_mask)][owned_mask]
        owned_pages = owned_pages[owned_pages >= 0]
        assert len(torch.unique(owned_pages)) == len(owned_pages)
        assert cache.available_size + len(owned_pages) == cache.num_pages

    assert len(query_ranges) > 1
    assert query_ranges[0][0] == 0
    assert query_ranges[-1][1] == 8
    assert all(
        left_end == right_start
        for (_, left_end), (right_start, _) in zip(query_ranges, query_ranges[1:])
    )
    assert req.usage_cached_tokens == 0
    assert kv_cache.calls == []
    assert saw_layer_transform
    _free_occurrence_request(req, cache, table)
    cache.check_integrity()


def test_paged_occurrence_unlocks_a_pinned_prefix_and_recomputes_when_needed() -> None:
    manager, cache, table, _ = _manager(num_pages=8, table_count=1)
    pending = _pending(uid=102)
    cached_pages = cache._allocate(6)
    cache.prefix_cache.insert_prefix(pending.radix_match_ids[:6], cached_pages)
    manager.pending_list.append(pending)

    batch = manager.schedule_next_batch(prefill_budget=8)

    assert batch is not None
    req = batch.reqs[0]
    assert isinstance(req, ChunkedReq)
    assert req.initial_active_cached_len == 0
    assert req.cached_len == 0
    assert cache.prefix_cache.size_info.protected_size == 0
    _free_occurrence_request(req, cache, table)
    cache.check_integrity()


def test_paged_occurrence_batches_two_requests_without_cross_request_pages() -> None:
    manager, cache, table, _ = _manager(num_pages=32, table_count=2)
    manager.pending_list.extend([_pending(uid=103), _pending(uid=104)])

    batch = manager.schedule_next_batch(prefill_budget=16)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [103, 104]
    metadata = build_occurrence_attention_batch(batch.reqs)
    assert metadata.num_queries == 16
    page_sets = [
        set(req.occurrence_pages[req.occurrence_pages >= 0].tolist()) for req in batch.reqs
    ]
    assert page_sets[0].isdisjoint(page_sets[1])
    for req in batch.reqs:
        _free_occurrence_request(req, cache, table)
    cache.check_integrity()


def test_paged_occurrence_rejects_only_an_impossible_minimum_working_set() -> None:
    manager, cache, _, _ = _manager(num_pages=7, table_count=1)
    manager.pending_list.append(_pending(uid=105))

    with pytest.raises(RepositionCapacityError) as raised:
        manager.schedule_next_batch(prefill_budget=8)

    assert raised.value.uid == 105
    assert raised.value.required_pages == 8
    assert raised.value.available_pages == 7
    cache.check_integrity()


def test_paged_occurrence_waits_for_pages_protected_by_another_request() -> None:
    manager, cache, table, _ = _manager(num_pages=10, table_count=1)
    pending = _pending(uid=106)
    blocker_keys = pending.radix_match_ids[:3].clone()
    blocker_keys[:, 1] += 1_000
    blocker = _protect_prefix(cache, blocker_keys)
    manager.pending_list.append(pending)

    assert manager.schedule_next_batch(prefill_budget=8) is None
    assert manager.pending_list == [pending]
    assert table.available_size == 1

    cache.unlock(blocker)
    batch = manager.schedule_next_batch(prefill_budget=8)
    assert batch is not None and len(batch.reqs) == 1
    _free_occurrence_request(batch.reqs[0], cache, table)
    cache.check_integrity()


def test_staged_reposition_waits_for_pages_protected_by_another_request() -> None:
    manager, cache, table, _ = _manager(num_pages=8, table_count=1)
    blocker = _protect_prefix(cache, torch.arange(200, 204, dtype=torch.int32))
    pending = PendingReq(
        uid=107,
        input_ids=torch.arange(10, 14, dtype=torch.int32),
        true_positions=torch.arange(4, dtype=torch.int32),
        raw_positions=torch.arange(4, dtype=torch.int32),
        radix_input_ids=torch.arange(10, 14, dtype=torch.int32),
        radix_match_ids=torch.arange(10, 14, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        radix_positions=torch.arange(4, dtype=torch.int32),
        radix_repos_info=torch.full((4,), -1, dtype=torch.int32),
        reposition_execution_mode="staged",
    )
    manager.pending_list.append(pending)

    assert manager.schedule_next_batch(prefill_budget=8) is None
    assert manager.pending_list == [pending]
    assert table.available_size == 1

    cache.unlock(blocker)
    batch = manager.schedule_next_batch(prefill_budget=8)
    assert batch is not None and len(batch.reqs) == 1
    req = batch.reqs[0]
    assert req.cached_len == 0
    cache.unlock(req.cache_handle)
    table.free(req.table_idx)
    cache.check_integrity()


def test_capacity_chunked_occurrence_request_runs_alone_until_completed() -> None:
    manager, cache, table, _ = _manager(num_pages=10, table_count=2)
    first = _pending(uid=108)
    second = _pending(uid=109)
    manager.pending_list.extend([first, second])

    batch = manager.schedule_next_batch(prefill_budget=16)

    assert batch is not None and len(batch.reqs) == 1
    assert isinstance(batch.reqs[0], ChunkedReq)
    assert [pending.uid for pending in manager.pending_list] == [108, 109]
    _free_occurrence_request(batch.reqs[0], cache, table)
    cache.check_integrity()


def test_occurrence_setup_failure_releases_prefix_lock_and_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, cache, table, _ = _manager(num_pages=12, table_count=1)
    pending = _pending(uid=110)
    cached_pages = cache._allocate(2)
    cache.prefix_cache.insert_prefix(pending.radix_match_ids[:2], cached_pages)
    manager.pending_list.append(pending)
    monkeypatch.setattr(
        table,
        "allocate",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic table failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic table failure"):
        manager.schedule_next_batch(prefill_budget=8)

    assert cache.prefix_cache.size_info.protected_size == 0
    assert table.available_size == 1
    cache.check_integrity()


def test_scheduler_rejection_releases_a_partial_occurrence_request() -> None:
    manager, cache, table, _ = _manager(num_pages=8, table_count=1)
    manager.pending_list.append(_pending(uid=111))

    for _ in range(8):
        try:
            batch = manager.schedule_next_batch(prefill_budget=8)
        except RepositionCapacityError:
            break
        assert batch is not None and len(batch.reqs) == 1
        req = batch.reqs[0]
        assert isinstance(req, ChunkedReq)
        _complete_intermediate_chunk(manager, req)
    else:
        pytest.fail("The synthetic eight-page plan never reached its impossible peak.")

    scheduler = object.__new__(Scheduler)
    scheduler.prefill_budget = 8
    scheduler.prefill_manager = manager
    scheduler.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None)
    scheduler.cache_manager = cache
    scheduler.table_manager = table
    scheduler.request_metrics = {111: object()}
    scheduler.context_sequence_uids = {111}
    replies: list[list[RequestRejectMsg]] = []
    scheduler.send_result = replies.append

    assert scheduler._schedule_next_batch() is None
    assert manager.pending_list == []
    assert table.available_size == 1
    assert scheduler.context_sequence_uids == set()
    assert replies[0][0].error_code == "reposition_working_set_exceeded"
    cache.check_integrity()


def test_abort_releases_a_completed_partial_occurrence_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, cache, table, _ = _manager(num_pages=10, table_count=1)
    pending = _pending(uid=112)
    manager.pending_list.append(pending)
    batch = manager.schedule_next_batch(prefill_budget=8)
    assert batch is not None and len(batch.reqs) == 1
    req = batch.reqs[0]
    assert isinstance(req, ChunkedReq)
    _complete_intermediate_chunk(manager, req)

    scheduler = object.__new__(Scheduler)
    scheduler.prefill_manager = manager
    scheduler.decode_manager = SimpleNamespace(abort_req=lambda _uid: None)
    scheduler.cache_manager = cache
    scheduler.table_manager = table
    scheduler.request_metrics = {112: object()}
    scheduler.context_sequence_uids = {112}
    scheduler.finished_reqs = set()
    monkeypatch.setattr(
        "minisgl.scheduler.scheduler.logger.debug_rank0", lambda *_args, **_kwargs: None
    )
    scheduler._process_one_msg(AbortBackendMsg(uid=112))

    assert manager.pending_list == []
    assert table.available_size == 1
    assert scheduler.context_sequence_uids == set()
    assert req not in scheduler.finished_reqs
    cache.check_integrity()


def test_abort_of_a_completed_occurrence_prefill_uses_normal_decode_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, cache, table, _ = _manager(num_pages=32, table_count=1)
    manager.pending_list.append(_pending(uid=113))
    batch = manager.schedule_next_batch(prefill_budget=8)
    assert batch is not None and len(batch.reqs) == 1
    req = batch.reqs[0]
    assert not isinstance(req, ChunkedReq)

    scheduler = object.__new__(Scheduler)
    scheduler.prefill_manager = manager
    scheduler.decode_manager = SimpleNamespace(abort_req=lambda _uid: req)
    scheduler.cache_manager = cache
    scheduler.table_manager = table
    scheduler.request_metrics = {113: object()}
    scheduler.context_sequence_uids = {113}
    scheduler.finished_reqs = set()
    cleanup_calls: list[str] = []
    scheduler._free_req_resources = lambda _req: cleanup_calls.append("decode")
    scheduler._free_aborted_occurrence_resources = lambda _req: cleanup_calls.append("partial")
    monkeypatch.setattr(
        "minisgl.scheduler.scheduler.logger.debug_rank0", lambda *_args, **_kwargs: None
    )

    scheduler._process_one_msg(AbortBackendMsg(uid=113))

    assert cleanup_calls == ["decode"]
    assert req in scheduler.finished_reqs
    _free_occurrence_request(req, cache, table)
    cache.check_integrity()
