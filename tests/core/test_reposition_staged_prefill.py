from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi")

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.kernel.radix_reposition import compile_radix_reposition_layout
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
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


def _pending() -> PendingReq:
    token_ids = torch.arange(6, dtype=torch.int32)
    drops = torch.tensor([3, 5], dtype=torch.int32)
    ranges = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
    repositions = torch.tensor([1, 2, 4], dtype=torch.int32)
    layout = compile_radix_reposition_layout(
        token_ids,
        drops,
        torch.tensor([0, 1, 2], dtype=torch.int32),
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
        drop_event_positions=drops,
        drop_range_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
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
    repositions = torch.tensor([2], dtype=torch.int32)
    layout = compile_radix_reposition_layout(
        token_ids,
        drops,
        torch.tensor([0, 1], dtype=torch.int32),
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
        drop_event_positions=drops,
        drop_range_offsets=torch.tensor([0, 1], dtype=torch.int32),
        drop_position_ranges=ranges,
        drop_effective_event_count=1,
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


def _materialize_batch(manager: PrefillManager, cache: CacheManager):
    batch = manager.schedule_next_batch(prefill_budget=64)
    assert batch is not None and len(batch.reqs) == 1
    req = batch.reqs[0]
    cache.allocate_paged(batch.reqs)
    req.complete_one()
    if isinstance(req, ChunkedReq):
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


def test_drop_timeline_stages_when_requested_reposition_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    manager, cache, table, kv_cache = _manager()
    pending = _pending_noop_reposition_after_tail_drop()
    manager.pending_list.append(pending)

    first = _materialize_batch(manager, cache)
    assert isinstance(first, ChunkedReq)
    assert pending.staged_active_raw.tolist() == [0, 1]
    assert pending.staged_actual_stage == 0

    final = _materialize_batch(manager, cache)
    assert not isinstance(final, ChunkedReq)
    assert final.raw_positions[: final.cached_len].tolist() == [0, 1, 3]
    assert final.true_positions[: final.cached_len].tolist() == [0, 1, 3]
    assert kv_cache.calls == []

    cache.cache_req(final, finished=True)
    assert bool(torch.all(final.staged_full_page_indices >= 0).item())
    table.free(final.table_idx)
    cache.check_integrity()
