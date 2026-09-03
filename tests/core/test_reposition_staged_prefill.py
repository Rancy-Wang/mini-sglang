from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tvm_ffi")

import minisgl.core as core
import minisgl.tokenizer.reposition_sequence as sequence_module
from minisgl.core import Req, SamplingParams
from minisgl.message import BaseBackendMsg, TokenizeMsg, WarmupAckMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq
from minisgl.tokenizer.reposition_sequence import RepositionSequenceState
from minisgl.tokenizer.tokenize import TokenizedResult


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


def _sequence(*, max_tokens: int = 2) -> RepositionSequenceState:
    token_ids = torch.arange(9, dtype=torch.int32)
    drops = torch.tensor([2, 7], dtype=torch.int32)
    drop_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    ranges = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
    reposition_boundaries = torch.tensor([4], dtype=torch.int32)
    reposition_offsets = torch.tensor([5], dtype=torch.int32)
    request = TokenizeMsg(
        uid=7,
        text="already tokenized",
        sampling_params=SamplingParams(max_tokens=max_tokens),
        reposition=[4],
        request_received_ns=123,
    )
    tokenized = TokenizedResult(
        input_ids=token_ids,
        true_positions=torch.arange(9, dtype=torch.int32),
        raw_positions=torch.arange(9, dtype=torch.int32),
        radix_input_ids=token_ids.to(torch.int64),
        radix_match_ids=token_ids.to(torch.int64),
        prefix_keep_mask=torch.ones(9, dtype=torch.int32),
        prompt_tokens=9,
        full_input_ids=token_ids,
        full_token_visible_until=_visible_until(9, drops, drop_offsets, ranges),
        full_keep_mask=torch.tensor([0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.int32),
        drop_event_positions=drops,
        drop_range_offsets=drop_offsets,
        drop_position_ranges=ranges,
        drop_effective_event_count=2,
        reposition_raw_boundaries=reposition_boundaries,
        reposition_insert_offsets=reposition_offsets,
        reposition_input_ids=token_ids,
        tokenize_invocations=1,
    )
    return RepositionSequenceState.pending(request, tokenized)


def _ack(uid: int, **metrics: int) -> WarmupAckMsg:
    return WarmupAckMsg(
        uid=uid,
        hit_ratio=1.0,
        cached_tokens=1,
        drop_skipped_tokens=0,
        finished=True,
        **metrics,
    )


def test_empty_reposition_builds_one_structured_retry_source_without_staging() -> None:
    token_ids = torch.arange(5, dtype=torch.int32)
    request = TokenizeMsg(
        uid=6,
        text="already tokenized",
        sampling_params=SamplingParams(max_tokens=1),
        reposition=[],
    )
    tokenized = TokenizedResult(
        input_ids=token_ids,
        true_positions=torch.arange(5, dtype=torch.int32),
        raw_positions=torch.arange(5, dtype=torch.int32),
        radix_input_ids=token_ids.to(torch.int64),
        radix_match_ids=token_ids.to(torch.int64),
        prefix_keep_mask=torch.ones(5, dtype=torch.int32),
        prompt_tokens=5,
        full_input_ids=token_ids,
        full_keep_mask=torch.ones(5, dtype=torch.int32),
        reposition_raw_boundaries=torch.empty(0, dtype=torch.int32),
        reposition_insert_offsets=torch.empty(0, dtype=torch.int32),
        reposition_input_ids=token_ids,
        tokenize_invocations=1,
    )
    state = RepositionSequenceState.pending(request, tokenized)
    state.compile([], step_token_budget=2)

    message = state.build_next_msg()

    assert message.raw_positions.tolist() == [0, 1, 2, 3, 4]
    assert message.radix_match_ids.ndim == 2
    assert not message.is_warmup
    assert not message.use_context_mask
    assert message.tokenize_invocations == 1


def test_tokenizer_sequence_compiles_once_and_waits_between_scheduler_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _sequence()
    calls = 0
    real_compile = sequence_module.compile_radix_reposition_layout

    def count_compile(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(sequence_module, "compile_radix_reposition_layout", count_compile)
    open_msg = state.open_msg()
    assert open_msg.full_token_count == 9
    state.compile([-101, -102], step_token_budget=64)
    assert calls == 1
    with pytest.raises(RuntimeError, match="compiled more than once"):
        state.compile([-101, -102], step_token_budget=64)

    # Scheduler acknowledgements carry cumulative snapshots seeded by the
    # previous Tokenizer turn; accepting one must not count that seed twice.
    state.radix_match_ns = 5
    first = state.build_next_msg()
    assert first.raw_positions.tolist() == [0, 1, 2, 3, 4]
    assert first.use_context_mask
    assert first.is_warmup
    assert first.radix_match_ns == 5
    with pytest.raises(RuntimeError, match="awaiting Scheduler"):
        state.build_next_msg()
    state.accept_ack(_ack(7, radix_match_ns=11))

    final = state.build_next_msg()
    assert final.raw_positions.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert final.use_context_mask
    assert not final.is_warmup
    assert final.context_post_prefill_keep_mask.tolist() == [0, 1, 1, 0, 1, 1, 1, 1, 1]
    assert final.tokenize_invocations == 1
    assert final.radix_compile_ns > 0
    assert final.radix_match_ns == 11
    assert final.retry_plan_ns == 0
    assert final.reposition_transition_count == 0
    assert calls == 1

    assert not any(field.name.startswith("staged_") for field in fields(Req))
    assert not any(field.name.startswith("staged_") for field in fields(PendingReq))


def test_each_scheduler_turn_reuses_the_previous_partial_radix_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    state = _sequence(max_tokens=1)
    state.compile([-201, -202], step_token_budget=64)
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

    cached_counts = []
    active_cached_counts = []
    for turn in range(2):
        # Exercise the real Tokenizer -> Scheduler ownership boundary.  Without
        # the wire copy, later state transitions could mutate this test's key.
        message = BaseBackendMsg.decoder(state.build_next_msg().encoder())
        manager.add_one_req(message)
        batch = manager.schedule_next_batch(prefill_budget=64)
        assert batch is not None and len(batch.reqs) == 1
        req = batch.reqs[0]
        cache.allocate_paged(batch.reqs)
        req.complete_one()
        cached_counts.append(req.radix_cached_tokens)
        active_cached_counts.append(req.initial_active_cached_len)
        ack = _ack(
            req.uid,
            radix_match_ns=req.radix_match_ns,
            retry_plan_ns=req.retry_plan_ns,
            reposition_transition_count=req.reposition_transition_count,
        )
        cache.cache_req(req, finished=True)
        table.free(req.table_idx)
        if turn == 0:
            state.accept_ack(ack)

    assert cached_counts[0] == 0
    assert cached_counts[1] > 0
    assert active_cached_counts[1] > 0
    assert kv_cache.calls
    cache.check_integrity()


def test_final_mask_prefill_compacts_decode_view_and_retains_owned_drop_pages() -> None:
    page_table = torch.full((1, 16), -1, dtype=torch.int32)
    token_pool = torch.zeros_like(page_table)
    scheduler = object.__new__(Scheduler)
    scheduler.table_manager = SimpleNamespace(page_table=page_table, token_pool=token_pool)
    prompt = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    keep_mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int32)
    req = Req(
        input_ids=prompt,
        true_positions=torch.arange(5, dtype=torch.int32),
        raw_positions=torch.arange(5, dtype=torch.int32),
        radix_input_ids=prompt.to(torch.int64),
        radix_match_ids=prompt.to(torch.int64),
        initial_full_match_indices=torch.tensor([10, 11, 12], dtype=torch.int32),
        initial_active_cached_len=3,
        true_seq_len=5,
        table_idx=0,
        cached_len=3,
        output_len=2,
        uid=9,
        sampling_params=SamplingParams(max_tokens=2),
        cache_handle=SimpleNamespace(),
        full_input_ids=prompt,
        full_token_visible_until=torch.full((5,), 6, dtype=torch.int32),
        full_keep_mask=keep_mask,
        use_context_mask=True,
        context_compact_stream=True,
        context_post_prefill_keep_mask=keep_mask,
        retry_transformed_mask=torch.tensor([False, True, False]),
    )
    page_table[0, :5] = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    token_pool[0, :6] = torch.tensor([10, 11, 12, 13, 14, 99], dtype=torch.int32)
    req.cached_len = 5
    req.device_len = 6
    req.max_device_len = 7
    req.true_positions = torch.cat((req.true_positions, torch.tensor([5], dtype=torch.int32)))
    req.raw_positions = torch.cat((req.raw_positions, torch.tensor([5], dtype=torch.int32)))

    scheduler._compact_context_after_prefill(req)

    assert req.input_ids.tolist() == [10, 12, 14]
    assert req.true_positions.tolist() == [0, 2, 4, 5]
    assert req.raw_positions.tolist() == [0, 2, 4, 5]
    assert page_table[0, :3].tolist() == [10, 12, 14]
    assert token_pool[0, :4].tolist() == [10, 12, 14, 99]
    assert req.inactive_cached_positions.tolist() == [1, 3]
    assert req.inactive_cached_pages.tolist() == [11, 13]
    assert req.initial_active_cached_len == 2
    assert req.retry_transformed_mask.tolist() == [False, False]
    assert (req.cached_len, req.device_len, req.max_device_len) == (3, 4, 5)
    assert not req.use_context_mask
    assert req.context_post_prefill_keep_mask is None
