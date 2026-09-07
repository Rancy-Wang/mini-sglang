"""Single-request scheduler lifecycles, plus opt-in real-model CUDA validation.

CPU tests use real Radix pages and scheduler transitions; sampled tokens are
synthetic. Set MINISGL_R3_MODEL to a local model path on an isolated CUDA device
for the separate real-model test. No resident server is used or modified.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.attention.base import build_context_attention_batch, build_context_attention_segments
from minisgl.core import SamplingParams
from minisgl.kernel.radix_reposition import compile_radix_reposition_layout
from minisgl.message import TokenizeMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillManager
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager
from minisgl.tokenizer.server import _build_user_msg
from minisgl.tokenizer.tokenize import TokenizedResult, TokenizeManager


def _tokens(uid=1, drop=True):
    full = torch.arange(100, 110, dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int32)
    events = torch.tensor([9], dtype=torch.int32) if drop else empty
    offsets = torch.tensor([0, 1], dtype=torch.int32) if drop else torch.zeros(1, dtype=torch.int32)
    ranges = torch.tensor([0, 5], dtype=torch.int32) if drop else empty
    layout = compile_radix_reposition_layout(full, events, offsets, ranges, empty, empty)
    keep = layout.keep_mask
    raw = torch.arange(10, dtype=torch.int32)[keep]
    visible = torch.full((10,), 100, dtype=torch.int32)
    visible[:5] = 9
    result = TokenizedResult(
        input_ids=full[keep],
        true_positions=raw,
        raw_positions=raw,
        radix_input_ids=layout.records[layout.token_to_key[raw.long()]],
        radix_match_ids=layout.records,
        prefix_keep_mask=keep[:-1].int(),
        prompt_tokens=10,
        full_input_ids=full,
        full_token_visible_until=visible,
        full_keep_mask=keep.int(),
        drop_event_positions=events,
        drop_range_offsets=offsets,
        drop_position_ranges=ranges,
        drop_effective_event_count=int(drop),
        radix_key_virtual_mask=layout.virtual_mask,
        radix_key_to_token=layout.key_to_token,
        radix_token_to_key=layout.token_to_key,
        radix_positions=layout.positions,
        radix_repos_info=layout.repos_info,
    )
    msg = TokenizeMsg(
        uid=uid,
        text="token fixture",
        sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
        use_context_mask=drop,
    )
    if not drop:
        # Exercise the actual minimal ordinary-key producer.
        result = TokenizeManager(SimpleNamespace())._ordinary_result(msg, full)
    return _build_user_msg(msg, result)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    ctx = core.Context(page_size=1)
    ctx.attn_backend = SimpleNamespace(supports_multi_context_mask_prefill=True)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    table = TableManager(8, torch.full((8, 128), -1, dtype=torch.int32))
    cache = CacheManager(512, 1, table.page_table, "radix")
    decode = DecodeManager(1)
    prefill = PrefillManager(cache, table, decode)
    scheduler = object.__new__(Scheduler)
    scheduler.table_manager, scheduler.cache_manager = table, cache
    scheduler.prefill_manager, scheduler.decode_manager = prefill, decode
    scheduler.finished_reqs, scheduler.context_sequence_uids = set(), set()
    scheduler.request_metrics, scheduler.eos_token_ids = {}, {0}
    replies = []
    scheduler.send_result = replies.extend
    return scheduler, replies


def _seed(cache, n):
    base = _tokens(drop=False)
    cache.prefix_cache.insert_prefix(base.radix_match_ids[:n], cache._allocate(n))


def _forward_cpu(scheduler, batch, token=42):
    cache, table = scheduler.cache_manager, scheduler.table_manager
    cache.allocate_paged(batch.reqs)
    for req in batch.reqs:
        if req.use_context_mask and req.usage_cached_tokens is None:
            metadata = build_context_attention_batch([req])
            req.record_context_cache_usage(metadata.cached_tokens[0], metadata.cached_positions[0])
        table.token_pool[req.table_idx, req.device_len] = token
        req.complete_one()
    scheduler.decode_manager.filter_reqs(batch.reqs)
    scheduler._process_last_data(
        (
            SimpleNamespace(batch=batch),
            (
                None,
                torch.full((len(batch.reqs),), token, dtype=torch.int32),
                SimpleNamespace(synchronize=lambda: None),
            ),
        )
    )


@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize(
    "hit,expected,masked", [(0, (0, 0), True), (8, (8, 0), True), (9, (4, 5), False)]
)
def test_single_formal_prefill_then_decode(runtime, concurrency, hit, expected, masked):
    scheduler, replies = runtime
    _seed(scheduler.cache_manager, hit)
    for uid in range(concurrency):
        scheduler.prefill_manager.add_one_req(_tokens(uid))
    batch = scheduler.prefill_manager.schedule_next_batch(128)
    assert batch is not None and len(batch.reqs) == concurrency
    assert not scheduler.prefill_manager.runnable
    assert all(req.use_context_mask == masked and not req.is_warmup for req in batch.reqs)
    _forward_cpu(scheduler, batch)
    assert len(replies) == concurrency
    assert all(reply.next_token == 42 and not reply.finished for reply in replies)
    for req in batch.reqs:
        assert (req.reported_cached_tokens, req.drop_skipped_tokens) == expected
        assert req.true_positions.tolist() == [5, 6, 7, 8, 9, 10]
        assert req.input_ids[-1] == 42
        assert not req.use_context_mask
    decode = scheduler.decode_manager.schedule_next_batch()
    assert decode is not None and not decode.is_prefill
    _forward_cpu(scheduler, decode, token=43)
    assert len(replies) == concurrency * 2
    assert all(reply.finished and reply.completion_tokens == 2 for reply in replies[concurrency:])
    scheduler.cache_manager.check_integrity()
    assert scheduler.table_manager.available_size == 8
    # Drop output commits the computed prefix under the same canonical base key.
    from minisgl.scheduler.utils import PendingReq

    ordinary = _tokens(drop=False)
    pending = PendingReq(
        uid=99,
        input_ids=ordinary.input_ids,
        true_positions=ordinary.true_positions,
        raw_positions=ordinary.raw_positions,
        radix_input_ids=ordinary.radix_input_ids,
        radix_match_ids=ordinary.radix_match_ids,
        sampling_params=ordinary.sampling_params,
    )
    match = scheduler.cache_manager.match_req(pending)
    assert match.active_cached_len == 9


@pytest.mark.parametrize("budget", [1, 2, 32])
def test_chunks_keep_initial_usage_and_sample_only_final_chunk(runtime, budget):
    scheduler, replies = runtime
    _seed(scheduler.cache_manager, 8)
    scheduler.prefill_manager.add_one_req(_tokens())
    forwards = 0
    while scheduler.prefill_manager.runnable:
        batch = scheduler.prefill_manager.schedule_next_batch(budget)
        assert batch is not None
        chunk = isinstance(batch.reqs[0], ChunkedReq)
        _forward_cpu(scheduler, batch)
        forwards += 1
        assert len(replies) == int(not chunk)
    assert forwards == (2 if budget == 1 else 1)
    req = batch.reqs[0]
    assert (req.reported_cached_tokens, req.drop_skipped_tokens) == (8, 0)
    _forward_cpu(scheduler, scheduler.decode_manager.schedule_next_batch())
    scheduler.cache_manager.check_integrity()


@pytest.mark.parametrize("abort_after_chunk", [False, True])
def test_formal_cancellation_releases_owned_pages(runtime, abort_after_chunk):
    scheduler, _ = runtime
    _seed(scheduler.cache_manager, 8)
    scheduler.prefill_manager.add_one_req(_tokens())
    batch = scheduler.prefill_manager.schedule_next_batch(1 if abort_after_chunk else 32)
    _forward_cpu(scheduler, batch)
    if abort_after_chunk:
        req = scheduler.prefill_manager.abort_req(1)
    else:
        req = scheduler.decode_manager.abort_req(1)
    assert req is not None
    scheduler._free_req_resources(req)
    scheduler.cache_manager.check_integrity()
    assert scheduler.table_manager.available_size == 8


def test_first_token_eos_compacts_and_frees(runtime):
    scheduler, replies = runtime
    msg = _tokens()
    msg.sampling_params.ignore_eos = False
    scheduler.prefill_manager.add_one_req(msg)
    _forward_cpu(scheduler, scheduler.prefill_manager.schedule_next_batch(32), token=0)
    assert len(replies) == 1 and replies[0].finished and replies[0].completion_tokens == 1
    assert not scheduler.decode_manager.runnable
    scheduler.cache_manager.check_integrity()


def test_overlap_finishes_compaction_before_scheduling_decode(runtime):
    scheduler, replies = runtime
    scheduler.prefill_manager.add_one_req(_tokens())
    batch = scheduler.prefill_manager.schedule_next_batch(32)
    scheduler.cache_manager.allocate_paged(batch.reqs)
    req = batch.reqs[0]
    metadata = build_context_attention_batch([req])
    req.record_context_cache_usage(metadata.cached_tokens[0], metadata.cached_positions[0])
    req.complete_one()
    scheduler.receive_msg = lambda **kwargs: []

    def schedule():
        assert not req.use_context_mask
        assert req.input_ids[-1] == 42
        assert len(req.input_ids) == 6
        return None

    scheduler._schedule_next_batch = schedule
    data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32), SimpleNamespace(synchronize=lambda: None)),
    )
    assert scheduler.overlap_loop(data) is None
    assert len(replies) == 1
    scheduler._free_req_resources(req)
    scheduler.cache_manager.check_integrity()


@pytest.mark.parametrize("window", [None, 2, 4])
@pytest.mark.parametrize("hit", [0, 8, 9])
def test_actual_query_visibility_matches_independent_dense_oracle(runtime, window, hit):
    scheduler, _ = runtime
    scheduler.prefill_manager.has_sliding_window = window is not None
    _seed(scheduler.cache_manager, hit)
    message = _tokens()
    scheduler.prefill_manager.add_one_req(message)
    req = scheduler.prefill_manager.schedule_next_batch(32).reqs[0]
    raw = req.raw_positions.long()
    actual = torch.zeros((req.extend_len, 10), dtype=torch.bool)
    if req.use_context_mask:
        segments = build_context_attention_segments(
            message.full_token_visible_until,
            query_start=req.cached_len,
            query_length=req.extend_len,
            key_length=req.device_len,
            raw_positions=raw,
            true_positions=req.true_positions,
            sliding_window=None if window is None else window - 1,
        )
        for segment in segments:
            queries = segment.query_end - segment.query_start
            prefix = len(segment.key_positions) - queries
            for q in range(queries):
                actual[segment.query_start + q, raw[segment.key_positions[: prefix + q + 1]]] = True
    else:
        for i, slot in enumerate(range(req.cached_len, req.device_len)):
            keys = raw[: slot + 1]
            if window is not None:
                keys = keys[keys >= raw[slot] - window + 1]
            actual[i, keys] = True
    keys = torch.arange(10)
    queries = raw[req.cached_len :, None]
    expected = (keys <= queries) & ((keys >= 5) | (queries < 9))
    if window is not None:
        expected &= keys >= queries - window + 1
    assert torch.equal(actual, expected)


def test_ordinary_cache_match_does_not_invoke_retry(runtime, monkeypatch):
    scheduler, replies = runtime
    _seed(scheduler.cache_manager, 8)

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary canonical keys must not use Retry planning.")

    monkeypatch.setattr(scheduler.cache_manager.prefix_cache, "match_retry_prefix", forbidden)
    scheduler.prefill_manager.add_one_req(_tokens(drop=False))
    _forward_cpu(scheduler, scheduler.prefill_manager.schedule_next_batch(32))
    _forward_cpu(scheduler, scheduler.decode_manager.schedule_next_batch())
    assert replies[-1].cached_tokens == 8
    scheduler.cache_manager.check_integrity()


@pytest.mark.skipif(
    not os.environ.get("MINISGL_R3_MODEL"), reason="set MINISGL_R3_MODEL on an isolated CUDA device"
)
def test_real_model_single_request_prefill_and_decode():
    subprocess.run([sys.executable, __file__, "--real-model"], check=True, timeout=900)


@torch.inference_mode()
def _real_model():
    from minisgl.distributed import DistributedInfo
    from minisgl.scheduler.config import SchedulerConfig
    from minisgl.utils import load_tokenizer

    model = os.environ["MINISGL_R3_MODEL"]
    config = SchedulerConfig(
        model_path=model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        offline_mode=True,
        attention_backend="fa3",
        cuda_graph_bs=[],
        use_pynccl=False,
        max_running_req=8,
        max_seq_len_override=512,
        num_page_override=2048,
        max_extend_tokens=512,
    )
    scheduler = Scheduler(config)
    tokenizer = TokenizeManager(load_tokenizer(model))
    messages = [
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "I remember 42."},
        {"role": "user", "content": "What number was mentioned?"},
    ]
    try:
        for concurrency in (1, 4):
            for drop in (False, True):
                replies = []
                scheduler.send_result = replies.extend
                for uid in range(concurrency):
                    msg = TokenizeMsg(
                        uid=uid,
                        text=messages,
                        sampling_params=SamplingParams(
                            max_tokens=2, ignore_eos=True, temperature=0
                        ),
                        drop_message={1: [0]} if drop else None,
                        use_context_mask=drop,
                    )
                    tokenized = tokenizer.tokenize([msg])[0]
                    assert (
                        tokenized.tokenize_invocations == tokenized.chat_template_invocations == 1
                    )
                    scheduler._process_one_msg(_build_user_msg(msg, tokenized))
                phases = []
                with scheduler.engine_stream_ctx:
                    while scheduler.prefill_manager.runnable or scheduler.decode_manager.runnable:
                        forward = scheduler._schedule_next_batch()
                        assert forward is not None
                        phases.append((forward.batch.is_prefill, len(forward.batch.reqs)))
                        assert all(not req.is_warmup for req in forward.batch.reqs)
                        scheduler._process_last_data((forward, scheduler._forward(forward)))
                assert phases == [(True, concurrency), (False, concurrency)], phases
                assert len(replies) == concurrency * 2
                assert all(reply.completion_tokens == 2 for reply in replies if reply.finished)
                scheduler.cache_manager.check_integrity()
                print({"concurrency": concurrency, "drop": drop, "phases": phases}, flush=True)
    finally:
        scheduler.shutdown()


if __name__ == "__main__" and "--real-model" in sys.argv:
    _real_model()
