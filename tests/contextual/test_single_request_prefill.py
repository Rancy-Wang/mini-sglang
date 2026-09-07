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
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
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


def test_different_ordinary_requests_batch_without_context_work(runtime, monkeypatch):
    scheduler, replies = runtime
    _seed(scheduler.cache_manager, 8)

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary requests must bypass Context planning and Retry.")

    monkeypatch.setattr(
        scheduler.prefill_manager.cache_manager.prefix_cache, "match_retry_prefix", forbidden
    )
    for uid in range(4):
        msg = _tokens(uid, drop=False)
        msg.input_ids[-1] += uid
        msg.radix_input_ids[-1, 1] = msg.input_ids[-1]
        scheduler.prefill_manager.add_one_req(msg)
    batch = scheduler.prefill_manager.schedule_next_batch(32)
    assert len(batch.reqs) == 4
    assert all(
        not req.use_context_mask and req.context_post_prefill_keep_mask is None
        for req in batch.reqs
    )
    _forward_cpu(scheduler, batch)
    _forward_cpu(scheduler, scheduler.decode_manager.schedule_next_batch())
    assert len(replies) == 8
    assert all(reply.cached_tokens == 8 for reply in replies if reply.finished)
    scheduler.cache_manager.check_integrity()


def test_first_token_stop_sequence_finishes_without_decode(runtime):
    scheduler, replies = runtime
    msg = _tokens()
    msg.stop, msg.stop_token_seqs = ["stop"], [[42]]
    scheduler.prefill_manager.add_one_req(msg)
    _forward_cpu(scheduler, scheduler.prefill_manager.schedule_next_batch(32))
    assert len(replies) == 1 and replies[0].matched_stop == "stop"
    assert replies[0].finished and not scheduler.decode_manager.runnable
    scheduler.cache_manager.check_integrity()


def _dense_model_attention(backend, q, k, v, layer_id, batch, *, sinks=None, sliding_window=None):
    """Per-query oracle uses raw lifetimes and true positions, never segment metadata."""
    backend.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    heads, dim = q.shape[1:]
    keys = backend.kvcache.k_cache(layer_id)
    values = backend.kvcache.v_cache(layer_id)
    kv_heads = keys.shape[-2]
    keys, values = keys.reshape(-1, kv_heads, dim), values.reshape(-1, kv_heads, dim)
    output = torch.empty_like(q)
    offset = 0
    for req in batch.padded_reqs:
        raw = req.raw_positions[: req.device_len]
        true = req.true_positions[: req.device_len]
        for slot in range(req.cached_len, req.device_len):
            allowed = torch.arange(req.device_len) <= slot
            if req.use_context_mask:
                allowed &= req.full_token_visible_until[raw.long()] > raw[slot]
            if sliding_window is not None:
                allowed &= true >= true[slot] - sliding_window
            pages = core.get_global_ctx().page_table[req.table_idx, : req.device_len]
            selected = pages[allowed.to(pages.device)].long()
            key = keys[selected].repeat_interleave(heads // kv_heads, dim=1).float()
            value = values[selected].repeat_interleave(heads // kv_heads, dim=1).float()
            logits = torch.einsum("hd,khd->hk", q[offset].float(), key) * backend.scale
            if sinks is not None:
                logits = torch.cat((logits, sinks.float().view(heads, 1)), dim=1)
            probabilities = torch.softmax(logits, dim=-1)[:, : len(selected)]
            output[offset] = torch.einsum("hk,khd->hd", probabilities, value).to(q.dtype)
            offset += 1
    assert offset == len(q)
    return output


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
        attention_backend="fa",
        cuda_graph_bs=[],
        use_pynccl=False,
        max_running_req=8,
        max_seq_len_override=512,
        num_page_override=2048,
        max_extend_tokens=512,
    )
    scheduler = Scheduler(config)
    tokenizer = TokenizeManager(load_tokenizer(model))
    backend = scheduler.engine.attn_backend
    fast_attention = backend.forward
    model_forward = scheduler.engine.model.forward
    checked_logits = []
    numerical_failures = []

    def checked_model_forward():
        actual = model_forward()
        backend.forward = lambda *args, **kwargs: _dense_model_attention(backend, *args, **kwargs)
        try:
            reference = model_forward()
        finally:
            backend.forward = fast_attention
        close = torch.isclose(actual.float(), reference.float(), atol=0.15, rtol=0.02)
        greedy_equal = torch.equal(actual.argmax(-1), reference.argmax(-1))
        comparison = {
            "max_abs_logit_error": (actual.float() - reference.float()).abs().max().item(),
            "mismatched_logits": (~close).sum().item(),
            "total_logits": actual.numel(),
            "actual_greedy": actual.argmax(-1).tolist(),
            "reference_greedy": reference.argmax(-1).tolist(),
            "greedy_equal": greedy_equal,
        }
        # The oracle wrote reference KV. Restore production KV before the next
        # decode; these extra model calls are validation only, not scheduler stages.
        restored = model_forward()
        torch.testing.assert_close(restored, actual, atol=0, rtol=0)
        checked_logits.append(comparison)
        return actual

    scheduler.engine.model.forward = checked_model_forward
    messages = [
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "I remember 42."},
        {"role": "user", "content": "What number was mentioned?"},
    ]
    try:
        for concurrency in (1, 4):
            for case, drop in (("ordinary", False), ("cold_mask", True), ("warm_extend", True)):
                if case != "warm_extend":
                    evicted = scheduler.cache_manager.prefix_cache.evict(
                        scheduler.cache_manager.prefix_cache.size_info.total_size
                    )
                    scheduler.cache_manager._free(evicted)
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
                masks = []
                checked_logits.clear()
                with scheduler.engine_stream_ctx:
                    while scheduler.prefill_manager.runnable or scheduler.decode_manager.runnable:
                        forward = scheduler._schedule_next_batch()
                        assert forward is not None
                        phases.append((forward.batch.is_prefill, len(forward.batch.reqs)))
                        if forward.batch.is_prefill:
                            masks.extend(req.use_context_mask for req in forward.batch.reqs)
                        assert all(not req.is_warmup for req in forward.batch.reqs)
                        scheduler._process_last_data((forward, scheduler._forward(forward)))
                assert phases == [(True, concurrency), (False, concurrency)], phases
                assert len(checked_logits) == 2
                assert masks == [case == "cold_mask"] * concurrency
                assert len(replies) == concurrency * 2
                assert all(reply.completion_tokens == 2 for reply in replies if reply.finished)
                scheduler.cache_manager.check_integrity()
                usage = [
                    {
                        "uid": reply.uid,
                        "prompt_tokens": reply.prompt_tokens,
                        "cached_tokens": reply.cached_tokens,
                        "drop_skipped_tokens": reply.drop_skipped_tokens,
                        "repos_tokens": reply.repos_tokens,
                        "completion_tokens": reply.completion_tokens,
                    }
                    for reply in replies
                    if reply.finished
                ]
                assert len(usage) == concurrency
                for item in usage:
                    cached = item["cached_tokens"]
                    skipped = item["drop_skipped_tokens"]
                    repos = item["repos_tokens"]
                    assert 0 <= cached + skipped + repos <= item["prompt_tokens"]
                    assert min(cached, skipped, repos) >= 0
                    assert repos == 0
                    if case == "warm_extend":
                        assert cached > 0 and skipped > 0
                    else:
                        assert cached == skipped == 0
                failed = any(
                    item["mismatched_logits"] or not item["greedy_equal"]
                    for item in checked_logits
                )
                if failed:
                    numerical_failures.append((concurrency, case, list(checked_logits)))
                print(
                    {
                        "concurrency": concurrency,
                        "case": case,
                        "phases": phases,
                        "masks": masks,
                        "usage": usage,
                        "output_tokens": [(reply.uid, reply.next_token) for reply in replies],
                        "numerical_checks": checked_logits,
                        "dense_logits_and_greedy": "fail" if failed else "pass",
                    },
                    flush=True,
                )
        # Finish all usage/lifecycle scenarios before reporting strict numerical
        # failures. The original atol/rtol and greedy equality remain required.
        assert not numerical_failures, numerical_failures
    finally:
        scheduler.shutdown()


if __name__ == "__main__" and "--real-model" in sys.argv:
    _real_model()
