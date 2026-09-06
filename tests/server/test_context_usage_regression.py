"""Real CPU Radix/Prefill producers joined to the HTTP usage collectors.

Tokenization and model forward are excluded; the ten-token cache history is
constructed explicitly so that the expected counts do not depend on text.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
import minisgl.server.api_server as api
from minisgl.attention.base import build_context_attention_batch
from minisgl.core import SamplingParams
from minisgl.kernel.radix_reposition import compile_radix_reposition_layout
from minisgl.message import UserReply
from minisgl.message.metrics import RequestMetricsState
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import PrefillAdder
from minisgl.scheduler.utils import PendingReq


@pytest.fixture
def producer(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
    full = torch.arange(100, 110, dtype=torch.int32)
    empty = torch.empty(0, dtype=torch.int32)
    events = torch.tensor([9], dtype=torch.int32)
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    ranges = torch.tensor([0, 5], dtype=torch.int32)
    layout = compile_radix_reposition_layout(full, events, offsets, ranges, empty, empty)
    keep = layout.keep_mask
    raw = torch.arange(10, dtype=torch.int32)[keep]
    visible = torch.full((10,), 11, dtype=torch.int32)
    visible[:5] = 9

    def make(cached_prefix, warmup):
        page_table = torch.full((2, 64), -1, dtype=torch.int32)
        cache = CacheManager(64, 1, page_table, "radix")
        n = int(layout.token_to_key[cached_prefix])
        virtual = layout.virtual_mask[:n]
        pages = torch.full((n,), -1, dtype=torch.int32)
        pages[~virtual] = cache._allocate(cached_prefix)
        cache.prefix_cache.insert_prefix(layout.records[:n], pages, virtual)
        pending = PendingReq(
            uid=1,
            input_ids=full[keep],
            true_positions=raw,
            raw_positions=raw,
            radix_input_ids=layout.records[layout.token_to_key[raw.long()]],
            radix_match_ids=layout.records,
            sampling_params=SamplingParams(max_tokens=1),
            prompt_tokens=10,
            is_warmup=warmup,
            prefix_keep_mask=keep[:-1].int(),
            full_input_ids=full,
            full_token_visible_until=visible,
            full_keep_mask=keep.int(),
            drop_event_positions=events,
            drop_range_offsets=offsets,
            drop_position_ranges=ranges,
            drop_effective_event_count=1,
            use_context_mask=warmup,
            radix_key_virtual_mask=layout.virtual_mask,
            radix_key_to_token=layout.key_to_token,
            radix_token_to_key=layout.token_to_key,
            radix_positions=layout.positions,
            radix_repos_info=layout.repos_info,
        )
        table = SimpleNamespace(
            available_size=2,
            token_pool=torch.zeros((2, 64), dtype=torch.int32),
            page_table=page_table,
            allocate=lambda: 0,
            free=lambda _: None,
        )
        adder = PrefillAdder(32, 0, cache, table)
        req = adder.try_add_one(pending, adder.plan_context_prefill(pending))
        assert req is not None
        if req.use_context_mask:
            context = build_context_attention_batch([req])
            req.record_context_cache_usage(context.cached_tokens[0], context.cached_positions[0])
        return api.CacheUsageReport(
            req.reported_cached_tokens, req.drop_skipped_tokens, req.reported_repos_tokens
        )

    return make


def _response(monkeypatch, warm, final, *, stream, include_usage=True, reposition=False):
    manager = api.FrontendManager(
        config=SimpleNamespace(
            model_path="Qwen3-1.7B",
            tool_call_parser="auto",
            reasoning_parser="auto",
            radix_drop_key_mode="delta-marker",
        ),
        send_tokenizer=None,
        recv_tokenizer=None,
    )
    metrics = RequestMetricsState(
        request_received_ns=0,
        prompt_tokens=10,
        active_prompt_tokens=5,
        # Deliberately incompatible diagnostic count: usage must use its own reply.
        drop_skipped_tokens=9,
    )
    metrics.observe_token(1, visible=True)

    async def acknowledgements(uid):
        yield UserReply(uid=uid, incremental_output="a", finished=False)
        yield UserReply(
            uid=uid,
            incremental_output="b",
            finished=True,
            finish_reason="stop",
            cached_tokens=final.cached_tokens,
            drop_skipped_tokens=final.drop_skipped_tokens,
            repos_tokens=final.repos_tokens,
            prompt_tokens=10,
            completion_tokens=2,
            server_metrics=metrics.finish(2),
        )

    async def warmup(*args):
        return warm

    async def send_one(msg):
        pass

    async def connected():
        return False

    manager.wait_for_ack = acknowledgements
    manager.run_contextual_warmup = warmup
    manager.send_one = send_one
    monkeypatch.setattr(api, "get_global_state", lambda: manager)
    request = api.OpenAICompletionRequest(
        model="test",
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
        drop_message={1: [0]},
        reposition=[1] if reposition else None,
        stream=stream,
        stream_options={"include_usage": include_usage},
    )

    async def run():
        response = await api.v1_completions(request, SimpleNamespace(is_disconnected=connected))
        if not stream:
            return response["usage"]
        chunks = [
            chunk.decode().strip().removeprefix("data: ") async for chunk in response.body_iterator
        ]
        assert chunks[-1] == "[DONE]"
        usages = [json.loads(chunk)["usage"] for chunk in chunks[:-1] if '"usage"' in chunk]
        assert len(usages) == int(include_usage)
        return usages[0] if usages else None

    return asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "warm_prefix,expected", [(8, (8, 0)), (3, (3, 0)), (0, (0, 0)), (9, (4, 5))]
)
def test_http_keeps_one_real_warmup_snapshot(monkeypatch, producer, stream, warm_prefix, expected):
    warm = producer(warm_prefix, True)
    final = producer(9, False)
    assert (warm.cached_tokens, warm.drop_skipped_tokens) == expected
    assert (final.cached_tokens, final.drop_skipped_tokens) == (4, 5)
    usage = _response(monkeypatch, warm, final, stream=stream)
    assert usage["prompt_tokens"] == 10
    assert usage["total_tokens"] == 12
    if expected == (0, 0):
        assert "prompt_tokens_details" not in usage
    else:
        assert usage["prompt_tokens_details"] == dict(
            zip(("cached_tokens", "drop_skipped_tokens"), expected)
        )


def test_real_warmup_stream_without_usage_still_finishes(monkeypatch, producer):
    assert (
        _response(
            monkeypatch, producer(8, True), producer(9, False), stream=True, include_usage=False
        )
        is None
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("final", [api.CacheUsageReport(4, 2, 2), api.CacheUsageReport(0, 0, 0)])
def test_reposition_http_uses_final_triple_including_zero(monkeypatch, stream, final):
    usage = _response(monkeypatch, None, final, stream=stream, reposition=True)
    assert usage["prompt_tokens_details"] == {
        "cached_tokens": final.cached_tokens,
        "drop_skipped_tokens": final.drop_skipped_tokens,
        "repos_tokens": final.repos_tokens,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_no_warmup_http_uses_backend_snapshot(monkeypatch, stream):
    usage = _response(monkeypatch, None, api.CacheUsageReport(4, 5), stream=stream)
    assert usage["prompt_tokens_details"] == {"cached_tokens": 4, "drop_skipped_tokens": 5}
