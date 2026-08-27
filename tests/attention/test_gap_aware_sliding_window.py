from __future__ import annotations

from types import SimpleNamespace

import minisgl.attention.fa as fa_module
import pytest
import torch
from minisgl.attention.base import (
    batch_needs_gap_aware_sliding_window,
    build_sliding_window_attention_batch,
    compile_context_page_tables,
    sliding_window_crosses_gap,
    validate_active_true_positions,
)
from minisgl.attention.fa import FAMetadata, FlashAttentionBackend
from minisgl.attention.fi import FIMetadata, FlashInferBackend
from minisgl.engine.graph import GraphRunner


def _req(
    positions: torch.Tensor | list[int],
    *,
    cached_len: int,
    table_idx: int = 0,
) -> SimpleNamespace:
    true_positions = torch.as_tensor(positions, dtype=torch.int32, device="cpu")
    return SimpleNamespace(
        true_positions=true_positions,
        device_len=len(true_positions),
        cached_len=cached_len,
        table_idx=table_idx,
    )


def _segment_keys(metadata, segment_idx: int) -> torch.Tensor:
    start = int(metadata.cu_seqlens_k[segment_idx])
    end = int(metadata.cu_seqlens_k[segment_idx + 1])
    return metadata.key_positions[start:end].to(dtype=torch.int64)


def test_reported_decode_window_selects_only_103_absolute_keys() -> None:
    positions = torch.cat(
        [
            torch.arange(0, 3578, dtype=torch.int32),
            torch.arange(4475, 4578, dtype=torch.int32),
        ]
    )
    req = _req(positions, cached_len=len(positions) - 1)

    assert sliding_window_crosses_gap(
        positions,
        device_len=len(positions),
        sliding_window=128,
    )
    metadata = build_sliding_window_attention_batch([req], sliding_window=128)
    compact_keys = _segment_keys(metadata, 0)
    absolute_keys = positions[compact_keys]

    assert torch.equal(absolute_keys, torch.arange(4475, 4578, dtype=torch.int32))
    assert len(absolute_keys) == 103
    assert not bool(torch.any((absolute_keys >= 3553) & (absolute_keys < 3578)))


def test_prefill_windows_match_absolute_position_oracle() -> None:
    positions = torch.tensor([0, 1, 5, 6, 7, 12], dtype=torch.int32)
    req = _req(positions, cached_len=2)
    metadata = build_sliding_window_attention_batch([req], sliding_window=4)

    assert metadata.num_queries == req.device_len - req.cached_len
    for segment_idx, query_idx in enumerate(range(req.cached_len, req.device_len)):
        compact_keys = _segment_keys(metadata, segment_idx)
        absolute_keys = positions[compact_keys]
        query_position = positions[query_idx]
        oracle = positions[: query_idx + 1]
        oracle = oracle[(oracle >= query_position - 3) & (oracle <= query_position)]
        assert torch.equal(absolute_keys, oracle)


def test_mixed_batch_compiles_compatible_segments_for_every_request() -> None:
    gapped = _req([0, 1, 10, 11], cached_len=3, table_idx=2)
    continuous = _req([20, 21, 22, 23], cached_len=3, table_idx=5)

    assert batch_needs_gap_aware_sliding_window(
        [gapped, continuous],
        sliding_window=4,
        decode_only=True,
    )
    metadata = build_sliding_window_attention_batch(
        [gapped, continuous],
        sliding_window=4,
    )

    assert metadata.segment_table_indices.tolist() == [2, 5]
    assert gapped.true_positions[_segment_keys(metadata, 0)].tolist() == [10, 11]
    assert continuous.true_positions[_segment_keys(metadata, 1)].tolist() == [20, 21, 22, 23]


def test_historical_gap_keeps_fast_path_when_recent_window_is_contiguous() -> None:
    positions = torch.cat(
        [
            torch.arange(0, 10, dtype=torch.int32),
            torch.arange(100, 228, dtype=torch.int32),
        ]
    )
    req = _req(positions, cached_len=len(positions) - 1)

    assert not sliding_window_crosses_gap(
        positions,
        device_len=len(positions),
        sliding_window=128,
    )
    assert not batch_needs_gap_aware_sliding_window(
        [req],
        sliding_window=128,
        decode_only=True,
    )


def test_decode_graph_eligibility_recovers_at_128_contiguous_tokens() -> None:
    positions = torch.cat(
        [
            torch.arange(0, 3578, dtype=torch.int32),
            torch.arange(4475, 4578, dtype=torch.int32),
        ]
    )

    for continuous_tail in range(103, 129):
        if continuous_tail > 103:
            positions = torch.cat(
                [positions, torch.tensor([int(positions[-1]) + 1], dtype=torch.int32)]
            )
        assert sliding_window_crosses_gap(
            positions,
            device_len=len(positions),
            sliding_window=128,
        ) is (continuous_tail < 128)


@pytest.mark.parametrize("backend_cls", [FlashInferBackend, FlashAttentionBackend])
def test_attention_backend_rejects_graph_for_any_gapped_request(backend_cls) -> None:
    backend = object.__new__(backend_cls)
    backend.config = SimpleNamespace(sliding_window=4)
    gapped = _req([0, 1, 10, 11], cached_len=3)
    continuous = _req([0, 1, 2, 3], cached_len=3)
    batch = SimpleNamespace(reqs=[gapped, continuous])

    assert not backend.can_use_cuda_graph(batch)
    batch.reqs = [continuous]
    assert backend.can_use_cuda_graph(batch)


def test_graph_runner_uses_backend_eligibility_and_ignores_padding() -> None:
    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = 4
    runner.attn_backend = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    batch = SimpleNamespace(is_decode=True, size=2)

    assert not runner.can_use_cuda_graph(batch)
    runner.attn_backend = SimpleNamespace(can_use_cuda_graph=lambda batch: True)
    assert runner.can_use_cuda_graph(batch)


def test_compiled_page_tables_preserve_segment_ownership_and_padding() -> None:
    first = _req([0, 1, 10, 11], cached_len=3, table_idx=0)
    second = _req([20, 21, 22, 23], cached_len=3, table_idx=1)
    metadata = build_sliding_window_attention_batch([first, second], sliding_window=4)
    page_table = torch.tensor(
        [
            [100, 101, 102, 103],
            [200, 201, 202, 203],
        ],
        dtype=torch.int32,
    )

    compiled = compile_context_page_tables(page_table, metadata)

    assert compiled.flat_indices.tolist() == [102, 103, 200, 201, 202, 203]
    assert compiled.padded_page_table.tolist() == [
        [102, 103, 0, 0],
        [200, 201, 202, 203],
    ]


@pytest.mark.parametrize(
    ("positions", "device_len", "message"),
    [
        (torch.tensor([[0, 1]], dtype=torch.int32), 2, "one-dimensional"),
        (torch.tensor([0, 1], dtype=torch.int32), 1, "exactly device_len"),
        (torch.tensor([0, 0], dtype=torch.int32), 2, "strictly increasing"),
        (torch.tensor([1, 0], dtype=torch.int32), 2, "strictly increasing"),
    ],
)
def test_true_position_validation(
    positions: torch.Tensor,
    device_len: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_active_true_positions(positions, device_len=device_len)


class _FakeKVCache:
    def __init__(self) -> None:
        self.cache = torch.zeros((4, 1, 1, 2))

    def store_kv(self, *args) -> None:
        return None

    def k_cache(self, layer_id: int) -> torch.Tensor:
        return self.cache

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self.cache


def test_flashattention_uses_segments_only_for_sliding_layers(monkeypatch) -> None:
    backend = object.__new__(FlashAttentionBackend)
    backend.kvcache = _FakeKVCache()
    backend.scale = 1.0
    segment = SimpleNamespace(query_start=0, query_end=1)
    metadata = FAMetadata(
        cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cache_seqlens=torch.tensor([1], dtype=torch.int32),
        max_seqlen_k=1,
        max_seqlen_q=1,
        page_table=torch.tensor([[0]], dtype=torch.int32),
        sliding_context_segments=segment,
    )
    batch = SimpleNamespace(
        attn_metadata=metadata,
        out_loc=torch.tensor([0], dtype=torch.int32),
    )
    q = torch.zeros((1, 1, 2))
    k = torch.zeros((1, 2))
    v = torch.zeros((1, 2))
    calls = []

    def segmented(**kwargs):
        calls.append(("segmented", kwargs["window_size"], kwargs["sinks"]))
        return torch.ones_like(q)

    def ordinary(**kwargs):
        calls.append(("ordinary", kwargs["window_size"], kwargs["sinks"]))
        return torch.full_like(q, 2)

    monkeypatch.setattr(fa_module, "_fa3_context_mask_impl", segmented)
    monkeypatch.setattr(fa_module, "_fa_sgl_impl", ordinary)

    sinks = torch.tensor([0.0])
    sliding_out = backend.forward(q, k, v, 0, batch, sinks=sinks, sliding_window=127)
    full_out = backend.forward(q, k, v, 0, batch, sinks=sinks)

    assert torch.equal(sliding_out, torch.ones_like(q))
    assert torch.equal(full_out, torch.full_like(q, 2))
    assert calls == [
        ("segmented", (-1, -1), sinks),
        ("ordinary", (-1, -1), sinks),
    ]


def test_flashinfer_gap_decode_uses_decode_wrapper_and_preserves_sinks() -> None:
    backend = object.__new__(FlashInferBackend)
    backend.kvcache = _FakeKVCache()
    decode_wrappers = []
    initialization = []

    class Wrapper:
        def run(self, *, q, paged_kv_cache, return_lse=False, **kwargs):
            del paged_kv_cache, kwargs
            out = torch.ones_like(q)
            if return_lse:
                return out, torch.zeros(out.shape[:2])
            return out

    def new_decode_wrapper():
        wrapper = Wrapper()
        decode_wrappers.append(wrapper)
        return wrapper

    def fail_prefill_wrapper():
        raise AssertionError("Gap-aware decode must use the paged decode wrapper.")

    def initialize(metadata, wrapper, *, is_decode, window_left):
        initialization.append((metadata, wrapper, is_decode, window_left))

    backend._new_decode_wrapper = new_decode_wrapper
    backend._new_prefill_wrapper = fail_prefill_wrapper
    backend._initialize_metadata_once = initialize

    segment = SimpleNamespace(
        query_start=0,
        query_end=1,
        is_decode=True,
        wrappers={},
    )
    metadata = object.__new__(FIMetadata)
    metadata.context_segments = None
    metadata.sliding_context_segments = segment
    metadata.is_decode = True
    metadata.graph_bs = None
    batch = SimpleNamespace(
        attn_metadata=metadata,
        out_loc=torch.tensor([0], dtype=torch.int32),
    )
    q = torch.zeros((1, 1, 2))
    k = torch.zeros((1, 2))
    v = torch.zeros((1, 2))

    out = backend.forward(
        q,
        k,
        v,
        0,
        batch,
        sinks=torch.tensor([0.0]),
        sliding_window=127,
    )

    torch.testing.assert_close(out, torch.full_like(q, 0.5))
    assert len(decode_wrappers) == 1
    assert initialization == [(segment, decode_wrappers[0], True, -1)]
