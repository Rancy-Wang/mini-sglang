import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from minisgl.attention.base import (
    HybridBackend,
    build_context_attention_batch,
    build_context_attention_segments,
    compile_context_page_tables,
)
from minisgl.attention.fa import (
    FAContextSegmentMetadata,
    FAMetadata,
    FlashAttentionBackend,
    _fa3_context_mask_impl,
    _fa_sgl_impl,
)
from minisgl.attention.fi import FIMetadata, FlashInferBackend
from minisgl.core import build_context_visibility_mask_reference
from minisgl.layers.attention import AttentionLayer


class _FakeKVCache:
    def __init__(self):
        self.stored = None
        self.key = torch.empty(2, 1, 1, 4)
        self.value = torch.empty(2, 1, 1, 4)

    def store_kv(self, k, v, out_loc, layer_id):
        self.stored = (k, v, out_loc, layer_id)

    def k_cache(self, layer_id):
        return self.key

    def v_cache(self, layer_id):
        return self.value


def _fa_metadata():
    return FAMetadata(
        cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cache_seqlens=torch.tensor([2], dtype=torch.int32),
        max_seqlen_k=2,
        max_seqlen_q=1,
        page_table=torch.tensor([[0, 1]], dtype=torch.int32),
    )


def test_flash_attention_forwards_sinks_and_window(monkeypatch):
    backend = object.__new__(FlashAttentionBackend)
    backend.kvcache = _FakeKVCache()
    backend.scale = 0.125
    backend.version = 3
    batch = SimpleNamespace(attn_metadata=_fa_metadata(), out_loc=object())
    q = torch.empty(1, 1, 4)
    k = torch.empty(1, 1, 4)
    v = torch.empty(1, 1, 4)
    sinks = torch.empty(1)
    expected = torch.empty_like(q)
    implementation = MagicMock(return_value=expected)
    monkeypatch.setattr("minisgl.attention.fa._fa_sgl_impl", implementation)

    output = backend.forward(q, k, v, 3, batch, sinks=sinks, sliding_window=127)

    assert output is expected
    call = implementation.call_args.kwargs
    assert call["sinks"] is sinks
    assert call["window_size"] == (127, 0)
    assert call["context_mask_aux"] is None


def test_fa3_kernel_receives_sinks_and_window(monkeypatch):
    output = torch.ones(2, 1, 4)
    lse = torch.zeros(1, 2)
    kernel = MagicMock(return_value=(output, lse, None, None))
    package = ModuleType("sgl_kernel")
    package.__path__ = []
    module = ModuleType("sgl_kernel.flash_attn")
    module.flash_attn_with_kvcache = kernel
    monkeypatch.setitem(sys.modules, "sgl_kernel", package)
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_attn", module)
    tensor = torch.empty(2, 1, 4)
    sinks = torch.zeros(1)

    corrected = _fa_sgl_impl(
        q=tensor,
        k_cache=tensor,
        v_cache=tensor,
        page_table=tensor,
        cache_seqlens=tensor,
        cu_seqlens_q=tensor,
        cu_seqlens_k=tensor,
        max_seqlen_q=1,
        softmax_scale=0.125,
        version=3,
        window_size=(127, 0),
        sinks=sinks,
    )

    call = kernel.call_args.kwargs
    assert call["sinks"] is None
    assert call["return_softmax_lse"] is True
    assert call["window_size"] == (127, 0)
    assert torch.equal(corrected, torch.full_like(output, 0.5))


def test_context_segments_match_dense_mask_with_cached_prefix_and_drops():
    visible_until = torch.tensor([4, 9, 6, 4, 9, 6, 9, 9, 9], dtype=torch.int32)
    query_start = 3
    query_length = 6
    segments = build_context_attention_segments(
        visible_until,
        query_start=query_start,
        query_length=query_length,
        key_length=9,
    )
    reconstructed = torch.zeros((query_length, 9), dtype=torch.bool)
    for segment in segments:
        query_count = segment.query_end - segment.query_start
        prefix_count = len(segment.key_positions) - query_count
        for local_query in range(query_count):
            reconstructed[
                segment.query_start + local_query,
                segment.key_positions[: prefix_count + local_query + 1],
            ] = True

    expected = build_context_visibility_mask_reference(
        visible_until,
        query_positions=torch.arange(query_start, query_start + query_length),
        key_positions=torch.arange(9),
    )
    assert torch.equal(reconstructed, expected)


def test_context_sliding_segments_use_absolute_positions_across_drops():
    visible_until = torch.tensor([4, 6, 3, 6, 6, 6], dtype=torch.int32)
    query_start = 3
    query_length = 3
    window_left = 2
    segments = build_context_attention_segments(
        visible_until,
        query_start=query_start,
        query_length=query_length,
        key_length=6,
        sliding_window=window_left,
    )
    reconstructed = torch.zeros((query_length, 6), dtype=torch.bool)
    for segment in segments:
        assert segment.query_end - segment.query_start == 1
        reconstructed[segment.query_start, segment.key_positions] = True

    query_positions = torch.arange(query_start, query_start + query_length)
    expected = build_context_visibility_mask_reference(
        visible_until,
        query_positions=query_positions,
        key_positions=torch.arange(6),
    )
    expected &= torch.arange(6).unsqueeze(0) >= (query_positions - window_left).unsqueeze(1)
    assert torch.equal(reconstructed, expected)


def _context_batch_requests():
    return (
        SimpleNamespace(
            table_idx=1,
            cached_len=2,
            extend_len=4,
            device_len=6,
            full_token_visible_until=torch.tensor(
                [4, 6, 6, 4, 6, 6], dtype=torch.int32
            ),
        ),
        SimpleNamespace(
            table_idx=3,
            cached_len=1,
            extend_len=4,
            device_len=5,
            full_token_visible_until=torch.tensor(
                [3, 5, 3, 5, 5], dtype=torch.int32
            ),
        ),
    )


def test_multi_request_context_batch_preserves_flattened_q_and_table_ownership():
    requests = _context_batch_requests()
    context_batch = build_context_attention_batch(requests)
    page_table = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)

    compiled = compile_context_page_tables(page_table, context_batch)

    assert context_batch.num_queries == sum(req.extend_len for req in requests)
    assert context_batch.segment_table_indices.tolist() == [1, 1, 3, 3]
    request_q_start = {1: 0, 3: requests[0].extend_len}
    reconstructed = {
        req.table_idx: torch.zeros((req.extend_len, req.device_len), dtype=torch.bool)
        for req in requests
    }
    for segment_idx, table_idx in enumerate(context_batch.segment_table_indices.tolist()):
        query_start = int(context_batch.cu_seqlens_q[segment_idx])
        query_end = int(context_batch.cu_seqlens_q[segment_idx + 1])
        key_start = int(context_batch.cu_seqlens_k[segment_idx])
        key_end = int(context_batch.cu_seqlens_k[segment_idx + 1])
        positions = context_batch.key_positions[key_start:key_end].to(torch.int64)
        expected = page_table[table_idx].index_select(0, positions)
        assert torch.equal(compiled.flat_indices[key_start:key_end], expected)
        assert torch.equal(
            compiled.padded_page_table[segment_idx, : len(expected)], expected
        )
        query_count = query_end - query_start
        prefix_count = len(positions) - query_count
        local_query_start = query_start - request_q_start[table_idx]
        for query_offset in range(query_count):
            reconstructed[table_idx][
                local_query_start + query_offset,
                positions[: prefix_count + query_offset + 1],
            ] = True

    for req in requests:
        expected_mask = build_context_visibility_mask_reference(
            req.full_token_visible_until,
            query_positions=torch.arange(req.cached_len, req.device_len),
            key_positions=torch.arange(req.device_len),
        )
        assert torch.equal(reconstructed[req.table_idx], expected_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton")
def test_context_page_table_gpu_kernel_matches_cpu_reference():
    context_batch = build_context_attention_batch(
        _context_batch_requests(), sliding_window=2
    )
    page_table_cpu = torch.arange(4 * 8, dtype=torch.int32).view(4, 8)

    expected = compile_context_page_tables(page_table_cpu, context_batch)
    actual = compile_context_page_tables(page_table_cpu.cuda(), context_batch)
    torch.cuda.synchronize()

    assert torch.equal(actual.flat_indices.cpu(), expected.flat_indices)
    assert torch.equal(actual.padded_page_table.cpu(), expected.padded_page_table)


def test_flash_attention_uses_preclipped_context_segments_for_sliding(monkeypatch):
    backend = object.__new__(FlashAttentionBackend)
    backend.kvcache = _FakeKVCache()
    backend.scale = 0.125
    backend.version = 3
    metadata = _fa_metadata()
    metadata.context_segments = (object(),)
    metadata.sliding_context_segments = (object(), object())
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=object())
    expected = torch.empty(1, 1, 4)
    implementation = MagicMock(return_value=expected)
    monkeypatch.setattr("minisgl.attention.fa._fa3_context_mask_impl", implementation)
    q = torch.empty(1, 1, 4)

    output = backend.forward(q, q, q, 0, batch, sliding_window=127)

    assert output is expected
    call = implementation.call_args.kwargs
    assert call["segments"] is metadata.sliding_context_segments
    assert call["window_size"] == (-1, -1)


def test_fa3_context_segments_preserve_window_and_sinks(monkeypatch):
    kernel = MagicMock(side_effect=lambda **kwargs: kwargs["q"])
    monkeypatch.setattr("minisgl.attention.fa._fa_sgl_impl", kernel)
    sinks = torch.empty(1)
    segments = FAContextSegmentMetadata(
        query_start=0,
        query_end=3,
        page_table=torch.tensor([[7, 8, 9], [8, 10, 0]], dtype=torch.int32),
        cache_seqlens=torch.tensor([3, 2], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_seqlen_q=2,
    )
    q = torch.empty(3, 1, 4)

    output = _fa3_context_mask_impl(
        q=q,
        k_cache=torch.empty(1),
        v_cache=torch.empty(1),
        segments=segments,
        softmax_scale=0.125,
        window_size=(127, 0),
        sinks=sinks,
    )

    assert output.shape == q.shape
    kernel.assert_called_once()
    call = kernel.call_args
    assert call.kwargs["version"] == 3
    assert call.kwargs["window_size"] == (127, 0)
    assert call.kwargs["sinks"] is sinks
    assert call.kwargs["max_seqlen_q"] == 2


def _fi_metadata():
    metadata = object.__new__(FIMetadata)
    metadata.is_decode = False
    metadata.context_segments = None
    metadata.graph_bs = None
    metadata.initialized_wrappers = set()
    return metadata


def test_flashinfer_forwards_sinks_and_matching_window():
    backend = object.__new__(FlashInferBackend)
    backend.kvcache = _FakeKVCache()
    backend._initialize_metadata_once = MagicMock()
    wrapper = MagicMock()
    wrapper.run.return_value = (torch.ones(1, 1, 4), torch.ones(1, 1))
    backend._ordinary_wrapper = MagicMock(return_value=wrapper)
    metadata = _fi_metadata()
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=object())
    q = torch.empty(1, 1, 4)
    k = torch.empty(1, 1, 4)
    v = torch.empty(1, 1, 4)
    sinks = torch.zeros(1)

    output = backend.forward(q, k, v, 5, batch, sinks=sinks, sliding_window=127)

    call = wrapper.run.call_args.kwargs
    assert call["q"] is q
    assert call["return_lse"] is True
    assert call["window_left"] == 127
    assert torch.allclose(output, torch.full_like(output, 2 / 3))
    backend._initialize_metadata_once.assert_called_once_with(
        metadata,
        wrapper,
        is_decode=False,
        window_left=127,
    )


def test_flashinfer_ordinary_forward_keeps_original_run_arguments():
    backend = object.__new__(FlashInferBackend)
    backend.kvcache = _FakeKVCache()
    backend._initialize_metadata_once = MagicMock()
    wrapper = MagicMock()
    backend._ordinary_wrapper = MagicMock(return_value=wrapper)
    metadata = _fi_metadata()
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=object())
    q = torch.empty(1, 1, 4)

    backend.forward(q, q, q, 0, batch)

    assert set(wrapper.run.call_args.kwargs) == {"q", "paged_kv_cache", "window_left"}
    assert wrapper.run.call_args.kwargs["window_left"] == -1


def test_flashinfer_plans_full_and_sliding_wrappers_with_matching_windows():
    backend = object.__new__(FlashInferBackend)
    backend.qo_head_local = 2
    backend.kv_head_local = 1
    backend.config = SimpleNamespace(head_dim=4)
    backend.kvcache = SimpleNamespace(dtype=torch.bfloat16)
    backend.last_event = MagicMock()
    metadata = _fi_metadata()
    metadata.cu_seqlens_q_cpu = torch.tensor([0, 2], dtype=torch.int32)
    metadata.cu_seqlens_k_cpu = torch.tensor([0, 3], dtype=torch.int32)
    metadata.indices = torch.tensor([4, 5, 6], dtype=torch.int32)
    metadata.last_page_len_cpu = torch.tensor([1], dtype=torch.int32)
    metadata.seq_lens_cpu = torch.tensor([3], dtype=torch.int32)
    metadata.num_qo_heads = 2
    metadata.num_kv_heads = 1
    metadata.head_dim = 4
    metadata.page_size = 1
    metadata.pos_encoding_mode = "NONE"
    metadata.dtype = torch.bfloat16
    full_wrapper = MagicMock()
    sliding_wrapper = MagicMock()

    backend._initialize_metadata_once(
        metadata, full_wrapper, is_decode=False, window_left=-1
    )
    backend._initialize_metadata_once(
        metadata, sliding_wrapper, is_decode=False, window_left=127
    )

    assert full_wrapper.plan.call_args.kwargs["window_left"] == -1
    assert sliding_wrapper.plan.call_args.kwargs["window_left"] == 127


def test_flashinfer_uses_preclipped_context_segments_for_sliding():
    backend = object.__new__(FlashInferBackend)
    backend.kvcache = _FakeKVCache()
    backend._initialize_metadata_once = MagicMock()
    wrapper = MagicMock()
    wrapper.run.side_effect = lambda **kwargs: kwargs["q"]
    backend._new_prefill_wrapper = MagicMock(return_value=wrapper)
    metadata = _fi_metadata()
    metadata.context_segments = SimpleNamespace(query_start=0, query_end=1, wrappers={})
    metadata.sliding_context_segments = SimpleNamespace(
        query_start=0, query_end=1, wrappers={}
    )
    batch = SimpleNamespace(attn_metadata=metadata, out_loc=object())
    q = torch.empty(1, 1, 4)

    output = backend.forward(q, q, q, 0, batch, sliding_window=127)

    assert torch.equal(output, q)
    backend._initialize_metadata_once.assert_called_once_with(
        metadata.sliding_context_segments,
        wrapper,
        is_decode=False,
        window_left=-1,
    )
    assert wrapper.run.call_args.kwargs["window_left"] == -1


def test_hybrid_ordinary_forward_does_not_expand_legacy_backend_call():
    prefill_backend = MagicMock()
    decode_backend = MagicMock()
    backend = HybridBackend(prefill_backend, decode_backend)
    batch = SimpleNamespace(is_prefill=True)
    q = object()
    k = object()
    v = object()

    backend.forward(q, k, v, 2, batch)

    prefill_backend.forward.assert_called_once_with(q, k, v, 2, batch)
    decode_backend.forward.assert_not_called()


def test_attention_layer_forwards_gpt_oss_parameters(monkeypatch):
    layer = object.__new__(AttentionLayer)
    layer.layer_id = 4
    layer.num_qo_heads = 1
    layer.num_kv_heads = 1
    layer.head_dim = 4
    layer.qo_attn_dim = 4
    layer.kv_attn_dim = 4
    layer.q_norm = None
    layer.k_norm = None
    layer.rotary = SimpleNamespace(forward=lambda positions, q, k: (q, k))
    layer.sinks = torch.empty(1)
    layer.sliding_window = 127
    backend = MagicMock()
    backend.forward.return_value = torch.empty(2, 1, 4)
    batch = SimpleNamespace(positions=torch.arange(2))
    monkeypatch.setattr(
        "minisgl.layers.attention.get_global_ctx",
        lambda: SimpleNamespace(batch=batch, attn_backend=backend),
    )

    output = layer.forward(torch.empty(2, 12))

    assert output.shape == (2, 4)
    call = backend.forward.call_args
    assert call.args[3:] == (4, batch)
    assert call.kwargs["sinks"] is layer.sinks
    assert call.kwargs["sliding_window"] == 127
