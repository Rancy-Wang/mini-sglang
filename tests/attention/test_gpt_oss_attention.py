import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch
from minisgl.attention.base import HybridBackend, build_context_attention_segments
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


def test_fa3_context_segments_preserve_window_and_sinks(monkeypatch):
    kernel = MagicMock(side_effect=lambda **kwargs: kwargs["q"])
    monkeypatch.setattr("minisgl.attention.fa._fa_sgl_impl", kernel)
    sinks = torch.empty(1)
    segments = (
        FAContextSegmentMetadata(
            query_start=0,
            query_end=2,
            page_table=torch.tensor([[7, 8, 9]], dtype=torch.int32),
            cache_seqlens=torch.tensor([3], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
        ),
        FAContextSegmentMetadata(
            query_start=2,
            query_end=3,
            page_table=torch.tensor([[8, 10]], dtype=torch.int32),
            cache_seqlens=torch.tensor([2], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
        ),
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
    assert kernel.call_count == 2
    for call in kernel.call_args_list:
        assert call.kwargs["version"] == 3
        assert call.kwargs["window_size"] == (127, 0)
        assert call.kwargs["sinks"] is sinks


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
