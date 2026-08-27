from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.base import (
    build_sliding_window_attention_batch,
    compile_context_page_tables,
)
from minisgl.attention.fa import _fa_sgl_impl


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for attention-kernel numerical tests.")
    return torch.device("cuda:0")


def _case(device: torch.device):
    torch.manual_seed(0)
    positions = torch.cat(
        [
            torch.arange(0, 20, dtype=torch.int32),
            torch.arange(50, 57, dtype=torch.int32),
        ]
    )
    req = SimpleNamespace(
        true_positions=positions,
        device_len=len(positions),
        cached_len=len(positions) - 3,
        table_idx=0,
    )
    metadata = build_sliding_window_attention_batch([req], sliding_window=8)
    page_table = torch.arange(
        len(positions),
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    compiled = compile_context_page_tables(page_table, metadata)
    num_kv_heads, num_qo_heads, head_dim = 1, 4, 64
    k_cache = torch.randn(
        len(positions),
        1,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(
        metadata.num_queries,
        num_qo_heads,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    return metadata, compiled, q, k_cache, v_cache


def _oracle(metadata, q, k_cache, v_cache):
    num_qo_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    refs = []
    logsumexp = []
    for segment_idx in range(metadata.num_segments):
        start = int(metadata.cu_seqlens_k[segment_idx])
        end = int(metadata.cu_seqlens_k[segment_idx + 1])
        keys = metadata.key_positions[start:end].to(
            dtype=torch.int64,
            device=q.device,
        )
        k = (
            k_cache[:, 0]
            .index_select(0, keys)
            .repeat_interleave(
                num_qo_heads // num_kv_heads,
                dim=1,
            )
        )
        v = (
            v_cache[:, 0]
            .index_select(0, keys)
            .repeat_interleave(
                num_qo_heads // num_kv_heads,
                dim=1,
            )
        )
        scores = torch.einsum("hd,khd->hk", q[segment_idx].float(), k.float())
        scores /= math.sqrt(q.shape[-1])
        refs.append(torch.einsum("hk,khd->hd", scores.softmax(-1), v.float()))
        logsumexp.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(refs).to(q.dtype), torch.stack(logsumexp)


def test_flashinfer_gap_aware_prefill_matches_absolute_oracle() -> None:
    flashinfer = pytest.importorskip("flashinfer")
    device = _cuda_or_skip()
    metadata, compiled, q, k_cache, v_cache = _case(device)
    reference, reference_lse = _oracle(metadata, q, k_cache, v_cache)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend="fa2",
    )
    cu_seqlens_q = metadata.cu_seqlens_q.pin_memory()
    cu_seqlens_k = metadata.cu_seqlens_k.pin_memory()
    seq_lens = (metadata.cu_seqlens_k[1:] - metadata.cu_seqlens_k[:-1]).pin_memory()
    wrapper.plan(
        qo_indptr=cu_seqlens_q,
        paged_kv_indptr=cu_seqlens_k,
        paged_kv_indices=compiled.flat_indices,
        paged_kv_last_page_len=torch.ones(
            metadata.num_segments,
            dtype=torch.int32,
            pin_memory=True,
        ),
        num_qo_heads=q.shape[1],
        num_kv_heads=k_cache.shape[2],
        head_dim_qk=q.shape[2],
        page_size=1,
        pos_encoding_mode="NONE",
        window_left=-1,
        seq_lens=seq_lens,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        causal=True,
    )

    out, lse = wrapper.run(
        q=q,
        paged_kv_cache=(k_cache, v_cache),
        window_left=-1,
        return_lse=True,
    )
    torch.testing.assert_close(out, reference, atol=2e-3, rtol=2e-3)

    sinks = torch.linspace(-1.0, 1.0, q.shape[1], device=device)
    sink_scale = torch.sigmoid(lse.float() * math.log(2.0) - sinks.reshape(1, -1))
    expected_scale = torch.sigmoid(reference_lse - sinks.reshape(1, -1))
    torch.testing.assert_close(sink_scale, expected_scale, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        out.float() * sink_scale.unsqueeze(-1),
        reference.float() * expected_scale.unsqueeze(-1),
        atol=2e-3,
        rtol=2e-3,
    )


def test_flashinfer_gap_aware_decode_matches_absolute_oracle() -> None:
    flashinfer = pytest.importorskip("flashinfer")
    device = _cuda_or_skip()
    metadata, compiled, q, k_cache, v_cache = _case(device)
    reference, _ = _oracle(metadata, q, k_cache, v_cache)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        use_tensor_cores=True,
        kv_layout="NHD",
        backend="fa2",
    )
    wrapper.plan(
        indptr=metadata.cu_seqlens_k.pin_memory(),
        indices=compiled.flat_indices,
        last_page_len=torch.ones(
            metadata.num_segments,
            dtype=torch.int32,
            pin_memory=True,
        ),
        num_qo_heads=q.shape[1],
        num_kv_heads=k_cache.shape[2],
        head_dim=q.shape[2],
        page_size=1,
        pos_encoding_mode="NONE",
        window_left=-1,
        seq_lens=(metadata.cu_seqlens_k[1:] - metadata.cu_seqlens_k[:-1]).pin_memory(),
        data_type=k_cache.dtype,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )

    out = wrapper.run(
        q=q,
        paged_kv_cache=(k_cache, v_cache),
        window_left=-1,
    )
    torch.testing.assert_close(out, reference, atol=2e-3, rtol=2e-3)


def test_flashattention_gap_aware_prefill_matches_absolute_oracle() -> None:
    device = _cuda_or_skip()
    pytest.importorskip("sgl_kernel.flash_attn")
    metadata, compiled, q, k_cache, v_cache = _case(device)
    reference, _ = _oracle(metadata, q, k_cache, v_cache)
    cu_seqlens_q = metadata.cu_seqlens_q.to(device)
    cu_seqlens_k = metadata.cu_seqlens_k.to(device)

    out = _fa_sgl_impl(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=compiled.padded_page_table,
        cache_seqlens=cu_seqlens_k[1:] - cu_seqlens_k[:-1],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q,
        softmax_scale=q.shape[-1] ** -0.5,
        window_size=(-1, -1),
    )

    torch.testing.assert_close(out, reference, atol=2e-3, rtol=2e-3)
