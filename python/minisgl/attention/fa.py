from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx

from .base import (
    BaseAttnBackend,
    BaseAttnMetadata,
    batch_needs_gap_aware_sliding_window,
    build_context_attention_batch,
    build_sliding_window_attention_batch,
    compile_context_page_tables,
)
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass(frozen=True)
class FAContextSegmentMetadata:
    query_start: int
    query_end: int
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor
    context_segments: FAContextSegmentMetadata | None = None
    sliding_context_segments: FAContextSegmentMetadata | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


def is_fa_context_mask_supported(device: torch.device | int | None = None) -> bool:
    """Whether the exact FA3 segmented adapter is available."""

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major in (8, 9)


def validate_fa_context_mask_support(device: torch.device | int | None = None) -> None:
    if not is_fa_context_mask_supported(device):
        capability = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None
        raise ValueError(
            "--contextual-prefill-mode mask with FlashAttention requires the FA3 segmented "
            "adapter on SM80/SM90; "
            f"current CUDA capability is {capability}."
        )

    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention context-mask mode on SM80/SM90 requires a working "
            "sgl-kernel FA3 installation for the selected architecture."
        ) from exc


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5

    def validate_context_mask_prefill(self, device: torch.device | int | None = None) -> None:
        validate_fa_context_mask_support(device)

    @property
    def supports_multi_context_mask_prefill(self) -> bool:
        return True

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return not batch_needs_gap_aware_sliding_window(
            batch.reqs,
            sliding_window=self.config.sliding_window,
            decode_only=True,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        sinks: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        segments = None
        window_size = (sliding_window, 0) if sliding_window is not None else (-1, -1)
        if sliding_window is not None and metadata.sliding_context_segments is not None:
            segments = metadata.sliding_context_segments
            window_size = (-1, -1)
        elif metadata.context_segments is not None:
            if sliding_window is not None:
                raise RuntimeError(
                    "FlashAttention context-mask metadata is missing absolute-position "
                    "sliding segments."
                )
            segments = metadata.context_segments

        if segments is not None:
            return _fa3_context_mask_impl(
                q=q,
                k_cache=self.kvcache.k_cache(layer_id),
                v_cache=self.kvcache.v_cache(layer_id),
                segments=segments,
                softmax_scale=self.scale,
                window_size=window_size,
                sinks=sinks,
            )
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            window_size=(sliding_window, 0) if sliding_window is not None else (-1, -1),
            sinks=sinks,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        masked_reqs = [req for req in reqs if req.use_context_mask]
        if masked_reqs:
            if not batch.is_prefill or len(masked_reqs) != len(reqs):
                raise RuntimeError(
                    "FlashAttention context-mask Prefill cannot mix masked and ordinary requests."
                )
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")

        context_segments = None
        sliding_context_segments = None

        def _compile_fa_segments(context_batch):
            compiled = compile_context_page_tables(page_table, context_batch)
            context_cu_q = context_batch.cu_seqlens_q.pin_memory().to(device, non_blocking=True)
            context_cu_k = context_batch.cu_seqlens_k.pin_memory().to(device, non_blocking=True)
            return FAContextSegmentMetadata(
                query_start=0,
                query_end=context_batch.num_queries,
                page_table=compiled.padded_page_table,
                cache_seqlens=(context_cu_k[1:] - context_cu_k[:-1]),
                cu_seqlens_q=context_cu_q,
                cu_seqlens_k=context_cu_k,
                max_seqlen_q=context_batch.max_seqlen_q,
            )

        if masked_reqs:
            if self.page_size != 1:
                raise RuntimeError(
                    "FlashAttention context-mask Prefill currently requires page_size=1."
                )

            def _compile_fa_context(sliding_window: int | None):
                context_batch = build_context_attention_batch(
                    masked_reqs, sliding_window=sliding_window
                )
                if sliding_window is None:
                    for req, cached_tokens in zip(
                        masked_reqs, context_batch.cached_tokens, strict=True
                    ):
                        if req.usage_cached_tokens is None:
                            req.record_context_cache_usage(cached_tokens)
                return _compile_fa_segments(context_batch)

            context_segments = _compile_fa_context(None)
            if self.config.sliding_window is not None:
                sliding_context_segments = _compile_fa_context(self.config.sliding_window - 1)
        elif batch_needs_gap_aware_sliding_window(
            reqs,
            sliding_window=self.config.sliding_window,
            decode_only=batch.is_decode,
        ):
            if self.page_size != 1:
                raise RuntimeError(
                    "Gap-aware FlashAttention sliding windows currently require page_size=1."
                )
            assert self.config.sliding_window is not None
            sliding_batch = build_sliding_window_attention_batch(
                reqs,
                sliding_window=self.config.sliding_window,
            )
            sliding_context_segments = _compile_fa_segments(sliding_batch)
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
            context_segments=context_segments,
            sliding_context_segments=sliding_context_segments,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def _fa3_context_mask_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    segments: FAContextSegmentMetadata,
    softmax_scale: float,
    window_size: Tuple[int, int],
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Run exact Context mask Prefill as causal FA3 over active-KV segments."""

    query = q[segments.query_start : segments.query_end]
    return _fa_sgl_impl(
        q=query,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=segments.page_table,
        cache_seqlens=segments.cache_seqlens,
        cu_seqlens_q=segments.cu_seqlens_q,
        cu_seqlens_k=segments.cu_seqlens_k,
        max_seqlen_q=segments.max_seqlen_q,
        softmax_scale=softmax_scale,
        window_size=window_size,
        sinks=sinks,
    )


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    kwargs = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k_new": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "softmax_scale": softmax_scale,
        "sm_margin": sm_margin,
        "window_size": window_size,
        "softcap": softcap,
        "num_splits": num_splits,
        "pack_gqa": pack_gqa,
        "causal": causal,
        "ver": 3,
    }
    if sinks is not None:
        # sgl-kernel FA3 accepts a sinks argument on SM80, but its kernel does
        # not include it in the softmax denominator. Recover the exact GPT-OSS
        # result from the ordinary output and per-row natural-log LSE:
        #   sum(exp(scores) * V) / (sum(exp(scores)) + exp(sink)).
        out, lse, *_ = flash_attn_with_kvcache(  # type: ignore
            **kwargs,
            sinks=None,
            return_softmax_lse=True,
        )
        if lse.shape != (out.shape[1], out.shape[0]):
            raise RuntimeError(
                "Unexpected FA3 LSE shape while applying GPT-OSS attention sinks: "
                f"output={tuple(out.shape)}, lse={tuple(lse.shape)}."
            )
        sink_scale = torch.sigmoid(lse.float() - sinks.float().reshape(-1, 1)).transpose(0, 1)
        return (out.float() * sink_scale.unsqueeze(-1)).to(out.dtype)

    return flash_attn_with_kvcache(  # type: ignore
        **kwargs,
        sinks=sinks,
    )
