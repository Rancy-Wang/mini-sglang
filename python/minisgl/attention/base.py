from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is only installed for the Linux CUDA runtime.
    triton = None
    tl = None

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass(frozen=True)
class ContextAttentionSegment:
    """One causal-attention segment over a compacted active-KV view."""

    query_start: int
    query_end: int
    key_positions: torch.Tensor


@dataclass(frozen=True)
class ContextAttentionBatch:
    """Ragged Context segments for a flattened multi-request Prefill batch."""

    segment_table_indices: torch.Tensor
    key_positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

    @property
    def num_segments(self) -> int:
        return len(self.segment_table_indices)

    @property
    def num_queries(self) -> int:
        return int(self.cu_seqlens_q[-1])


@dataclass(frozen=True)
class CompiledContextPageTables:
    """Backend layouts emitted together by the Context page-table compiler."""

    flat_indices: torch.Tensor
    padded_page_table: torch.Tensor


if triton is not None:

    @triton.jit
    def _compile_context_page_tables_kernel(
        page_table_ptr,
        segment_table_indices_ptr,
        key_positions_ptr,
        key_offsets_ptr,
        flat_indices_ptr,
        padded_page_table_ptr,
        page_table_stride,
        padded_stride,
        max_seqlen_k,
        BLOCK_SIZE: tl.constexpr,
    ):
        segment_idx = tl.program_id(0)
        key_block_idx = tl.program_id(1)
        local_key_idx = key_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        key_start = tl.load(key_offsets_ptr + segment_idx)
        key_end = tl.load(key_offsets_ptr + segment_idx + 1)
        key_count = key_end - key_start
        valid = local_key_idx < key_count
        key_position = tl.load(
            key_positions_ptr + key_start + local_key_idx,
            mask=valid,
            other=0,
        )
        table_idx = tl.load(segment_table_indices_ptr + segment_idx)
        page = tl.load(
            page_table_ptr + table_idx * page_table_stride + key_position,
            mask=valid,
            other=0,
        )
        tl.store(flat_indices_ptr + key_start + local_key_idx, page, mask=valid)
        padded_offset = segment_idx * padded_stride + local_key_idx
        tl.store(
            padded_page_table_ptr + padded_offset,
            page,
            mask=local_key_idx < max_seqlen_k,
        )

else:
    _compile_context_page_tables_kernel = None


def build_context_attention_segments(
    full_token_visible_until: torch.Tensor,
    *,
    query_start: int,
    query_length: int,
    key_length: int,
    sliding_window: int | None = None,
) -> tuple[ContextAttentionSegment, ...]:
    """Compile the exact token-position Drop mask into causal KV segments.

    Query offsets in the result are relative to the input Q tensor. Key
    positions stay on the full, absolute token axis so callers can gather the
    corresponding page-table entries without changing RoPE positions.
    """

    if full_token_visible_until.ndim != 1 or not full_token_visible_until.is_cpu:
        raise ValueError("full_token_visible_until must be a one-dimensional CPU tensor.")
    if query_start < 0 or query_length < 1 or key_length < 1:
        raise ValueError("Context attention lengths must be positive and query_start non-negative.")
    if sliding_window is not None and sliding_window < 0:
        raise ValueError("sliding_window must be non-negative when provided.")
    query_end = query_start + query_length
    if query_end > key_length or key_length > len(full_token_visible_until):
        raise ValueError("Context attention query/key bounds exceed the full token stream.")

    visible_until = full_token_visible_until[:key_length].to(dtype=torch.int64)
    key_positions = torch.arange(key_length, dtype=torch.int64, device="cpu")
    if bool(torch.any(visible_until <= key_positions).item()):
        raise ValueError("A token cannot become invisible before it has been computed.")

    if sliding_window is not None:
        # A compacted active-KV view has gaps in its absolute positions, so a
        # kernel window over compact indices is not equivalent to a model
        # window. Preselect each query's exact absolute-position window and run
        # causal attention without an additional kernel-side window.
        segments = []
        for absolute_query in range(query_start, query_end):
            prefix_positions = key_positions[
                max(0, absolute_query - sliding_window) : absolute_query
            ]
            active_prefix = prefix_positions[visible_until[prefix_positions] > absolute_query]
            segments.append(
                ContextAttentionSegment(
                    query_start=absolute_query - query_start,
                    query_end=absolute_query - query_start + 1,
                    key_positions=torch.cat(
                        (active_prefix, key_positions[absolute_query : absolute_query + 1])
                    ),
                )
            )
        return tuple(segments)

    boundaries = {query_start, query_end}
    expiring = visible_until[(visible_until > query_start) & (visible_until < query_end)]
    boundaries.update(int(position) for position in torch.unique(expiring).tolist())
    ordered = sorted(boundaries)

    segments = []
    for absolute_start, absolute_end in zip(ordered, ordered[1:]):
        prefix_positions = key_positions[:absolute_start]
        active_prefix = prefix_positions[visible_until[:absolute_start] > absolute_start]
        local_queries = key_positions[absolute_start:absolute_end]
        segments.append(
            ContextAttentionSegment(
                query_start=absolute_start - query_start,
                query_end=absolute_end - query_start,
                key_positions=torch.cat((active_prefix, local_queries)),
            )
        )
    return tuple(segments)


def build_context_attention_batch(
    reqs: Sequence,
    *,
    sliding_window: int | None = None,
) -> ContextAttentionBatch:
    """Compile per-request visibility into one ordered ragged attention batch.

    Segments are emitted request-by-request and query-order-preserving, so their
    concatenated Q layout is identical to the scheduler's flattened Prefill Q.
    Absolute key positions and table ownership remain explicit until the GPU
    page-table compiler resolves them.
    """

    if not reqs:
        raise ValueError("Context attention batching requires at least one request.")

    segment_table_indices = []
    key_positions = []
    query_lengths = []
    key_lengths = []
    expected_query_offset = 0
    for req in reqs:
        if req.full_token_visible_until is None:
            raise RuntimeError("Context-mask Prefill request is missing visibility metadata.")
        segments = build_context_attention_segments(
            req.full_token_visible_until,
            query_start=req.cached_len,
            query_length=req.extend_len,
            key_length=req.device_len,
            sliding_window=sliding_window,
        )
        local_query_offset = 0
        for segment in segments:
            if segment.query_start != local_query_offset:
                raise RuntimeError("Context segments do not form a contiguous query partition.")
            query_length = segment.query_end - segment.query_start
            segment_table_indices.append(req.table_idx)
            key_positions.append(segment.key_positions.to(dtype=torch.int32))
            query_lengths.append(query_length)
            key_lengths.append(len(segment.key_positions))
            local_query_offset = segment.query_end
        if local_query_offset != req.extend_len:
            raise RuntimeError("Context segments do not cover the complete request extension.")
        expected_query_offset += req.extend_len

    cu_seqlens_q = torch.tensor([0] + query_lengths, dtype=torch.int32).cumsum_(dim=0)
    cu_seqlens_k = torch.tensor([0] + key_lengths, dtype=torch.int32).cumsum_(dim=0)
    if int(cu_seqlens_q[-1]) != expected_query_offset:
        raise RuntimeError("Context batch query layout diverged from flattened Prefill Q.")
    return ContextAttentionBatch(
        segment_table_indices=torch.tensor(segment_table_indices, dtype=torch.int32),
        key_positions=torch.cat(key_positions),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(key_lengths),
    )


def compile_context_page_tables(
    page_table: torch.Tensor,
    context_batch: ContextAttentionBatch,
) -> CompiledContextPageTables:
    """Resolve absolute Context keys into FA/FI page layouts in one GPU kernel."""

    num_segments = context_batch.num_segments
    total_keys = len(context_batch.key_positions)
    flat_indices = torch.empty(total_keys, dtype=page_table.dtype, device=page_table.device)
    padded_page_table = torch.empty(
        (num_segments, context_batch.max_seqlen_k),
        dtype=page_table.dtype,
        device=page_table.device,
    )

    if page_table.is_cuda:
        if _compile_context_page_tables_kernel is None or triton is None:
            raise RuntimeError("CUDA Context mask compilation requires Triton.")
        table_indices = context_batch.segment_table_indices.pin_memory().to(
            page_table.device, non_blocking=True
        )
        key_positions = context_batch.key_positions.pin_memory().to(
            page_table.device, non_blocking=True
        )
        key_offsets = context_batch.cu_seqlens_k.pin_memory().to(
            page_table.device, non_blocking=True
        )
        block_size = 256
        grid = (num_segments, triton.cdiv(context_batch.max_seqlen_k, block_size))
        _compile_context_page_tables_kernel[grid](
            page_table,
            table_indices,
            key_positions,
            key_offsets,
            flat_indices,
            padded_page_table,
            page_table.stride(0),
            padded_page_table.stride(0),
            context_batch.max_seqlen_k,
            BLOCK_SIZE=block_size,
        )
    else:
        for segment_idx, table_idx in enumerate(context_batch.segment_table_indices.tolist()):
            key_start = int(context_batch.cu_seqlens_k[segment_idx])
            key_end = int(context_batch.cu_seqlens_k[segment_idx + 1])
            positions = context_batch.key_positions[key_start:key_end].to(dtype=torch.int64)
            pages = page_table[table_idx].index_select(0, positions)
            flat_indices[key_start:key_end] = pages
            padded_page_table[segment_idx, : len(pages)] = pages
            padded_page_table[segment_idx, len(pages) :] = 0

    return CompiledContextPageTables(
        flat_indices=flat_indices,
        padded_page_table=padded_page_table,
    )


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @property
    def supports_multi_context_mask_prefill(self) -> bool:
        return False

    @abstractmethod
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
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    @property
    def supports_multi_context_mask_prefill(self) -> bool:
        return self.prefill_backend.supports_multi_context_mask_prefill

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
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        if sinks is None and sliding_window is None:
            return backend.forward(q, k, v, layer_id, batch)
        return backend.forward(
            q,
            k,
            v,
            layer_id,
            batch,
            sinks=sinks,
            sliding_window=sliding_window,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
