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

    cached_tokens: tuple[int, ...]
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


def validate_active_true_positions(
    true_positions: torch.Tensor,
    *,
    device_len: int,
) -> torch.Tensor:
    """Validate and return the active absolute token positions as int64 on CPU."""

    if true_positions.ndim != 1 or not true_positions.is_cpu:
        raise ValueError("true_positions must be a one-dimensional CPU tensor.")
    if device_len < 1 or device_len != len(true_positions):
        raise ValueError(
            "true_positions must contain exactly device_len active positions, got "
            f"{len(true_positions)} positions for device_len={device_len}."
        )
    active_positions = true_positions.to(dtype=torch.int64)
    if len(active_positions) > 1 and bool(
        torch.any(active_positions[1:] <= active_positions[:-1]).item()
    ):
        raise ValueError("true_positions must be strictly increasing.")
    return active_positions


def sliding_window_crosses_gap(
    true_positions: torch.Tensor,
    *,
    device_len: int,
    sliding_window: int,
) -> bool:
    """Whether the last decode query's absolute window crosses a dropped span."""

    if sliding_window < 1:
        raise ValueError("sliding_window must be positive.")
    if true_positions.ndim != 1 or not true_positions.is_cpu:
        raise ValueError("true_positions must be a one-dimensional CPU tensor.")
    if device_len < 1 or device_len != len(true_positions):
        raise ValueError(
            "true_positions must contain exactly device_len active positions, got "
            f"{len(true_positions)} positions for device_len={device_len}."
        )
    token_count = min(sliding_window, device_len)
    recent_positions = true_positions[device_len - token_count : device_len].to(dtype=torch.int64)
    if len(recent_positions) > 1 and bool(
        torch.any(recent_positions[1:] <= recent_positions[:-1]).item()
    ):
        raise ValueError("true_positions must be strictly increasing.")
    return int(recent_positions[-1] - recent_positions[0]) + 1 != token_count


def batch_needs_gap_aware_sliding_window(
    reqs: Sequence,
    *,
    sliding_window: int | None,
    decode_only: bool = False,
) -> bool:
    """Whether any request/query cannot use a compact-index kernel window."""

    if sliding_window is None:
        return False
    if sliding_window < 1:
        raise ValueError("sliding_window must be positive.")

    if decode_only:
        return any(
            sliding_window_crosses_gap(
                req.true_positions,
                device_len=req.device_len,
                sliding_window=sliding_window,
            )
            for req in reqs
        )

    for req in reqs:
        active_positions = validate_active_true_positions(
            req.true_positions,
            device_len=req.device_len,
        )
        query_start = req.cached_len
        if not 0 <= query_start < req.device_len:
            raise ValueError("Sliding-window queries must satisfy 0 <= query_start < device_len.")
        for query_idx in range(query_start, req.device_len):
            token_count = min(sliding_window, query_idx + 1)
            first_idx = query_idx + 1 - token_count
            if int(active_positions[query_idx] - active_positions[first_idx]) + 1 != token_count:
                return True
    return False


def build_sliding_window_attention_batch(
    reqs: Sequence,
    *,
    sliding_window: int,
) -> ContextAttentionBatch:
    """Build exact absolute-position windows over compact active KV slots."""

    if not reqs:
        raise ValueError("Sliding-window batching requires at least one request.")
    if sliding_window < 1:
        raise ValueError("sliding_window must be positive.")

    segment_table_indices = []
    key_positions = []
    key_lengths = []
    cached_tokens = []
    expected_query_count = 0
    window_left = sliding_window - 1
    for req in reqs:
        active_positions = validate_active_true_positions(
            req.true_positions,
            device_len=req.device_len,
        )
        if not 0 <= req.cached_len < req.device_len:
            raise ValueError("Sliding-window requests must satisfy 0 <= cached_len < device_len.")
        first_key_count = None
        for query_idx in range(req.cached_len, req.device_len):
            query_position = active_positions[query_idx]
            left_idx = int(
                torch.searchsorted(
                    active_positions[: query_idx + 1],
                    query_position - window_left,
                    side="left",
                ).item()
            )
            compact_keys = torch.arange(left_idx, query_idx + 1, dtype=torch.int32)
            if first_key_count is None:
                first_key_count = len(compact_keys)
            segment_table_indices.append(req.table_idx)
            key_positions.append(compact_keys)
            key_lengths.append(len(compact_keys))
            expected_query_count += 1
        assert first_key_count is not None
        cached_tokens.append(first_key_count - 1)

    cu_seqlens_q = torch.arange(expected_query_count + 1, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0] + key_lengths, dtype=torch.int32).cumsum_(dim=0)
    return ContextAttentionBatch(
        cached_tokens=tuple(cached_tokens),
        segment_table_indices=torch.tensor(segment_table_indices, dtype=torch.int32),
        key_positions=torch.cat(key_positions),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=max(key_lengths),
    )


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
    raw_positions: torch.Tensor | None = None,
    true_positions: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> tuple[ContextAttentionSegment, ...]:
    """Compile an exact raw-position Drop mask into causal KV segments.

    Query offsets in the result are relative to the input Q tensor. Key
    positions are compact page-table slots. ``raw_positions`` supplies the
    immutable full-token coordinate used by Drop visibility, while
    ``true_positions`` supplies the current RoPE coordinate used by an optional
    sliding window. Keeping those axes separate is required after Reposition.
    """

    if full_token_visible_until.ndim != 1 or not full_token_visible_until.is_cpu:
        raise ValueError("full_token_visible_until must be a one-dimensional CPU tensor.")
    if query_start < 0 or query_length < 1 or key_length < 1:
        raise ValueError("Context attention lengths must be positive and query_start non-negative.")
    if sliding_window is not None and sliding_window < 0:
        raise ValueError("sliding_window must be non-negative when provided.")
    query_end = query_start + query_length
    if query_end > key_length:
        raise ValueError("Context attention query bounds exceed the compact KV stream.")

    if raw_positions is None:
        raw_positions = torch.arange(key_length, dtype=torch.int32, device="cpu")
    if raw_positions.ndim != 1 or not raw_positions.is_cpu or len(raw_positions) != key_length:
        raise ValueError("raw_positions must be a CPU vector covering the compact KV stream.")
    raw_positions = raw_positions.to(dtype=torch.int64)
    if len(raw_positions) > 1 and bool(
        torch.any(raw_positions[1:] <= raw_positions[:-1]).item()
    ):
        raise ValueError("raw_positions must be strictly increasing.")
    if len(raw_positions) == 0 or int(raw_positions[-1]) >= len(full_token_visible_until):
        raise ValueError("Context raw positions exceed full_token_visible_until.")

    table_slots = torch.arange(key_length, dtype=torch.int64, device="cpu")
    visible_until = full_token_visible_until[raw_positions].to(dtype=torch.int64)
    if bool(torch.any(visible_until <= raw_positions).item()):
        raise ValueError("A token cannot become invisible before it has been computed.")

    if sliding_window is not None:
        # A compacted active-KV view has gaps in its absolute positions, so a
        # kernel window over compact indices is not equivalent to a model
        # window. Preselect each query's exact absolute-position window and run
        # causal attention without an additional kernel-side window.
        segments = []
        if true_positions is None:
            true_positions = raw_positions
        if (
            true_positions.ndim != 1
            or not true_positions.is_cpu
            or len(true_positions) != key_length
        ):
            raise ValueError("true_positions must be a CPU vector covering the compact KV stream.")
        true_positions = true_positions.to(dtype=torch.int64)
        if len(true_positions) > 1 and bool(
            torch.any(true_positions[1:] <= true_positions[:-1]).item()
        ):
            raise ValueError("true_positions must be strictly increasing.")
        for query_slot in range(query_start, query_end):
            query_raw = raw_positions[query_slot]
            query_true = true_positions[query_slot]
            prefix_slots = table_slots[:query_slot]
            active_prefix = prefix_slots[
                (visible_until[:query_slot] > query_raw)
                & (true_positions[:query_slot] >= query_true - sliding_window)
            ]
            segments.append(
                ContextAttentionSegment(
                    query_start=query_slot - query_start,
                    query_end=query_slot - query_start + 1,
                    key_positions=torch.cat(
                        (active_prefix, table_slots[query_slot : query_slot + 1])
                    ),
                )
            )
        return tuple(segments)

    # A segment may share one causal attention call while its visible prefix is
    # stable. An expiry at raw position p starts a new segment at the first
    # compact query whose raw position is >= p. This remains correct when Drop
    # has left holes in the compact table.
    boundaries = {query_start, query_end}
    query_raw_positions = raw_positions[query_start:query_end]
    for expiry in torch.unique(visible_until[:query_end]).tolist():
        local_boundary = int(
            torch.searchsorted(query_raw_positions, int(expiry), side="left").item()
        )
        if 0 < local_boundary < query_length:
            boundaries.add(query_start + local_boundary)
    ordered = sorted(boundaries)

    segments = []
    for compact_start, compact_end in zip(ordered, ordered[1:]):
        query_raw = raw_positions[compact_start]
        prefix_slots = table_slots[:compact_start]
        active_prefix = prefix_slots[visible_until[:compact_start] > query_raw]
        local_queries = table_slots[compact_start:compact_end]
        segments.append(
            ContextAttentionSegment(
                query_start=compact_start - query_start,
                query_end=compact_end - query_start,
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
    Compact key slots and table ownership remain explicit until the GPU
    page-table compiler resolves them.
    """

    if not reqs:
        raise ValueError("Context attention batching requires at least one request.")

    segment_table_indices = []
    key_positions = []
    query_lengths = []
    key_lengths = []
    cached_tokens = []
    expected_query_offset = 0
    for req in reqs:
        if req.full_token_visible_until is None:
            raise RuntimeError("Context-mask Prefill request is missing visibility metadata.")
        segments = build_context_attention_segments(
            req.full_token_visible_until,
            query_start=req.cached_len,
            query_length=req.extend_len,
            key_length=req.device_len,
            raw_positions=(
                req.raw_positions[: req.device_len]
                if getattr(req, "raw_positions", None) is not None
                else None
            ),
            true_positions=(
                req.true_positions[: req.device_len]
                if getattr(req, "true_positions", None) is not None
                else None
            ),
            sliding_window=sliding_window,
        )
        first_segment = segments[0]
        first_query_length = first_segment.query_end - first_segment.query_start
        cached_tokens.append(len(first_segment.key_positions) - first_query_length)
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
        cached_tokens=tuple(cached_tokens),
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
    """Resolve compact Context keys into FA/FI page layouts in one GPU kernel."""

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
    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return True

    def validate_context_mask_prefill(self, device: torch.device | int | None = None) -> None:
        raise ValueError(
            f"Context-mask Prefill is not supported by {type(self).__name__}. "
            "Select a FlashInfer or FlashAttention Prefill backend, or use "
            "--contextual-prefill-mode staged."
        )

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

    def validate_context_mask_prefill(self, device: torch.device | int | None = None) -> None:
        self.prefill_backend.validate_context_mask_prefill(device)

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return self.decode_backend.can_use_cuda_graph(batch)

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
