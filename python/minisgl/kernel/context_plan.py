from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


_CONTEXT_PLAN_MODULE: Module | None = None
_CONTEXT_PLAN_LOAD_ERROR: Exception | None = None


def _is_cpu_int32_vector(tensor: torch.Tensor) -> bool:
    return (
        tensor.device.type == "cpu"
        and tensor.dtype == torch.int32
        and tensor.ndim == 1
        and tensor.is_contiguous()
    )


def _is_cpu_bool_vector(tensor: torch.Tensor) -> bool:
    return (
        tensor.device.type == "cpu"
        and tensor.dtype == torch.bool
        and tensor.ndim == 1
        and tensor.is_contiguous()
    )


def _checked_int32_offsets(
    key_lengths: torch.Tensor,
    *,
    description: str = "Sliding attention keys",
) -> torch.Tensor:
    total_keys = int(key_lengths.sum(dtype=torch.int64).item())
    if total_keys > torch.iinfo(torch.int32).max:
        raise RuntimeError(f"{description} exceed int32 CSR capacity.")
    key_offsets = torch.empty(len(key_lengths) + 1, dtype=torch.int32, device="cpu")
    key_offsets[0] = 0
    torch.cumsum(key_lengths, dim=0, out=key_offsets[1:])
    return key_offsets


def _load_context_plan_module() -> Module:
    global _CONTEXT_PLAN_LOAD_ERROR, _CONTEXT_PLAN_MODULE
    if _CONTEXT_PLAN_MODULE is not None:
        return _CONTEXT_PLAN_MODULE
    if _CONTEXT_PLAN_LOAD_ERROR is not None:
        raise RuntimeError(
            "The Context planner kernel is unavailable."
        ) from _CONTEXT_PLAN_LOAD_ERROR
    try:
        _CONTEXT_PLAN_MODULE = load_aot("context_plan", cpp_files=["context_plan.cpp"])
    except Exception as exc:
        _CONTEXT_PLAN_LOAD_ERROR = exc
        raise
    return _CONTEXT_PLAN_MODULE


def preload_context_plan_kernel() -> None:
    _load_context_plan_module()


def prewarm_context_plan_variants() -> None:
    """Execute representative full, sliding, and occurrence planner calls."""

    preload_context_plan_kernel()
    never = torch.iinfo(torch.int32).max
    visible_until = torch.full((2,), never, dtype=torch.int32, device="cpu")
    raw_positions = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
    true_positions = raw_positions.clone()
    full = try_build_context_full_plan(
        visible_until,
        raw_positions,
        query_start=1,
        query_length=1,
    )
    sliding = try_build_context_sliding_plan(
        visible_until,
        raw_positions,
        true_positions,
        query_start=1,
        query_length=1,
        sliding_window=1,
    )
    occurrence = try_build_occurrence_sliding_plan(
        raw_positions,
        true_positions,
        torch.tensor([1], dtype=torch.int32, device="cpu"),
        torch.tensor([2], dtype=torch.int32, device="cpu"),
        torch.tensor([0, 2], dtype=torch.int32, device="cpu"),
        raw_positions,
        true_positions,
        cached_len=1,
        device_len=2,
        initial_cached_len=1,
        sliding_window=1,
        occurrence_base=0,
    )
    capacity = try_build_occurrence_capacity_index(
        raw_positions,
        true_positions,
        raw_positions,
        raw_positions,
        torch.tensor([1], dtype=torch.int32, device="cpu"),
        torch.tensor([2], dtype=torch.int32, device="cpu"),
        torch.tensor([0, 2], dtype=torch.int32, device="cpu"),
        raw_positions,
        torch.zeros(2, dtype=torch.bool, device="cpu"),
        torch.ones(2, dtype=torch.bool, device="cpu"),
        torch.tensor([0], dtype=torch.int32, device="cpu"),
        chunk_start=1,
        max_chunk_end=2,
        output_len=1,
    )
    if full is None or sliding is None or occurrence is None or capacity is None:
        raise RuntimeError("Context planner warmup did not execute every serving variant.")


def try_build_context_full_plan(
    full_token_visible_until: torch.Tensor,
    raw_positions: torch.Tensor,
    *,
    query_start: int,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build one request's full-attention Context CSR with two AOT passes."""

    try:
        module = _load_context_plan_module()
    except Exception:
        return None

    if not all(
        _is_cpu_int32_vector(tensor) for tensor in (full_token_visible_until, raw_positions)
    ):
        return None

    query_count = int(query_length)
    query_lengths_capacity = torch.empty(query_count, dtype=torch.int32, device="cpu")
    key_lengths_capacity = torch.empty(query_count, dtype=torch.int32, device="cpu")
    status = torch.zeros(1, dtype=torch.int64, device="cpu")
    module.count_context_full_keys(
        full_token_visible_until,
        raw_positions,
        int(query_start),
        query_count,
        query_lengths_capacity,
        key_lengths_capacity,
        status,
    )
    segment_count = int(status[0].item())
    if segment_count < 1 or segment_count > query_count:
        raise RuntimeError("The Context full-attention planner returned an invalid segment count.")
    query_lengths = query_lengths_capacity[:segment_count]
    key_lengths = key_lengths_capacity[:segment_count]
    key_offsets = _checked_int32_offsets(
        key_lengths,
        description="Full attention keys",
    )
    key_positions = torch.empty(int(key_offsets[-1].item()), dtype=torch.int32, device="cpu")
    module.fill_context_full_keys(
        full_token_visible_until,
        raw_positions,
        int(query_start),
        query_count,
        query_lengths,
        key_offsets,
        key_positions,
    )
    return query_lengths, key_offsets, key_positions


def try_build_context_sliding_plan(
    full_token_visible_until: torch.Tensor,
    raw_positions: torch.Tensor,
    true_positions: torch.Tensor,
    *,
    query_start: int,
    query_length: int,
    sliding_window: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build one request's sliding Context CSR keys, or None without the AOT module."""

    try:
        module = _load_context_plan_module()
    except Exception:
        return None

    if not all(
        _is_cpu_int32_vector(tensor)
        for tensor in (full_token_visible_until, raw_positions, true_positions)
    ):
        return None

    query_count = int(query_length)
    key_lengths = torch.empty(query_count, dtype=torch.int32, device="cpu")
    module.count_context_sliding_keys(
        full_token_visible_until,
        raw_positions,
        true_positions,
        int(query_start),
        query_count,
        int(sliding_window),
        key_lengths,
    )
    key_offsets = _checked_int32_offsets(key_lengths)
    key_positions = torch.empty(int(key_offsets[-1].item()), dtype=torch.int32, device="cpu")
    module.fill_context_sliding_keys(
        full_token_visible_until,
        raw_positions,
        true_positions,
        int(query_start),
        query_count,
        int(sliding_window),
        key_offsets,
        key_positions,
    )
    return key_offsets, key_positions


def try_build_occurrence_sliding_plan(
    occurrence_raw_tokens: torch.Tensor,
    occurrence_positions: torch.Tensor,
    segment_query_starts: torch.Tensor,
    segment_query_ends: torch.Tensor,
    segment_key_offsets: torch.Tensor,
    segment_key_occurrences: torch.Tensor,
    true_positions: torch.Tensor,
    *,
    cached_len: int,
    device_len: int,
    initial_cached_len: int,
    sliding_window: int,
    occurrence_base: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build one request's occurrence sliding CSR keys when segment positions are ordered."""

    try:
        module = _load_context_plan_module()
    except Exception:
        return None

    if not all(
        _is_cpu_int32_vector(tensor)
        for tensor in (
            occurrence_raw_tokens,
            occurrence_positions,
            segment_query_starts,
            segment_query_ends,
            segment_key_offsets,
            segment_key_occurrences,
            true_positions,
        )
    ):
        return None

    query_count = int(device_len) - int(cached_len)
    key_lengths = torch.empty(query_count, dtype=torch.int32, device="cpu")
    cached_mask = torch.zeros(int(initial_cached_len), dtype=torch.bool, device="cpu")
    status = torch.zeros(1, dtype=torch.int64, device="cpu")
    module.count_occurrence_sliding_keys(
        occurrence_raw_tokens,
        occurrence_positions,
        segment_query_starts,
        segment_query_ends,
        segment_key_offsets,
        segment_key_occurrences,
        true_positions,
        int(cached_len),
        int(device_len),
        int(initial_cached_len),
        int(sliding_window),
        key_lengths,
        cached_mask,
        status,
    )
    if int(status[0]) != 0:
        return None

    key_offsets = _checked_int32_offsets(key_lengths)
    key_positions = torch.empty(int(key_offsets[-1].item()), dtype=torch.int32, device="cpu")
    module.fill_occurrence_sliding_keys(
        occurrence_positions,
        segment_query_starts,
        segment_query_ends,
        segment_key_offsets,
        segment_key_occurrences,
        true_positions,
        int(cached_len),
        int(device_len),
        int(sliding_window),
        int(occurrence_base),
        key_offsets,
        key_positions,
    )
    cached_positions = torch.nonzero(cached_mask, as_tuple=False).view(-1).to(torch.int64)
    return key_offsets, key_positions, cached_positions


def try_build_occurrence_capacity_index(
    occurrence_raw_tokens: torch.Tensor,
    occurrence_positions: torch.Tensor,
    birth_occurrences: torch.Tensor,
    terminal_occurrences: torch.Tensor,
    segment_query_starts: torch.Tensor,
    segment_query_ends: torch.Tensor,
    segment_key_offsets: torch.Tensor,
    segment_key_occurrences: torch.Tensor,
    terminal_owned: torch.Tensor,
    final_keep: torch.Tensor,
    initial_source_positions: torch.Tensor | None = None,
    *,
    chunk_start: int,
    max_chunk_end: int,
    output_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Precompute exact occurrence capacity for every candidate chunk endpoint."""

    try:
        module = _load_context_plan_module()
    except Exception:
        return None

    int32_inputs = (
        occurrence_raw_tokens,
        occurrence_positions,
        birth_occurrences,
        terminal_occurrences,
        segment_query_starts,
        segment_query_ends,
        segment_key_offsets,
        segment_key_occurrences,
    )
    if not all(_is_cpu_int32_vector(tensor) for tensor in int32_inputs):
        return None
    if not _is_cpu_bool_vector(terminal_owned) or not _is_cpu_bool_vector(final_keep):
        return None
    if initial_source_positions is None:
        initial_source_positions = torch.empty(0, dtype=torch.int32, device="cpu")
    if not _is_cpu_int32_vector(initial_source_positions):
        return None

    endpoint_count = int(max_chunk_end) - int(chunk_start)
    if endpoint_count <= 0:
        raise ValueError("Occurrence capacity index requires at least one endpoint.")
    first_required_end = torch.empty(len(occurrence_raw_tokens), dtype=torch.int32, device="cpu")
    current_allocations = torch.empty(endpoint_count, dtype=torch.int64, device="cpu")
    persistent_allocations = torch.empty(endpoint_count, dtype=torch.int64, device="cpu")
    future_reserve = torch.empty(endpoint_count, dtype=torch.int64, device="cpu")
    module.build_occurrence_capacity_index(
        *int32_inputs,
        terminal_owned,
        final_keep,
        initial_source_positions,
        int(chunk_start),
        int(max_chunk_end),
        int(output_len),
        first_required_end,
        current_allocations,
        persistent_allocations,
        future_reserve,
    )
    return (
        first_required_end,
        current_allocations,
        persistent_allocations,
        future_reserve,
    )


def first_mask_free_conflict_event(
    active_positions: torch.Tensor,
    event_positions: torch.Tensor,
    range_offsets: torch.Tensor,
    position_ranges: torch.Tensor,
    *,
    active_cached_len: int,
    effective_event_count: int,
) -> int | None:
    """Return the first conflicting effective event, or None when compact Extend is exact."""

    result = int(
        _load_context_plan_module().first_mask_free_conflict_event(
            active_positions,
            event_positions,
            range_offsets,
            position_ranges,
            int(active_cached_len),
            int(effective_event_count),
        )
    )
    return None if result < 0 else result
