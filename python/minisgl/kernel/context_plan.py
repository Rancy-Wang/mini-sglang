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


def _checked_int32_offsets(key_lengths: torch.Tensor) -> torch.Tensor:
    total_keys = int(key_lengths.sum(dtype=torch.int64).item())
    if total_keys > torch.iinfo(torch.int32).max:
        raise RuntimeError("Sliding attention keys exceed int32 CSR capacity.")
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
