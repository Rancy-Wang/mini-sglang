from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


_CONTEXT_PLAN_MODULE: Module | None = None
_CONTEXT_PLAN_LOAD_ERROR: Exception | None = None


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
