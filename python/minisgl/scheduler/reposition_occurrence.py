from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from minisgl.tokenizer.reposition_occurrence import RepositionOccurrencePlan

_COMPACT_OCCURRENCE_SENTINEL = -1


@dataclass(frozen=True)
class CompactRepositionOccurrenceLayout:
    """Immutable Reposition state needed to materialize occurrence attention lazily."""

    birth_positions: torch.Tensor
    birth_stages: torch.Tensor
    transition_offsets: torch.Tensor
    transition_raw_tokens: torch.Tensor
    transition_old_positions: torch.Tensor
    transition_new_positions: torch.Tensor


def pack_compact_occurrence_pending_fields(
    *,
    birth_positions: torch.Tensor | None,
    birth_stages: torch.Tensor | None,
    transition_offsets: torch.Tensor | None,
    transition_raw_tokens: torch.Tensor | None,
    transition_old_positions: torch.Tensor | None,
    transition_new_positions: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    """Pack wire layout tensors into existing scheduler-pending storage.

    ``PendingReq`` predates the compact wire layout.  Keeping this adapter at
    the tokenizer/scheduler boundary avoids retaining two copies of the same
    transition program while a request waits for Radix matching.
    """

    layout = (
        birth_positions,
        birth_stages,
        transition_offsets,
        transition_raw_tokens,
        transition_old_positions,
        transition_new_positions,
    )
    if not any(tensor is not None for tensor in layout):
        return {}
    if not all(tensor is not None for tensor in layout):
        raise ValueError("Compact occurrence layout must be provided as one complete set.")
    return {
        "occurrence_raw_tokens": birth_positions,
        "occurrence_positions": birth_stages,
        "occurrence_birth_indices": transition_offsets,
        "occurrence_terminal_indices": transition_raw_tokens,
        "occurrence_segment_query_starts": transition_old_positions,
        "occurrence_segment_query_ends": transition_new_positions,
        "occurrence_segment_key_offsets": torch.tensor(
            [_COMPACT_OCCURRENCE_SENTINEL], dtype=torch.int32, device="cpu"
        ),
        "occurrence_segment_key_indices": torch.empty(0, dtype=torch.int32, device="cpu"),
    }


def unpack_compact_occurrence_pending_fields(req: Any) -> CompactRepositionOccurrenceLayout | None:
    offsets = req.occurrence_segment_key_offsets
    if (
        offsets is None
        or offsets.ndim != 1
        or len(offsets) != 1
        or int(offsets[0]) != _COMPACT_OCCURRENCE_SENTINEL
    ):
        return None
    tensors = (
        req.occurrence_raw_tokens,
        req.occurrence_positions,
        req.occurrence_birth_indices,
        req.occurrence_terminal_indices,
        req.occurrence_segment_query_starts,
        req.occurrence_segment_query_ends,
    )
    if not all(tensor is not None for tensor in tensors):
        raise ValueError("Compact occurrence layout lost one or more transition tensors.")
    return CompactRepositionOccurrenceLayout(*tensors)


def install_occurrence_plan(req: Any, plan: RepositionOccurrencePlan) -> None:
    req.occurrence_raw_tokens = plan.occurrence_raw_tokens
    req.occurrence_positions = plan.occurrence_positions
    req.occurrence_birth_indices = plan.birth_occurrences
    req.occurrence_terminal_indices = plan.terminal_occurrences
    req.occurrence_segment_query_starts = plan.segment_query_starts
    req.occurrence_segment_query_ends = plan.segment_query_ends
    req.occurrence_segment_key_offsets = plan.segment_key_offsets
    req.occurrence_segment_key_indices = plan.segment_key_occurrences


def compile_occurrence_window(
    layout: CompactRepositionOccurrenceLayout,
    full_token_visible_until: torch.Tensor,
    terminal_positions: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
) -> RepositionOccurrencePlan:
    """Compile the same ordered plan using request-local native CPU arrays.

    All inputs are read-only views. Working arrays belong to this invocation;
    there is no mutable global scratch or trusted-validation flag. In particular
    the final materialization still covers *all* raw tokens, including dropped
    tokens needed by the final-position Radix cache.
    """
    tensors = (
        layout.birth_positions, layout.birth_stages, layout.transition_offsets,
        layout.transition_raw_tokens, layout.transition_old_positions,
        layout.transition_new_positions, full_token_visible_until, terminal_positions,
    )
    if any(t.device.type != "cpu" or t.dtype != torch.int32 or t.ndim != 1 for t in tensors):
        raise ValueError("Compact occurrence inputs must be one-dimensional CPU int32 tensors.")
    birth, stages, offsets, changed_raw, old, new, expiry, terminal = (
        t.numpy() for t in tensors
    )
    n = len(birth)
    if n < 1 or len(stages) != n:
        raise ValueError("Compact occurrence birth metadata must cover a nonempty prompt.")
    if len(expiry) != n or len(terminal) != n:
        raise ValueError("Occurrence visibility and terminal positions must cover the prompt.")
    if not 0 <= query_start < query_end <= n:
        raise ValueError("Occurrence query window is outside the raw prompt.")
    if len(offsets) < 2:
        raise ValueError("Paged-occurrence requires at least one effective Reposition stage.")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Occurrence transition offsets must start at zero and be monotonic.")
    if not offsets[-1] == len(changed_raw) == len(old) == len(new):
        raise ValueError("Occurrence transition offsets do not cover the transition arrays.")
    if np.any(birth < 0) or np.any(new < 0):
        raise ValueError("Occurrence positions must be non-negative.")
    stage_count = len(offsets) - 1
    if np.any(stages < 0) or np.any(stages > stage_count):
        raise ValueError("Occurrence birth stages are outside the Reposition program.")
    if np.any(stages[1:] < stages[:-1]):
        raise ValueError("Occurrence birth stages must preserve raw-token order.")
    raw = np.arange(n, dtype=np.int32)
    if np.any(expiry <= raw):
        raise ValueError("A token cannot become invisible before it has been computed.")
    bounds = np.searchsorted(stages, np.arange(stage_count + 2))
    current_ids = raw.copy()
    current_pos = birth.copy()
    materialized_pos = birth.copy()
    raw_parts, pos_parts = [raw], [birth]
    next_id = n
    starts, ends, keys, key_offsets = [], [], [], [0]

    def materialize(indices):
        nonlocal next_id
        stale = indices[materialized_pos[indices] != current_pos[indices]]
        count = len(stale)
        if not count:
            return
        if next_id + count > np.iinfo(np.int32).max:
            raise ValueError("Occurrence IDs exceed int32 capacity.")
        raw_parts.append(stale)
        pos_parts.append(current_pos[stale])
        current_ids[stale] = np.arange(next_id, next_id + count, dtype=np.int32)
        materialized_pos[stale] = current_pos[stale]
        next_id += count

    covered = query_start
    for stage in range(stage_count + 1):
        if stage:
            begin, end = int(offsets[stage - 1]), int(offsets[stage])
            ids = changed_raw[begin:end]
            if np.any(ids < 0) or np.any(ids >= n):
                raise ValueError("Reposition transition references an invalid raw token.")
            if np.any(ids[1:] <= ids[:-1]) and len(np.unique(ids)) != len(ids):
                raise ValueError("One Reposition stage cannot transition a raw token twice.")
            if not np.array_equal(current_pos[ids], old[begin:end]):
                raise ValueError("Reposition transition old positions do not match current state.")
            current_pos[ids] = new[begin:end]
        local_start = max(query_start, int(bounds[stage]))
        local_end = min(query_end, int(bounds[stage + 1]))
        if local_start >= local_end:
            continue
        if local_start != covered:
            raise RuntimeError("Occurrence stages do not cover the requested query window.")
        values = expiry[:local_end]
        cuts = [local_start, *np.unique(values[(values > local_start) & (values < local_end)]),
                local_end]
        for start, end in zip(cuts, cuts[1:]):
            active = raw[:start][expiry[:start] > start]
            materialize(active)
            selected = np.concatenate((current_ids[active], raw[start:end]))
            if not len(selected) or selected[-1] != end - 1:
                raise RuntimeError("Occurrence segment does not end at its final query token.")
            starts.append(start)
            ends.append(end)
            keys.append(selected)
            key_offsets.append(key_offsets[-1] + len(selected))
        covered = local_end
    if covered != query_end:
        raise RuntimeError("Occurrence stages do not cover the requested query window.")
    if not np.array_equal(current_pos, terminal):
        raise ValueError("Compact occurrence transitions disagree with terminal Radix positions.")
    materialize(raw)
    if key_offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError("Occurrence segment offsets exceed int32 capacity.")
    all_raw = np.concatenate(raw_parts)
    if not np.array_equal(all_raw[current_ids], raw):
        raise RuntimeError("Terminal occurrences do not cover the raw stream in order.")
    return RepositionOccurrencePlan(
        occurrence_raw_tokens=torch.from_numpy(all_raw),
        occurrence_positions=torch.from_numpy(np.concatenate(pos_parts)),
        birth_occurrences=torch.from_numpy(raw),
        terminal_occurrences=torch.from_numpy(current_ids),
        segment_query_starts=torch.from_numpy(np.asarray(starts, dtype=np.int32)),
        segment_query_ends=torch.from_numpy(np.asarray(ends, dtype=np.int32)),
        segment_key_offsets=torch.from_numpy(np.asarray(key_offsets, dtype=np.int32)),
        segment_key_occurrences=torch.from_numpy(np.concatenate(keys)),
    )


def prewarm_occurrence_window_compiler() -> None:
    """Exercise full- and partial-prefix compiler branches before serving."""

    layout = CompactRepositionOccurrenceLayout(
        birth_positions=torch.tensor([0, 1, 1], dtype=torch.int32),
        birth_stages=torch.tensor([0, 0, 1], dtype=torch.int32),
        transition_offsets=torch.tensor([0, 1], dtype=torch.int32),
        transition_raw_tokens=torch.tensor([1], dtype=torch.int32),
        transition_old_positions=torch.tensor([1], dtype=torch.int32),
        transition_new_positions=torch.tensor([0], dtype=torch.int32),
    )
    visibility = torch.full((3,), 4, dtype=torch.int32)
    terminal_positions = torch.tensor([0, 0, 1], dtype=torch.int32)
    compile_occurrence_window(
        layout,
        visibility,
        terminal_positions,
        query_start=0,
        query_end=3,
    )
    compile_occurrence_window(
        layout,
        visibility,
        terminal_positions,
        query_start=2,
        query_end=3,
    )


__all__ = [
    "CompactRepositionOccurrenceLayout",
    "compile_occurrence_window",
    "install_occurrence_plan",
    "pack_compact_occurrence_pending_fields",
    "prewarm_occurrence_window_compiler",
    "unpack_compact_occurrence_pending_fields",
]
