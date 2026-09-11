from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    """Compile only the occurrence states visible to one post-match query window.

    Reposition stages before ``query_start`` are folded into one current
    occurrence per referenced token.  Intermediate occurrences are emitted only
    for stages that actually own queries in the requested window.  Birth and
    terminal maps still cover the complete raw stream so chunk-capacity and
    final page ownership remain exact.
    """

    vectors = (
        layout.birth_positions,
        layout.birth_stages,
        layout.transition_offsets,
        layout.transition_raw_tokens,
        layout.transition_old_positions,
        layout.transition_new_positions,
        full_token_visible_until,
        terminal_positions,
    )
    if any(
        tensor.device.type != "cpu" or tensor.dtype != torch.int32 or tensor.ndim != 1
        for tensor in vectors
    ):
        raise ValueError("Compact occurrence inputs must be one-dimensional CPU int32 tensors.")

    token_count = len(layout.birth_positions)
    if token_count < 1 or len(layout.birth_stages) != token_count:
        raise ValueError("Compact occurrence birth metadata must cover a nonempty prompt.")
    if len(full_token_visible_until) != token_count or len(terminal_positions) != token_count:
        raise ValueError("Occurrence visibility and terminal positions must cover the prompt.")
    if not 0 <= query_start < query_end <= token_count:
        raise ValueError("Occurrence query window is outside the raw prompt.")
    if len(layout.transition_offsets) < 2:
        raise ValueError("Paged-occurrence requires at least one effective Reposition stage.")
    if int(layout.transition_offsets[0]) != 0 or bool(
        torch.any(layout.transition_offsets[1:] < layout.transition_offsets[:-1]).item()
    ):
        raise ValueError("Occurrence transition offsets must start at zero and be monotonic.")
    transition_count = int(layout.transition_offsets[-1])
    if not (
        transition_count
        == len(layout.transition_raw_tokens)
        == len(layout.transition_old_positions)
        == len(layout.transition_new_positions)
    ):
        raise ValueError("Occurrence transition offsets do not cover the transition arrays.")
    if bool(torch.any(layout.birth_positions < 0).item()) or bool(
        torch.any(layout.transition_new_positions < 0).item()
    ):
        raise ValueError("Occurrence positions must be non-negative.")

    stage_count = len(layout.transition_offsets) - 1
    birth_stages = layout.birth_stages.to(torch.int64)
    if bool(torch.any(birth_stages < 0).item()) or bool(
        torch.any(birth_stages > stage_count).item()
    ):
        raise ValueError("Occurrence birth stages are outside the Reposition program.")
    if len(birth_stages) > 1 and bool(torch.any(birth_stages[1:] < birth_stages[:-1]).item()):
        raise ValueError("Occurrence birth stages must preserve raw-token order.")

    raw = torch.arange(token_count, dtype=torch.int64, device="cpu")
    visible_until = full_token_visible_until.to(torch.int64)
    if bool(torch.any(visible_until <= raw).item()):
        raise ValueError("A token cannot become invisible before it has been computed.")

    birth_occurrences = torch.arange(token_count, dtype=torch.int32, device="cpu")
    occurrence_raw_parts = [raw.to(torch.int32)]
    occurrence_position_parts = [layout.birth_positions]
    current_occurrences = birth_occurrences.clone()
    current_positions = layout.birth_positions.clone()
    materialized_positions = layout.birth_positions.clone()
    next_occurrence = token_count

    segment_query_starts: list[int] = []
    segment_query_ends: list[int] = []
    segment_keys: list[torch.Tensor] = []
    segment_key_offsets = [0]

    def materialize_current(raw_tokens: torch.Tensor) -> None:
        nonlocal next_occurrence
        if len(raw_tokens) == 0:
            return
        stale = materialized_positions[raw_tokens] != current_positions[raw_tokens]
        stale_raw = raw_tokens[stale]
        if len(stale_raw) == 0:
            return
        new_occurrences = torch.arange(
            next_occurrence,
            next_occurrence + len(stale_raw),
            dtype=torch.int32,
            device="cpu",
        )
        occurrence_raw_parts.append(stale_raw.to(torch.int32))
        occurrence_position_parts.append(current_positions[stale_raw].clone())
        current_occurrences[stale_raw] = new_occurrences
        materialized_positions[stale_raw] = current_positions[stale_raw]
        next_occurrence += len(stale_raw)

    covered_query = query_start
    for stage in range(stage_count + 1):
        if stage > 0:
            begin = int(layout.transition_offsets[stage - 1])
            end = int(layout.transition_offsets[stage])
            transition_raw = layout.transition_raw_tokens[begin:end].to(torch.int64)
            transition_old = layout.transition_old_positions[begin:end]
            transition_new = layout.transition_new_positions[begin:end]
            if bool(torch.any(transition_raw < 0).item()) or bool(
                torch.any(transition_raw >= token_count).item()
            ):
                raise ValueError("Reposition transition references an invalid raw token.")
            if len(torch.unique(transition_raw)) != len(transition_raw):
                raise ValueError("One Reposition stage cannot transition a raw token twice.")
            if not torch.equal(current_positions[transition_raw], transition_old):
                raise ValueError("Reposition transition old positions do not match current state.")
            current_positions[transition_raw] = transition_new

        stage_queries = torch.nonzero(birth_stages == stage, as_tuple=False).view(-1)
        if len(stage_queries) == 0:
            continue
        stage_start = int(stage_queries[0])
        stage_end = int(stage_queries[-1]) + 1
        if stage_end - stage_start != len(stage_queries):
            raise ValueError("One Reposition stage must own a contiguous raw-query interval.")
        local_query_start = max(query_start, stage_start)
        local_query_end = min(query_end, stage_end)
        if local_query_start >= local_query_end:
            continue
        if local_query_start != covered_query:
            raise RuntimeError("Occurrence stages do not cover the requested query window.")

        expiries = visible_until[:local_query_end]
        internal_expiries = expiries[(expiries > local_query_start) & (expiries < local_query_end)]
        boundaries = [local_query_start]
        if len(internal_expiries) > 0:
            boundaries.extend(int(value) for value in torch.unique(internal_expiries).tolist())
        boundaries.append(local_query_end)
        boundaries = sorted(set(boundaries))
        for local_start, local_end in zip(boundaries, boundaries[1:]):
            prefix_raw = raw[:local_start]
            active_prefix = prefix_raw[visible_until[:local_start] > local_start]
            materialize_current(active_prefix)
            keys = torch.cat(
                (
                    current_occurrences[active_prefix],
                    birth_occurrences[local_start:local_end],
                )
            ).to(torch.int32)
            if len(keys) == 0 or int(keys[-1]) != local_end - 1:
                raise RuntimeError("Occurrence segment does not end at its final query token.")
            segment_query_starts.append(local_start)
            segment_query_ends.append(local_end)
            segment_keys.append(keys)
            segment_key_offsets.append(segment_key_offsets[-1] + len(keys))
        covered_query = local_query_end

    if covered_query != query_end:
        raise RuntimeError("Occurrence stages do not cover the requested query window.")
    if not torch.equal(current_positions, terminal_positions):
        raise ValueError("Compact occurrence transitions disagree with terminal Radix positions.")

    materialize_current(raw)
    terminal_occurrences = current_occurrences.clone()
    occurrence_raw_tokens = torch.cat(occurrence_raw_parts).contiguous()
    occurrence_positions = torch.cat(occurrence_position_parts).contiguous()
    if not torch.equal(
        occurrence_raw_tokens[terminal_occurrences.to(torch.int64)], raw.to(torch.int32)
    ):
        raise RuntimeError("Terminal occurrences do not cover the raw stream in order.")

    return RepositionOccurrencePlan(
        occurrence_raw_tokens=occurrence_raw_tokens,
        occurrence_positions=occurrence_positions,
        birth_occurrences=birth_occurrences,
        terminal_occurrences=terminal_occurrences,
        segment_query_starts=torch.tensor(segment_query_starts, dtype=torch.int32),
        segment_query_ends=torch.tensor(segment_query_ends, dtype=torch.int32),
        segment_key_offsets=torch.tensor(segment_key_offsets, dtype=torch.int32),
        segment_key_occurrences=torch.cat(segment_keys).contiguous(),
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
