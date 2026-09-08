from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.kernel.radix_reposition import RadixRepositionLayout


@dataclass(frozen=True)
class RepositionOccurrencePlan:
    """One-shot attention/KV plan for an effective Reposition request.

    Every ``(raw token, RoPE position)`` pair owns a distinct occurrence.  The
    scheduler later assigns one page from the ordinary KV page allocator to
    every occurrence; K and V always share that page index.
    """

    occurrence_raw_tokens: torch.Tensor
    occurrence_positions: torch.Tensor
    birth_occurrences: torch.Tensor
    terminal_occurrences: torch.Tensor
    segment_query_starts: torch.Tensor
    segment_query_ends: torch.Tensor
    segment_key_offsets: torch.Tensor
    segment_key_occurrences: torch.Tensor

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrence_raw_tokens)

    @property
    def segment_count(self) -> int:
        return len(self.segment_query_starts)


def compile_reposition_occurrence_plan(
    layout: RadixRepositionLayout,
    full_token_visible_until: torch.Tensor | None,
) -> RepositionOccurrencePlan:
    """Expand a staged layout into one request-major ragged attention plan.

    The expansion is performed once in the tokenizer process, before Scheduler
    metadata preparation.  Segment keys are occurrence IDs rather than page
    IDs, so this result is device-independent and safe to serialize over IPC.
    """

    token_count = len(layout.birth_positions)
    if token_count < 1:
        raise ValueError("Reposition occurrence expansion requires at least one token.")
    cpu_vectors = (
        layout.birth_positions,
        layout.birth_stages,
        layout.transition_offsets,
        layout.transition_raw_tokens,
        layout.transition_old_positions,
        layout.transition_new_positions,
    )
    if any(tensor.device.type != "cpu" or tensor.ndim != 1 for tensor in cpu_vectors):
        raise ValueError("Reposition occurrence inputs must be one-dimensional CPU tensors.")
    if len(layout.transition_offsets) <= 1:
        raise ValueError("Occurrence expansion requires at least one effective Reposition stage.")
    if not (
        len(layout.transition_raw_tokens)
        == len(layout.transition_old_positions)
        == len(layout.transition_new_positions)
    ):
        raise ValueError("Reposition transition arrays have different lengths.")
    if int(layout.transition_offsets[-1]) != len(layout.transition_raw_tokens):
        raise ValueError("Reposition transition offsets do not cover all transitions.")
    if int(layout.transition_offsets[0]) != 0 or bool(
        torch.any(layout.transition_offsets[1:] < layout.transition_offsets[:-1]).item()
    ):
        raise ValueError("Reposition transition offsets must start at zero and be monotonic.")
    if len(layout.birth_stages) != token_count:
        raise ValueError("Reposition birth stages must cover the raw token stream.")
    if bool(torch.any(layout.birth_positions < 0).item()) or bool(
        torch.any(layout.transition_new_positions < 0).item()
    ):
        raise ValueError("Reposition occurrence positions must be non-negative.")

    if full_token_visible_until is None:
        visible_until = torch.full((token_count,), token_count + 1, dtype=torch.int64, device="cpu")
    else:
        if (
            full_token_visible_until.device.type != "cpu"
            or full_token_visible_until.ndim != 1
            or len(full_token_visible_until) != token_count
        ):
            raise ValueError(
                "full_token_visible_until must be a CPU vector covering the raw token stream."
            )
        visible_until = full_token_visible_until.to(dtype=torch.int64)
    raw = torch.arange(token_count, dtype=torch.int64, device="cpu")
    if bool(torch.any(visible_until <= raw).item()):
        raise ValueError("A token cannot become invisible before it has been computed.")

    # Birth occurrences occupy [0, token_count).  Each transition gets a new
    # occurrence even when another stage references the same raw token.
    occurrence_raw_tokens = [raw.to(torch.int32)]
    occurrence_positions = [layout.birth_positions.to(torch.int32)]
    birth_occurrences = torch.arange(token_count, dtype=torch.int32, device="cpu")
    current_occurrences = birth_occurrences.clone()
    current_positions = layout.birth_positions.to(torch.int32).clone()

    segment_query_starts: list[int] = []
    segment_query_ends: list[int] = []
    segment_keys: list[torch.Tensor] = []
    key_offsets = [0]
    stage_count = len(layout.transition_offsets) - 1
    birth_stages = layout.birth_stages.to(dtype=torch.int64)
    if bool(torch.any(birth_stages < 0).item()) or int(torch.max(birth_stages)) > stage_count:
        raise ValueError("Token birth stages are outside the effective Reposition stages.")
    if len(birth_stages) > 1 and bool(torch.any(birth_stages[1:] < birth_stages[:-1]).item()):
        raise ValueError("Token birth stages must preserve raw-token order.")

    for stage in range(stage_count + 1):
        if stage > 0:
            begin = int(layout.transition_offsets[stage - 1])
            end = int(layout.transition_offsets[stage])
            transition_raw = layout.transition_raw_tokens[begin:end].to(torch.int64)
            transition_positions = layout.transition_new_positions[begin:end].to(torch.int32)
            transition_old_positions = layout.transition_old_positions[begin:end].to(torch.int32)
            if bool(torch.any(transition_raw < 0).item()) or bool(
                torch.any(transition_raw >= token_count).item()
            ):
                raise ValueError("Reposition transition references an invalid raw token.")
            if len(torch.unique(transition_raw)) != len(transition_raw):
                raise ValueError("One Reposition stage cannot transition a raw token twice.")
            if not torch.equal(current_positions[transition_raw], transition_old_positions):
                raise ValueError("Reposition transition old positions do not match current state.")
            first_occurrence = sum(len(part) for part in occurrence_raw_tokens)
            new_occurrences = torch.arange(
                first_occurrence,
                first_occurrence + len(transition_raw),
                dtype=torch.int32,
                device="cpu",
            )
            occurrence_raw_tokens.append(transition_raw.to(torch.int32))
            occurrence_positions.append(transition_positions)
            current_occurrences[transition_raw] = new_occurrences
            current_positions[transition_raw] = transition_positions

        stage_queries = torch.nonzero(birth_stages == stage, as_tuple=False).view(-1)
        if len(stage_queries) == 0:
            continue
        query_start = int(stage_queries[0])
        query_end = int(stage_queries[-1]) + 1
        if query_end - query_start != len(stage_queries):
            raise ValueError("One Reposition stage must own a contiguous raw-query interval.")

        boundaries = {query_start, query_end}
        for expiry in torch.unique(visible_until[:query_end]).tolist():
            expiry = int(expiry)
            if query_start < expiry < query_end:
                boundaries.add(expiry)
        ordered = sorted(boundaries)
        for local_start, local_end in zip(ordered, ordered[1:]):
            prefix_raw = raw[:local_start]
            active_prefix = prefix_raw[visible_until[:local_start] > local_start]
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
            key_offsets.append(key_offsets[-1] + len(keys))

    terminal_occurrences = current_occurrences
    flat_raw = torch.cat(occurrence_raw_tokens).contiguous()
    flat_positions = torch.cat(occurrence_positions).contiguous()
    if len(flat_raw) != token_count + len(layout.transition_raw_tokens):
        raise RuntimeError("Occurrence count diverged from births plus transitions.")
    if not torch.equal(flat_raw[terminal_occurrences.to(torch.int64)], raw.to(torch.int32)):
        raise RuntimeError("Terminal occurrences do not cover the raw stream in order.")
    return RepositionOccurrencePlan(
        occurrence_raw_tokens=flat_raw,
        occurrence_positions=flat_positions,
        birth_occurrences=birth_occurrences,
        terminal_occurrences=terminal_occurrences,
        segment_query_starts=torch.tensor(segment_query_starts, dtype=torch.int32),
        segment_query_ends=torch.tensor(segment_query_ends, dtype=torch.int32),
        segment_key_offsets=torch.tensor(key_offsets, dtype=torch.int32),
        segment_key_occurrences=torch.cat(segment_keys).contiguous(),
    )


__all__ = ["RepositionOccurrencePlan", "compile_reposition_occurrence_plan"]
