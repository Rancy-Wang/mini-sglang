from __future__ import annotations

import functools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import load_aot

TOKEN_KIND = 0
DELTA_KIND = 1
REPOSITION_KIND = 2


@dataclass(frozen=True)
class RadixRepositionLayout:
    drop_insert_offsets: torch.Tensor
    drop_range_offsets: torch.Tensor
    drop_ranges: torch.Tensor
    records: torch.Tensor
    virtual_mask: torch.Tensor
    key_to_token: torch.Tensor
    token_to_key: torch.Tensor
    positions: torch.Tensor
    repos_info: torch.Tensor
    keep_mask: torch.Tensor
    materialized_stage: torch.Tensor
    birth_positions: torch.Tensor
    birth_stages: torch.Tensor
    transition_offsets: torch.Tensor
    transition_raw_tokens: torch.Tensor
    transition_old_positions: torch.Tensor
    transition_new_positions: torch.Tensor
    effective_reposition_stages: torch.Tensor
    drop_event_to_key: torch.Tensor
    effective_repositions: torch.Tensor
    ignored_repositions: torch.Tensor
    next_position: int
    current_reposition: int
    compile_ns: int

    @property
    def keys(self) -> torch.Tensor:
        return self.records


@dataclass(frozen=True)
class RadixRepositionInput:
    """One request for the bounded CPU Radix compiler batch entry point."""

    token_ids: torch.Tensor
    drop_insert_offsets: torch.Tensor
    drop_range_offsets: torch.Tensor
    drop_ranges: torch.Tensor
    reposition_raw_boundaries: torch.Tensor
    reposition_insert_offsets: torch.Tensor


@functools.cache
def _load_module():
    return load_aot(
        "radix_reposition",
        cpp_files=["radix_reposition.cpp"],
    )


def compile_radix_reposition_layout(
    token_ids: torch.Tensor,
    drop_insert_offsets: torch.Tensor,
    drop_range_offsets: torch.Tensor,
    drop_ranges: torch.Tensor,
    reposition_raw_boundaries: torch.Tensor,
    reposition_insert_offsets: torch.Tensor,
) -> RadixRepositionLayout:
    """Compile token/Drop/Reposition events into a CPU structured Radix key."""

    if token_ids.device.type != "cpu" or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Reposition Radix token IDs must be a CPU int32/int64 tensor.")
    token_ids = token_ids.contiguous()
    drop_insert_offsets = drop_insert_offsets.to(device="cpu", dtype=torch.int32).contiguous()
    drop_range_offsets = drop_range_offsets.to(device="cpu", dtype=torch.int32).contiguous()
    drop_ranges = drop_ranges.to(device="cpu", dtype=torch.int32).contiguous()
    reposition_raw_boundaries = reposition_raw_boundaries.to(
        device="cpu", dtype=torch.int32
    ).contiguous()
    reposition_insert_offsets = reposition_insert_offsets.to(
        device="cpu", dtype=torch.int32
    ).contiguous()

    compile_started_ns = time.perf_counter_ns()
    token_count = len(token_ids)
    reposition_count = len(reposition_raw_boundaries)
    transition_counts = torch.zeros(reposition_count, dtype=torch.int32, device="cpu")
    count_status = torch.zeros(2, dtype=torch.int64, device="cpu")
    _load_module().count_radix_reposition_transitions(
        token_count,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        reposition_raw_boundaries,
        reposition_insert_offsets,
        transition_counts,
        count_status,
    )
    if int(count_status[0]) == 1:
        boundary = int(count_status[1])
        raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
    transition_count = int(transition_counts.sum().item())
    range_count = len(drop_ranges) // 2
    capacity = token_count + range_count + reposition_count
    records = torch.empty((capacity, 4), dtype=torch.int32, device="cpu")
    virtual_mask = torch.empty(capacity, dtype=torch.bool, device="cpu")
    key_to_token = torch.empty(capacity, dtype=torch.int64, device="cpu")
    token_to_key = torch.empty(token_count, dtype=torch.int64, device="cpu")
    positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    repos_info = torch.empty(token_count, dtype=torch.int32, device="cpu")
    keep_mask = torch.empty(token_count, dtype=torch.bool, device="cpu")
    materialized_stage = torch.empty(token_count, dtype=torch.int32, device="cpu")
    birth_positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    birth_stages = torch.empty(token_count, dtype=torch.int32, device="cpu")
    transition_offsets = torch.empty(reposition_count + 1, dtype=torch.int32, device="cpu")
    transition_raw_tokens = torch.empty(transition_count, dtype=torch.int32, device="cpu")
    transition_old_positions = torch.empty(transition_count, dtype=torch.int32, device="cpu")
    transition_new_positions = torch.empty(transition_count, dtype=torch.int32, device="cpu")
    effective_reposition_stages = torch.full(
        (reposition_count,), -1, dtype=torch.int32, device="cpu"
    )
    drop_event_to_key = torch.full(
        (len(drop_insert_offsets),), -1, dtype=torch.int64, device="cpu"
    )
    effective = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    ignored = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    status = torch.zeros(6, dtype=torch.int64, device="cpu")

    _load_module().compile_radix_reposition_layout(
        token_ids,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        reposition_raw_boundaries,
        reposition_insert_offsets,
        records,
        virtual_mask,
        key_to_token,
        token_to_key,
        positions,
        repos_info,
        keep_mask,
        materialized_stage,
        birth_positions,
        birth_stages,
        transition_offsets,
        transition_raw_tokens,
        transition_old_positions,
        transition_new_positions,
        effective_reposition_stages,
        drop_event_to_key,
        effective,
        ignored,
        status,
    )

    if int(status[0]) == 1:
        boundary = int(status[4])
        raise ValueError(f"Reposition at raw boundary {boundary} has no active tokens.")
    if int(status[0]) == 2:
        token_id = int(status[4])
        raise ValueError(
            f"Reposition Radix token ID {token_id} is outside the non-negative int32 range."
        )

    key_len = int(status[1])
    effective_stage_count = int(status[2])
    return RadixRepositionLayout(
        drop_insert_offsets=drop_insert_offsets,
        drop_range_offsets=drop_range_offsets,
        drop_ranges=drop_ranges,
        records=records[:key_len],
        virtual_mask=virtual_mask[:key_len],
        key_to_token=key_to_token[:key_len],
        token_to_key=token_to_key,
        positions=positions,
        repos_info=repos_info,
        keep_mask=keep_mask,
        materialized_stage=materialized_stage,
        birth_positions=birth_positions,
        birth_stages=birth_stages,
        transition_offsets=transition_offsets[: effective_stage_count + 1],
        transition_raw_tokens=transition_raw_tokens,
        transition_old_positions=transition_old_positions,
        transition_new_positions=transition_new_positions,
        effective_reposition_stages=effective_reposition_stages,
        drop_event_to_key=drop_event_to_key,
        effective_repositions=effective,
        ignored_repositions=ignored,
        next_position=int(status[3]),
        current_reposition=int(status[5]),
        compile_ns=time.perf_counter_ns() - compile_started_ns,
    )


def validate_radix_reposition_records(
    records: torch.Tensor,
    *,
    token_count: int | None = None,
    require_materialized: bool = False,
) -> None:
    """Batch-validate direct structured records without Python row iteration."""

    if (
        records.device.type != "cpu"
        or records.dtype != torch.int32
        or records.ndim != 2
        or records.shape[1] != 4
    ):
        raise ValueError("Structured Radix records must be CPU int32 [N, 4].")
    if token_count is not None and token_count < 0:
        raise ValueError("Structured Radix token_count must be non-negative.")

    kinds = records[:, 0]
    is_token = kinds == TOKEN_KIND
    is_delta = kinds == DELTA_KIND
    is_reposition = kinds == REPOSITION_KIND
    known = is_token | is_delta | is_reposition

    def first_bad(mask: torch.Tensor) -> int:
        return int(torch.nonzero(mask, as_tuple=False)[0].item())

    if bool(torch.any(~known)):
        row = first_bad(~known)
        raise ValueError(f"Unknown structured Radix record kind at row {row}: {int(kinds[row])}")

    invalid_token = is_token & (records[:, 1] < 0)
    if bool(torch.any(invalid_token)):
        raise ValueError("Token records require non-negative token IDs.")

    field_one = records[:, 1].to(dtype=torch.int64)
    field_two = records[:, 2].to(dtype=torch.int64)
    delta_start = -field_one - 1
    delta_end = -field_two - 1
    invalid_delta = is_delta & (
        (records[:, 1] >= 0)
        | (records[:, 2] >= 0)
        | (records[:, 3] != -1)
        | (delta_end <= delta_start)
    )
    if bool(torch.any(invalid_delta)):
        row = first_bad(invalid_delta)
        raise ValueError(f"Invalid direct Delta range record at row {row}.")
    if token_count is not None:
        outside = is_delta & (delta_end > token_count)
        if bool(torch.any(outside)):
            row = first_bad(outside)
            raise ValueError(
                f"Delta range [{int(delta_start[row])}, {int(delta_end[row])}) "
                f"exceeds token length {token_count}."
            )

    if len(records) > 1:
        adjacent_delta = is_delta[:-1] & is_delta[1:]
        noncanonical = adjacent_delta & (delta_start[1:] <= delta_end[:-1])
        if bool(torch.any(noncanonical)):
            raise ValueError("Consecutive Delta ranges must be canonical and disjoint.")

    invalid_reposition = is_reposition & (
        (records[:, 1] < 0) | (records[:, 2] != -1) | (records[:, 3] != -1)
    )
    if bool(torch.any(invalid_reposition)):
        row = first_bad(invalid_reposition)
        raise ValueError(f"Invalid Reposition record at row {row}.")

    if require_materialized:
        token_flags = is_token.to(torch.int64)
        materialized_before = torch.cumsum(token_flags, dim=0) - token_flags
        premature_delta = is_delta & (delta_end > materialized_before)
        if bool(torch.any(premature_delta)):
            row = first_bad(premature_delta)
            raise ValueError(
                f"Delta range [{int(delta_start[row])}, {int(delta_end[row])}) precedes "
                f"its materialization boundary {int(materialized_before[row])}."
            )
        misplaced_reposition = is_reposition & (
            field_one + 1 != materialized_before
        )
        if bool(torch.any(misplaced_reposition)):
            raise ValueError(
                "Reposition raw boundary does not match its record insertion point."
            )


def compile_radix_reposition_layout_batch(
    requests: Sequence[RadixRepositionInput],
    *,
    max_workers: int = 4,
) -> tuple[RadixRepositionLayout, ...]:
    """Compile independent requests concurrently with bounded CPU workers."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if not requests:
        return ()

    _load_module()

    def compile_one(request: RadixRepositionInput) -> RadixRepositionLayout:
        return compile_radix_reposition_layout(
            request.token_ids,
            request.drop_insert_offsets,
            request.drop_range_offsets,
            request.drop_ranges,
            request.reposition_raw_boundaries,
            request.reposition_insert_offsets,
        )

    worker_count = min(len(requests), max_workers)
    if worker_count == 1:
        return tuple(compile_one(request) for request in requests)
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="radix-reposition",
    ) as executor:
        return tuple(executor.map(compile_one, requests))


__all__ = [
    "DELTA_KIND",
    "REPOSITION_KIND",
    "TOKEN_KIND",
    "RadixRepositionInput",
    "RadixRepositionLayout",
    "compile_radix_reposition_layout",
    "compile_radix_reposition_layout_batch",
    "validate_radix_reposition_records",
]
