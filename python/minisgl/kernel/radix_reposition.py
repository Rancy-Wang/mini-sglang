from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from .utils import load_aot


TOKEN_KIND = 0
DELTA_KIND = 1
REPOSITION_KIND = 2


@dataclass(frozen=True)
class RadixRepositionLayout:
    records: torch.Tensor
    virtual_mask: torch.Tensor
    key_to_token: torch.Tensor
    token_to_key: torch.Tensor
    positions: torch.Tensor
    repos_info: torch.Tensor
    keep_mask: torch.Tensor
    materialized_stage: torch.Tensor
    effective_repositions: torch.Tensor
    ignored_repositions: torch.Tensor
    next_position: int
    current_reposition: int

    @property
    def keys(self) -> torch.Tensor:
        return self.records


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
    delta_marker_ids: torch.Tensor,
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
    delta_marker_ids = delta_marker_ids.to(device="cpu", dtype=torch.int32).contiguous()
    reposition_raw_boundaries = reposition_raw_boundaries.to(
        device="cpu", dtype=torch.int32
    ).contiguous()
    reposition_insert_offsets = reposition_insert_offsets.to(
        device="cpu", dtype=torch.int32
    ).contiguous()

    token_count = len(token_ids)
    reposition_count = len(reposition_raw_boundaries)
    capacity = token_count + len(drop_insert_offsets) + reposition_count
    records = torch.empty((capacity, 4), dtype=torch.int32, device="cpu")
    virtual_mask = torch.empty(capacity, dtype=torch.bool, device="cpu")
    key_to_token = torch.empty(capacity, dtype=torch.int64, device="cpu")
    token_to_key = torch.empty(token_count, dtype=torch.int64, device="cpu")
    positions = torch.empty(token_count, dtype=torch.int32, device="cpu")
    repos_info = torch.empty(token_count, dtype=torch.int32, device="cpu")
    keep_mask = torch.empty(token_count, dtype=torch.bool, device="cpu")
    materialized_stage = torch.empty(token_count, dtype=torch.int32, device="cpu")
    effective = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    ignored = torch.zeros(reposition_count, dtype=torch.bool, device="cpu")
    status = torch.zeros(6, dtype=torch.int64, device="cpu")

    _load_module().compile_radix_reposition_layout(
        token_ids,
        drop_insert_offsets,
        drop_range_offsets,
        drop_ranges,
        delta_marker_ids,
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
    return RadixRepositionLayout(
        records=records[:key_len],
        virtual_mask=virtual_mask[:key_len],
        key_to_token=key_to_token[:key_len],
        token_to_key=token_to_key,
        positions=positions,
        repos_info=repos_info,
        keep_mask=keep_mask,
        materialized_stage=materialized_stage,
        effective_repositions=effective,
        ignored_repositions=ignored,
        next_position=int(status[3]),
        current_reposition=int(status[5]),
    )


__all__ = [
    "DELTA_KIND",
    "REPOSITION_KIND",
    "TOKEN_KIND",
    "RadixRepositionLayout",
    "compile_radix_reposition_layout",
]
