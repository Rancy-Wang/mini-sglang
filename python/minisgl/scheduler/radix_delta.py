from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

CanonicalDelta = tuple[tuple[int, int], ...]


def canonicalize_delta(dropped_positions: Iterable[int]) -> CanonicalDelta:
    """Return sorted, de-duplicated half-open ranges for token positions."""

    positions = sorted({int(position) for position in dropped_positions})
    if any(position < 0 for position in positions):
        raise ValueError(f"Delta token positions must be non-negative: {positions}")
    if not positions:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        ranges.append((start, previous + 1))
        start = previous = position
    ranges.append((start, previous + 1))
    return tuple(ranges)


def canonicalize_delta_ranges(ranges: Iterable[Sequence[int]]) -> CanonicalDelta:
    normalized: list[tuple[int, int]] = []
    for raw_range in ranges:
        if len(raw_range) != 2:
            raise ValueError(f"Delta ranges must contain two endpoints: {raw_range}")
        start, end = (int(raw_range[0]), int(raw_range[1]))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid half-open delta range: [{start}, {end})")
        normalized.append((start, end))
    if not normalized:
        return ()
    normalized.sort()
    merged: list[tuple[int, int]] = [normalized[0]]
    for start, end in normalized[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def encode_delta_endpoint(position: int) -> int:
    position = int(position)
    if position < 0 or position > (1 << 31) - 1:
        raise ValueError(f"Delta endpoint is outside the int32 token range: {position}")
    return -(position + 1)


def decode_delta_record(record: Sequence[int]) -> tuple[int, int]:
    """Decode ``[1, -(start+1), -(end+1), -1]`` into ``[start, end)``."""

    if len(record) != 4:
        raise ValueError("A Delta record must contain four int32 fields.")
    kind, encoded_start, encoded_end, sentinel = (int(value) for value in record)
    if kind != 1 or encoded_start >= 0 or encoded_end >= 0 or sentinel != -1:
        raise ValueError(f"Invalid direct Delta range record: {list(record)}")
    start = -encoded_start - 1
    end = -encoded_end - 1
    if end <= start:
        raise ValueError(f"Invalid decoded half-open Delta range: [{start}, {end})")
    return start, end


def validate_delta_records(
    records: torch.Tensor,
    *,
    token_count: int | None = None,
    require_materialized: bool = False,
) -> None:
    """Validate direct structured records and canonical consecutive Delta blocks."""

    if (
        records.device.type != "cpu"
        or records.dtype != torch.int32
        or records.ndim != 2
        or records.shape[1] != 4
    ):
        raise ValueError("Structured Radix records must be CPU int32 [N, 4].")
    previous_delta_end: int | None = None
    materialized_tokens = 0
    for record in records.tolist():
        kind = int(record[0])
        if kind == 0:
            if int(record[1]) < 0:
                raise ValueError("Token records require non-negative token IDs.")
            previous_delta_end = None
            materialized_tokens += 1
        elif kind == 1:
            start, end = decode_delta_record(record)
            if token_count is not None and end > token_count:
                raise ValueError(
                    f"Delta range [{start}, {end}) exceeds token length {token_count}."
                )
            if require_materialized and end > materialized_tokens:
                raise ValueError(
                    f"Delta range [{start}, {end}) precedes its materialization boundary "
                    f"{materialized_tokens}."
                )
            if previous_delta_end is not None and start <= previous_delta_end:
                raise ValueError("Consecutive Delta ranges must be canonical and disjoint.")
            previous_delta_end = end
        elif kind == 2:
            if int(record[1]) < 0 or record[2:] != [-1, -1]:
                raise ValueError(f"Invalid Reposition record: {record}")
            if require_materialized and int(record[1]) + 1 != materialized_tokens:
                raise ValueError(
                    "Reposition raw boundary does not match its record insertion point."
                )
            previous_delta_end = None
        else:
            raise ValueError(f"Unknown structured Radix record kind: {kind}")
