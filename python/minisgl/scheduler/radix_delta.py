from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

CanonicalDelta = tuple[tuple[int, int], ...]


def canonicalize_delta(dropped_ids: Iterable[int]) -> CanonicalDelta:
    """Return sorted, de-duplicated half-open ranges for one newly effective delta."""

    ids = sorted({int(msg_id) for msg_id in dropped_ids})
    if any(msg_id < 0 for msg_id in ids):
        raise ValueError(f"Delta message IDs must be non-negative: {ids}")
    if not ids:
        return ()

    ranges: list[tuple[int, int]] = []
    start = previous = ids[0]
    for msg_id in ids[1:]:
        if msg_id == previous + 1:
            previous = msg_id
            continue
        ranges.append((start, previous + 1))
        start = previous = msg_id
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


@dataclass
class DeltaMarkerRegistry:
    """Intern canonical drop deltas into scheduler-local negative int64 keys."""

    _markers: dict[CanonicalDelta, int] = field(default_factory=dict)
    _next_marker: int = -1

    def intern(self, dropped_ids: Iterable[int]) -> int:
        return self.intern_canonical(canonicalize_delta(dropped_ids))

    def intern_canonical(self, delta: Iterable[Sequence[int]]) -> int:
        canonical = canonicalize_delta_ranges(delta)
        if not canonical:
            raise ValueError("An empty delta must not create a Radix marker.")
        existing = self._markers.get(canonical)
        if existing is not None:
            return existing
        if self._next_marker < -(1 << 63):
            raise RuntimeError("Exhausted signed int64 delta-marker namespace.")
        marker = self._next_marker
        self._next_marker -= 1
        self._markers[canonical] = marker
        return marker

    @property
    def size(self) -> int:
        return len(self._markers)


@dataclass(frozen=True)
class DeltaRadixLayout:
    keys: torch.Tensor
    virtual_mask: torch.Tensor
    key_to_token: torch.Tensor
    token_to_key: torch.Tensor


def key_prefix_len_for_token_boundary(layout: DeltaRadixLayout, token_boundary: int) -> int:
    """Map a full-token boundary to a key prefix, including markers at the boundary."""

    token_boundary = int(token_boundary)
    token_count = len(layout.token_to_key)
    if token_boundary < 0 or token_boundary > token_count:
        raise ValueError(f"Token boundary {token_boundary} is outside token length {token_count}.")
    if token_boundary == token_count:
        return len(layout.keys)
    return int(layout.token_to_key[token_boundary].item())


def inject_delta_markers(
    full_radix_ids: torch.Tensor,
    marker_meta: list[dict],
    registry: DeltaMarkerRegistry,
) -> DeltaRadixLayout | None:
    """Insert one virtual marker per canonical insertion boundary."""

    if full_radix_ids.device.type != "cpu" or full_radix_ids.dtype != torch.int64:
        raise ValueError("full_radix_ids must be a CPU int64 tensor.")
    if full_radix_ids.ndim != 1:
        raise ValueError("full_radix_ids must be one-dimensional.")
    if not marker_meta:
        return None

    deltas_by_pos: dict[int, list[Sequence[int]]] = {}
    for marker in marker_meta:
        insertion_pos = int(marker["insertion_pos"])
        if insertion_pos < 0 or insertion_pos > len(full_radix_ids):
            raise ValueError(
                f"Delta marker insertion {insertion_pos} is outside token length "
                f"{len(full_radix_ids)}."
            )
        canonical = canonicalize_delta_ranges(marker.get("delta", ()))
        if not canonical:
            continue
        deltas_by_pos.setdefault(insertion_pos, []).extend(canonical)
    if not deltas_by_pos:
        return None

    keys: list[int] = []
    virtual_mask: list[bool] = []
    key_to_token: list[int] = []
    token_to_key = torch.empty(len(full_radix_ids), dtype=torch.int64, device="cpu")
    for token_pos in range(len(full_radix_ids) + 1):
        delta = deltas_by_pos.get(token_pos)
        if delta:
            keys.append(registry.intern_canonical(delta))
            virtual_mask.append(True)
            key_to_token.append(-1)
        if token_pos == len(full_radix_ids):
            continue
        token_to_key[token_pos] = len(keys)
        keys.append(int(full_radix_ids[token_pos].item()))
        virtual_mask.append(False)
        key_to_token.append(token_pos)

    return DeltaRadixLayout(
        keys=torch.tensor(keys, dtype=torch.int64, device="cpu"),
        virtual_mask=torch.tensor(virtual_mask, dtype=torch.bool, device="cpu"),
        key_to_token=torch.tensor(key_to_token, dtype=torch.int64, device="cpu"),
        token_to_key=token_to_key,
    )
