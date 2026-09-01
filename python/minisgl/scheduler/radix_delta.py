from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

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


@dataclass
class DeltaMarkerRegistry:
    """Intern exact position deltas and retain only cached/in-flight markers."""

    _markers: dict[CanonicalDelta, int] = field(default_factory=dict)
    _canonical_by_marker: dict[int, CanonicalDelta] = field(default_factory=dict)
    _request_refs: Counter[int] = field(default_factory=Counter)
    _tree_refs: Counter[int] = field(default_factory=Counter)
    _next_marker: int = -1

    def intern(self, dropped_positions: Iterable[int]) -> int:
        return self.intern_canonical(canonicalize_delta(dropped_positions))

    def intern_canonical(self, delta: Iterable[Sequence[int]]) -> int:
        canonical = canonicalize_delta_ranges(delta)
        if not canonical:
            raise ValueError("An empty delta must not create a Radix marker.")
        existing = self._markers.get(canonical)
        if existing is not None:
            return existing
        if self._next_marker < -(1 << 31):
            raise RuntimeError("Exhausted signed int32 delta-marker namespace.")
        marker = self._next_marker
        self._next_marker -= 1
        self._markers[canonical] = marker
        self._canonical_by_marker[marker] = canonical
        return marker

    def acquire_canonical(self, delta: Iterable[Sequence[int]]) -> int:
        marker = self.intern_canonical(delta)
        self._request_refs[marker] += 1
        return marker

    def release_request_refs(self, marker_ids: Iterable[int]) -> None:
        for raw_marker in marker_ids:
            marker = int(raw_marker)
            if self._request_refs[marker] <= 0:
                raise RuntimeError(f"Delta marker {marker} has no request reference to release.")
            self._request_refs[marker] -= 1
            self._discard_if_unused(marker)

    def add_tree_refs(self, marker_ids: Iterable[int]) -> None:
        for raw_marker in marker_ids:
            marker = int(raw_marker)
            if marker not in self._canonical_by_marker:
                raise RuntimeError(f"Radix inserted an unknown delta marker: {marker}")
            self._tree_refs[marker] += 1

    def remove_tree_refs(self, marker_ids: Iterable[int]) -> None:
        for raw_marker in marker_ids:
            marker = int(raw_marker)
            if self._tree_refs[marker] <= 0:
                raise RuntimeError(f"Delta marker {marker} has no tree reference to release.")
            self._tree_refs[marker] -= 1
            self._discard_if_unused(marker)

    def canonical_for(self, marker: int) -> CanonicalDelta:
        try:
            return self._canonical_by_marker[int(marker)]
        except KeyError as exc:
            raise RuntimeError(f"Unknown delta marker: {marker}") from exc

    def check_tree_refs(self, actual: Counter[int]) -> None:
        expected = Counter({marker: count for marker, count in self._tree_refs.items() if count})
        if actual != expected:
            raise RuntimeError(
                "Delta marker tree reference mismatch: "
                f"registry={dict(expected)}, actual={dict(actual)}"
            )
        unknown = set(actual) - set(self._canonical_by_marker)
        if unknown:
            raise RuntimeError(f"Radix tree contains unknown delta markers: {sorted(unknown)}")

    def _discard_if_unused(self, marker: int) -> None:
        if self._request_refs[marker] != 0 or self._tree_refs[marker] != 0:
            return
        canonical = self._canonical_by_marker.pop(marker)
        del self._markers[canonical]
        self._request_refs.pop(marker, None)
        self._tree_refs.pop(marker, None)

    @property
    def size(self) -> int:
        return len(self._markers)

    @property
    def request_ref_count(self) -> int:
        return sum(self._request_refs.values())

    @property
    def tree_ref_count(self) -> int:
        return sum(self._tree_refs.values())


@dataclass(frozen=True)
class DeltaRadixLayout:
    keys: torch.Tensor
    virtual_mask: torch.Tensor
    key_to_token: torch.Tensor
    token_to_key: torch.Tensor
    marker_ids: tuple[int, ...]


def select_effective_delta_events(
    event_positions: torch.Tensor,
    range_offsets: torch.Tensor,
    position_ranges: torch.Tensor,
    final_keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only Drop events that are effective for this request target.

    Context warmups may carry future Drop events for the same full message stream.
    Those events must not become Radix markers until their position ranges are
    hidden by the request's target-specific final keep mask.
    """

    if (
        final_keep_mask.device.type != "cpu"
        or final_keep_mask.ndim != 1
        or final_keep_mask.dtype not in (torch.bool, torch.int32, torch.int64)
    ):
        raise ValueError("final_keep_mask must be a CPU 1D bool/integer tensor.")
    _validate_position_wire(
        len(final_keep_mask), event_positions, range_offsets, position_ranges
    )
    keep_mask = final_keep_mask != 0
    selected_positions: list[int] = []
    selected_ranges: list[tuple[int, int]] = []
    selected_offsets: list[int] = [0]
    ranges = position_ranges.view(-1, 2)
    for event_idx, raw_event_position in enumerate(event_positions.tolist()):
        range_start = int(range_offsets[event_idx].item())
        range_end = int(range_offsets[event_idx + 1].item())
        event_ranges = [
            (int(start), int(end))
            for start, end in ranges[range_start:range_end].tolist()
        ]
        effective: list[bool] = []
        for start, end in event_ranges:
            segment = keep_mask[start:end]
            all_kept = bool(torch.all(segment).item())
            all_dropped = bool(torch.all(~segment).item())
            if not (all_kept or all_dropped):
                raise ValueError(
                    "A target-specific keep mask partially cuts a Drop delta range: "
                    f"event={raw_event_position}, range=[{start}, {end})"
                )
            effective.append(all_dropped)
        if effective and any(state != effective[0] for state in effective[1:]):
            raise ValueError(
                "One Drop event has inconsistent target-specific range visibility: "
                f"event={raw_event_position}, effective={effective}"
            )
        if effective and effective[0]:
            selected_positions.append(int(raw_event_position))
            selected_ranges.extend(event_ranges)
            selected_offsets.append(len(selected_ranges))

    return (
        torch.tensor(selected_positions, dtype=event_positions.dtype, device="cpu"),
        torch.tensor(selected_offsets, dtype=range_offsets.dtype, device="cpu"),
        torch.tensor(
            selected_ranges, dtype=position_ranges.dtype, device="cpu"
        ).reshape(-1),
    )


def key_prefix_len_for_token_boundary(layout: DeltaRadixLayout, token_boundary: int) -> int:
    """Map a full-token boundary to a key prefix, including markers at the boundary."""

    token_boundary = int(token_boundary)
    token_count = len(layout.token_to_key)
    if token_boundary < 0 or token_boundary > token_count:
        raise ValueError(f"Token boundary {token_boundary} is outside token length {token_count}.")
    if token_boundary == token_count:
        return len(layout.keys)
    return int(layout.token_to_key[token_boundary].item())


def _validate_position_wire(
    full_token_count: int,
    event_positions: torch.Tensor,
    range_offsets: torch.Tensor,
    position_ranges: torch.Tensor,
) -> None:
    tensors = (event_positions, range_offsets, position_ranges)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("Token-position Drop metadata must use CPU tensors.")
    if event_positions.ndim != 1 or range_offsets.ndim != 1 or position_ranges.ndim != 1:
        raise ValueError("Token-position Drop metadata has an invalid rank.")
    if len(position_ranges) % 2 != 0:
        raise ValueError("drop_position_ranges must contain flattened start/end pairs.")
    if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in tensors):
        raise ValueError("Token-position Drop metadata must use integer tensors.")
    if len(range_offsets) != len(event_positions) + 1:
        raise ValueError("drop_range_offsets must have event_count + 1 entries.")
    if len(range_offsets) == 0 or int(range_offsets[0].item()) != 0:
        raise ValueError("drop_range_offsets must start at zero.")
    if int(range_offsets[-1].item()) * 2 != len(position_ranges):
        raise ValueError("drop_range_offsets does not cover all position ranges.")
    if len(range_offsets) > 1 and bool(torch.any(range_offsets[1:] < range_offsets[:-1]).item()):
        raise ValueError("drop_range_offsets must be monotonically non-decreasing.")
    if len(event_positions) > 1 and bool(
        torch.any(event_positions[1:] <= event_positions[:-1]).item()
    ):
        raise ValueError("drop_event_positions must be strictly increasing.")
    if len(event_positions) > 0 and (
        bool(torch.any(event_positions < 0).item())
        or bool(torch.any(event_positions > full_token_count).item())
    ):
        raise ValueError("A Drop event position is outside the full token stream.")


def inject_delta_markers(
    full_radix_ids: torch.Tensor,
    event_positions: torch.Tensor,
    range_offsets: torch.Tensor,
    position_ranges: torch.Tensor,
    registry: DeltaMarkerRegistry,
) -> DeltaRadixLayout | None:
    """Insert one virtual marker for each absolute token-position Drop event."""

    if full_radix_ids.device.type != "cpu" or full_radix_ids.dtype != torch.int64:
        raise ValueError("full_radix_ids must be a CPU int64 tensor.")
    if full_radix_ids.ndim != 1:
        raise ValueError("full_radix_ids must be one-dimensional.")
    _validate_position_wire(len(full_radix_ids), event_positions, range_offsets, position_ranges)
    if len(event_positions) == 0:
        return None

    deltas_by_pos: dict[int, CanonicalDelta] = {}
    for event_idx, raw_position in enumerate(event_positions.tolist()):
        insertion_pos = int(raw_position)
        range_start = int(range_offsets[event_idx].item())
        range_end = int(range_offsets[event_idx + 1].item())
        canonical = canonicalize_delta_ranges(
            position_ranges[2 * range_start : 2 * range_end].view(-1, 2).tolist()
        )
        if not canonical:
            raise ValueError("A token-position Drop event must contain at least one range.")
        if canonical[-1][1] > insertion_pos:
            raise ValueError(
                "A Drop event cannot hide a token before it has been computed: "
                f"position={insertion_pos}, ranges={canonical}"
            )
        deltas_by_pos[insertion_pos] = canonical

    keys: list[int] = []
    virtual_mask: list[bool] = []
    key_to_token: list[int] = []
    marker_ids: list[int] = []
    token_to_key = torch.empty(len(full_radix_ids), dtype=torch.int64, device="cpu")
    try:
        for token_pos in range(len(full_radix_ids) + 1):
            delta = deltas_by_pos.get(token_pos)
            if delta:
                marker = registry.acquire_canonical(delta)
                marker_ids.append(marker)
                keys.append(marker)
                virtual_mask.append(True)
                key_to_token.append(-1)
            if token_pos == len(full_radix_ids):
                continue
            token_to_key[token_pos] = len(keys)
            keys.append(int(full_radix_ids[token_pos].item()))
            virtual_mask.append(False)
            key_to_token.append(token_pos)
    except Exception:
        registry.release_request_refs(marker_ids)
        raise

    return DeltaRadixLayout(
        keys=torch.tensor(keys, dtype=torch.int64, device="cpu"),
        virtual_mask=torch.tensor(virtual_mask, dtype=torch.bool, device="cpu"),
        key_to_token=torch.tensor(key_to_token, dtype=torch.int64, device="cpu"),
        token_to_key=token_to_key,
        marker_ids=tuple(marker_ids),
    )


def acquire_delta_marker_ids(
    full_token_count: int,
    event_positions: torch.Tensor,
    range_offsets: torch.Tensor,
    position_ranges: torch.Tensor,
    registry: DeltaMarkerRegistry,
) -> tuple[int, ...]:
    """Acquire one canonical marker per token-granular Drop event."""

    _validate_position_wire(
        full_token_count, event_positions, range_offsets, position_ranges
    )
    ranges = position_ranges.view(-1, 2)
    marker_ids: list[int] = []
    try:
        for event_idx, insertion_pos in enumerate(event_positions.tolist()):
            range_start = int(range_offsets[event_idx])
            range_end = int(range_offsets[event_idx + 1])
            canonical = canonicalize_delta_ranges(ranges[range_start:range_end].tolist())
            if not canonical:
                raise ValueError("A token-position Drop event must contain a range.")
            if canonical[-1][1] > int(insertion_pos):
                raise ValueError(
                    "A Drop event cannot hide tokens before they are computed: "
                    f"position={insertion_pos}, ranges={canonical}"
                )
            marker_ids.append(registry.acquire_canonical(canonical))
    except Exception:
        registry.release_request_refs(marker_ids)
        raise
    return tuple(marker_ids)
