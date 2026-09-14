from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def mask_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def proven_skip_ranges(handle, required_raw: torch.Tensor) -> list[tuple[int, int]]:
    """Only a Delta on this matched path authorizes releasing an ancestor KV."""
    if not handle.cached_len:
        return []
    records = handle.get_matched_keys().numpy()
    if records.ndim != 2 or records.shape[1] != 4:
        return []
    real = records[:, 0] == 0
    raw_keys = np.flatnonzero(real)
    count = len(raw_keys)
    required = required_raw.numpy()
    if required.dtype != np.bool_ or len(required) != count:
        raise ValueError("Drop lock demand must cover the matched raw prefix.")
    dropped = np.zeros(count, dtype=np.bool_)
    raw_before = np.cumsum(real)
    for key in np.flatnonzero(records[:, 0] == 1):
        start, end = -int(records[key, 1]) - 1, -int(records[key, 2]) - 1
        if not 0 <= start < end <= int(raw_before[key]):
            raise ValueError("Matched Delta references a non-ancestor token range.")
        dropped[start:end] = True
    skip = np.zeros(len(records), dtype=np.bool_)
    skip[raw_keys] = dropped & ~required
    return mask_ranges(skip)


@dataclass(frozen=True)
class RecoveryPlan:
    # Continuous intervals use the existing prefill attention path. Resident
    # gaps between intervals are reused, including the matched suffix.
    intervals: tuple[tuple[int, int], ...]
    required_prefix: torch.Tensor
    matched_length: int

    @property
    def start(self) -> int:
        return self.intervals[0][0]

    def next_interval(self, cursor: int) -> tuple[int, int]:
        for start, end in self.intervals:
            if cursor < end:
                return max(start, cursor), end
        raise ValueError("Recovery cursor is past the query stream.")


def plan_recovery(
    resident: torch.Tensor,
    visible_until: torch.Tensor,
    input_length: int,
) -> RecoveryPlan:
    """Close missing KV dependencies backwards without scanning the Radix tree.

    A missing token t is needed iff a planned later query q < expiry[t]
    reads it. A reverse scan over missing tokens closes this relation: once a
    token is needed it becomes the earliest query for all preceding tokens.
    """
    present = resident.numpy()
    expiry = visible_until.numpy()
    matched = len(present)
    if not 0 <= matched < input_length or len(expiry) < input_length:
        raise ValueError("Recovery metadata must leave an uncached query suffix.")
    missing = np.flatnonzero(~present)
    needed = np.zeros(input_length, dtype=np.bool_)
    needed[matched:] = True
    earliest = matched
    for raw in missing[::-1]:
        if expiry[raw] > earliest:
            needed[raw] = True
            earliest = int(raw)
    # A resident token is held only when some planned query can read it.
    next_query = np.minimum.accumulate(
        np.where(needed, np.arange(input_length), input_length)[::-1]
    )[::-1]
    required = expiry[:matched] > next_query[1:matched + 1]
    required |= needed[:matched]
    return RecoveryPlan(tuple(mask_ranges(needed)), torch.from_numpy(required), matched)


def build_drop_capacity_index(req, owned, source_positions, exact, start, end):
    """Vectorized activation curves; omit terminal copies needed only for dropped cacheback.

    Work is O(occurrences + referenced keys + query chunk), independent of the
    number of capacity probes. All inputs and outputs stay on the CPU.
    """
    raw = req.occurrence_raw_tokens.numpy()
    pos = req.occurrence_positions.numpy()
    birth = req.occurrence_birth_indices.numpy()
    terminal = req.occurrence_terminal_indices.numpy()
    n = len(birth)
    keep = (req.full_keep_mask.numpy().astype(np.bool_) if req.full_keep_mask is not None
            else np.ones(n, dtype=np.bool_))
    owner = owned.numpy()
    source = source_positions.numpy()
    first = np.full(len(raw), end + 1, dtype=np.int32)
    q = np.arange(start, end)
    first[birth[q]] = q + 1
    np.minimum.at(first, terminal[q[keep[q]]], q[keep[q]] + 1)
    if end == n:
        old = np.flatnonzero(keep[:len(source)])
        np.minimum.at(first, terminal[old], n)
    starts = req.occurrence_segment_query_starts.numpy()
    ends = req.occurrence_segment_query_ends.numpy()
    offsets = req.occurrence_segment_key_offsets.numpy()
    keys = req.occurrence_segment_key_indices.numpy()
    for a, b, x, y in zip(starts, ends, offsets[:-1], offsets[1:], strict=True):
        left, right = max(start, int(a)), min(end, int(b))
        if left >= right:
            continue
        prefix = int(y - x - (b - a))
        selected = keys[x:x + prefix + right - a]
        activation = np.maximum(left, a + np.arange(len(selected)) - prefix) + 1
        np.minimum.at(first, selected, activation)

    ids = np.flatnonzero(first <= end)
    r = raw[ids]
    prior = r < start
    canonical = pos[birth[r]].copy()
    matched = r < len(source)
    canonical[matched] = source[r[matched]]
    reuse = owner[r] & (pos[ids] == pos[terminal[r]])
    canonical[reuse] = pos[terminal[r[reuse]]]
    new = ~prior | (pos[ids] != canonical)
    retry = prior & matched & (r >= exact) & (ids == terminal[r]) & ~owner[r]
    late_retry = retry & ~new & (end == n)
    persistent = new & ((ids == birth[r]) & ~prior | (ids == terminal[r]) & keep[r])
    activation = first[ids] - start - 1

    def curve(mask):
        return np.cumsum(np.bincount(activation[mask], minlength=end - start), dtype=np.int64)

    current, retained = curve(new), curve(persistent)
    if end == n:
        current[-1] += int(late_retry.sum())
        retained[-1] += int((late_retry & keep[r]).sum())
    canonical_terminal = pos[birth].copy()
    canonical_terminal[:len(source)] = source
    terminal_needed = keep & ~owner & (
        (pos[terminal] != canonical_terminal)
        | ((np.arange(n) < len(source)) & (np.arange(n) >= exact))
    )
    acquire = np.full(n, end + 1, dtype=np.int32)
    acquired = persistent & (ids == terminal[r])
    np.minimum.at(acquire, r[acquired], first[ids[acquired]])
    if end == n:
        np.minimum.at(acquire, r[late_retry & keep[r]], n)
    # Birth == terminal still satisfies final ownership for fresh tokens.
    terminal_remaining = int(terminal_needed.sum()) - np.cumsum(np.bincount(
        acquire[terminal_needed & (acquire <= end)] - start - 1,
        minlength=end - start), dtype=np.int64)
    pending = np.zeros(n, dtype=np.int64)
    for a, b in req.drop_recovery_plan.intervals:
        pending[max(a, start):b] = 1
    future_birth = np.cumsum(pending[::-1])[::-1]
    future_birth = np.r_[future_birth, 0][np.arange(start + 1, end + 1)]
    future = future_birth + terminal_remaining + req.output_len
    return tuple(torch.from_numpy(a) for a in (first, current, retained, future))
