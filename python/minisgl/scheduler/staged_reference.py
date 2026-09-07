"""Cold sequential Drop reference. No mask compiler or cross-request Radix state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle

if TYPE_CHECKING:
    from minisgl.core import Req
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .table import TableManager


@dataclass(frozen=True)
class PrivateCacheHandle(BaseCacheHandle):
    """A sentinel that must never be passed to a prefix cache."""

    def get_matched_indices(self) -> torch.Tensor:
        raise RuntimeError("The staged reference has no Radix handle.")

    def get_matched_virtual_mask(self) -> torch.Tensor:
        raise RuntimeError("The staged reference has no Radix handle.")

    @property
    def physical_cached_len(self) -> int:
        return 0


@dataclass
class StagedReferenceState:
    full_ids: torch.Tensor
    events: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    cursor: int = 0
    event_index: int = 0
    segment_end: int = 0
    segments: int = 0
    forward_complete: bool = False
    prefill_done: bool = False
    released: bool = False
    owned_pages: torch.Tensor | None = None

    @classmethod
    def from_message(cls, msg: UserMsg) -> StagedReferenceState:
        if msg.is_warmup or msg.use_context_mask or msg.radix_current_reposition != -1:
            raise ValueError("The staged reference cannot use warmup, masks or Reposition.")
        ids = msg.input_ids
        expected = torch.arange(len(ids), dtype=torch.int32)
        if ids.ndim != 1 or not ids.is_cpu or ids.dtype != torch.int32 or len(ids) == 0:
            raise ValueError("The staged reference needs a nonempty canonical CPU token stream.")
        if not torch.equal(msg.raw_positions, expected) or not torch.equal(
            msg.true_positions, expected
        ):
            raise ValueError("Reference input must retain every canonical absolute position.")
        positions, offsets, ranges = (
            msg.drop_event_positions, msg.drop_range_offsets, msg.drop_position_ranges
        )
        if any(t is None or not t.is_cpu or t.ndim != 1 or t.dtype != torch.int32
               for t in (positions, offsets, ranges)):
            raise ValueError("Reference Drop events require CPU int32 CSR tensors.")
        assert positions is not None and offsets is not None and ranges is not None
        count = msg.drop_effective_event_count
        if (not 0 < count <= len(positions) or len(offsets) != len(positions) + 1
                or int(offsets[0]) != 0 or int(offsets[-1]) * 2 != len(ranges)
                or bool(torch.any(offsets[1:] < offsets[:-1]))):
            raise ValueError("Invalid reference Drop event offsets/count.")
        events = []
        previous = 0
        for i in range(count):
            boundary = int(positions[i])
            if not previous <= boundary <= len(ids):
                raise ValueError("Reference Drop boundaries are not chronological.")
            pairs = tuple(tuple(pair) for pair in ranges[
                2 * int(offsets[i]):2 * int(offsets[i + 1])
            ].view(-1, 2).tolist())
            if any(not 0 <= start < end <= boundary for start, end in pairs):
                raise ValueError("A Drop cannot remove a token before its forward completes.")
            events.append((boundary, pairs))
            previous = boundary
        return cls(ids, tuple(events))

    @property
    def remaining_input(self) -> int:
        return len(self.full_ids) - self.cursor

    def next_end(self, budget: int) -> int:
        if self.released or self.prefill_done or self.forward_complete or budget <= 0:
            raise RuntimeError("Reference stage scheduled in an invalid state.")
        # Zero-length events may only have empty ranges; consume them without a forward.
        while self.event_index < len(self.events) and self.events[self.event_index][0] == self.cursor:
            if self.events[self.event_index][1]:
                raise RuntimeError("A reference Drop was not applied after its preceding segment.")
            self.event_index += 1
        boundary = (self.events[self.event_index][0]
                    if self.event_index < len(self.events) else len(self.full_ids))
        self.segment_end = min(boundary, self.cursor + budget, len(self.full_ids))
        if self.segment_end <= self.cursor:
            raise RuntimeError("Reference Prefill made no progress.")
        return self.segment_end

    def register_pages(self, pages: torch.Tensor) -> None:
        if self.released:
            raise RuntimeError("Cannot allocate pages for a released reference.")
        self.owned_pages = pages.clone() if self.owned_pages is None else torch.cat(
            (self.owned_pages, pages)
        )
        if len(torch.unique(self.owned_pages)) != len(self.owned_pages):
            raise RuntimeError("Reference KV page ownership contains duplicates.")

    def finish_segment(self, req: Req, table: TableManager, cache: CacheManager) -> bool:
        if not self.forward_complete or self.prefill_done or self.released:
            raise RuntimeError("Reference completion must follow exactly one forward.")
        if req.cached_len != req.device_len or req.device_len != len(req.input_ids):
            raise RuntimeError("An intermediate sample changed the reference prompt state.")
        if self.owned_pages is None:
            raise RuntimeError("Reference forward has no owned KV pages.")
        pages = table.page_table[req.table_idx, :req.cached_len].clone()
        if not torch.equal(pages, self.owned_pages):
            raise RuntimeError("Reference page table disagrees with private ownership.")
        if int(req.raw_positions[-1]) + 1 != self.segment_end:
            raise RuntimeError("Reference did not finish its scheduled canonical segment.")
        self.cursor = self.segment_end
        self.segments += 1
        keep = torch.ones(len(req.input_ids), dtype=torch.bool)
        while self.event_index < len(self.events) and self.events[self.event_index][0] == self.cursor:
            for start, end in self.events[self.event_index][1]:
                keep &= ~((req.raw_positions >= start) & (req.raw_positions < end))
            self.event_index += 1
        if not bool(torch.all(keep)):
            device_keep = keep.to(pages.device, non_blocking=True)
            survivors = pages[device_keep]
            cache.free_reference_pages(pages[~device_keep])
            self.owned_pages = survivors
            n = len(survivors)
            table.page_table[req.table_idx, :n].copy_(survivors)
            pool = table.token_pool[req.table_idx, :req.cached_len]
            pool[:n].copy_(pool[device_keep])
            table.page_table[req.table_idx, n:req.cached_len].fill_(-1)
            req.input_ids = req.input_ids[keep]
            req.true_positions = req.true_positions[keep]
            req.raw_positions = req.raw_positions[keep]
            req.radix_input_ids = req.radix_input_ids[keep]
            req.cached_len = req.device_len = n
        req.max_device_len = req.device_len + self.remaining_input + req.output_len
        self.forward_complete = False
        self.prefill_done = self.cursor == len(self.full_ids)
        if self.prefill_done:
            if len(req.input_ids) == 0:
                raise RuntimeError("Reference has no active token at generation.")
            req.true_seq_len = len(self.full_ids)
        return self.prefill_done
