"""CPU-only transition plans and resource-scoped GPU completion fences.

These objects do not allocate KV pages or change admission order. A scheduler
remains the sole writer, with at most its existing one batch of lookahead.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class CompactPlan:
    prompt_len: int
    keep: torch.Tensor
    keep_indices: torch.Tensor
    dropped_indices: torch.Tensor
    inactive_positions: torch.Tensor | None
    input_ids: torch.Tensor
    true_positions: torch.Tensor
    raw_positions: torch.Tensor
    radix_input_ids: torch.Tensor
    initial_cached_len: int
    retry_mask: torch.Tensor | None
    terminal_owned: torch.Tensor | None

    @classmethod
    def build(cls, req, prompt_len: int):
        mask = req.context_post_prefill_keep_mask
        raw = req.raw_positions[:prompt_len].to(dtype=torch.int64, device="cpu")
        if mask is None or not len(raw) or int(raw[-1]) >= len(mask):
            raise RuntimeError("Post-Prefill keep mask does not cover the prompt raw positions.")
        keep = (mask[raw] != 0).to(torch.bool)
        indices = torch.nonzero(keep, as_tuple=False).view(-1)
        if not len(indices):
            raise RuntimeError("Cannot Drop every prompt token before generation.")
        owned = req.occurrence_terminal_owned_mask
        if owned is not None:
            if len(owned) != prompt_len:
                raise RuntimeError("Occurrence-owned pages do not cover the prompt stream.")
        else:
            owned = torch.arange(prompt_len) >= req.initial_active_cached_len
            if req.retry_transformed_mask is not None:
                owned[:len(req.retry_transformed_mask)] |= req.retry_transformed_mask
        dropped = (~keep) & owned
        dropped_indices = torch.nonzero(dropped, as_tuple=False).view(-1)
        inactive = req.inactive_cached_positions
        if len(dropped_indices):
            selected = raw[dropped]
            inactive = selected if inactive is None else torch.cat((inactive, selected))
        positions = req.true_positions[:prompt_len]
        if req.reposition_execution_mode == "paged-occurrence":
            if req.radix_positions is None:
                raise RuntimeError("Paged-occurrence compaction requires final Radix positions.")
            positions = req.radix_positions[raw]
        initial_keep = keep[:req.initial_active_cached_len]
        return cls(
            prompt_len, keep, indices, dropped_indices, inactive,
            req.input_ids[keep].contiguous(), positions[keep].contiguous(),
            req.raw_positions[:prompt_len][keep].contiguous(),
            req.radix_input_ids[keep].contiguous(), int(initial_keep.count_nonzero()),
            None if req.retry_transformed_mask is None else req.retry_transformed_mask[initial_keep].contiguous(),
            None if req.occurrence_terminal_owned_mask is None else req.occurrence_terminal_owned_mask[keep].contiguous(),
        )


@dataclass(eq=False)
class TransitionFence:
    """Retain sources until their last read; fence only actual consumers."""

    event: Any
    resources: tuple = ()
    waited_streams: set = field(default_factory=set)

    def wait_on(self, stream):
        key = stream.cuda_stream
        if key not in self.waited_streams:
            stream.wait_event(self.event)
            self.waited_streams.add(key)

    def release_if_ready(self) -> bool:
        if not self.event.query():
            return False
        self.resources = ()
        return True


class TransitionRetirement:
    """Bounded metadata retention, NOT a queue of prematurely freed KV pages."""

    def __init__(self, capacity=2):
        self.capacity = capacity
        self.pending = []

    def collect(self):
        self.pending[:] = [fence for fence in self.pending if not fence.release_if_ready()]

    def add(self, fence):
        self.collect()
        if len(self.pending) >= self.capacity:
            # Resource pressure is a deliberate safe fallback, never an
            # unbounded allocation or a change to request admission order.
            self.pending[0].event.synchronize()
            self.collect()
        self.pending.append(fence)

    def clear_after_synchronize(self):
        for fence in self.pending:
            fence.resources = ()
        self.pending.clear()
