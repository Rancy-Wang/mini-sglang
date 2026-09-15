"""Pinned, batch-packed indices for post-Prefill compaction.

Packing runs on the scheduler stream BEFORE forward. Consumers only enqueue
index_select operations; neither retirement nor reuse waits on the host.
"""

from dataclasses import dataclass, field

import torch


@dataclass(eq=False)
class _Buffer:
    host: torch.Tensor
    device: torch.Tensor
    pending: int = 0
    events: dict = field(default_factory=dict)

    def available(self) -> bool:
        return self.pending == 0 and all(event.query() for event in self.events.values())


@dataclass(eq=False)
class CompactIndexLease:
    """One request's views; retain BOTH buffers until its GPU reads complete."""

    keep: torch.Tensor
    dropped_owned: torch.Tensor
    _buffer: _Buffer
    _released: bool = False

    def release(self, stream: torch.cuda.Stream) -> None:
        if self._released:
            raise RuntimeError("Compact index lease was released twice.")
        # Re-recording on one stream covers all its earlier consumers. Abort
        # cleanup on another stream gets its own fence, never a host wait.
        key = stream.cuda_stream
        event = self._buffer.events.get(key)
        if event is None:
            event = self._buffer.events[key] = torch.cuda.Event()
        event.record(stream)
        self._buffer.pending -= 1
        self._released = True


class CompactIndexPool:
    def __init__(self, device: torch.device, capacity: int, slots: int = 2):
        self.device = device
        self.capacity = max(1, capacity)
        self._buffers = [self._allocate(self.capacity) for _ in range(slots)]
        # Pre-touch pinned pages and exercise the H2D path during serving warmup.
        # Scheduler startup already synchronizes before accepting requests.
        for buffer in self._buffers:
            buffer.host.zero_()
            buffer.device.copy_(buffer.host, non_blocking=True)
            event = torch.cuda.Event()
            stream = torch.cuda.current_stream(device)
            event.record(stream)
            buffer.events[stream.cuda_stream] = event

    def _allocate(self, capacity: int) -> _Buffer:
        return _Buffer(
            torch.empty(capacity, dtype=torch.int64, device="cpu", pin_memory=True),
            torch.empty(capacity, dtype=torch.int64, device=self.device),
        )

    def pack(self, indices: list[tuple[torch.Tensor, torch.Tensor]]) -> list[CompactIndexLease]:
        if not indices:
            return []
        flat = [tensor for pair in indices for tensor in pair]
        if any(t.device.type != "cpu" or t.dtype != torch.int64 or t.ndim != 1 for t in flat):
            raise ValueError("Compact indices must be one-dimensional CPU int64 tensors.")
        size = sum(t.numel() for t in flat)
        buffer = next((b for b in self._buffers if len(b.host) >= size and b.available()), None)
        if buffer is None:
            # Raw occurrence streams can exceed the fixed decode-table width.
            # Grow before forward; never wait for an in-flight slot to become free.
            self.capacity = max(self.capacity, 1 << max(0, size - 1).bit_length())
            buffer = self._allocate(self.capacity)
            self._buffers = [b for b in self._buffers
                             if len(b.host) >= self.capacity or not b.available()]
            self._buffers.append(buffer)
        # Keep completed event objects for re-recording on the next use; no
        # per-request event destruction/reallocation in the steady-state path.
        offset = 0
        leases = []
        for keep, dropped in indices:
            end_keep = offset + keep.numel()
            end = end_keep + dropped.numel()
            buffer.host[offset:end_keep].copy_(keep)
            buffer.host[end_keep:end].copy_(dropped)
            leases.append(CompactIndexLease(buffer.device[offset:end_keep],
                                            buffer.device[end_keep:end], buffer))
            offset = end
        buffer.pending = len(leases)
        # One pinned H2D per physical prefill batch, on the current scheduler
        # stream. The existing engine.wait_stream supplies the forward dependency.
        buffer.device[:size].copy_(buffer.host[:size], non_blocking=True)
        return leases

    def clear_after_synchronize(self) -> None:
        """Shutdown only: caller has synchronized the device, including consumers."""
        self._buffers.clear()
