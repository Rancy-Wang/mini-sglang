from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass(frozen=True)
class ContextAttentionSegment:
    """One causal-attention segment over a compacted active-KV view."""

    query_start: int
    query_end: int
    key_positions: torch.Tensor


def build_context_attention_segments(
    full_token_visible_until: torch.Tensor,
    *,
    query_start: int,
    query_length: int,
    key_length: int,
) -> tuple[ContextAttentionSegment, ...]:
    """Compile the exact token-position Drop mask into causal KV segments.

    Query offsets in the result are relative to the input Q tensor. Key
    positions stay on the full, absolute token axis so callers can gather the
    corresponding page-table entries without changing RoPE positions.
    """

    if full_token_visible_until.ndim != 1 or not full_token_visible_until.is_cpu:
        raise ValueError("full_token_visible_until must be a one-dimensional CPU tensor.")
    if query_start < 0 or query_length < 1 or key_length < 1:
        raise ValueError("Context attention lengths must be positive and query_start non-negative.")
    query_end = query_start + query_length
    if query_end > key_length or key_length > len(full_token_visible_until):
        raise ValueError("Context attention query/key bounds exceed the full token stream.")

    visible_until = full_token_visible_until[:key_length].to(dtype=torch.int64)
    key_positions = torch.arange(key_length, dtype=torch.int64, device="cpu")
    if bool(torch.any(visible_until <= key_positions).item()):
        raise ValueError("A token cannot become invisible before it has been computed.")

    boundaries = {query_start, query_end}
    boundaries.update(
        int(position)
        for position in visible_until.tolist()
        if query_start < int(position) < query_end
    )
    ordered = sorted(boundaries)

    segments = []
    for absolute_start, absolute_end in zip(ordered, ordered[1:]):
        prefix_positions = key_positions[:absolute_start]
        active_prefix = prefix_positions[visible_until[:absolute_start] > absolute_start]
        local_queries = key_positions[absolute_start:absolute_end]
        segments.append(
            ContextAttentionSegment(
                query_start=absolute_start - query_start,
                query_end=absolute_end - query_start,
                key_positions=torch.cat((active_prefix, local_queries)),
            )
        )
    return tuple(segments)


@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        sinks: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


class HybridBackend(BaseAttnBackend):
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        sinks: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        if sinks is None and sliding_window is None:
            return backend.forward(q, k, v, layer_id, batch)
        return backend.forward(
            q,
            k,
            v,
            layer_id,
            batch,
            sinks=sinks,
            sliding_window=sliding_window,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
