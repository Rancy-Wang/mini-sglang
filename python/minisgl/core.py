from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    true_positions: torch.Tensor  # cpu tensor, absolute position for each input token
    radix_input_ids: torch.Tensor  # cpu tensor, int64 encoded key ids for radix
    radix_match_ids: torch.Tensor  # cpu tensor, full int64 encoded key ids for radix matching
    initial_full_match_indices: torch.Tensor  # tensor for initially matched full-prefix page indices
    initial_active_cached_len: int
    true_seq_len: int
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    stop: List[str] | None = None
    stop_token_seqs: List[List[int]] | None = None
    prefix_keep_mask: torch.Tensor | None = None  # cpu tensor for full->active prefix filtering
    is_warmup: bool = False
    cache_hit_ratio: float = 1.0

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        assert self.true_positions.is_cpu
        assert self.radix_input_ids.is_cpu
        assert self.radix_match_ids.is_cpu
        if self.prefix_keep_mask is not None:
            assert self.prefix_keep_mask.is_cpu
        assert len(self.input_ids) == len(self.true_positions)
        assert len(self.input_ids) == len(self.radix_input_ids)
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        assert 0 <= self.initial_active_cached_len <= self.cached_len
        assert self.true_seq_len >= int(self.true_positions[self.device_len - 1].item()) + 1

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        # `complete_one` is called immediately after forward.
        # Update position metadata here so both overlap and normal loops
        # can schedule the next batch with consistent absolute positions.
        self.cached_len = self.device_len
        self.device_len += 1
        self.true_seq_len += 1
        next_pos = torch.tensor([self.true_seq_len - 1], dtype=torch.int32, device="cpu")
        self.true_positions = torch.cat([self.true_positions, next_pos])

    def append_host(self, next_token: torch.Tensor) -> None:
        # Position is already appended in `complete_one`.
        self.input_ids = torch.cat([self.input_ids, next_token])
        next_token_key = next_token.to(dtype=torch.int64, device="cpu")
        self.radix_input_ids = torch.cat([self.radix_input_ids, next_token_key])
        self.radix_match_ids = torch.cat([self.radix_match_ids, next_token_key])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def match_stop(self) -> tuple[bool, str | None]:
        if not self.stop_token_seqs:
            return False, None
        input_ids = self.input_ids.tolist()
        for idx, stop_seq in enumerate(self.stop_token_seqs):
            if len(stop_seq) == 0 or len(stop_seq) > len(input_ids):
                continue
            if input_ids[-len(stop_seq) :] == stop_seq:
                if self.stop is not None and idx < len(self.stop):
                    return True, self.stop[idx]
                return True, None
        return False, None

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"true_seq_len={self.true_seq_len}, max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
