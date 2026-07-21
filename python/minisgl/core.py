from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch


def build_context_visibility_mask_reference(
    full_kv_owner: torch.Tensor,
    full_query_epoch: torch.Tensor,
    drop_visible_until: torch.Tensor,
    *,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the exact dense Drop Message mask used as the CPU correctness oracle."""

    tensors = (full_kv_owner, full_query_epoch, drop_visible_until)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("Context-mask metadata tensors must be one-dimensional.")
    if len(full_kv_owner) != len(full_query_epoch):
        raise ValueError("full_kv_owner and full_query_epoch must have the same length.")
    if len(full_query_epoch) > 1 and bool(
        torch.any(full_query_epoch[1:] < full_query_epoch[:-1]).item()
    ):
        raise ValueError("full_query_epoch must be monotonically non-decreasing.")
    if len(full_kv_owner) > 0:
        if bool(torch.any(full_kv_owner < 0).item()) or bool(
            torch.any(full_kv_owner >= len(drop_visible_until)).item()
        ):
            raise ValueError("full_kv_owner contains an out-of-range message ID.")

    device = full_kv_owner.device
    if query_positions is None:
        query_positions = torch.arange(len(full_query_epoch), dtype=torch.int64, device=device)
    if key_positions is None:
        key_positions = torch.arange(len(full_kv_owner), dtype=torch.int64, device=device)
    if query_positions.ndim != 1 or key_positions.ndim != 1:
        raise ValueError("query_positions and key_positions must be one-dimensional.")
    query_positions = query_positions.to(dtype=torch.int64, device=device)
    key_positions = key_positions.to(dtype=torch.int64, device=device)
    for name, positions in (("query_positions", query_positions), ("key_positions", key_positions)):
        if len(positions) > 0 and (
            bool(torch.any(positions < 0).item())
            or bool(torch.any(positions >= len(full_kv_owner)).item())
        ):
            raise ValueError(f"{name} contains an out-of-range full-token position.")

    query_epoch = full_query_epoch[query_positions]
    key_owner = full_kv_owner[key_positions]
    visible_until = drop_visible_until[key_owner.to(dtype=torch.int64)]
    causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    visible = query_epoch.unsqueeze(1) <= visible_until.unsqueeze(0)
    return causal & visible

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
    full_input_ids: torch.Tensor | None = None
    full_kv_owner: torch.Tensor | None = None
    full_query_epoch: torch.Tensor | None = None
    drop_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    use_context_mask: bool = False

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        assert self.true_positions.is_cpu
        assert self.radix_input_ids.is_cpu
        assert self.radix_match_ids.is_cpu
        if self.use_context_mask and not self.is_warmup:
            raise ValueError("Context-mask Prefill is restricted to internal warmup requests.")
        if self.prefix_keep_mask is not None:
            assert self.prefix_keep_mask.is_cpu
        assert len(self.input_ids) == len(self.true_positions)
        assert len(self.input_ids) == len(self.radix_input_ids)
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        assert 0 <= self.initial_active_cached_len <= self.cached_len
        assert self.true_seq_len >= int(self.true_positions[self.device_len - 1].item()) + 1

        context_tensors = (
            self.full_input_ids,
            self.full_kv_owner,
            self.full_query_epoch,
            self.drop_visible_until,
            self.full_keep_mask,
        )
        if any(tensor is not None for tensor in context_tensors):
            if not all(tensor is not None for tensor in context_tensors):
                raise ValueError("Context-mask metadata must be provided as one complete set.")
            assert self.full_input_ids is not None
            assert self.full_kv_owner is not None
            assert self.full_query_epoch is not None
            assert self.drop_visible_until is not None
            assert self.full_keep_mask is not None
            for tensor in context_tensors:
                assert tensor is not None and tensor.is_cpu and tensor.ndim == 1
                if tensor.dtype != torch.int32:
                    raise ValueError("Context-mask metadata tensors must use torch.int32.")
            full_len = len(self.full_input_ids)
            if not (
                len(self.full_kv_owner)
                == len(self.full_query_epoch)
                == len(self.full_keep_mask)
                == len(self.radix_match_ids)
                == full_len
            ):
                raise ValueError("Full context-mask tensors and Radix keys must have equal lengths.")
            if not torch.equal(
                self.input_ids,
                self.full_input_ids[self.true_positions.to(dtype=torch.int64)],
            ):
                raise ValueError("Active input_ids do not match full_input_ids at true_positions.")
            if len(self.full_query_epoch) > 1 and bool(
                torch.any(self.full_query_epoch[1:] < self.full_query_epoch[:-1]).item()
            ):
                raise ValueError("full_query_epoch must be monotonically non-decreasing.")
            if bool(torch.any(self.full_kv_owner < 0).item()) or bool(
                torch.any(self.full_kv_owner >= len(self.drop_visible_until)).item()
            ):
                raise ValueError("full_kv_owner contains an out-of-range message ID.")
        if self.use_context_mask and not all(tensor is not None for tensor in context_tensors):
            raise ValueError("Context-mask Prefill requires complete context metadata.")

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
