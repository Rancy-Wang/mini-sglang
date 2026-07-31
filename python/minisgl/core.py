from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch


def build_context_visibility_mask_reference(
    full_token_visible_until: torch.Tensor,
    *,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the exact dense token-position Drop mask used as the CPU oracle."""

    if full_token_visible_until.ndim != 1:
        raise ValueError("full_token_visible_until must be one-dimensional.")
    device = full_token_visible_until.device
    if query_positions is None:
        query_positions = torch.arange(
            len(full_token_visible_until), dtype=torch.int64, device=device
        )
    if key_positions is None:
        key_positions = torch.arange(
            len(full_token_visible_until), dtype=torch.int64, device=device
        )
    if query_positions.ndim != 1 or key_positions.ndim != 1:
        raise ValueError("query_positions and key_positions must be one-dimensional.")
    query_positions = query_positions.to(dtype=torch.int64, device=device)
    key_positions = key_positions.to(dtype=torch.int64, device=device)
    for name, positions in (("query_positions", query_positions), ("key_positions", key_positions)):
        if len(positions) > 0 and (
            bool(torch.any(positions < 0).item())
            or bool(torch.any(positions >= len(full_token_visible_until)).item())
        ):
            raise ValueError(f"{name} contains an out-of-range full-token position.")

    causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    visible_until = full_token_visible_until[key_positions]
    visible = query_positions.unsqueeze(1) < visible_until.unsqueeze(0)
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
    full_token_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    use_context_mask: bool = False
    radix_key_virtual_mask: torch.Tensor | None = None
    radix_key_to_token: torch.Tensor | None = None
    radix_token_to_key: torch.Tensor | None = None
    radix_commit_key_len: int | None = None
    radix_marker_ids: tuple[int, ...] = ()

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
        radix_layout = (
            self.radix_key_virtual_mask,
            self.radix_key_to_token,
            self.radix_token_to_key,
        )
        if any(tensor is not None for tensor in radix_layout):
            if not all(tensor is not None for tensor in radix_layout):
                raise ValueError("Delta-marker Radix layout must be provided as one complete set.")
            virtual_mask, key_to_token, token_to_key = radix_layout
            assert virtual_mask is not None
            assert key_to_token is not None
            assert token_to_key is not None
            for tensor in radix_layout:
                assert tensor is not None and tensor.is_cpu and tensor.ndim == 1
            if virtual_mask.dtype != torch.bool:
                raise ValueError("radix_key_virtual_mask must use torch.bool.")
            if key_to_token.dtype != torch.int64 or token_to_key.dtype != torch.int64:
                raise ValueError("Delta-marker Radix mappings must use torch.int64.")
            if len(virtual_mask) != len(self.radix_match_ids) or len(key_to_token) != len(
                self.radix_match_ids
            ):
                raise ValueError("Delta-marker key-axis tensors must match radix_match_ids.")
            if len(token_to_key) == 0 and len(self.input_ids) > 0:
                raise ValueError("Delta-marker token_to_key must cover the input token stream.")
            if bool(torch.any(key_to_token[virtual_mask] != -1).item()):
                raise ValueError("Virtual Radix keys must map to token -1.")
            real_key_positions = torch.nonzero(~virtual_mask, as_tuple=False).view(-1)
            if not torch.equal(
                key_to_token[real_key_positions],
                torch.arange(len(token_to_key), dtype=torch.int64, device="cpu"),
            ):
                raise ValueError("Real Radix keys must preserve full-token order.")
            if not torch.equal(token_to_key, real_key_positions):
                raise ValueError("radix_token_to_key is not the inverse key mapping.")
            virtual_keys = self.radix_match_ids[virtual_mask].tolist()
            if virtual_keys != list(self.radix_marker_ids):
                raise ValueError("radix_marker_ids do not match the virtual Radix key stream.")
        elif self.radix_marker_ids:
            raise ValueError("radix_marker_ids require a delta-marker Radix layout.")
        if self.radix_commit_key_len is not None:
            if self.radix_key_virtual_mask is None:
                raise ValueError(
                    "radix_commit_key_len requires a delta-marker Radix layout."
                )
            if not 0 <= self.radix_commit_key_len <= len(self.radix_match_ids):
                raise ValueError("radix_commit_key_len is outside the Radix key stream.")
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        assert 0 <= self.initial_active_cached_len <= self.cached_len
        assert self.true_seq_len >= int(self.true_positions[self.device_len - 1].item()) + 1

        context_tensors = (
            self.full_input_ids,
            self.full_token_visible_until,
            self.full_keep_mask,
        )
        if any(tensor is not None for tensor in context_tensors):
            if not all(tensor is not None for tensor in context_tensors):
                raise ValueError("Context-mask metadata must be provided as one complete set.")
            assert self.full_input_ids is not None
            assert self.full_token_visible_until is not None
            assert self.full_keep_mask is not None
            for tensor in context_tensors:
                assert tensor is not None and tensor.is_cpu and tensor.ndim == 1
                if tensor.dtype != torch.int32:
                    raise ValueError("Context-mask metadata tensors must use torch.int32.")
            full_len = len(self.full_input_ids)
            if not len(self.full_token_visible_until) == len(self.full_keep_mask) == full_len:
                raise ValueError("Full context-mask tensors and Radix keys must have equal lengths.")
            if self.radix_token_to_key is None:
                if len(self.radix_match_ids) != full_len:
                    raise ValueError(
                        "Full context-mask tensors and Radix keys must have equal lengths."
                    )
            elif len(self.radix_token_to_key) != full_len:
                raise ValueError(
                    "Full context-mask tensors and delta-marker token mapping must have equal lengths."
                )
            if not torch.equal(
                self.input_ids,
                self.full_input_ids[self.true_positions.to(dtype=torch.int64)],
            ):
                raise ValueError("Active input_ids do not match full_input_ids at true_positions.")
            key_positions = torch.arange(full_len, dtype=torch.int32, device="cpu")
            if bool(torch.any(self.full_token_visible_until <= key_positions).item()):
                raise ValueError(
                    "A token cannot become invisible before it has been computed."
                )
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
        if self.radix_key_virtual_mask is not None:
            assert self.radix_key_to_token is not None
            assert self.radix_token_to_key is not None
            token_pos = len(self.radix_token_to_key)
            key_pos = len(self.radix_match_ids) - 1
            self.radix_key_virtual_mask = torch.cat(
                [
                    self.radix_key_virtual_mask,
                    torch.tensor([False], dtype=torch.bool, device="cpu"),
                ]
            )
            self.radix_key_to_token = torch.cat(
                [
                    self.radix_key_to_token,
                    torch.tensor([token_pos], dtype=torch.int64, device="cpu"),
                ]
            )
            self.radix_token_to_key = torch.cat(
                [
                    self.radix_token_to_key,
                    torch.tensor([key_pos], dtype=torch.int64, device="cpu"),
                ]
            )

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
