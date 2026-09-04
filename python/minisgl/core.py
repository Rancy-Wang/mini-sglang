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
    seed: int | None = None

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    true_positions: torch.Tensor  # cpu tensor, current KV position for each active token
    raw_positions: torch.Tensor  # cpu tensor, immutable full-token position
    radix_input_ids: torch.Tensor  # cpu tensor, int64 encoded key ids for radix
    radix_match_ids: torch.Tensor  # cpu tensor, full int64 encoded key ids for radix matching
    initial_full_match_indices: (
        torch.Tensor
    )  # tensor for initially matched full-prefix page indices
    initial_active_cached_len: int
    true_seq_len: int
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    prompt_tokens: int = 0
    stop: List[str] | None = None
    stop_token_seqs: List[List[int]] | None = None
    prefix_keep_mask: torch.Tensor | None = None  # cpu tensor for full->active prefix filtering
    is_warmup: bool = False
    cache_reuse_ratio: float = 1.0
    radix_cached_tokens: int = 0
    usage_cached_tokens: int | None = None
    drop_skipped_tokens: int = 0
    full_input_ids: torch.Tensor | None = None
    full_token_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    use_context_mask: bool = False
    context_compact_stream: bool = False
    context_post_prefill_keep_mask: torch.Tensor | None = None
    radix_key_virtual_mask: torch.Tensor | None = None
    radix_key_to_token: torch.Tensor | None = None
    radix_token_to_key: torch.Tensor | None = None
    radix_commit_key_len: int | None = None
    radix_positions: torch.Tensor | None = None
    radix_repos_info: torch.Tensor | None = None
    radix_next_position: int | None = None
    radix_current_reposition: int = -1
    retry_transformed_mask: torch.Tensor | None = None
    inactive_cached_positions: torch.Tensor | None = None
    inactive_cached_pages: torch.Tensor | None = None
    tokenize_invocations: int = 1
    radix_compile_ns: int = 0
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    reposition_transition_count: int = 0
    reposition_h2d_bytes: int = 0
    reposition_d2h_bytes: int = 0

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        assert self.true_positions.is_cpu
        if self.true_positions.ndim != 1:
            raise ValueError("true_positions must be one-dimensional.")
        if len(self.true_positions) > 1 and bool(
            torch.any(self.true_positions[1:] <= self.true_positions[:-1]).item()
        ):
            raise ValueError("true_positions must be strictly increasing.")
        if self.raw_positions.ndim != 1 or not self.raw_positions.is_cpu:
            raise ValueError("raw_positions must be a one-dimensional CPU tensor.")
        if len(self.raw_positions) > 1 and bool(
            torch.any(self.raw_positions[1:] <= self.raw_positions[:-1]).item()
        ):
            raise ValueError("raw_positions must be strictly increasing.")
        assert self.radix_input_ids.is_cpu
        assert self.radix_match_ids.is_cpu
        if self.use_context_mask and not (
            self.is_warmup or self.context_post_prefill_keep_mask is not None
        ):
            raise ValueError(
                "Context-mask Prefill is restricted to internal warmup or Reposition requests."
            )
        if self.context_post_prefill_keep_mask is not None:
            keep_mask = self.context_post_prefill_keep_mask
            if (
                not keep_mask.is_cpu
                or keep_mask.ndim != 1
                or keep_mask.dtype not in (torch.bool, torch.int32)
            ):
                raise ValueError(
                    "context_post_prefill_keep_mask must be a CPU bool or int32 vector."
                )
            if len(self.raw_positions) == 0 or int(self.raw_positions[-1]) >= len(keep_mask):
                raise ValueError(
                    "context_post_prefill_keep_mask does not cover the compact raw stream."
                )
        if self.prefix_keep_mask is not None:
            assert self.prefix_keep_mask.is_cpu
        assert len(self.input_ids) == len(self.true_positions)
        assert len(self.input_ids) == len(self.raw_positions)
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
            if self.radix_match_ids.ndim == 2:
                from minisgl.kernel.radix_reposition import (
                    validate_radix_reposition_records,
                )

                validate_radix_reposition_records(
                    self.radix_match_ids,
                    token_count=len(token_to_key),
                    require_materialized=True,
                )
                expected_virtual = self.radix_match_ids[:, 0] != 0
                if not torch.equal(virtual_mask, expected_virtual):
                    raise ValueError("Structured Radix virtual kinds disagree with the mask.")
        if self.radix_commit_key_len is not None:
            if self.radix_key_virtual_mask is None:
                raise ValueError("radix_commit_key_len requires a delta-marker Radix layout.")
            if not 0 <= self.radix_commit_key_len <= len(self.radix_match_ids):
                raise ValueError("radix_commit_key_len is outside the Radix key stream.")
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        assert 0 <= self.initial_active_cached_len <= self.cached_len
        if self.retry_transformed_mask is not None:
            if (
                not self.retry_transformed_mask.is_cpu
                or self.retry_transformed_mask.dtype != torch.bool
                or self.retry_transformed_mask.ndim != 1
                or len(self.retry_transformed_mask) != self.initial_active_cached_len
            ):
                raise ValueError(
                    "retry_transformed_mask must be a CPU bool vector covering the initial "
                    "active cache prefix."
                )
        inactive_retry = (
            self.inactive_cached_positions,
            self.inactive_cached_pages,
        )
        if any(tensor is not None for tensor in inactive_retry):
            if not all(tensor is not None for tensor in inactive_retry):
                raise ValueError("Inactive cached positions and pages must be provided together.")
            inactive_positions, inactive_pages = inactive_retry
            assert inactive_positions is not None
            assert inactive_pages is not None
            if (
                inactive_positions.device.type != "cpu"
                or inactive_positions.dtype != torch.int64
                or inactive_positions.ndim != 1
                or inactive_pages.dtype != torch.int32
                or inactive_pages.ndim != 1
                or len(inactive_positions) != len(inactive_pages)
            ):
                raise ValueError("Inactive cached metadata has an invalid layout.")
        if self.radix_cached_tokens < 0:
            raise ValueError("radix_cached_tokens must be non-negative.")
        if self.usage_cached_tokens is not None:
            self._validate_context_cache_usage(self.usage_cached_tokens)
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
                raise ValueError(
                    "Full context-mask tensors and Radix keys must have equal lengths."
                )
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
                self.full_input_ids[self.raw_positions.to(dtype=torch.int64)],
            ):
                raise ValueError("Active input_ids do not match full_input_ids at raw_positions.")
            key_positions = torch.arange(full_len, dtype=torch.int32, device="cpu")
            if bool(torch.any(self.full_token_visible_until <= key_positions).item()):
                raise ValueError("A token cannot become invisible before it has been computed.")
        if self.use_context_mask and not all(tensor is not None for tensor in context_tensors):
            raise ValueError("Context-mask Prefill requires complete context metadata.")

    def _validate_context_cache_usage(self, cached_tokens: int) -> None:
        if not 0 <= cached_tokens <= self.radix_cached_tokens:
            raise ValueError(
                "Attention cache usage must satisfy 0 <= cached <= Radix-matched, got "
                f"{cached_tokens}, {self.radix_cached_tokens}."
            )

    def record_context_cache_usage(self, cached_tokens: int) -> None:
        """Record distinct Radix-hit tokens that enter full Context attention."""

        self._validate_context_cache_usage(cached_tokens)
        if self.usage_cached_tokens is not None:
            if self.usage_cached_tokens != cached_tokens:
                raise RuntimeError(
                    "Context cache usage changed after it was recorded: "
                    f"{self.usage_cached_tokens} != {cached_tokens}."
                )
            return
        self.usage_cached_tokens = cached_tokens
        self.drop_skipped_tokens = self.radix_cached_tokens - cached_tokens

    @property
    def reported_cached_tokens(self) -> int:
        if self.usage_cached_tokens is None:
            raise RuntimeError("Context cache usage was not recorded before reporting.")
        return self.usage_cached_tokens

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
        position = (
            self.radix_next_position if self.radix_next_position is not None else self.true_seq_len
        )
        next_pos = torch.tensor([position], dtype=torch.int32, device="cpu")
        self.true_positions = torch.cat([self.true_positions, next_pos])
        self.true_seq_len = max(self.true_seq_len, position + 1)
        if self.radix_next_position is not None:
            self.radix_next_position += 1
        if self.radix_token_to_key is not None:
            pending_host_tokens = len(self.raw_positions) - len(self.input_ids)
            if pending_host_tokens < 0:
                raise RuntimeError("Raw positions fell behind the host token stream.")
            raw_position = len(self.radix_token_to_key) + pending_host_tokens
        else:
            raw_position = int(self.raw_positions[-1]) + 1
        self.raw_positions = torch.cat(
            [
                self.raw_positions,
                torch.tensor([raw_position], dtype=torch.int32, device="cpu"),
            ]
        )

    def append_host(self, next_token: torch.Tensor) -> None:
        # Overlap scheduling can finish the following decode before this sampled
        # token reaches the CPU. Pair it with its own queued position instead of
        # the newest (possibly one-token-ahead) position.
        host_token_index = len(self.input_ids)
        if host_token_index >= len(self.true_positions) or host_token_index >= len(
            self.raw_positions
        ):
            raise RuntimeError("A sampled token arrived before its position metadata.")
        host_true_position = int(self.true_positions[host_token_index])
        host_raw_position = int(self.raw_positions[host_token_index])
        if self.radix_token_to_key is not None and host_raw_position != len(
            self.radix_token_to_key
        ):
            raise RuntimeError("Generated-token raw positions are not contiguous.")
        self.input_ids = torch.cat([self.input_ids, next_token])
        if self.radix_match_ids.ndim == 2:
            next_token_key = torch.tensor(
                [
                    [
                        0,
                        int(next_token[0]),
                        self.radix_current_reposition,
                        host_true_position,
                    ]
                ],
                dtype=torch.int32,
                device="cpu",
            )
        else:
            next_token_key = next_token.to(dtype=torch.int64, device="cpu")
        self.radix_input_ids = torch.cat([self.radix_input_ids, next_token_key])
        self.radix_match_ids = torch.cat([self.radix_match_ids, next_token_key])
        if self.radix_positions is not None:
            self.radix_positions = torch.cat(
                [
                    self.radix_positions,
                    torch.tensor([host_true_position], dtype=torch.int32, device="cpu"),
                ]
            )
        if self.radix_repos_info is not None:
            self.radix_repos_info = torch.cat(
                [
                    self.radix_repos_info,
                    torch.tensor([self.radix_current_reposition], dtype=torch.int32, device="cpu"),
                ]
            )
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

    @property
    def completion_tokens(self) -> int:
        active_prompt_tokens = self.max_device_len - self.output_len
        return self.device_len - active_prompt_tokens

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
