from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

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
    tool_grammar: dict[str, Any] | None = None

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


def validate_occurrence_positions(req, model_limit: int, rope_limit: int) -> int:
    """Check execution positions once at admission, independently of raw length."""
    positions = req.occurrence_positions
    terminal = req.occurrence_terminal_indices
    keep = req.full_keep_mask
    if keep is None:
        raise ValueError("Occurrence position validation requires a final active mask.")
    execution_positions: torch.Tensor
    if positions is not None and terminal is not None:
        if not positions.is_cpu or positions.ndim != 1 or len(positions) == 0:
            raise ValueError("Occurrence positions must be a nonempty CPU vector.")
        if len(terminal) != len(req.input_ids) or len(keep) != len(terminal):
            raise ValueError("Occurrence terminal and active maps must cover raw input.")
        if int(terminal.min()) < 0 or int(terminal.max()) >= len(positions):
            raise ValueError("Occurrence terminal references are out of range.")
        terminal_positions = positions[terminal.to(torch.int64)]
        execution_positions = positions
    else:
        compact = (
            getattr(req, "occurrence_layout_birth_positions", None),
            getattr(req, "occurrence_layout_birth_stages", None),
            getattr(req, "occurrence_layout_transition_offsets", None),
            getattr(req, "occurrence_layout_transition_raw_tokens", None),
            getattr(req, "occurrence_layout_transition_old_positions", None),
            getattr(req, "occurrence_layout_transition_new_positions", None),
        )
        if not all(tensor is not None for tensor in compact):
            raise ValueError("Occurrence position validation requires a complete compact layout.")
        for tensor in compact:
            assert tensor is not None
            if not tensor.is_cpu or tensor.dtype != torch.int32 or tensor.ndim != 1:
                raise ValueError("Compact occurrence layout tensors must be CPU int32 vectors.")
        birth_positions, birth_stages, offsets, raw_tokens, old_positions, new_positions = compact
        assert birth_positions is not None
        assert birth_stages is not None
        assert offsets is not None
        assert raw_tokens is not None
        assert old_positions is not None
        assert new_positions is not None
        token_count = len(req.input_ids)
        if len(birth_positions) != token_count or len(birth_stages) != token_count:
            raise ValueError("Compact occurrence birth metadata must cover raw input.")
        if len(keep) != token_count:
            raise ValueError("Occurrence active mask must cover raw input.")
        if len(offsets) < 2 or int(offsets[0]) != 0 or bool(
            torch.any(offsets[1:] < offsets[:-1]).item()
        ):
            raise ValueError("Compact occurrence transition offsets are invalid.")
        transition_count = int(offsets[-1])
        if not transition_count == len(raw_tokens) == len(old_positions) == len(new_positions):
            raise ValueError("Compact occurrence transition arrays have different lengths.")
        if bool(torch.any(raw_tokens < 0).item()) or bool(
            torch.any(raw_tokens >= token_count).item()
        ):
            raise ValueError("Compact occurrence transitions reference invalid raw tokens.")
        terminal_positions = req.radix_positions
        if (
            terminal_positions is None
            or not terminal_positions.is_cpu
            or terminal_positions.dtype != torch.int32
            or terminal_positions.ndim != 1
            or len(terminal_positions) != token_count
        ):
            raise ValueError("Compact occurrence layout requires terminal Radix positions.")
        execution_positions = torch.cat(
            (birth_positions, old_positions, new_positions, terminal_positions)
        )
        if len(execution_positions) == 0:
            raise ValueError("Compact occurrence layout has no execution positions.")
    if int(execution_positions.min()) < 0 or int(execution_positions.max()) >= min(
        model_limit, rope_limit
    ):
        raise ValueError("An occurrence execution position exceeds the model/RoPE limit.")
    active = keep.to(torch.bool)
    active_count = int(torch.count_nonzero(active))
    if active_count == 0:
        raise ValueError("Occurrence prompt must retain an active token.")
    active_positions = terminal_positions[active]
    next_position = req.radix_next_position
    if next_position is None or next_position <= int(active_positions.max()):
        raise ValueError("Next position does not cover active terminal positions.")
    return active_count


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
    usage_cached_tokens: int | None = None  # all reused KV, including Retry RoPE
    usage_repos_tokens: int | None = None  # frozen before post-Prefill compaction
    context_usage_cached_positions: torch.Tensor | None = None
    drop_skipped_tokens: int = 0
    full_input_ids: torch.Tensor | None = None
    full_token_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    use_context_mask: bool = False
    context_compact_stream: bool = False
    context_post_prefill_keep_mask: torch.Tensor | None = None
    context_decode_keep_mask: torch.Tensor | None = None
    context_decode_keep_indices: torch.Tensor | None = None
    context_decode_dropped_owned_indices: torch.Tensor | None = None
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
    reposition_execution_mode: Literal["staged", "paged-occurrence"] | None = None
    occurrence_raw_tokens: torch.Tensor | None = None
    occurrence_positions: torch.Tensor | None = None
    occurrence_birth_indices: torch.Tensor | None = None
    occurrence_terminal_indices: torch.Tensor | None = None
    occurrence_segment_query_starts: torch.Tensor | None = None
    occurrence_segment_query_ends: torch.Tensor | None = None
    occurrence_segment_key_offsets: torch.Tensor | None = None
    occurrence_segment_key_indices: torch.Tensor | None = None
    occurrence_pages: torch.Tensor | None = None
    occurrence_transient_pages: torch.Tensor | None = None
    occurrence_birth_pages: torch.Tensor | None = None
    occurrence_birth_owned_mask: torch.Tensor | None = None
    occurrence_transform_source_pages: torch.Tensor | None = None
    occurrence_transform_destination_pages: torch.Tensor | None = None
    occurrence_transform_position_pairs: torch.Tensor | None = None
    occurrence_terminal_owned_mask: torch.Tensor | None = None
    occurrence_initial_source_positions: torch.Tensor | None = None
    occurrence_exact_full_cached_len: int | None = None
    occurrence_same_position_retry_copy_count: int = 0
    occurrence_repositioned_cached_mask: torch.Tensor | None = None
    occurrence_inflight: bool = False
    occurrence_abort_deferred: bool = False
    occurrence_external_storage: bool = False
    _host_append_buffers: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        assert self.true_positions.is_cpu
        if self.true_positions.ndim != 1:
            raise ValueError("true_positions must be one-dimensional.")
        if (
            self.reposition_execution_mode != "paged-occurrence"
            and len(self.true_positions) > 1
            and bool(torch.any(self.true_positions[1:] <= self.true_positions[:-1]).item())
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
            raise ValueError("Context-mask Prefill requires warmup or a final active keep mask.")
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
            if self.usage_repos_tokens is None:
                if self.use_context_mask:
                    raise ValueError("Recorded Context usage requires a frozen Retry count.")
                self.usage_repos_tokens = (
                    int(self.retry_transformed_mask.sum().item())
                    if self.retry_transformed_mask is not None
                    else 0
                )
            if not 0 <= self.usage_repos_tokens <= self.usage_cached_tokens:
                raise ValueError("Retry usage exceeds the reused cache prefix.")
        occurrence_plan = (
            self.occurrence_raw_tokens,
            self.occurrence_positions,
            self.occurrence_birth_indices,
            self.occurrence_terminal_indices,
            self.occurrence_segment_query_starts,
            self.occurrence_segment_query_ends,
            self.occurrence_segment_key_offsets,
            self.occurrence_segment_key_indices,
        )
        if self.reposition_execution_mode == "paged-occurrence":
            if not all(tensor is not None for tensor in occurrence_plan):
                raise ValueError("Paged-occurrence Reposition requires a complete occurrence plan.")
            for tensor in occurrence_plan:
                assert tensor is not None
                if tensor.device.type != "cpu" or tensor.dtype != torch.int32:
                    raise ValueError("Occurrence plan tensors must be CPU int32 tensors.")
            assert self.occurrence_raw_tokens is not None
            assert self.occurrence_positions is not None
            assert self.occurrence_birth_indices is not None
            assert self.occurrence_terminal_indices is not None
            assert self.occurrence_segment_query_starts is not None
            assert self.occurrence_segment_query_ends is not None
            assert self.occurrence_segment_key_offsets is not None
            assert self.occurrence_segment_key_indices is not None
            occurrence_count = len(self.occurrence_raw_tokens)
            plan_token_count = len(self.occurrence_birth_indices)
            if occurrence_count < plan_token_count:
                raise ValueError("Occurrence plan has fewer occurrences than prompt tokens.")
            if len(self.occurrence_positions) != occurrence_count:
                raise ValueError("Occurrence token and position vectors have different lengths.")
            if len(self.occurrence_terminal_indices) != plan_token_count:
                raise ValueError("Occurrence birth/terminal maps must cover the prompt stream.")
            if len(self.input_ids) > plan_token_count:
                raise ValueError("Chunked occurrence input exceeds the full prompt plan.")
            if (
                len(self.occurrence_segment_query_starts) != len(self.occurrence_segment_query_ends)
                or len(self.occurrence_segment_key_offsets)
                != len(self.occurrence_segment_query_starts) + 1
            ):
                raise ValueError("Occurrence segment metadata has inconsistent lengths.")
            if int(self.occurrence_segment_key_offsets[-1]) != len(
                self.occurrence_segment_key_indices
            ):
                raise ValueError("Occurrence segment offsets do not cover their key indices.")
            if int(self.occurrence_segment_key_offsets[0]) != 0 or bool(
                torch.any(
                    self.occurrence_segment_key_offsets[1:]
                    < self.occurrence_segment_key_offsets[:-1]
                ).item()
            ):
                raise ValueError(
                    "Occurrence segment key offsets must start at zero and be monotonic."
                )
            if len(self.occurrence_segment_query_starts) == 0:
                raise ValueError("Occurrence plan must contain at least one query segment.")
            if (
                int(self.occurrence_segment_query_starts[0]) > self.cached_len
                or int(self.occurrence_segment_query_ends[-1]) < self.device_len
                or int(self.occurrence_segment_query_starts[0]) < 0
                or int(self.occurrence_segment_query_ends[-1]) > plan_token_count
                or bool(
                    torch.any(
                        self.occurrence_segment_query_starts[1:]
                        != self.occurrence_segment_query_ends[:-1]
                    ).item()
                )
                or bool(
                    torch.any(
                        self.occurrence_segment_query_ends <= self.occurrence_segment_query_starts
                    ).item()
                )
            ):
                raise ValueError(
                    "Occurrence query segments must contiguously cover the request extension."
                )
            if bool(torch.any(self.occurrence_raw_tokens < 0).item()) or bool(
                torch.any(self.occurrence_raw_tokens >= plan_token_count).item()
            ):
                raise ValueError("Occurrence raw-token indices are outside the prompt stream.")
            occurrence_refs = torch.cat(
                (
                    self.occurrence_birth_indices,
                    self.occurrence_terminal_indices,
                    self.occurrence_segment_key_indices,
                )
            )
            if bool(torch.any(occurrence_refs < 0).item()) or bool(
                torch.any(occurrence_refs >= occurrence_count).item()
            ):
                raise ValueError("Occurrence plan references an invalid occurrence index.")
            prompt_raw = torch.arange(plan_token_count, dtype=torch.int32, device="cpu")
            birth_refs = self.occurrence_birth_indices.to(torch.int64)
            terminal_refs = self.occurrence_terminal_indices.to(torch.int64)
            if not torch.equal(
                self.occurrence_raw_tokens[birth_refs], prompt_raw
            ) or not torch.equal(
                self.occurrence_positions[birth_refs[: len(self.input_ids)]],
                self.true_positions[: len(self.input_ids)].to(torch.int32),
            ):
                raise ValueError("Occurrence birth map disagrees with the prompt token stream.")
            if not torch.equal(self.occurrence_raw_tokens[terminal_refs], prompt_raw):
                raise ValueError("Occurrence terminal map does not cover raw tokens in order.")
            terminal_positions = self.occurrence_positions[terminal_refs]
            if self.radix_positions is not None and not torch.equal(
                terminal_positions, self.radix_positions.to(torch.int32)
            ):
                raise ValueError("Occurrence terminal positions disagree with Radix positions.")
            active_terminal_positions = terminal_positions
            if self.context_post_prefill_keep_mask is not None:
                final_keep = self.context_post_prefill_keep_mask
                if len(final_keep) != plan_token_count:
                    raise ValueError("Occurrence final keep mask must cover the raw prompt.")
                active_terminal_positions = terminal_positions[final_keep.to(torch.bool)]
            if len(active_terminal_positions) == 0:
                raise ValueError("Occurrence prompt must retain an active token.")
            if self.true_seq_len < int(torch.max(active_terminal_positions).item()) + 1:
                raise ValueError("true_seq_len does not cover terminal occurrence positions.")
            if self.occurrence_terminal_owned_mask is not None and (
                not self.occurrence_terminal_owned_mask.is_cpu
                or self.occurrence_terminal_owned_mask.dtype != torch.bool
                or self.occurrence_terminal_owned_mask.ndim != 1
                or len(self.occurrence_terminal_owned_mask) != plan_token_count
            ):
                raise ValueError(
                    "Occurrence-owned mask must be a CPU bool vector covering the prompt."
                )
            if self.occurrence_birth_pages is None or self.occurrence_birth_owned_mask is None:
                raise ValueError("Paged-occurrence requires canonical birth page ownership.")
            if (
                self.occurrence_birth_pages.ndim != 1
                or len(self.occurrence_birth_pages) != plan_token_count
            ):
                raise ValueError("Occurrence birth pages must cover the full prompt.")
            if (
                not self.occurrence_birth_owned_mask.is_cpu
                or self.occurrence_birth_owned_mask.dtype != torch.bool
                or self.occurrence_birth_owned_mask.ndim != 1
                or len(self.occurrence_birth_owned_mask) != plan_token_count
            ):
                raise ValueError("Occurrence birth ownership must cover the full prompt.")
            if self.occurrence_initial_source_positions is not None and (
                not self.occurrence_initial_source_positions.is_cpu
                or self.occurrence_initial_source_positions.dtype != torch.int32
                or self.occurrence_initial_source_positions.ndim != 1
                or len(self.occurrence_initial_source_positions) != self.initial_active_cached_len
            ):
                raise ValueError("Occurrence source positions must cover the initial cache hits.")
            if self.occurrence_exact_full_cached_len is not None and not (
                0 <= self.occurrence_exact_full_cached_len <= self.initial_active_cached_len
            ):
                raise ValueError("Occurrence exact prefix must lie within its source prefix.")
            if self.occurrence_same_position_retry_copy_count < 0:
                raise ValueError("Occurrence Retry copy count must be non-negative.")
            if self.occurrence_repositioned_cached_mask is not None and (
                not self.occurrence_repositioned_cached_mask.is_cpu
                or self.occurrence_repositioned_cached_mask.dtype != torch.bool
                or self.occurrence_repositioned_cached_mask.ndim != 1
                or len(self.occurrence_repositioned_cached_mask) != self.initial_active_cached_len
            ):
                raise ValueError(
                    "Occurrence repositioned-cache mask must cover the initial cache hits."
                )
            transforms = (
                self.occurrence_transform_source_pages,
                self.occurrence_transform_destination_pages,
                self.occurrence_transform_position_pairs,
            )
            if any(tensor is not None for tensor in transforms):
                if not all(tensor is not None for tensor in transforms):
                    raise ValueError("Occurrence layer transform metadata must be complete.")
                source_pages, destination_pages, position_pairs = transforms
                assert source_pages is not None
                assert destination_pages is not None
                assert position_pairs is not None
                if (
                    source_pages.dtype != torch.int32
                    or destination_pages.dtype != torch.int32
                    or position_pairs.dtype != torch.int32
                    or source_pages.ndim != 1
                    or destination_pages.ndim != 1
                    or position_pairs.ndim != 2
                    or position_pairs.shape[1] != 2
                    or len(source_pages) != len(destination_pages)
                    or len(source_pages) != len(position_pairs)
                ):
                    raise ValueError("Occurrence layer transform metadata has invalid shape/dtype.")
        elif any(tensor is not None for tensor in occurrence_plan):
            raise ValueError("Occurrence metadata requires paged-occurrence execution mode.")
        else:
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

    def record_context_cache_usage(
        self, cached_tokens: int, cached_positions: torch.Tensor | None = None
    ) -> None:
        """Record distinct Radix-hit tokens that enter full Context attention."""

        self._validate_context_cache_usage(cached_tokens)
        if cached_positions is None:
            if self.context_usage_cached_positions is not None:
                raise ValueError("Chunked Context cache usage requires cache positions.")
            if self.usage_cached_tokens is not None:
                if self.usage_cached_tokens != cached_tokens:
                    raise RuntimeError(
                        "Context cache usage changed after it was recorded: "
                        f"{self.usage_cached_tokens} != {cached_tokens}."
                    )
                return
            used_positions = None
        else:
            if (
                not cached_positions.is_cpu
                or cached_positions.ndim != 1
                or len(cached_positions) != cached_tokens
                or len(torch.unique(cached_positions)) != cached_tokens
                or bool(torch.any(cached_positions < 0).item())
                or bool(torch.any(cached_positions >= self.initial_active_cached_len).item())
            ):
                raise ValueError("Attention cache positions must identify distinct initial hits.")
            used_positions = cached_positions.to(dtype=torch.int64, device="cpu")
            if self.context_usage_cached_positions is not None:
                used_positions = torch.unique(
                    torch.cat((self.context_usage_cached_positions, used_positions))
                )
            self.context_usage_cached_positions = used_positions
            cached_tokens = len(used_positions)
            self._validate_context_cache_usage(cached_tokens)
        repos_tokens = 0
        repositioned_mask = (
            self.occurrence_repositioned_cached_mask
            if self.occurrence_repositioned_cached_mask is not None
            else self.retry_transformed_mask
        )
        if repositioned_mask is not None:
            if used_positions is None:
                raise ValueError("Retry usage requires the attention cache positions.")
            repos_tokens = int(repositioned_mask[used_positions].sum().item())
        self.usage_repos_tokens = repos_tokens
        self.usage_cached_tokens = cached_tokens
        self.drop_skipped_tokens = self.radix_cached_tokens - cached_tokens

    @property
    def reported_cached_tokens(self) -> int:
        if self.usage_cached_tokens is None:
            raise RuntimeError("Context cache usage was not recorded before reporting.")
        return self.usage_cached_tokens - self.reported_repos_tokens

    @property
    def reported_repos_tokens(self) -> int:
        if self.usage_repos_tokens is None:
            raise RuntimeError("Retry cache usage was not recorded before reporting.")
        return self.usage_repos_tokens

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def _append_host_tensor(self, name: str, value: torch.Tensor) -> None:
        """Append to a CPU tensor without copying its full prefix every token."""

        current = getattr(self, name)
        assert current is not None and current.is_cpu and value.is_cpu
        if value.ndim != current.ndim or value.shape[1:] != current.shape[1:]:
            raise ValueError(f"{name} append has incompatible shape.")
        length = len(current)
        required = length + len(value)
        dtype = torch.promote_types(current.dtype, value.dtype)
        buffer = self._host_append_buffers.get(name)
        if (
            buffer is None
            or buffer.data_ptr() != current.data_ptr()
            or buffer.dtype != dtype
            or required > len(buffer)
        ):
            capacity = length + max(self.output_len + 1, len(value), 2)
            if buffer is not None and buffer.data_ptr() == current.data_ptr():
                capacity = max(capacity, len(buffer) * 2)
            buffer = torch.empty(
                (capacity, *current.shape[1:]), dtype=dtype, device="cpu"
            )
            buffer[:length].copy_(current)
            self._host_append_buffers[name] = buffer
        buffer[length:required].copy_(value)
        setattr(self, name, buffer[:required])

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
        self._append_host_tensor("true_positions", next_pos)
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
        self._append_host_tensor(
            "raw_positions", torch.tensor([raw_position], dtype=torch.int32, device="cpu")
        )

    @property
    def sample_is_committed(self) -> bool:
        """Whether this forward's sampled token belongs to the generated stream."""

        # ChunkedReq deliberately overrides append_host because its sampled row
        # is padding, not output.  Avoid importing the scheduler subclass here.
        return type(self).append_host is Req.append_host

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
        self._append_host_tensor("input_ids", next_token)
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
        self._append_host_tensor("radix_input_ids", next_token_key)
        self._append_host_tensor("radix_match_ids", next_token_key)
        if self.radix_positions is not None:
            self._append_host_tensor(
                "radix_positions",
                torch.tensor([host_true_position], dtype=torch.int32, device="cpu"),
            )
        if self.radix_repos_info is not None:
            self._append_host_tensor(
                "radix_repos_info",
                torch.tensor([self.radix_current_reposition], dtype=torch.int32, device="cpu"),
            )
        if self.radix_key_virtual_mask is not None:
            assert self.radix_key_to_token is not None
            assert self.radix_token_to_key is not None
            token_pos = len(self.radix_token_to_key)
            key_pos = len(self.radix_match_ids) - 1
            self._append_host_tensor(
                "radix_key_virtual_mask",
                torch.tensor([False], dtype=torch.bool, device="cpu"),
            )
            self._append_host_tensor(
                "radix_key_to_token",
                torch.tensor([token_pos], dtype=torch.int64, device="cpu"),
            )
            self._append_host_tensor(
                "radix_token_to_key",
                torch.tensor([key_pos], dtype=torch.int64, device="cpu"),
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
        max_stop_len = max(
            (len(seq) for seq in self.stop_token_seqs if 0 < len(seq) <= len(self.input_ids)),
            default=0,
        )
        if max_stop_len == 0:
            return False, None
        suffix = self.input_ids[-max_stop_len:].tolist()
        for idx, stop_seq in enumerate(self.stop_token_seqs):
            if len(stop_seq) == 0 or len(stop_seq) > len(self.input_ids):
                continue
            if suffix[-len(stop_seq) :] == stop_seq:
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
    occurrence_source_pages: torch.Tensor | None = field(default=None, init=False)
    occurrence_destination_pages: torch.Tensor | None = field(default=None, init=False)
    occurrence_position_pairs: torch.Tensor | None = field(default=None, init=False)
    occurrence_rope_cache: torch.Tensor | None = field(default=None, init=False)
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
