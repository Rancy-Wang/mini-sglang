from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.kernel.context_plan import first_mask_free_conflict_event
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)
_sparse_kernel_failure_logged = False


def _calculate_cache_reuse_ratio(
    cached_len: int,
    matchable_prefix_len: int,
) -> float:
    if not 0 <= cached_len <= matchable_prefix_len:
        raise ValueError(
            "Cache reuse lengths must satisfy 0 <= cached <= matchable, got "
            f"{cached_len}, {matchable_prefix_len}."
        )
    return 1.0 if matchable_prefix_len == 0 else cached_len / matchable_prefix_len


def _supports_multi_context_mask_prefill() -> bool:
    try:
        backend = get_global_ctx().attn_backend
    except AssertionError:
        return False
    return backend.supports_multi_context_mask_prefill


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass(frozen=True)
class ContextPrefillPlan:
    use_context_mask: bool
    input_ids: torch.Tensor
    true_positions: torch.Tensor
    raw_positions: torch.Tensor
    radix_input_ids: torch.Tensor
    cache_handle: BaseCacheHandle
    cached_indices: torch.Tensor
    cached_len: int
    initial_full_match_indices: torch.Tensor
    reason: str
    radix_cached_tokens: int
    usage_cached_tokens: int | None


@dataclass(frozen=True)
class PrefillAllocation:
    cache_handle: BaseCacheHandle
    table_idx: int
    cache_reuse_ratio: float
    initial_full_match_indices: torch.Tensor
    cached_len: int
    radix_cached_tokens: int
    usage_cached_tokens: int | None
    retry_transformed_mask: torch.Tensor | None
    retry_inactive_transformed_positions: torch.Tensor | None
    retry_inactive_transformed_pages: torch.Tensor | None
    radix_actual_materialized_stage: int = 0


def _mask_free_context_reason_reference(
    req: PendingReq,
    *,
    active_cached_len: int,
    has_sliding_window: bool,
) -> str | None:
    """Return None only when compact causal Extend exactly equals the Drop mask."""

    if has_sliding_window:
        return "sliding_window_requires_absolute_key_selection"
    if (
        req.full_input_ids is None
        or req.full_token_visible_until is None
        or req.full_keep_mask is None
    ):
        return "missing_context_metadata"

    full_len = len(req.full_input_ids)
    if not len(req.full_token_visible_until) == len(req.full_keep_mask) == full_len:
        return "invalid_context_metadata_length"

    keep_mask = req.full_keep_mask != 0
    active_positions = req.true_positions.to(dtype=torch.int64, device="cpu")
    expected_positions = torch.nonzero(keep_mask, as_tuple=False).view(-1).to(torch.int64)
    if not torch.equal(active_positions, expected_positions):
        return "active_stream_does_not_match_keep_mask"
    if not 0 <= active_cached_len < len(active_positions):
        return "no_uncached_active_token"

    query_positions = active_positions[active_cached_len:]
    visible_until = req.full_token_visible_until.to(dtype=torch.int64, device="cpu")
    full_positions = torch.arange(full_len, dtype=torch.int64, device="cpu")
    if bool(torch.any(visible_until <= full_positions).item()):
        return "invalid_visibility_lifetime"

    # For every new active query q, ordinary compact causal attention exposes
    # exactly the final keep-set prefix. It is equivalent to the Drop mask iff
    # every dropped prefix token has expired and every kept prefix token remains
    # visible at q. Prefix extrema make this proof linear in the full token count.
    never_expires = torch.iinfo(torch.int64).max
    dropped_expiry = torch.where(
        keep_mask,
        torch.full_like(visible_until, -1),
        visible_until,
    )
    kept_expiry = torch.where(
        keep_mask,
        visible_until,
        torch.full_like(visible_until, never_expires),
    )
    max_dropped_expiry = torch.cummax(dropped_expiry, dim=0).values[query_positions]
    min_kept_expiry = torch.cummin(kept_expiry, dim=0).values[query_positions]
    if bool(
        torch.any(
            (max_dropped_expiry > query_positions) | (min_kept_expiry <= query_positions)
        ).item()
    ):
        return "visibility_changes_within_extend"
    return None


def _mask_free_context_reason(
    req: PendingReq,
    *,
    active_cached_len: int,
    has_sliding_window: bool,
) -> str | None:
    """Use the sparse CPU kernel, falling back to the proven O(N) reference."""

    if has_sliding_window:
        return "sliding_window_requires_absolute_key_selection"
    drop_wire = (
        req.drop_event_positions,
        req.drop_range_offsets,
        req.drop_position_ranges,
    )
    if not all(tensor is not None for tensor in drop_wire):
        return "missing_sparse_drop_metadata"
    if not 0 <= active_cached_len < len(req.true_positions):
        return "no_uncached_active_token"
    if req.drop_effective_event_count < 0:
        return _mask_free_context_reason_reference(
            req,
            active_cached_len=active_cached_len,
            has_sliding_window=False,
        )
    event_positions, range_offsets, position_ranges = drop_wire
    assert event_positions is not None
    assert range_offsets is not None
    assert position_ranges is not None
    try:
        conflict = first_mask_free_conflict_event(
            req.true_positions,
            event_positions,
            range_offsets,
            position_ranges,
            active_cached_len=active_cached_len,
            effective_event_count=req.drop_effective_event_count,
        )
    except Exception:
        global _sparse_kernel_failure_logged
        if not _sparse_kernel_failure_logged:
            logger.warning(
                "Sparse Context planner kernel failed; using the O(N) reference.",
                exc_info=True,
            )
            _sparse_kernel_failure_logged = True
        return _mask_free_context_reason_reference(
            req,
            active_cached_len=active_cached_len,
            has_sliding_window=False,
        )
    return None if conflict is None else "visibility_changes_within_extend"


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    has_sliding_window: bool = False
    enable_mask_free_context_prefill: bool = True
    kv_cache: BaseKVCachePool | None = None
    retry_rope_cache: torch.Tensor | None = None

    @classmethod
    def _has_reposition_timeline(cls, req: PendingReq) -> bool:
        return (
            req.reposition_input_ids is not None
            and req.reposition_raw_boundaries is not None
            and len(req.reposition_raw_boundaries) > 0
            and (
                cls._effective_drop_count(req) > 0
                or (
                    req.reposition_transition_offsets is not None
                    and len(req.reposition_transition_offsets) > 1
                )
            )
        )

    @staticmethod
    def _effective_drop_count(req: PendingReq) -> int:
        if req.drop_event_positions is None:
            return 0
        if req.drop_effective_event_count < 0:
            return len(req.drop_event_positions)
        return min(req.drop_effective_event_count, len(req.drop_event_positions))

    @classmethod
    def _next_staged_event_boundary(cls, req: PendingReq) -> int | None:
        candidates: list[int] = []
        drop_count = cls._effective_drop_count(req)
        if req.drop_event_positions is not None and req.staged_drop_cursor < drop_count:
            candidates.append(int(req.drop_event_positions[req.staged_drop_cursor]))
        if (
            req.reposition_insert_offsets is not None
            and req.reposition_effective_stages is not None
        ):
            for index in range(req.staged_reposition_cursor, len(req.reposition_insert_offsets)):
                if int(req.reposition_effective_stages[index]) > 0:
                    candidates.append(int(req.reposition_insert_offsets[index]))
                    break
        return min(candidates) if candidates else None

    @classmethod
    def _prepare_staged_segment(cls, req: PendingReq) -> None:
        if (
            req.reposition_input_ids is None
            or req.reposition_birth_positions is None
            or req.reposition_birth_stages is None
            or req.staged_active_raw is None
            or req.staged_current_positions is None
            or req.radix_match_ids is None
            or req.radix_token_to_key is None
        ):
            raise RuntimeError("Reposition staged-prefill metadata is incomplete.")
        raw_count = len(req.reposition_input_ids)
        next_event = cls._next_staged_event_boundary(req)
        segment_end = raw_count if next_event is None else next_event
        if not req.staged_raw_cursor < segment_end <= raw_count:
            raise RuntimeError(
                "Reposition timeline did not leave an uncached token before its next event."
            )
        new_raw = torch.arange(req.staged_raw_cursor, segment_end, dtype=torch.int32, device="cpu")
        expected_birth_stage = req.reposition_birth_stages[new_raw.to(torch.int64)]
        if bool(torch.any(expected_birth_stage != req.staged_actual_stage).item()):
            raise RuntimeError(
                "Reposition token birth stage disagrees with the scheduler stage cursor."
            )
        input_raw = torch.cat([req.staged_active_raw, new_raw])
        raw_index = input_raw.to(torch.int64)
        req.input_ids = req.reposition_input_ids[raw_index].contiguous()
        req.true_positions = req.staged_current_positions[raw_index].contiguous()
        req.raw_positions = input_raw.contiguous()
        req.radix_input_ids = req.radix_match_ids[req.radix_token_to_key[raw_index]].contiguous()
        req.staged_segment_end = segment_end
        req.staged_final_segment = next_event is None
        if req.staged_final_segment and segment_end == req.staged_raw_cursor:
            raise RuntimeError("Final Reposition stage must contain an uncached prompt token.")

    @classmethod
    def _initialize_staged_state(cls, req: PendingReq) -> None:
        if (
            req.reposition_input_ids is None
            or req.reposition_birth_positions is None
            or req.reposition_birth_stages is None
            or req.reposition_transition_offsets is None
            or req.reposition_transition_raw_tokens is None
            or req.reposition_transition_old_positions is None
            or req.reposition_transition_new_positions is None
            or req.reposition_effective_stages is None
        ):
            raise RuntimeError("Reposition compiler did not provide a complete staged timeline.")
        raw_count = len(req.reposition_input_ids)
        if len(req.reposition_birth_positions) != raw_count:
            raise RuntimeError("Reposition birth positions do not cover the raw token stream.")
        req.staged_raw_cursor = 0
        req.staged_drop_cursor = 0
        req.staged_reposition_cursor = 0
        next_event = cls._next_staged_event_boundary(req)
        if next_event is not None and next_event >= raw_count:
            raise RuntimeError(
                "Reposition/Drop after the final prompt token cannot produce valid generation logits."
            )
        req.staged_reposition = True
        req.staged_ready = True
        req.staged_actual_stage = 0
        req.staged_active_raw = torch.empty(0, dtype=torch.int32, device="cpu")
        req.staged_current_positions = req.reposition_birth_positions.clone()
        cls._prepare_staged_segment(req)

    @classmethod
    def _staged_transition_count_at_segment_end(cls, req: PendingReq) -> int:
        if (
            req.reposition_insert_offsets is None
            or req.reposition_effective_stages is None
            or req.reposition_transition_offsets is None
        ):
            return 0
        count = 0
        for index in range(req.staged_reposition_cursor, len(req.reposition_insert_offsets)):
            boundary = int(req.reposition_insert_offsets[index])
            if boundary > req.staged_segment_end:
                break
            if boundary != req.staged_segment_end:
                continue
            stage = int(req.reposition_effective_stages[index])
            if stage > 0:
                count += int(
                    req.reposition_transition_offsets[stage]
                    - req.reposition_transition_offsets[stage - 1]
                )
        return count

    def _allocate_staged(self, req: PendingReq) -> PrefillAllocation | None:
        if req.staged_cache_handle is not None or req.staged_table_idx is not None:
            raise RuntimeError("Reposition staged resources were allocated twice.")
        if self.table_manager.available_size == 0:
            return None
        transition_count = self._staged_transition_count_at_segment_end(req)
        needed = len(req.input_ids) + transition_count + req.output_len
        if needed + self.reserved_size > self.cache_manager.available_size:
            return None
        root_match = self.cache_manager.match_empty_req(req)
        self.cache_manager.lock(root_match.handle)
        if needed + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(root_match.handle)
            return None
        table_idx = self.table_manager.allocate()
        req.staged_cache_handle = root_match.handle
        req.staged_table_idx = table_idx
        req.staged_full_page_indices = torch.full(
            (len(req.reposition_input_ids),),
            -1,
            dtype=torch.int32,
            device=self.cache_manager.device,
        )
        return PrefillAllocation(
            cache_handle=root_match.handle,
            table_idx=table_idx,
            cache_reuse_ratio=0.0,
            initial_full_match_indices=root_match.full_match_indices,
            cached_len=0,
            radix_cached_tokens=0,
            usage_cached_tokens=0,
            retry_transformed_mask=None,
            retry_inactive_transformed_positions=None,
            retry_inactive_transformed_pages=None,
            radix_actual_materialized_stage=0,
        )

    def plan_context_prefill(self, req: PendingReq) -> ContextPrefillPlan | None:
        if not req.use_context_mask or req.chunked_req is not None:
            return None
        full_match = self.cache_manager.match_full_req(req)
        if full_match is None:
            return None
        active_match = self.cache_manager.derive_active_match(req, full_match)
        fallback_reason = (
            _mask_free_context_reason(
                req,
                active_cached_len=active_match.active_cached_len,
                has_sliding_window=self.has_sliding_window,
            )
            if self.enable_mask_free_context_prefill
            else "mask_free_disabled"
        )
        if fallback_reason is None:
            radix_cached_tokens = full_match.handle.physical_cached_len
            if active_match.active_cached_len > radix_cached_tokens:
                raise RuntimeError("Active cache usage exceeds resident Radix matches.")
            logger.debug(
                "Context warmup %s selected mask-free Extend with %d active cache hits.",
                req.uid,
                active_match.active_cached_len,
            )
            return ContextPrefillPlan(
                use_context_mask=False,
                input_ids=req.input_ids,
                true_positions=req.true_positions,
                raw_positions=req.raw_positions,
                radix_input_ids=req.radix_input_ids,
                cache_handle=active_match.handle,
                cached_indices=active_match.active_match_indices,
                cached_len=active_match.active_cached_len,
                initial_full_match_indices=active_match.full_match_indices,
                reason="mask_free_visibility_equivalent",
                radix_cached_tokens=radix_cached_tokens,
                usage_cached_tokens=active_match.active_cached_len,
            )

        assert req.full_input_ids is not None
        full_positions = torch.arange(len(req.full_input_ids), dtype=torch.int32, device="cpu")
        full_radix_input_ids = (
            req.radix_match_ids[req.radix_token_to_key]
            if req.radix_token_to_key is not None
            else req.radix_match_ids
        )
        assert full_radix_input_ids is not None
        logger.debug(
            "Context warmup %s retained mask Prefill: %s.",
            req.uid,
            fallback_reason,
        )
        return ContextPrefillPlan(
            use_context_mask=True,
            input_ids=req.full_input_ids,
            true_positions=full_positions,
            raw_positions=full_positions,
            radix_input_ids=full_radix_input_ids,
            cache_handle=full_match.handle,
            cached_indices=full_match.safe_match_indices,
            cached_len=full_match.safe_cached_len,
            initial_full_match_indices=full_match.full_match_indices,
            reason=fallback_reason,
            radix_cached_tokens=full_match.handle.physical_cached_len,
            usage_cached_tokens=None,
        )

    def _try_allocate_one(
        self,
        req: PendingReq,
        context_plan: ContextPrefillPlan | None = None,
    ) -> PrefillAllocation | None:
        if self.table_manager.available_size == 0:
            return None

        original_stream = None
        retry_plan = None
        retry_active_full_positions = None
        if context_plan is not None:
            original_stream = (
                req.input_ids,
                req.true_positions,
                req.raw_positions,
                req.radix_input_ids,
                req.use_context_mask,
            )
            req.input_ids = context_plan.input_ids
            req.true_positions = context_plan.true_positions
            req.raw_positions = context_plan.raw_positions
            req.radix_input_ids = context_plan.radix_input_ids
            req.use_context_mask = context_plan.use_context_mask
            cache_handle = context_plan.cache_handle
            cached_len = context_plan.cached_len
            cached_indices = context_plan.cached_indices
            initial_full_match_indices = context_plan.initial_full_match_indices
            radix_cached_tokens = context_plan.radix_cached_tokens
            usage_cached_tokens = context_plan.usage_cached_tokens
        elif req.use_context_mask:
            match = self.cache_manager.match_full_req(req)
            if match is None:
                return None
            cache_handle = match.handle
            cached_len = match.safe_cached_len
            cached_indices = match.safe_match_indices
            initial_full_match_indices = match.full_match_indices
            radix_cached_tokens = match.handle.physical_cached_len
            usage_cached_tokens = None
        else:
            match_started_ns = time.perf_counter_ns()
            match = self.cache_manager.match_req(req)
            match_elapsed_ns = time.perf_counter_ns() - match_started_ns
            match_retry_plan_ns = 0 if match is None else match.retry_plan_ns
            req.radix_match_ns += max(0, match_elapsed_ns - match_retry_plan_ns)
            if match is None:
                return None
            req.retry_plan_ns += match.retry_plan_ns
            if self._has_reposition_timeline(req) and match.active_cached_len < max(
                req.input_len - 1, 0
            ):
                self._initialize_staged_state(req)
                return self._allocate_staged(req)
            cache_handle = match.handle
            cached_len = match.active_cached_len
            cached_indices = match.active_match_indices
            initial_full_match_indices = match.full_match_indices[: match.full_cached_len]
            radix_cached_tokens = cached_len
            usage_cached_tokens = cached_len
            retry_plan = match.retry_plan
            retry_active_full_positions = match.active_full_positions
        full_prefix_len, active_prefix_len = self.cache_manager.matchable_prefix_lens(req)
        cache_reuse_ratio = _calculate_cache_reuse_ratio(
            cached_len,
            full_prefix_len if req.use_context_mask else active_prefix_len,
        )
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        retry_transformed_mask = None
        retry_page_count = 0 if retry_plan is None else len(retry_plan)
        estimated_len = extend_len + req.output_len + retry_page_count

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            if original_stream is not None:
                (
                    req.input_ids,
                    req.true_positions,
                    req.raw_positions,
                    req.radix_input_ids,
                    req.use_context_mask,
                ) = original_stream
            return None
        self.cache_manager.lock(cache_handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(cache_handle)
            if original_stream is not None:
                (
                    req.input_ids,
                    req.true_positions,
                    req.raw_positions,
                    req.radix_input_ids,
                    req.use_context_mask,
                ) = original_stream
            return None

        table_idx = self.table_manager.allocate()
        retry_pages = torch.empty(0, dtype=torch.int32, device=cached_indices.device)
        retry_inactive_positions = None
        retry_inactive_pages = None
        try:
            if retry_page_count > 0:
                if self.kv_cache is None or self.retry_rope_cache is None:
                    raise RuntimeError("Retry Reposition KV transform is not configured.")
                if retry_plan is None or retry_active_full_positions is None:
                    raise RuntimeError("Retry Reposition plan metadata is incomplete.")
                changed_old_positions = retry_plan[:, 2]
                changed_new_positions = retry_plan[:, 3]
                rope_cache_len = len(self.retry_rope_cache)
                if (
                    int(torch.min(changed_old_positions).item()) < 0
                    or int(torch.min(changed_new_positions).item()) < 0
                    or int(torch.max(changed_old_positions).item()) >= rope_cache_len
                    or int(torch.max(changed_new_positions).item()) >= rope_cache_len
                ):
                    raise RuntimeError("Retry Reposition position exceeds the RoPE cache.")
                if req.radix_token_to_key is None:
                    raise RuntimeError("Retry Reposition requires a structured token mapping.")
                full_to_active = torch.full(
                    (len(req.radix_token_to_key),), -1, dtype=torch.int32, device="cpu"
                )
                full_to_active[retry_active_full_positions] = torch.arange(
                    len(retry_active_full_positions), dtype=torch.int32, device="cpu"
                )
                changed_active_indices = full_to_active[retry_plan[:, 1].to(torch.int64)]
                retry_metadata = (
                    torch.column_stack(
                        (
                            retry_plan[:, 0],
                            retry_plan[:, 1],
                            changed_active_indices,
                            changed_old_positions,
                            changed_new_positions,
                        ),
                    )
                    .pin_memory()
                    .to(device=self.cache_manager.device, non_blocking=True)
                )
                req.reposition_h2d_bytes += retry_metadata.numel() * retry_metadata.element_size()
                req.reposition_transition_count += retry_page_count
                source_full_device = retry_metadata[:, 0].to(torch.int64)
                target_full_device = retry_metadata[:, 1].to(torch.int64)
                source_pages = initial_full_match_indices[source_full_device]
                retry_pages = self.cache_manager.allocate_retry_pages(retry_page_count)
                self.kv_cache.retry_reposition(
                    source_pages,
                    retry_pages,
                    retry_metadata[:, 3:],
                    self.retry_rope_cache,
                )
                initial_full_match_indices = initial_full_match_indices.clone()
                initial_full_match_indices[target_full_device] = retry_pages
                active_rows_cpu = changed_active_indices >= 0
                retry_transformed_mask = torch.zeros(cached_len, dtype=torch.bool, device="cpu")
                changed_active = changed_active_indices[active_rows_cpu].to(torch.int64)
                retry_transformed_mask[changed_active] = True
                if bool(torch.any(active_rows_cpu).item()):
                    active_rows = retry_metadata[:, 2] >= 0
                    cached_indices = cached_indices.clone()
                    active_indices = retry_metadata[active_rows, 2].to(torch.int64)
                    cached_indices[active_indices] = retry_pages[active_rows]
                inactive_rows_cpu = changed_active_indices < 0
                if bool(torch.any(inactive_rows_cpu).item()):
                    retry_inactive_positions = retry_plan[inactive_rows_cpu, 1].to(torch.int64)
                    retry_inactive_pages = retry_pages[retry_metadata[:, 2] < 0]
            if cached_len > 0:  # NOTE: set the cached part
                device_ids = self.table_manager.token_pool[table_idx][:cached_len]
                page_entry = self.table_manager.page_table[table_idx][:cached_len]
                device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
                page_entry.copy_(cached_indices)
        except Exception:
            self.cache_manager.free_retry_pages(retry_pages)
            self.table_manager.free(table_idx)
            self.cache_manager.unlock(cache_handle)
            if original_stream is not None:
                (
                    req.input_ids,
                    req.true_positions,
                    req.raw_positions,
                    req.radix_input_ids,
                    req.use_context_mask,
                ) = original_stream
            raise

        return PrefillAllocation(
            cache_handle=cache_handle,
            table_idx=table_idx,
            cache_reuse_ratio=cache_reuse_ratio,
            initial_full_match_indices=initial_full_match_indices.clone(),
            cached_len=cached_len,
            radix_cached_tokens=radix_cached_tokens,
            usage_cached_tokens=usage_cached_tokens,
            retry_transformed_mask=retry_transformed_mask,
            retry_inactive_transformed_positions=retry_inactive_positions,
            retry_inactive_transformed_pages=retry_inactive_pages,
            radix_actual_materialized_stage=(
                len(req.reposition_transition_offsets) - 1
                if req.reposition_transition_offsets is not None
                else 0
            ),
        )

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        cache_reuse_ratio: float,
        initial_full_match_indices: torch.Tensor,
        initial_active_cached_len: int,
        radix_cached_tokens: int,
        usage_cached_tokens: int | None,
        retry_transformed_mask: torch.Tensor | None,
        retry_inactive_transformed_positions: torch.Tensor | None,
        retry_inactive_transformed_pages: torch.Tensor | None,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len or (
            pending_req.staged_reposition and not pending_req.staged_final_segment
        )
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        if pending_req.staged_reposition:
            self.reserved_size += self._staged_transition_count_at_segment_end(pending_req)
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            true_positions=pending_req.true_positions[: cached_len + chunk_size],
            raw_positions=pending_req.raw_positions[: cached_len + chunk_size],
            radix_input_ids=pending_req.radix_input_ids[: cached_len + chunk_size],
            radix_match_ids=(
                pending_req.radix_match_ids
                if pending_req.radix_match_ids is not None
                else pending_req.radix_input_ids
            ),
            initial_full_match_indices=initial_full_match_indices,
            initial_active_cached_len=initial_active_cached_len,
            true_seq_len=(
                pending_req.radix_next_position
                if pending_req.radix_next_position is not None
                and (not pending_req.staged_reposition or pending_req.staged_final_segment)
                else int(pending_req.true_positions[cached_len + chunk_size - 1].item()) + 1
            ),
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            prompt_tokens=pending_req.prompt_tokens,
            stop=pending_req.stop,
            stop_token_seqs=pending_req.stop_token_seqs,
            prefix_keep_mask=pending_req.prefix_keep_mask,
            is_warmup=pending_req.is_warmup,
            cache_reuse_ratio=cache_reuse_ratio,
            radix_cached_tokens=radix_cached_tokens,
            usage_cached_tokens=usage_cached_tokens,
            drop_skipped_tokens=(
                radix_cached_tokens - usage_cached_tokens if usage_cached_tokens is not None else 0
            ),
            full_input_ids=(pending_req.full_input_ids if pending_req.use_context_mask else None),
            full_token_visible_until=(
                pending_req.full_token_visible_until if pending_req.use_context_mask else None
            ),
            full_keep_mask=(pending_req.full_keep_mask if pending_req.use_context_mask else None),
            use_context_mask=pending_req.use_context_mask,
            radix_key_virtual_mask=pending_req.radix_key_virtual_mask,
            radix_key_to_token=pending_req.radix_key_to_token,
            radix_token_to_key=pending_req.radix_token_to_key,
            radix_commit_key_len=pending_req.radix_commit_key_len,
            radix_marker_ids=pending_req.radix_marker_ids,
            radix_positions=pending_req.radix_positions,
            radix_repos_info=pending_req.radix_repos_info,
            radix_materialized_stage=(
                pending_req.radix_materialized_stage
                if not pending_req.staged_reposition or pending_req.staged_final_segment
                else None
            ),
            reposition_transition_offsets=pending_req.reposition_transition_offsets,
            staged_full_page_indices=(
                pending_req.staged_full_page_indices
                if pending_req.staged_reposition and pending_req.staged_final_segment
                else None
            ),
            staged_reposition=pending_req.staged_reposition,
            radix_actual_materialized_stage=(
                pending_req.staged_actual_stage
                if pending_req.staged_reposition
                else (
                    len(pending_req.reposition_transition_offsets) - 1
                    if pending_req.reposition_transition_offsets is not None
                    else 0
                )
            ),
            radix_next_position=(
                pending_req.radix_next_position
                if not pending_req.staged_reposition or pending_req.staged_final_segment
                else None
            ),
            radix_current_reposition=pending_req.radix_current_reposition,
            retry_transformed_mask=retry_transformed_mask,
            retry_inactive_transformed_positions=retry_inactive_transformed_positions,
            retry_inactive_transformed_pages=retry_inactive_transformed_pages,
            tokenize_invocations=pending_req.tokenize_invocations,
            context_stage_count=pending_req.context_stage_count,
            radix_compile_ns=pending_req.radix_compile_ns,
            radix_match_ns=pending_req.radix_match_ns,
            retry_plan_ns=pending_req.retry_plan_ns,
            reposition_transition_count=pending_req.reposition_transition_count,
            reposition_h2d_bytes=pending_req.reposition_h2d_bytes,
            reposition_d2h_bytes=pending_req.reposition_d2h_bytes,
        )

    def try_add_one(
        self,
        pending_req: PendingReq,
        context_plan: ContextPrefillPlan | None = None,
    ) -> Req | None:
        if self.token_budget <= 0:
            return None

        if pending_req.staged_reposition and not pending_req.staged_ready:
            return None

        if pending_req.staged_reposition and pending_req.chunked_req is None:
            if pending_req.staged_cache_handle is None or pending_req.staged_table_idx is None:
                resource = self._allocate_staged(pending_req)
                if resource is None:
                    return None
                cache_handle = resource.cache_handle
                table_idx = resource.table_idx
                initial_full = resource.initial_full_match_indices
            else:
                cache_handle = pending_req.staged_cache_handle
                table_idx = pending_req.staged_table_idx
                initial_full = torch.empty(0, dtype=torch.int32, device=self.cache_manager.device)
            cached_len = (
                len(pending_req.staged_active_raw)
                if pending_req.staged_active_raw is not None
                else 0
            )
            transition_count = self._staged_transition_count_at_segment_end(pending_req)
            remain_len = pending_req.input_len - cached_len
            if (
                remain_len + transition_count + pending_req.output_len + self.reserved_size
                > self.cache_manager.available_size
            ):
                return None
            result = self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cached_len,
                cache_reuse_ratio=0.0,
                initial_full_match_indices=initial_full,
                initial_active_cached_len=0,
                radix_cached_tokens=0,
                usage_cached_tokens=0,
                retry_transformed_mask=None,
                retry_inactive_transformed_positions=None,
                retry_inactive_transformed_pages=None,
            )
            if isinstance(result, ChunkedReq):
                pending_req.staged_ready = False
            return result

        if chunked_req := pending_req.chunked_req:
            result = self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
                cache_reuse_ratio=chunked_req.cache_reuse_ratio,
                initial_full_match_indices=chunked_req.initial_full_match_indices,
                initial_active_cached_len=chunked_req.initial_active_cached_len,
                radix_cached_tokens=chunked_req.radix_cached_tokens,
                usage_cached_tokens=chunked_req.usage_cached_tokens,
                retry_transformed_mask=chunked_req.retry_transformed_mask,
                retry_inactive_transformed_positions=(
                    chunked_req.retry_inactive_transformed_positions
                ),
                retry_inactive_transformed_pages=chunked_req.retry_inactive_transformed_pages,
            )
            if pending_req.staged_reposition and isinstance(result, ChunkedReq):
                pending_req.staged_ready = False
            return result

        if resource := self._try_allocate_one(pending_req, context_plan):
            result = self._add_one_req(
                pending_req=pending_req,
                cache_handle=resource.cache_handle,
                table_idx=resource.table_idx,
                cached_len=resource.cached_len,
                cache_reuse_ratio=resource.cache_reuse_ratio,
                initial_full_match_indices=resource.initial_full_match_indices,
                initial_active_cached_len=resource.cached_len,
                radix_cached_tokens=resource.radix_cached_tokens,
                usage_cached_tokens=resource.usage_cached_tokens,
                retry_transformed_mask=resource.retry_transformed_mask,
                retry_inactive_transformed_positions=(
                    resource.retry_inactive_transformed_positions
                ),
                retry_inactive_transformed_pages=resource.retry_inactive_transformed_pages,
            )
            if pending_req.staged_reposition and isinstance(result, ChunkedReq):
                pending_req.staged_ready = False
            return result

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    has_sliding_window: bool = False
    enable_mask_free_context_prefill: bool = True
    kv_cache: BaseKVCachePool | None = None
    retry_rope_cache: torch.Tensor | None = None
    pending_list: List[PendingReq] = field(default_factory=list)
    aborted_staged_chunks: dict[int, PendingReq] = field(default_factory=dict)

    def add_one_req(self, req: UserMsg) -> None:
        if req.use_context_mask:
            if not req.is_warmup:
                raise ValueError("Context-mask Prefill is restricted to warmup requests.")
            if req.full_input_ids is None or req.radix_match_ids is None:
                raise ValueError(
                    "Context-mask Prefill requires a full token stream and Radix keys."
                )
        self.pending_list.append(
            PendingReq(
                uid=req.uid,
                input_ids=req.input_ids,
                true_positions=req.true_positions,
                raw_positions=req.raw_positions,
                radix_input_ids=req.radix_input_ids,
                radix_match_ids=req.radix_match_ids,
                sampling_params=req.sampling_params,
                prompt_tokens=req.prompt_tokens or len(req.input_ids),
                stop=req.stop,
                stop_token_seqs=req.stop_token_seqs,
                is_warmup=req.is_warmup,
                internal_uid=req.internal_uid,
                prefix_keep_mask=req.prefix_keep_mask,
                full_input_ids=req.full_input_ids,
                full_token_visible_until=req.full_token_visible_until,
                full_keep_mask=req.full_keep_mask,
                drop_event_positions=req.drop_event_positions,
                drop_range_offsets=req.drop_range_offsets,
                drop_position_ranges=req.drop_position_ranges,
                drop_effective_event_count=req.drop_effective_event_count,
                use_context_mask=req.use_context_mask,
                radix_key_virtual_mask=req.radix_key_virtual_mask,
                radix_key_to_token=req.radix_key_to_token,
                radix_token_to_key=req.radix_token_to_key,
                radix_commit_key_len=req.radix_commit_key_len,
                radix_marker_ids=tuple(req.radix_marker_ids or ()),
                radix_positions=req.radix_positions,
                radix_repos_info=req.radix_repos_info,
                radix_materialized_stage=req.radix_materialized_stage,
                reposition_raw_boundaries=req.reposition_raw_boundaries,
                reposition_insert_offsets=req.reposition_insert_offsets,
                reposition_input_ids=req.reposition_input_ids,
                reposition_birth_positions=req.reposition_birth_positions,
                reposition_birth_stages=req.reposition_birth_stages,
                reposition_transition_offsets=req.reposition_transition_offsets,
                reposition_transition_raw_tokens=req.reposition_transition_raw_tokens,
                reposition_transition_old_positions=req.reposition_transition_old_positions,
                reposition_transition_new_positions=req.reposition_transition_new_positions,
                reposition_effective_stages=req.reposition_effective_stages,
                radix_next_position=req.radix_next_position,
                radix_current_reposition=req.radix_current_reposition,
                tokenize_invocations=req.tokenize_invocations,
                context_stage_count=req.context_stage_count,
                radix_compile_ns=req.radix_compile_ns,
            )
        )

    def _pending_for_chunk(self, chunk: ChunkedReq) -> PendingReq | None:
        for pending in self.pending_list:
            if pending.chunked_req is chunk:
                return pending
        return None

    def complete_chunk(self, chunk: ChunkedReq) -> None:
        aborted = self.aborted_staged_chunks.pop(id(chunk), None)
        pending = aborted or self._pending_for_chunk(chunk)
        if pending is None or not pending.staged_reposition:
            return
        if pending.staged_full_page_indices is None:
            raise RuntimeError("Staged Reposition lost its full raw-to-page map.")
        materialized_len = chunk.cached_len
        raw_positions = chunk.raw_positions[:materialized_len]
        pages = self.table_manager.page_table[chunk.table_idx, :materialized_len]
        raw_device = raw_positions.pin_memory().to(
            device=pages.device, dtype=torch.int64, non_blocking=True
        )
        pending.staged_full_page_indices[raw_device] = pages
        pending.radix_match_ns = chunk.radix_match_ns
        pending.retry_plan_ns = chunk.retry_plan_ns
        pending.reposition_transition_count = chunk.reposition_transition_count
        pending.reposition_h2d_bytes = chunk.reposition_h2d_bytes
        pending.reposition_d2h_bytes = chunk.reposition_d2h_bytes
        pending.reposition_h2d_bytes += raw_positions.numel() * raw_positions.element_size()

        if aborted is not None:
            pending.chunked_req = None
            self.release_staged(pending)
            return

        if materialized_len < pending.input_len:
            pending.staged_ready = True
            return

        pending.staged_raw_cursor = pending.staged_segment_end
        pending.staged_active_raw = raw_positions.clone()

        drop_count = PrefillAdder._effective_drop_count(pending)
        if (
            pending.drop_event_positions is not None
            and pending.drop_range_offsets is not None
            and pending.drop_position_ranges is not None
        ):
            while (
                pending.staged_drop_cursor < drop_count
                and int(pending.drop_event_positions[pending.staged_drop_cursor])
                == pending.staged_raw_cursor
            ):
                event = pending.staged_drop_cursor
                keep = torch.ones(len(pending.staged_active_raw), dtype=torch.bool, device="cpu")
                start_offset = int(pending.drop_range_offsets[event])
                end_offset = int(pending.drop_range_offsets[event + 1])
                for range_index in range(start_offset, end_offset):
                    start = int(pending.drop_position_ranges[2 * range_index])
                    end = int(pending.drop_position_ranges[2 * range_index + 1])
                    keep &= ~(
                        (pending.staged_active_raw >= start) & (pending.staged_active_raw < end)
                    )
                pending.staged_active_raw = pending.staged_active_raw[keep]
                pending.staged_drop_cursor += 1

        if (
            pending.reposition_insert_offsets is not None
            and pending.reposition_effective_stages is not None
            and pending.reposition_transition_offsets is not None
            and pending.reposition_transition_raw_tokens is not None
            and pending.reposition_transition_old_positions is not None
            and pending.reposition_transition_new_positions is not None
            and pending.staged_current_positions is not None
        ):
            while pending.staged_reposition_cursor < len(pending.reposition_insert_offsets):
                event = pending.staged_reposition_cursor
                boundary = int(pending.reposition_insert_offsets[event])
                if boundary > pending.staged_raw_cursor:
                    break
                stage = int(pending.reposition_effective_stages[event])
                pending.staged_reposition_cursor += 1
                if stage < 0:
                    continue
                if boundary != pending.staged_raw_cursor:
                    raise RuntimeError(
                        "An effective Reposition stage was skipped by staged Prefill."
                    )
                if stage != pending.staged_actual_stage + 1:
                    raise RuntimeError("Reposition stage mapping is not contiguous.")
                begin = int(pending.reposition_transition_offsets[stage - 1])
                end = int(pending.reposition_transition_offsets[stage])
                raw_tokens = pending.reposition_transition_raw_tokens[begin:end]
                old_positions = pending.reposition_transition_old_positions[begin:end]
                new_positions = pending.reposition_transition_new_positions[begin:end]
                if len(raw_tokens) > 0:
                    if self.kv_cache is None or self.retry_rope_cache is None:
                        raise RuntimeError("Staged Reposition KV transform is not configured.")
                    rope_cache_len = len(self.retry_rope_cache)
                    if (
                        int(torch.min(old_positions).item()) < 0
                        or int(torch.min(new_positions).item()) < 0
                        or int(torch.max(old_positions).item()) >= rope_cache_len
                        or int(torch.max(new_positions).item()) >= rope_cache_len
                    ):
                        raise RuntimeError("Staged Reposition position exceeds the RoPE cache.")
                    active_lookup = torch.zeros(
                        len(pending.reposition_input_ids), dtype=torch.bool, device="cpu"
                    )
                    active_lookup[pending.staged_active_raw.to(torch.int64)] = True
                    if not bool(torch.all(active_lookup[raw_tokens.to(torch.int64)]).item()):
                        raise RuntimeError("Reposition transition references an inactive token.")
                    transition_cpu = torch.column_stack(
                        (raw_tokens, old_positions, new_positions)
                    ).pin_memory()
                    transition_device = transition_cpu.to(
                        device=self.cache_manager.device, non_blocking=True
                    )
                    raw_token_device = transition_device[:, 0].to(torch.int64)
                    source_pages = pending.staged_full_page_indices[raw_token_device]
                    destination_pages = self.cache_manager.allocate_retry_pages(len(raw_tokens))
                    self.kv_cache.retry_reposition(
                        source_pages,
                        destination_pages,
                        transition_device[:, 1:],
                        self.retry_rope_cache,
                    )
                    pending.staged_full_page_indices[raw_token_device] = destination_pages
                    self.cache_manager.free_retry_pages(source_pages)
                    pending.reposition_transition_count += len(raw_tokens)
                    pending.reposition_h2d_bytes += (
                        transition_device.numel() * transition_device.element_size()
                    )
                    pending.staged_current_positions[raw_tokens.to(torch.int64)] = new_positions
                pending.staged_actual_stage = stage

        active_device = pending.staged_active_raw.pin_memory().to(
            device=self.cache_manager.device, dtype=torch.int64, non_blocking=True
        )
        pending.reposition_h2d_bytes += (
            pending.staged_active_raw.numel() * pending.staged_active_raw.element_size()
        )
        active_pages = pending.staged_full_page_indices[active_device]
        self.table_manager.page_table[chunk.table_idx, : len(pending.staged_active_raw)].copy_(
            active_pages
        )
        pending.chunked_req = None
        PrefillAdder._prepare_staged_segment(pending)
        pending.staged_ready = True

    def release_staged(self, pending: PendingReq) -> None:
        if pending.staged_full_page_indices is not None:
            valid = pending.staged_full_page_indices[pending.staged_full_page_indices >= 0]
            self.cache_manager.free_retry_pages(valid)
            pending.staged_full_page_indices = None
        if pending.staged_cache_handle is not None:
            self.cache_manager.unlock(pending.staged_cache_handle)
            pending.staged_cache_handle = None
        if pending.staged_table_idx is not None:
            self.table_manager.free(pending.staged_table_idx)
            pending.staged_table_idx = None

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            has_sliding_window=self.has_sliding_window,
            enable_mask_free_context_prefill=self.enable_mask_free_context_prefill,
            kv_cache=self.kv_cache,
            retry_rope_cache=self.retry_rope_cache,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        supports_multi_context_mask = _supports_multi_context_mask_prefill()
        for pending_req in self.pending_list:
            context_plan = (
                adder.plan_context_prefill(pending_req)
                if pending_req.use_context_mask and pending_req.chunked_req is None
                else None
            )
            if pending_req.use_context_mask and pending_req.chunked_req is None:
                if context_plan is None:
                    break
                planned_context_mask = context_plan.use_context_mask
            else:
                planned_context_mask = pending_req.use_context_mask
            if len(reqs) > 0:
                first_uses_context_mask = reqs[0].use_context_mask
                if planned_context_mask != first_uses_context_mask:
                    break
                if planned_context_mask and not supports_multi_context_mask:
                    break
            if req := adder.try_add_one(pending_req, context_plan):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                if pending_req.use_context_mask and not supports_multi_context_mask:
                    break
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | PendingReq | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                if req.staged_reposition:
                    if req.chunked_req is not None:
                        self.aborted_staged_chunks[id(req.chunked_req)] = req
                    else:
                        self.release_staged(req)
                    return req
                return req.chunked_req or req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
