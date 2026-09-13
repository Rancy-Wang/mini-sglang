from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.kernel.context_plan import (
    first_mask_free_conflict_event,
    try_build_occurrence_capacity_index,
)
from minisgl.utils import init_logger

from .reposition_occurrence import (
    compile_occurrence_window,
    install_occurrence_plan,
    pack_compact_occurrence_pending_fields,
    unpack_compact_occurrence_pending_fields,
)
from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)
_sparse_kernel_failure_logged = False


class OccurrenceInputError(ValueError):
    def __init__(self, uid: int, detail: str) -> None:
        super().__init__(detail)
        self.uid = uid


@dataclass
class RepositionCapacityError(RuntimeError):
    uid: int
    required_pages: int
    available_pages: int
    matched_pages: int
    retry_pages: int

    @property
    def signature(self) -> tuple[int, int, int, int]:
        return (
            self.required_pages,
            self.available_pages,
            self.matched_pages,
            self.retry_pages,
        )

    def __str__(self) -> str:
        return (
            "Reposition cannot make progress with the current KV capacity: "
            f"needs {self.required_pages} allocatable pages after pinning its source, "
            f"but only {self.available_pages} remain "
            f"(matched={self.matched_pages}, retry={self.retry_pages})."
        )


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
    except (AssertionError, AttributeError):
        return False
    return bool(getattr(backend, "supports_multi_context_mask_prefill", False))


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
    retry_plan: torch.Tensor | None = None
    retry_active_full_positions: torch.Tensor | None = None


@dataclass(frozen=True)
class PrefillAllocation:
    cache_handle: BaseCacheHandle
    table_idx: int
    cache_reuse_ratio: float
    initial_full_match_indices: torch.Tensor
    cached_len: int
    radix_cached_tokens: int
    usage_cached_tokens: int | None
    usage_repos_tokens: int | None
    retry_transformed_mask: torch.Tensor | None
    inactive_cached_positions: torch.Tensor | None
    inactive_cached_pages: torch.Tensor | None
    chunk_size: int | None = None
    reserved_pages: int | None = None
    context_usage_cached_positions: torch.Tensor | None = None
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
    occurrence_allocated_pages: torch.Tensor | None = None


def _mask_free_context_reason_reference(
    req: PendingReq,
    *,
    active_cached_len: int,
    has_sliding_window: bool,
) -> str | None:
    """Return None only when compact causal Extend exactly equals the Drop mask."""

    # Ordinary attention already selects sliding windows by absolute positions.
    # Equality of full visibility also proves equality after the same window intersection.
    del has_sliding_window
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
    active_positions = req.raw_positions.to(dtype=torch.int64, device="cpu")
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

    # Ordinary attention already selects sliding windows by absolute positions.
    # Equality of full visibility also proves equality after the same window intersection.
    del has_sliding_window
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
            req.raw_positions,
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
    initial_token_budget: int = field(init=False)

    def __post_init__(self) -> None:
        self.initial_token_budget = self.token_budget

    @staticmethod
    def _occurrence_required_ids(
        req: PendingReq | Req,
        start: int,
        end: int,
        *,
        source_count: int = 0,
    ) -> torch.Tensor:
        """Return the full-plan occurrence IDs referenced by one query chunk."""

        assert req.occurrence_birth_indices is not None
        assert req.occurrence_terminal_indices is not None
        assert req.occurrence_segment_query_starts is not None
        assert req.occurrence_segment_query_ends is not None
        assert req.occurrence_segment_key_offsets is not None
        assert req.occurrence_segment_key_indices is not None
        parts = [
            req.occurrence_birth_indices[start:end],
            req.occurrence_terminal_indices[start:end],
        ]
        if end == len(req.occurrence_birth_indices) and source_count:
            # Finalize every borrowed source page before the request enters
            # Decode. This also covers dropped tokens that are cacheable later.
            parts.append(req.occurrence_terminal_indices[:source_count])
        for segment_index, (raw_start_tensor, raw_end_tensor) in enumerate(
            zip(
                req.occurrence_segment_query_starts,
                req.occurrence_segment_query_ends,
                strict=True,
            )
        ):
            raw_start = int(raw_start_tensor)
            raw_end = int(raw_end_tensor)
            query_start = max(start, raw_start)
            query_end = min(end, raw_end)
            if query_start >= query_end:
                continue
            key_start = int(req.occurrence_segment_key_offsets[segment_index])
            key_end = int(req.occurrence_segment_key_offsets[segment_index + 1])
            segment_keys = req.occurrence_segment_key_indices[key_start:key_end]
            prefix_length = len(segment_keys) - (raw_end - raw_start)
            if prefix_length < 0:
                raise RuntimeError("Occurrence segment has fewer keys than local queries.")
            parts.append(segment_keys[: prefix_length + query_end - raw_start])
        required = torch.unique(torch.cat(parts).to(torch.int64))
        if len(required) == 0:
            raise RuntimeError("Occurrence chunk has no physical page requirements.")
        return required

    @staticmethod
    def _occurrence_capacity_for_chunk(
        req: PendingReq | Req,
        *,
        start: int,
        end: int,
        terminal_owned: torch.Tensor,
        initial_source_positions: torch.Tensor | None = None,
        exact_full_cached_len: int | None = None,
    ) -> tuple[torch.Tensor, int, int, int]:
        """Return IDs, current allocations, persistent allocations, and future reserve."""

        assert req.occurrence_raw_tokens is not None
        assert req.occurrence_positions is not None
        assert req.occurrence_birth_indices is not None
        assert req.occurrence_terminal_indices is not None
        source_count = 0 if initial_source_positions is None else len(initial_source_positions)
        if exact_full_cached_len is None:
            exact_full_cached_len = source_count
        required = PrefillAdder._occurrence_required_ids(
            req, start, end, source_count=source_count
        )
        required_raw = req.occurrence_raw_tokens[required].to(torch.int64)
        required_positions = req.occurrence_positions[required]
        terminal_ids = req.occurrence_terminal_indices.to(torch.int64)
        terminal_positions = req.occurrence_positions[terminal_ids]
        birth_ids = req.occurrence_birth_indices.to(torch.int64)
        birth_positions = req.occurrence_positions[birth_ids]

        prior = required_raw < start
        current = (required_raw >= start) & (required_raw < end)
        if bool(torch.any(~(prior | current)).item()):
            raise RuntimeError("Occurrence chunk references an uncomputed future token.")
        prior_raw = required_raw[prior]
        prior_birth_ids = req.occurrence_birth_indices[prior_raw].to(torch.int64)
        canonical_positions = req.occurrence_positions[prior_birth_ids]
        if initial_source_positions is None:
            initial_source_positions = torch.empty(0, dtype=torch.int32, device="cpu")
        matched_prior = prior_raw < len(initial_source_positions)
        canonical_positions[matched_prior] = initial_source_positions[
            prior_raw[matched_prior]
        ]
        reuse_terminal = terminal_owned[prior_raw] & (
            required_positions[prior] == terminal_positions[prior_raw]
        )
        canonical_positions[reuse_terminal] = terminal_positions[prior_raw[reuse_terminal]]
        prior_new = required_positions[prior] != canonical_positions
        if end == len(birth_ids):
            retry_source = (prior_raw >= exact_full_cached_len) & matched_prior
            terminal_required = required[prior] == terminal_ids[prior_raw]
            prior_new |= retry_source & terminal_required & (~terminal_owned[prior_raw])
        current_new = int(torch.count_nonzero(current).item())
        current_allocations = int(torch.count_nonzero(prior_new).item()) + current_new
        persistent_prior = prior_new & (required[prior] == terminal_ids[prior_raw])
        fresh_birth_is_terminal = (
            req.occurrence_birth_indices[start:end] == req.occurrence_terminal_indices[start:end]
        )
        persistent_allocations = (
            int(torch.count_nonzero(persistent_prior).item())
            + (end - start)
            + int(torch.count_nonzero(~fresh_birth_is_terminal).item())
        )

        owner_after = terminal_owned.clone()
        if len(prior_raw) > 0 and bool(torch.any(persistent_prior).item()):
            owner_after[prior_raw[persistent_prior]] = True
        owner_after[start:end] = True
        final_keep = (
            req.full_keep_mask.to(dtype=torch.bool, device="cpu")
            if req.full_keep_mask is not None
            else torch.ones(len(owner_after), dtype=torch.bool, device="cpu")
        )
        if len(final_keep) != len(owner_after):
            raise RuntimeError("Occurrence final keep mask does not cover the prompt plan.")
        # Source pages outside the exact-key prefix belong to a different
        # Radix branch, even when their final position happens to be equal.
        source_raw = torch.arange(len(owner_after), device="cpu") < source_count
        canonical_terminal_positions = birth_positions.clone()
        canonical_terminal_positions[source_raw] = initial_source_positions
        retry_source = source_raw & (
            torch.arange(len(owner_after), device="cpu") >= exact_full_cached_len
        )
        needs_terminal = (~owner_after) & (
            (terminal_positions != canonical_terminal_positions) | retry_source
        )
        output_reserve = req.output_len
        future_birth_pages = len(owner_after) - end
        future_reserve = (
            future_birth_pages + int(torch.count_nonzero(needs_terminal).item()) + output_reserve
        )
        return required, current_allocations, persistent_allocations, future_reserve

    def _try_allocate_occurrence(
        self,
        req: PendingReq,
        chunked_req: ChunkedReq | None = None,
    ) -> PrefillAllocation | None:
        if chunked_req is None and self.table_manager.available_size == 0:
            return None
        if self.kv_cache is None or self.retry_rope_cache is None:
            raise RuntimeError("Paged-occurrence KV materialization is not configured.")
        initial_allocation = chunked_req is None
        compact_layout = (
            unpack_compact_occurrence_pending_fields(req) if initial_allocation else None
        )
        if compact_layout is not None:
            # PendingReq intentionally reuses its legacy occurrence slots for the
            # compact wire program.  Preserve the immutable source program after
            # installing a runtime plan so a later scheduling attempt can adapt
            # to a changed exact-prefix match.
            setattr(req, "_compact_occurrence_layout", compact_layout)
        elif initial_allocation:
            compact_layout = getattr(req, "_compact_occurrence_layout", None)

        occurrence_raw = None
        occurrence_positions = None
        birth_ids = None
        terminal_ids = None
        occurrence_count = 0
        plan_token_count = 0

        def refresh_occurrence_plan() -> None:
            nonlocal occurrence_count, occurrence_positions, occurrence_raw
            nonlocal birth_ids, plan_token_count, terminal_ids
            plan = (
                req.occurrence_raw_tokens,
                req.occurrence_positions,
                req.occurrence_birth_indices,
                req.occurrence_terminal_indices,
                req.occurrence_segment_query_starts,
                req.occurrence_segment_query_ends,
                req.occurrence_segment_key_offsets,
                req.occurrence_segment_key_indices,
            )
            if not all(tensor is not None for tensor in plan):
                raise RuntimeError("Paged-occurrence request is missing its occurrence plan.")
            occurrence_raw, occurrence_positions, birth_ids, terminal_ids = plan[:4]
            assert occurrence_raw is not None
            assert occurrence_positions is not None
            assert birth_ids is not None
            assert terminal_ids is not None
            occurrence_count = len(occurrence_raw)
            plan_token_count = len(birth_ids)

        if compact_layout is None:
            refresh_occurrence_plan()
        initial_resources_live = False
        match = None
        fallback_to_empty = False
        while True:
            if initial_allocation:
                match_started_ns = time.perf_counter_ns()
                match = (
                    self.cache_manager.match_empty_req(req)
                    if fallback_to_empty
                    else self.cache_manager.match_occurrence_req(req)
                )
                match_elapsed_ns = time.perf_counter_ns() - match_started_ns
                match_retry_plan_ns = 0 if match is None else match.retry_plan_ns
                req.radix_match_ns += max(0, match_elapsed_ns - match_retry_plan_ns)
                if match is None:
                    return None
                if match.retry_plan is not None or match.retry_plan_ns != 0:
                    raise RuntimeError("Paged-occurrence matching produced a staged Retry plan.")
                req.retry_plan_ns += match.retry_plan_ns
                cached_len = match.full_cached_len
                exact_full_cached_len = match.exact_full_cached_len
                if match.active_cached_len != cached_len:
                    raise RuntimeError(
                        "Paged-occurrence matching requires the complete full prefix."
                    )
                if cached_len >= req.input_len:
                    raise RuntimeError(
                        "Prefix matching must leave at least one occurrence query token."
                    )
                if compact_layout is not None:
                    if req.full_token_visible_until is None or req.radix_positions is None:
                        raise RuntimeError(
                            "Compact occurrence compilation requires visibility and terminal positions."
                        )
                    install_occurrence_plan(
                        req,
                        compile_occurrence_window(
                            compact_layout,
                            req.full_token_visible_until,
                            req.radix_positions,
                            query_start=cached_len,
                            query_end=req.input_len,
                        ),
                    )
                    refresh_occurrence_plan()
                cache_handle = match.handle
                table_idx: int | None = None
                cache_locked = False
                try:
                    self.cache_manager.lock(cache_handle)
                    cache_locked = True
                    table_idx = self.table_manager.allocate()
                    self.table_manager.prepare_occurrence(table_idx, plan_token_count)
                    source_pages = match.full_match_indices[:cached_len].clone()
                    matched_virtual = (
                        cache_handle.get_matched_virtual_mask()[: cache_handle.cached_len]
                        if cache_handle.cached_len > 0
                        else torch.empty(0, dtype=torch.bool, device="cpu")
                    )
                    source_records = (
                        cache_handle.get_matched_keys()[: cache_handle.cached_len]
                        if cache_handle.cached_len > 0
                        else torch.empty((0, 4), dtype=torch.int32, device="cpu")
                    )
                    source_positions = (
                        source_records[~matched_virtual, 3].to(dtype=torch.int32, device="cpu")
                        if cached_len > 0
                        else torch.empty(0, dtype=torch.int32, device="cpu")
                    )
                    if len(source_positions) != cached_len:
                        raise RuntimeError(
                            "Matched source positions do not cover the cached prefix."
                        )
                    terminal_positions = occurrence_positions[
                        terminal_ids[:cached_len].to(torch.int64)
                    ]
                    if not torch.equal(
                        source_positions[:exact_full_cached_len],
                        terminal_positions[:exact_full_cached_len],
                    ):
                        raise RuntimeError(
                            "Exact occurrence pages are not keyed by final positions."
                        )
                    same_position_retry_copy_count = int(
                        torch.count_nonzero(
                            source_positions[exact_full_cached_len:]
                            == terminal_positions[exact_full_cached_len:]
                        ).item()
                    )
                    terminal_owned = torch.zeros(plan_token_count, dtype=torch.bool, device="cpu")
                    birth_pages = torch.full(
                        (plan_token_count,),
                        -1,
                        dtype=torch.int32,
                        device=self.cache_manager.device,
                    )
                    birth_owned = torch.zeros(plan_token_count, dtype=torch.bool, device="cpu")
                    repositioned_cached = torch.zeros(cached_len, dtype=torch.bool, device="cpu")
                    usage_positions = None
                    usage_cached_tokens = None
                    usage_repos_tokens = None
                    radix_cached_tokens = cache_handle.physical_cached_len
                    full_prefix_len, _ = self.cache_manager.matchable_prefix_lens(req)
                    cache_reuse_ratio = _calculate_cache_reuse_ratio(cached_len, full_prefix_len)
                    table = self.table_manager.occurrence_pages(table_idx)
                    table[:plan_token_count].fill_(-1)
                    if cached_len > 0:
                        birth_pages[:cached_len].copy_(source_pages)
                        table[:cached_len].copy_(source_pages)
                        self.table_manager.occurrence_tokens(table_idx)[:cached_len].copy_(
                            req.input_ids[:cached_len].pin_memory(), non_blocking=True
                        )
                except Exception:
                    if table_idx is not None:
                        self.table_manager.free(table_idx)
                    if cache_locked:
                        self.cache_manager.unlock(cache_handle)
                    raise
                assert table_idx is not None
                initial_resources_live = True
            else:
                assert chunked_req is not None
                cached_len = chunked_req.cached_len
                cache_handle = chunked_req.cache_handle
                table_idx = chunked_req.table_idx
                source_pages = chunked_req.initial_full_match_indices
                source_positions = chunked_req.occurrence_initial_source_positions
                exact_full_cached_len = chunked_req.occurrence_exact_full_cached_len
                same_position_retry_copy_count = (
                    chunked_req.occurrence_same_position_retry_copy_count
                )
                terminal_owned = (
                    None
                    if chunked_req.occurrence_terminal_owned_mask is None
                    else chunked_req.occurrence_terminal_owned_mask.clone()
                )
                birth_pages = (
                    None
                    if chunked_req.occurrence_birth_pages is None
                    else chunked_req.occurrence_birth_pages.clone()
                )
                birth_owned = (
                    None
                    if chunked_req.occurrence_birth_owned_mask is None
                    else chunked_req.occurrence_birth_owned_mask.clone()
                )
                repositioned_cached = (
                    None
                    if chunked_req.occurrence_repositioned_cached_mask is None
                    else chunked_req.occurrence_repositioned_cached_mask.clone()
                )
                if (
                    source_positions is None
                    or terminal_owned is None
                    or birth_pages is None
                    or birth_owned is None
                    or repositioned_cached is None
                ):
                    raise RuntimeError("Chunked occurrence request lost its persistent metadata.")
                if exact_full_cached_len is None:
                    exact_full_cached_len = len(source_positions)
                usage_positions = chunked_req.context_usage_cached_positions
                usage_cached_tokens = chunked_req.usage_cached_tokens
                usage_repos_tokens = chunked_req.usage_repos_tokens
                radix_cached_tokens = chunked_req.radix_cached_tokens
                cache_reuse_ratio = chunked_req.cache_reuse_ratio

            try:
                max_end = min(req.input_len, cached_len + self.token_budget)
                best: tuple[torch.Tensor, int, int, int] | None = None
                best_end: int | None = None
                available_pages = self.cache_manager.available_size
                if max_end > cached_len:
                    assert occurrence_raw is not None
                    assert occurrence_positions is not None
                    assert birth_ids is not None
                    assert terminal_ids is not None
                    assert req.occurrence_segment_query_starts is not None
                    assert req.occurrence_segment_query_ends is not None
                    assert req.occurrence_segment_key_offsets is not None
                    assert req.occurrence_segment_key_indices is not None
                    final_keep = (
                        req.full_keep_mask.to(dtype=torch.bool, device="cpu").contiguous()
                        if req.full_keep_mask is not None
                        else torch.ones(
                            plan_token_count,
                            dtype=torch.bool,
                            device="cpu",
                        )
                    )
                    capacity_index = try_build_occurrence_capacity_index(
                        occurrence_raw,
                        occurrence_positions,
                        birth_ids,
                        terminal_ids,
                        req.occurrence_segment_query_starts,
                        req.occurrence_segment_query_ends,
                        req.occurrence_segment_key_offsets,
                        req.occurrence_segment_key_indices,
                        terminal_owned.contiguous(),
                        final_keep,
                        source_positions.contiguous(),
                        chunk_start=cached_len,
                        max_chunk_end=max_end,
                        output_len=req.output_len,
                    )

                    if capacity_index is not None:
                        first_required_end, current_curve, persistent_curve, future_curve = (
                            capacity_index
                        )

                        def indexed_capacity(end: int) -> tuple[int, int, int]:
                            index = end - cached_len - 1
                            final_chunk = end == plan_token_count
                            return (
                                int(current_curve[index])
                                + (same_position_retry_copy_count if final_chunk else 0),
                                int(persistent_curve[index])
                                + (same_position_retry_copy_count if final_chunk else 0),
                                int(future_curve[index])
                                + (0 if final_chunk else same_position_retry_copy_count),
                            )

                        # Most scheduling attempts can consume the whole token-budget
                        # window. Test that endpoint first and only search when it does
                        # not fit; every later predicate lookup is O(1).
                        current_pages, persistent_pages, future_pages = indexed_capacity(max_end)
                        required_pages = max(
                            current_pages,
                            persistent_pages + future_pages,
                        )
                        if required_pages + self.reserved_size <= available_pages:
                            best_end = max_end
                        else:
                            low = cached_len + 1
                            high = max_end - 1
                            while low <= high:
                                candidate = (low + high) // 2
                                current_pages, persistent_pages, future_pages = indexed_capacity(
                                    candidate
                                )
                                required_pages = max(
                                    current_pages,
                                    persistent_pages + future_pages,
                                )
                                if required_pages + self.reserved_size <= available_pages:
                                    best_end = candidate
                                    low = candidate + 1
                                else:
                                    high = candidate - 1
                        if best_end is not None:
                            current_pages, persistent_pages, future_pages = indexed_capacity(
                                best_end
                            )
                            required = (
                                torch.nonzero(
                                    first_required_end <= best_end,
                                    as_tuple=False,
                                )
                                .view(-1)
                                .to(torch.int64)
                            )
                            if len(required) == 0:
                                raise RuntimeError(
                                    "Occurrence chunk has no physical page requirements."
                                )
                            best = (
                                required,
                                current_pages,
                                persistent_pages,
                                future_pages,
                            )
                    else:
                        # Keep a semantics-identical reference fallback for platforms
                        # where the CPU AOT planner cannot be loaded.
                        capacity_cache: dict[int, tuple[torch.Tensor, int, int, int]] = {}

                        def reference_capacity(
                            end: int,
                        ) -> tuple[torch.Tensor, int, int, int]:
                            capacity = capacity_cache.get(end)
                            if capacity is None:
                                capacity = self._occurrence_capacity_for_chunk(
                                    req,
                                    start=cached_len,
                                    end=end,
                                    terminal_owned=terminal_owned,
                                    initial_source_positions=source_positions,
                                    exact_full_cached_len=exact_full_cached_len,
                                )
                                capacity_cache[end] = capacity
                            return capacity

                        capacity = reference_capacity(max_end)
                        _, current_pages, persistent_pages, future_pages = capacity
                        required_pages = max(
                            current_pages,
                            persistent_pages + future_pages,
                        )
                        if required_pages + self.reserved_size <= available_pages:
                            best = capacity
                            best_end = max_end
                        else:
                            low = cached_len + 1
                            high = max_end - 1
                            while low <= high:
                                candidate = (low + high) // 2
                                capacity = reference_capacity(candidate)
                                _, current_pages, persistent_pages, future_pages = capacity
                                required_pages = max(
                                    current_pages,
                                    persistent_pages + future_pages,
                                )
                                if required_pages + self.reserved_size <= available_pages:
                                    best = capacity
                                    best_end = candidate
                                    low = candidate + 1
                                else:
                                    high = candidate - 1
            except Exception:
                if initial_resources_live:
                    self.table_manager.free(table_idx)
                    self.cache_manager.unlock(cache_handle)
                    initial_resources_live = False
                raise

            if best is not None:
                assert best_end is not None
                chunk_end = best_end
                break
            if initial_resources_live:
                self.table_manager.free(table_idx)
                self.cache_manager.unlock(cache_handle)
                initial_resources_live = False
                if cached_len > 0 and not fallback_to_empty:
                    fallback_to_empty = True
                    continue
            minimum = self._occurrence_capacity_for_chunk(
                req,
                start=cached_len,
                end=cached_len + 1,
                terminal_owned=terminal_owned,
                initial_source_positions=source_positions,
                exact_full_cached_len=exact_full_cached_len,
            )
            _, current_pages, persistent_pages, future_pages = minimum
            minimum_pages = max(current_pages, persistent_pages + future_pages)
            self_pinned_pages = len(torch.unique(source_pages)) + int(
                torch.count_nonzero(terminal_owned).item()
            )
            if bool(torch.any(birth_owned).item()):
                owned_birth_pages = birth_pages[
                    birth_owned.pin_memory().to(self.cache_manager.device, non_blocking=True)
                ]
                terminal_pages = self.table_manager.occurrence_pages(table_idx)[
                    : len(terminal_owned)
                ][terminal_owned.pin_memory().to(self.cache_manager.device, non_blocking=True)]
                self_pinned_pages = len(
                    torch.unique(torch.cat((source_pages, terminal_pages, owned_birth_pages)))
                )
            theoretical_available_pages = self.cache_manager.num_pages - self_pinned_pages
            if self.reserved_size > 0 or minimum_pages <= theoretical_available_pages:
                # Pages protected by other running requests are transient pressure.
                # Keep this request pending so normal Decode/Prefill completion can
                # release them instead of turning pressure into a terminal failure.
                return None
            raise RepositionCapacityError(
                uid=req.uid,
                required_pages=minimum_pages,
                available_pages=available_pages,
                matched_pages=cache_handle.physical_cached_len,
                retry_pages=current_pages - 1,
            )

        required_ids, allocation_count, _, future_reserve = best
        required_raw = occurrence_raw[required_ids].to(torch.int64)
        required_positions = occurrence_positions[required_ids]
        prior = required_raw < cached_len
        prior_raw = required_raw[prior]
        prior_birth_ids = birth_ids[prior_raw].to(torch.int64)
        canonical_positions = occurrence_positions[prior_birth_ids]
        canonical_pages = torch.empty(
            len(prior_raw), dtype=torch.int32, device=self.cache_manager.device
        )
        if len(prior_raw) > 0:
            prior_raw_device = prior_raw.pin_memory().to(
                self.cache_manager.device, non_blocking=True
            )
            matched_prior = prior_raw < len(source_positions)
            canonical_positions[matched_prior] = source_positions[
                prior_raw[matched_prior]
            ]
            reuse_terminal = terminal_owned[prior_raw] & (
                required_positions[prior]
                == occurrence_positions[terminal_ids[prior_raw].to(torch.int64)]
            )
            canonical_positions[reuse_terminal] = required_positions[prior][reuse_terminal]
            birth_source = ~reuse_terminal
            birth_source_device = birth_source.pin_memory().to(
                self.cache_manager.device, non_blocking=True
            )
            if bool(torch.any(birth_source).item()):
                canonical_pages[birth_source_device] = birth_pages[
                    prior_raw_device[birth_source_device]
                ]
            if bool(torch.any(reuse_terminal).item()):
                terminal_source_device = reuse_terminal.pin_memory().to(
                    self.cache_manager.device, non_blocking=True
                )
                canonical_pages[terminal_source_device] = self.table_manager.occurrence_pages(
                    table_idx
                )[prior_raw_device[terminal_source_device]]
            if bool(torch.any(canonical_pages < 0).item()):
                raise RuntimeError("A computed occurrence token has no retained KV page.")
        prior_new = required_positions[prior] != canonical_positions
        if chunk_end == plan_token_count:
            retry_source = (prior_raw >= exact_full_cached_len) & (
                prior_raw < len(source_positions)
            )
            terminal_required = required_ids[prior] == terminal_ids[prior_raw]
            prior_new |= retry_source & terminal_required & (~terminal_owned[prior_raw])
        new_mask = torch.zeros(len(required_ids), dtype=torch.bool, device="cpu")
        new_mask[prior] = prior_new
        new_mask[~prior] = True
        if int(torch.count_nonzero(new_mask).item()) != allocation_count:
            raise RuntimeError("Occurrence capacity estimate disagrees with page materialization.")

        allocated_pages = torch.empty(0, dtype=torch.int32, device=self.cache_manager.device)
        try:
            allocated_pages = self.cache_manager.allocate_occurrence_pages(allocation_count)
            required_device = required_ids.pin_memory().to(
                self.cache_manager.device, non_blocking=True
            )
            new_device = new_mask.pin_memory().to(self.cache_manager.device, non_blocking=True)
            runtime_pages = torch.full(
                (occurrence_count,), -1, dtype=torch.int32, device=self.cache_manager.device
            )
            runtime_pages[required_device[new_device]] = allocated_pages
            if len(prior_raw) > 0:
                reuse_prior = ~prior_new
                if bool(torch.any(reuse_prior).item()):
                    prior_required = (
                        required_ids[prior][reuse_prior]
                        .pin_memory()
                        .to(self.cache_manager.device, non_blocking=True)
                    )
                    reuse_device = reuse_prior.pin_memory().to(
                        self.cache_manager.device, non_blocking=True
                    )
                    runtime_pages[prior_required] = canonical_pages[reuse_device]

            allocated_ids = required_ids[new_mask]
            allocated_raw = occurrence_raw[allocated_ids].to(torch.int64)
            terminal_for_allocated = terminal_ids[allocated_raw].to(torch.int64)
            persistent_allocated = allocated_ids == terminal_for_allocated
            persistent_ids = allocated_ids[persistent_allocated]
            persistent_raw = allocated_raw[persistent_allocated]
            persistent_pages = runtime_pages[
                persistent_ids.pin_memory().to(self.cache_manager.device, non_blocking=True)
            ]

            cached_transform = prior_new
            cached_ids = required_ids[prior][cached_transform]
            cached_raw = prior_raw[cached_transform]
            cached_source_pages = torch.empty(
                0, dtype=torch.int32, device=self.cache_manager.device
            )
            cached_destination_pages = torch.empty(
                0, dtype=torch.int32, device=self.cache_manager.device
            )
            cached_position_pairs = torch.empty(
                (0, 2), dtype=torch.int32, device=self.cache_manager.device
            )
            if len(cached_ids) > 0:
                cached_reuse_device = cached_transform.pin_memory().to(
                    self.cache_manager.device, non_blocking=True
                )
                cached_source_pages = canonical_pages[cached_reuse_device]
                cached_destination_pages = runtime_pages[
                    cached_ids.pin_memory().to(self.cache_manager.device, non_blocking=True)
                ]
                cached_pairs_cpu = torch.column_stack(
                    (canonical_positions[cached_transform], occurrence_positions[cached_ids])
                ).to(torch.int32)
                self._validate_occurrence_rope_positions(cached_pairs_cpu)
                cached_position_pairs = cached_pairs_cpu.pin_memory().to(
                    self.cache_manager.device, non_blocking=True
                )
                req.reposition_h2d_bytes += (
                    cached_pairs_cpu.numel() * cached_pairs_cpu.element_size()
                )

            fresh = required_raw >= cached_len
            fresh_ids = required_ids[fresh]
            fresh_raw = required_raw[fresh]
            fresh_birth_ids = birth_ids[fresh_raw].to(torch.int64)
            fresh_transition = fresh_ids != fresh_birth_ids
            transition_ids = fresh_ids[fresh_transition]
            transition_raw = fresh_raw[fresh_transition]
            transition_birth_ids = birth_ids[transition_raw].to(torch.int64)
            fresh_pairs_cpu = torch.column_stack(
                (
                    occurrence_positions[transition_birth_ids],
                    occurrence_positions[transition_ids],
                )
            ).to(torch.int32)
            self._validate_occurrence_rope_positions(fresh_pairs_cpu)
            fresh_source_pages = runtime_pages[
                transition_birth_ids.pin_memory().to(self.cache_manager.device, non_blocking=True)
            ]
            fresh_destination_pages = runtime_pages[
                transition_ids.pin_memory().to(self.cache_manager.device, non_blocking=True)
            ]
            fresh_position_pairs = fresh_pairs_cpu.pin_memory().to(
                self.cache_manager.device, non_blocking=True
            )
            req.reposition_h2d_bytes += fresh_pairs_cpu.numel() * fresh_pairs_cpu.element_size()
            transform_source_pages = torch.cat((cached_source_pages, fresh_source_pages))
            transform_destination_pages = torch.cat(
                (cached_destination_pages, fresh_destination_pages)
            )
            transform_position_pairs = torch.cat((cached_position_pairs, fresh_position_pairs))

            fresh_birth_occurrences = birth_ids[fresh_raw].to(torch.int64)
            fresh_birth_pages = runtime_pages[
                fresh_birth_occurrences.pin_memory().to(
                    self.cache_manager.device, non_blocking=True
                )
            ]
            birth_pages = birth_pages.clone()
            birth_owned = birth_owned.clone()
            birth_pages.index_copy_(
                0,
                fresh_raw.pin_memory().to(self.cache_manager.device, non_blocking=True),
                fresh_birth_pages,
            )
            birth_owned[fresh_raw] = True

            retained_mask = torch.zeros(occurrence_count, dtype=torch.bool, device="cpu")
            retained_mask[persistent_ids] = True
            retained_mask[fresh_birth_occurrences] = True
            transient_ids = allocated_ids[~retained_mask[allocated_ids]]
            transient_pages = runtime_pages[
                transient_ids.pin_memory().to(self.cache_manager.device, non_blocking=True)
            ]
            if len(cached_raw) > 0:
                changed_initial = cached_raw[
                    (cached_raw < len(repositioned_cached))
                    & (cached_pairs_cpu[:, 0] != cached_pairs_cpu[:, 1])
                ]
                repositioned_cached[changed_initial] = True
            if len(persistent_ids) > 0:
                persistent_raw_device = persistent_raw.pin_memory().to(
                    self.cache_manager.device, non_blocking=True
                )
                self.table_manager.occurrence_pages(table_idx).index_copy_(
                    0, persistent_raw_device, persistent_pages
                )
                terminal_owned[persistent_raw] = True
        except Exception:
            self.cache_manager.free_occurrence_pages(allocated_pages)
            if initial_resources_live:
                self.table_manager.free(table_idx)
                self.cache_manager.unlock(cache_handle)
                initial_resources_live = False
            raise

        return PrefillAllocation(
            cache_handle=cache_handle,
            table_idx=table_idx,
            cache_reuse_ratio=cache_reuse_ratio,
            initial_full_match_indices=source_pages,
            cached_len=cached_len,
            radix_cached_tokens=radix_cached_tokens,
            usage_cached_tokens=usage_cached_tokens,
            usage_repos_tokens=usage_repos_tokens,
            retry_transformed_mask=None,
            inactive_cached_positions=None,
            inactive_cached_pages=None,
            chunk_size=chunk_end - cached_len,
            reserved_pages=future_reserve,
            context_usage_cached_positions=usage_positions,
            occurrence_pages=runtime_pages,
            occurrence_transient_pages=transient_pages,
            occurrence_birth_pages=birth_pages,
            occurrence_birth_owned_mask=birth_owned,
            occurrence_transform_source_pages=transform_source_pages,
            occurrence_transform_destination_pages=transform_destination_pages,
            occurrence_transform_position_pairs=transform_position_pairs,
            occurrence_terminal_owned_mask=terminal_owned,
            occurrence_initial_source_positions=source_positions,
            occurrence_exact_full_cached_len=exact_full_cached_len,
            occurrence_same_position_retry_copy_count=same_position_retry_copy_count,
            occurrence_repositioned_cached_mask=repositioned_cached,
            occurrence_allocated_pages=allocated_pages,
        )

    def _validate_occurrence_rope_positions(self, position_pairs: torch.Tensor) -> None:
        if len(position_pairs) == 0:
            return
        assert self.retry_rope_cache is not None
        if int(torch.min(position_pairs).item()) < 0 or int(
            torch.max(position_pairs).item()
        ) >= len(self.retry_rope_cache):
            raise RuntimeError("Paged-occurrence position exceeds the RoPE cache.")

    def plan_context_prefill(self, req: PendingReq) -> ContextPrefillPlan | None:
        if not req.use_context_mask or req.chunked_req is not None:
            return None
        structured_retry = (
            req.context_compact_stream
            and req.radix_match_ids is not None
            and req.radix_match_ids.ndim == 2
        )
        retry_plan = None
        retry_active_full_positions = None
        if structured_retry:
            match_started_ns = time.perf_counter_ns()
            active_match = self.cache_manager.match_req(req)
            match_elapsed_ns = time.perf_counter_ns() - match_started_ns
            match_retry_plan_ns = 0 if active_match is None else active_match.retry_plan_ns
            req.radix_match_ns += max(0, match_elapsed_ns - match_retry_plan_ns)
            if active_match is None:
                return None
            req.retry_plan_ns += active_match.retry_plan_ns
            radix_cached_tokens = active_match.handle.physical_cached_len
            retry_plan = active_match.retry_plan
            retry_active_full_positions = active_match.active_full_positions
        else:
            full_match = self.cache_manager.match_full_req(req)
            if full_match is None:
                return None
            active_match = self.cache_manager.derive_active_match(req, full_match)
            radix_cached_tokens = full_match.handle.physical_cached_len
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
            if active_match.active_cached_len > radix_cached_tokens:
                raise RuntimeError("Active cache usage exceeds resident Radix matches.")
            logger.debug(
                "Context request %s selected mask-free Extend with %d active cache hits.",
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
                retry_plan=retry_plan,
                retry_active_full_positions=retry_active_full_positions,
            )

        if req.context_compact_stream:
            logger.debug(
                "Context request %s retained compact mask Prefill: %s.",
                req.uid,
                fallback_reason,
            )
            return ContextPrefillPlan(
                use_context_mask=True,
                input_ids=req.input_ids,
                true_positions=req.true_positions,
                raw_positions=req.raw_positions,
                radix_input_ids=req.radix_input_ids,
                cache_handle=active_match.handle,
                cached_indices=active_match.active_match_indices,
                cached_len=active_match.active_cached_len,
                initial_full_match_indices=active_match.full_match_indices,
                reason=fallback_reason,
                radix_cached_tokens=radix_cached_tokens,
                usage_cached_tokens=None,
                retry_plan=retry_plan,
                retry_active_full_positions=retry_active_full_positions,
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
            "Context request %s retained mask Prefill: %s.",
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
        if req.reposition_execution_mode == "paged-occurrence":
            if context_plan is not None:
                raise RuntimeError("Paged-occurrence Reposition cannot use the mask planner.")
            return self._try_allocate_occurrence(req)
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
            retry_plan = context_plan.retry_plan
            retry_active_full_positions = context_plan.retry_active_full_positions
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
            cache_handle = match.handle
            cached_len = match.active_cached_len
            cached_indices = match.active_match_indices
            initial_full_match_indices = match.full_match_indices[: match.full_cached_len]
            radix_cached_tokens = match.handle.physical_cached_len
            usage_cached_tokens = cached_len
            retry_plan = match.retry_plan
            retry_active_full_positions = match.active_full_positions
        full_prefix_len, active_prefix_len = self.cache_manager.matchable_prefix_lens(req)
        is_reposition = req.radix_positions is not None and req.radix_repos_info is not None
        original_matched_pages = cache_handle.physical_cached_len
        empty_fallback_used = False

        def restore_context_stream() -> None:
            if original_stream is not None:
                (
                    req.input_ids,
                    req.true_positions,
                    req.raw_positions,
                    req.radix_input_ids,
                    req.use_context_mask,
                ) = original_stream

        while True:
            cache_reuse_ratio = _calculate_cache_reuse_ratio(
                cached_len,
                full_prefix_len if req.use_context_mask else active_prefix_len,
            )
            # TODO: better estimate policy
            extend_len = req.input_len - cached_len
            retry_transformed_mask = None
            retry_page_count = 0 if retry_plan is None else len(retry_plan)
            estimated_len = extend_len + req.output_len + retry_page_count
            available_pages = self.cache_manager.available_size
            if estimated_len + self.reserved_size <= available_pages:
                self.cache_manager.lock(cache_handle)
                available_pages = self.cache_manager.available_size
                if estimated_len + self.reserved_size <= available_pages:
                    break
                self.cache_manager.unlock(cache_handle)

            # Ordinary scheduler pressure is transient. Let Decode or an earlier
            # Prefill batch release its reservation before trying this request.
            if self.reserved_size > 0:
                restore_context_stream()
                return None

            # A large matched prefix can become the only thing preventing its
            # own Reposition step from allocating the extension/retry pages.
            # Unlock it and recompute the stage from the empty root so normal
            # cache eviction can reclaim those pages.
            if is_reposition and cached_len > 0 and not empty_fallback_used:
                empty_match = self.cache_manager.match_empty_req(req)
                cache_handle = empty_match.handle
                cached_len = 0
                cached_indices = empty_match.active_match_indices
                initial_full_match_indices = empty_match.full_match_indices
                radix_cached_tokens = 0
                usage_cached_tokens = None if req.use_context_mask else 0
                retry_plan = None
                retry_active_full_positions = None
                empty_fallback_used = True
                continue

            restore_context_stream()
            total_capacity = self.cache_manager.num_pages * self.cache_manager.page_size
            if is_reposition and estimated_len > total_capacity:
                raise RepositionCapacityError(
                    uid=req.uid,
                    required_pages=estimated_len,
                    available_pages=available_pages,
                    matched_pages=original_matched_pages,
                    retry_pages=retry_page_count,
                )
            # This request fits after pages protected by another running request
            # are released. Leave it pending and let the normal scheduler advance
            # Decode/Prefill rather than rejecting transient pressure.
            return None

        table_idx: int | None = None
        retry_pages = torch.empty(0, dtype=torch.int32, device=cached_indices.device)
        retry_inactive_positions = None
        retry_inactive_pages = None
        try:
            table_idx = self.table_manager.allocate()
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
            if table_idx is not None:
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

        assert table_idx is not None
        return PrefillAllocation(
            cache_handle=cache_handle,
            table_idx=table_idx,
            cache_reuse_ratio=cache_reuse_ratio,
            initial_full_match_indices=initial_full_match_indices.clone(),
            cached_len=cached_len,
            radix_cached_tokens=radix_cached_tokens,
            usage_cached_tokens=usage_cached_tokens,
            usage_repos_tokens=None,
            retry_transformed_mask=retry_transformed_mask,
            inactive_cached_positions=retry_inactive_positions,
            inactive_cached_pages=retry_inactive_pages,
        )

    def _add_one_req(self, **kwargs) -> Req:
        """Transfer an allocation to Req, or roll it back before propagating failure."""
        allocated = kwargs.pop("occurrence_allocated_pages", None)
        pending = kwargs["pending_req"]
        budget, reserved = self.token_budget, self.reserved_size
        try:
            return self._construct_req(**kwargs)
        except Exception as exc:
            self.token_budget, self.reserved_size = budget, reserved
            if allocated is not None:
                slot = kwargs["table_idx"]
                table = self.table_manager.occurrence_pages(slot)
                owned = kwargs["occurrence_terminal_owned_mask"]
                # Error-only rollback. Never release borrowed source pages.
                newly_owned = torch.isin(table[: len(owned)], allocated).cpu()
                owned[newly_owned] = False
                table[: len(owned)][newly_owned.to(table.device)] = -1
                self.cache_manager.free_occurrence_pages(allocated)
                if pending.chunked_req is None:
                    self.cache_manager.unlock(kwargs["cache_handle"])
                    self.table_manager.free(slot)
                if isinstance(exc, ValueError):
                    raise OccurrenceInputError(pending.uid, str(exc)) from exc
            raise

    def _construct_req(
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
        inactive_cached_positions: torch.Tensor | None,
        inactive_cached_pages: torch.Tensor | None,
        usage_repos_tokens: int | None = None,
        occurrence_pages: torch.Tensor | None = None,
        occurrence_transient_pages: torch.Tensor | None = None,
        occurrence_birth_pages: torch.Tensor | None = None,
        occurrence_birth_owned_mask: torch.Tensor | None = None,
        occurrence_transform_source_pages: torch.Tensor | None = None,
        occurrence_transform_destination_pages: torch.Tensor | None = None,
        occurrence_transform_position_pairs: torch.Tensor | None = None,
        occurrence_terminal_owned_mask: torch.Tensor | None = None,
        occurrence_initial_source_positions: torch.Tensor | None = None,
        occurrence_exact_full_cached_len: int | None = None,
        occurrence_same_position_retry_copy_count: int = 0,
        occurrence_repositioned_cached_mask: torch.Tensor | None = None,
        context_usage_cached_positions: torch.Tensor | None = None,
        chunk_size_override: int | None = None,
        occurrence_reserved_pages: int | None = None,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        is_occurrence = pending_req.reposition_execution_mode == "paged-occurrence"
        chunk_size = (
            chunk_size_override
            if chunk_size_override is not None
            else min(self.token_budget, remain_len)
        )
        if not 0 < chunk_size <= min(self.token_budget, remain_len):
            raise RuntimeError("Prefill allocation returned an invalid query chunk size.")
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        if is_occurrence:
            if occurrence_reserved_pages is None:
                raise RuntimeError("Occurrence allocation omitted its future KV reservation.")
            self.reserved_size += occurrence_reserved_pages
        else:
            self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = (
            self.table_manager.occurrence_tokens(table_idx)[_slice]
            if is_occurrence
            else self.table_manager.token_pool[table_idx, _slice]
        )
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            occurrence_external_storage=(
                is_occurrence and self.table_manager.has_occurrence_storage(table_idx)
            ),
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
            usage_repos_tokens=(
                usage_repos_tokens
                if usage_repos_tokens is not None
                else (
                    pending_req.chunked_req.usage_repos_tokens
                    if pending_req.chunked_req is not None
                    else None
                )
            ),
            context_usage_cached_positions=context_usage_cached_positions,
            drop_skipped_tokens=(
                radix_cached_tokens - usage_cached_tokens if usage_cached_tokens is not None else 0
            ),
            full_input_ids=(pending_req.full_input_ids if pending_req.use_context_mask else None),
            full_token_visible_until=(
                pending_req.full_token_visible_until if pending_req.use_context_mask else None
            ),
            full_keep_mask=(pending_req.full_keep_mask if pending_req.use_context_mask else None),
            use_context_mask=pending_req.use_context_mask,
            context_compact_stream=pending_req.context_compact_stream,
            context_post_prefill_keep_mask=(
                pending_req.context_post_prefill_keep_mask if pending_req.use_context_mask else None
            ),
            radix_key_virtual_mask=pending_req.radix_key_virtual_mask,
            radix_key_to_token=pending_req.radix_key_to_token,
            radix_token_to_key=pending_req.radix_token_to_key,
            radix_commit_key_len=pending_req.radix_commit_key_len,
            radix_positions=pending_req.radix_positions,
            radix_repos_info=pending_req.radix_repos_info,
            radix_next_position=pending_req.radix_next_position,
            radix_current_reposition=pending_req.radix_current_reposition,
            retry_transformed_mask=retry_transformed_mask,
            inactive_cached_positions=inactive_cached_positions,
            inactive_cached_pages=inactive_cached_pages,
            tokenize_invocations=pending_req.tokenize_invocations,
            radix_compile_ns=pending_req.radix_compile_ns,
            radix_match_ns=pending_req.radix_match_ns,
            retry_plan_ns=pending_req.retry_plan_ns,
            reposition_transition_count=pending_req.reposition_transition_count,
            reposition_h2d_bytes=pending_req.reposition_h2d_bytes,
            reposition_d2h_bytes=pending_req.reposition_d2h_bytes,
            reposition_execution_mode=pending_req.reposition_execution_mode,
            occurrence_raw_tokens=pending_req.occurrence_raw_tokens,
            occurrence_positions=pending_req.occurrence_positions,
            occurrence_birth_indices=pending_req.occurrence_birth_indices,
            occurrence_terminal_indices=pending_req.occurrence_terminal_indices,
            occurrence_segment_query_starts=pending_req.occurrence_segment_query_starts,
            occurrence_segment_query_ends=pending_req.occurrence_segment_query_ends,
            occurrence_segment_key_offsets=pending_req.occurrence_segment_key_offsets,
            occurrence_segment_key_indices=pending_req.occurrence_segment_key_indices,
            occurrence_pages=occurrence_pages,
            occurrence_transient_pages=occurrence_transient_pages,
            occurrence_birth_pages=occurrence_birth_pages,
            occurrence_birth_owned_mask=occurrence_birth_owned_mask,
            occurrence_transform_source_pages=occurrence_transform_source_pages,
            occurrence_transform_destination_pages=occurrence_transform_destination_pages,
            occurrence_transform_position_pairs=occurrence_transform_position_pairs,
            occurrence_terminal_owned_mask=occurrence_terminal_owned_mask,
            occurrence_initial_source_positions=occurrence_initial_source_positions,
            occurrence_exact_full_cached_len=occurrence_exact_full_cached_len,
            occurrence_same_position_retry_copy_count=occurrence_same_position_retry_copy_count,
            occurrence_repositioned_cached_mask=occurrence_repositioned_cached_mask,
        )

    def try_add_one(
        self,
        pending_req: PendingReq,
        context_plan: ContextPrefillPlan | None = None,
    ) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            if (
                pending_req.reposition_execution_mode == "paged-occurrence"
                and chunked_req.occurrence_inflight
            ):
                return None
            if pending_req.reposition_execution_mode == "paged-occurrence":
                resource = self._try_allocate_occurrence(pending_req, chunked_req)
                if resource is None:
                    return None
                return self._add_one_req(
                    pending_req=pending_req,
                    cache_handle=resource.cache_handle,
                    table_idx=resource.table_idx,
                    cached_len=resource.cached_len,
                    cache_reuse_ratio=resource.cache_reuse_ratio,
                    initial_full_match_indices=resource.initial_full_match_indices,
                    initial_active_cached_len=chunked_req.initial_active_cached_len,
                    radix_cached_tokens=resource.radix_cached_tokens,
                    usage_cached_tokens=resource.usage_cached_tokens,
                    retry_transformed_mask=resource.retry_transformed_mask,
                    inactive_cached_positions=resource.inactive_cached_positions,
                    inactive_cached_pages=resource.inactive_cached_pages,
                    usage_repos_tokens=resource.usage_repos_tokens,
                    occurrence_pages=resource.occurrence_pages,
                    occurrence_transient_pages=resource.occurrence_transient_pages,
                    occurrence_birth_pages=resource.occurrence_birth_pages,
                    occurrence_birth_owned_mask=resource.occurrence_birth_owned_mask,
                    occurrence_transform_source_pages=(resource.occurrence_transform_source_pages),
                    occurrence_transform_destination_pages=(
                        resource.occurrence_transform_destination_pages
                    ),
                    occurrence_transform_position_pairs=(
                        resource.occurrence_transform_position_pairs
                    ),
                    occurrence_terminal_owned_mask=resource.occurrence_terminal_owned_mask,
                    occurrence_initial_source_positions=resource.occurrence_initial_source_positions,
                    occurrence_exact_full_cached_len=resource.occurrence_exact_full_cached_len,
                    occurrence_same_position_retry_copy_count=(
                        resource.occurrence_same_position_retry_copy_count
                    ),
                    occurrence_repositioned_cached_mask=(
                        resource.occurrence_repositioned_cached_mask
                    ),
                    context_usage_cached_positions=resource.context_usage_cached_positions,
                    chunk_size_override=resource.chunk_size,
                    occurrence_reserved_pages=resource.reserved_pages,
                    occurrence_allocated_pages=resource.occurrence_allocated_pages,
                )
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
                inactive_cached_positions=chunked_req.inactive_cached_positions,
                inactive_cached_pages=chunked_req.inactive_cached_pages,
                usage_repos_tokens=chunked_req.usage_repos_tokens,
            )
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
                inactive_cached_positions=resource.inactive_cached_positions,
                inactive_cached_pages=resource.inactive_cached_pages,
                usage_repos_tokens=resource.usage_repos_tokens,
                occurrence_pages=resource.occurrence_pages,
                occurrence_transient_pages=resource.occurrence_transient_pages,
                occurrence_birth_pages=resource.occurrence_birth_pages,
                occurrence_birth_owned_mask=resource.occurrence_birth_owned_mask,
                occurrence_transform_source_pages=resource.occurrence_transform_source_pages,
                occurrence_transform_destination_pages=(
                    resource.occurrence_transform_destination_pages
                ),
                occurrence_transform_position_pairs=resource.occurrence_transform_position_pairs,
                occurrence_terminal_owned_mask=resource.occurrence_terminal_owned_mask,
                occurrence_initial_source_positions=resource.occurrence_initial_source_positions,
                occurrence_exact_full_cached_len=resource.occurrence_exact_full_cached_len,
                occurrence_same_position_retry_copy_count=(
                    resource.occurrence_same_position_retry_copy_count
                ),
                occurrence_repositioned_cached_mask=(resource.occurrence_repositioned_cached_mask),
                context_usage_cached_positions=resource.context_usage_cached_positions,
                chunk_size_override=resource.chunk_size,
                occurrence_reserved_pages=resource.reserved_pages,
                occurrence_allocated_pages=resource.occurrence_allocated_pages,
            )
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

    def add_one_req(self, req: UserMsg) -> None:
        if req.use_context_mask:
            if not req.is_warmup and req.context_post_prefill_keep_mask is None:
                raise ValueError(
                    "Context-mask Prefill requires warmup or a final active keep mask."
                )
            if req.full_input_ids is None or req.radix_match_ids is None:
                raise ValueError(
                    "Context-mask Prefill requires a full token stream and Radix keys."
                )
        occurrence_fields = pack_compact_occurrence_pending_fields(
            birth_positions=req.occurrence_layout_birth_positions,
            birth_stages=req.occurrence_layout_birth_stages,
            transition_offsets=req.occurrence_layout_transition_offsets,
            transition_raw_tokens=req.occurrence_layout_transition_raw_tokens,
            transition_old_positions=req.occurrence_layout_transition_old_positions,
            transition_new_positions=req.occurrence_layout_transition_new_positions,
        )
        if not occurrence_fields:
            occurrence_fields = {
                "occurrence_raw_tokens": req.occurrence_raw_tokens,
                "occurrence_positions": req.occurrence_positions,
                "occurrence_birth_indices": req.occurrence_birth_indices,
                "occurrence_terminal_indices": req.occurrence_terminal_indices,
                "occurrence_segment_query_starts": req.occurrence_segment_query_starts,
                "occurrence_segment_query_ends": req.occurrence_segment_query_ends,
                "occurrence_segment_key_offsets": req.occurrence_segment_key_offsets,
                "occurrence_segment_key_indices": req.occurrence_segment_key_indices,
            }
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
                context_compact_stream=req.context_compact_stream,
                context_post_prefill_keep_mask=req.context_post_prefill_keep_mask,
                radix_key_virtual_mask=req.radix_key_virtual_mask,
                radix_key_to_token=req.radix_key_to_token,
                radix_token_to_key=req.radix_token_to_key,
                radix_commit_key_len=req.radix_commit_key_len,
                radix_positions=req.radix_positions,
                radix_repos_info=req.radix_repos_info,
                radix_next_position=req.radix_next_position,
                radix_current_reposition=req.radix_current_reposition,
                tokenize_invocations=req.tokenize_invocations,
                radix_compile_ns=req.radix_compile_ns,
                radix_match_ns=req.radix_match_ns,
                retry_plan_ns=req.retry_plan_ns,
                reposition_transition_count=req.reposition_transition_count,
                reposition_h2d_bytes=req.reposition_h2d_bytes,
                reposition_d2h_bytes=req.reposition_d2h_bytes,
                reposition_execution_mode=req.reposition_execution_mode,
                **occurrence_fields,
            )
        )

    def complete_chunk(self, chunk: ChunkedReq) -> None:
        if chunk.reposition_execution_mode != "paged-occurrence":
            return
        if not chunk.occurrence_inflight or chunk.occurrence_transient_pages is None:
            raise RuntimeError("Completed occurrence chunk has no in-flight page metadata.")
        self.cache_manager.free_occurrence_pages(chunk.occurrence_transient_pages)
        chunk.occurrence_transient_pages = None
        chunk.occurrence_pages = None
        chunk.occurrence_transform_source_pages = None
        chunk.occurrence_transform_destination_pages = None
        chunk.occurrence_transform_position_pairs = None
        chunk.occurrence_inflight = False
        # Dropped terminal pages are still final Radix candidates. Do not
        # recycle them before the finished-request commit (or abort cleanup).

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
                if (
                    pending_req.use_context_mask
                    and pending_req.chunked_req is None
                    and pending_req.reposition_execution_mode != "paged-occurrence"
                )
                else None
            )
            if (
                pending_req.use_context_mask
                and pending_req.chunked_req is None
                and pending_req.reposition_execution_mode != "paged-occurrence"
            ):
                if context_plan is None:
                    break
                planned_context_mask = context_plan.use_context_mask
            else:
                planned_context_mask = pending_req.use_context_mask
            if len(reqs) > 0:
                first_uses_context_mask = reqs[0].use_context_mask
                if planned_context_mask != first_uses_context_mask:
                    break
                if pending_req.reposition_execution_mode != reqs[0].reposition_execution_mode:
                    break
                if planned_context_mask and not supports_multi_context_mask:
                    break
            try:
                req = adder.try_add_one(pending_req, context_plan)
            except (RepositionCapacityError, OccurrenceInputError):
                if not reqs:
                    raise
                break
            if req:
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
                if (
                    isinstance(req, ChunkedReq)
                    and req.reposition_execution_mode == "paged-occurrence"
                ):
                    # A partial occurrence request retains terminal pages between
                    # forwards. Finish it before starting another partial request,
                    # preventing mutually pinned working sets under page pressure.
                    # Requests that fit in one forward remain batchable.
                    break
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
                return req.chunked_req or req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
