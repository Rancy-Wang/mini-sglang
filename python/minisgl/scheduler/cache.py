from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from minisgl.kvcache.base import InsertResult

    from .utils import PendingReq


@dataclass(frozen=True)
class ContextMatchResult:
    handle: BaseCacheHandle
    full_match_indices: torch.Tensor
    full_cached_len: int
    active_match_indices: torch.Tensor
    active_cached_len: int
    initial_active_cached_len: int
    active_full_positions: torch.Tensor
    retry_plan: torch.Tensor | None = None
    retry_plan_ns: int = 0


@dataclass(frozen=True)
class FullMatchResult:
    handle: BaseCacheHandle
    full_match_indices: torch.Tensor
    full_cached_len: int
    safe_match_indices: torch.Tensor
    safe_cached_len: int


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        *,
        drop_aware_eviction: bool = False,
    ):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.drop_aware_eviction = drop_aware_eviction
        if drop_aware_eviction:
            from minisgl.kvcache.radix_cache import RadixPrefixCache

            if not isinstance(self.prefix_cache, RadixPrefixCache):
                raise ValueError("Drop-aware eviction requires the Radix prefix cache.")
            self.prefix_cache.enable_drop_aware_eviction()
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def bind_delta_marker_registry(self, registry) -> None:
        from minisgl.kvcache.radix_cache import RadixPrefixCache

        if isinstance(self.prefix_cache, RadixPrefixCache):
            self.prefix_cache.bind_delta_marker_registry(registry)

    def _matched_indices(self, handle: BaseCacheHandle) -> torch.Tensor:
        if handle.cached_len == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        return handle.get_matched_indices()

    def _match_and_prune_legacy_holes(
        self,
        radix_query: torch.Tensor,
        virtual_mask: torch.Tensor | None = None,
    ) -> tuple[BaseCacheHandle, torch.Tensor, torch.Tensor] | None:
        result = self.prefix_cache.match_prefix(radix_query, virtual_mask)
        handle = result.cuda_handle
        indices = self._matched_indices(handle)
        matched_virtual_mask = (
            handle.get_matched_virtual_mask()
            if handle.cached_len > 0
            else torch.empty(0, dtype=torch.bool, device="cpu")
        )
        value_virtual_mask = matched_virtual_mask.to(device=indices.device, non_blocking=True)
        holes = torch.nonzero((indices < 0) & (~value_virtual_mask), as_tuple=False).view(-1)
        if len(holes) == 0:
            return handle, indices, matched_virtual_mask

        if self.drop_aware_eviction:
            return handle, indices, matched_virtual_mask

        from minisgl.kvcache.radix_cache import RadixPrefixCache

        if not isinstance(self.prefix_cache, RadixPrefixCache):
            raise RuntimeError("A non-Radix prefix cache returned a negative page slot.")
        valid_prefix_len = int(holes[0].item())
        released = self.prefix_cache.prune_suffix(radix_query, valid_prefix_len, virtual_mask)
        if released is None:
            return None
        self._free(released)

        result = self.prefix_cache.match_prefix(radix_query, virtual_mask)
        handle = result.cuda_handle
        indices = self._matched_indices(handle)
        matched_virtual_mask = (
            handle.get_matched_virtual_mask()
            if handle.cached_len > 0
            else torch.empty(0, dtype=torch.bool, device="cpu")
        )
        value_virtual_mask = matched_virtual_mask.to(device=indices.device, non_blocking=True)
        if handle.cached_len != valid_prefix_len or bool(
            torch.any((indices < 0) & (~value_virtual_mask)).item()
        ):
            raise RuntimeError(
                "Radix legacy-hole pruning did not leave the expected safe prefix:"
                f" expected={valid_prefix_len}, actual={handle.cached_len}"
            )
        return handle, indices, matched_virtual_mask

    @staticmethod
    def _radix_query_prefix(
        req: PendingReq,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if req.radix_match_ids is None:
            raise ValueError("Prefix matching requires radix_match_ids.")
        if req.radix_token_to_key is None:
            return req.radix_match_ids[:-1], None
        if req.radix_key_virtual_mask is None:
            raise ValueError("Delta-marker key mapping requires a virtual mask.")
        last_token_pos = int(req.raw_positions[-1].item())
        if last_token_pos < 0 or last_token_pos >= len(req.radix_token_to_key):
            raise ValueError("The final input token is outside radix_token_to_key.")
        key_prefix_len = int(req.radix_token_to_key[last_token_pos].item())
        if req.radix_commit_key_len is not None:
            key_prefix_len = min(key_prefix_len, req.radix_commit_key_len)
        return (
            req.radix_match_ids[:key_prefix_len],
            req.radix_key_virtual_mask[:key_prefix_len],
        )

    @classmethod
    def matchable_prefix_lens(cls, req: PendingReq) -> tuple[int, int]:
        radix_query, query_virtual_mask = cls._radix_query_prefix(req)
        if query_virtual_mask is None:
            prefix_len = max(req.input_len - 1, 0)
            return prefix_len, prefix_len
        full_token_prefix_len = int(torch.count_nonzero(~query_virtual_mask).item())
        active_token_prefix_len = int(
            torch.count_nonzero(req.raw_positions < full_token_prefix_len).item()
        )
        return full_token_prefix_len, active_token_prefix_len

    def match_req(self, req: PendingReq) -> ContextMatchResult | None:
        assert req.input_len > 0, "Input length must be greater than 0."
        radix_query, query_virtual_mask = self._radix_query_prefix(req)
        matched = self._match_and_prune_legacy_holes(radix_query, query_virtual_mask)
        if matched is None:
            return None
        handle, key_match_indices, matched_virtual_mask = matched
        used_retry = False
        if req.radix_match_ids is not None and req.radix_match_ids.ndim == 2:
            from minisgl.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache

            if not isinstance(self.prefix_cache, RadixPrefixCache) or not isinstance(
                handle, RadixCacheHandle
            ):
                raise RuntimeError("Structured Retry requires the Radix prefix cache.")
            retry_handle = self.prefix_cache.match_retry_prefix(
                radix_query,
                (
                    query_virtual_mask
                    if query_virtual_mask is not None
                    else torch.zeros(len(radix_query), dtype=torch.bool, device="cpu")
                ),
                handle,
            )
            if retry_handle.cached_len > handle.cached_len:
                retry_indices = self._matched_indices(retry_handle)[: retry_handle.cached_len]
                retry_virtual = retry_handle.get_matched_virtual_mask()[: retry_handle.cached_len]
                value_virtual = retry_virtual.to(device=retry_indices.device, non_blocking=True)
                if not bool(torch.any((retry_indices < 0) & (~value_virtual)).item()):
                    handle = retry_handle
                    key_match_indices = retry_indices
                    matched_virtual_mask = retry_virtual
                    used_retry = True
        full_match_indices = key_match_indices[
            (~matched_virtual_mask).to(device=key_match_indices.device, non_blocking=True)
        ]
        result = self._derive_active_match(req, handle, full_match_indices)
        if not used_retry:
            return result

        if req.radix_key_to_token is None:
            raise RuntimeError("Structured Retry requires a target key-to-token mapping.")
        from minisgl.kernel.radix import fast_compare_retry_radix_records_plan

        source_records = handle.get_matched_keys()[: handle.cached_len]
        source_key_to_token = torch.full((handle.cached_len,), -1, dtype=torch.int64, device="cpu")
        source_key_to_token[~matched_virtual_mask] = torch.arange(
            result.full_cached_len, dtype=torch.int64, device="cpu"
        )
        retry_started_ns = time.perf_counter_ns()
        matched_len, retry_plan = fast_compare_retry_radix_records_plan(
            source_records,
            radix_query[: handle.cached_len],
            source_key_to_token,
            req.radix_key_to_token[: handle.cached_len],
        )
        retry_plan_ns = time.perf_counter_ns() - retry_started_ns
        if matched_len != handle.cached_len:
            raise RuntimeError("Retry plan compiler disagrees with the selected Radix path.")
        return ContextMatchResult(
            handle=result.handle,
            full_match_indices=result.full_match_indices,
            full_cached_len=result.full_cached_len,
            active_match_indices=result.active_match_indices,
            active_cached_len=result.active_cached_len,
            initial_active_cached_len=result.active_cached_len,
            active_full_positions=result.active_full_positions,
            retry_plan=retry_plan,
            retry_plan_ns=retry_plan_ns,
        )

    def _derive_active_match(
        self,
        req: PendingReq,
        handle: BaseCacheHandle,
        full_match_indices: torch.Tensor,
    ) -> ContextMatchResult:
        active_match_indices = full_match_indices
        active_full_positions = torch.arange(
            len(full_match_indices), dtype=torch.int64, device="cpu"
        )
        if req.prefix_keep_mask is not None and len(full_match_indices) > 0:
            if len(req.prefix_keep_mask) < len(full_match_indices):
                raise RuntimeError(
                    "prefix_keep_mask is shorter than matched full-token prefix:"
                    f" {len(req.prefix_keep_mask)} < {len(full_match_indices)}"
                )
            keep_mask_cpu = (req.prefix_keep_mask[: len(full_match_indices)] != 0).to(
                device="cpu", dtype=torch.bool
            )
            keep_mask = keep_mask_cpu.to(
                device=full_match_indices.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            if self.drop_aware_eviction:
                kept_indices = full_match_indices[keep_mask]
                kept_holes = torch.nonzero(kept_indices < 0, as_tuple=False).view(-1)
                active_cached_len = (
                    int(kept_holes[0].item()) if len(kept_holes) > 0 else len(kept_indices)
                )
                active_match_indices = kept_indices[:active_cached_len]
                active_full_positions = torch.nonzero(keep_mask_cpu, as_tuple=False).view(-1)[
                    :active_cached_len
                ]
            else:
                active_match_indices = full_match_indices[keep_mask]
                active_full_positions = torch.nonzero(keep_mask_cpu, as_tuple=False).view(-1)
        elif self.drop_aware_eviction and len(full_match_indices) > 0:
            holes = torch.nonzero(full_match_indices < 0, as_tuple=False).view(-1)
            active_cached_len = int(holes[0].item()) if len(holes) > 0 else len(full_match_indices)
            active_match_indices = full_match_indices[:active_cached_len]
            active_full_positions = active_full_positions[:active_cached_len]
        if self.drop_aware_eviction:
            from minisgl.kvcache.radix_cache import RadixPrefixCache

            assert isinstance(self.prefix_cache, RadixPrefixCache)
            handle = self.prefix_cache.with_pinned_slots(handle, active_match_indices)
        return ContextMatchResult(
            handle=handle,
            full_match_indices=full_match_indices,
            full_cached_len=len(full_match_indices),
            active_match_indices=active_match_indices,
            active_cached_len=len(active_match_indices),
            initial_active_cached_len=len(active_match_indices),
            active_full_positions=active_full_positions,
        )

    def allocate_retry_pages(self, count: int) -> torch.Tensor:
        if self.page_size != 1:
            raise RuntimeError("Retry Reposition requires page_size=1.")
        if count < 0:
            raise ValueError("Retry Reposition page count must be non-negative.")
        if count > self.available_size:
            raise RuntimeError(
                f"Retry Reposition needs {count} KV pages, but only {self.available_size} "
                "pages can be made available."
            )
        return self._allocate(count)

    def free_retry_pages(self, indices: torch.Tensor) -> None:
        self._free(indices)

    def derive_active_match(
        self, req: PendingReq, full_match: FullMatchResult
    ) -> ContextMatchResult:
        """Reuse one full Radix lookup to derive the resident active-token prefix."""

        return self._derive_active_match(
            req,
            full_match.handle,
            full_match.full_match_indices,
        )

    def match_full_req(self, req: PendingReq) -> FullMatchResult | None:
        assert req.input_len > 0, "Input length must be greater than 0."
        radix_query, query_virtual_mask = self._radix_query_prefix(req)
        matched = self._match_and_prune_legacy_holes(radix_query, query_virtual_mask)
        if matched is None:
            return None
        handle, key_match_indices, matched_virtual_mask = matched
        full_match_indices = key_match_indices[
            (~matched_virtual_mask).to(device=key_match_indices.device, non_blocking=True)
        ]
        safe_match_indices = full_match_indices
        if self.drop_aware_eviction and len(full_match_indices) > 0:
            holes = torch.nonzero(full_match_indices < 0, as_tuple=False).view(-1)
            safe_len = int(holes[0].item()) if len(holes) > 0 else len(full_match_indices)
            safe_match_indices = full_match_indices[:safe_len]
            from minisgl.kvcache.radix_cache import RadixPrefixCache

            assert isinstance(self.prefix_cache, RadixPrefixCache)
            handle = self.prefix_cache.with_pinned_slots(handle, safe_match_indices)
        return FullMatchResult(
            handle=handle,
            full_match_indices=full_match_indices,
            full_cached_len=len(full_match_indices),
            safe_match_indices=safe_match_indices,
            safe_cached_len=len(safe_match_indices),
        )

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        if self.drop_aware_eviction:
            if req.radix_key_virtual_mask is not None:
                if finished:
                    self._cache_finished_drop_aware_delta_req(req)
                return
            self._cache_drop_aware_linear_req(req, finished=finished)
            return
        if req.radix_key_virtual_mask is not None:
            if not finished:
                return
            self._cache_finished_delta_req(req)
            return
        sparse = len(req.radix_match_ids) != len(req.radix_input_ids)
        if req.use_context_mask:
            if not finished:
                return
            self._cache_finished_full_req(req)
            return
        if sparse:
            if not finished:
                return
            self._cache_finished_sparse_req(req)
            return
        self._cache_linear_req(req, finished=finished)

    def _cache_finished_drop_aware_delta_req(self, req: Req) -> None:
        from minisgl.kvcache.radix_cache import RadixPrefixCache

        if self.page_size != 1:
            raise RuntimeError("Drop-aware delta commit requires page_size=1.")
        if not isinstance(self.prefix_cache, RadixPrefixCache):
            raise RuntimeError("Drop-aware delta commit requires the Radix prefix cache.")
        assert req.radix_key_virtual_mask is not None
        assert req.radix_key_to_token is not None
        assert req.radix_token_to_key is not None

        old_handle = req.cache_handle
        all_active_indices = self.page_table[req.table_idx, : req.cached_len]
        all_active_positions = req.raw_positions[: req.cached_len].to(
            dtype=torch.int64, device="cpu"
        )
        key_prefix_len = self._delta_key_prefix_len(req)
        try:
            key_virtual_mask = req.radix_key_virtual_mask[:key_prefix_len]
            key_to_token = req.radix_key_to_token[:key_prefix_len]
            full_token_prefix_len = int(torch.count_nonzero(~key_virtual_mask).item())
            within_prefix = all_active_positions < full_token_prefix_len
            active_indices = all_active_indices[
                within_prefix.to(
                    device=all_active_indices.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            ]
            active_positions = all_active_positions[within_prefix]
            excluded_indices = all_active_indices[
                (~within_prefix).to(
                    device=all_active_indices.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            ]
            if req.initial_active_cached_len > len(active_indices):
                raise RuntimeError("Initial active cache prefix exceeds the commit boundary.")

            full_indices = torch.full(
                (full_token_prefix_len,),
                -1,
                dtype=torch.int32,
                device=active_indices.device,
            )
            initial_full_len = min(len(req.initial_full_match_indices), full_token_prefix_len)
            if initial_full_len > 0:
                full_indices[:initial_full_len] = req.initial_full_match_indices[:initial_full_len]

            active_positions_device = active_positions.to(
                device=active_indices.device, non_blocking=True
            )
            active_slots = torch.arange(
                len(active_indices), dtype=torch.int64, device=active_indices.device
            )
            initially_matched = active_slots < req.initial_active_cached_len
            if bool(torch.any(initially_matched).item()):
                matched_positions = active_positions_device[initially_matched]
                previous = full_indices[matched_positions]
                current = active_indices[initially_matched]
                if bool(torch.any(previous < 0).item()) or not torch.equal(previous, current):
                    raise RuntimeError("Matched Drop-aware tokens use different KV slots.")
            newly_computed = ~initially_matched
            if bool(torch.any(newly_computed).item()):
                computed_positions = active_positions_device[newly_computed]
                computed_indices = active_indices[newly_computed]
                missing = full_indices[computed_positions] < 0
                full_indices[computed_positions[missing]] = computed_indices[missing]

            keep_mask = torch.ones(full_token_prefix_len, dtype=torch.bool, device="cpu")
            if req.full_keep_mask is not None:
                # Context metadata describes the original prompt only. Decode
                # appends generated tokens to the Radix/token axes, and those
                # positions are always visible for the finished request.
                keep_len = min(len(req.full_keep_mask), full_token_prefix_len)
                keep_mask[:keep_len] = req.full_keep_mask[:keep_len] != 0
            else:
                if req.prefix_keep_mask is not None:
                    keep_len = min(len(req.prefix_keep_mask), full_token_prefix_len)
                    keep_mask[:keep_len] = req.prefix_keep_mask[:keep_len] != 0
            kept_device = keep_mask.to(device=full_indices.device, non_blocking=True)
            # Dropped slots are deliberately not pinned and may have been evicted
            # and reassigned after the initial structural match. Never reuse that
            # stale snapshot during the finished commit. Existing resident nodes
            # remain canonical inside commit_drop_prefix; holes stay holes.
            full_indices[~kept_device] = -1
            missing_kept = torch.nonzero(kept_device & (full_indices < 0), as_tuple=False).view(-1)
            if len(missing_kept) > 0:
                raise RuntimeError(
                    "Kept Drop-aware tokens are missing KV slots at commit: "
                    f"{missing_kept[:16].tolist()}"
                )

            key_indices = torch.full(
                (key_prefix_len,), -1, dtype=torch.int32, device=active_indices.device
            )
            real_key_mask = ~key_virtual_mask
            real_token_positions = key_to_token[real_key_mask]
            key_indices[real_key_mask.to(device=active_indices.device)] = full_indices[
                real_token_positions.to(device=active_indices.device)
            ]
            result = self.prefix_cache.commit_drop_prefix(
                req.radix_match_ids[:key_prefix_len],
                key_indices,
                key_virtual_mask,
                key_to_token,
                keep_mask,
            )

            newly_allocated = active_slots >= req.initial_active_cached_len
            active_key_positions = req.radix_token_to_key[active_positions]
            canonical_active = result.canonical_indices[
                active_key_positions.to(device=active_indices.device)
            ]
            adopted = canonical_active == active_indices
            self._free(
                torch.cat(
                    [
                        active_indices[newly_allocated & (~adopted)],
                        excluded_indices,
                    ]
                )
            )
        finally:
            self.unlock(old_handle)

    def _cache_drop_aware_linear_req(self, req: Req, *, finished: bool) -> None:
        from minisgl.kvcache.radix_cache import RadixPrefixCache

        if not isinstance(self.prefix_cache, RadixPrefixCache):
            raise RuntimeError("Drop-aware linear commit requires the Radix prefix cache.")
        if self.page_size != 1:
            raise RuntimeError("Drop-aware linear commit requires page_size=1.")
        old_handle = req.cache_handle
        candidates = self.page_table[req.table_idx, : req.cached_len].clone()
        input_ids = req.radix_input_ids[: req.cached_len]
        virtual_mask = torch.zeros(len(input_ids), dtype=torch.bool, device="cpu")
        key_to_token = torch.arange(len(input_ids), dtype=torch.int64, device="cpu")
        keep_mask = torch.ones(len(input_ids), dtype=torch.bool, device="cpu")
        try:
            result = self.prefix_cache.commit_drop_prefix(
                input_ids,
                candidates,
                virtual_mask,
                key_to_token,
                keep_mask,
            )
            canonical = result.canonical_indices
            slots = torch.arange(len(candidates), dtype=torch.int64, device=candidates.device)
            newly_allocated = slots >= req.initial_active_cached_len
            duplicates = newly_allocated & (canonical != candidates)
            self.page_table[req.table_idx, : req.cached_len].copy_(canonical)
            self.unlock(old_handle)
            self._free(candidates[duplicates])
            if not finished:
                new_handle = self.prefix_cache.with_pinned_slots(result.handle, canonical)
                req.cache_handle = new_handle
                self.lock(new_handle)
        except Exception:
            # The old handle remains the request's lease unless commit reached the explicit unlock.
            raise

    @staticmethod
    def _delta_key_prefix_len(req: Req) -> int:
        assert req.radix_token_to_key is not None
        if req.cached_len < len(req.input_ids):
            next_token_pos = int(req.raw_positions[req.cached_len].item())
            if next_token_pos < 0 or next_token_pos >= len(req.radix_token_to_key):
                raise RuntimeError("The next active token is outside radix_token_to_key.")
            key_prefix_len = int(req.radix_token_to_key[next_token_pos].item())
        else:
            last_token_pos = int(req.raw_positions[len(req.input_ids) - 1].item())
            if last_token_pos < 0 or last_token_pos >= len(req.radix_token_to_key):
                raise RuntimeError("The final active token is outside radix_token_to_key.")
            token_boundary = last_token_pos + 1
            key_prefix_len = (
                int(req.radix_token_to_key[token_boundary].item())
                if token_boundary < len(req.radix_token_to_key)
                else len(req.radix_match_ids)
            )
        if req.radix_commit_key_len is not None:
            key_prefix_len = min(key_prefix_len, req.radix_commit_key_len)
        return key_prefix_len

    def _cache_finished_delta_req(self, req: Req) -> None:
        if self.page_size != 1:
            raise RuntimeError("Delta-marker caching requires page_size=1.")
        assert req.radix_key_virtual_mask is not None
        assert req.radix_key_to_token is not None
        assert req.radix_token_to_key is not None

        old_handle = req.cache_handle
        all_active_indices = self.page_table[req.table_idx, : req.cached_len]
        all_active_positions = req.raw_positions[: req.cached_len].to(
            dtype=torch.int64, device="cpu"
        )
        key_prefix_len = self._delta_key_prefix_len(req)
        try:
            if len(all_active_indices) != len(all_active_positions):
                raise RuntimeError(
                    "Active cache indices and true positions have different lengths."
                )
            if key_prefix_len < old_handle.cached_len:
                raise RuntimeError("Delta-marker key prefix regressed below the matched prefix.")

            key_virtual_mask = req.radix_key_virtual_mask[:key_prefix_len]
            full_token_prefix_len = int(torch.count_nonzero(~key_virtual_mask).item())
            within_prefix = all_active_positions < full_token_prefix_len
            active_indices = all_active_indices[
                within_prefix.to(
                    device=all_active_indices.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            ]
            active_positions = all_active_positions[within_prefix]
            excluded_indices = all_active_indices[
                (~within_prefix).to(
                    device=all_active_indices.device,
                    dtype=torch.bool,
                    non_blocking=True,
                )
            ]
            if req.initial_active_cached_len > len(active_indices):
                raise RuntimeError(
                    "Matched active prefix exceeds the delta-marker commit boundary."
                )
            full_indices = torch.full(
                (full_token_prefix_len,),
                -1,
                dtype=torch.int32,
                device=active_indices.device,
            )
            filled = torch.zeros(
                full_token_prefix_len,
                dtype=torch.bool,
                device=active_indices.device,
            )

            old_full_cached_len = old_handle.physical_cached_len
            if old_full_cached_len > 0:
                if len(req.initial_full_match_indices) < old_full_cached_len:
                    raise RuntimeError(
                        "Initial full-token match indices are shorter than the cache handle."
                    )
                full_indices[:old_full_cached_len] = req.initial_full_match_indices[
                    :old_full_cached_len
                ]
                filled[:old_full_cached_len] = True

            if bool(torch.any(active_positions >= full_token_prefix_len).item()):
                raise RuntimeError("A cached active token lies outside the full-token prefix.")
            active_positions_device = active_positions.to(
                device=active_indices.device, non_blocking=True
            )
            overlap = active_positions < old_full_cached_len
            if bool(torch.any(overlap).item()):
                transformed = torch.zeros(len(active_indices), dtype=torch.bool, device="cpu")
                if req.retry_transformed_mask is not None:
                    transformed[: len(req.retry_transformed_mask)] = req.retry_transformed_mask
                ordinary_overlap = overlap & (~transformed)
                ordinary_device = ordinary_overlap.to(
                    device=active_indices.device, non_blocking=True
                )
                if bool(torch.any(ordinary_overlap).item()) and not torch.equal(
                    full_indices[active_positions_device[ordinary_device]],
                    active_indices[ordinary_device],
                ):
                    raise RuntimeError("Matched delta-marker tokens use different KV slots.")
            full_indices[active_positions_device] = active_indices
            filled[active_positions_device] = True

            inactive_positions = req.inactive_cached_positions
            inactive_pages = req.inactive_cached_pages
            if inactive_positions is not None and inactive_pages is not None:
                if bool(torch.any(inactive_positions < 0).item()):
                    raise RuntimeError("An inactive cached token has a negative raw position.")
                if bool(torch.any(inactive_positions >= full_token_prefix_len).item()):
                    raise RuntimeError("An inactive cached token lies outside the commit prefix.")
                inactive_device = inactive_positions.to(
                    device=active_indices.device, dtype=torch.int64, non_blocking=True
                )
                full_indices[inactive_device] = inactive_pages
                filled[inactive_device] = True

            missing_positions = torch.nonzero(~filled, as_tuple=False).view(-1)
            cacheable_full_len = (
                int(missing_positions[0].item())
                if len(missing_positions) > 0
                else full_token_prefix_len
            )
            if cacheable_full_len < old_full_cached_len:
                raise RuntimeError("A new real-token hole overlaps the matched Radix prefix.")
            cacheable_key_len = (
                int(req.radix_token_to_key[cacheable_full_len].item())
                if cacheable_full_len < full_token_prefix_len
                else key_prefix_len
            )

            commit_virtual_mask = req.radix_key_virtual_mask[:cacheable_key_len]
            commit_key_to_token = req.radix_key_to_token[:cacheable_key_len]
            key_indices = torch.full(
                (cacheable_key_len,),
                -1,
                dtype=torch.int32,
                device=active_indices.device,
            )
            real_key_mask = ~commit_virtual_mask
            real_token_positions = commit_key_to_token[real_key_mask]
            key_indices[real_key_mask.to(device=active_indices.device, non_blocking=True)] = (
                full_indices[
                    real_token_positions.to(device=active_indices.device, non_blocking=True)
                ]
            )

            insert_result = self.prefix_cache.insert_prefix(
                req.radix_match_ids[:cacheable_key_len],
                key_indices,
                commit_virtual_mask,
            )
            active_key_positions = req.radix_token_to_key[active_positions]
            self._free_finished_candidates(
                req,
                active_indices,
                active_key_positions,
                insert_result,
                excluded_indices,
            )
        finally:
            self.unlock(old_handle)

    def _cache_linear_req(self, req: Req, *, finished: bool) -> None:
        insert_ids = req.radix_input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        if finished:
            self.unlock(old_handle)
            self._free(page_indices[old_handle.cached_len : cached_len])
            self._free(page_indices[new_handle.cached_len :])
            return

        duplicate_slice = slice(old_handle.cached_len, cached_len)
        duplicate_indices = page_indices[duplicate_slice].clone()
        self.lock(new_handle)
        try:
            if cached_len > old_handle.cached_len:
                canonical_indices = new_handle.get_matched_indices()
                page_indices[duplicate_slice].copy_(canonical_indices[duplicate_slice])
        except Exception:
            self.unlock(new_handle)
            raise
        self.unlock(old_handle)
        req.cache_handle = new_handle
        self._free(duplicate_indices)

    def _free_finished_candidates(
        self,
        req: Req,
        candidates: torch.Tensor,
        candidate_key_positions: torch.Tensor,
        insert_result: InsertResult,
        excluded: torch.Tensor | None = None,
    ) -> None:
        """Apply main's three cache regions after mapping active tokens to Radix keys."""

        if len(candidates) != len(candidate_key_positions):
            raise RuntimeError("Candidate pages and Radix key positions have different lengths.")
        candidate_key_positions = candidate_key_positions.to(
            device=candidates.device, dtype=torch.int64, non_blocking=True
        )
        canonical_indices = insert_result.handle.get_matched_indices()

        def adopted_pages(pages: torch.Tensor, key_positions: torch.Tensor) -> torch.Tensor:
            key_positions = key_positions.to(
                device=pages.device, dtype=torch.int64, non_blocking=True
            )
            in_committed_key = (key_positions >= 0) & (
                key_positions < insert_result.handle.cached_len
            )
            adopted = torch.zeros(len(pages), dtype=torch.bool, device=pages.device)
            selected_positions = key_positions[in_committed_key]
            adopted[in_committed_key] = (
                canonical_indices[selected_positions] == pages[in_committed_key]
            )
            return adopted

        active_slots = torch.arange(len(candidates), dtype=torch.int64, device=candidates.device)
        newly_allocated = active_slots >= req.initial_active_cached_len
        if req.retry_transformed_mask is not None:
            transformed = req.retry_transformed_mask.to(
                device=candidates.device, dtype=torch.bool, non_blocking=True
            )
            if len(transformed) > len(newly_allocated):
                raise RuntimeError("Retry transformed-page mask exceeds the active cache prefix.")
            newly_allocated[: len(transformed)] |= transformed
        adopted = adopted_pages(candidates, candidate_key_positions)
        released = candidates[newly_allocated & (~adopted)]
        inactive_positions = req.inactive_cached_positions
        inactive_pages = req.inactive_cached_pages
        if inactive_positions is not None and inactive_pages is not None:
            if req.radix_token_to_key is None:
                raise RuntimeError("Inactive cached pages require a structured Radix mapping.")
            inactive_key_positions = req.radix_token_to_key[inactive_positions].to(
                device=inactive_pages.device, dtype=torch.int64, non_blocking=True
            )
            inactive_adopted = adopted_pages(inactive_pages, inactive_key_positions)
            released = torch.cat([released, inactive_pages[~inactive_adopted]])
        if excluded is not None:
            released = torch.cat([released, excluded])
        self._free(released)

    def _cache_finished_sparse_req(self, req: Req) -> None:
        if self.page_size != 1:
            raise RuntimeError("Drop-message sparse caching currently requires page_size=1.")
        old_handle = req.cache_handle
        active_indices = self.page_table[req.table_idx, : req.cached_len]
        active_positions = req.raw_positions[: req.cached_len]
        old_full_cached_len = old_handle.cached_len
        try:
            if len(active_indices) != len(active_positions):
                raise RuntimeError(
                    "Active cache indices and true positions have different lengths."
                )
            full_cached_len = int(active_positions[-1].item()) + 1
            if full_cached_len > len(req.radix_match_ids):
                raise RuntimeError("Full cached length exceeds Radix key length.")

            full_indices = torch.full(
                (full_cached_len,), -1, dtype=torch.int32, device=active_indices.device
            )
            filled = torch.zeros(full_cached_len, dtype=torch.bool, device=active_indices.device)
            if old_full_cached_len > 0:
                full_indices[:old_full_cached_len] = req.initial_full_match_indices[
                    :old_full_cached_len
                ]
                filled[:old_full_cached_len] = True

            full_positions = active_positions.to(
                dtype=torch.int64, device=active_indices.device, non_blocking=True
            )
            overlap = full_positions < old_full_cached_len
            if bool(torch.any(overlap).item()):
                if not torch.equal(full_indices[full_positions[overlap]], active_indices[overlap]):
                    raise RuntimeError("Cached full/active overlap uses different KV slots.")
            full_indices[full_positions] = active_indices
            filled[full_positions] = True

            if not bool(torch.all(filled).item()):
                expected_keep = torch.ones(
                    full_cached_len, dtype=torch.bool, device=active_indices.device
                )
                if req.prefix_keep_mask is not None:
                    keep_len = min(len(req.prefix_keep_mask), full_cached_len)
                    expected_keep[:keep_len] = (req.prefix_keep_mask[:keep_len] != 0).to(
                        device=active_indices.device, dtype=torch.bool, non_blocking=True
                    )
                unexpected = (~filled) & expected_keep
                if bool(torch.any(unexpected).item()):
                    positions = torch.nonzero(unexpected, as_tuple=False).view(-1)
                    raise RuntimeError(
                        "Missing kept KV positions while rebuilding full Radix values:"
                        f" {positions[:16].tolist()}"
                    )

            holes = torch.nonzero(full_indices < 0, as_tuple=False).view(-1)
            cacheable_len = int(holes[0].item()) if len(holes) > 0 else full_cached_len
            if cacheable_len < old_full_cached_len:
                raise RuntimeError("A new sparse hole overlaps the matched safe prefix.")

            insert_result = self.prefix_cache.insert_prefix(
                req.radix_match_ids[:cacheable_len],
                full_indices[:cacheable_len],
            )
            self._free_finished_candidates(
                req,
                active_indices,
                full_positions,
                insert_result,
            )
        finally:
            self.unlock(old_handle)

    def _cache_finished_full_req(self, req: Req) -> None:
        if self.page_size != 1:
            raise RuntimeError("Context-mask full-stream caching currently requires page_size=1.")
        old_handle = req.cache_handle
        full_indices = self.page_table[req.table_idx, : req.cached_len]
        try:
            if len(full_indices) == 0 or bool(torch.any(full_indices < 0).item()):
                raise RuntimeError("Full-stream Prefill produced invalid KV slots.")
            insert_result = self.prefix_cache.insert_prefix(
                req.radix_match_ids[: req.cached_len], full_indices
            )
            if insert_result.cached_len < req.initial_active_cached_len:
                raise RuntimeError("Full-stream Radix prefix regressed during commit.")
            self._free_finished_candidates(
                req,
                full_indices,
                torch.arange(len(full_indices), dtype=torch.int64, device=full_indices.device),
                insert_result,
            )
        finally:
            self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.drop_aware_eviction:
            from minisgl.kvcache.radix_cache import RadixPrefixCache

            assert isinstance(self.prefix_cache, RadixPrefixCache)
            free_slots = [int(slot) for slot in self.free_slots.tolist()]
            free_set = set(free_slots)
            if len(free_set) != len(free_slots):
                raise RuntimeError("Drop-aware CacheManager contains duplicate free KV slots.")
            resident = set(self.prefix_cache.resident_slots)
            overlap = free_set & resident
            if overlap:
                raise RuntimeError(
                    f"Drop-aware free/resident KV slot overlap: {sorted(overlap)[:16]}"
                )
            expected = set(range(self.num_pages))
            if free_set | resident != expected:
                missing = expected - free_set - resident
                unexpected = (free_set | resident) - expected
                raise RuntimeError(
                    "Drop-aware KV slot partition is incomplete: "
                    f"missing={sorted(missing)[:16]}, unexpected={sorted(unexpected)[:16]}"
                )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) == 0:
            return
        flat = indices.view(-1).to(dtype=torch.int64, device=self.device, non_blocking=True)
        max_slot = self.num_pages * self.page_size
        if bool(torch.any(flat < 0).item()) or bool(torch.any(flat >= max_slot).item()):
            bad = flat[(flat < 0) | (flat >= max_slot)]
            raise RuntimeError(f"Attempted to free invalid cache slots: {bad[:16].tolist()}")
        page_starts = flat[:: self.page_size].to(dtype=torch.int32)
        if bool(torch.any(page_starts % self.page_size != 0).item()):
            raise RuntimeError("Attempted to free a cache range that is not page aligned.")
        self.free_slots = torch.cat([self.free_slots, page_starts])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
