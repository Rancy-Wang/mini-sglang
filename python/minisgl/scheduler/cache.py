from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


@dataclass(frozen=True)
class ContextMatchResult:
    handle: BaseCacheHandle
    full_match_indices: torch.Tensor
    full_cached_len: int
    active_match_indices: torch.Tensor
    active_cached_len: int


@dataclass(frozen=True)
class FullMatchResult:
    handle: BaseCacheHandle
    full_match_indices: torch.Tensor
    full_cached_len: int
    safe_match_indices: torch.Tensor
    safe_cached_len: int


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def _matched_indices(self, handle: BaseCacheHandle) -> torch.Tensor:
        if handle.cached_len == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        return handle.get_matched_indices()

    def _match_and_prune_legacy_holes(
        self, radix_query: torch.Tensor
    ) -> tuple[BaseCacheHandle, torch.Tensor] | None:
        result = self.prefix_cache.match_prefix(radix_query)
        handle = result.cuda_handle
        indices = self._matched_indices(handle)
        holes = torch.nonzero(indices < 0, as_tuple=False).view(-1)
        if len(holes) == 0:
            return handle, indices

        from minisgl.kvcache.radix_cache import RadixPrefixCache

        if not isinstance(self.prefix_cache, RadixPrefixCache):
            raise RuntimeError("A non-Radix prefix cache returned a negative page slot.")
        valid_prefix_len = int(holes[0].item())
        released = self.prefix_cache.prune_suffix(radix_query, valid_prefix_len)
        if released is None:
            return None
        self._free(released)

        result = self.prefix_cache.match_prefix(radix_query)
        handle = result.cuda_handle
        indices = self._matched_indices(handle)
        if handle.cached_len != valid_prefix_len or bool(torch.any(indices < 0).item()):
            raise RuntimeError(
                "Radix legacy-hole pruning did not leave the expected safe prefix:"
                f" expected={valid_prefix_len}, actual={handle.cached_len}"
            )
        return handle, indices

    def match_req(self, req: PendingReq) -> ContextMatchResult | None:
        assert req.input_len > 0, "Input length must be greater than 0."
        radix_query = req.radix_match_ids
        matched = self._match_and_prune_legacy_holes(radix_query[:-1])
        if matched is None:
            return None
        handle, full_match_indices = matched
        active_match_indices = full_match_indices
        if req.prefix_keep_mask is not None and len(full_match_indices) > 0:
            if len(req.prefix_keep_mask) < len(full_match_indices):
                raise RuntimeError(
                    "prefix_keep_mask is shorter than matched full prefix:"
                    f" {len(req.prefix_keep_mask)} < {len(full_match_indices)}"
                )
            keep_mask = (req.prefix_keep_mask[: len(full_match_indices)] != 0).to(
                device=full_match_indices.device, dtype=torch.bool, non_blocking=True
            )
            active_match_indices = full_match_indices[keep_mask]
        return ContextMatchResult(
            handle=handle,
            full_match_indices=full_match_indices,
            full_cached_len=handle.cached_len,
            active_match_indices=active_match_indices,
            active_cached_len=len(active_match_indices),
        )

    def match_full_req(self, req: PendingReq) -> FullMatchResult | None:
        assert req.input_len > 0, "Input length must be greater than 0."
        matched = self._match_and_prune_legacy_holes(req.radix_match_ids[:-1])
        if matched is None:
            return None
        handle, indices = matched
        return FullMatchResult(
            handle=handle,
            full_match_indices=indices,
            full_cached_len=handle.cached_len,
            safe_match_indices=indices,
            safe_cached_len=handle.cached_len,
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

    def _cache_linear_req(self, req: Req, *, finished: bool) -> None:
        insert_ids = req.radix_input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def _cache_finished_sparse_req(self, req: Req) -> None:
        if self.page_size != 1:
            raise RuntimeError("Drop-message sparse caching currently requires page_size=1.")
        old_handle = req.cache_handle
        active_indices = self.page_table[req.table_idx, : req.cached_len]
        active_positions = req.true_positions[: req.cached_len]
        old_full_cached_len = old_handle.cached_len
        try:
            if len(active_indices) != len(active_positions):
                raise RuntimeError("Active cache indices and true positions have different lengths.")
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
                if not torch.equal(
                    full_indices[full_positions[overlap]], active_indices[overlap]
                ):
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
            in_cache_len = insert_result.cached_len
            active_slots = torch.arange(
                len(active_indices), dtype=torch.int64, device=active_indices.device
            )
            newly_allocated = active_slots >= req.initial_active_cached_len
            adopted = (full_positions >= in_cache_len) & (full_positions < cacheable_len)
            self._free(active_indices[newly_allocated & (~adopted)])
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
            slots = torch.arange(
                len(full_indices), dtype=torch.int64, device=full_indices.device
            )
            duplicates = (
                (slots >= req.initial_active_cached_len)
                & (slots < insert_result.cached_len)
            )
            self._free(full_indices[duplicates])
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
