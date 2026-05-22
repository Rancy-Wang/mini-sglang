from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .utils import PendingReq

logger = init_logger(__name__)


@dataclass(frozen=True)
class MatchResult:
    handle: BaseCacheHandle
    full_match_indices: torch.Tensor
    full_cached_len: int
    active_match_indices: torch.Tensor
    active_cached_len: int


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) == 0:
            return
        flat = indices.view(-1).to(dtype=torch.int64, device=self.device, non_blocking=True)
        if bool(torch.any(flat < 0).item()) or bool(torch.any(flat >= self.num_pages).item()):
            bad = flat[(flat < 0) | (flat >= self.num_pages)]
            raise RuntimeError(f"Attempted to free invalid cache slots: {bad[:16].tolist()}")
        flat = torch.unique(flat.to(dtype=torch.int32))
        if len(flat) == 0:
            return
        if len(self._free_slots) > 0:
            existing = torch.isin(flat, self._free_slots)
            if bool(torch.any(existing).item()):
                dup = flat[existing]
                raise RuntimeError(f"Double-free detected for cache slots: {dup[:16].tolist()}")
        self._free_slots = torch.cat([self._free_slots, flat])

    def match_req(self, req: PendingReq) -> MatchResult:
        assert req.input_len > 0, "Input length must be greater than 0."
        radix_query = req.radix_match_ids if req.radix_match_ids is not None else req.radix_input_ids
        query_len = len(radix_query)
        handle, full_match_indices = self.manager.match_prefix(radix_query[: query_len - 1])
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
        if len(active_match_indices) > 0 and bool(torch.any(active_match_indices < 0).item()):
            neg_mask = active_match_indices < 0
            neg_pos = torch.nonzero(neg_mask, as_tuple=False).view(-1)
            neg_count = len(neg_pos)
            neg_preview = active_match_indices[neg_pos[:16]].tolist()
            logger.warning(
                "Active matched indices contain invalid negative slots."
                " Falling back to root prefix match."
                " uid=%s full_match_len=%s active_match_len=%s neg_count=%s neg_preview=%s",
                req.uid,
                len(full_match_indices),
                len(active_match_indices),
                neg_count,
                neg_preview,
            )
            root_handle, root_match_indices = self.manager.match_prefix(radix_query[:0])
            return MatchResult(
                handle=root_handle,
                full_match_indices=root_match_indices,
                full_cached_len=root_handle.cached_len,
                active_match_indices=root_match_indices,
                active_cached_len=len(root_match_indices),
            )

        return MatchResult(
            handle=handle,
            full_match_indices=full_match_indices,
            full_cached_len=handle.cached_len,
            active_match_indices=active_match_indices,
            active_cached_len=len(active_match_indices),
        )

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        if len(evicted) > 0:
            evicted = evicted.view(-1).to(dtype=torch.int32, device=self.device, non_blocking=True)
            if bool(torch.any(evicted < 0).item()) or bool(torch.any(evicted >= self.num_pages).item()):
                bad = evicted[(evicted < 0) | (evicted >= self.num_pages)]
                raise RuntimeError(f"Evicted invalid cache slots: {bad[:16].tolist()}")
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        full_radix_ids: torch.Tensor,
        active_indices: torch.Tensor,
        active_true_positions: torch.Tensor,
        initial_full_match_indices: torch.Tensor,
        initial_active_cached_len: int,
        prefix_keep_mask: torch.Tensor | None = None,
    ) -> None:
        old_full_cached_len = old_handle.cached_len
        try:
            if len(active_indices) != len(active_true_positions):
                raise RuntimeError(
                    "Length mismatch for active_indices and active_true_positions:"
                    f" {len(active_indices)} vs {len(active_true_positions)}"
                )
            if len(active_indices) == 0:
                raise RuntimeError("Cannot cache finished req with empty active indices.")

            full_cached_len = int(active_true_positions[-1].item()) + 1
            if full_cached_len > len(full_radix_ids):
                raise RuntimeError(
                    f"Full cached length {full_cached_len} exceeds full_radix_ids length {len(full_radix_ids)}."
                )
            if full_cached_len < old_full_cached_len:
                raise RuntimeError(
                    "Full cached length regressed below old cached length:"
                    f" {full_cached_len} < {old_full_cached_len}"
                )
            if old_full_cached_len > len(initial_full_match_indices):
                raise RuntimeError(
                    "Initial full match indices are shorter than old cached length:"
                    f" {len(initial_full_match_indices)} < {old_full_cached_len}"
                )

            full_indices = torch.full(
                (full_cached_len,), -1, dtype=torch.int32, device=active_indices.device
            )
            filled = torch.zeros((full_cached_len,), dtype=torch.bool, device=active_indices.device)

            if old_full_cached_len > 0:
                full_indices[:old_full_cached_len] = initial_full_match_indices[:old_full_cached_len]
                filled[:old_full_cached_len] = True

            full_positions = active_true_positions.to(
                dtype=torch.int64, device=active_indices.device, non_blocking=True
            )
            if bool(torch.any(full_positions < 0).item()) or bool(
                torch.any(full_positions >= full_cached_len).item()
            ):
                raise RuntimeError("Encountered out-of-range true position while reconstructing full indices.")

            overlap = full_positions < old_full_cached_len
            if bool(torch.any(overlap).item()):
                overlap_positions = full_positions[overlap]
                seeded_overlap = full_indices[overlap_positions]
                active_overlap = active_indices[overlap]
                if not torch.equal(seeded_overlap, active_overlap):
                    raise RuntimeError(
                        "Mismatch between initial full-match indices and active mapped indices"
                        " in already-cached overlap region."
                    )

            full_indices[full_positions] = active_indices
            filled[full_positions] = True
            if not bool(torch.all(filled).item()):
                missing = ~filled
                expected_keep = torch.ones(
                    (full_cached_len,), dtype=torch.bool, device=active_indices.device
                )
                if prefix_keep_mask is not None and len(prefix_keep_mask) > 0:
                    keep_len = min(len(prefix_keep_mask), full_cached_len)
                    expected_keep[:keep_len] = (prefix_keep_mask[:keep_len] != 0).to(
                        device=active_indices.device, dtype=torch.bool, non_blocking=True
                    )

                unexpected_missing = missing & expected_keep
                if bool(torch.any(unexpected_missing).item()):
                    missing_positions = torch.nonzero(unexpected_missing, as_tuple=False).view(-1)
                    missing_preview = missing_positions[:16].tolist()
                    raise RuntimeError(
                        "Failed to reconstruct full-index mapping for kept positions;"
                        f" missing kept positions (first 16): {missing_preview}"
                    )

                dropped_missing = missing & (~expected_keep)
                full_indices[dropped_missing] = -1
                filled[dropped_missing] = True

            if not bool(torch.all(filled).item()):
                missing_positions = torch.nonzero(~filled, as_tuple=False).view(-1)
                missing_preview = missing_positions[:16].tolist()
                raise RuntimeError(
                    "Failed to reconstruct complete full-index mapping for finished request;"
                    f" missing positions (first 16): {missing_preview}"
                )

            in_cache_full_len = self.manager.insert_prefix(
                full_radix_ids[:full_cached_len], full_indices
            )

            active_slots = torch.arange(
                len(active_indices), device=active_indices.device, dtype=torch.int64
            )
            dedup_mask = (
                (active_slots >= initial_active_cached_len)
                & (full_positions >= old_full_cached_len)
                & (full_positions < in_cache_full_len)
                & (active_indices >= 0)
            )
            self._free(active_indices[dedup_mask])
        finally:
            self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
        if len(self._free_slots) > 0:
            if bool(torch.any(self._free_slots < 0).item()) or bool(
                torch.any(self._free_slots >= self.num_pages).item()
            ):
                raise RuntimeError("CacheManager integrity check failed: free_slots contain invalid index.")
            if len(torch.unique(self._free_slots)) != len(self._free_slots):
                raise RuntimeError("CacheManager integrity check failed: free_slots contain duplicates.")
