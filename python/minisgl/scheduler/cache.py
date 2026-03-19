from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str, table_manager=None):
        # TODO: support page_size > 1
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self._table_manager = table_manager
        self.manager = create_cache_manager(device=device, type=type, table_manager=table_manager)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

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

        # Not enough free slots, try evicting reusable tables one by one
        if self._table_manager is not None:
            while len(self._free_slots) < needed_len:
                evicted = self._table_manager.evict_one()
                if evicted is None:
                    break
                self._free(evicted.to(self.device))

        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # Fallback to cache manager's own eviction
        evicted = self.manager.evict(needed_len - len(self._free_slots))
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        total_size = self.manager.size_info.total_size
        if len(self._free_slots) + total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({total_size}) != num_pages({self.num_pages})"
            )
