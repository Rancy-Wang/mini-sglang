from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _compute_physical_indices(
        self,
        boundaries: List[Tuple[int, int, str, int]],
        drop_ids: List[int],
        current_msg_id: int,
    ) -> dict:
        """
        Compute the physical cache indices for each message based on historical drops.

        After drops, the physical cache is compressed. This method maps each
        message's absolute boundaries to its actual physical position in the
        PREVIOUS turn's cache state (before new_drop_ids are applied).

        Args:
            boundaries: List of (start, end, role, msg_id) for all messages
            drop_ids: Message IDs that were already dropped in previous rounds
                      (NOT including new_drop_ids, as those haven't been dropped yet)
            current_msg_id: The current message being processed

        Returns:
            Dict mapping msg_id -> (physical_start, physical_end)
        """
        dropped = set(drop_ids or [])
        physical_map = {}
        physical_offset = 0

        for (start, end, _, msg_id) in boundaries:
            if msg_id >= current_msg_id:
                break

            msg_len = end - start

            if msg_id in dropped:
                # This message was already dropped, skip it
                continue

            # This message exists in the physical cache
            physical_map[msg_id] = (physical_offset, physical_offset + msg_len)
            physical_offset += msg_len

        return physical_map

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(match_indices)

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        is_table_reuse: bool = False,
        previous_cached_len: int = 0,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            message_id=pending_req.message_id,
            drop_ids=pending_req.drop_ids,
            true_seq_len=pending_req.true_seq_len,
            is_table_reuse=is_table_reuse,
            previous_cached_len=previous_cached_len,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # Table reuse path: reconstruct KV cache with correct physical indices after drops
        if pending_req.table_idx is not None and pending_req.boundaries is not None:
            # Reusing table_idx - reconstruct retained KV cache without allocating new resources
            table_idx = pending_req.table_idx
            retained_input_ids = []
            retained_page_indices = []
            true_seq_len = 0
            cached_len = 0
            current_msg_start = 0

            # Compute physical indices based on historical drops only
            # (new_drop_ids haven't been applied yet, so those pages still exist in compressed form)
            physical_map = self._compute_physical_indices(
                boundaries=pending_req.boundaries,
                drop_ids=pending_req.drop_ids,
                current_msg_id=pending_req.message_id,
            )

            # Combine all drop IDs for filtering retained messages
            all_drop_ids = set(pending_req.drop_ids or []) | set(pending_req.new_drop_ids or [])

            for (start, end, _, msg_id) in pending_req.boundaries:
                if msg_id == pending_req.message_id:
                    # Found current message start
                    current_msg_start = start
                    break

                msg_len = end - start
                true_seq_len += msg_len  # Always count for absolute position

                # Check if this message should be dropped (either already or now)
                if msg_id in all_drop_ids:
                    continue

                # This message is retained
                # Use physical position for BOTH token_pool and page_table access
                # (both are stored in compressed form after previous drops)
                phys_start, phys_end = physical_map[msg_id]

                # Get token ids from token_pool using physical position
                old_token_ids = self.table_manager.token_pool[table_idx][phys_start:phys_end].cpu()
                retained_input_ids.append(old_token_ids)

                # Get page indices from page_table using physical position
                old_pages = self.table_manager.page_table[table_idx][phys_start:phys_end].cpu()
                retained_page_indices.append(old_pages)

                cached_len += msg_len

            # Reconstruct the table entries and update input_ids
            if retained_input_ids:
                new_input_ids = torch.cat(retained_input_ids)
                new_page_indices = torch.cat(retained_page_indices)

                # Update token_pool and page_table with retained tokens (compacting them)
                device_ids = self.table_manager.token_pool[table_idx][:cached_len]
                page_entry = self.table_manager.page_table[table_idx][:cached_len]
                device_ids.copy_(new_input_ids.pin_memory(), non_blocking=True)
                page_entry.copy_(new_page_indices.pin_memory(), non_blocking=True)

                # Update input_ids: retained tokens + current message onwards (from original input)
                pending_req.input_ids = torch.cat([new_input_ids, pending_req.input_ids[current_msg_start:]])
                print(f"Message {pending_req.message_id}: Reusing table_idx {table_idx} with {cached_len} retained tokens, true_seq_len {true_seq_len}")
            else:
                # No retained tokens, just use current message onwards
                pending_req.input_ids = pending_req.input_ids[current_msg_start:]

            pending_req.cached_len = cached_len
            pending_req.true_seq_len = true_seq_len

            cache_handle = torch.empty(0, dtype=torch.int32, device=device_ids.device)

            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cached_len,
                is_table_reuse=True,
                previous_cached_len=cached_len,  # The reconstructed cache is already protected
            )

        # Normal path: allocate new resources
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            cached_len = 0

            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cached_len,
                is_table_reuse=False,
                previous_cached_len=0,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(
            uid=req.uid,
            input_ids=req.input_ids,
            sampling_params=req.sampling_params,
            table_idx=req.table_idx,
            boundaries=req.boundaries,
            message_id=req.message_id,
            drop_ids=req.drop_ids,
            new_drop_ids=req.new_drop_ids,
            true_seq_len=req.true_seq_len,
        ))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
