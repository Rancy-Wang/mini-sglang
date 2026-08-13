from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.utils import init_logger, page_count

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


def _page_granular_keep_mask(keep_mask: torch.Tensor, page_size: int) -> torch.Tensor:
    if keep_mask.ndim != 1:
        raise ValueError("page-granular keep_mask must be one-dimensional.")
    if page_size <= 1:
        return keep_mask
    result = keep_mask.to(dtype=torch.bool, device="cpu").clone()
    for start in range(0, len(result), page_size):
        end = min(start + page_size, len(result))
        page = result[start:end]
        if bool(torch.any(page).item()) and not bool(torch.all(page).item()):
            result[start:end] = True
    return result.to(dtype=keep_mask.dtype)


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


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _estimate_tokens_to_reserve(
        self,
        *,
        input_len: int,
        output_len: int,
        cached_len: int,
        compact_cached_prefix: bool,
    ) -> int:
        page_size = self.cache_manager.page_size
        effective_cached_len = 0 if compact_cached_prefix else cached_len
        return (
            page_count(input_len + output_len, page_size)
            - page_count(effective_cached_len, page_size)
        ) * page_size

    def _try_allocate_one(
        self, req: PendingReq
    ) -> Tuple[BaseCacheHandle, int, float, torch.Tensor, int, bool, int] | None:
        if self.table_manager.available_size == 0:
            return None

        if req.use_context_mask:
            match = self.cache_manager.match_full_req(req)
            if match is None:
                return None
            cache_handle = match.handle
            cached_len = match.safe_cached_len
            cached_indices = match.safe_match_indices
            initial_full_match_indices = match.full_match_indices
            requires_compaction = False
        else:
            match = self.cache_manager.match_req(req)
            if match is None:
                return None
            cache_handle = match.handle
            cached_len = match.active_cached_len
            cached_indices = match.active_match_indices
            initial_full_match_indices = match.full_match_indices[: match.full_cached_len]
            requires_compaction = match.requires_compaction
        effective_prefix_len = self.cache_manager.matchable_active_prefix_len(req)
        hit_ratio = 1.0 if effective_prefix_len == 0 else cached_len / effective_prefix_len
        estimated_len = self._estimate_tokens_to_reserve(
            input_len=req.input_len,
            output_len=req.output_len,
            cached_len=cached_len,
            compact_cached_prefix=requires_compaction,
        )

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(cache_handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(cache_handle)
            return None

        table_idx = self.table_manager.allocate()
        try:
            if cached_len > 0:  # NOTE: set the cached part
                device_ids = self.table_manager.token_pool[table_idx][:cached_len]
                page_entry = self.table_manager.page_table[table_idx][:cached_len]
                device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
                page_entry.copy_(cached_indices)
        except Exception:
            self.table_manager.free(table_idx)
            self.cache_manager.unlock(cache_handle)
            raise

        return (
            cache_handle,
            table_idx,
            hit_ratio,
            initial_full_match_indices.clone(),
            cached_len,
            requires_compaction,
            estimated_len,
        )

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        cache_hit_ratio: float,
        initial_full_match_indices: torch.Tensor,
        initial_active_cached_len: int,
        compact_cached_prefix: bool,
        estimated_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += estimated_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            true_positions=pending_req.true_positions[: cached_len + chunk_size],
            radix_input_ids=pending_req.radix_input_ids[: cached_len + chunk_size],
            radix_match_ids=(
                pending_req.radix_match_ids
                if pending_req.radix_match_ids is not None
                else pending_req.radix_input_ids
            ),
            initial_full_match_indices=initial_full_match_indices,
            initial_active_cached_len=initial_active_cached_len,
            true_seq_len=int(pending_req.true_positions[cached_len + chunk_size - 1].item()) + 1,
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            stop=pending_req.stop,
            stop_token_seqs=pending_req.stop_token_seqs,
            prefix_keep_mask=pending_req.prefix_keep_mask,
            is_warmup=pending_req.is_warmup,
            cache_hit_ratio=cache_hit_ratio,
            full_input_ids=pending_req.full_input_ids,
            full_token_visible_until=pending_req.full_token_visible_until,
            full_keep_mask=pending_req.full_keep_mask,
            use_context_mask=pending_req.use_context_mask,
            radix_key_virtual_mask=pending_req.radix_key_virtual_mask,
            radix_key_to_token=pending_req.radix_key_to_token,
            radix_token_to_key=pending_req.radix_token_to_key,
            radix_commit_key_len=pending_req.radix_commit_key_len,
            radix_marker_ids=pending_req.radix_marker_ids,
            compact_cached_prefix=compact_cached_prefix,
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
                cache_hit_ratio=chunked_req.cache_hit_ratio,
                initial_full_match_indices=chunked_req.initial_full_match_indices,
                initial_active_cached_len=chunked_req.initial_active_cached_len,
                compact_cached_prefix=chunked_req.compact_cached_prefix,
                estimated_len=self._estimate_tokens_to_reserve(
                    input_len=pending_req.input_len,
                    output_len=pending_req.output_len,
                    cached_len=chunked_req.cached_len,
                    compact_cached_prefix=chunked_req.compact_cached_prefix,
                ),
            )

        if resource := self._try_allocate_one(pending_req):
            (
                cache_handle,
                table_idx,
                cache_hit_ratio,
                initial_full_match_indices,
                initial_active_cached_len,
                compact_cached_prefix,
                estimated_len,
            ) = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=initial_active_cached_len,
                cache_hit_ratio=cache_hit_ratio,
                initial_full_match_indices=initial_full_match_indices,
                initial_active_cached_len=initial_active_cached_len,
                compact_cached_prefix=compact_cached_prefix,
                estimated_len=estimated_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        input_ids = req.input_ids
        true_positions = req.true_positions
        radix_input_ids = req.radix_input_ids
        prefix_keep_mask = req.prefix_keep_mask
        full_keep_mask = req.full_keep_mask
        if prefix_keep_mask is not None and req.radix_match_ids is not None:
            page_size = self.cache_manager.page_size
            prefix_keep_mask = _page_granular_keep_mask(prefix_keep_mask, page_size)
            full_mask = torch.ones(len(req.radix_match_ids), dtype=torch.int32, device="cpu")
            full_mask[: len(prefix_keep_mask)] = prefix_keep_mask.to(dtype=torch.int32)
            if full_keep_mask is not None:
                full_mask[: len(full_keep_mask)] = _page_granular_keep_mask(
                    full_keep_mask, page_size
                ).to(dtype=torch.int32)
            active_mask = full_mask.to(dtype=torch.bool)
            full_ids = req.full_input_ids if req.full_input_ids is not None else req.radix_match_ids
            input_ids = full_ids.to(dtype=req.input_ids.dtype)[active_mask].contiguous()
            true_positions = torch.arange(len(full_ids), dtype=torch.int32, device="cpu")[
                active_mask
            ]
            radix_input_ids = req.radix_match_ids[active_mask].contiguous()
            full_keep_mask = full_mask
        if req.use_context_mask:
            if not req.is_warmup:
                raise ValueError("Context-mask Prefill is restricted to warmup requests.")
            if req.full_input_ids is None or req.radix_match_ids is None:
                raise ValueError(
                    "Context-mask Prefill requires a full token stream and Radix keys."
                )
            input_ids = req.full_input_ids
            true_positions = torch.arange(len(input_ids), dtype=torch.int32, device="cpu")
            radix_input_ids = (
                req.radix_match_ids[req.radix_token_to_key]
                if req.radix_token_to_key is not None
                else req.radix_match_ids
            )
        self.pending_list.append(
            PendingReq(
                uid=req.uid,
                input_ids=input_ids,
                true_positions=true_positions,
                radix_input_ids=radix_input_ids,
                radix_match_ids=req.radix_match_ids,
                sampling_params=req.sampling_params,
                stop=req.stop,
                stop_token_seqs=req.stop_token_seqs,
                is_warmup=req.is_warmup,
                internal_uid=req.internal_uid,
                prefix_keep_mask=prefix_keep_mask,
                full_input_ids=req.full_input_ids,
                full_token_visible_until=req.full_token_visible_until,
                full_keep_mask=full_keep_mask,
                use_context_mask=req.use_context_mask,
                radix_key_virtual_mask=req.radix_key_virtual_mask,
                radix_key_to_token=req.radix_key_to_token,
                radix_token_to_key=req.radix_token_to_key,
                radix_commit_key_len=req.radix_commit_key_len,
                radix_marker_ids=tuple(req.radix_marker_ids or ()),
            )
        )

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
        supports_multi_context_mask = _supports_multi_context_mask_prefill()
        for pending_req in self.pending_list:
            if len(reqs) > 0:
                first_uses_context_mask = reqs[0].use_context_mask
                if pending_req.use_context_mask != first_uses_context_mask:
                    break
                if pending_req.use_context_mask and not supports_multi_context_mask:
                    break
            if req := adder.try_add_one(pending_req):
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
                return req.chunked_req or req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
