from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
from minisgl.core import Req, SamplingParams
from minisgl.attention.utils import (
    make_backend_page_table,
    make_last_page_len_cpu,
    make_page_indptr_cpu,
    make_paged_kv_indices,
)
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import PrefillAdder
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _make_cache_manager(num_pages: int = 8, page_size: int = 4) -> CacheManager:
    page_table = torch.empty((1, num_pages * page_size), dtype=torch.int32)
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix")


def _empty_handle(cm: CacheManager):
    return cm.prefix_cache.match_prefix(torch.empty(0, dtype=torch.int64)).cuda_handle


class _FakeKVCache:
    def __init__(self):
        self.copies = []

    def copy_slots(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        self.copies.append((src.clone(), dst.clone()))


def test_attention_page_helpers_convert_token_table_to_pages():
    page_size = 4
    page_table = torch.stack(
        [torch.arange(0, 16, dtype=torch.int32), torch.arange(16, 32, dtype=torch.int32)]
    )
    reqs = [SimpleNamespace(table_idx=0, device_len=7), SimpleNamespace(table_idx=1, device_len=5)]

    backend_table = make_backend_page_table(page_table, reqs, max_seqlen_k=7, page_size=page_size)
    assert backend_table.tolist() == [[0, 1], [4, 5]]

    indices = make_paged_kv_indices(page_table, reqs, page_size=page_size)
    assert indices.tolist() == [0, 1, 4, 5]

    indptr = make_page_indptr_cpu([7, 5, 8], page_size, pin_memory=False)
    assert indptr.tolist() == [0, 2, 4, 6]

    last_page_len = make_last_page_len_cpu([7, 5, 8], page_size, pin_memory=False)
    assert last_page_len.tolist() == [3, 1, 4]


def test_sparse_match_truncates_at_mixed_page_for_page_granular_scheme():
    page_size = 4
    cm = _make_cache_manager(page_size=page_size)
    input_ids = torch.arange(page_size * 4, dtype=torch.int64)
    indices = torch.arange(page_size * 4, dtype=torch.int32)
    cm.prefix_cache.insert_prefix(input_ids, indices)

    keep_mask = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        dtype=torch.int32,
    )
    pending = PendingReq(
        uid=1,
        input_ids=torch.arange(11, dtype=torch.int32),
        true_positions=torch.tensor([0, 1, 2, 3, 8, 10, 11, 12, 13, 14, 15], dtype=torch.int32),
        radix_input_ids=torch.arange(11, dtype=torch.int64),
        radix_match_ids=torch.arange(page_size * 4 + 1, dtype=torch.int64),
        sampling_params=SamplingParams(max_tokens=1),
        prefix_keep_mask=keep_mask,
    )

    match = cm.match_req(pending)

    assert match is not None
    assert match.full_cached_len == page_size
    assert match.handle.cached_len == 16
    assert match.active_cached_len == page_size
    assert match.active_match_indices.tolist() == [0, 1, 2, 3]
    assert not match.requires_compaction


def test_radix_virtual_markers_allow_page_aligned_real_tokens():
    page_size = 4
    cm = _make_cache_manager(page_size=page_size)
    input_ids = torch.tensor([10, 11, -1, 12, 13], dtype=torch.int64)
    indices = torch.tensor([0, 1, -1, 2, 3], dtype=torch.int32)
    virtual_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool)

    insert_result = cm.prefix_cache.insert_prefix(input_ids, indices, virtual_mask)

    assert insert_result.handle.cached_len == len(input_ids)
    match = cm.prefix_cache.match_prefix(input_ids, virtual_mask).cuda_handle
    assert match.cached_len == page_size
    assert match.physical_cached_len == page_size - 1
    assert match.get_matched_indices().tolist() == indices[:page_size].tolist()


def test_radix_virtual_markers_reject_unaligned_real_tokens():
    page_size = 4
    cm = _make_cache_manager(page_size=page_size)
    input_ids = torch.tensor([10, -1, 11], dtype=torch.int64)
    indices = torch.tensor([0, -1, 1], dtype=torch.int32)
    virtual_mask = torch.tensor([0, 1, 0], dtype=torch.bool)

    with pytest.raises(ValueError, match="page alignment"):
        cm.prefix_cache.insert_prefix(input_ids, indices, virtual_mask)


def test_sparse_cached_prefix_page_granular_recomputes_after_mixed_page():
    page_size = 4
    cm = _make_cache_manager(num_pages=8, page_size=page_size)
    fake_kv_cache = _FakeKVCache()
    core.get_global_ctx().kv_cache = fake_kv_cache
    source_indices = cm._page_to_token(cm._allocate(4))
    cm.prefix_cache.insert_prefix(torch.arange(16, dtype=torch.int64), source_indices)

    keep_mask = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        dtype=torch.int32,
    )
    active_positions = torch.cat(
        [
            torch.nonzero(keep_mask != 0, as_tuple=False).view(-1).to(torch.int32),
            torch.tensor([16], dtype=torch.int32),
        ]
    )
    pending = PendingReq(
        uid=1,
        input_ids=torch.arange(len(active_positions), dtype=torch.int32),
        true_positions=active_positions,
        radix_input_ids=torch.arange(len(active_positions), dtype=torch.int64),
        radix_match_ids=torch.arange(17, dtype=torch.int64),
        sampling_params=SamplingParams(max_tokens=1),
        prefix_keep_mask=keep_mask,
    )
    table_manager = TableManager(max_running_reqs=1, page_table=cm.page_table)
    adder = PrefillAdder(
        token_budget=len(active_positions),
        reserved_size=0,
        cache_manager=cm,
        table_manager=table_manager,
    )

    req = adder.try_add_one(pending)
    assert req is not None
    assert not req.compact_cached_prefix

    cm.allocate_paged([req])

    assert not req.compact_cached_prefix
    assert req.cache_handle.cached_len == 16
    assert req.cached_len == page_size
    assert req.initial_active_cached_len == page_size
    assert fake_kv_cache.copies == []
    assert cm.page_table[req.table_idx, :4].tolist() == [0, 1, 2, 3]

    req.cached_len = req.device_len
    cm._cache_finished_sparse_req(req)
    cm.check_integrity()


def test_sparse_finished_cache_inserts_only_complete_pages_for_page_size_gt_one():
    page_size = 4
    cm = _make_cache_manager(num_pages=4, page_size=page_size)
    cm._allocate(2)
    cm.page_table[0, :8] = torch.arange(8, dtype=torch.int32)
    req = Req(
        input_ids=torch.arange(7, dtype=torch.int32),
        true_positions=torch.tensor([0, 1, 2, 3, 4, 6, 7], dtype=torch.int32),
        radix_input_ids=torch.arange(7, dtype=torch.int64),
        radix_match_ids=torch.arange(7, dtype=torch.int64),
        initial_full_match_indices=torch.empty(0, dtype=torch.int32),
        initial_active_cached_len=0,
        true_seq_len=8,
        table_idx=0,
        cached_len=6,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=_empty_handle(cm),
        prefix_keep_mask=torch.tensor([1, 1, 1, 1, 1, 0, 1], dtype=torch.int32),
    )

    cm._cache_finished_sparse_req(req)

    match = cm.prefix_cache.match_prefix(torch.arange(7, dtype=torch.int64)).cuda_handle
    assert match.cached_len == page_size
    assert page_size in cm.free_slots.tolist()
    cm.check_integrity()


def test_context_full_stream_finished_cache_accepts_partial_final_page():
    page_size = 4
    cm = _make_cache_manager(num_pages=4, page_size=page_size)
    cm._allocate(2)
    cm.page_table[0, :8] = torch.arange(8, dtype=torch.int32)
    req = Req(
        input_ids=torch.arange(7, dtype=torch.int32),
        true_positions=torch.arange(7, dtype=torch.int32),
        radix_input_ids=torch.arange(7, dtype=torch.int64),
        radix_match_ids=torch.arange(7, dtype=torch.int64),
        initial_full_match_indices=torch.empty(0, dtype=torch.int32),
        initial_active_cached_len=0,
        true_seq_len=7,
        table_idx=0,
        cached_len=6,
        output_len=1,
        uid=1,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=_empty_handle(cm),
    )

    cm._cache_finished_full_req(req)

    match = cm.prefix_cache.match_prefix(torch.arange(7, dtype=torch.int64)).cuda_handle
    assert match.cached_len == page_size
    cm.check_integrity()



def _make_pending_req(uid: int, *, input_len: int = 1, output_len: int = 1) -> PendingReq:
    input_ids = torch.arange(input_len, dtype=torch.int32) + uid * 10
    radix_input_ids = input_ids.to(dtype=torch.int64)
    return PendingReq(
        uid=uid,
        input_ids=input_ids,
        true_positions=torch.arange(input_len, dtype=torch.int32),
        radix_input_ids=radix_input_ids,
        radix_match_ids=radix_input_ids,
        sampling_params=SamplingParams(max_tokens=output_len),
    )


def test_prefill_reservation_accounts_for_whole_pages():
    page_size = 4
    page_table = torch.empty((2, page_size), dtype=torch.int32)
    ctx = core.Context(page_size=page_size)
    core.set_global_ctx(ctx)
    cm = CacheManager(num_pages=1, page_size=page_size, page_table=page_table, type="radix")
    table_manager = TableManager(max_running_reqs=2, page_table=page_table)
    adder = PrefillAdder(
        token_budget=page_size,
        reserved_size=0,
        cache_manager=cm,
        table_manager=table_manager,
    )

    assert adder.try_add_one(_make_pending_req(1)) is not None
    assert adder.reserved_size == page_size
    assert adder.try_add_one(_make_pending_req(2)) is None
