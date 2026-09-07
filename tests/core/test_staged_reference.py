"""CPU state/page tests; model and attention correctness are checked by the GPU runner."""
from types import SimpleNamespace

import minisgl.core as core
import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import UserMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager, ReferenceCapacityError
from minisgl.scheduler.scheduler import Scheduler
from minisgl.scheduler.table import TableManager


def message(uid=7, *, length=93, events=((49, ((0, 25),)), (79, ((25, 49),))), output=3):
    ids = torch.arange(100, 100 + length, dtype=torch.int32)
    positions = torch.arange(length, dtype=torch.int32)
    offsets = [0]
    flat = []
    for _, pairs in events:
        flat.extend(x for pair in pairs for x in pair)
        offsets.append(len(flat) // 2)
    return UserMsg(
        uid=uid, input_ids=ids, true_positions=positions, raw_positions=positions,
        radix_input_ids=ids.long(), sampling_params=SamplingParams(max_tokens=output, ignore_eos=True),
        prompt_tokens=length, staged_reference=True,
        drop_event_positions=torch.tensor([n for n, _ in events], dtype=torch.int32),
        drop_range_offsets=torch.tensor(offsets, dtype=torch.int32),
        drop_position_ranges=torch.tensor(flat, dtype=torch.int32),
        drop_effective_event_count=len(events),
    )


def scheduler(monkeypatch, pages=256):
    monkeypatch.setattr(torch.Tensor, 'pin_memory', lambda self: self)
    monkeypatch.setattr(core, '_GLOBAL_CTX', core.Context(page_size=1))
    import minisgl.distributed.info as info
    monkeypatch.setattr(info, '_TP_INFO', info.DistributedInfo(0, 1))
    s = Scheduler.__new__(Scheduler)
    table = torch.full((5, pages), -1, dtype=torch.int32)
    s.table_manager = TableManager(4, table)
    s.token_pool = s.table_manager.token_pool
    s.cache_manager = CacheManager(pages, 1, table, 'radix')
    s.decode_manager = DecodeManager(1)
    s.prefill_manager = PrefillManager(s.cache_manager, s.table_manager, s.decode_manager)
    s.finished_reqs, s.context_sequence_uids, s.request_metrics = set(), set(), {}
    s.eos_token_ids = set()
    s.replies = []
    s.send_result = s.replies.extend
    return s


def advance(s, batch, token):
    s.cache_manager.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()
    output = SimpleNamespace(next_tokens_gpu=torch.full((len(batch.reqs),), token, dtype=torch.int32))
    # _process_last_data accepts the real ForwardOutput tuple protocol.
    from collections import namedtuple
    Output = namedtuple('Output', 'next_tokens_gpu next_tokens_cpu copy_done_event')
    output = Output(output.next_tokens_gpu, output.next_tokens_gpu,
                    SimpleNamespace(synchronize=lambda: None))
    s._process_last_data((SimpleNamespace(batch=batch), output))


@pytest.mark.parametrize('budget', [1, 24, 25, 45, 46, 48, 49, 50, 78, 79, 93, 100])
def test_each_query_visibility_and_final_sample(monkeypatch, budget):
    s = scheduler(monkeypatch)
    msg = message()
    # Any access to cross-request matching/commit is a reference failure.
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference must not use Radix')
    monkeypatch.setattr(s.cache_manager, 'match_req', forbidden)
    monkeypatch.setattr(s.cache_manager.prefix_cache, 'insert_prefix', forbidden)
    s.prefill_manager.add_one_req(msg)
    seen = []
    while s.prefill_manager.runnable:
        batch = s.prefill_manager.schedule_next_batch(budget)
        assert batch is not None
        req = batch.reqs[0]
        assert not req.use_context_mask and req.usage_cached_tokens == 0
        keys = req.raw_positions[:req.device_len].tolist()
        queries = req.raw_positions[req.cached_len:req.device_len].tolist()
        for q in queries:
            expected = set(range(q + 1))
            if q >= 49:
                expected -= set(range(25))
            if q >= 79:
                expected -= set(range(25, 49))
            assert {k for k in keys if k <= q} == expected
        seen.extend(queries)
        advance(s, batch, 701 + len(seen))
        if seen[-1] < 92:
            assert s.replies == [] and not s.decode_manager.runnable
    assert seen == list(range(93))
    assert len(s.replies) == 1 and s.replies[0].next_token == 794
    assert req.raw_positions.tolist() == list(range(49, 94))
    assert req.true_positions.tolist() == list(range(49, 94))
    assert req.input_ids[-1] == s.token_pool[req.table_idx, req.cached_len] == 794
    assert req.completion_tokens == 1
    while s.decode_manager.runnable:
        advance(s, s.decode_manager.schedule_next_batch(), 800)
    assert len(s.replies) == 3
    terminal = s.replies[-1]
    assert (terminal.prompt_tokens, terminal.completion_tokens) == (93, 3)
    assert (terminal.cached_tokens, terminal.drop_skipped_tokens, terminal.repos_tokens) == (0, 0, 0)
    assert req.reference_state.released
    s.cache_manager.check_integrity()
    assert s.table_manager.available_size == 4


@pytest.mark.parametrize('when', ['pending', 'allocated', 'between', 'decode'])
def test_cancel_releases_only_private_pages(monkeypatch, when):
    s = scheduler(monkeypatch)
    # A separate cached request's pages must survive all reference exits.
    other = s.cache_manager._allocate(3)
    s.cache_manager.prefix_cache.insert_prefix(torch.tensor([1, 2, 3]), other)
    s.prefill_manager.add_one_req(message())
    req = None
    if when != 'pending':
        batch = s.prefill_manager.schedule_next_batch(49)
        req = batch.reqs[0]
        if when == 'allocated':
            s.cache_manager.allocate_paged(batch.reqs)
        else:
            advance(s, batch, 123)
        if when == 'decode':
            while s.prefill_manager.runnable:
                advance(s, s.prefill_manager.schedule_next_batch(100), 123)
    from minisgl.message import AbortBackendMsg
    s._process_one_msg(AbortBackendMsg(uid=7))
    assert not s.prefill_manager.runnable and not s.decode_manager.runnable
    s.cache_manager.check_integrity()
    assert s.cache_manager.prefix_cache.size_info.total_size == 3
    assert s.table_manager.available_size == 4
    if req is not None:
        assert req.reference_state.released
        with pytest.raises(RuntimeError, match='twice'):
            s.cache_manager.cache_req(req, finished=True)


def test_capacity_reject_and_batch_isolation(monkeypatch):
    s = scheduler(monkeypatch, pages=32)
    s.prefill_manager.add_one_req(message())
    with pytest.raises(ReferenceCapacityError):
        s.prefill_manager.schedule_next_batch(32)
    assert s.table_manager.available_size == 4
    s.prefill_manager.abort_req(7)
    s.cache_manager.check_integrity()
    for uid in range(4):
        s.prefill_manager.add_one_req(message(uid, length=6, events=((3, ((0, 2),)),), output=1))
    all_pages = []
    while s.prefill_manager.runnable:
        batch = s.prefill_manager.schedule_next_batch(32)
        assert len(batch.reqs) == 4
        advance(s, batch, 9)
        all_pages = [r.reference_state.owned_pages for r in batch.reqs]
        flat = torch.cat([p for p in all_pages if p is not None]) if any(p is not None for p in all_pages) else torch.empty(0)
        assert len(torch.unique(flat)) == len(flat)
    s.cache_manager.check_integrity()
    assert len(s.replies) == 4 and s.table_manager.available_size == 4


def test_same_boundary_events_and_invalid_future_drop(monkeypatch):
    s = scheduler(monkeypatch)
    s.prefill_manager.add_one_req(message(length=6, events=((3, ((0, 1),)), (3, ((1, 2),)))))
    advance(s, s.prefill_manager.schedule_next_batch(6), 10)
    req = s.prefill_manager.pending_list[0].reference_req
    assert req.raw_positions.tolist() == [2]
    advance(s, s.prefill_manager.schedule_next_batch(6), 10)
    while s.decode_manager.runnable:
        advance(s, s.decode_manager.schedule_next_batch(), 10)
    s.cache_manager.check_integrity()
    with pytest.raises(ValueError, match='before'):
        s.prefill_manager.add_one_req(message(length=6, events=((3, ((0, 4),)),)))
