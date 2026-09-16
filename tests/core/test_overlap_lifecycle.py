"""Transition planning equivalence and bounded cross-stream ownership."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "_overlap_state_unit",
    Path(__file__).resolve().parents[2] / "python/minisgl/scheduler/overlap_state.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Event:
    def __init__(self):
        self.ready = False
        self.syncs = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.syncs += 1
        self.ready = True


def test_fence_only_waits_consumers_once_per_stream():
    event = Event()
    resource = object()
    fence = module.TransitionFence(event, (resource,))
    waits = []
    p1 = SimpleNamespace(cuda_stream=1, wait_event=waits.append)
    # Independent preparation does not visit the P1 fence.
    assert waits == []
    fence.wait_on(p1)
    fence.wait_on(p1)
    assert waits == [event]
    fence.wait_on(SimpleNamespace(cuda_stream=2, wait_event=waits.append))
    assert waits == [event, event]
    assert not fence.release_if_ready() and fence.resources == (resource,)
    event.ready = True
    assert fence.release_if_ready() and fence.resources == ()


def test_retirement_is_bounded_and_never_releases_inflight_sources():
    queue = module.TransitionRetirement(capacity=2)
    fences = [module.TransitionFence(Event(), (object(),)) for _ in range(3)]
    queue.add(fences[0])
    queue.add(fences[1])
    queue.collect()
    assert all(f.resources for f in fences)
    queue.add(fences[2])
    assert fences[0].event.syncs == 1 and not fences[0].resources
    assert queue.pending == fences[1:] and all(f.resources for f in queue.pending)
    for fence in queue.pending:
        fence.event.ready = True
    queue.collect()
    assert queue.pending == []


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_cpu_plan_matches_original_compaction_without_mutating_request(owned, paged, seed):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.arange(17, dtype=torch.int32) * 2
    mask = torch.randint(0, 2, (34,), generator=generator, dtype=torch.int32)
    mask[0] = 1
    req = SimpleNamespace(
        input_ids=torch.arange(17, dtype=torch.int32), radix_input_ids=torch.arange(17),
        raw_positions=raw, true_positions=raw.clone(), radix_positions=torch.arange(34) // 2,
        context_post_prefill_keep_mask=mask, initial_active_cached_len=5,
        retry_transformed_mask=torch.tensor([False, True, False, True, False]),
        inactive_cached_positions=torch.tensor([99], dtype=torch.int32),
        occurrence_terminal_owned_mask=(torch.arange(17) % 3 != 0) if owned else None,
        reposition_execution_mode="paged-occurrence" if paged else "staged",
    )
    original = {key: value.clone() for key, value in vars(req).items()
                if isinstance(value, torch.Tensor)}
    keep = mask[raw.long()] != 0
    expected_owned = req.occurrence_terminal_owned_mask
    if expected_owned is None:
        expected_owned = torch.arange(17) >= 5
        expected_owned[:5] |= req.retry_transformed_mask
    plan = module.CompactPlan.build(req, 17)
    assert torch.equal(plan.keep_indices, keep.nonzero().view(-1))
    assert torch.equal(plan.dropped_indices, ((~keep) & expected_owned).nonzero().view(-1))
    assert torch.equal(plan.input_ids, req.input_ids[keep])
    assert torch.equal(plan.raw_positions, raw[keep])
    assert torch.equal(plan.true_positions, (raw // 2 if paged else raw)[keep])
    assert plan.initial_cached_len == int(keep[:5].sum())
    assert torch.equal(plan.retry_mask, req.retry_transformed_mask[keep[:5]])
    assert torch.equal(plan.inactive_positions, torch.cat((
        req.inactive_cached_positions, raw[(~keep) & expected_owned])))
    for name, before in original.items():
        assert torch.equal(getattr(req, name), before), name


def test_plan_rejects_all_dropped():
    req = SimpleNamespace(context_post_prefill_keep_mask=torch.zeros(4),
                          raw_positions=torch.arange(4))
    with pytest.raises(RuntimeError, match="every prompt token"):
        module.CompactPlan.build(req, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_compaction_fence_with_external_storage_and_independent_preparation():
    from minisgl.core import Req, SamplingParams
    from minisgl.scheduler.compact_indices import CompactIndexPool
    from minisgl.scheduler.overlap_state import CompactPlan, TransitionRetirement
    from minisgl.scheduler.scheduler import ForwardInput, Scheduler
    from minisgl.scheduler.table import TableManager

    device = torch.device("cuda", 0)
    prepare, engine = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    scheduler = object.__new__(Scheduler)
    scheduler.device = device
    scheduler.transition_retirement = TransitionRetirement()
    scheduler.request_metrics = {}
    scheduler.decode_manager = SimpleNamespace(filter_reqs=lambda _: None)
    with torch.cuda.stream(prepare):
        table = TableManager(1, torch.full((2, 8), -1, dtype=torch.int32, device=device))
        slot = table.allocate()
        table.prepare_occurrence(slot, 9)
        table.occurrence_pages(slot).copy_(torch.arange(10, 19, dtype=torch.int32))
        table.occurrence_tokens(slot)[:9].copy_(torch.arange(100, 109, dtype=torch.int32))
        scheduler.table_manager, scheduler.token_pool = table, table.token_pool
        scheduler.compact_index_pool = CompactIndexPool(device, capacity=32)
        raw = torch.arange(9, dtype=torch.int32)
        prompt = raw + 100
        keep = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32)
        req = Req(
            input_ids=prompt, true_positions=raw, raw_positions=raw,
            radix_input_ids=prompt.long(), radix_match_ids=prompt.long(), true_seq_len=9,
            table_idx=slot, cached_len=3, output_len=2, uid=1,
            sampling_params=SamplingParams(max_tokens=2), cache_handle=SimpleNamespace(),
            initial_active_cached_len=3,
            initial_full_match_indices=torch.arange(10, 13, dtype=torch.int32, device=device),
            context_post_prefill_keep_mask=keep, occurrence_external_storage=True,
            reposition_execution_mode="paged-occurrence", radix_positions=raw,
            occurrence_raw_tokens=raw, occurrence_positions=raw,
            occurrence_birth_indices=raw, occurrence_terminal_indices=raw,
            occurrence_segment_query_starts=torch.tensor([0], dtype=torch.int32),
            occurrence_segment_query_ends=torch.tensor([9], dtype=torch.int32),
            occurrence_segment_key_offsets=torch.tensor([0, 9], dtype=torch.int32),
            occurrence_segment_key_indices=raw,
            occurrence_birth_pages=torch.arange(10, 19, dtype=torch.int32, device=device),
            occurrence_birth_owned_mask=(raw >= 3) | (raw == 1),
            retry_transformed_mask=torch.tensor([False, True, False]),
            inactive_cached_positions=torch.tensor([99], dtype=torch.int64),
            inactive_cached_pages=torch.tensor([40], dtype=torch.int32, device=device),
        )
        req.context_compact_plan = plan = CompactPlan.build(req, 9)
        req.context_decode_index_lease = scheduler.compact_index_pool.pack([
            (plan.keep_indices, plan.dropped_indices)])[0]
        batch = SimpleNamespace(reqs=[req], padded_reqs=[req])
        mapping = (torch.full((6,), slot, dtype=torch.int64, device=device),
                   torch.zeros(6, dtype=torch.int64, device=device))
        output_mapping = (torch.tensor([slot], device=device), torch.tensor([-1], device=device))
    def forward(*args):
        torch.cuda._sleep(5_000_000)
        req.complete_one()
        return SimpleNamespace(next_tokens_gpu=torch.tensor([999], dtype=torch.int32, device=device))
    scheduler.engine = SimpleNamespace(stream=engine, forward_batch=forward)
    with torch.cuda.stream(engine):
        engine.wait_stream(prepare)
        scheduler._forward(ForwardInput(batch, None, mapping, output_mapping))
    assert req.context_transition.resources and not table.has_occurrence_storage(slot)
    with torch.cuda.stream(prepare):
        # Unrelated preparation has no P1 table dependency or blanket wait.
        unrelated = torch.arange(4096, dtype=torch.int32, device=device)
        scheduler._wait_for_transition(req)
        active = table.page_table[slot, :5].clone()
        tokens = table.token_pool[slot, :6].clone()
        inactive = req.inactive_cached_pages.clone()
    prepare.synchronize()
    assert active.tolist() == [10, 12, 14, 16, 18]
    assert tokens.tolist() == [100, 102, 104, 106, 108, 999]
    assert inactive.tolist() == [40, 11, 13, 15, 17]
    assert unrelated[-1].item() == 4095
    scheduler.transition_retirement.collect()
    assert scheduler.transition_retirement.pending == []
    assert req.context_transition.resources == ()
