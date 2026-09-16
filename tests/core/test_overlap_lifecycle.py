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
        raw_positions=raw, true_positions=raw.clone(), radix_positions=torch.arange(34),
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
    assert torch.equal(plan.true_positions, raw[keep])
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
