from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

# This small pool depends only on torch, not the Linux-only scheduler runtime.
spec = importlib.util.spec_from_file_location(
    "_compact_indices_unit",
    Path(__file__).resolve().parents[2] / "python/minisgl/scheduler/compact_indices.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
CompactIndexPool = module.CompactIndexPool


@pytest.fixture
def cpu_pool(monkeypatch):
    """Deterministic event model: test lifetime even on a CPU-only machine."""
    class Event:
        def __init__(self):
            self.ready = True
            self.records = []

        def query(self):
            return self.ready

        def record(self, stream):
            self.records.append(stream.cuda_stream)

    monkeypatch.setattr(CompactIndexPool, "_allocate", lambda self, capacity: module._Buffer(
        torch.empty(capacity, dtype=torch.int64), torch.empty(capacity, dtype=torch.int64)))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=1))
    return CompactIndexPool(torch.device("cpu"), capacity=8)


def pair(keep=(0, 2, 4), dropped=(1, 3)):
    return torch.tensor(keep, dtype=torch.int64), torch.tensor(dropped, dtype=torch.int64)


def test_batch_packs_disjoint_views_and_empty_drop(cpu_pool):
    leases = cpu_pool.pack([pair(), pair((0, 1), ())])
    assert leases[0]._buffer is leases[1]._buffer
    assert leases[0].keep.tolist() == [0, 2, 4]
    assert leases[0].dropped_owned.tolist() == [1, 3]
    assert leases[1].keep.tolist() == [0, 1]
    assert leases[1].dropped_owned.numel() == 0
    assert leases[0]._buffer.pending == 2


def test_no_reuse_until_all_consumers_and_all_streams_finish(cpu_pool):
    leases = cpu_pool.pack([pair(), pair((0,), ())])
    block = leases[0]._buffer
    leases[0].release(SimpleNamespace(cuda_stream=2))
    assert not block.available()  # second request has not even enqueued its reads
    leases[1].release(SimpleNamespace(cuda_stream=3))
    block.events[2].ready = False
    original = block.host.clone()
    other = cpu_pool.pack([pair((7,), ())])[0]
    assert other._buffer is not block
    assert torch.equal(block.host, original)
    block.events[2].ready = True
    assert block.available()
    reused = cpu_pool.pack([pair((6,), ())])[0]
    assert reused._buffer is block
    with pytest.raises(RuntimeError, match="twice"):
        leases[0].release(SimpleNamespace(cuda_stream=2))


def test_abort_before_forward_fences_the_transfer(cpu_pool):
    lease = cpu_pool.pack([pair()])[0]
    lease.release(SimpleNamespace(cuda_stream=1))
    lease._buffer.events[1].ready = False
    assert not lease._buffer.available()
    lease._buffer.events[1].ready = True
    assert lease._buffer.available()


def test_growth_keeps_inflight_raw_streams_alive(cpu_pool):
    first = cpu_pool.pack([pair()])[0]
    before = first._buffer.host.clone()
    large = cpu_pool.pack([(torch.arange(131443), torch.empty(0, dtype=torch.int64))])[0]
    assert len(large.keep) == 131443
    assert torch.equal(large.keep, torch.arange(131443))
    assert torch.equal(first._buffer.host, before)
    assert first._buffer in cpu_pool._buffers


def test_no_work_and_invalid_indices(cpu_pool):
    assert cpu_pool.pack([]) == []
    with pytest.raises(ValueError, match="CPU int64"):
        cpu_pool.pack([(torch.tensor([0], dtype=torch.int32), pair()[1])])
    with pytest.raises(ValueError, match="CPU int64"):
        cpu_pool.pack([(torch.zeros(1, 1, dtype=torch.int64), pair()[1])])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_pinned_transfer_and_cross_stream_consumption():
    device = torch.device("cuda", 0)
    prepare, engine = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with torch.cuda.stream(prepare):
        pool = CompactIndexPool(device, capacity=32)
    torch.cuda.synchronize(device)  # serving startup warmup
    outputs = []
    for i in range(12):
        with torch.cuda.stream(prepare):
            leases = pool.pack([pair((i, i + 1), (i + 2,)), pair((i + 3,), ())])
            assert leases[0]._buffer.host.is_pinned()
        with torch.cuda.stream(engine):
            engine.wait_stream(prepare)
            for lease in leases:
                outputs.append((lease.keep.clone(), lease.dropped_owned.clone()))
                lease.release(engine)
    torch.cuda.synchronize(device)
    for i in range(12):
        assert outputs[2*i][0].tolist() == [i, i + 1]
        assert outputs[2*i][1].tolist() == [i + 2]
        assert outputs[2*i+1][0].tolist() == [i + 3]
    assert all(block.available() for block in pool._buffers)
    pool.clear_after_synchronize()
    assert not pool._buffers
