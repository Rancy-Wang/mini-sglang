import threading
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minisgl.server.launch import _start_worker_watchdog


def test_runtime_worker_loss_notifies_once():
    worker = SimpleNamespace(exitcode=None)
    failure = threading.Event()
    calls = []

    def failed():
        calls.append(True)
        failure.set()

    cancel = _start_worker_watchdog([worker], failed, poll_interval_s=0.01)
    try:
        worker.exitcode = 1
        assert failure.wait(1)
    finally:
        cancel()
    assert calls == [True]


def test_normal_shutdown_cancels_watchdog_before_workers_stop():
    worker = SimpleNamespace(exitcode=None)
    calls = []
    cancel = _start_worker_watchdog([worker], lambda: calls.append(True))
    cancel()
    worker.exitcode = 0
    assert calls == []


@pytest.fixture
def r4_harness(monkeypatch):
    monkeypatch.delenv("MINISGL_R4_OBSERVE", raising=False)
    monkeypatch.delenv("MINISGL_R4_REFERENCE_SHIM", raising=False)
    path = Path(__file__).resolve().parents[2] / "scripts/validate_paged_occurrence_regression.py"
    spec = importlib.util.spec_from_file_location("r4_validation_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["timing", "exact", "pressure"])
def test_r4_observers_preserve_forward_and_keyword_only_commit(
    r4_harness, monkeypatch, tmp_path, mode,
):
    import minisgl.attention.base as attention
    from minisgl.core import Req
    from minisgl.engine.engine import Engine
    from minisgl.engine.sample import Sampler
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.scheduler import Scheduler

    # Register every patched production attribute for restoration after this
    # test. The harness itself exists only in an isolated serving process.
    for owner, names in (
        (attention, ("build_context_attention_batch", "build_occurrence_attention_batch",
                     "compile_context_page_tables")),
        (Engine, ("forward_batch",)), (Sampler, ("sample",)), (Req, ("append_host",)),
        (CacheManager, ("_allocate", "cache_req")), (Scheduler, ("run_when_idle",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))
    # A pre-imported backend must not silently bypass the timing/CSR observer.
    backend = SimpleNamespace(compile_context_page_tables=attention.compile_context_page_tables)
    monkeypatch.setitem(r4_harness.sys.modules, "minisgl.attention.fi", backend)
    monkeypatch.setitem(r4_harness.sys.modules, "minisgl.attention.fa", SimpleNamespace())
    monkeypatch.setenv("MINISGL_R4_OBSERVE", str(tmp_path))
    monkeypatch.setenv("MINISGL_R4_OBSERVE_MODE", mode)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    commits = []

    def commit(self, req, *, finished):
        commits.append((req.uid, finished))
        return "committed"

    monkeypatch.setattr(CacheManager, "cache_req", commit)
    monkeypatch.setattr(Sampler, "sample", lambda self, logits, args: logits.argmax(-1))
    monkeypatch.setattr(Req, "append_host", lambda self, token: token)
    monkeypatch.setattr(Scheduler, "run_when_idle", lambda self: None)
    import minisgl.scheduler.scheduler as scheduler_module
    for name in ("_make_input_tuple", "_make_write_tuple"):
        monkeypatch.setattr(scheduler_module, name, lambda batch, device: ("mapping", device))

    def forward(self, batch, args):
        req = batch.reqs[0]
        req.cached_len = req.device_len
        req.device_len += 1
        return Sampler.sample(None, torch.tensor([[0., 1.]]), args)

    monkeypatch.setattr(Engine, "forward_batch", forward)
    r4_harness.install_observers()
    req = SimpleNamespace(uid="r4-smoke", cached_len=0, device_len=2, table_idx=0,
                          input_ids=torch.tensor([10, 11]), true_positions=torch.tensor([0, 1, 2]),
                          raw_positions=torch.tensor([0, 1, 2]), radix_current_reposition=1)
    batch = SimpleNamespace(reqs=[req], size=1, phase="prefill", out_loc=torch.tensor([1, 0]),
                            occurrence_destination_pages=torch.tensor([2]),
                            occurrence_position_pairs=torch.tensor([[9, 1]]))
    kv = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
    engine = SimpleNamespace(graph_runner=SimpleNamespace(can_use_cuda_graph=lambda _: False),
                             kv_cache=SimpleNamespace(num_layers=1, k_cache=lambda _: kv,
                                                      v_cache=lambda _: kv))
    assert Engine.forward_batch(engine, batch, None).tolist() == [1]
    assert scheduler_module._make_input_tuple(batch, "cpu") == ("mapping", "cpu")
    assert scheduler_module._make_write_tuple(batch, "cpu") == ("mapping", "cpu")
    assert Req.append_host(req, torch.tensor([12])).tolist() == [12]
    manager = SimpleNamespace(prefix_cache=SimpleNamespace(
        root_node=SimpleNamespace(children={}), _ordinary_slot_nodes={7: [SimpleNamespace(uuid=3)]}))
    if mode == "pressure":
        (tmp_path / "record_owners.request").touch()
    assert CacheManager.cache_req(manager, req, finished=True) == "committed"
    assert CacheManager.cache_req(manager, req, finished=False) == "committed"
    assert commits == [(req.uid, True), (req.uid, False)]
    plan = attention.ContextAttentionBatch(
        cached_tokens=(0,), segment_table_indices=torch.tensor([0], dtype=torch.int32),
        key_positions=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32), max_seqlen_q=2, max_seqlen_k=2,
    )
    assert backend.compile_context_page_tables(torch.tensor([[7, 8]]), plan,
                                                output_layout="flat").flat_indices.tolist() == [7, 8]
    rows = [json.loads(line) for path in tmp_path.glob("observer-*.jsonl")
            for line in path.read_text().splitlines()]
    kinds = [row["kind"] for row in rows]
    assert "forward" in kinds and "page_tables" in kinds
    assert kinds.count("mapping") == 2
    if mode == "pressure":
        assert kinds.count("completed_tree") == 1
        assert next(row for row in rows if row["kind"] == "completed_tree")[
            "physical_owners"] == {"7": [3]}
        # Exercise the genuine allocator/DFS with a tiny CPU pool as a harness
        # regression only. This does NOT satisfy R4's real model/GPU-pool gate.
        import minisgl.core as core

        monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))
        cache = CacheManager(8, 1, torch.empty((1,)), type="radix")
        pages = cache._allocate(5)
        cache.prefix_cache.insert_prefix(torch.tensor([1, 2, 7]), pages[:3])
        cache.prefix_cache.insert_prefix(torch.tensor([3, 4]), pages[3:])
        # This newer branch shares the protected branch's physical pages; after
        # reclaiming all EXCLUSIVE evictable memory it can legitimately remain.
        cache.prefix_cache.insert_prefix(torch.tensor([5, 6]), pages[:2])
        (tmp_path / "pressure.request").touch()
        scheduler = SimpleNamespace(cache_manager=cache)
        Scheduler.run_when_idle(scheduler)
        Scheduler.run_when_idle(scheduler)  # The trigger is consumed only once.
        assert len(cache.free_slots) == cache.num_pages == 8
        pressure_rows = [json.loads(line) for path in tmp_path.glob("observer-*.jsonl")
                         for line in path.read_text().splitlines()]
        protected = next(row for row in pressure_rows if row["kind"] == "pressure_protected")
        assert protected["protected_survive"] and protected["remaining_eligible"]
        assert sum(row["kind"] == "pressure_complete" for row in pressure_rows) == 1
    elif mode == "exact":
        assert {"logits", "kv", "reposition_kv", "token"} <= set(kinds)
        assert next(row for row in rows if row["kind"] == "kv")["pages"] == 2


def test_r4_records_plaintext_server_failure(r4_harness):
    class Client:
        async def post(self, url, json):
            class Response:
                status_code = 500
                text = "Internal Server Error"

                def json(self):
                    raise ValueError("not JSON")
            return Response()

    record = asyncio.run(r4_harness.send(Client(), "http://127.0.0.1:1", {"messages": []}, 1))
    assert record["status_code"] == 500
    assert record["response_text"] == "Internal Server Error"
    assert "error" not in record  # Do not obscure a server error with a JSON parsing error.


def test_r4_rolling_interface_counts_tool_responses_not_assistant_turns(r4_harness):
    messages = [{"role": "system"}, {"role": "user"}]
    tools = []
    for _ in range(14):
        messages.extend([{"role": "assistant"}, {"role": "tool"}])
        tools.append(len(messages) - 1)
    assert r4_harness.rolling_interface(messages[:tools[11] + 1]) == {}
    assert r4_harness.rolling_interface(messages) == {
        "drop_message": {str(tools[12]): [tools[0]], str(tools[13]): [tools[1]]},
        "reposition": [tools[12], tools[13]],
    }


def test_r4_csr_partition_uses_query_boundaries_for_shared_direct_table(r4_harness):
    assert list(r4_harness.request_segment_ranges(torch.tensor([0, 2, 3, 5, 8]), [3, 5])) == [
        (0, 2), (2, 4)]
    with pytest.raises(ValueError, match="boundary"):
        list(r4_harness.request_segment_ranges(torch.tensor([0, 2, 3]), [1, 2]))
    with pytest.raises(ValueError, match="unassigned"):
        list(r4_harness.request_segment_ranges(torch.tensor([0, 2, 3]), [2]))


def test_r4_paired_gate_does_not_treat_turns_as_independent_samples(r4_harness):
    assert r4_harness.paired_noninferiority([100.] * 5, [101.] * 5)["noninferior_2_percent"]
    assert not r4_harness.paired_noninferiority([100.] * 5, [103.] * 5)["noninferior_2_percent"]
    assert not r4_harness.paired_noninferiority([100.] * 5, [90., 110., 90., 110., 100.])[
        "noninferior_2_percent"]
    with pytest.raises(ValueError, match="five"):
        r4_harness.paired_noninferiority([100.] * 100, [100.] * 100)


def test_r4_stress_decode_budget_preserves_prompt_and_schedule(r4_harness):
    original = {"messages": [{"role": "user", "content": "hello"}],
                "drop_message": {"13": [1]}, "reposition": [13], "max_tokens": 1}
    assert r4_harness.wave_payload(original, None) == original
    updated = r4_harness.wave_payload(original, 8)
    assert updated == {**original, "max_tokens": 8, "ignore_eos": True}
    assert original["max_tokens"] == 1
    with pytest.raises(ValueError, match="positive"):
        r4_harness.wave_payload(original, 0)


@pytest.mark.parametrize("url", ["https://127.0.0.1", "http://example.com", "http://192.0.2.1",
                                 "http://user:password@127.0.0.1"])
def test_r4_rejects_non_loopback_destinations(r4_harness, url):
    with pytest.raises(ValueError):
        r4_harness.loopback_url(url)
