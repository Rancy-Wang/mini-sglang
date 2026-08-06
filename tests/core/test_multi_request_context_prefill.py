from types import SimpleNamespace

from minisgl.scheduler.prefill import PrefillAdder, PrefillManager


def _manager(mask_flags):
    pending = [
        SimpleNamespace(uid=uid, use_context_mask=masked, chunked_req=None)
        for uid, masked in enumerate(mask_flags)
    ]
    return PrefillManager(
        cache_manager=object(),
        table_manager=object(),
        decode_manager=SimpleNamespace(inflight_tokens=0),
        pending_list=pending,
    )


def _return_scheduled_req(_adder, pending):
    return SimpleNamespace(uid=pending.uid, use_context_mask=pending.use_context_mask)


def test_scheduler_batches_consecutive_context_mask_requests(monkeypatch):
    manager = _manager([True, True, False])
    monkeypatch.setattr(
        "minisgl.scheduler.prefill._supports_multi_context_mask_prefill", lambda: True
    )
    monkeypatch.setattr(PrefillAdder, "try_add_one", _return_scheduled_req)

    batch = manager.schedule_next_batch(prefill_budget=32)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [0, 1]
    assert [req.uid for req in manager.pending_list] == [2]


def test_scheduler_keeps_masked_and_ordinary_prefill_batches_separate(monkeypatch):
    manager = _manager([False, True, True])
    monkeypatch.setattr(
        "minisgl.scheduler.prefill._supports_multi_context_mask_prefill", lambda: True
    )
    monkeypatch.setattr(PrefillAdder, "try_add_one", _return_scheduled_req)

    batch = manager.schedule_next_batch(prefill_budget=32)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [0]
    assert [req.uid for req in manager.pending_list] == [1, 2]


def test_scheduler_preserves_single_request_fallback_for_other_backends(monkeypatch):
    manager = _manager([True, True])
    monkeypatch.setattr(
        "minisgl.scheduler.prefill._supports_multi_context_mask_prefill", lambda: False
    )
    monkeypatch.setattr(PrefillAdder, "try_add_one", _return_scheduled_req)

    batch = manager.schedule_next_batch(prefill_budget=32)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [0]
    assert [req.uid for req in manager.pending_list] == [1]
