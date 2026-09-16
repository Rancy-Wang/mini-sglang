from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import pytest

from minisgl.attention.fi import FlashInferBackend, FIMetadata


def metadata():
    result = object.__new__(FIMetadata)
    result.initialized_wrappers = set()
    result.wrappers = {}
    result.cu_seqlens_q_cpu = torch.tensor([0, 2], dtype=torch.int32)
    result.cu_seqlens_k_cpu = torch.tensor([0, 3], dtype=torch.int32)
    result.indices = torch.arange(3, dtype=torch.int32)
    result.last_page_len_cpu = torch.ones(1, dtype=torch.int32)
    result.seq_lens_cpu = torch.tensor([3], dtype=torch.int32)
    result.is_decode = False
    result.graph_bs = None
    result.context_segments = result.sliding_context_segments = None
    return result


def backend(monkeypatch):
    obj = object.__new__(FlashInferBackend)
    obj.config = SimpleNamespace(head_dim=4, sliding_window=128)
    obj.qo_head_local, obj.kv_head_local = 2, 1
    obj.kvcache = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "Event", MagicMock)
    return obj


def test_waits_only_for_same_pinned_buffer_and_only_once_per_metadata(monkeypatch):
    obj = backend(monkeypatch)
    full, sliding = MagicMock(), MagicMock()
    first = metadata()
    obj._initialize_metadata_once(first, full, is_decode=False, window_left=-1)
    full_event = obj._plan_events[full]
    obj._initialize_metadata_once(first, sliding, is_decode=False, window_left=127)
    assert full_event.synchronize.call_count == 0
    obj._initialize_metadata_once(first, full, is_decode=False, window_left=-1)
    assert full.plan.call_count == 1
    second = metadata()
    obj._initialize_metadata_once(second, full, is_decode=False, window_left=-1)
    assert full_event.synchronize.call_count == 1 and full.plan.call_count == 2
    assert obj._plan_events[sliding].synchronize.call_count == 0


def test_two_slots_preserve_p1_metadata_and_graph_wrapper_identity(monkeypatch):
    obj = backend(monkeypatch)
    obj._plan_slot = 0
    obj._wrapper_slots = {(False, "ordinary", window): (MagicMock(), MagicMock())
                          for window in (-1, 127)}
    batches = [SimpleNamespace(attn_metadata=metadata()) for _ in range(3)]
    for batch in batches:
        obj.prepare_for_forward(batch)
    for window in (-1, 127):
        a, b = obj._wrapper_slots[(False, "ordinary", window)]
        assert batches[0].attn_metadata.wrappers[window] is a
        assert batches[1].attn_metadata.wrappers[window] is b
        assert batches[2].attn_metadata.wrappers[window] is a
        assert obj._plan_events[a].synchronize.call_count == 1
        assert obj._plan_events[b].synchronize.call_count == 0
    graph_wrapper = MagicMock()
    obj.graph_wrappers = {(8, -1): graph_wrapper}
    batches[0].attn_metadata.graph_bs = 8
    assert obj._ordinary_wrapper(batches[0].attn_metadata, -1) is graph_wrapper


def test_full_and_sliding_segments_are_preplanned_with_distinct_storage(monkeypatch):
    obj = backend(monkeypatch)
    obj._plan_slot = 0
    obj._wrapper_slots = {(False, kind, -1): (MagicMock(), MagicMock())
                          for kind in ("full", "sliding")}
    data = metadata()
    data.context_segments, data.sliding_context_segments = metadata(), metadata()
    obj.prepare_for_forward(SimpleNamespace(attn_metadata=data))
    full = data.context_segments.wrappers[-1]
    sliding = data.sliding_context_segments.wrappers[-1]
    assert full is not sliding
    assert full.plan.call_args.kwargs["window_left"] == -1
    assert sliding.plan.call_args.kwargs["window_left"] == -1
    assert obj._plan_events[full].synchronize.call_count == 0
    assert obj._plan_events[sliding].synchronize.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_two_slot_prefill_plans_preserve_outputs_across_queued_batches():
    """Real FI H2D/plan/run, full and sliding, without changing model math."""
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    device = torch.device("cuda", 0)
    stream = torch.cuda.Stream(device=device)
    torch.manual_seed(13)
    with torch.cuda.stream(stream):
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        obj = object.__new__(FlashInferBackend)
        obj.config = SimpleNamespace(head_dim=64, sliding_window=128)
        obj.qo_head_local, obj.kv_head_local = 4, 2
        obj.kvcache = SimpleNamespace(dtype=torch.bfloat16)
        obj._plan_slot = 0
        def wrapper():
            return BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
        obj._wrapper_slots = {(False, "ordinary", window): (wrapper(), wrapper())
                              for window in (-1, 127)}
        reference_wrapper = wrapper()
        k = torch.randn((400, 1, 2, 64), dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        inputs = []
        for size in (1, 7, 1, 7):
            data = metadata()
            data.cu_seqlens_q_cpu = torch.arange(size + 1, dtype=torch.int32) * 3
            data.cu_seqlens_k_cpu = torch.arange(size + 1, dtype=torch.int32) * 48
            data.indices = torch.randperm(400, device=device)[:size * 48].to(torch.int32)
            data.last_page_len_cpu = torch.ones(size, dtype=torch.int32)
            data.seq_lens_cpu = torch.full((size,), 48, dtype=torch.int32)
            q = torch.randn((size * 3, 4, 64), dtype=torch.bfloat16, device=device)
            inputs.append((data, q))
        references = []
        for data, q in inputs:
            pair = []
            for window in (-1, 127):
                # Independent synchronous reference, same FA2 plan and inputs.
                data.initialized_wrappers.clear()
                obj._initialize_metadata_once(data, reference_wrapper,
                                              is_decode=False, window_left=window)
                pair.append(reference_wrapper.run(q, (k, v), window_left=window).clone())
                stream.synchronize()
            data.initialized_wrappers.clear()
            references.append(pair)
        outputs = []
        for data, q in inputs:
            obj.prepare_for_forward(SimpleNamespace(attn_metadata=data))
            outputs.append([data.wrappers[window].run(q, (k, v), window_left=window)
                            for window in (-1, 127)])
        stream.synchronize()
        for expected, actual in zip(references, outputs, strict=True):
            for left, right in zip(expected, actual, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
