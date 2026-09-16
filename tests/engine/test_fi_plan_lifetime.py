from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

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
