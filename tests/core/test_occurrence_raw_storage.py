from types import SimpleNamespace

import pytest
import torch
from minisgl.core import validate_occurrence_positions
from minisgl.scheduler.scheduler import ForwardInput, Scheduler
from minisgl.scheduler.table import TableManager


@pytest.mark.parametrize("raw_length", [7, 8, 9, 131443])
def test_raw_storage_preserves_graph_tables_and_releases_on_free(raw_length):
    fixed = torch.zeros((2, 8), dtype=torch.int32)
    manager = TableManager(1, fixed)
    slot = manager.allocate()
    page_ptr, token_ptr = fixed.data_ptr(), manager.token_pool.data_ptr()
    manager.prepare_occurrence(slot, raw_length)
    assert manager.has_occurrence_storage(slot) == (raw_length + 1 > 8)
    pages, tokens = manager.occurrence_pages(slot), manager.occurrence_tokens(slot)
    assert len(pages) >= raw_length
    assert len(tokens) >= raw_length + 1
    pages[raw_length - 1] = 123
    tokens[raw_length] = 456
    assert fixed.data_ptr() == page_ptr
    assert manager.token_pool.data_ptr() == token_ptr
    manager.free(slot)
    assert not manager.has_occurrence_storage(slot)
    assert manager.available_size == 1


def _request():
    return SimpleNamespace(
        input_ids=torch.arange(4),
        occurrence_positions=torch.tensor([0, 100, 1, 2], dtype=torch.int32),
        occurrence_terminal_indices=torch.arange(4, dtype=torch.int32),
        full_keep_mask=torch.tensor([1, 0, 1, 1], dtype=torch.int32),
        radix_next_position=3,
    )


def test_dropped_terminal_does_not_bound_active_next_position():
    assert validate_occurrence_positions(_request(), 128, 128) == 3


def test_historical_execution_still_checks_rope_limit():
    with pytest.raises(ValueError, match="model/RoPE"):
        validate_occurrence_positions(_request(), 128, 100)


def test_drop_without_reposition_allows_position_holes():
    req = _request()
    req.occurrence_positions[-1] = 9
    req.radix_next_position = 10
    assert validate_occurrence_positions(req, 128, 128) == 3
    req.radix_next_position = 9
    with pytest.raises(ValueError, match="active terminal"):
        validate_occurrence_positions(req, 128, 128)


def test_compact_occurrence_layout_validates_without_expanded_plan():
    req = SimpleNamespace(
        input_ids=torch.arange(4),
        occurrence_positions=None,
        occurrence_terminal_indices=None,
        occurrence_layout_birth_positions=torch.tensor([0, 1, 1, 2], dtype=torch.int32),
        occurrence_layout_birth_stages=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        occurrence_layout_transition_offsets=torch.tensor([0, 1], dtype=torch.int32),
        occurrence_layout_transition_raw_tokens=torch.tensor([1], dtype=torch.int32),
        occurrence_layout_transition_old_positions=torch.tensor([1], dtype=torch.int32),
        occurrence_layout_transition_new_positions=torch.tensor([0], dtype=torch.int32),
        full_keep_mask=torch.tensor([1, 0, 1, 1], dtype=torch.int32),
        radix_positions=torch.tensor([0, 0, 1, 2], dtype=torch.int32),
        radix_next_position=3,
    )

    assert validate_occurrence_positions(req, 128, 128) == 3
    req.occurrence_layout_transition_old_positions[0] = 128
    with pytest.raises(ValueError, match="model/RoPE"):
        validate_occurrence_positions(req, 128, 128)


def test_external_sample_write_uses_post_forward_cached_length():
    table = TableManager(1, torch.zeros((2, 8), dtype=torch.int32))
    slot = table.allocate()
    table.prepare_occurrence(slot, 9)
    table.occurrence_tokens(slot)[:9] = torch.arange(9)
    req = SimpleNamespace(
        occurrence_external_storage=True, table_idx=slot, cached_len=7,
        device_len=9, can_decode=True,
    )
    batch = SimpleNamespace(reqs=[req], padded_reqs=[req])

    def forward(current_batch, args):
        assert current_batch.input_ids.tolist() == [7, 8]
        req.cached_len, req.device_len = 9, 10
        req.can_decode = False  # max_tokens=1 still commits its sampled token.
        return SimpleNamespace(next_tokens_gpu=torch.tensor([999], dtype=torch.int32))

    scheduler = object.__new__(Scheduler)
    scheduler.table_manager = table
    scheduler.token_pool = table.token_pool
    scheduler.engine = SimpleNamespace(forward_batch=forward)
    scheduler.decode_manager = SimpleNamespace(filter_reqs=lambda _: None)
    scheduler._forward(ForwardInput(
        batch, None, (torch.tensor([slot, slot]), torch.tensor([0, 0])),
        (torch.tensor([slot]), torch.tensor([-1])),
    ))
    assert table.occurrence_tokens(slot)[9] == 999


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_compaction_lease_preserves_pages_positions_sample_and_ownership(external, prepared, monkeypatch):
    from minisgl.core import Req, SamplingParams

    table = TableManager(1, torch.full((2, 8 if external else 16), -1, dtype=torch.int32))
    slot = table.allocate()
    table.prepare_occurrence(slot, 9)
    table.occurrence_pages(slot)[:9] = torch.arange(10, 19, dtype=torch.int32)
    table.occurrence_tokens(slot)[:10] = torch.arange(100, 110, dtype=torch.int32)
    keep = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32)
    prompt = torch.arange(100, 109, dtype=torch.int32)
    req = Req(input_ids=prompt, true_positions=torch.arange(9, dtype=torch.int32),
              raw_positions=torch.arange(9, dtype=torch.int32), radix_input_ids=prompt.to(torch.int64),
              radix_match_ids=prompt.to(torch.int64), true_seq_len=9, table_idx=slot,
              cached_len=3, output_len=2, uid=1, sampling_params=SamplingParams(max_tokens=2),
              cache_handle=SimpleNamespace(), initial_active_cached_len=3,
              context_post_prefill_keep_mask=keep, occurrence_external_storage=external,
              reposition_execution_mode="paged-occurrence", radix_positions=torch.arange(9),
              retry_transformed_mask=torch.tensor([False, True, False]))
    req.cached_len, req.device_len, req.max_device_len = 9, 10, 11
    req.true_positions = req.raw_positions = torch.arange(10, dtype=torch.int32)
    scheduler = object.__new__(Scheduler)
    scheduler.table_manager = table
    retired = []
    if prepared:
        req.context_decode_keep_mask = keep.bool()
        req.context_decode_keep_indices = torch.tensor([0, 2, 4, 6, 8])
        req.context_decode_dropped_owned_indices = torch.tensor([1, 3, 5, 7])
        req.context_decode_index_lease = SimpleNamespace(
            keep=req.context_decode_keep_indices.clone(),
            dropped_owned=req.context_decode_dropped_owned_indices.clone())
        original_to = torch.Tensor.to
        def guarded_to(tensor, *args, **kwargs):
            assert tensor is not req.context_decode_keep_indices
            assert tensor is not req.context_decode_dropped_owned_indices
            return original_to(tensor, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, "to", guarded_to)
        def retire(request):
            retired.append(request.context_decode_index_lease)
            request.context_decode_index_lease = None
        scheduler._release_compact_indices = retire
    scheduler._compact_context_after_prefill(req)
    assert table.page_table[slot, :5].tolist() == [10, 12, 14, 16, 18]
    assert table.token_pool[slot, :6].tolist() == [100, 102, 104, 106, 108, 109]
    assert req.input_ids.tolist() == [100, 102, 104, 106, 108]
    assert req.true_positions.tolist() == req.raw_positions.tolist() == [0, 2, 4, 6, 8, 9]
    assert req.inactive_cached_pages.tolist() == [11, 13, 15, 17]
    assert req.inactive_cached_positions.tolist() == [1, 3, 5, 7]
    assert (req.cached_len, req.device_len, req.initial_active_cached_len) == (5, 6, 2)
    assert req.context_decode_index_lease is None
    assert not table.has_occurrence_storage(slot)
    assert len(retired) == int(prepared)
