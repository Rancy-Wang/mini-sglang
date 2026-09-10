from types import SimpleNamespace

import pytest
import torch
from minisgl.core import validate_occurrence_positions
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.scheduler import ForwardInput, Scheduler


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
