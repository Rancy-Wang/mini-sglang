from types import SimpleNamespace

import pytest
import torch
from minisgl.core import validate_occurrence_positions
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
