"""Generated-token CPU metadata must retain its exact logical sequence."""

from __future__ import annotations

import pytest
import torch

from minisgl.core import Req, SamplingParams


def _req(*, structured: bool, output_len: int = 2) -> Req:
    tokens = torch.tensor([10, 11, 12], dtype=torch.int32)
    positions = torch.arange(3, dtype=torch.int32)
    keys = (
        torch.column_stack(
            (
                torch.zeros(3, dtype=torch.int32),
                tokens,
                torch.full((3,), -1, dtype=torch.int32),
                positions,
            )
        )
        if structured
        else tokens.to(torch.int64)
    )
    return Req(
        input_ids=tokens,
        true_positions=positions.clone(),
        raw_positions=positions.clone(),
        radix_input_ids=keys.clone(),
        radix_match_ids=keys.clone(),
        initial_full_match_indices=torch.empty(0, dtype=torch.int32),
        initial_active_cached_len=0,
        true_seq_len=3,
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=object(),
        radix_key_virtual_mask=(torch.zeros(3, dtype=torch.bool) if structured else None),
        radix_key_to_token=(torch.arange(3, dtype=torch.int64) if structured else None),
        radix_token_to_key=(torch.arange(3, dtype=torch.int64) if structured else None),
        radix_positions=(positions.clone() if structured else None),
        radix_repos_info=(torch.full((3,), -1, dtype=torch.int32) if structured else None),
        radix_next_position=(3 if structured else None),
    )


@pytest.mark.parametrize("structured", [False, True])
def test_overlapped_tokens_keep_their_own_positions(structured: bool) -> None:
    req = _req(structured=structured, output_len=2)
    req.complete_one()
    req.complete_one()  # The second GPU forward can precede the first CPU append.
    req.append_host(torch.tensor([21], dtype=torch.int32))
    req.append_host(torch.tensor([22], dtype=torch.int32))

    assert req.input_ids.tolist() == [10, 11, 12, 21, 22]
    assert req.true_positions.tolist() == [0, 1, 2, 3, 4]
    assert req.raw_positions.tolist() == [0, 1, 2, 3, 4]
    if structured:
        assert req.radix_match_ids[-2:].tolist() == [[0, 21, -1, 3], [0, 22, -1, 4]]
        assert req.radix_input_ids[-2:].tolist() == [[0, 21, -1, 3], [0, 22, -1, 4]]
        assert req.radix_positions.tolist() == [0, 1, 2, 3, 4]
        assert req.radix_repos_info.tolist() == [-1, -1, -1, -1, -1]
        assert req.radix_key_virtual_mask.tolist() == [False] * 5
        assert req.radix_key_to_token.tolist() == list(range(5))
        assert req.radix_token_to_key.tolist() == list(range(5))
    else:
        assert req.radix_match_ids.tolist() == [10, 11, 12, 21, 22]


def test_host_buffer_rebuilds_after_external_compaction() -> None:
    req = _req(structured=False)
    req.complete_one()
    req.append_host(torch.tensor([21], dtype=torch.int32))
    old_view = req.input_ids
    req.input_ids = req.input_ids[[0, 2, 3]].contiguous()
    req._append_host_tensor("input_ids", torch.tensor([22], dtype=torch.int32))
    assert req.input_ids.tolist() == [10, 12, 21, 22]
    assert old_view.tolist() == [10, 11, 12, 21]


def test_host_buffer_grows_without_changing_prior_logical_values() -> None:
    req = _req(structured=False, output_len=1)
    for token in (21, 22, 23, 24):
        req._append_host_tensor("input_ids", torch.tensor([token], dtype=torch.int32))
    assert req.input_ids.tolist() == [10, 11, 12, 21, 22, 23, 24]


def test_match_stop_checks_only_a_bounded_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    req = _req(structured=False)
    req.input_ids = torch.arange(10_000, dtype=torch.int32)
    req.stop_token_seqs = [[9998, 9999], [9999], [], list(range(10_001))]
    req.stop = ["long", "short", "empty", "impossible"]
    observed_lengths: list[int] = []
    original_tolist = torch.Tensor.tolist

    def record_tolist(tensor: torch.Tensor):
        observed_lengths.append(len(tensor))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", record_tolist)
    assert req.match_stop() == (True, "long")
    assert observed_lengths == [2]
