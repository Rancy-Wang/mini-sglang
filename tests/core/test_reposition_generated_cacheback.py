from __future__ import annotations

import minisgl.core as core
import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.core import Req, SamplingParams
from minisgl.kernel.radix_reposition import TOKEN_KIND, compile_radix_reposition_layout
from minisgl.message import TokenizeMsg, WarmupAckMsg
from minisgl.scheduler.cache import CacheManager
from minisgl.tokenizer.reposition_sequence import RepositionSequenceState
from minisgl.tokenizer.tokenize import TokenizedResult


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    yield
    core._GLOBAL_CTX = old_ctx


def _final_message():
    token_ids = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
    drop_positions = torch.tensor([4], dtype=torch.int32)
    drop_offsets = torch.tensor([0, 1], dtype=torch.int32)
    drop_ranges = torch.tensor([0, 2], dtype=torch.int32)
    reposition_boundaries = torch.tensor([3], dtype=torch.int32)
    reposition_offsets = reposition_boundaries + 1
    layout = compile_radix_reposition_layout(
        token_ids,
        drop_positions,
        drop_offsets,
        drop_ranges,
        reposition_boundaries,
        reposition_offsets,
    )
    request = TokenizeMsg(
        uid=71,
        text="already tokenized",
        sampling_params=SamplingParams(max_tokens=3),
        reposition=[1],
    )
    visible_until = torch.full((6,), torch.iinfo(torch.int32).max, dtype=torch.int32)
    visible_until[:2] = 4
    tokenized = TokenizedResult(
        input_ids=token_ids[layout.keep_mask],
        true_positions=layout.positions[layout.keep_mask],
        raw_positions=torch.nonzero(layout.keep_mask, as_tuple=False).view(-1).to(torch.int32),
        radix_input_ids=layout.records[layout.token_to_key[layout.keep_mask]],
        radix_match_ids=layout.records,
        prefix_keep_mask=layout.keep_mask[:-1].to(torch.int32),
        prompt_tokens=6,
        full_input_ids=token_ids,
        full_token_visible_until=visible_until,
        full_keep_mask=layout.keep_mask.to(torch.int32),
        reposition_raw_boundaries=reposition_boundaries,
        reposition_insert_offsets=reposition_offsets,
        reposition_input_ids=token_ids,
        reposition_layout=layout,
        tokenize_invocations=1,
        chat_template_invocations=1,
    )
    state = RepositionSequenceState.pending(request, tokenized)
    state.activate(step_token_budget=32)
    first = state.build_next_msg()
    assert first.is_warmup
    state.accept_ack(
        WarmupAckMsg(
            uid=71,
            hit_ratio=0.0,
            cached_tokens=0,
            drop_skipped_tokens=0,
            finished=True,
        )
    )
    final = state.build_next_msg()
    assert not final.is_warmup
    assert final.radix_commit_key_len is None
    return final


def test_final_reposition_cacheback_includes_two_computed_generated_tokens() -> None:
    message = _final_message()
    page_table = torch.full((1, 32), -1, dtype=torch.int32)
    cache = CacheManager(32, 1, page_table, "radix")
    history_pages = cache._allocate(4)
    source = cache.prefix_cache.insert_prefix(
        message.radix_match_ids[:4],
        history_pages,
    )
    source_handle = source.handle
    cache.lock(source_handle)
    req = Req(
        input_ids=message.input_ids,
        true_positions=message.true_positions,
        raw_positions=message.raw_positions,
        radix_input_ids=message.radix_input_ids,
        radix_match_ids=message.radix_match_ids,
        initial_full_match_indices=history_pages.clone(),
        initial_active_cached_len=2,
        true_seq_len=int(message.radix_next_position),
        table_idx=0,
        cached_len=2,
        output_len=3,
        uid=message.uid,
        sampling_params=message.sampling_params,
        cache_handle=source_handle,
        prompt_tokens=message.prompt_tokens or 0,
        prefix_keep_mask=message.prefix_keep_mask,
        radix_key_virtual_mask=message.radix_key_virtual_mask,
        radix_key_to_token=message.radix_key_to_token,
        radix_token_to_key=message.radix_token_to_key,
        radix_commit_key_len=message.radix_commit_key_len,
        radix_positions=message.radix_positions,
        radix_repos_info=message.radix_repos_info,
        radix_next_position=message.radix_next_position,
        radix_current_reposition=message.radix_current_reposition,
    )

    page_table[0, :2] = history_pages[2:]
    prompt_pages = cache._allocate(len(req.input_ids) - req.cached_len)
    page_table[0, 2 : 2 + len(prompt_pages)] = prompt_pages
    generated_pages: list[int] = []
    for token_id in (100, 101, 102):
        req.complete_one()
        req.append_host(torch.tensor([token_id], dtype=torch.int32))
        if token_id != 102:
            page = cache._allocate(1)
            page_table[0, req.cached_len] = page[0]
            generated_pages.append(int(page[0]))

    expected_key_len = CacheManager._delta_key_prefix_len(req)
    expected_records = req.radix_match_ids[:expected_key_len].clone()
    expected_virtual = req.radix_key_virtual_mask[:expected_key_len].clone()
    assert expected_records[~expected_virtual][-2:, 1].tolist() == [100, 101]
    assert expected_records[~expected_virtual][-2:, 0].tolist() == [TOKEN_KIND, TOKEN_KIND]

    cache.cache_req(req, finished=True)

    matched = cache.prefix_cache.match_prefix(expected_records, expected_virtual).cuda_handle
    assert matched.cached_len == expected_key_len
    real_pages = matched.get_matched_indices()[~expected_virtual]
    assert real_pages[-2:].tolist() == generated_pages
    cache.check_integrity()
