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
        usage_cached_tokens=0,
        usage_repos_tokens=0,
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


@pytest.mark.parametrize("structured", [False, True])
def test_completion_usage_counts_committed_tokens_not_overlap_lookahead(structured):
    req = _req(structured=structured, output_len=2)
    req.complete_one()
    req.complete_one()
    assert req.reported_completion_tokens == 0
    req.append_host(torch.tensor([21], dtype=torch.int32))
    assert req.reported_completion_tokens == 1
    assert not req.can_decode  # No more GPU work; one host output is still pending.
    req.append_host(torch.tensor([22], dtype=torch.int32))
    assert req.reported_completion_tokens == 2


@pytest.mark.parametrize("structured", [False, True])
def test_completion_usage_is_invariant_under_prompt_compaction(structured):
    req = _req(structured=structured, output_len=2)
    req.complete_one()
    req.append_host(torch.tensor([21], dtype=torch.int32))
    assert req.reported_completion_tokens == 1
    # The scheduler removes one prompt token from both the host stream and the
    # device budget; raw/full prompt_tokens is deliberately not the denominator.
    req.input_ids = req.input_ids[[0, 2, 3]].contiguous()
    req.max_device_len -= 1
    req.device_len -= 1
    assert req.reported_completion_tokens == 1


@pytest.mark.parametrize("structured", [False, True])
def test_seeded_sampling_keeps_device_offset_separate_from_reported_usage(structured, monkeypatch):
    from types import SimpleNamespace

    import minisgl.engine.sample as sampling

    # This checks the production prepare path without requiring a CUDA transfer.
    monkeypatch.setattr(sampling, "make_device_tensor",
                        lambda data, dtype, device: torch.tensor(data, dtype=dtype))
    req = _req(structured=structured, output_len=3)
    req.sampling_params = SamplingParams(max_tokens=3, temperature=0.8, seed=17)
    sampler = sampling.Sampler(device=torch.device("cpu"), vocab_size=100)
    batch = SimpleNamespace(reqs=[req])

    def check(offset, committed):
        args = sampler.prepare(batch)
        # Exact pre-change formula consumed by the RNG, not the usage counter.
        baseline_offset = req.device_len - (req.max_device_len - req.output_len)
        assert args.offsets.tolist() == [baseline_offset] == [offset]
        assert args.seeds.tolist() == [17]
        assert req.reported_completion_tokens == committed

    check(0, 0)
    req.complete_one()
    check(1, 0)
    req.complete_one()  # Overlap can schedule while CPU output is still pending.
    check(2, 0)
    req.append_host(torch.tensor([21], dtype=torch.int32))
    check(2, 1)


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


def _host_scheduler():
    from contextlib import nullcontext
    from types import SimpleNamespace

    from minisgl.scheduler.decode import DecodeManager
    from minisgl.scheduler.scheduler import Scheduler

    scheduler = object.__new__(Scheduler)
    scheduler.finished_reqs = set()
    scheduler.request_metrics = {}
    scheduler.eos_token_ids = {99}
    scheduler.decode_manager = DecodeManager(page_size=1)
    scheduler.cache_manager = SimpleNamespace(
        lazy_free_region=nullcontext, cache_req=lambda *args, **kwargs: None,
    )
    scheduler._wait_for_transition = lambda req: None
    freed, replies = [], []
    scheduler._free_req_resources = freed.append
    scheduler.send_result = replies.extend
    return scheduler, freed, replies


def _deliver(scheduler, req, token):
    from types import SimpleNamespace

    batch = SimpleNamespace(reqs=[req], is_prefill=False)
    copy_done = SimpleNamespace(synchronize=lambda: None)
    scheduler._process_last_data((SimpleNamespace(batch=batch),
                                 (None, torch.tensor([token]), copy_done)))


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("output_len", [1, 2, 5])
def test_scheduler_drains_exact_length_before_freeing(structured, overlap, output_len):
    req = _req(structured=structured, output_len=output_len)
    req.sampling_params.ignore_eos = True
    scheduler, freed, replies = _host_scheduler()
    submitted = 0
    for committed in range(output_len):
        target = min(output_len, committed + (2 if overlap else 1))
        while submitted < target:
            req.complete_one()
            scheduler.decode_manager.filter_reqs([req])
            submitted += 1
        # Even an EOS sample must be delivered when ignore_eos is enabled.
        _deliver(scheduler, req, 99)
        final = committed == output_len - 1
        assert replies[-1].finished is final
        assert replies[-1].finish_reason == ("length" if final else None)
        assert freed == ([req] if final else [])
    assert [reply.next_token for reply in replies] == [99] * output_len
    assert replies[-1].completion_tokens == output_len
    assert req.reported_completion_tokens == output_len
    assert not scheduler.decode_manager.runnable


@pytest.mark.parametrize("stop_kind", ["eos", "explicit"])
def test_scheduler_early_stop_drains_stale_overlap_without_double_free(stop_kind):
    req = _req(structured=True, output_len=5)
    req.sampling_params.ignore_eos = stop_kind == "explicit"
    if stop_kind == "explicit":
        req.stop_token_seqs, req.stop = [[99]], ["custom stop"]
    scheduler, freed, replies = _host_scheduler()
    req.complete_one()
    req.complete_one()
    scheduler.decode_manager.filter_reqs([req])
    _deliver(scheduler, req, 99)
    _deliver(scheduler, req, 21)
    assert len(replies) == 1
    assert replies[0].finished and replies[0].finish_reason == "stop"
    assert req.reported_completion_tokens == 1
    assert freed == [req]
    assert not scheduler.decode_manager.runnable
