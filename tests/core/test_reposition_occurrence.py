from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import minisgl.attention.base as attention_base
import minisgl.core as core
import minisgl.scheduler.prefill as prefill_module
import pytest
import torch
from minisgl.attention.base import (
    build_occurrence_attention_batch,
    compile_context_page_tables,
)
from minisgl.distributed import DistributedInfo
from minisgl.kernel.radix_reposition import RadixRepositionLayout
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.config import SchedulerConfig
from minisgl.scheduler.prefill import PrefillManager, RepositionCapacityError
from minisgl.scheduler.reposition_occurrence import (
    CompactRepositionOccurrenceLayout,
    compile_occurrence_window,
    pack_compact_occurrence_pending_fields,
    unpack_compact_occurrence_pending_fields,
)
from minisgl.tokenizer.reposition_occurrence import compile_reposition_occurrence_plan


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    yield
    core._GLOBAL_CTX = old_ctx


def _layout() -> RadixRepositionLayout:
    token_count = 8
    return RadixRepositionLayout(
        drop_insert_offsets=torch.tensor([4, 7], dtype=torch.int32),
        drop_range_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
        drop_ranges=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        records=torch.empty((0, 4), dtype=torch.int32),
        virtual_mask=torch.empty(0, dtype=torch.bool),
        key_to_token=torch.empty(0, dtype=torch.int64),
        token_to_key=torch.arange(token_count, dtype=torch.int64),
        positions=torch.tensor([0, 0, 1, 1, 2, 3, 4, 5], dtype=torch.int32),
        repos_info=torch.zeros(token_count, dtype=torch.int32),
        keep_mask=torch.tensor([False, True, False, True, True, True, True, True]),
        materialized_stage=torch.zeros(token_count, dtype=torch.int32),
        birth_positions=torch.tensor([0, 1, 2, 3, 3, 4, 5, 5], dtype=torch.int32),
        birth_stages=torch.tensor([0, 0, 0, 0, 1, 1, 1, 2], dtype=torch.int32),
        transition_offsets=torch.tensor([0, 3, 7], dtype=torch.int32),
        transition_raw_tokens=torch.tensor([1, 2, 3, 3, 4, 5, 6], dtype=torch.int32),
        transition_old_positions=torch.tensor([1, 2, 3, 2, 3, 4, 5], dtype=torch.int32),
        transition_new_positions=torch.tensor([0, 1, 2, 1, 2, 3, 4], dtype=torch.int32),
        effective_reposition_stages=torch.tensor([1, 2], dtype=torch.int32),
        drop_event_to_key=torch.tensor([-1, -1], dtype=torch.int64),
        effective_repositions=torch.tensor([True, True]),
        ignored_repositions=torch.tensor([False, False]),
        next_position=6,
        current_reposition=1,
        compile_ns=0,
    )


def _plan():
    visible_until = torch.tensor([4, 9, 7, 9, 9, 9, 9, 9], dtype=torch.int32)
    return compile_reposition_occurrence_plan(_layout(), visible_until)


def test_occurrence_expansion_assigns_one_identity_per_raw_position_pair() -> None:
    plan = _plan()

    assert plan.occurrence_count == 15
    assert plan.birth_occurrences.tolist() == list(range(8))
    assert plan.occurrence_raw_tokens.tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        1,
        2,
        3,
        3,
        4,
        5,
        6,
    ]
    assert plan.occurrence_positions.tolist() == [
        0,
        1,
        2,
        3,
        3,
        4,
        5,
        5,
        0,
        1,
        2,
        1,
        2,
        3,
        4,
    ]
    assert plan.terminal_occurrences.tolist() == [0, 8, 9, 11, 12, 13, 14, 7]
    terminal_positions = plan.occurrence_positions[plan.terminal_occurrences.to(torch.int64)]
    assert torch.equal(terminal_positions, _layout().positions)

    keys = [
        plan.segment_key_occurrences[
            plan.segment_key_offsets[index] : plan.segment_key_offsets[index + 1]
        ].tolist()
        for index in range(plan.segment_count)
    ]
    assert list(zip(plan.segment_query_starts.tolist(), plan.segment_query_ends.tolist())) == [
        (0, 4),
        (4, 7),
        (7, 8),
    ]
    assert keys == [
        [0, 1, 2, 3],
        [8, 9, 10, 4, 5, 6],
        [8, 11, 12, 13, 14, 7],
    ]
    raw_three_occurrences = torch.nonzero(plan.occurrence_raw_tokens == 3, as_tuple=False).view(-1)
    assert raw_three_occurrences.tolist() == [3, 10, 11]
    assert plan.occurrence_positions[raw_three_occurrences].tolist() == [3, 2, 1]


def test_occurrence_expansion_rejects_stale_transition_source_position() -> None:
    layout = _layout()
    stale = layout.transition_old_positions.clone()
    stale[-4] = 3

    with pytest.raises(ValueError, match="old positions"):
        compile_reposition_occurrence_plan(
            replace(layout, transition_old_positions=stale),
            torch.tensor([4, 9, 7, 9, 9, 9, 9, 9], dtype=torch.int32),
        )


def _compact_layout(layout: RadixRepositionLayout) -> CompactRepositionOccurrenceLayout:
    return CompactRepositionOccurrenceLayout(
        birth_positions=layout.birth_positions,
        birth_stages=layout.birth_stages,
        transition_offsets=layout.transition_offsets,
        transition_raw_tokens=layout.transition_raw_tokens,
        transition_old_positions=layout.transition_old_positions,
        transition_new_positions=layout.transition_new_positions,
    )


def _semantic_window(plan, query_start: int, query_end: int):
    semantic = []
    for index, (segment_start, segment_end) in enumerate(
        zip(plan.segment_query_starts.tolist(), plan.segment_query_ends.tolist(), strict=True)
    ):
        local_start = max(query_start, segment_start)
        local_end = min(query_end, segment_end)
        if local_start >= local_end:
            continue
        key_start = int(plan.segment_key_offsets[index])
        key_end = int(plan.segment_key_offsets[index + 1])
        keys = plan.segment_key_occurrences[key_start:key_end]
        prefix_length = len(keys) - (segment_end - segment_start)
        keys = keys[: prefix_length + local_end - segment_start].to(torch.int64)
        semantic.append(
            (
                local_start,
                local_end,
                list(
                    zip(
                        plan.occurrence_raw_tokens[keys].tolist(),
                        plan.occurrence_positions[keys].tolist(),
                        strict=True,
                    )
                ),
            )
        )
    return semantic


def test_compact_occurrence_pending_adapter_round_trips_without_expansion() -> None:
    compact = _compact_layout(_layout())
    packed = pack_compact_occurrence_pending_fields(
        birth_positions=compact.birth_positions,
        birth_stages=compact.birth_stages,
        transition_offsets=compact.transition_offsets,
        transition_raw_tokens=compact.transition_raw_tokens,
        transition_old_positions=compact.transition_old_positions,
        transition_new_positions=compact.transition_new_positions,
    )

    restored = unpack_compact_occurrence_pending_fields(SimpleNamespace(**packed))

    assert restored is not None
    for name in (
        "birth_positions",
        "birth_stages",
        "transition_offsets",
        "transition_raw_tokens",
        "transition_old_positions",
        "transition_new_positions",
    ):
        assert torch.equal(getattr(restored, name), getattr(compact, name))
    assert packed["occurrence_segment_key_offsets"].tolist() == [-1]
    assert packed["occurrence_segment_key_indices"].numel() == 0


@pytest.mark.parametrize("query_start", [0, 2, 4, 7])
def test_lazy_occurrence_window_matches_full_plan_semantics(query_start: int) -> None:
    layout = _layout()
    visibility = torch.tensor([4, 9, 7, 9, 9, 9, 9, 9], dtype=torch.int32)
    full = compile_reposition_occurrence_plan(layout, visibility)
    lazy = compile_occurrence_window(
        _compact_layout(layout),
        visibility,
        layout.positions,
        query_start=query_start,
        query_end=8,
    )

    assert _semantic_window(lazy, query_start, 8) == _semantic_window(full, query_start, 8)
    assert torch.equal(
        lazy.occurrence_positions[lazy.birth_occurrences.to(torch.int64)],
        full.occurrence_positions[full.birth_occurrences.to(torch.int64)],
    )
    assert torch.equal(
        lazy.occurrence_positions[lazy.terminal_occurrences.to(torch.int64)],
        full.occurrence_positions[full.terminal_occurrences.to(torch.int64)],
    )
    if query_start == 7:
        assert lazy.occurrence_count < full.occurrence_count


def _runtime_req(plan, *, cached_len: int, page_base: int, table_idx: int):
    return SimpleNamespace(
        reposition_execution_mode="paged-occurrence",
        occurrence_raw_tokens=plan.occurrence_raw_tokens,
        occurrence_positions=plan.occurrence_positions,
        occurrence_segment_query_starts=plan.segment_query_starts,
        occurrence_segment_query_ends=plan.segment_query_ends,
        occurrence_segment_key_offsets=plan.segment_key_offsets,
        occurrence_segment_key_indices=plan.segment_key_occurrences,
        occurrence_pages=torch.arange(
            page_base,
            page_base + plan.occurrence_count,
            dtype=torch.int32,
        ),
        cached_len=cached_len,
        device_len=8,
        extend_len=8 - cached_len,
        table_idx=table_idx,
        true_positions=_layout().birth_positions,
    )


def test_occurrence_attention_batches_partial_hits_for_multiple_requests() -> None:
    plan = _plan()
    reqs = [
        _runtime_req(plan, cached_len=2, page_base=100, table_idx=3),
        _runtime_req(plan, cached_len=4, page_base=200, table_idx=7),
    ]

    batch = build_occurrence_attention_batch(reqs)

    assert batch.num_queries == 10
    assert batch.cu_seqlens_q.tolist() == [0, 2, 5, 6, 9, 10]
    assert batch.key_positions[:4].tolist() == [0, 1, 2, 3]
    assert batch.key_positions[-6:].tolist() == [23, 26, 27, 28, 29, 22]
    assert batch.direct_pages is not None
    assert len(torch.unique(batch.direct_pages)) == 2 * plan.occurrence_count

    compiled = compile_context_page_tables(
        torch.full((1, 1), -1, dtype=torch.int32),
        batch,
    )
    expected = batch.direct_pages[batch.key_positions.to(torch.int64)]
    assert torch.equal(compiled.flat_indices, expected)


def test_occurrence_sliding_aot_batch_matches_python_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    reqs = [
        _runtime_req(plan, cached_len=2, page_base=100, table_idx=3),
        _runtime_req(plan, cached_len=4, page_base=200, table_idx=7),
    ]
    for req in reqs:
        req.initial_active_cached_len = req.cached_len

    with monkeypatch.context() as fallback:
        fallback.setattr(
            attention_base,
            "try_build_occurrence_sliding_plan",
            lambda *args, **kwargs: None,
        )
        expected = attention_base.build_occurrence_attention_batch(reqs, sliding_window=2)

    actual = attention_base.build_occurrence_attention_batch(reqs, sliding_window=2)

    assert actual.cached_tokens == expected.cached_tokens
    assert actual.max_seqlen_q == expected.max_seqlen_q
    assert actual.max_seqlen_k == expected.max_seqlen_k
    assert len(actual.cached_positions) == len(expected.cached_positions)
    for actual_positions, expected_positions in zip(
        actual.cached_positions, expected.cached_positions, strict=True
    ):
        assert torch.equal(actual_positions, expected_positions)
    for name in (
        "segment_table_indices",
        "key_positions",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "direct_pages",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name))


def test_occurrence_owned_prompt_prefix_allows_generated_cache_candidates() -> None:
    page_table = torch.full((1, 8), -1, dtype=torch.int32)
    manager = CacheManager(16, 1, page_table, "radix")
    candidates = manager._allocate(3)
    canonical = manager._allocate(3)
    key = torch.tensor([10, 11, 12], dtype=torch.int64)
    insert_result = manager.prefix_cache.insert_prefix(key, canonical)
    req = SimpleNamespace(
        initial_active_cached_len=2,
        retry_transformed_mask=None,
        occurrence_terminal_owned_mask=torch.tensor([True, True]),
        inactive_cached_positions=None,
        inactive_cached_pages=None,
        radix_token_to_key=None,
    )

    manager._free_finished_candidates(
        req,
        candidates,
        torch.arange(3, dtype=torch.int64),
        insert_result,
    )

    assert set(candidates.tolist()).issubset(set(manager.free_slots.tolist()))
    assert not set(canonical.tolist()) & set(manager.free_slots.tolist())
    manager.check_integrity()


def test_paged_occurrence_is_the_scheduler_default() -> None:
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float16,
    )

    assert config.reposition_execution_mode == "paged-occurrence"


def test_occurrence_capacity_failure_preserves_an_already_allocated_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core.get_global_ctx().attn_backend = SimpleNamespace(supports_multi_context_mask_prefill=True)
    allocated = SimpleNamespace(
        uid=31,
        use_context_mask=True,
        reposition_execution_mode="paged-occurrence",
    )

    class _FakeAdder:
        def __init__(self, **_kwargs):
            self.calls = 0
            self.reserved_size = 0

        def try_add_one(self, pending_req, _context_plan):
            self.calls += 1
            if self.calls == 1:
                return allocated
            raise RepositionCapacityError(
                uid=pending_req.uid,
                required_pages=12,
                available_pages=3,
                matched_pages=4,
                retry_pages=8,
            )

    monkeypatch.setattr(prefill_module, "PrefillAdder", _FakeAdder)
    manager = PrefillManager(
        cache_manager=SimpleNamespace(),
        table_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(inflight_tokens=0),
    )
    manager.pending_list.extend(
        [
            SimpleNamespace(
                uid=31,
                use_context_mask=True,
                chunked_req=None,
                reposition_execution_mode="paged-occurrence",
            ),
            SimpleNamespace(
                uid=32,
                use_context_mask=True,
                chunked_req=None,
                reposition_execution_mode="paged-occurrence",
            ),
        ]
    )

    batch = manager.schedule_next_batch(prefill_budget=32)

    assert batch is not None
    assert batch.reqs == [allocated]
    assert [req.uid for req in manager.pending_list] == [32]
