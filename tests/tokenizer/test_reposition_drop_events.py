from __future__ import annotations

from dataclasses import replace

import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.tokenizer.drop_rules import (
    DropCompileContext,
    KeepTextDropRule,
    MessageDropRule,
    TextDropRule,
    ThinkingDropRule,
    TokenDropEvents,
)
from minisgl.tokenizer.tokenize import TokenizeManager, resolve_reposition_token_boundaries


def _context() -> DropCompileContext:
    def compile_events(events: dict[int, list[tuple[int, int]]]) -> TokenDropEvents:
        flat = [value for ranges in events.values() for item in ranges for value in item]
        return TokenDropEvents(
            event_insert_offsets=torch.tensor(list(events), dtype=torch.int32),
            range_offsets=torch.tensor([0, len(flat) // 2], dtype=torch.int32),
            raw_ranges=torch.tensor(flat, dtype=torch.int32),
            full_token_visible_until=torch.full((12,), 12, dtype=torch.int32),
            effective_event_count=len(events),
            effective_ranges=tuple(item for ranges in events.values() for item in ranges),
        )

    return DropCompileContext(
        raw_messages=(
            {"role": "user", "content": "alpha"},
            {"role": "assistant", "content": "thought"},
        ),
        owner_ranges={2: ((1, 3),), 3: ((5, 7),)},
        provenance=object(),
        full_input_ids=tuple(range(12)),
        owners=(2,) * 4 + (3,) * 8,
        target_offset=2,
        normalized_message_count=4,
        is_gpt_oss=False,
        harmony_thinking_ranges={},
        normalize_content=str,
        rendered_source_start=lambda *args, **kwargs: 0,
        token_ranges_for_char_spans=lambda *args, **kwargs: [(1, 3)],
        canonicalize_ranges=lambda ranges: sorted(set(ranges)),
        position_ranges_from_ids=lambda ids: [(min(ids), max(ids) + 1)] if ids else [],
        find_owned_subsequence=lambda *args, **kwargs: (5, 7),
        encode_text=lambda text: [1],
        compile_events=compile_events,
    )


def test_drop_rules_share_one_token_event_compilation_interface() -> None:
    context = _context()
    rules = (
        MessageDropRule({1: (0,)}),
        TextDropRule.from_payload(
            {
                "type": "text_drop",
                "drop_messages": [
                    {"role": "user", "content": "alpha"},
                    {"role": "assistant", "content": None},
                ],
            },
            context.raw_messages,
        ),
        KeepTextDropRule(
            full_messages=tuple(dict(message) for message in context.raw_messages),
            keep_spans=(None, (0, 0)),
        ),
        ThinkingDropRule({1: "thought"}),
    )

    compiled = [rule.compile_token_drop_events(context) for rule in rules]

    assert [events.event_insert_offsets.tolist() for events in compiled] == [
        [3],
        [2],
        [3],
        [3],
    ]
    assert all(events.raw_ranges.numel() > 0 for events in compiled)


def test_thinking_drop_uses_harmony_ranges_without_offset_provenance() -> None:
    context = replace(
        _context(),
        provenance=None,
        is_gpt_oss=True,
        harmony_thinking_ranges={1: ((8, 10),)},
    )

    events = ThinkingDropRule({1: "thought"}).compile_token_drop_events(context)

    assert events.event_insert_offsets.tolist() == [3]
    assert events.raw_ranges.tolist() == [8, 10]


def test_reposition_resolver_uses_explicit_public_owner_mapping() -> None:
    resolved = resolve_reposition_token_boundaries(
        [0, 2],
        {4: [(1, 3), (5, 6)], 9: [(8, 10)]},
        {0: 4, 2: 9},
    )

    assert resolved.raw_boundaries.tolist() == [5, 9]
    assert resolved.insert_offsets.tolist() == [6, 10]


def test_tokenizer_filters_nonprefix_effective_drop_events_before_radix_compile() -> None:
    manager = object.__new__(TokenizeManager)
    manager.radix_drop_key_mode = "delta-marker"
    events = TokenDropEvents(
        event_insert_offsets=torch.tensor([2, 4], dtype=torch.int32),
        range_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
        raw_ranges=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        full_token_visible_until=torch.full((4,), 4, dtype=torch.int32),
        effective_event_count=-1,
        effective_ranges=((2, 3),),
    )

    layout = manager._compile_delta_layout(
        torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        events,
        torch.tensor([1, 1, 0, 1], dtype=torch.bool),
        None,
        None,
    )

    assert layout is not None
    assert layout.drop_insert_offsets.tolist() == [4]
    assert layout.drop_range_offsets.tolist() == [0, 1]
    assert layout.drop_ranges.tolist() == [2, 3]
    assert layout.records[-1].tolist() == [1, -3, -4, -1]


def test_target_filter_splits_mixed_current_and_future_ranges_at_one_insertion() -> None:
    events = TokenDropEvents(
        event_insert_offsets=torch.tensor([4], dtype=torch.int32),
        range_offsets=torch.tensor([0, 1], dtype=torch.int32),
        raw_ranges=torch.tensor([0, 4], dtype=torch.int32),
        full_token_visible_until=torch.full((4,), 4, dtype=torch.int32),
        effective_event_count=-1,
        effective_ranges=((0, 2),),
    )

    positions, offsets, ranges = TokenizeManager._select_effective_delta_wire(
        events,
        torch.tensor([0, 0, 1, 1], dtype=torch.bool),
    )

    assert positions.tolist() == [4]
    assert offsets.tolist() == [0, 1]
    assert ranges.tolist() == [0, 2]


def test_target_filter_preserves_multiple_effective_fragments_at_one_insertion() -> None:
    events = TokenDropEvents(
        event_insert_offsets=torch.tensor([6], dtype=torch.int32),
        range_offsets=torch.tensor([0, 1], dtype=torch.int32),
        raw_ranges=torch.tensor([0, 6], dtype=torch.int32),
        full_token_visible_until=torch.full((6,), 6, dtype=torch.int32),
        effective_event_count=-1,
        effective_ranges=((0, 2), (4, 6)),
    )

    positions, offsets, ranges = TokenizeManager._select_effective_delta_wire(
        events,
        torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
    )

    assert positions.tolist() == [6]
    assert offsets.tolist() == [0, 2]
    assert ranges.tolist() == [0, 2, 4, 6]


@pytest.mark.parametrize("reposition", [[1, 1], [2, 1], [True], [-1]])
def test_reposition_resolver_rejects_noncanonical_ids(reposition: list[int]) -> None:
    with pytest.raises(ValueError):
        resolve_reposition_token_boundaries(reposition, {1: [(0, 1)]}, {1: 1})
