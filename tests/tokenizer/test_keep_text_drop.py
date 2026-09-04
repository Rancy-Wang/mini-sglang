from __future__ import annotations

import pytest

from minisgl.tokenizer.drop_rules import KeepTextDropRule, parse_drop_rule
from minisgl.tokenizer.template_provenance import TemplateTokenProvenance
from minisgl.tokenizer.tokenize import TokenizeManager


def test_keep_text_drop_projects_rightmost_and_round_trips_wire() -> None:
    full_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "repeat target old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "repeat target newest"},
    ]
    visible = [{"role": "user", "content": "target"}]
    parsed = parse_drop_rule({"type": "keep_text_drop", "full_messages": full_messages}, visible)
    assert isinstance(parsed, KeepTextDropRule)
    assert parsed.keep_spans == (None, None, None, (7, 13))

    wire = parsed.to_wire()
    restored = parse_drop_rule(wire, full_messages, allow_internal=True)
    assert restored == parsed


def test_keep_text_drop_protocol_metadata_is_part_of_matching() -> None:
    call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q":"x"}'},
    }
    full_messages = [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "assistant", "content": ""},
    ]
    visible = [{"role": "assistant", "content": "", "tool_calls": [call]}]
    parsed = parse_drop_rule({"type": "keep_text_drop", "full_messages": full_messages}, visible)
    assert isinstance(parsed, KeepTextDropRule)
    assert parsed.keep_spans == ((0, 0), None)


def test_keep_text_drop_force_falls_back_to_visible_prompt() -> None:
    visible = [{"role": "user", "content": "not in history"}]
    parsed = parse_drop_rule(
        {
            "type": "keep_text_drop",
            "full_messages": [{"role": "user", "content": "different"}],
            "force": True,
        },
        visible,
    )
    assert isinstance(parsed, KeepTextDropRule)
    assert parsed.use_visible_as_full
    assert parsed.full_messages == tuple(visible)


def test_keep_text_drop_mismatch_is_rejected_without_force() -> None:
    with pytest.raises(ValueError, match="projection failed"):
        parse_drop_rule(
            {
                "type": "keep_text_drop",
                "full_messages": [{"role": "user", "content": "different"}],
            },
            [{"role": "user", "content": "not in history"}],
        )


class _FakeTokenizer:
    name_or_path = "fake"


def test_partial_keep_preserves_template_and_boundary_crossing_tokens() -> None:
    rendered = "[U]alpha assistant[/U][A]"
    provenance = TemplateTokenProvenance(
        input_ids=[10, 11, 12, 13, 14],
        owners=[0, 0, 0, 0, 1],
        offsets=[(0, 3), (3, 8), (8, 18), (18, 22), (22, 25)],
        rendered_text=rendered,
        char_owners=[0] * 22 + [1] * 3,
        cross_owner_tokens=0,
    )
    rule = KeepTextDropRule(
        full_messages=({"role": "user", "content": "alpha assistant"},),
        keep_spans=((6, 12),),  # "assist" cuts through the token " assistant".
    )
    manager = TokenizeManager(_FakeTokenizer())
    events = manager._compile_rule_position_events(
        rule,
        raw_messages=list(rule.full_messages),
        owner_ranges={0: [(0, 4)], 1: [(4, 5)]},
        provenance=provenance,
        full_input_ids=provenance.input_ids,
        owners=provenance.owners,
        target_offset=0,
        normalized_message_count=1,
    )

    # Only the complete non-selected content token "alpha" is dropped.  [U], [/U],
    # the generation prompt, and the boundary-crossing " assistant" token remain.
    assert events == {0: [(1, 2)]}


def test_unselected_message_drops_its_template_tokens_too() -> None:
    provenance = TemplateTokenProvenance(
        input_ids=[10, 11, 12, 13],
        owners=[0, 0, 1, 2],
        offsets=[(0, 3), (3, 7), (7, 11), (11, 14)],
        rendered_text="[U]text[/U][A]",
        char_owners=[0] * 7 + [1] * 4 + [2] * 3,
        cross_owner_tokens=0,
    )
    rule = KeepTextDropRule(
        full_messages=(
            {"role": "user", "content": "text"},
            {"role": "assistant", "content": ""},
        ),
        keep_spans=(None, (0, 0)),
    )
    manager = TokenizeManager(_FakeTokenizer())
    events = manager._compile_rule_position_events(
        rule,
        raw_messages=list(rule.full_messages),
        owner_ranges={0: [(0, 2)], 1: [(2, 3)], 2: [(3, 4)]},
        provenance=provenance,
        full_input_ids=provenance.input_ids,
        owners=provenance.owners,
        target_offset=0,
        normalized_message_count=2,
    )
    assert events == {1: [(0, 2)]}
