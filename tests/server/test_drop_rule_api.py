from __future__ import annotations

import pytest
from fastapi import HTTPException

from minisgl.server.api_server import _parse_request_drop_rule


MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer", "reasoning_content": "private"},
    {"role": "user", "content": "next"},
]


def _parse(drop_rule=None, legacy=None):
    return _parse_request_drop_rule(
        drop_rule=drop_rule,
        legacy_drop_message=legacy,
        messages=MESSAGES,
        radix_drop_key_mode="delta-marker",
    )


def test_legacy_drop_message_converts_to_message_drop_wire():
    assert _parse(legacy={3: [1, 2]}) == {
        "type": "message_drop",
        "drop_messages": {"3": [1, 2]},
    }


def test_legacy_and_new_drop_interfaces_are_mutually_exclusive():
    with pytest.raises(HTTPException) as error:
        _parse(
            drop_rule={"type": "message_drop", "drop_messages": {"3": [1]}},
            legacy={3: [1]},
        )
    assert error.value.status_code == 400


def test_text_drop_wire_is_order_aligned_and_server_sets_private_trigger():
    wire = _parse(
        drop_rule={
            "type": "text_drop",
            "drop_messages": [
                {"role": "system", "content": None},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": ["ans", "wer"]},
                {"role": "user", "content": ""},
            ],
        }
    )
    assert wire["type"] == "text_drop"
    assert wire["_trigger_message_id"] == 3
    assert len(wire["drop_messages"]) == len(MESSAGES)


def test_request_cannot_spoof_internal_text_trigger():
    with pytest.raises(HTTPException, match="reserved internal field"):
        _parse(
            drop_rule={
                "type": "text_drop",
                "_trigger_message_id": 1,
                "drop_messages": [
                    {"role": message["role"], "content": None} for message in MESSAGES
                ],
            }
        )


def test_all_drop_rules_require_delta_marker_radix_mode():
    with pytest.raises(HTTPException, match="delta-marker"):
        _parse_request_drop_rule(
            drop_rule={"type": "thinking_drop"},
            legacy_drop_message=None,
            messages=MESSAGES,
            radix_drop_key_mode="symbol",
        )
