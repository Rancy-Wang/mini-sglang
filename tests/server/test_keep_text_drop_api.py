from __future__ import annotations

from minisgl.server.api_server import _parse_request_drop_rule


def test_api_uses_full_messages_for_successful_projection() -> None:
    visible = [{"role": "user", "content": "keep"}]
    full = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "please keep this"},
    ]
    wire, effective = _parse_request_drop_rule(
        drop_rule={"type": "keep_text_drop", "full_messages": full},
        legacy_drop_message=None,
        messages=visible,
        radix_drop_key_mode="delta-marker",
    )
    assert effective == full
    assert wire is not None
    assert wire["_keep_spans"] == [None, [7, 11]]


def test_api_force_fallback_is_plain_inference_even_without_drop_mode() -> None:
    visible = [{"role": "user", "content": "visible prompt"}]
    wire, effective = _parse_request_drop_rule(
        drop_rule={
            "type": "keep_text_drop",
            "full_messages": [{"role": "user", "content": "different history"}],
            "force": True,
        },
        legacy_drop_message=None,
        messages=visible,
        radix_drop_key_mode="symbol",
    )
    assert wire is None
    assert effective == visible
