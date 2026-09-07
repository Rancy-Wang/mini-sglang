from __future__ import annotations

import pytest

from minisgl.kernel.text_match import (
    find_ordered_latest,
    find_ordered_latest_reference,
)


@pytest.mark.parametrize("matcher", [find_ordered_latest_reference, find_ordered_latest])
def test_ordered_latest_prefers_rightmost_distinct_messages(matcher) -> None:
    sources = ["same old", "middle", "same newest", "same last"]
    matches = matcher(
        sources,
        ["same", "same"],
        source_keys=[1, 2, 1, 1],
        pattern_keys=[1, 1],
    )
    assert matches == [(2, 0, 4), (3, 0, 4)]


@pytest.mark.parametrize("matcher", [find_ordered_latest_reference, find_ordered_latest])
def test_ordered_latest_handles_utf8_empty_text_and_protocol_keys(matcher) -> None:
    matches = matcher(
        ["旧记录", "", "保留中文后缀"],
        ["", "中文"],
        source_keys=[1, 2, 1],
        pattern_keys=[2, 1],
    )
    assert matches == [(1, 0, 0), (2, 2, 4)]


def test_ordered_latest_rejects_out_of_order_projection() -> None:
    with pytest.raises(ValueError, match="pattern 0"):
        find_ordered_latest_reference(
            ["first", "second"],
            ["second", "first"],
            source_keys=[1, 1],
            pattern_keys=[1, 1],
        )
