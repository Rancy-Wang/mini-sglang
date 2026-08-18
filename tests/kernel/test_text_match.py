from __future__ import annotations

import random

import pytest

from minisgl.kernel.text_match import find_all, find_all_reference


def _naive(text, patterns):
    result = []
    for pattern in patterns:
        spans = []
        start = 0
        while True:
            found = text.find(pattern, start)
            if found < 0:
                break
            spans.append((found, found + len(pattern)))
            start = found + 1
        result.append(spans)
    return result


def test_aho_matcher_counts_overlaps_and_unicode_character_offsets():
    text = "ababa你你aba"
    patterns = ["aba", "bab", "你", "你你"]
    expected = _naive(text, patterns)

    assert find_all_reference(text, patterns) == expected
    assert find_all(text, patterns) == expected


def test_aot_aho_kernel_matches_reference_without_python_fallback():
    text = "aaaa你🙂aaaa"
    patterns = ["a", "aa", "aaa", "你🙂"]
    assert find_all(text, patterns, allow_fallback=False) == find_all_reference(text, patterns)


def test_aho_matcher_randomized_differential():
    rng = random.Random(20260818)
    alphabet = "abc你🙂"
    for _ in range(100):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 80)))
        patterns = [
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 8)))
            for _ in range(rng.randrange(1, 12))
        ]
        assert find_all_reference(text, patterns) == _naive(text, patterns)
        assert find_all(text, patterns) == _naive(text, patterns)


def test_aho_matcher_rejects_empty_patterns_and_output_overflow():
    with pytest.raises(ValueError, match="empty patterns"):
        find_all("abc", [""])
    with pytest.raises(ValueError, match="maximum of 2 matches"):
        find_all_reference("aaaa", ["a"], max_matches=2)
