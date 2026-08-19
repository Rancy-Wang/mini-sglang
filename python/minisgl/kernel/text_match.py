from __future__ import annotations

import functools
from collections import deque
from typing import Sequence

import torch

from .utils import load_aot


MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_PATTERNS = 4096
MAX_PATTERN_BYTES = 1024 * 1024
MAX_MATCHES = 1_000_000


@functools.cache
def _load_text_match_module():
    return load_aot("text_match", cpp_files=["text_match.cpp"])


def _validate_inputs(text: str, patterns: Sequence[str], max_matches: int) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(patterns) > MAX_PATTERNS:
        raise ValueError(f"too many text-drop patterns; maximum is {MAX_PATTERNS}")
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > MAX_TEXT_BYTES:
        raise ValueError(f"text is too large for text matching; maximum is {MAX_TEXT_BYTES} bytes")
    total_pattern_bytes = 0
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise TypeError("every pattern must be a string")
        if not pattern:
            raise ValueError("empty patterns are not matchable")
        total_pattern_bytes += len(pattern.encode("utf-8"))
    if total_pattern_bytes > MAX_PATTERN_BYTES:
        raise ValueError(
            "text-drop patterns are too large; "
            f"maximum combined UTF-8 size is {MAX_PATTERN_BYTES} bytes"
        )
    if max_matches <= 0 or max_matches > MAX_MATCHES:
        raise ValueError(f"max_matches must be in [1, {MAX_MATCHES}]")


def _byte_to_char_boundaries(text: str) -> dict[int, int]:
    boundaries = {0: 0}
    byte_pos = 0
    for char_pos, char in enumerate(text, start=1):
        byte_pos += len(char.encode("utf-8"))
        boundaries[byte_pos] = char_pos
    return boundaries


def _decode_byte_matches(
    text: str,
    pattern_count: int,
    matches: Sequence[tuple[int, int, int]],
) -> list[list[tuple[int, int]]]:
    boundaries = _byte_to_char_boundaries(text)
    result: list[list[tuple[int, int]]] = [[] for _ in range(pattern_count)]
    for pattern_id, byte_start, byte_end in matches:
        if byte_start not in boundaries or byte_end not in boundaries:
            raise RuntimeError("text matcher returned a span inside a UTF-8 code point")
        result[pattern_id].append((boundaries[byte_start], boundaries[byte_end]))
    for spans in result:
        spans.sort()
    return result


def find_all_reference(
    text: str,
    patterns: Sequence[str],
    *,
    max_matches: int = MAX_MATCHES,
) -> list[list[tuple[int, int]]]:
    """Aho-Corasick reference implementation with overlapping matches."""

    patterns = list(patterns)
    _validate_inputs(text, patterns, max_matches)
    if not patterns:
        return []

    encoded_patterns = [pattern.encode("utf-8") for pattern in patterns]
    transitions: list[dict[int, int]] = [{}]
    failure = [0]
    output: list[list[int]] = [[]]
    for pattern_id, pattern in enumerate(encoded_patterns):
        node = 0
        for byte in pattern:
            next_node = transitions[node].get(byte)
            if next_node is None:
                next_node = len(transitions)
                transitions[node][byte] = next_node
                transitions.append({})
                failure.append(0)
                output.append([])
            node = next_node
        output[node].append(pattern_id)

    queue: deque[int] = deque(transitions[0].values())
    while queue:
        node = queue.popleft()
        for byte, child in transitions[node].items():
            queue.append(child)
            fallback = failure[node]
            while fallback and byte not in transitions[fallback]:
                fallback = failure[fallback]
            failure[child] = transitions[fallback].get(byte, 0)
            output[child].extend(output[failure[child]])

    raw_matches: list[tuple[int, int, int]] = []
    node = 0
    for byte_end, byte in enumerate(text.encode("utf-8"), start=1):
        while node and byte not in transitions[node]:
            node = failure[node]
        node = transitions[node].get(byte, 0)
        for pattern_id in output[node]:
            byte_start = byte_end - len(encoded_patterns[pattern_id])
            raw_matches.append((pattern_id, byte_start, byte_end))
            if len(raw_matches) > max_matches:
                raise ValueError(f"text matching exceeds the maximum of {max_matches} matches")
    return _decode_byte_matches(text, len(patterns), raw_matches)


def find_all(
    text: str,
    patterns: Sequence[str],
    *,
    max_matches: int = MAX_MATCHES,
    allow_fallback: bool = True,
) -> list[list[tuple[int, int]]]:
    """Find all overlapping UTF-8 pattern occurrences in O(input + patterns + output)."""

    patterns = list(patterns)
    _validate_inputs(text, patterns, max_matches)
    if not patterns:
        return []

    source_bytes = text.encode("utf-8")
    pattern_bytes = [pattern.encode("utf-8") for pattern in patterns]
    flat_patterns = b"".join(pattern_bytes)
    offsets = [0]
    for pattern in pattern_bytes:
        offsets.append(offsets[-1] + len(pattern))

    source = torch.tensor(list(source_bytes), dtype=torch.int32, device="cpu")
    flat = torch.tensor(list(flat_patterns), dtype=torch.int32, device="cpu")
    pattern_offsets = torch.tensor(offsets, dtype=torch.int64, device="cpu")
    capacity = min(max_matches, max(1, len(source_bytes) * len(patterns)))
    output = torch.empty((capacity, 3), dtype=torch.int64, device="cpu")
    try:
        count = int(
            _load_text_match_module().aho_find_all(
                source, flat, pattern_offsets, output, int(capacity)
            )
        )
    except Exception:
        if not allow_fallback:
            raise
        return find_all_reference(text, patterns, max_matches=max_matches)
    if count < 0:
        raise ValueError(f"text matching exceeds the maximum of {max_matches} matches")
    raw = [tuple(map(int, row)) for row in output[:count].tolist()]
    return _decode_byte_matches(text, len(patterns), raw)
