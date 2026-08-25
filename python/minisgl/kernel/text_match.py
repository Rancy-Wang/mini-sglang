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


def _validate_ordered_inputs(
    sources: Sequence[str],
    patterns: Sequence[str],
    source_keys: Sequence[int],
    pattern_keys: Sequence[int],
) -> tuple[list[bytes], list[bytes]]:
    if len(sources) != len(source_keys):
        raise ValueError("source_keys must provide one key per source")
    if len(patterns) != len(pattern_keys):
        raise ValueError("pattern_keys must provide one key per pattern")
    if len(sources) > MAX_PATTERNS or len(patterns) > MAX_PATTERNS:
        raise ValueError(f"ordered text matching supports at most {MAX_PATTERNS} entries")
    if not all(isinstance(source, str) for source in sources):
        raise TypeError("every source must be a string")
    if not all(isinstance(pattern, str) for pattern in patterns):
        raise TypeError("every pattern must be a string")
    encoded_sources = [source.encode("utf-8") for source in sources]
    encoded_patterns = [pattern.encode("utf-8") for pattern in patterns]
    if sum(map(len, encoded_sources)) > MAX_TEXT_BYTES:
        raise ValueError(f"ordered text sources are too large; maximum is {MAX_TEXT_BYTES} bytes")
    if sum(map(len, encoded_patterns)) > MAX_PATTERN_BYTES:
        raise ValueError(
            "ordered text patterns are too large; "
            f"maximum combined UTF-8 size is {MAX_PATTERN_BYTES} bytes"
        )
    if not all(isinstance(key, int) and not isinstance(key, bool) for key in source_keys):
        raise TypeError("every source key must be an integer")
    if not all(isinstance(key, int) and not isinstance(key, bool) for key in pattern_keys):
        raise TypeError("every pattern key must be an integer")
    return encoded_sources, encoded_patterns


def _bytes_to_int_tensor(value: bytes) -> torch.Tensor:
    if not value:
        return torch.empty(0, dtype=torch.int32, device="cpu")
    return torch.frombuffer(bytearray(value), dtype=torch.uint8).to(dtype=torch.int32)


def find_ordered_latest_reference(
    sources: Sequence[str],
    patterns: Sequence[str],
    *,
    source_keys: Sequence[int],
    pattern_keys: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Match ordered patterns to distinct sources, preferring the latest valid spans."""

    sources = list(sources)
    patterns = list(patterns)
    source_keys = list(source_keys)
    pattern_keys = list(pattern_keys)
    _validate_ordered_inputs(sources, patterns, source_keys, pattern_keys)

    result: list[tuple[int, int, int] | None] = [None] * len(patterns)
    source_id = len(sources) - 1
    for pattern_id in range(len(patterns) - 1, -1, -1):
        pattern = patterns[pattern_id]
        while source_id >= 0:
            if source_keys[source_id] == pattern_keys[pattern_id]:
                start = sources[source_id].rfind(pattern)
                if start >= 0:
                    result[pattern_id] = (source_id, start, start + len(pattern))
                    source_id -= 1
                    break
            source_id -= 1
        if result[pattern_id] is None:
            raise ValueError(
                f"ordered text pattern {pattern_id} has no compatible match in full_messages"
            )
    return [match for match in result if match is not None]


def find_ordered_latest(
    sources: Sequence[str],
    patterns: Sequence[str],
    *,
    source_keys: Sequence[int],
    pattern_keys: Sequence[int],
    allow_fallback: bool = True,
) -> list[tuple[int, int, int]]:
    """Linear right-to-left ordered matching with a CPU AOT KMP implementation."""

    sources = list(sources)
    patterns = list(patterns)
    source_keys = list(source_keys)
    pattern_keys = list(pattern_keys)
    source_bytes, pattern_bytes = _validate_ordered_inputs(
        sources, patterns, source_keys, pattern_keys
    )
    if not patterns:
        return []

    flat_sources = b"".join(source_bytes)
    flat_patterns = b"".join(pattern_bytes)
    source_offsets = [0]
    pattern_offsets = [0]
    for source in source_bytes:
        source_offsets.append(source_offsets[-1] + len(source))
    for pattern in pattern_bytes:
        pattern_offsets.append(pattern_offsets[-1] + len(pattern))

    source = _bytes_to_int_tensor(flat_sources)
    pattern = _bytes_to_int_tensor(flat_patterns)
    source_offset_tensor = torch.tensor(source_offsets, dtype=torch.int64, device="cpu")
    pattern_offset_tensor = torch.tensor(pattern_offsets, dtype=torch.int64, device="cpu")
    source_key_tensor = torch.tensor(source_keys, dtype=torch.int64, device="cpu")
    pattern_key_tensor = torch.tensor(pattern_keys, dtype=torch.int64, device="cpu")
    output = torch.empty((len(patterns), 3), dtype=torch.int64, device="cpu")
    try:
        missing = int(
            _load_text_match_module().ordered_latest_find(
                source,
                source_offset_tensor,
                source_key_tensor,
                pattern,
                pattern_offset_tensor,
                pattern_key_tensor,
                output,
            )
        )
    except Exception:
        if not allow_fallback:
            raise
        return find_ordered_latest_reference(
            sources,
            patterns,
            source_keys=source_keys,
            pattern_keys=pattern_keys,
        )
    if missing >= 0:
        raise ValueError(f"ordered text pattern {missing} has no compatible match in full_messages")

    byte_boundaries: dict[int, dict[int, int]] = {}
    result: list[tuple[int, int, int]] = []
    for source_id, byte_start, byte_end in output.tolist():
        source_id = int(source_id)
        byte_start = int(byte_start)
        byte_end = int(byte_end)
        boundaries = byte_boundaries.setdefault(
            source_id, _byte_to_char_boundaries(sources[source_id])
        )
        if byte_start not in boundaries or byte_end not in boundaries:
            raise RuntimeError("ordered text matcher returned a span inside a UTF-8 code point")
        result.append((source_id, boundaries[byte_start], boundaries[byte_end]))
    return result
