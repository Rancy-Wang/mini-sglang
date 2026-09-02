from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from minisgl.kernel.text_match import find_all, find_ordered_latest


MAX_DROP_MESSAGES = 4096
MAX_SELECTORS_PER_MESSAGE = 1024


def _role(message: Mapping[str, Any]) -> str:
    role = str(message.get("role", "")).lower()
    return "tool" if role == "function" else role


def _content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("text_drop requires every selected message content to be a string or null")
    return content


def _matchable_content(message: Mapping[str, Any], *, field: str) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part_id, part in enumerate(content):
            if not isinstance(part, Mapping):
                raise ValueError(f"{field}.content[{part_id}] must be an object")
            part_type = part.get("type")
            value = part.get("text") if part_type in {"text", "input_text"} else None
            if part_type == "thinking":
                value = part.get("thinking")
            if not isinstance(value, str):
                raise ValueError(
                    f"{field}.content supports only text, input_text, and thinking parts"
                )
            parts.append(value)
        return "".join(parts)
    raise ValueError(f"{field}.content must be a string, text-part list, or null")


def _protocol_fingerprint(message: Mapping[str, Any]) -> str:
    role = _role(message)
    protocol: dict[str, Any] = {"role": role}
    if role == "assistant":
        protocol["reasoning_content"] = message.get("reasoning_content")
        protocol["tool_calls"] = message.get("tool_calls")
    elif role == "tool":
        protocol["name"] = message.get("name")
        protocol["tool_call_id"] = message.get("tool_call_id")
    elif message.get("name") is not None:
        protocol["name"] = message.get("name")
    return json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be a positive integer")
    if isinstance(value, str) and value.strip() != str(result):
        raise ValueError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    if value == 0 or value == "0":
        return 0
    return _positive_int(value, field=field)


@dataclass(frozen=True)
class _TextSelection:
    segments: tuple[str, ...]
    occurrences: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    whole_message: bool


@dataclass(frozen=True)
class TokenDropEvents:
    """Token-level Drop events shared by tokenizer and Radix compilation."""

    event_insert_offsets: torch.Tensor
    range_offsets: torch.Tensor
    raw_ranges: torch.Tensor
    full_token_visible_until: torch.Tensor
    effective_event_count: int
    effective_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DropCompileContext:
    """Tokenizer services needed by every DropRule implementation."""

    raw_messages: Sequence[Mapping[str, Any]]
    owner_ranges: Mapping[int, Sequence[tuple[int, int]]]
    provenance: Any | None
    full_input_ids: Sequence[int]
    owners: Sequence[int]
    target_offset: int
    normalized_message_count: int
    is_gpt_oss: bool
    harmony_thinking_ranges: Mapping[int, Sequence[tuple[int, int]]]
    normalize_content: Callable[[Any], str]
    rendered_source_start: Callable[..., int]
    token_ranges_for_char_spans: Callable[..., list[tuple[int, int]]]
    canonicalize_ranges: Callable[[Sequence[tuple[int, int]]], list[tuple[int, int]]]
    position_ranges_from_ids: Callable[[list[int]], list[tuple[int, int]]]
    find_owned_subsequence: Callable[..., tuple[int, int]]
    encode_text: Callable[[str], list[int]]
    compile_events: Callable[[dict[int, list[tuple[int, int]]]], TokenDropEvents]


@dataclass(frozen=True)
class MessageDropRule:
    """Drop complete chat-template message ownership ranges at named triggers."""

    drop_messages: dict[int, tuple[int, ...]]
    type: str = "message_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> MessageDropRule:
        raw = payload.get("drop_messages")
        if not isinstance(raw, Mapping):
            raise ValueError("message_drop.drop_messages must be an object of trigger-to-ID lists")
        normalized: dict[int, tuple[int, ...]] = {}
        unknown = set(payload) - {"type", "drop_messages"}
        if unknown:
            raise ValueError(f"message_drop has unsupported fields: {sorted(unknown)}")
        for raw_trigger, raw_ids in raw.items():
            trigger = _nonnegative_int(raw_trigger, field="message_drop trigger")
            if trigger >= (1 << 63):
                raise ValueError("message_drop trigger is outside the signed int64 range")
            if not isinstance(raw_ids, list):
                raise ValueError(f"message_drop.drop_messages[{trigger}] must be a list")
            ids: list[int] = []
            for raw_id in raw_ids:
                message_id = _nonnegative_int(raw_id, field="message_drop message ID")
                if message_id >= (1 << 63):
                    raise ValueError("message_drop message ID is outside the signed int64 range")
                if message_id > trigger:
                    raise ValueError(
                        f"message_drop event {trigger} cannot drop future message {message_id}"
                    )
                if trigger < len(messages) and message_id >= len(messages):
                    raise ValueError(
                        f"message_drop message ID {message_id} is outside the current message range"
                    )
                ids.append(message_id)
            normalized[trigger] = tuple(ids)
        return cls(normalized)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "drop_messages": {
                str(trigger): list(message_ids)
                for trigger, message_ids in self.drop_messages.items()
            },
        }

    def position_events(self, context: DropCompileContext) -> dict[int, list[tuple[int, int]]]:
        events: dict[int, list[tuple[int, int]]] = {}
        effective: set[int] = set()
        for raw_trigger in sorted(self.drop_messages):
            trigger = raw_trigger + context.target_offset
            if trigger >= context.normalized_message_count:
                continue
            shifted = {
                message_id + context.target_offset for message_id in self.drop_messages[raw_trigger]
            }
            newly_effective = shifted - effective
            effective.update(shifted)
            ranges = [
                item
                for message_id in sorted(newly_effective)
                for item in context.owner_ranges.get(message_id, ())
            ]
            if ranges:
                events[trigger] = ranges
        return events

    def compile_token_drop_events(self, context: DropCompileContext) -> TokenDropEvents:
        return context.compile_events(self.position_events(context))


@dataclass(frozen=True)
class TextDropRule:
    """Drop user-selected raw-content substrings after the latest user Prefill."""

    drop_messages: tuple[dict[str, Any], ...]
    selections: tuple[_TextSelection | None, ...]
    trigger_message_id: int
    type: str = "text_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> TextDropRule:
        unknown_payload = set(payload) - {"type", "drop_messages", "_trigger_message_id"}
        if unknown_payload:
            raise ValueError(f"text_drop has unsupported fields: {sorted(unknown_payload)}")
        raw_entries = payload.get("drop_messages")
        if not isinstance(raw_entries, list):
            raise ValueError("text_drop.drop_messages must be a list aligned with messages")
        if len(raw_entries) != len(messages):
            raise ValueError(
                "text_drop.drop_messages must have exactly the same length as messages"
            )
        if len(raw_entries) > MAX_DROP_MESSAGES:
            raise ValueError(f"text_drop supports at most {MAX_DROP_MESSAGES} messages")

        latest_user = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if _role(messages[index]) == "user"
            ),
            None,
        )
        internal_trigger = payload.get("_trigger_message_id")
        if internal_trigger is not None:
            latest_user = int(internal_trigger)
        if latest_user is None:
            raise ValueError("text_drop requires at least one user message")

        normalized_entries: list[dict[str, Any]] = []
        selections: list[_TextSelection | None] = []
        for index, (entry, message) in enumerate(zip(raw_entries, messages, strict=True)):
            if not isinstance(entry, Mapping):
                raise ValueError(f"text_drop.drop_messages[{index}] must be an object")
            unknown = set(entry) - {"role", "content", "occurrence"}
            if unknown:
                raise ValueError(
                    f"text_drop.drop_messages[{index}] has unsupported fields: {sorted(unknown)}"
                )
            role = str(entry.get("role", "")).lower()
            expected_role = str(message.get("role", "")).lower()
            if role != expected_role:
                raise ValueError(
                    f"text_drop.drop_messages[{index}].role must match messages[{index}].role"
                )
            if "occurence" in entry:
                raise ValueError("use the spelling 'occurrence', not 'occurence'")
            raw_selector = entry.get("content")
            occurrence_supplied = "occurrence" in entry
            raw_occurrence = entry.get("occurrence")

            if raw_selector is None:
                segments: list[str] = []
            elif isinstance(raw_selector, str):
                segments = [] if raw_selector == "" else [raw_selector]
            elif isinstance(raw_selector, list) and all(
                isinstance(item, str) for item in raw_selector
            ):
                if len(raw_selector) > MAX_SELECTORS_PER_MESSAGE:
                    raise ValueError(
                        f"text_drop.drop_messages[{index}] has too many content selectors"
                    )
                empty = [item == "" for item in raw_selector]
                if any(empty) and not all(empty):
                    raise ValueError(
                        f"text_drop.drop_messages[{index}].content cannot mix empty "
                        "and non-empty strings"
                    )
                segments = [] if all(empty) else list(raw_selector)
            else:
                raise ValueError(
                    f"text_drop.drop_messages[{index}].content must be null, a string, or list[str]"
                )

            if occurrence_supplied and not segments:
                raise ValueError(
                    f"text_drop.drop_messages[{index}].occurrence is invalid without content"
                )
            if not segments:
                normalized_entries.append({"role": expected_role, "content": raw_selector})
                selections.append(None)
                continue

            if occurrence_supplied:
                if isinstance(raw_selector, str):
                    if isinstance(raw_occurrence, list):
                        raise ValueError(
                            f"text_drop.drop_messages[{index}].occurrence must be one "
                            "positive integer"
                        )
                    occurrences = [
                        _positive_int(raw_occurrence, field=f"drop_messages[{index}].occurrence")
                    ]
                else:
                    if not isinstance(raw_occurrence, list) or len(raw_occurrence) != len(segments):
                        raise ValueError(
                            f"text_drop.drop_messages[{index}].occurrence must provide "
                            "one value per content segment"
                        )
                    occurrences = [
                        _positive_int(value, field=f"drop_messages[{index}].occurrence[{part}]")
                        for part, value in enumerate(raw_occurrence)
                    ]
            else:
                occurrences = [1] * len(segments)

            source = _content(message)
            matches = find_all(source, segments)
            spans: list[tuple[int, int]] = []
            for part, (segment, occurrence, candidates) in enumerate(
                zip(segments, occurrences, matches, strict=True)
            ):
                if occurrence > len(candidates):
                    raise ValueError(
                        "text_drop selector is not the requested occurrence of a substring of "
                        f"messages[{index}].content (selector {part}, occurrence {occurrence})"
                    )
                spans.append(candidates[occurrence - 1])
            covered = _merge_ranges(spans)
            whole_message = bool(source) and covered == [(0, len(source))]
            normalized = {"role": expected_role, "content": raw_selector}
            if occurrence_supplied:
                normalized["occurrence"] = raw_occurrence
            normalized_entries.append(normalized)
            selections.append(
                _TextSelection(tuple(segments), tuple(occurrences), tuple(spans), whole_message)
            )
        return cls(tuple(normalized_entries), tuple(selections), latest_user)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "drop_messages": [dict(entry) for entry in self.drop_messages],
            "_trigger_message_id": self.trigger_message_id,
        }

    def position_events(self, context: DropCompileContext) -> dict[int, list[tuple[int, int]]]:
        trigger = self.trigger_message_id + context.target_offset
        if trigger >= context.normalized_message_count:
            return {}
        ranges: list[tuple[int, int]] = []
        for raw_message_id, selection in enumerate(self.selections):
            if selection is None:
                continue
            owner = raw_message_id + context.target_offset
            if selection.whole_message:
                ranges.extend(context.owner_ranges.get(owner, ()))
                continue
            if context.provenance is None:
                raise ValueError(
                    "Partial text_drop requires a fast tokenizer with canonical offset mapping"
                )
            source = context.normalize_content(context.raw_messages[raw_message_id].get("content"))
            rendered_start = context.rendered_source_start(
                context.provenance,
                owner=owner,
                source=source,
                field="content",
            )
            rendered_spans = [
                (rendered_start + start, rendered_start + end) for start, end in selection.spans
            ]
            ranges.extend(
                context.token_ranges_for_char_spans(
                    context.provenance,
                    owner=owner,
                    spans=rendered_spans,
                    field="text_drop content",
                    allow_empty=True,
                )
            )
        canonical = context.canonicalize_ranges(ranges)
        return {trigger: canonical} if canonical else {}

    def compile_token_drop_events(self, context: DropCompileContext) -> TokenDropEvents:
        return context.compile_events(self.position_events(context))


@dataclass(frozen=True)
class KeepTextDropRule:
    """Keep an ordered visible-text projection over a complete Radix history."""

    full_messages: tuple[dict[str, Any], ...]
    keep_spans: tuple[tuple[int, int] | None, ...]
    force: bool = False
    use_visible_as_full: bool = False
    fallback_reason: str | None = None
    type: str = "keep_text_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        *,
        allow_internal: bool = False,
    ) -> KeepTextDropRule:
        internal = "_keep_spans" in payload
        allowed = (
            {"type", "force", "_keep_spans"}
            if internal and allow_internal
            else {"type", "full_messages", "force"}
        )
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"keep_text_drop has unsupported fields: {sorted(unknown)}")

        force = payload.get("force", False)
        if not isinstance(force, bool):
            raise ValueError("keep_text_drop.force must be a boolean")
        if internal:
            if not allow_internal:
                raise ValueError("drop_rule contains a reserved internal field")
            raw_full: Sequence[Mapping[str, Any]] = messages
        else:
            public_full = payload.get("full_messages")
            if not isinstance(public_full, list):
                raise ValueError("keep_text_drop.full_messages must be a list")
            if not all(isinstance(message, Mapping) for message in public_full):
                raise ValueError("every keep_text_drop.full_messages entry must be an object")
            raw_full = public_full
        if not raw_full:
            raise ValueError("keep_text_drop.full_messages must not be empty")
        if len(raw_full) > MAX_DROP_MESSAGES:
            raise ValueError(f"keep_text_drop supports at most {MAX_DROP_MESSAGES} full_messages")
        full_messages = tuple(dict(message) for message in raw_full)
        for message_id, message in enumerate(full_messages):
            role = _role(message)
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"keep_text_drop.full_messages[{message_id}].role is invalid")
            _matchable_content(message, field=f"keep_text_drop.full_messages[{message_id}]")

        if internal:
            raw_spans = payload.get("_keep_spans")
            if not isinstance(raw_spans, list) or len(raw_spans) != len(full_messages):
                raise ValueError(
                    "internal keep_text_drop._keep_spans must align with full_messages"
                )
            keep_spans: list[tuple[int, int] | None] = []
            for message_id, (raw_span, message) in enumerate(
                zip(raw_spans, full_messages, strict=True)
            ):
                if raw_span is None:
                    keep_spans.append(None)
                    continue
                if (
                    not isinstance(raw_span, list)
                    or len(raw_span) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) for value in raw_span
                    )
                ):
                    raise ValueError(
                        f"internal keep_text_drop._keep_spans[{message_id}] is invalid"
                    )
                start, end = raw_span
                source = _matchable_content(
                    message, field=f"keep_text_drop.full_messages[{message_id}]"
                )
                if start < 0 or end < start or end > len(source):
                    raise ValueError(
                        f"internal keep_text_drop._keep_spans[{message_id}] is out of range"
                    )
                keep_spans.append((start, end))
            return cls(full_messages, tuple(keep_spans), force=force)

        if not messages:
            raise ValueError("keep_text_drop requires at least one visible message")
        if len(messages) > len(full_messages):
            reason = "visible messages cannot outnumber keep_text_drop.full_messages"
            if force:
                return cls(
                    tuple(dict(message) for message in messages),
                    tuple(),
                    force=True,
                    use_visible_as_full=True,
                    fallback_reason=reason,
                )
            raise ValueError(reason)

        full_contents = [
            _matchable_content(message, field=f"keep_text_drop.full_messages[{message_id}]")
            for message_id, message in enumerate(full_messages)
        ]
        visible_contents = [
            _matchable_content(message, field=f"messages[{message_id}]")
            for message_id, message in enumerate(messages)
        ]
        fingerprints = [
            *(_protocol_fingerprint(message) for message in full_messages),
            *(_protocol_fingerprint(message) for message in messages),
        ]
        key_ids = {value: key_id for key_id, value in enumerate(dict.fromkeys(fingerprints))}
        full_keys = [key_ids[_protocol_fingerprint(message)] for message in full_messages]
        visible_keys = [key_ids[_protocol_fingerprint(message)] for message in messages]
        try:
            matches = find_ordered_latest(
                full_contents,
                visible_contents,
                source_keys=full_keys,
                pattern_keys=visible_keys,
            )
        except ValueError as exc:
            reason = str(exc)
            if force:
                return cls(
                    tuple(dict(message) for message in messages),
                    tuple(),
                    force=True,
                    use_visible_as_full=True,
                    fallback_reason=reason,
                )
            raise ValueError(f"keep_text_drop projection failed: {reason}") from exc

        keep_spans: list[tuple[int, int] | None] = [None] * len(full_messages)
        for source_id, start, end in matches:
            keep_spans[source_id] = (start, end)
        return cls(full_messages, tuple(keep_spans), force=force)

    def to_wire(self) -> dict[str, Any]:
        if self.use_visible_as_full:
            raise RuntimeError("forced keep_text_drop fallback has no Drop wire payload")
        return {
            "type": self.type,
            "force": self.force,
            "_keep_spans": [list(span) if span is not None else None for span in self.keep_spans],
        }

    def has_drop(self) -> bool:
        return any(
            span is None
            or span
            != (
                0,
                len(
                    _matchable_content(message, field=f"keep_text_drop.full_messages[{message_id}]")
                ),
            )
            for message_id, (message, span) in enumerate(
                zip(self.full_messages, self.keep_spans, strict=True)
            )
        )

    def position_events(self, context: DropCompileContext) -> dict[int, list[tuple[int, int]]]:
        if context.provenance is None:
            raise ValueError("keep_text_drop requires exact chat-template token provenance")
        ranges: list[tuple[int, int]] = []
        for raw_message_id, keep_span in enumerate(self.keep_spans):
            owner = raw_message_id + context.target_offset
            if keep_span is None:
                ranges.extend(context.owner_ranges.get(owner, ()))
                continue

            source = context.normalize_content(context.raw_messages[raw_message_id].get("content"))
            if not source:
                continue
            rendered_start = context.rendered_source_start(
                context.provenance,
                owner=owner,
                source=source,
                field="keep_text_drop content",
                prefer_latest=True,
            )
            full_span = (rendered_start, rendered_start + len(source))
            selected_span = (
                rendered_start + keep_span[0],
                rendered_start + keep_span[1],
            )
            content_ranges = context.token_ranges_for_char_spans(
                context.provenance,
                owner=owner,
                spans=[full_span],
                field="keep_text_drop full content",
                boundary_mode="contained",
                allow_empty=True,
            )
            kept_ranges = context.token_ranges_for_char_spans(
                context.provenance,
                owner=owner,
                spans=[selected_span],
                field="keep_text_drop selected content",
                boundary_mode="overlap",
                allow_empty=keep_span[0] == keep_span[1],
            )
            content_ids = {
                token_id for start, end in content_ranges for token_id in range(start, end)
            }
            kept_ids = {token_id for start, end in kept_ranges for token_id in range(start, end)}
            ranges.extend(context.position_ranges_from_ids(list(content_ids - kept_ids)))

        trigger = max(context.normalized_message_count - 1, 0)
        canonical = context.canonicalize_ranges(ranges)
        return {trigger: canonical} if canonical else {}

    def compile_token_drop_events(self, context: DropCompileContext) -> TokenDropEvents:
        return context.compile_events(self.position_events(context))


@dataclass(frozen=True)
class ThinkingDropRule:
    """Retain structured assistant thinking in the full stream, then drop its KV."""

    thinking_by_message: dict[int, str]
    type: str = "thinking_drop"

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> ThinkingDropRule:
        allowed = {"type"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"thinking_drop has unsupported fields: {sorted(unknown)}")
        thinking: dict[int, str] = {}
        for index, message in enumerate(messages):
            if _role(message) != "assistant":
                continue
            structured = message.get("reasoning_content")
            if structured is not None and not isinstance(structured, str):
                raise ValueError(f"messages[{index}].reasoning_content must be a string or null")
            structured = structured or None
            inline = _extract_leading_think(_content(message), message_id=index)
            if structured is not None and inline is not None:
                raise ValueError(
                    f"messages[{index}] cannot provide both reasoning_content and "
                    "a leading <think> block"
                )
            source = structured if structured is not None else inline
            if source:
                thinking[index] = source
        if not thinking:
            raise ValueError(
                "thinking_drop requires at least one assistant reasoning_content or "
                "leading <think> block"
            )
        return cls(thinking)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type}

    def position_events(self, context: DropCompileContext) -> dict[int, list[tuple[int, int]]]:
        events: dict[int, list[tuple[int, int]]] = {}
        for raw_message_id, source in self.thinking_by_message.items():
            owner = raw_message_id + context.target_offset
            if owner >= context.normalized_message_count:
                continue
            if context.provenance is not None:
                rendered_start = context.rendered_source_start(
                    context.provenance,
                    owner=owner,
                    source=source,
                    field="thinking",
                )
                ranges = context.token_ranges_for_char_spans(
                    context.provenance,
                    owner=owner,
                    spans=[(rendered_start, rendered_start + len(source))],
                    field="thinking",
                )
            elif context.is_gpt_oss:
                ranges = list(context.harmony_thinking_ranges.get(raw_message_id, ()))
                if not ranges:
                    raise ValueError(
                        f"Cannot map thinking for messages[{raw_message_id}] into "
                        "the retained Harmony analysis component"
                    )
            else:
                ranges = [
                    context.find_owned_subsequence(
                        context.full_input_ids,
                        context.owners,
                        context.encode_text(source),
                        owner=owner,
                        field="thinking",
                    )
                ]
            events[owner] = context.canonicalize_ranges(ranges)
        return events

    def compile_token_drop_events(self, context: DropCompileContext) -> TokenDropEvents:
        return context.compile_events(self.position_events(context))


DropRule = MessageDropRule | TextDropRule | KeepTextDropRule | ThinkingDropRule


def parse_drop_rule(
    payload: Mapping[str, Any] | None,
    messages: Sequence[Mapping[str, Any]],
    *,
    legacy_drop_message: Mapping[Any, Any] | None = None,
    allow_internal: bool = False,
) -> DropRule | None:
    if payload is not None and legacy_drop_message is not None:
        raise ValueError("drop_rule and legacy drop_message cannot be supplied together")
    if payload is None:
        if legacy_drop_message is None:
            return None
        payload = {"type": "message_drop", "drop_messages": legacy_drop_message}
    if not isinstance(payload, Mapping):
        raise ValueError("drop_rule must be an object")
    if any(str(field).startswith("_") for field in payload) and not allow_internal:
        raise ValueError("drop_rule contains a reserved internal field")
    rule_type = payload.get("type")
    if rule_type == "message_drop":
        return MessageDropRule.from_payload(payload, messages)
    if rule_type == "text_drop":
        return TextDropRule.from_payload(payload, messages)
    if rule_type == "keep_text_drop":
        return KeepTextDropRule.from_payload(payload, messages, allow_internal=allow_internal)
    if rule_type == "thinking_drop":
        return ThinkingDropRule.from_payload(payload, messages)
    raise ValueError(
        "drop_rule.type must be one of: message_drop, text_drop, " "keep_text_drop, thinking_drop"
    )


def project_drop_rule_for_prefix(
    payload: Mapping[str, Any] | None,
    prefix_len: int,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    rule_type = payload.get("type")
    if rule_type == "message_drop":
        return {
            "type": rule_type,
            "drop_messages": {
                str(trigger): ids
                for trigger, ids in payload.get("drop_messages", {}).items()
                if int(trigger) < prefix_len
            },
        }
    if rule_type == "text_drop":
        trigger = int(payload.get("_trigger_message_id", prefix_len))
        return {
            "type": rule_type,
            "drop_messages": list(payload.get("drop_messages", []))[:prefix_len],
            "_trigger_message_id": trigger,
        }
    if rule_type == "keep_text_drop":
        return {
            "type": rule_type,
            "force": bool(payload.get("force", False)),
            "_keep_spans": list(payload.get("_keep_spans", []))[:prefix_len],
        }
    if rule_type == "thinking_drop":
        return {"type": rule_type}
    raise ValueError(f"unsupported drop rule type: {rule_type!r}")


def _extract_leading_think(content: str, *, message_id: int) -> str | None:
    has_tag = "<think>" in content or "</think>" in content
    if not has_tag:
        return None
    if not content.startswith("<think>"):
        raise ValueError(f"messages[{message_id}] has a non-leading or malformed <think> block")
    close = content.find("</think>", len("<think>"))
    if close < 0:
        raise ValueError(f"messages[{message_id}] has an unclosed <think> block")
    reasoning = content[len("<think>") : close]
    remainder = content[close + len("</think>") :]
    if (
        "<think>" in reasoning
        or "</think>" in reasoning
        or "<think>" in remainder
        or "</think>" in remainder
    ):
        raise ValueError(f"messages[{message_id}] has nested or multiple <think> blocks")
    return reasoning


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
