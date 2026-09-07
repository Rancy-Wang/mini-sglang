from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

from pydantic import BaseModel, Field, model_validator

MESSAGE_SHAPES: Tuple[str, ...] = (
    "reasoning",
    "content",
    "toolcall",
    "reasoning+content",
    "reasoning+toolcall",
    "content+toolcall",
    "reasoning+content+toolcall",
)

CorrectnessMethod = Literal["full", "summary"]
WorkloadMethod = Literal["full", "drop_kv", "summary", "summary_drop_kv"]


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def classify_assistant_message(message: Dict[str, Any]) -> str | None:
    """Classify one assistant message by its non-empty public fields."""

    if message.get("role") != "assistant":
        return None
    parts: List[str] = []
    if _nonempty_text(message.get("reasoning_content")):
        parts.append("reasoning")
    if _nonempty_text(message.get("content")):
        parts.append("content")
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        parts.append("toolcall")
    return "+".join(parts) or None


def request_hash(request: Dict[str, Any]) -> str:
    canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MatchConfig(BaseModel):
    mode: Literal["exact", "prefix", "keywords"] = "exact"
    prefix_chars: int | None = None
    keywords: List[str] = Field(default_factory=list)
    compare_reasoning: bool = True
    compare_tool_calls: bool = True

    @model_validator(mode="after")
    def validate_mode_options(self) -> MatchConfig:
        if self.mode == "prefix" and (self.prefix_chars is None or self.prefix_chars <= 0):
            raise ValueError("prefix mode requires a positive prefix_chars value.")
        if self.mode == "keywords" and not self.keywords:
            raise ValueError("keywords mode requires at least one keyword.")
        return self


class CaseMetadata(BaseModel):
    source: str
    method: WorkloadMethod
    model: str
    summary_triggered: bool | None = None
    target_message_index: int | None = None
    tags: Dict[str, Any] = Field(default_factory=dict)


class OracleResult(BaseModel):
    message: Dict[str, Any]
    finish_reason: str | None = None


class CaptureRecord(BaseModel):
    capture_id: str
    captured_at_ns: int
    request: Dict[str, Any]
    request_sha256: str

    @model_validator(mode="after")
    def validate_hash(self) -> CaptureRecord:
        actual = request_hash(self.request)
        if self.request_sha256 != actual:
            raise ValueError(
                f"Capture {self.capture_id!r} request hash mismatch: "
                f"expected {self.request_sha256}, got {actual}."
            )
        return self


class TrajectoryTask(BaseModel):
    """One full-history task whose captured requests are ordered by turn."""

    task_id: str
    source_path: str
    turns: List[CaptureRecord]


class ManifestCase(BaseModel):
    case_id: str
    request: Dict[str, Any]
    request_sha256: str
    metadata: CaseMetadata
    matcher: MatchConfig = Field(default_factory=MatchConfig)
    oracle: OracleResult | None = None

    @model_validator(mode="after")
    def validate_case(self) -> ManifestCase:
        actual = request_hash(self.request)
        if self.request_sha256 != actual:
            raise ValueError(
                f"Case {self.case_id!r} request hash mismatch: "
                f"expected {self.request_sha256}, got {actual}."
            )
        messages = self.request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Case {self.case_id!r} requires a non-empty messages list.")
        index = self.metadata.target_message_index
        if index is not None:
            if not -len(messages) <= index < len(messages):
                raise ValueError(f"Case {self.case_id!r} target_message_index is out of range.")
            if classify_assistant_message(messages[index]) is None:
                raise ValueError(
                    f"Case {self.case_id!r} target_message_index must select a non-empty "
                    "assistant message."
                )
        return self

    def detected_shapes(self) -> Tuple[str, ...]:
        messages = self.request["messages"]
        index = self.metadata.target_message_index
        if index is not None:
            shape = classify_assistant_message(messages[index])
            assert shape is not None
            return (shape,)
        found = {
            shape
            for message in messages
            if (shape := classify_assistant_message(message)) is not None
        }
        return tuple(shape for shape in MESSAGE_SHAPES if shape in found)

    def ensure_correctness_scope(self) -> None:
        if self.metadata.method not in {"full", "summary"}:
            raise ValueError(
                f"Case {self.case_id!r} uses {self.metadata.method!r}; "
                "DropKV correctness is deferred."
            )
        has_drop_payload = (
            self.request.get("drop_rule") is not None
            or self.request.get("drop_message") is not None
        )
        if has_drop_payload:
            raise ValueError(
                f"Correctness case {self.case_id!r} contains a Drop payload; "
                "DropKV correctness is deferred."
            )
        if self.metadata.method == "summary" and self.metadata.summary_triggered is not True:
            raise ValueError(
                f"Summary correctness case {self.case_id!r} must record summary_triggered=true."
            )

    def has_drop_payload(self) -> bool:
        return (
            self.request.get("drop_rule") is not None
            or self.request.get("drop_message") is not None
        )

    def ensure_performance_scope(self) -> None:
        if self.metadata.summary_triggered is None:
            raise ValueError(
                f"Performance case {self.case_id!r} must explicitly record whether Summary "
                "was triggered."
            )
        expected_summary = self.metadata.method in {"summary", "summary_drop_kv"}
        if self.metadata.summary_triggered is not expected_summary:
            raise ValueError(
                f"Performance case {self.case_id!r} method={self.metadata.method!r} conflicts "
                f"with summary_triggered={self.metadata.summary_triggered!r}."
            )
        expected_drop = self.metadata.method in {"drop_kv", "summary_drop_kv"}
        if self.has_drop_payload() is not expected_drop:
            raise ValueError(
                f"Performance case {self.case_id!r} method={self.metadata.method!r} conflicts "
                f"with drop_requested={self.has_drop_payload()!r}."
            )


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            records.append(value)
    return records


def load_capture_records(path: str | Path) -> List[CaptureRecord]:
    return [CaptureRecord.model_validate(record) for record in _load_jsonl(path)]


def load_full_trajectories(
    directory: str | Path,
    *,
    max_turns: int,
    min_tasks: int,
) -> List[TrajectoryTask]:
    """Load one full-history trajectory per JSONL file under ``directory``."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Trajectory directory does not exist or is not a directory: {root}")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive.")
    if min_tasks <= 0:
        raise ValueError("min_tasks must be positive.")

    paths = sorted(root.rglob("*.jsonl"), key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"Trajectory directory contains no JSONL files: {root}")

    tasks: List[TrajectoryTask] = []
    for path in paths:
        relative = path.relative_to(root)
        records = load_capture_records(path)
        if len(records) < max_turns:
            raise ValueError(
                f"Trajectory {relative.as_posix()!r} has {len(records)} turns; "
                f"at least {max_turns} are required."
            )

        previous_messages: List[Dict[str, Any]] | None = None
        for turn_id, record in enumerate(records[:max_turns], start=1):
            request = record.request
            if request.get("drop_rule") is not None or request.get("drop_message") is not None:
                raise ValueError(
                    f"Trajectory {relative.as_posix()!r} turn {turn_id} contains a Drop payload; "
                    "bench-trajectories requires full no-drop captures."
                )
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Trajectory {relative.as_posix()!r} turn {turn_id} requires a non-empty "
                    "messages list."
                )
            if (
                previous_messages is not None
                and messages[: len(previous_messages)] != previous_messages
            ):
                raise ValueError(
                    f"Trajectory {relative.as_posix()!r} turn {turn_id} is not a full-history "
                    "extension of the previous turn."
                )
            previous_messages = messages

        tasks.append(
            TrajectoryTask(
                task_id=relative.with_suffix("").as_posix(),
                source_path=relative.as_posix(),
                turns=records[:max_turns],
            )
        )

    if len(tasks) < min_tasks:
        raise ValueError(
            f"Trajectory directory has {len(tasks)} tasks; at least {min_tasks} are required."
        )
    return tasks


def load_manifest(path: str | Path) -> List[ManifestCase]:
    cases = [ManifestCase.model_validate(record) for record in _load_jsonl(path)]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Manifest case_id values must be unique.")
    return cases


def dump_jsonl(path: str | Path, values: Iterable[BaseModel | Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for value in values:
            payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
            file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def coverage_matrix(
    cases: Sequence[ManifestCase],
    *,
    models: Sequence[str],
    methods: Sequence[CorrectnessMethod] = ("full", "summary"),
) -> Dict[str, Any]:
    expected = {
        (model, method, shape)
        for model in models
        for method in methods
        for shape in MESSAGE_SHAPES
    }
    counts = {cell: 0 for cell in expected}
    excluded: List[Dict[str, str]] = []

    for case in cases:
        try:
            case.ensure_correctness_scope()
        except ValueError as exc:
            excluded.append({"case_id": case.case_id, "reason": str(exc)})
            continue
        for shape in case.detected_shapes():
            cell = (case.metadata.model, case.metadata.method, shape)
            if cell in counts:
                counts[cell] += 1

    cells = [
        {"model": model, "method": method, "shape": shape, "count": counts[(model, method, shape)]}
        for model in models
        for method in methods
        for shape in MESSAGE_SHAPES
    ]
    missing = [cell for cell in cells if cell["count"] == 0]
    return {
        "expected_cells": len(expected),
        "covered_cells": len(expected) - len(missing),
        "missing_cells": missing,
        "cells": cells,
        "excluded_cases": excluded,
    }
