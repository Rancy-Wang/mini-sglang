from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from minisgl.benchmark.reposition_trajectory import inspect_message

ROLLING_WINDOW_TURNS = 5
CAPTURE_NAME = "request.json.gz"
_CASE_PATTERN = re.compile(r"^case_(.+)$")
_CALL_PATTERN = re.compile(r"^(\d+)_agent_(\d+)_retry_(\d+)$")


@dataclass(frozen=True)
class TurnGroup:
    """One completed assistant turn and all immediately following tool results."""

    number: int
    assistant_message_id: int
    message_ids: tuple[int, ...]

    @property
    def trigger_message_id(self) -> int:
        return self.message_ids[-1]


@dataclass(frozen=True)
class RollingPlan:
    incoming_turn: int
    protected_prefix_ids: tuple[int, ...]
    turns: tuple[TurnGroup, ...]
    drop_messages: dict[str, list[int]]
    reposition: tuple[int, ...]

    def interface(self, *, include_reposition: bool = True) -> dict[str, Any]:
        if not self.drop_messages:
            return {}
        result: dict[str, Any] = {
            "drop_rule": {
                "type": "message_drop",
                "drop_messages": copy.deepcopy(self.drop_messages),
            }
        }
        if include_reposition:
            result["reposition"] = list(self.reposition)
        return result

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["turns"] = [asdict(turn) for turn in self.turns]
        return result


@dataclass(frozen=True)
class CapturedRequest:
    case_id: str
    physical_id: int
    logical_id: int
    retry_id: int
    path: Path
    request: dict[str, Any]
    provenance: str = "raw_capture"


@dataclass(frozen=True)
class ReplayTask:
    case_id: str
    requests: tuple[CapturedRequest, ...]
    prompt_tokens_hint: int | None = None


def browsecomp_plus_tools() -> list[dict[str, Any]]:
    """Return the stable tool surface needed to reconstruct readable BCP trajectories."""

    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the BrowseComp Plus document index.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_document",
                "description": "Fetch one full document by document ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"docid": {"type": "string"}},
                    "required": ["docid"],
                },
            },
        },
    ]


def derive_turn_groups(
    messages: Sequence[dict[str, Any]],
) -> tuple[tuple[int, ...], tuple[TurnGroup, ...]]:
    """Derive rolling ownership from public, zero-based message IDs.

    A turn owns exactly its assistant message and consecutive tool responses. Other roles remain
    outside turn ownership and therefore cannot be removed by this rolling policy.
    """

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("messages must be a sequence")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("every message must be an object")
    first_assistant = next(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        len(messages),
    )
    protected = tuple(range(first_assistant))
    turns: list[TurnGroup] = []
    index = first_assistant
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role != "assistant":
            if role == "tool":
                raise ValueError(f"messages[{index}] is a tool response without an assistant owner")
            index += 1
            continue

        message_ids = [index]
        declared_call_ids: set[str] = set()
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                if isinstance(call_id, str):
                    declared_call_ids.add(call_id)
        index += 1
        while index < len(messages) and messages[index].get("role") == "tool":
            tool_call_id = messages[index].get("tool_call_id")
            if (
                isinstance(tool_call_id, str)
                and declared_call_ids
                and tool_call_id not in declared_call_ids
            ):
                raise ValueError(
                    f"messages[{index}] tool_call_id is not declared by its assistant turn"
                )
            message_ids.append(index)
            index += 1
        turns.append(
            TurnGroup(
                number=len(turns) + 1,
                assistant_message_id=message_ids[0],
                message_ids=tuple(message_ids),
            )
        )

    return protected, tuple(turns)


def build_rolling_plan(
    messages: Sequence[dict[str, Any]],
    *,
    max_active_turns: int = ROLLING_WINDOW_TURNS,
) -> RollingPlan:
    """Build all cumulative Drop+Reposition events for the next assistant generation."""

    if max_active_turns < 1:
        raise ValueError("max_active_turns must be positive")
    protected, turns = derive_turn_groups(messages)
    incoming_turn = len(turns) + 1
    drop_messages: dict[str, list[int]] = {}
    reposition: list[int] = []
    first_dropping_turn = max_active_turns + 1
    for generated_turn in range(first_dropping_turn, incoming_turn + 1):
        dropped = turns[generated_turn - max_active_turns - 1]
        trigger = turns[generated_turn - 2].trigger_message_id
        if trigger <= dropped.trigger_message_id:
            raise ValueError("rolling trigger must follow every message it drops")
        key = str(trigger)
        if key in drop_messages:
            raise ValueError(f"duplicate rolling trigger message ID: {trigger}")
        drop_messages[key] = list(dropped.message_ids)
        reposition.append(trigger)

    dropped_ids = {item for ids in drop_messages.values() for item in ids}
    if dropped_ids.intersection(protected):
        raise ValueError("rolling policy attempted to drop the protected prefix")
    return RollingPlan(
        incoming_turn=incoming_turn,
        protected_prefix_ids=protected,
        turns=turns,
        drop_messages=drop_messages,
        reposition=tuple(reposition),
    )


def apply_rolling_interface(
    request: dict[str, Any],
    *,
    mode: str,
    max_active_turns: int = ROLLING_WINDOW_TURNS,
) -> tuple[dict[str, Any], RollingPlan]:
    """Return a copy in full, rolling, or diagnostic rolling-drop-only mode."""

    if mode not in {"full", "rolling", "rolling-drop-only"}:
        raise ValueError(f"unknown interface mode: {mode}")
    result = copy.deepcopy(request)
    messages = result.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("request.messages must be an array of objects")
    result.pop("drop_message", None)
    result.pop("drop_rule", None)
    result.pop("reposition", None)
    plan = build_rolling_plan(messages, max_active_turns=max_active_turns)
    if mode != "full":
        result.update(plan.interface(include_reposition=mode == "rolling"))
    return result, plan


def apply_fixed_shape_interface(request: dict[str, Any], *, shape: str) -> dict[str, Any]:
    """Build one of the six fixed correctness shapes used by the request matrix."""

    result, plan = apply_rolling_interface(request, mode="full")
    events = list(plan.drop_messages.items())
    if shape == "two_drop_reposition":
        selected = events[:2]
        if len(selected) < 2:
            raise ValueError("two_drop_reposition needs at least two effective events")
        reposition = [int(trigger) for trigger, _ in selected]
    elif shape == "rolling_three_or_four":
        selected = events[-4:]
        reposition = [int(trigger) for trigger, _ in selected]
        if len(selected) < 3:
            raise ValueError("rolling_three_or_four needs at least three effective events")
    elif shape == "drops_before_later_reposition":
        selected = events
        if len(selected) < 2:
            raise ValueError("drops_before_later_reposition needs at least two effective events")
        reposition = [int(selected[-1][0])] if len(selected) >= 2 else []
    elif shape == "same_boundary":
        selected = events[-1:]
        reposition = [int(selected[0][0])] if selected else []
    elif shape == "overlapping_newly_effective":
        selected = events[:2]
        if len(selected) < 2:
            raise ValueError("overlapping_newly_effective needs at least two effective events")
        selected[1] = (selected[1][0], selected[0][1] + selected[1][1])
        reposition = [int(trigger) for trigger, _ in selected]
    elif shape == "long_partial_rehit":
        selected = events
        reposition = [int(trigger) for trigger, _ in selected]
    else:
        raise ValueError(f"unknown fixed shape: {shape}")
    if not selected:
        raise ValueError(f"{shape} needs at least one effective rolling event")
    result["drop_rule"] = {
        "type": "message_drop",
        "drop_messages": {trigger: list(ids) for trigger, ids in selected},
    }
    result["reposition"] = reposition
    return result


FIXED_SHAPES = (
    "two_drop_reposition",
    "rolling_three_or_four",
    "drops_before_later_reposition",
    "same_boundary",
    "overlapping_newly_effective",
    "long_partial_rehit",
)


def materialize_fixed_request_cases(
    tasks: Sequence[ReplayTask],
    output_dir: Path,
    *,
    model: str | None = None,
    max_tokens: int = 2048,
) -> list[Path]:
    """Write 6 semantic shapes x stream/non-stream, each as cold/partial/warm replay."""

    candidates = [task for task in tasks if len(task.requests) >= 9]
    if not candidates:
        raise ValueError("fixed request cases need a task with at least nine captured turns")
    task = max(candidates, key=lambda item: item.prompt_tokens_hint or len(item.requests))
    adjacent = []
    for cold_candidate, partial_candidate in zip(task.requests, task.requests[1:]):
        cold_messages = cold_candidate.request.get("messages")
        partial_messages = partial_candidate.request.get("messages")
        if (
            isinstance(cold_messages, list)
            and isinstance(partial_messages, list)
            and partial_messages[: len(cold_messages)] == cold_messages
        ):
            adjacent.append((cold_candidate, partial_candidate))
    if not adjacent:
        raise ValueError("fixed request task has no prefix-extending adjacent request pair")
    cold_source, partial_source = adjacent[-1]
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for shape in FIXED_SHAPES:
        cold = apply_fixed_shape_interface(cold_source.request, shape=shape)
        partial = apply_fixed_shape_interface(partial_source.request, shape=shape)
        for stream in (False, True):
            chain = [cold, partial, copy.deepcopy(partial)]
            for request in chain:
                request["stream"] = stream
                request["max_tokens"] = max_tokens
                request["temperature"] = 0.0
                request["top_p"] = 1.0
                request["top_k"] = -1
                if model is not None:
                    request["model"] = model
                else:
                    request.pop("model", None)
                if stream:
                    request["stream_options"] = {"include_usage": True}
                else:
                    request.pop("stream_options", None)
            suffix = "stream" if stream else "nonstream"
            path = output_dir / f"{shape}__{suffix}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as output:
                for request in chain:
                    output.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
            paths.append(path)
            manifest.append(
                {
                    "path": path.name,
                    "shape": shape,
                    "stream": stream,
                    "replay": ["cold", "partial_hit", "warm_rehit"],
                    "source_case_id": task.case_id,
                    "source_provenance": cold_source.provenance,
                }
            )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def request_sha256(request: dict[str, Any]) -> str:
    data = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    request = payload.get("request", payload)
    if not isinstance(request, dict):
        raise ValueError(f"{path} contains no request object")
    return request


def _capture_identity(path: Path) -> tuple[str, int, int, int] | None:
    case_id: str | None = None
    call: tuple[int, int, int] | None = None
    for part in path.parts:
        case_match = _CASE_PATTERN.match(part)
        if case_match:
            case_id = case_match.group(1)
        call_match = _CALL_PATTERN.match(part)
        if call_match:
            call = tuple(int(item) for item in call_match.groups())
    if case_id is None or call is None:
        return None
    return case_id, *call


def discover_captured_requests(root: Path) -> list[CapturedRequest]:
    """Discover readable captures without failing the whole scan on protected directories."""

    captures: list[CapturedRequest] = []
    errors: list[OSError] = []
    for directory, _, filenames in os.walk(root, onerror=errors.append):
        if CAPTURE_NAME not in filenames:
            continue
        path = Path(directory) / CAPTURE_NAME
        identity = _capture_identity(path)
        if identity is None:
            continue
        case_id, physical_id, logical_id, retry_id = identity
        try:
            request = _load_json(path)
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError, ValueError):
            continue
        captures.append(
            CapturedRequest(
                case_id=case_id,
                physical_id=physical_id,
                logical_id=logical_id,
                retry_id=retry_id,
                path=path,
                request=request,
            )
        )
    return sorted(
        captures,
        key=lambda item: (item.case_id, item.physical_id, item.logical_id, item.retry_id),
    )


def load_trajectory_replay_tasks(
    paths: Iterable[Path],
    *,
    model: str | None = None,
    prompt_token_hints: dict[str, int] | None = None,
) -> list[ReplayTask]:
    """Reconstruct request prefixes when protected backend capture directories are unreadable."""

    tasks: list[ReplayTask] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = str(row.get("case_id", "")) if isinstance(row, dict) else ""
                trajectory = row.get("trajectory") if isinstance(row, dict) else None
                if not case_id or not isinstance(trajectory, list):
                    raise ValueError(f"{path}:{line_number} has no case_id/trajectory")
                if not all(isinstance(message, dict) for message in trajectory):
                    raise ValueError(f"{path}:{line_number} trajectory contains a non-object")
                captures: list[CapturedRequest] = []
                for message_id, message in enumerate(trajectory):
                    if message.get("role") != "assistant":
                        continue
                    request: dict[str, Any] = {
                        "messages": copy.deepcopy(trajectory[:message_id]),
                        "tools": browsecomp_plus_tools(),
                        "stream": True,
                    }
                    if model is not None:
                        request["model"] = model
                    captures.append(
                        CapturedRequest(
                            case_id=case_id,
                            physical_id=len(captures),
                            logical_id=len(captures),
                            retry_id=0,
                            path=path,
                            request=request,
                            provenance="trajectory_reconstructed",
                        )
                    )
                tasks.append(
                    ReplayTask(
                        case_id=case_id,
                        requests=tuple(captures),
                        prompt_tokens_hint=(prompt_token_hints or {}).get(case_id),
                    )
                )
    return sorted(tasks, key=lambda item: item.case_id)


def load_rollout_prompt_token_hints(paths: Iterable[Path]) -> dict[str, int]:
    hints: dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                    raise ValueError(f"{path}:{line_number} has no rollout metadata")
                case_id = str(row.get("case_id", ""))
                maximum = 0
                for call in row["metadata"].get("model_calls", []):
                    usage = call.get("usage") if isinstance(call, dict) else None
                    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    if isinstance(prompt_tokens, int):
                        maximum = max(maximum, prompt_tokens)
                if case_id and maximum:
                    hints[case_id] = max(hints.get(case_id, 0), maximum)
    return hints


def select_task_set(
    tasks: Iterable[ReplayTask],
    *,
    limit: int,
    preferred_case_ids: Sequence[str] = (),
    minimum_requests: int = 6,
) -> list[ReplayTask]:
    """Select fixed IDs while preserving 32k/64k/100k+ prompt-length coverage when present."""

    if limit < 1 or minimum_requests < 1:
        raise ValueError("limit and minimum_requests must be positive")
    eligible = {task.case_id: task for task in tasks if len(task.requests) >= minimum_requests}
    selected: list[str] = []
    for case_id in map(str, preferred_case_ids):
        if case_id in eligible and case_id not in selected:
            selected.append(case_id)

    bins = ((100_000, None), (64_000, 100_000), (32_000, 64_000))
    for lower, upper in bins:
        candidates = [
            task
            for task in eligible.values()
            if task.case_id not in selected
            and task.prompt_tokens_hint is not None
            and task.prompt_tokens_hint >= lower
            and (upper is None or task.prompt_tokens_hint < upper)
        ]
        if candidates and len(selected) < limit:
            selected.append(max(candidates, key=lambda task: task.prompt_tokens_hint or 0).case_id)

    remaining = sorted(
        (task for task in eligible.values() if task.case_id not in selected),
        key=lambda task: (-(task.prompt_tokens_hint or 0), task.case_id),
    )
    selected.extend(task.case_id for task in remaining[: max(0, limit - len(selected))])
    return [eligible[case_id] for case_id in selected[:limit]]


def select_replay_tasks(
    captures: Iterable[CapturedRequest],
    *,
    limit: int,
    preferred_case_ids: Sequence[str] = (),
    minimum_requests: int = 6,
) -> list[ReplayTask]:
    if limit < 1 or minimum_requests < 1:
        raise ValueError("limit and minimum_requests must be positive")
    grouped: dict[str, dict[tuple[int, int], CapturedRequest]] = {}
    for capture in captures:
        key = (capture.physical_id, capture.logical_id)
        current = grouped.setdefault(capture.case_id, {}).get(key)
        if current is None or capture.retry_id > current.retry_id:
            grouped[capture.case_id][key] = capture
    eligible = {
        case_id: tuple(sorted(items.values(), key=lambda item: (item.physical_id, item.logical_id)))
        for case_id, items in grouped.items()
        if len(items) >= minimum_requests
    }
    preferred = [str(case_id) for case_id in preferred_case_ids]
    order = [case_id for case_id in preferred if case_id in eligible]
    order.extend(case_id for case_id in sorted(eligible) if case_id not in set(order))
    return [ReplayTask(case_id, eligible[case_id]) for case_id in order[:limit]]


def audit_parsed_response(
    parsed: dict[str, Any],
    *,
    request: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message = parsed.get("message")
    if not isinstance(message, dict):
        return {"issues": ["transport:missing_parsed_message"], "inspection": {}}
    issues, inspection = inspect_message(message, tools=request.get("tools"), expected=expected)
    tool_calls = inspection.get("canonical_tool_calls", [])
    finish_reason = parsed.get("finish_reason")
    if finish_reason == "tool_calls" and not tool_calls:
        issues.append("tool_calls:finish_without_calls")
    if tool_calls and finish_reason not in {"tool_calls", "stop", None}:
        issues.append("tool_calls:unexpected_finish_reason")
    return {"issues": sorted(set(issues)), "inspection": inspection}


def attribute_issues(
    candidate_issues: Sequence[str], baseline_issues: Sequence[str]
) -> list[dict[str, str]]:
    """Use the paired full-context run to avoid blaming a model symptom on the system."""

    baseline = set(baseline_issues)
    result: list[dict[str, str]] = []
    for issue in sorted(set(candidate_issues)):
        if issue.startswith("transport:") or issue.startswith("system:"):
            attribution = "system"
        elif issue in baseline:
            attribution = "model_or_benchmark"
        else:
            attribution = "reposition_system_suspect"
        result.append({"issue": issue, "attribution": attribution})
    return result


def append_gzip_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def render_text_trajectory(records: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for record in records:
        label = " / ".join(str(record.get(key, "?")) for key in ("case_id", "mode", "turn"))
        message = record.get("response", {}).get("message", {})
        blocks.append(f"===== {label} =====")
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if reasoning:
            blocks.append("[reasoning]\n" + str(reasoning))
        if content:
            blocks.append("[content]\n" + str(content))
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if calls:
            blocks.append("[tool_calls]\n" + json.dumps(calls, ensure_ascii=False, indent=2))
        issues = record.get("audit", {}).get("issues", [])
        blocks.append("[issues] " + (", ".join(issues) if issues else "none"))
    return "\n\n".join(blocks) + ("\n" if blocks else "")
