"""R8 validation helpers and an opt-in live BCP HTTP replay.

The ordinary pytest cases are CPU-only.  Set ``MINISGL_R8_BASE_URL`` and
``MINISGL_R8_INPUT`` to replay BCP 822/844 against an already running server.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def validate_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not calls:
        raise AssertionError("Expected at least one generated tool call.")
    normalized = []
    for index, call in enumerate(calls):
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise AssertionError(f"tool_calls[{index}] has no function name: {call!r}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise AssertionError(f"tool_calls[{index}].function.arguments is not a string.")
        value = json.loads(arguments)
        if not isinstance(value, dict):
            raise AssertionError(f"tool_calls[{index}] arguments must decode to an object.")
        normalized.append({"name": function["name"], "arguments": value})
    return normalized


def assemble_stream_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    names: dict[int, str] = {}
    arguments: dict[int, list[str]] = defaultdict(list)
    finish_reason = None
    content: list[str] = []
    reasoning: list[str] = []
    for event in events:
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning.append(delta["reasoning_content"])
        for call in delta.get("tool_calls") or []:
            index = int(call.get("index", 0))
            function = call.get("function") or {}
            if function.get("name"):
                names[index] = function["name"]
            if isinstance(function.get("arguments"), str):
                arguments[index].append(function["arguments"])
    calls = [
        {
            "type": "function",
            "function": {"name": names[index], "arguments": "".join(arguments[index])},
        }
        for index in sorted(names)
    ]
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": calls,
        "finish_reason": finish_reason,
    }


def parse_sse_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    events = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise AssertionError(f"Unexpected SSE line: {raw!r}")
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))
    return events


def test_valid_tool_calls_cover_escapes_unicode_nesting_and_multiple_calls():
    calls = [
        {
            "function": {
                "name": "search",
                "arguments": json.dumps(
                    {
                        "query": '中文🙂 "quoted" \\ path',
                        "nested": [{"values": [1, True, None]}, []],
                    },
                    ensure_ascii=False,
                ),
            }
        },
        {"function": {"name": "lookup", "arguments": '{"id":2}'}},
    ]
    assert validate_tool_calls(calls) == [
        {
            "name": "search",
            "arguments": {
                "query": '中文🙂 "quoted" \\ path',
                "nested": [{"values": [1, True, None]}, []],
            },
        },
        {"name": "lookup", "arguments": {"id": 2}},
    ]


def test_incomplete_or_non_object_arguments_are_not_accepted_as_tool_calls():
    for arguments in ('{"query":"unfinished', "[]", '"scalar"'):
        try:
            validate_tool_calls([{"function": {"name": "search", "arguments": arguments}}])
        except (AssertionError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError(f"Invalid arguments unexpectedly passed: {arguments!r}")


def test_stream_assembly_preserves_fragmented_unicode_and_nested_arguments():
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "search", "arguments": '{"q":"中'}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '文","x":[1,{"y":true}]}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    assembled = assemble_stream_events(events)
    assert assembled["finish_reason"] == "tool_calls"
    assert validate_tool_calls(assembled["tool_calls"]) == [
        {"name": "search", "arguments": {"q": "中文", "x": [1, {"y": True}]}}
    ]


def _load_bcp_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(gzip.decompress(path.read_bytes()))
    selected = [row for row in payload if str(row.get("case_id")) in {"822", "844"}]
    if {str(row["case_id"]) for row in selected} != {"822", "844"}:
        raise AssertionError("The input artifact must contain BCP cases 822 and 844.")
    return sorted(selected, key=lambda row: str(row["case_id"]))


def _first_tool_name(request: dict[str, Any]) -> str:
    tools = request.get("tools") or []
    if not tools:
        raise AssertionError("The tool-choice validation case has no tools.")
    function = tools[0].get("function") or {}
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise AssertionError(f"The first validation tool has no name: {tools[0]!r}")
    return name


def run_live(base_url: str, input_path: Path, output_path: Path, timeout: float) -> None:
    import httpx

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    results = []
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        cases = _load_bcp_cases(input_path)
        for case in cases:
            expected = None
            modes = []
            for stream in (False, True):
                request = dict(case["request"])
                request.update(
                    temperature=0,
                    seed=17,
                    max_tokens=512,
                    ignore_eos=False,
                    stream=stream,
                )
                if stream:
                    with client.stream("POST", endpoint, json=request) as response:
                        response.raise_for_status()
                        message = assemble_stream_events(parse_sse_lines(response.iter_lines()))
                else:
                    response = client.post(endpoint, json=request)
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    message = dict(choice["message"])
                    message["finish_reason"] = choice.get("finish_reason")
                normalized = validate_tool_calls(message.get("tool_calls") or [])
                if expected is None:
                    expected = normalized
                elif normalized != expected:
                    raise AssertionError(
                        f"BCP {case['case_id']} full/stream tool calls differ: "
                        f"{expected!r} != {normalized!r}"
                    )
                modes.append({"stream": stream, "message": message, "normalized": normalized})
            results.append({"case_id": case["case_id"], "modes": modes})

        choice_results = []
        for label, case, tool_choice in (
            ("required", cases[0], "required"),
            (
                "forced",
                cases[1],
                {
                    "type": "function",
                    "function": {"name": _first_tool_name(cases[1]["request"])},
                },
            ),
        ):
            request = dict(case["request"])
            request.update(
                temperature=0,
                seed=17,
                max_tokens=512,
                ignore_eos=False,
                stream=False,
                tool_choice=tool_choice,
            )
            response = client.post(endpoint, json=request)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            normalized = validate_tool_calls(choice["message"].get("tool_calls") or [])
            if label == "forced":
                forced_name = tool_choice["function"]["name"]
                if any(call["name"] != forced_name for call in normalized):
                    raise AssertionError(
                        f"Forced tool {forced_name!r} produced another call: {normalized!r}"
                    )
            choice_results.append(
                {
                    "mode": label,
                    "case_id": case["case_id"],
                    "finish_reason": choice.get("finish_reason"),
                    "normalized": normalized,
                }
            )

        truncated_request = dict(cases[0]["request"])
        truncated_request.update(
            temperature=0,
            seed=17,
            max_tokens=1,
            ignore_eos=False,
            stream=False,
            tool_choice="required",
        )
        truncated_response = client.post(endpoint, json=truncated_request)
        truncated_response.raise_for_status()
        truncated_choice = truncated_response.json()["choices"][0]
        truncated_message = truncated_choice["message"]
        if truncated_choice.get("finish_reason") != "length":
            raise AssertionError(f"Truncated grammar did not finish by length: {truncated_choice!r}")
        if truncated_message.get("tool_calls"):
            raise AssertionError(
                f"Truncated grammar was repaired into a tool call: {truncated_message!r}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "PASS",
                "endpoint": endpoint,
                "results": results,
                "tool_choice_results": choice_results,
                "truncation": {
                    "finish_reason": truncated_choice.get("finish_reason"),
                    "message": truncated_message,
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_live(args.base_url, args.input, args.output, args.timeout)
    print(json.dumps({"status": "PASS", "output": str(args.output)}))
