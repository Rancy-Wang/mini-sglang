from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List


SUPPORTED_TOOL_CALL_PARSERS = ("qwen", "qwen25", "qwen3_coder", "llama3", "gpt-oss")
SUPPORTED_REASONING_PARSERS = ("qwen3", "deepseek-r1", "gpt-oss")


def infer_tool_call_parser(model_path: str, configured: str | None = "auto") -> str | None:
    """Resolve the SGLang-compatible parser used by the supported model families."""

    if configured not in {None, "auto"}:
        return "qwen" if configured == "qwen25" else configured
    model = model_path.lower().replace("_", "-")
    if "gpt-oss" in model or "gptoss" in model:
        return "gpt-oss"
    if "qwen3-coder" in model:
        return "qwen3_coder"
    if "deepseek-r1-distill-llama" in model or "llama" in model:
        return "llama3"
    if "qwen" in model or "deepseek-r1-distill-qwen" in model:
        return "qwen"
    return None


def infer_reasoning_parser(model_path: str, configured: str | None = "auto") -> str | None:
    if configured not in {None, "auto"}:
        return configured
    model = model_path.lower().replace("_", "-")
    if "gpt-oss" in model or "gptoss" in model:
        return "gpt-oss"
    if "deepseek-r1" in model:
        return "deepseek-r1"
    if "qwen3" in model or "agenticqwen" in model:
        return "qwen3"
    return None


def _tool_name(tool: Dict[str, Any]) -> str | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _tool_call(name: str, arguments: Any, index: int, call_id: str | None = None) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "index": index,
        "function": {"name": name, "arguments": arguments},
    }


@dataclass(frozen=True)
class ParsedResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] | None = None


@dataclass(frozen=True)
class StreamPiece:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class _ReasoningParser:
    def __init__(
        self,
        parser: str | None,
        force_reasoning: bool,
        stream_reasoning: bool = True,
        tool_marker: str | None = None,
    ) -> None:
        self.parser = parser
        self.stream_reasoning = stream_reasoning
        self.tool_marker = tool_marker
        self.in_reasoning = parser == "deepseek-r1" or force_reasoning
        self.seen_start = False
        self.buffer = ""

    def parse_full(self, text: str) -> tuple[str, str]:
        if self.parser not in {"qwen3", "deepseek-r1"}:
            return "", text
        in_reasoning = self.parser == "deepseek-r1" or self.in_reasoning or "<think>" in text
        if not in_reasoning:
            return "", text
        while text.startswith("<think>"):
            text = text[len("<think>") :]
        if "</think>" in text:
            reasoning, content = text.split("</think>", 1)
            return reasoning, content
        if self.tool_marker and self.tool_marker in text:
            split = text.index(self.tool_marker)
            return text[:split], text[split:]
        return text, ""

    @staticmethod
    def _hold_partial(text: str, markers: tuple[str, ...]) -> tuple[str, str]:
        hold = 0
        for marker in markers:
            for size in range(1, min(len(text), len(marker) - 1) + 1):
                if marker.startswith(text[-size:]):
                    hold = max(hold, size)
        return (text[:-hold], text[-hold:]) if hold else (text, "")

    def feed(self, text: str) -> tuple[str, str]:
        if self.parser not in {"qwen3", "deepseek-r1"}:
            return "", text
        self.buffer += text
        if not self.seen_start and "<think>" in self.buffer:
            before, self.buffer = self.buffer.split("<think>", 1)
            self.seen_start = True
            self.in_reasoning = True
            return "", before
        if not self.in_reasoning:
            emitted, self.buffer = self._hold_partial(self.buffer, ("<think>",))
            return "", emitted
        if "</think>" in self.buffer:
            reasoning, content = self.buffer.split("</think>", 1)
            self.buffer = ""
            self.in_reasoning = False
            return reasoning, content
        if self.tool_marker and self.tool_marker in self.buffer:
            split = self.buffer.index(self.tool_marker)
            reasoning, content = self.buffer[:split], self.buffer[split:]
            self.buffer = ""
            self.in_reasoning = False
            return reasoning, content
        if not self.stream_reasoning:
            return "", ""
        markers = ("</think>", self.tool_marker) if self.tool_marker else ("</think>",)
        emitted, self.buffer = self._hold_partial(self.buffer, markers)
        return emitted, ""

    def finish(self) -> tuple[str, str]:
        text, self.buffer = self.buffer, ""
        return (text, "") if self.in_reasoning else ("", text)


def _parse_qwen(text: str, tools: List[Dict[str, Any]]) -> tuple[str, list[dict]]:
    begin, end = "<tool_call>\n", "\n</tool_call>"
    first = text.find(begin)
    if first < 0:
        return text, []
    names = {_tool_name(tool) for tool in tools}
    calls: list[dict] = []
    pattern = re.compile(re.escape(begin) + r"(.*?)" + re.escape(end), re.DOTALL)
    for match in pattern.finditer(text):
        try:
            item = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        items = item if isinstance(item, list) else [item]
        for raw in items:
            if not isinstance(raw, dict) or raw.get("name") not in names:
                continue
            calls.append(_tool_call(raw["name"], raw.get("arguments", {}), len(calls)))
    return text[:first].strip(), calls


def _schema_properties(tools: List[Dict[str, Any]], name: str) -> dict:
    for tool in tools:
        fn = tool.get("function", {})
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters", {})
            return params.get("properties", {}) if isinstance(params, dict) else {}
    return {}


def _convert_qwen3_coder_value(value: str, spec: Any) -> Any:
    kind = spec.get("type") if isinstance(spec, dict) else None
    if value.lower() == "null":
        return None
    if kind in {"integer", "number", "boolean", "object", "array"}:
        try:
            return json.loads(value.lower() if kind == "boolean" else value)
        except (TypeError, ValueError):
            pass
    try:
        return ast.literal_eval(value) if kind in {"object", "array"} else value
    except (SyntaxError, ValueError):
        return value


def _parse_qwen3_coder(text: str, tools: List[Dict[str, Any]]) -> tuple[str, list[dict]]:
    first = text.find("<tool_call>")
    if first < 0:
        return text, []
    names = {_tool_name(tool) for tool in tools}
    calls: list[dict] = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        for fn_match in re.finditer(r"<function=([^>]+)>(.*?)</function>", block, re.DOTALL):
            name, body = fn_match.groups()
            if name not in names:
                continue
            properties = _schema_properties(tools, name)
            arguments = {}
            for param in re.finditer(
                r"<parameter=([^>]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)",
                body,
                re.DOTALL,
            ):
                key, value = param.groups()
                arguments[key] = _convert_qwen3_coder_value(
                    value.removeprefix("\n").removesuffix("\n"), properties.get(key)
                )
            calls.append(_tool_call(name, arguments, len(calls)))
    return text[:first], calls


def _parse_llama(text: str, tools: List[Dict[str, Any]]) -> tuple[str, list[dict]]:
    if "<|python_tag|>" in text:
        normal, action = text.split("<|python_tag|>", 1)
    elif text.startswith("{"):
        normal, action = "", text
    else:
        return text, []
    names = {_tool_name(tool) for tool in tools}
    calls: list[dict] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(action):
        while cursor < len(action) and (action[cursor].isspace() or action[cursor] == ";"):
            cursor += 1
        try:
            item, used = decoder.raw_decode(action[cursor:])
        except json.JSONDecodeError:
            try:
                item = ast.literal_eval(action[cursor:])
                used = len(action) - cursor
            except (SyntaxError, ValueError):
                break
        cursor += used
        if isinstance(item, dict) and item.get("name") in names:
            calls.append(_tool_call(item["name"], item.get("arguments", {}), len(calls)))
    return normal, calls


@lru_cache(maxsize=1)
def _harmony_encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _parse_harmony(text: str, tools: List[Dict[str, Any]]) -> ParsedResponse:
    from openai_harmony import HarmonyError, Role, StreamableParser

    encoding = _harmony_encoding()
    parser = StreamableParser(encoding, Role.ASSISTANT, strict=False)
    try:
        for token in encoding.encode(text, allowed_special="all"):
            parser.process(int(token))
    except HarmonyError:
        # GPT-OSS occasionally places `to=functions.*` after the channel or
        # content-type field.  The streaming parser accepts that unambiguous
        # header, so keep non-stream responses behaviorally identical instead
        # of turning a recoverable tool call into HTTP 500.
        fallback = _HarmonyStream(tools)
        pieces = (fallback.feed(text), fallback.finish())
        calls = [call for piece in pieces for call in piece.tool_calls]
        return ParsedResponse(
            content="".join(piece.content for piece in pieces),
            reasoning_content="".join(piece.reasoning_content for piece in pieces),
            tool_calls=calls or None,
        )

    entries: list[tuple[str | None, str | None, str]] = []
    for message in parser.messages:
        content = "".join(
            text
            for part in getattr(message, "content", ())
            if isinstance((text := getattr(part, "text", None)), str)
        )
        entries.append((message.channel, message.recipient, content))
    if parser.current_content:
        entries.append((parser.current_channel, parser.current_recipient, parser.current_content))

    names = {_tool_name(tool) for tool in tools}
    content_parts, reasoning_parts, calls = [], [], []
    for channel, recipient, content in entries:
        if recipient:
            name = recipient.split(".")[-1]
            if name not in names:
                continue
            try:
                arguments = json.loads(content) if content.strip() else {}
            except json.JSONDecodeError:
                continue
            calls.append(_tool_call(name, arguments, len(calls)))
        elif channel == "analysis":
            reasoning_parts.append(content)
        elif channel in {"final", "commentary", None}:
            content_parts.append(content)
    return ParsedResponse("".join(content_parts), "".join(reasoning_parts), calls or None)


class _BufferedToolStream:
    """Emit normal text immediately and each native tool call as soon as it closes."""

    def __init__(self, parser: str, tools: List[Dict[str, Any]]) -> None:
        self.parser = parser
        self.tools = tools
        self.buffer = ""
        self.emitted_calls = 0
        self.begin = {
            "qwen": "<tool_call>\n",
            "qwen3_coder": "<tool_call>",
            "llama3": "<|python_tag|>",
        }[parser]
        self.end = {
            "qwen": "\n</tool_call>",
            "qwen3_coder": "</tool_call>",
            "llama3": "",
        }[parser]

    @staticmethod
    def _partial_suffix(text: str, marker: str) -> int:
        for size in range(min(len(text), len(marker) - 1), 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0

    def feed(self, text: str) -> StreamPiece:
        self.buffer += text
        if self.parser == "llama3":
            start = self.buffer.find(self.begin)
            if start >= 0:
                content, self.buffer = self.buffer[:start], self.buffer[start:]
                return StreamPiece(content=content)
            if self.buffer.lstrip().startswith("{"):
                return StreamPiece()
        start = self.buffer.find(self.begin)
        if start < 0:
            hold = self._partial_suffix(self.buffer, self.begin)
            content = self.buffer[:-hold] if hold else self.buffer
            self.buffer = self.buffer[-hold:] if hold else ""
            return StreamPiece(content=content)

        content = self.buffer[:start]
        if self.parser == "llama3":
            return StreamPiece(content=content)
        close = self.buffer.find(self.end, start + len(self.begin))
        if close < 0:
            self.buffer = self.buffer[start:]
            return StreamPiece(content=content)
        call_end = close + len(self.end)
        call_text = self.buffer[start:call_end]
        self.buffer = self.buffer[call_end:]
        _, calls = (
            _parse_qwen(call_text, self.tools)
            if self.parser == "qwen"
            else _parse_qwen3_coder(call_text, self.tools)
        )
        deltas = []
        for call in calls:
            call["index"] = self.emitted_calls
            self.emitted_calls += 1
            deltas.append(call)
        tail = self.feed("") if self.buffer else StreamPiece()
        return StreamPiece(
            content=content + tail.content,
            tool_calls=deltas + tail.tool_calls,
        )

    def finish(self) -> StreamPiece:
        if self.parser == "llama3":
            content, calls = _parse_llama(self.buffer, self.tools)
            self.buffer = ""
            return StreamPiece(content=content, tool_calls=calls)
        content, self.buffer = self.buffer, ""
        return StreamPiece(content=content)


class _HarmonyStream:
    """Incrementally separate Harmony channels without leaking control tokens."""

    _TERMINATORS = ("<|end|>", "<|call|>", "<|return|>")

    def __init__(self, tools: List[Dict[str, Any]], stream_reasoning: bool = True) -> None:
        self.names = {_tool_name(tool) for tool in tools}
        self.stream_reasoning = stream_reasoning
        self.buffer = ""
        self.reasoning_buffer = ""
        self.state = "seek"
        self.channel: str | None = None
        self.recipient: str | None = None
        self.tool_index = 0

    @staticmethod
    def _hold_partial(text: str, markers: tuple[str, ...]) -> tuple[str, str]:
        hold = 0
        for marker in markers:
            for size in range(1, min(len(text), len(marker) - 1) + 1):
                if marker.startswith(text[-size:]):
                    hold = max(hold, size)
        return (text[:-hold], text[-hold:]) if hold else (text, "")

    def _piece_for_content(self, content: str, *, terminal: bool) -> StreamPiece:
        if self.recipient:
            if not terminal:
                return StreamPiece()
            name = self.recipient.split(".")[-1]
            if name not in self.names:
                return StreamPiece()
            try:
                arguments = json.loads(content) if content.strip() else {}
            except json.JSONDecodeError:
                return StreamPiece()
            call = _tool_call(name, arguments, self.tool_index)
            self.tool_index += 1
            return StreamPiece(tool_calls=[call])
        if self.channel == "analysis":
            if not self.stream_reasoning:
                self.reasoning_buffer += content
                if not terminal:
                    return StreamPiece()
                content = self.reasoning_buffer
                self.reasoning_buffer = ""
            return StreamPiece(reasoning_content=content)
        if self.channel in {"final", "commentary"}:
            return StreamPiece(content=content)
        return StreamPiece()

    def feed(self, text: str) -> StreamPiece:
        self.buffer += text
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[dict] = []

        def append(piece: StreamPiece) -> None:
            content_parts.append(piece.content)
            reasoning_parts.append(piece.reasoning_content)
            calls.extend(piece.tool_calls)

        while self.buffer:
            if self.state == "seek":
                marker = self.buffer.find("<|channel|>")
                if marker < 0:
                    # Harmony output normally starts with structural markers.
                    # Preserve a possible split marker and pass through genuine
                    # plain text produced by compatibility templates.
                    emitted, self.buffer = self._hold_partial(
                        self.buffer, ("<|start|>", "<|channel|>")
                    )
                    if "<|" not in emitted and not emitted.endswith("assistant"):
                        content_parts.append(emitted)
                    break
                self.buffer = self.buffer[marker + len("<|channel|>") :]
                self.state = "header"
                continue

            if self.state == "header":
                marker = self.buffer.find("<|message|>")
                if marker < 0:
                    break
                header = self.buffer[:marker].strip()
                self.buffer = self.buffer[marker + len("<|message|>") :]
                self.channel = header.split(None, 1)[0].lower() if header else None
                recipient = re.search(r"\bto=([^\s<]+)", header)
                self.recipient = recipient.group(1) if recipient else None
                self.state = "content"
                continue

            positions = [
                (self.buffer.find(marker), marker)
                for marker in self._TERMINATORS
                if self.buffer.find(marker) >= 0
            ]
            if not positions:
                if self.recipient:
                    break
                emitted, self.buffer = self._hold_partial(self.buffer, self._TERMINATORS)
                append(self._piece_for_content(emitted, terminal=False))
                break
            position, marker = min(positions, key=lambda item: item[0])
            append(self._piece_for_content(self.buffer[:position], terminal=True))
            self.buffer = self.buffer[position + len(marker) :]
            self.state = "seek"
            self.channel = self.recipient = None

        return StreamPiece("".join(content_parts), "".join(reasoning_parts), calls)

    def finish(self) -> StreamPiece:
        if self.state == "content":
            piece = self._piece_for_content(self.buffer, terminal=True)
        else:
            piece = StreamPiece()
        self.buffer = ""
        return piece


class ChatResponseParser:
    """Small serving adapter matching SGLang 0.5.15/0.5.16 model protocols."""

    def __init__(
        self,
        *,
        model_path: str,
        tools: List[Dict[str, Any]] | None,
        tool_call_parser: str | None,
        reasoning_parser: str | None,
        enable_thinking: bool | None,
        separate_reasoning: bool,
        stream_reasoning: bool = True,
    ) -> None:
        self.tools = tools or []
        self.tool_parser = infer_tool_call_parser(model_path, tool_call_parser) if tools else None
        self.reasoning_name = infer_reasoning_parser(model_path, reasoning_parser)
        self.separate_reasoning = separate_reasoning
        tool_marker = {
            "qwen": "<tool_call>",
            "qwen3_coder": "<tool_call>",
            "llama3": "<|python_tag|>",
        }.get(self.tool_parser)
        self.reasoning = _ReasoningParser(
            self.reasoning_name if separate_reasoning else None,
            force_reasoning=enable_thinking is True,
            stream_reasoning=stream_reasoning,
            tool_marker=tool_marker,
        )
        self.tool_stream = (
            _BufferedToolStream(self.tool_parser, self.tools)
            if self.tool_parser in {"qwen", "qwen3_coder", "llama3"}
            else None
        )
        self.harmony_stream = (
            _HarmonyStream(self.tools, stream_reasoning=stream_reasoning)
            if self.tool_parser == "gpt-oss" or self.reasoning_name == "gpt-oss"
            else None
        )
        self.has_tool_calls = False

    def parse_full(self, text: str) -> ParsedResponse:
        if self.tool_parser == "gpt-oss" or self.reasoning_name == "gpt-oss":
            return _parse_harmony(text, self.tools)
        reasoning, content = self.reasoning.parse_full(text)
        calls: list[dict] = []
        if self.tool_parser == "qwen":
            content, calls = _parse_qwen(content, self.tools)
        elif self.tool_parser == "qwen3_coder":
            content, calls = _parse_qwen3_coder(content, self.tools)
        elif self.tool_parser == "llama3":
            content, calls = _parse_llama(content, self.tools)
        return ParsedResponse(content, reasoning, calls or None)

    def feed(self, text: str) -> StreamPiece:
        if self.tool_parser == "gpt-oss" or self.reasoning_name == "gpt-oss":
            assert self.harmony_stream is not None
            piece = self.harmony_stream.feed(text)
            if piece.tool_calls:
                self.has_tool_calls = True
            return piece
        reasoning, content = self.reasoning.feed(text)
        parsed = self.tool_stream.feed(content) if self.tool_stream is not None else StreamPiece(content)
        if parsed.tool_calls:
            self.has_tool_calls = True
        return StreamPiece(parsed.content, reasoning, parsed.tool_calls)

    def finish(self) -> StreamPiece:
        if self.tool_parser == "gpt-oss" or self.reasoning_name == "gpt-oss":
            assert self.harmony_stream is not None
            piece = self.harmony_stream.finish()
            if piece.tool_calls:
                self.has_tool_calls = True
            return piece
        reasoning, content = self.reasoning.finish()
        parsed = self.tool_stream.feed(content) if self.tool_stream is not None else StreamPiece(content)
        tail = self.tool_stream.finish() if self.tool_stream is not None else StreamPiece()
        calls = parsed.tool_calls + tail.tool_calls
        if calls:
            self.has_tool_calls = True
        return StreamPiece(parsed.content + tail.content, reasoning, calls)
