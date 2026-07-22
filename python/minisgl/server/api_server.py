from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    RequestErrorReply,
    TokenizeMsg,
    UserReply,
    WarmupReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


class RequestRejected(RuntimeError):
    def __init__(self, status_code: int, error_code: str, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail


def _http_error(exc: RequestRejected) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"message": exc.detail, "type": exc.error_code, "code": exc.error_code},
    )


def _stream_error_event(exc: RequestRejected) -> bytes:
    payload = {
        "error": {
            "message": exc.detail,
            "type": exc.error_code,
            "code": exc.error_code,
        }
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _unwrap_msg(
    msg: BaseFrontendMsg,
) -> List[UserReply | WarmupReply | RequestErrorReply]:
    if isinstance(msg, BatchFrontendMsg):
        result: List[UserReply | WarmupReply | RequestErrorReply] = []
        for reply in msg.data:
            assert isinstance(reply, (UserReply, WarmupReply, RequestErrorReply))
            result.append(reply)
        return result
    assert isinstance(msg, (UserReply, WarmupReply, RequestErrorReply))
    return [msg]


def _validate_drop_message(
    drop_message: Dict[int, List[int]],
    *,
    radix_drop_key_mode: str = "symbol",
    message_count: int | None = None,
) -> None:
    for raw_n, raw_ids in drop_message.items():
        n = int(raw_n)
        if n < 0:
            raise HTTPException(status_code=400, detail=f"drop_message key must be non-negative: {n}")
        if radix_drop_key_mode == "bitmask" and n >= 32:
            raise HTTPException(status_code=400, detail=f"drop_message key out of range [0, 31]: {n}")
        if radix_drop_key_mode == "symbol" and n >= (1 << 63):
            raise HTTPException(status_code=400, detail=f"drop_message key out of int64 range: {n}")
        for raw_id in raw_ids:
            msg_id = int(raw_id)
            if msg_id < 0:
                raise HTTPException(
                    status_code=400, detail=f"drop_message id must be non-negative: {msg_id}"
                )
            if radix_drop_key_mode == "bitmask" and msg_id >= 32:
                raise HTTPException(
                    status_code=400, detail=f"drop_message id out of range [0, 31]: {msg_id}"
                )
            if radix_drop_key_mode == "symbol" and msg_id >= (1 << 63):
                raise HTTPException(
                    status_code=400, detail=f"drop_message id out of int64 range: {msg_id}"
                )
            # Future schedules are intentionally accepted. For an event that
            # already applies to this prompt, however, every referenced message
            # must already exist.
            if message_count is not None and n < message_count:
                if msg_id >= message_count:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"drop_message id {msg_id} is outside the current message "
                            f"range [0, {message_count - 1}] for trigger {n}"
                        ),
                    )
            if msg_id > n:
                raise HTTPException(
                    status_code=400,
                    detail=f"drop_message event {n} cannot drop future message {msg_id}",
                )


def _to_wire_drop_message(drop_message: Dict[int, List[int]] | None) -> Dict[str, List[int]] | None:
    if drop_message is None:
        return None
    return {str(int(raw_n)): [int(raw_id) for raw_id in raw_ids] for raw_n, raw_ids in drop_message.items()}


def _extract_tool_function_name(tool: Dict[str, Any]) -> str | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str) and len(name.strip()) > 0:
            return name.strip()
    name = tool.get("name")
    if isinstance(name, str) and len(name.strip()) > 0:
        return name.strip()
    return None


def _tool_choice_mode(tool_choice: str | Dict[str, Any] | None) -> str:
    if isinstance(tool_choice, dict):
        return "function"
    if isinstance(tool_choice, str):
        lowered = tool_choice.strip().lower()
        if lowered in {"auto", "required", "none"}:
            return lowered
        return "function"
    return "auto"


def _tool_choice_forced_name(tool_choice: str | Dict[str, Any] | None) -> str | None:
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and len(name.strip()) > 0:
                return name.strip()
    if isinstance(tool_choice, str):
        text = tool_choice.strip()
        if len(text) == 0 or text.lower() in {"auto", "required", "none"}:
            return None
        return text
    return None


def _normalize_tool_choice(
    tools: List[Dict[str, Any]] | None,
    tool_choice: str | Dict[str, Any] | None,
) -> str | Dict[str, Any]:
    available_tools = tools or []
    available_names = {
        name for name in (_extract_tool_function_name(tool) for tool in available_tools) if name is not None
    }

    if tool_choice is None:
        return "auto" if len(available_tools) > 0 else "none"

    normalized: str | Dict[str, Any]
    if isinstance(tool_choice, str):
        raw = tool_choice.strip()
        lowered = raw.lower()
        if lowered in {"auto", "required", "none"}:
            normalized = lowered
        elif len(raw) > 0:
            normalized = {"type": "function", "function": {"name": raw}}
        else:
            normalized = "auto" if len(available_tools) > 0 else "none"
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise HTTPException(status_code=400, detail="tool_choice.type must be 'function'.")
        fn = tool_choice.get("function")
        if not isinstance(fn, dict):
            raise HTTPException(status_code=400, detail="tool_choice.function must be an object.")
        name = fn.get("name")
        if not isinstance(name, str) or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="tool_choice.function.name must be a non-empty string.")
        normalized = {"type": "function", "function": {"name": name.strip()}}
    else:
        raise HTTPException(status_code=400, detail="tool_choice must be a string or an object.")

    mode = _tool_choice_mode(normalized)
    if mode == "required" and len(available_tools) == 0:
        raise HTTPException(status_code=400, detail="tools cannot be empty when tool_choice is 'required'.")
    if mode == "function" and len(available_tools) == 0:
        raise HTTPException(
            status_code=400,
            detail="tools cannot be empty when tool_choice targets a specific function.",
        )
    forced_name = _tool_choice_forced_name(normalized)
    if forced_name is not None and forced_name not in available_names:
        raise HTTPException(
            status_code=400,
            detail=f"tool_choice function '{forced_name}' is not present in tools.",
        )
    return normalized


def _dedup_keep_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if len(value) == 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _infer_tool_stop_strings(model_path: str) -> List[str]:
    model = model_path.lower()
    # Borrowed from sglang function-call detector conventions.
    inferred: List[str] = ["</tool_call>"]
    if "qwen" in model:
        inferred.extend(["\n</tool_call>", "<|im_end|>"])
    if "glm" in model:
        inferred.append("</tool_call>")
    if "deepseek" in model:
        inferred.extend(["<｜tool▁calls▁end｜>", "<｜tool_calls_end｜>"])
    if "kimi" in model or "k2" in model:
        inferred.append("<|tool_calls_section_end|>")
    if "step" in model:
        inferred.append("<｜tool_calls_end｜>")
    return _dedup_keep_order(inferred)


def _build_effective_stop(
    user_stop: List[str] | None,
    tools: List[Dict[str, Any]] | None,
    tool_choice: str | Dict[str, Any] | None,
    model_path: str,
) -> List[str]:
    merged = list(user_stop or [])
    if tools and _tool_choice_mode(tool_choice) != "none":
        merged.extend(_infer_tool_stop_strings(model_path))
    return _dedup_keep_order(merged)


_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*"
    r"<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*"
    r"<\|tool_call_end\|>",
    re.DOTALL,
)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _format_tool_arguments(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _build_tool_call(name: str, arguments: Any, idx: int, call_id: str | None = None) -> Dict[str, Any]:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "index": idx,
        "function": {
            "name": name,
            "arguments": _format_tool_arguments(arguments),
        },
    }


def _coerce_tool_call_item(item: Any, idx: int) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    call_id = item.get("id")
    if isinstance(item.get("function"), dict):
        function = item["function"]
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = item.get("name")
        arguments = item.get("arguments")
    if not isinstance(name, str) or len(name) == 0:
        return None
    return _build_tool_call(name=name, arguments=arguments, idx=idx, call_id=call_id)


def _extract_json_objects(blob: str) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(blob):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = blob[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    parsed.append(obj)
    return parsed


def _parse_tool_calls_from_text(
    text: str,
    forced_tool_name: str | None = None,
) -> tuple[str, List[Dict[str, Any]] | None]:
    working_text = text
    tool_calls: List[Dict[str, Any]] = []

    def _append_candidates(payload: Any) -> None:
        if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            for item in payload["tool_calls"]:
                call = _coerce_tool_call_item(item, len(tool_calls))
                if call is not None:
                    tool_calls.append(call)
            return
        if isinstance(payload, dict):
            call = _coerce_tool_call_item(payload, len(tool_calls))
            if call is not None:
                tool_calls.append(call)
            return
        if isinstance(payload, list):
            for item in payload:
                call = _coerce_tool_call_item(item, len(tool_calls))
                if call is not None:
                    tool_calls.append(call)

    # Qwen / GLM style: <tool_call>...</tool_call>
    blocks = list(_TOOL_BLOCK_RE.finditer(working_text))
    for block in blocks:
        payload = block.group(1).strip()
        candidates: List[Any] = []
        try:
            parsed = json.loads(payload)
            candidates = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            # Fallback for very simple XML-ish GLM style: first line is name, rest are args
            lines = [line for line in payload.splitlines() if len(line.strip()) > 0]
            if len(lines) > 0:
                candidates = [{"name": lines[0].strip(), "arguments": "\n".join(lines[1:]).strip()}]

        _append_candidates(candidates)
    if len(blocks) > 0:
        working_text = _TOOL_BLOCK_RE.sub("", working_text).strip()

    # Kimi-K2 style sections.
    kimi_found = False
    for match in _KIMI_TOOL_CALL_RE.finditer(text):
        kimi_found = True
        raw_id = match.group("tool_call_id")
        raw_args = match.group("function_arguments")
        name = raw_id.split(":", 1)[0]
        if name.startswith("functions."):
            name = name[len("functions.") :]
        try:
            args = json.loads(raw_args)
        except Exception:
            args = raw_args
        _append_candidates(
            {
                "id": raw_id,
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        )
    if kimi_found:
        working_text = (
            working_text.replace("<|tool_calls_section_begin|>", "")
            .replace("<|tool_calls_section_end|>", "")
            .strip()
        )

    # DeepSeek / STEP style wrappers where JSON objects are embedded in a section.
    wrappers = [
        ("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
        ("<｜tool_calls_begin｜>", "<｜tool_calls_end｜>"),
    ]
    for begin, end in wrappers:
        if begin not in working_text or end not in working_text:
            continue
        section = working_text.split(begin, 1)[1].split(end, 1)[0]
        for obj in _extract_json_objects(section):
            _append_candidates(obj)
        working_text = working_text.replace(begin + section + end, "").strip()

    # Markdown fenced JSON block fallback.
    fenced_blocks = list(_JSON_FENCE_RE.finditer(working_text))
    for match in fenced_blocks:
        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        _append_candidates(parsed)
    if len(fenced_blocks) > 0:
        working_text = _JSON_FENCE_RE.sub("", working_text).strip()

    # Generic JSON fallback: entire output is a function call object/array.
    if len(tool_calls) == 0:
        stripped = working_text.strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        _append_candidates(parsed)
        if len(tool_calls) > 0:
            working_text = ""

    if forced_tool_name is not None and len(tool_calls) > 0:
        filtered: List[Dict[str, Any]] = []
        for call in tool_calls:
            fn = call.get("function")
            if isinstance(fn, dict) and fn.get("name") == forced_tool_name:
                filtered.append(call)
        tool_calls = filtered

    return working_text, (tool_calls if len(tool_calls) > 0 else None)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = Field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False
    drop_message: Dict[int, List[int]] | None = None
    enable_thinking: bool | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: str | Dict[str, Any] | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply | WarmupReply | RequestErrorReply]] = field(
        default_factory=dict
    )
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        timeout = self.config.request_timeout

        try:
            while True:
                try:
                    if timeout > 0:
                        await asyncio.wait_for(event.wait(), timeout=timeout)
                    else:
                        await event.wait()
                except asyncio.TimeoutError as exc:
                    logger.error("Timed out waiting for request %s after %.1fs", uid, timeout)
                    try:
                        await self.send_one(AbortMsg(uid=uid))
                    except Exception:
                        logger.exception("Failed to send timeout abort for request %s", uid)
                    raise RequestRejected(
                        status_code=504,
                        error_code="backend_timeout",
                        detail=f"No backend reply was received for {timeout:.1f} seconds.",
                    ) from exc
                event.clear()

                pending = self.ack_map.get(uid, [])
                self.ack_map[uid] = []
                for ack in pending:
                    if isinstance(ack, RequestErrorReply):
                        raise RequestRejected(
                            status_code=ack.status_code,
                            error_code=ack.error_code,
                            detail=ack.detail,
                        )
                    if ack.finished:
                        # Clean synchronously before exposing the terminal ack.
                        # Callers often return/break immediately after receiving it.
                        self.ack_map.pop(uid, None)
                        self.event_map.pop(uid, None)
                        yield ack
                        return
                    yield ack
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def wait_for_warmup(self, uid: int) -> WarmupReply:
        async for ack in self.wait_for_ack(uid):
            if isinstance(ack, WarmupReply):
                return ack
        raise RuntimeError("Warmup finished without WarmupReply")

    async def run_contextual_warmup(
        self,
        messages: List[Dict[str, Any]],
        drop_message: Dict[str, List[int]],
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
        tool_choice: str | Dict[str, Any] | None,
    ) -> None:
        # Event n changes visibility only for messages after n. A future event
        # therefore needs no special warmup for the current generation prompt.
        if not any(int(raw_n) < len(messages) for raw_n in drop_message):
            return
        warmup_target = max(len(messages) - 1, 0)
        warmup_uid = self.new_user()
        use_context_mask = self.config.contextual_prefill_mode != "staged"
        await self.send_one(
            TokenizeMsg(
                uid=warmup_uid,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
                target_msg_id=warmup_target,
                drop_message=drop_message,
                enable_thinking=enable_thinking,
                tools=tools,
                tool_choice=tool_choice,
                is_warmup=True,
                internal_uid=warmup_uid,
                use_context_mask=use_context_mask,
            )
        )
        warmup_ack = await self.wait_for_warmup(warmup_uid)
        if use_context_mask:
            return
        if warmup_ack.hit_ratio >= 0.95:
            return

        # Fallback: staged prefill by message prefixes.
        for end in range(1, len(messages)):
            staged_uid = self.new_user()
            await self.send_one(
                TokenizeMsg(
                    uid=staged_uid,
                    text=messages[:end],
                    sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
                    target_msg_id=end - 1,
                    drop_message=drop_message,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    tool_choice=tool_choice,
                    is_warmup=True,
                    internal_uid=staged_uid,
                )
            )
            await self.wait_for_warmup(staged_uid)

    async def stream_generate(self, uid: int):
        try:
            async for ack in self.wait_for_ack(uid):
                if not isinstance(ack, UserReply):
                    continue
                yield f"data: {ack.incremental_output}\n".encode()
                if ack.finished:
                    break
        except RequestRejected as exc:
            yield _stream_error_event(exc)
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(
        self,
        uid: int,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] | None = None,
    ):
        first_chunk = True
        buffered_output = ""
        final_finish_reason = "stop"
        matched_stop: str | None = None
        mode = _tool_choice_mode(tool_choice)
        forced_tool_name = _tool_choice_forced_name(tool_choice)
        require_tool_call = mode in {"required", "function"}
        buffer_mode = tools is not None and len(tools) > 0 and mode != "none"
        tool_call_missing = False
        try:
            async for ack in self.wait_for_ack(uid):
                if not isinstance(ack, UserReply):
                    continue
                if ack.finish_reason is not None:
                    final_finish_reason = ack.finish_reason
                if ack.matched_stop is not None:
                    matched_stop = ack.matched_stop

                if buffer_mode:
                    if ack.incremental_output:
                        buffered_output += ack.incremental_output
                    if ack.finished:
                        break
                    continue

                delta = {}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                if ack.incremental_output:
                    delta["content"] = ack.incremental_output

                chunk = {
                    "id": f"cmpl-{uid}",
                    "object": "text_completion.chunk",
                    "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

                if ack.finished:
                    break
        except RequestRejected as exc:
            yield _stream_error_event(exc)
            yield b"data: [DONE]\n\n"
            return

        if buffer_mode:
            clean_text, tool_calls = _parse_tool_calls_from_text(
                buffered_output, forced_tool_name=forced_tool_name
            )
            if tool_calls is not None and final_finish_reason == "stop":
                final_finish_reason = "tool_calls"
            if require_tool_call and tool_calls is None:
                final_finish_reason = "tool_call_missing"
                tool_call_missing = True

            delta: Dict[str, Any] = {"role": "assistant"}
            if tool_calls is not None:
                delta["tool_calls"] = tool_calls
            elif len(clean_text) > 0:
                delta["content"] = clean_text
            if tool_call_missing:
                delta["tool_call_missing"] = True

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": final_finish_reason,
                    "matched_stop": matched_stop,
                    "tool_call_missing": tool_call_missing,
                }
            ],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.drop_message is not None:
        _validate_drop_message(
            req.drop_message,
            radix_drop_key_mode=state.config.radix_drop_key_mode,
        )
    wire_drop_message = _to_wire_drop_message(req.drop_message)
    normalized_tool_choice: str | Dict[str, Any] = "none"
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
        normalized_tool_choice = _normalize_tool_choice(req.tools, req.tool_choice)
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt
        if req.drop_message is not None:
            raise HTTPException(
                status_code=400, detail="drop_message is only supported with chat `messages` input."
            )
        if req.tools is not None:
            raise HTTPException(status_code=400, detail="tools are only supported with chat `messages`.")
        if req.tool_choice is not None:
            raise HTTPException(
                status_code=400,
                detail="tool_choice is only supported with chat `messages` and `tools`.",
            )

    if req.drop_message is not None and isinstance(prompt, list):
        _validate_drop_message(
            req.drop_message,
            radix_drop_key_mode=state.config.radix_drop_key_mode,
            message_count=len(prompt),
        )

    effective_stop = _build_effective_stop(
        req.stop,
        req.tools,
        normalized_tool_choice,
        state.config.model_path,
    )

    if wire_drop_message is not None and isinstance(prompt, list):
        try:
            await state.run_contextual_warmup(
                prompt,
                wire_drop_message,
                req.enable_thinking,
                req.tools,
                normalized_tool_choice,
            )
        except RequestRejected as exc:
            raise _http_error(exc) from exc

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
            target_msg_id=(len(prompt) if isinstance(prompt, list) else None),
            drop_message=wire_drop_message,
            enable_thinking=req.enable_thinking,
            tools=req.tools,
            tool_choice=normalized_tool_choice,
            stop=effective_stop,
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(
                state.stream_chat_completions(
                    uid,
                    tools=req.tools,
                    tool_choice=normalized_tool_choice,
                ),
                request,
                uid,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    try:
        async for ack in state.wait_for_ack(uid):
            if not isinstance(ack, UserReply):
                continue
            full_content += ack.incremental_output
            if ack.finished:
                break
    except RequestRejected as exc:
        raise _http_error(exc) from exc

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    if req.drop_message is not None:
        _validate_drop_message(
            req.drop_message,
            radix_drop_key_mode=state.config.radix_drop_key_mode,
        )
    wire_drop_message = _to_wire_drop_message(req.drop_message)
    prompt = [msg.model_dump() for msg in req.messages]
    if req.drop_message is not None:
        _validate_drop_message(
            req.drop_message,
            radix_drop_key_mode=state.config.radix_drop_key_mode,
            message_count=len(prompt),
        )
    normalized_tool_choice = _normalize_tool_choice(req.tools, req.tool_choice)
    effective_stop = _build_effective_stop(
        req.stop,
        req.tools,
        normalized_tool_choice,
        state.config.model_path,
    )

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
            target_msg_id=len(prompt),
            drop_message=wire_drop_message,
            enable_thinking=req.enable_thinking,
            tools=req.tools,
            tool_choice=normalized_tool_choice,
            stop=effective_stop,
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
