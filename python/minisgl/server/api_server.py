from __future__ import annotations

import asyncio
import json
import time
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
from minisgl.tokenizer.drop_rules import (
    KeepTextDropRule,
    MessageDropRule,
    TextDropRule,
    ThinkingDropRule,
    parse_drop_rule,
    project_drop_rule_for_prefix,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs
from .response_parser import ChatResponseParser, infer_tool_call_parser

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
    radix_drop_key_mode: str = "delta-marker",
    message_count: int | None = None,
) -> None:
    if drop_message and radix_drop_key_mode != "delta-marker":
        raise HTTPException(
            status_code=400,
            detail=(
                "Drop Message now compiles to token-position ranges and requires "
                "radix_drop_key_mode='delta-marker'."
            ),
        )
    for raw_n, raw_ids in drop_message.items():
        n = int(raw_n)
        if n < 0:
            raise HTTPException(
                status_code=400, detail=f"drop_message key must be non-negative: {n}"
            )
        if radix_drop_key_mode == "bitmask" and n >= 32:
            raise HTTPException(
                status_code=400, detail=f"drop_message key out of range [0, 31]: {n}"
            )
        if radix_drop_key_mode in {"symbol", "delta-marker"} and n >= (1 << 63):
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
            if radix_drop_key_mode in {"symbol", "delta-marker"} and msg_id >= (1 << 63):
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
    return {
        str(int(raw_n)): [int(raw_id) for raw_id in raw_ids]
        for raw_n, raw_ids in drop_message.items()
    }


def _parse_request_drop_rule(
    *,
    drop_rule: Dict[str, Any] | None,
    legacy_drop_message: Dict[int, List[int]] | None,
    messages: List[Dict[str, Any]],
    radix_drop_key_mode: str,
) -> tuple[dict[str, Any] | None, List[Dict[str, Any]]]:
    try:
        parsed = parse_drop_rule(
            drop_rule,
            messages,
            legacy_drop_message=legacy_drop_message,
        )
    except ValueError as exc:
        logger.warning("Rejected invalid Drop rule: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(parsed, KeepTextDropRule):
        if parsed.use_visible_as_full:
            logger.warning(
                "keep_text_drop.force fallback: using visible messages as the full prompt: %s",
                parsed.fallback_reason,
            )
            return None, messages
        effective_messages = [dict(message) for message in parsed.full_messages]
    else:
        effective_messages = messages
    if parsed is not None and radix_drop_key_mode != "delta-marker":
        raise HTTPException(
            status_code=400,
            detail=(
                "Drop rules compile to token-position ranges and require "
                "radix_drop_key_mode='delta-marker'."
            ),
        )
    return (parsed.to_wire() if parsed is not None else None), effective_messages


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
        name
        for name in (_extract_tool_function_name(tool) for tool in available_tools)
        if name is not None
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
            raise HTTPException(
                status_code=400, detail="tool_choice.function.name must be a non-empty string."
            )
        normalized = {"type": "function", "function": {"name": name.strip()}}
    else:
        raise HTTPException(status_code=400, detail="tool_choice must be a string or an object.")

    mode = _tool_choice_mode(normalized)
    if mode == "required" and len(available_tools) == 0:
        raise HTTPException(
            status_code=400, detail="tools cannot be empty when tool_choice is 'required'."
        )
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


def _validate_tools(tools: List[Dict[str, Any]] | None) -> None:
    for index, tool in enumerate(tools or []):
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise HTTPException(
                status_code=400,
                detail=f"tools[{index}] must be an OpenAI function tool.",
            )
        function = tool.get("function")
        if not isinstance(function, dict):
            raise HTTPException(
                status_code=400,
                detail=f"tools[{index}].function must be an object.",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(
                status_code=400,
                detail=f"tools[{index}].function.name must be a non-empty string.",
            )
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise HTTPException(
                status_code=400,
                detail=f"tools[{index}].function.parameters must be a JSON Schema object.",
            )


def _normalize_stop(stop: str | List[str] | None) -> List[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _tools_for_choice(
    tools: List[Dict[str, Any]] | None,
    tool_choice: str | Dict[str, Any] | None,
) -> List[Dict[str, Any]] | None:
    forced_name = _tool_choice_forced_name(tool_choice)
    if not tools or forced_name is None:
        return tools
    return [tool for tool in tools if _extract_tool_function_name(tool) == forced_name]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | List[Dict[str, Any]] | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = Field(default=16, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | List[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False
    drop_message: Dict[int, List[int]] | None = None
    drop_rule: Dict[str, Any] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: str | Dict[str, Any] | None = None
    parallel_tool_calls: bool = True
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    seed: int | None = None


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
    stopped: bool = False
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
        drop_rule: Dict[str, Any],
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        tools: List[Dict[str, Any]] | None,
        tool_choice: str | Dict[str, Any] | None,
    ) -> int | None:
        # Event n changes visibility only for messages after n. A future event
        # therefore needs no special warmup for the current generation prompt.
        parsed_rule = parse_drop_rule(drop_rule, messages, allow_internal=True)
        if isinstance(parsed_rule, MessageDropRule):
            has_current_drop = any(
                trigger < len(messages) and bool(message_ids)
                for trigger, message_ids in parsed_rule.drop_messages.items()
            )
        elif isinstance(parsed_rule, TextDropRule):
            has_current_drop = any(selection is not None for selection in parsed_rule.selections)
        elif isinstance(parsed_rule, KeepTextDropRule):
            has_current_drop = parsed_rule.has_drop()
        else:
            has_current_drop = isinstance(parsed_rule, ThinkingDropRule) and bool(
                parsed_rule.thinking_by_message
            )
        if not has_current_drop:
            return None
        use_context_mask = self.config.contextual_prefill_mode == "mask"
        warmup_target = len(messages) if use_context_mask else max(len(messages) - 1, 0)
        warmup_uid = self.new_user()
        await self.send_one(
            TokenizeMsg(
                uid=warmup_uid,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
                target_msg_id=warmup_target,
                drop_rule=drop_rule,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                tools=tools,
                tool_choice=tool_choice,
                is_warmup=True,
                internal_uid=warmup_uid,
                use_context_mask=use_context_mask,
            )
        )
        warmup_ack = await self.wait_for_warmup(warmup_uid)
        if use_context_mask:
            return warmup_ack.cached_tokens
        if warmup_ack.hit_ratio >= 0.95:
            return warmup_ack.cached_tokens

        # Fallback: staged prefill by message prefixes.
        for end in range(1, len(messages)):
            staged_uid = self.new_user()
            staged_rule = project_drop_rule_for_prefix(drop_rule, end)
            if staged_rule is not None and staged_rule.get("type") == "thinking_drop":
                try:
                    parse_drop_rule(staged_rule, messages[:end], allow_internal=True)
                except ValueError:
                    staged_rule = None
            await self.send_one(
                TokenizeMsg(
                    uid=staged_uid,
                    text=messages[:end],
                    sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
                    target_msg_id=end - 1,
                    drop_rule=staged_rule,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                    tools=tools,
                    tool_choice=tool_choice,
                    is_warmup=True,
                    internal_uid=staged_uid,
                )
            )
            await self.wait_for_warmup(staged_uid)
        return warmup_ack.cached_tokens

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
        enable_thinking: bool | None = None,
        separate_reasoning: bool = True,
        stream_reasoning: bool = True,
        cache_report_cached_tokens: int | None = None,
    ):
        final_finish_reason = "stop"
        matched_stop: str | None = None
        cache_hit_ratio: float | None = None
        cached_tokens: int | None = cache_report_cached_tokens
        prompt_tokens = 0
        parser = ChatResponseParser(
            model_path=self.config.model_path,
            tools=tools if _tool_choice_mode(tool_choice) != "none" else None,
            tool_call_parser=self.config.tool_call_parser,
            reasoning_parser=self.config.reasoning_parser,
            enable_thinking=enable_thinking,
            separate_reasoning=separate_reasoning,
            stream_reasoning=stream_reasoning,
        )

        role_chunk = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [
                {"delta": {"role": "assistant", "content": ""}, "index": 0, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n".encode()

        def encode_piece(piece) -> bytes | None:
            delta: Dict[str, Any] = {}
            if piece.reasoning_content:
                delta["reasoning_content"] = piece.reasoning_content
            if piece.content:
                delta["content"] = piece.content
            if piece.tool_calls:
                delta["tool_calls"] = piece.tool_calls
            if not delta:
                return None
            chunk = {
                "id": f"chatcmpl-{uid}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            return f"data: {json.dumps(chunk)}\n\n".encode()

        try:
            async for ack in self.wait_for_ack(uid):
                if not isinstance(ack, UserReply):
                    continue
                if ack.finish_reason is not None:
                    final_finish_reason = ack.finish_reason
                if ack.matched_stop is not None:
                    matched_stop = ack.matched_stop
                if ack.cache_hit_ratio is not None:
                    cache_hit_ratio = ack.cache_hit_ratio
                if cached_tokens is None and ack.cached_tokens is not None:
                    cached_tokens = ack.cached_tokens
                if ack.prompt_tokens is not None:
                    prompt_tokens = ack.prompt_tokens
                if ack.incremental_output:
                    encoded = encode_piece(parser.feed(ack.incremental_output))
                    if encoded is not None:
                        yield encoded

                if ack.finished:
                    break
        except RequestRejected as exc:
            yield _stream_error_event(exc)
            yield b"data: [DONE]\n\n"
            return

        tail = parser.finish()
        encoded = encode_piece(tail)
        if encoded is not None:
            yield encoded
        if parser.has_tool_calls and final_finish_reason == "stop":
            final_finish_reason = "tool_calls"

        # send final finish_reason
        end_chunk = {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [
                {
                    "delta": {},
                    "index": 0,
                    "finish_reason": final_finish_reason,
                    "matched_stop": matched_stop,
                }
            ],
        }
        if cached_tokens is not None:
            end_chunk["cached_tokens"] = cached_tokens
            cache_hit_ratio = (
                1.0 if prompt_tokens == 0 else cached_tokens / prompt_tokens
            )
        if cache_hit_ratio is not None:
            end_chunk["cache_hit_ratio"] = cache_hit_ratio
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
        if self.stopped:
            return
        self.stopped = True
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
    wire_drop_rule: dict[str, Any] | None = None
    normalized_tool_choice: str | Dict[str, Any] = "none"
    _validate_tools(req.tools)
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
        normalized_tool_choice = _normalize_tool_choice(req.tools, req.tool_choice)
        if (
            req.tools
            and _tool_choice_mode(normalized_tool_choice) != "none"
            and infer_tool_call_parser(
                state.config.model_path, state.config.tool_call_parser
            )
            is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "No tool-call parser can be inferred for this model; set "
                    "--tool-call-parser explicitly."
                ),
            )
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt
        if req.drop_message is not None or req.drop_rule is not None:
            raise HTTPException(
                status_code=400, detail="drop_rule is only supported with chat `messages` input."
            )
        if req.tools is not None:
            raise HTTPException(
                status_code=400, detail="tools are only supported with chat `messages`."
            )
        if req.tool_choice is not None:
            raise HTTPException(
                status_code=400,
                detail="tool_choice is only supported with chat `messages` and `tools`.",
            )

    if isinstance(prompt, list):
        wire_drop_rule, prompt = _parse_request_drop_rule(
            drop_rule=req.drop_rule,
            legacy_drop_message=req.drop_message,
            messages=prompt,
            radix_drop_key_mode=state.config.radix_drop_key_mode,
        )

    effective_stop = _normalize_stop(req.stop)
    max_tokens = (
        req.max_completion_tokens
        if req.max_completion_tokens is not None
        else req.max_tokens
    )

    cache_report_cached_tokens: int | None = None
    if wire_drop_rule is not None and isinstance(prompt, list):
        try:
            cache_report_cached_tokens = await state.run_contextual_warmup(
                prompt,
                wire_drop_rule,
                req.enable_thinking,
                req.reasoning_effort,
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
                max_tokens=max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
            ),
            target_msg_id=(len(prompt) if isinstance(prompt, list) else None),
            drop_rule=wire_drop_rule,
            enable_thinking=req.enable_thinking,
            reasoning_effort=req.reasoning_effort,
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
                    tools=_tools_for_choice(req.tools, normalized_tool_choice),
                    tool_choice=normalized_tool_choice,
                    enable_thinking=req.enable_thinking,
                    separate_reasoning=req.separate_reasoning,
                    stream_reasoning=req.stream_reasoning,
                    cache_report_cached_tokens=cache_report_cached_tokens,
                ),
                request,
                uid,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    finish_reason = "stop"
    cache_hit_ratio: float | None = None
    cached_tokens: int | None = cache_report_cached_tokens
    prompt_tokens = 0
    completion_tokens = 0
    try:
        async for ack in state.wait_for_ack(uid):
            if not isinstance(ack, UserReply):
                continue
            full_content += ack.incremental_output
            if ack.finish_reason is not None:
                finish_reason = ack.finish_reason
            if ack.cache_hit_ratio is not None:
                cache_hit_ratio = ack.cache_hit_ratio
            if cached_tokens is None and ack.cached_tokens is not None:
                cached_tokens = ack.cached_tokens
            if ack.prompt_tokens is not None:
                prompt_tokens = ack.prompt_tokens
            if ack.completion_tokens is not None:
                completion_tokens = ack.completion_tokens
            if ack.finished:
                break
    except RequestRejected as exc:
        raise _http_error(exc) from exc

    parser = ChatResponseParser(
        model_path=state.config.model_path,
        tools=(
            _tools_for_choice(req.tools, normalized_tool_choice)
            if _tool_choice_mode(normalized_tool_choice) != "none"
            else None
        ),
        tool_call_parser=state.config.tool_call_parser,
        reasoning_parser=state.config.reasoning_parser,
        enable_thinking=req.enable_thinking,
        separate_reasoning=req.separate_reasoning,
    )
    parsed = parser.parse_full(full_content)
    response_message: Dict[str, Any] = {"role": "assistant", "content": parsed.content}
    if parsed.reasoning_content:
        response_message["reasoning_content"] = parsed.reasoning_content
    if parsed.tool_calls is not None:
        response_message["tool_calls"] = parsed.tool_calls
        if finish_reason == "stop":
            finish_reason = "tool_calls"

    prompt_tokens_details = (
        {"cached_tokens": cached_tokens}
        if cached_tokens is not None and cached_tokens > 0
        else None
    )
    response = {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": prompt_tokens_details,
        },
    }
    if cached_tokens is not None:
        cache_hit_ratio = 1.0 if prompt_tokens == 0 else cached_tokens / prompt_tokens
    if cache_hit_ratio is not None:
        response["cache_hit_ratio"] = cache_hit_ratio
    return response


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]
    wire_drop_rule, prompt = _parse_request_drop_rule(
        drop_rule=req.drop_rule,
        legacy_drop_message=req.drop_message,
        messages=prompt,
        radix_drop_key_mode=state.config.radix_drop_key_mode,
    )
    normalized_tool_choice = _normalize_tool_choice(req.tools, req.tool_choice)
    effective_stop = _normalize_stop(req.stop)

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=(
                    req.max_completion_tokens
                    if req.max_completion_tokens is not None
                    else req.max_tokens
                ),
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
            ),
            target_msg_id=len(prompt),
            drop_rule=wire_drop_rule,
            enable_thinking=req.enable_thinking,
            reasoning_effort=req.reasoning_effort,
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


def run_api_server(
    config: ServerArgs,
    start_backend: Callable[[], Callable[[], None]],
    run_shell: bool,
) -> None:
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

    stop_backend: Callable[[], None] | None = None
    try:
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

        stop_backend = start_backend()
        logger.info(f"API server is ready to serve on {host}:{port}")
        if not run_shell:
            uvicorn.run(app, host=host, port=port)
        else:
            asyncio.run(shell())
    finally:
        try:
            if _GLOBAL_STATE is not None:
                _GLOBAL_STATE.shutdown()
        finally:
            if stop_backend is not None:
                stop_backend()
            _GLOBAL_STATE = None
