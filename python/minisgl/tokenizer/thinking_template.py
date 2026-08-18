from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment


_PROBE_SENTINEL = "MINISGL_PRIVATE_THINKING_8e76bd9f"
_QWEN_HISTORY_GUARD = re.compile(
    r"({%-?\s*if\s+)loop\.index0\s*>\s*ns\.last_query_index(\s*-?%})"
)
_CAPABILITY_CACHE: dict[tuple[Any, ...], "ThinkingTemplatePlan"] = {}


@dataclass(frozen=True)
class ThinkingTemplatePlan:
    chat_template: str | None
    template_kwargs: dict[str, Any]
    capability: str
    fingerprint: str


def _template_fingerprint(chat_template: str) -> str:
    return hashlib.sha256(chat_template.encode("utf-8")).hexdigest()


def _probe_messages(*, with_reasoning: bool) -> list[dict[str, Any]]:
    assistant: dict[str, Any] = {"role": "assistant", "content": "PROBE_FINAL"}
    if with_reasoning:
        assistant["reasoning_content"] = _PROBE_SENTINEL
    return [
        {"role": "user", "content": "PROBE_USER_ONE"},
        assistant,
        {"role": "user", "content": "PROBE_USER_TWO"},
    ]


def _render_probe(
    tokenizer,
    *,
    tools: list[dict[str, Any]] | None,
    chat_template: str | None,
    preserve: bool,
    with_reasoning: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    if preserve:
        kwargs["preserve_thinking_history"] = True
    rendered = tokenizer.apply_chat_template(
        _probe_messages(with_reasoning=with_reasoning), **kwargs
    )
    if not isinstance(rendered, str):
        raise ValueError("thinking_history_not_preservable: tokenizer probe did not render text")
    return rendered


def _patch_qwen_history_guard(chat_template: str) -> str:
    # Parse first so malformed or non-Jinja templates fail closed.
    Environment().parse(chat_template)
    if "reasoning_content" not in chat_template or "ns.last_query_index" not in chat_template:
        raise ValueError(
            "thinking_history_not_preservable: template has no recognized structured-thinking guard"
        )
    patched, count = _QWEN_HISTORY_GUARD.subn(
        r"\1preserve_thinking_history or loop.index0 > ns.last_query_index\2",
        chat_template,
    )
    if count != 1:
        raise ValueError(
            "thinking_history_not_preservable: expected exactly one Qwen history guard"
        )
    Environment().parse(patched)
    return patched


def prepare_thinking_template(
    tokenizer,
    *,
    tools: list[dict[str, Any]] | None,
) -> ThinkingTemplatePlan:
    """Lazily prove native retention or build a guarded request-local Qwen adapter."""

    chat_template = tokenizer.get_chat_template(tools=tools)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("thinking_history_not_preservable: tokenizer has no chat template")
    fingerprint = _template_fingerprint(chat_template)
    cache_key = (
        id(tokenizer),
        str(getattr(tokenizer, "name_or_path", "")),
        type(tokenizer).__name__,
        fingerprint,
        bool(tools),
    )
    cached = _CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    native = _render_probe(
        tokenizer,
        tools=tools,
        chat_template=None,
        preserve=False,
        with_reasoning=True,
    )
    if native.count(_PROBE_SENTINEL) == 1:
        plan = ThinkingTemplatePlan(None, {}, "native", fingerprint)
        _CAPABILITY_CACHE[cache_key] = plan
        return plan

    patched = _patch_qwen_history_guard(chat_template)
    preserved = _render_probe(
        tokenizer,
        tools=tools,
        chat_template=patched,
        preserve=True,
        with_reasoning=True,
    )
    if preserved.count(_PROBE_SENTINEL) != 1:
        raise ValueError(
            "thinking_history_not_preservable: adapted template did not retain "
            "reasoning exactly once"
        )

    baseline_without_reasoning = _render_probe(
        tokenizer,
        tools=tools,
        chat_template=None,
        preserve=False,
        with_reasoning=False,
    )
    adapted_without_reasoning = _render_probe(
        tokenizer,
        tools=tools,
        chat_template=patched,
        preserve=True,
        with_reasoning=False,
    )
    if adapted_without_reasoning != baseline_without_reasoning:
        raise ValueError(
            "thinking_history_not_preservable: adapter changes non-thinking template output"
        )

    plan = ThinkingTemplatePlan(
        patched,
        {"preserve_thinking_history": True},
        "qwen_guard_adapter",
        fingerprint,
    )
    _CAPABILITY_CACHE[cache_key] = plan
    return plan
