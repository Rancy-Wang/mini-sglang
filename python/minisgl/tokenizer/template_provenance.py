from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from jinja2 import nodes
from jinja2.visitor import NodeTransformer
from transformers.utils.chat_template_utils import _compile_jinja_template


@dataclass(frozen=True)
class TemplateTokenProvenance:
    input_ids: list[int]
    owners: list[int]
    offsets: list[tuple[int, int]]
    rendered_text: str
    char_owners: list[int]
    cross_owner_tokens: int


class _TraceTemplateOutputs(NodeTransformer):
    """Wrap template output nodes with markers carrying active loop variables."""

    def __init__(self) -> None:
        self._loop_vars: list[str] = []

    @staticmethod
    def _target_names(target: nodes.Node) -> list[str]:
        if isinstance(target, nodes.Name):
            return [target.name]
        if isinstance(target, (nodes.List, nodes.Tuple)):
            names: list[str] = []
            for item in target.items:
                names.extend(_TraceTemplateOutputs._target_names(item))
            return names
        return []

    def visit_For(self, node: nodes.For, *args, **kwargs):
        names = self._target_names(node.target)
        self._loop_vars.extend(names)
        try:
            return self.generic_visit(node, *args, **kwargs)
        finally:
            if names:
                del self._loop_vars[-len(names) :]

    def visit_Output(self, node: nodes.Output, *args, **kwargs):
        node = self.generic_visit(node, *args, **kwargs)
        loop_values = [nodes.Name(name, "load") for name in reversed(self._loop_vars)]

        def marker_call(phase: str) -> nodes.Call:
            return nodes.Call(
                nodes.Name("_minisgl_owner_marker", "load"),
                [nodes.Const(phase), *loop_values],
                [],
                None,
                None,
            ).set_lineno(node.lineno)

        node.nodes = [marker_call("B"), *node.nodes, marker_call("E")]
        return node


@lru_cache(maxsize=32)
def _compile_traced_template(chat_template: str):
    compiled = _compile_jinja_template(chat_template)
    environment = compiled.environment
    traced_ast = _TraceTemplateOutputs().visit(environment.parse(chat_template))
    code = environment.compile(traced_ast)
    return environment.template_class.from_code(
        environment,
        code,
        environment.globals,
        None,
    )


def _render_traced_template(
    tokenizer,
    chat_template: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    enable_thinking: bool | None,
    extra_template_kwargs: dict[str, Any] | None,
) -> tuple[str, str, re.Pattern[str]]:
    nonce = uuid.uuid4().hex
    marker_prefix = f"\x00minisgl-owner-{nonce}:"
    owner_by_object = {id(message): msg_id for msg_id, message in enumerate(messages)}

    def owner_marker(phase: str, *loop_values: Any) -> str:
        owner = -1
        for value in loop_values:
            candidate = owner_by_object.get(id(value))
            if candidate is not None:
                owner = candidate
                break
        return f"{marker_prefix}{phase}:{owner}\x00"

    template_kwargs = dict(getattr(tokenizer, "special_tokens_map", {}))
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    template_kwargs.update(extra_template_kwargs or {})
    traced = _compile_traced_template(chat_template).render(
        messages=messages,
        tools=tools,
        documents=None,
        add_generation_prompt=add_generation_prompt,
        _minisgl_owner_marker=owner_marker,
        **template_kwargs,
    )
    pattern = re.compile(re.escape(marker_prefix) + r"([BE]):(-?\d+)\x00")
    return traced, marker_prefix, pattern


def _parse_character_owners(
    traced_text: str,
    canonical_text: str,
    canonical_no_generation_text: str,
    marker_pattern: re.Pattern[str],
    *,
    message_count: int,
    add_generation_prompt: bool,
) -> list[int]:
    clean_parts: list[str] = []
    owners: list[int] = []
    active: list[int] = []
    cursor = 0

    def active_owner() -> int:
        return next((owner for owner in reversed(active) if owner >= 0), -1)

    for match in marker_pattern.finditer(traced_text):
        chunk = traced_text[cursor : match.start()]
        clean_parts.append(chunk)
        owners.extend([active_owner()] * len(chunk))

        phase = match.group(1)
        marker_owner = int(match.group(2))
        if phase == "B":
            active.append(marker_owner)
        else:
            if not active:
                raise RuntimeError("Unbalanced chat-template provenance marker.")
            active.pop()
        cursor = match.end()

    tail = traced_text[cursor:]
    clean_parts.append(tail)
    owners.extend([active_owner()] * len(tail))
    if active:
        raise RuntimeError("Unbalanced chat-template provenance marker.")

    clean_text = "".join(clean_parts)
    if clean_text != canonical_text:
        raise RuntimeError(
            "Instrumented chat template changed canonical text; "
            "cannot construct reliable message ownership."
        )
    if len(owners) != len(canonical_text):
        raise RuntimeError("Character ownership length does not match canonical chat text.")
    if len(owners) == 0:
        return owners

    generation_start = None
    if add_generation_prompt and canonical_text.startswith(canonical_no_generation_text):
        generation_start = len(canonical_no_generation_text)

    known = [idx for idx, owner in enumerate(owners) if owner >= 0]
    if not known:
        if message_count > 1:
            raise RuntimeError(
                "Chat template emitted multiple messages outside traceable message loops."
            )
        fallback = message_count if add_generation_prompt and message_count == 0 else 0
        owners = [fallback] * len(owners)
        if generation_start is not None:
            owners[generation_start:] = [message_count] * (len(owners) - generation_start)
        return owners

    first_known = known[0]
    first_owner = owners[first_known]
    leading_owner = 0 if message_count > 0 else first_owner
    owners[:first_known] = [leading_owner] * first_known

    previous_owner = owners[first_known]
    for idx in range(first_known + 1, len(owners)):
        if owners[idx] < 0:
            owners[idx] = previous_owner
        else:
            previous_owner = owners[idx]

    if generation_start is not None:
        owners[generation_start:] = [message_count] * (len(owners) - generation_start)
    elif add_generation_prompt:
        last_known = known[-1]
        if last_known + 1 < len(owners):
            owners[last_known + 1 :] = [message_count] * (len(owners) - last_known - 1)
    return owners


def build_template_token_provenance(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    canonical_text: str,
    canonical_no_generation_text: str,
    expected_input_ids: list[int] | None,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    enable_thinking: bool | None,
    chat_template: str | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> TemplateTokenProvenance:
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError(
            "Drop Message ownership requires a fast tokenizer with offset_mapping support."
        )
    if chat_template is None:
        chat_template = tokenizer.get_chat_template(tools=tools)
    traced_text, _, marker_pattern = _render_traced_template(
        tokenizer,
        chat_template,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        extra_template_kwargs=template_kwargs,
    )
    char_owners = _parse_character_owners(
        traced_text,
        canonical_text,
        canonical_no_generation_text,
        marker_pattern,
        message_count=len(messages),
        add_generation_prompt=add_generation_prompt,
    )

    encoded = tokenizer(
        canonical_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    if expected_input_ids is not None and input_ids != [
        int(token_id) for token_id in expected_input_ids
    ]:
        raise RuntimeError(
            "Canonical chat text tokenization differs from apply_chat_template(tokenize=True)."
        )

    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    owners: list[int] = []
    cross_owner_tokens = 0
    previous_owner = 0
    for token_idx, (start, end) in enumerate(offsets):
        if start < 0 or end < start or end > len(char_owners):
            raise RuntimeError(f"Token {token_idx} has an invalid character offset.")
        if start == end:
            owner = previous_owner
        else:
            token_owners = char_owners[start:end]
            if any(owner < 0 for owner in token_owners):
                raise RuntimeError(f"Token {token_idx} covers an unowned template character.")
            owner = token_owners[0]
            if any(candidate != owner for candidate in token_owners[1:]):
                cross_owner_tokens += 1
        owners.append(owner)
        previous_owner = owner

    return TemplateTokenProvenance(
        input_ids=input_ids,
        owners=owners,
        offsets=offsets,
        rendered_text=canonical_text,
        char_owners=char_owners,
        cross_owner_tokens=cross_owner_tokens,
    )
