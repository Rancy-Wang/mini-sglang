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

    @staticmethod
    def _references_generation_prompt(node: nodes.Node) -> bool:
        if isinstance(node, nodes.Name) and node.name == "add_generation_prompt":
            return True
        return any(
            isinstance(candidate, nodes.Name)
            and candidate.name == "add_generation_prompt"
            for candidate in node.find_all(nodes.Name)
        )

    def _marker_call(self, phase: str, lineno: int) -> nodes.Call:
        loop_values = [nodes.Name(name, "load") for name in reversed(self._loop_vars)]
        return nodes.Call(
            nodes.Name("_minisgl_owner_marker", "load"),
            [nodes.Const(phase), *loop_values],
            [],
            None,
            None,
        ).set_lineno(lineno)

    def _marker_output(self, phase: str, lineno: int) -> nodes.Output:
        return nodes.Output([self._marker_call(phase, lineno)]).set_lineno(lineno)

    def visit_If(self, node: nodes.If, *args, **kwargs):
        traces_generation = self._references_generation_prompt(node.test)
        node = self.generic_visit(node, *args, **kwargs)
        if traces_generation:
            node.body = [
                self._marker_output("G", node.lineno),
                *node.body,
                self._marker_output("H", node.lineno),
            ]
        return node

    def visit_Output(self, node: nodes.Output, *args, **kwargs):
        traces_generation = self._references_generation_prompt(node)
        node = self.generic_visit(node, *args, **kwargs)
        traced_nodes = [
            self._marker_call("B", node.lineno),
            *node.nodes,
            self._marker_call("E", node.lineno),
        ]
        if traces_generation:
            traced_nodes = [
                self._marker_call("G", node.lineno),
                *traced_nodes,
                self._marker_call("H", node.lineno),
            ]
        node.nodes = traced_nodes
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
    pattern = re.compile(re.escape(marker_prefix) + r"([BEGH]):(-?\d+)\x00")
    return traced, marker_prefix, pattern


def _parse_character_owners(
    traced_text: str,
    marker_pattern: re.Pattern[str],
    *,
    message_count: int,
    add_generation_prompt: bool,
) -> tuple[str, list[int]]:
    clean_parts: list[str] = []
    owners: list[int] = []
    active: list[int] = []
    generation_depth = 0
    cursor = 0

    def active_owner() -> int:
        if generation_depth > 0:
            return message_count
        return next((owner for owner in reversed(active) if owner >= 0), -1)

    for match in marker_pattern.finditer(traced_text):
        chunk = traced_text[cursor : match.start()]
        clean_parts.append(chunk)
        owners.extend([active_owner()] * len(chunk))

        phase = match.group(1)
        marker_owner = int(match.group(2))
        if phase == "B":
            active.append(marker_owner)
        elif phase == "E":
            if not active:
                raise RuntimeError("Unbalanced chat-template provenance marker.")
            active.pop()
        elif phase == "G":
            generation_depth += 1
        else:
            if generation_depth == 0:
                raise RuntimeError("Unbalanced generation-prompt provenance marker.")
            generation_depth -= 1
        cursor = match.end()

    tail = traced_text[cursor:]
    clean_parts.append(tail)
    owners.extend([active_owner()] * len(tail))
    if active or generation_depth:
        raise RuntimeError("Unbalanced chat-template provenance marker.")

    clean_text = "".join(clean_parts)
    if len(owners) != len(clean_text):
        raise RuntimeError("Character ownership length does not match canonical chat text.")
    if len(owners) == 0:
        return clean_text, owners

    known = [idx for idx, owner in enumerate(owners) if owner >= 0]
    if not known:
        if message_count > 1:
            raise RuntimeError(
                "Chat template emitted multiple messages outside traceable message loops."
            )
        fallback = message_count if add_generation_prompt and message_count == 0 else 0
        owners = [fallback] * len(owners)
        return clean_text, owners

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

    if add_generation_prompt and message_count not in owners:
        last_known = known[-1]
        if last_known + 1 < len(owners):
            owners[last_known + 1 :] = [message_count] * (len(owners) - last_known - 1)
    return clean_text, owners


def build_template_token_provenance(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
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
    canonical_text, char_owners = _parse_character_owners(
        traced_text,
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
