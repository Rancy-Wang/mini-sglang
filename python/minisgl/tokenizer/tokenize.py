from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, List

import torch
from minisgl.kernel.radix_reposition import (
    RadixRepositionLayout,
    compile_radix_reposition_layout,
)
from minisgl.kernel.text_match import find_all
from minisgl.message import TokenizeMsg
from minisgl.message.tokenizer import get_gpt_oss_terminal_stop_token_ids
from minisgl.tokenizer.drop_rules import (
    DropCompileContext,
    KeepTextDropRule,
    MessageDropRule,
    TextDropRule,
    ThinkingDropRule,
    TokenDropEvents,
    parse_drop_rule,
)
from minisgl.tokenizer.template_provenance import (
    TemplateTokenProvenance,
    build_template_token_provenance,
)
from minisgl.tokenizer.thinking_template import prepare_thinking_template
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenizedResult:
    input_ids: torch.Tensor
    true_positions: torch.Tensor
    raw_positions: torch.Tensor
    radix_input_ids: torch.Tensor
    radix_match_ids: torch.Tensor | None
    prefix_keep_mask: torch.Tensor
    prompt_tokens: int
    full_input_ids: torch.Tensor | None = None
    full_token_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    drop_event_positions: torch.Tensor | None = None
    drop_range_offsets: torch.Tensor | None = None
    drop_position_ranges: torch.Tensor | None = None
    drop_effective_event_count: int = 0
    reposition_raw_boundaries: torch.Tensor | None = None
    reposition_insert_offsets: torch.Tensor | None = None
    reposition_input_ids: torch.Tensor | None = None
    radix_commit_token_len: int | None = None
    radix_commit_key_len: int | None = None
    radix_key_virtual_mask: torch.Tensor | None = None
    radix_key_to_token: torch.Tensor | None = None
    radix_token_to_key: torch.Tensor | None = None
    radix_positions: torch.Tensor | None = None
    radix_repos_info: torch.Tensor | None = None
    radix_next_position: int | None = None
    radix_current_reposition: int = -1
    reposition_layout: RadixRepositionLayout | None = None
    stop_token_seqs: List[List[int]] | None = None
    message_meta: dict | None = None
    tokenize_invocations: int = 1


@dataclass(frozen=True)
class PositionDropPlan:
    event_positions: torch.Tensor
    range_offsets: torch.Tensor
    position_ranges: torch.Tensor
    full_token_visible_until: torch.Tensor
    effective_event_count: int


@dataclass(frozen=True)
class TokenRepositionEvents:
    raw_boundaries: torch.Tensor
    insert_offsets: torch.Tensor


def resolve_reposition_token_boundaries(
    reposition_ids: List[int] | None,
    owner_ranges: dict[int, List[tuple[int, int]]],
    public_to_normalized_owner: dict[int, int],
) -> TokenRepositionEvents:
    """Translate public message IDs into exact raw-token boundaries."""

    raw_ids = reposition_ids or []
    if any(isinstance(raw_id, bool) or not isinstance(raw_id, int) for raw_id in raw_ids):
        raise ValueError("reposition must contain integer message IDs.")
    if raw_ids != sorted(set(raw_ids)):
        raise ValueError("reposition must contain strictly increasing unique message IDs.")
    boundaries: list[int] = []
    for raw_id in raw_ids:
        owner = public_to_normalized_owner.get(raw_id)
        if owner is None:
            raise ValueError(f"reposition message ID {raw_id} is outside the conversation.")
        ranges = owner_ranges.get(owner)
        if not ranges:
            raise ValueError(f"Cannot map reposition message ID {raw_id} into the token stream.")
        boundaries.append(max(end for _, end in ranges) - 1)
    if boundaries != sorted(set(boundaries)):
        raise ValueError("Chat template ownership does not preserve Reposition boundary order.")
    raw_boundaries = torch.tensor(boundaries, dtype=torch.int32, device="cpu")
    return TokenRepositionEvents(raw_boundaries, raw_boundaries + 1)


@dataclass(frozen=True)
class _HarmonyComponentOwnership:
    owner: int
    sources: tuple[tuple[int, str], ...] = ()
    is_analysis: bool = False


@dataclass(frozen=True)
class _HarmonyPrompt:
    conversation: Any
    components: list[Any]
    ownership: list[_HarmonyComponentOwnership]
    thinking_components: dict[int, tuple[int, str]]
    has_function_tools: bool


class TokenizeManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        radix_drop_key_mode: str = "delta-marker",
    ) -> None:
        if radix_drop_key_mode not in {"bitmask", "symbol", "delta-marker"}:
            raise ValueError(f"Unsupported radix drop key mode: {radix_drop_key_mode}")
        self.tokenizer = tokenizer
        self.radix_drop_key_mode = radix_drop_key_mode
        tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
        tokenizer_class = type(tokenizer).__name__.lower()
        self.is_gpt_oss = "gpt-oss" in tokenizer_name or "gptoss" in tokenizer_class
        self._reasoning_effort: str | None = None
        self._harmony_encoding = None
        self._preserve_harmony_thinking = False
        self._harmony_thinking_ranges: dict[int, List[tuple[int, int]]] = {}
        self._chat_template_override: str | None = None
        self._chat_template_kwargs: dict[str, Any] = {}
        self._template_requires_bare_tools = False
        # Be optimistic: many tokenizers accept extra kwargs via **kwargs even
        # when the explicit signature does not list `enable_thinking`.
        self._supports_enable_thinking = True

    @staticmethod
    def _empty_context_events() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(0, dtype=torch.int32, device="cpu"),
            torch.zeros(1, dtype=torch.int32, device="cpu"),
            torch.empty(0, dtype=torch.int32, device="cpu"),
        )

    def _compile_delta_layout(
        self,
        token_ids: torch.Tensor,
        token_drop_events: TokenDropEvents | None,
        final_keep_mask: torch.Tensor,
        reposition_raw_boundaries: torch.Tensor | None,
        reposition_insert_offsets: torch.Tensor | None,
    ) -> RadixRepositionLayout | None:
        if self.radix_drop_key_mode != "delta-marker":
            return None
        if token_drop_events is None:
            event_positions, range_offsets, ranges = self._empty_context_events()
        else:
            event_positions, range_offsets, ranges = self._select_effective_delta_wire(
                token_drop_events,
                final_keep_mask,
            )
        empty = torch.empty(0, dtype=torch.int32, device="cpu")
        return compile_radix_reposition_layout(
            token_ids,
            event_positions,
            range_offsets,
            ranges,
            reposition_raw_boundaries if reposition_raw_boundaries is not None else empty,
            reposition_insert_offsets if reposition_insert_offsets is not None else empty,
        )

    @staticmethod
    def _select_effective_delta_wire(
        events: TokenDropEvents,
        final_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select target-effective events before the one CPU Radix compilation."""

        event_count = min(events.effective_event_count, len(events.event_insert_offsets))
        if event_count >= 0:
            range_count = int(events.range_offsets[event_count])
            return (
                events.event_insert_offsets[:event_count].contiguous(),
                events.range_offsets[: event_count + 1].contiguous(),
                events.raw_ranges[: 2 * range_count].contiguous(),
            )

        keep_mask = final_keep_mask.to(device="cpu", dtype=torch.bool).view(-1)
        selected_positions: list[int] = []
        selected_ranges: list[tuple[int, int]] = []
        selected_offsets = [0]
        ranges = events.raw_ranges.view(-1, 2)
        for event_index, insertion in enumerate(events.event_insert_offsets.tolist()):
            begin = int(events.range_offsets[event_index])
            end = int(events.range_offsets[event_index + 1])
            event_ranges = [
                (int(start), int(finish))
                for start, finish in ranges[begin:end].tolist()
            ]
            effective: list[bool] = []
            for start, finish in event_ranges:
                segment = keep_mask[start:finish]
                all_kept = bool(torch.all(segment).item())
                all_dropped = bool(torch.all(~segment).item())
                if not (all_kept or all_dropped):
                    raise ValueError(
                        "A target-specific keep mask partially cuts a Drop delta range: "
                        f"event={insertion}, range=[{start}, {finish})"
                    )
                effective.append(all_dropped)
            if effective and any(state != effective[0] for state in effective[1:]):
                raise ValueError(
                    "One Drop event has inconsistent target-specific range visibility: "
                    f"event={insertion}, effective={effective}"
                )
            if effective and effective[0]:
                selected_positions.append(int(insertion))
                selected_ranges.extend(event_ranges)
                selected_offsets.append(len(selected_ranges))

        return (
            torch.tensor(selected_positions, dtype=torch.int32, device="cpu"),
            torch.tensor(selected_offsets, dtype=torch.int32, device="cpu"),
            torch.tensor(selected_ranges, dtype=torch.int32, device="cpu").reshape(-1),
        )

    def _call_chat_template(
        self,
        messages: List[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> Any:
        if self.is_gpt_oss:
            tokens = self._render_harmony_tokens(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            if tokenize:
                return tokens
            return self._get_harmony_encoding().decode(tokens)
        kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None and self._supports_enable_thinking:
            kwargs["enable_thinking"] = enable_thinking
        if tools is not None:
            kwargs["tools"] = self._effective_template_tools(tools)
        if self._chat_template_override is not None:
            kwargs["chat_template"] = self._chat_template_override
        kwargs.update(self._chat_template_kwargs)
        tried_bare_tools = self._template_requires_bare_tools
        while True:
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError as exc:
                # Only downgrade when the failure is specifically due to an
                # unknown template kwarg. Keep other TypeErrors visible.
                if "enable_thinking" in kwargs and "enable_thinking" in str(exc):
                    kwargs.pop("enable_thinking")
                    self._supports_enable_thinking = False
                    continue
                error = exc
            except Exception as exc:
                error = exc

            if tools is None or tried_bare_tools:
                raise error

            # Match SGLang's compatibility path: use OpenAI wrappers first,
            # then retry a flat function-only list for templates that reject it.
            flat_tools = self._flatten_tools(tools)
            if flat_tools == kwargs.get("tools"):
                raise error
            kwargs["tools"] = flat_tools
            self._template_requires_bare_tools = True
            tried_bare_tools = True

    def _get_harmony_encoding(self):
        if self._harmony_encoding is None:
            try:
                from openai_harmony import HarmonyEncodingName, load_harmony_encoding
            except ImportError as exc:
                raise RuntimeError("GPT-OSS chat requests require openai-harmony>=0.0.8.") from exc
            self._harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        return self._harmony_encoding

    def _build_harmony_prompt(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> _HarmonyPrompt:
        from openai_harmony import (
            Author,
            Conversation,
            DeveloperContent,
            ReasoningEffort,
            Role,
            SystemContent,
            ToolDescription,
        )
        from openai_harmony import (
            Message as HarmonyMessage,
        )

        effort_text = self._reasoning_effort
        if effort_text is None and enable_thinking is False:
            effort_text = "low"
        effort = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }.get(str(effort_text or "medium").lower())
        if effort is None:
            raise ValueError("reasoning_effort must be one of: low, medium, high")

        harmony_messages = [
            HarmonyMessage.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(effort)
                .with_conversation_start_date(date.today().isoformat())
                .with_required_channels(["analysis", "commentary", "final"]),
            )
        ]
        ownership = [_HarmonyComponentOwnership(owner=-1)]
        thinking_components: dict[int, tuple[int, str]] = {}

        instruction_sources = tuple(
            (raw_message_id, content)
            for raw_message_id, raw in enumerate(messages)
            if str(raw.get("role", "")).lower() in {"system", "developer"}
            and (content := self._normalize_message_content(raw.get("content")))
        )
        developer = DeveloperContent.new()
        if instruction_sources:
            developer.with_instructions("\n\n".join(content for _, content in instruction_sources))
        descriptions = []
        for tool in tools or []:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            descriptions.append(
                ToolDescription.new(
                    str(fn["name"]),
                    str(fn.get("description") or ""),
                    parameters=fn.get("parameters"),
                )
            )
        if descriptions:
            developer.with_function_tools(descriptions)
        if instruction_sources or descriptions:
            harmony_messages.append(HarmonyMessage.from_role_and_content(Role.DEVELOPER, developer))
            ownership.append(
                _HarmonyComponentOwnership(
                    owner=instruction_sources[0][0] if instruction_sources else -1,
                    sources=instruction_sources,
                )
            )

        tool_names: dict[str, str] = {}
        for raw_message_id, raw in enumerate(messages):
            role = str(raw.get("role", "user")).lower()
            if role in {"system", "developer"}:
                continue
            content = self._normalize_message_content(raw.get("content"))
            if role == "user":
                harmony_messages.append(HarmonyMessage.from_role_and_content(Role.USER, content))
                ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                continue
            if role == "assistant":
                calls = raw.get("tool_calls") or []
                if calls and content:
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(Role.ASSISTANT, content).with_channel(
                            "commentary"
                        )
                    )
                    ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                reasoning = raw.get("reasoning")
                legacy_reasoning = raw.get("reasoning_content")
                if (
                    isinstance(reasoning, str)
                    and isinstance(legacy_reasoning, str)
                    and reasoning != legacy_reasoning
                ):
                    raise ValueError(
                        "GPT-OSS assistant reasoning and reasoning_content must match "
                        "when both are provided."
                    )
                if not isinstance(reasoning, str):
                    reasoning = legacy_reasoning
                if isinstance(reasoning, str) and reasoning:
                    component_id = len(harmony_messages)
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(
                            Role.ASSISTANT, reasoning
                        ).with_channel("analysis")
                    )
                    thinking_components[component_id] = (raw_message_id, reasoning)
                    ownership.append(
                        _HarmonyComponentOwnership(
                            owner=raw_message_id,
                            is_analysis=True,
                        )
                    )
                if content and not calls:
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(Role.ASSISTANT, content).with_channel(
                            "final"
                        )
                    )
                    ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function")
                    if not isinstance(fn, dict) or not fn.get("name"):
                        continue
                    name = str(fn["name"])
                    if call.get("id") is not None:
                        tool_names[str(call["id"])] = name
                    arguments = fn.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = self._json_dumps(arguments)
                    harmony_messages.append(
                        HarmonyMessage.from_role_and_content(Role.ASSISTANT, arguments)
                        .with_channel("commentary")
                        .with_recipient(f"functions.{name}")
                        .with_content_type("json")
                    )
                    ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                continue
            if role in {"tool", "function"}:
                name = raw.get("name") or tool_names.get(str(raw.get("tool_call_id", "")))
                if not name:
                    raise ValueError(
                        "GPT-OSS tool results require name or a matching tool_call_id."
                    )
                harmony_messages.append(
                    HarmonyMessage.from_author_and_content(
                        Author.new(Role.TOOL, f"functions.{name}"), content
                    )
                    .with_channel("commentary")
                    .with_recipient("assistant")
                )
                ownership.append(_HarmonyComponentOwnership(owner=raw_message_id))
                continue
            raise ValueError(f"Unsupported GPT-OSS Harmony role: {role}")

        prompt = _HarmonyPrompt(
            conversation=Conversation.from_messages(harmony_messages),
            components=harmony_messages,
            ownership=ownership,
            thinking_components=thinking_components,
            has_function_tools=bool(descriptions),
        )
        return (
            prompt
            if self._preserve_harmony_thinking
            else self._drop_harmony_analysis_before_last_final(prompt)
        )

    @staticmethod
    def _drop_harmony_analysis_before_last_final(prompt: _HarmonyPrompt) -> _HarmonyPrompt:
        """Match vLLM's long-history cleanup while retaining component ownership."""

        from openai_harmony import Conversation

        last_final = -1
        for component_id in range(len(prompt.components) - 1, -1, -1):
            component = prompt.components[component_id]
            role = getattr(getattr(component, "author", None), "role", None)
            role = getattr(role, "value", role)
            if str(role).lower() == "assistant" and component.channel == "final":
                last_final = component_id
                break
        if last_final < 0:
            return prompt

        keep_ids = [
            component_id
            for component_id, component in enumerate(prompt.components)
            if not (component_id < last_final and component.channel == "analysis")
        ]
        if len(keep_ids) == len(prompt.components):
            return prompt

        remap = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
        components = [prompt.components[component_id] for component_id in keep_ids]
        return _HarmonyPrompt(
            conversation=Conversation.from_messages(components),
            components=components,
            ownership=[prompt.ownership[component_id] for component_id in keep_ids],
            thinking_components={
                remap[component_id]: source
                for component_id, source in prompt.thinking_components.items()
                if component_id in remap
            },
            has_function_tools=prompt.has_function_tools,
        )

    def _render_harmony_tokens(
        self,
        messages: List[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> List[int]:
        from openai_harmony import RenderConversationConfig, RenderOptions, Role

        prompt = self._build_harmony_prompt(
            messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        encoding = self._get_harmony_encoding()
        config = RenderConversationConfig(auto_drop_analysis=False)
        if self._preserve_harmony_thinking:
            render_options = RenderOptions(
                conversation_has_function_tools=prompt.has_function_tools
            )
            component_stream: List[int] = []
            thinking_ranges: dict[int, List[tuple[int, int]]] = {}
            for component_id, component in enumerate(prompt.components):
                component_ids = [
                    int(token_id) for token_id in encoding.render(component, render_options)
                ]
                thinking_source = prompt.thinking_components.get(component_id)
                if thinking_source is not None:
                    raw_message_id, source = thinking_source
                    needle = [int(token_id) for token_id in encoding.encode(source)]
                    local_start, local_end = self._find_owned_token_subsequence(
                        component_ids,
                        [raw_message_id] * len(component_ids),
                        needle,
                        owner=raw_message_id,
                        field="thinking",
                    )
                    thinking_ranges.setdefault(raw_message_id, []).append(
                        (
                            len(component_stream) + local_start,
                            len(component_stream) + local_end,
                        )
                    )
                component_stream.extend(component_ids)
            self._harmony_thinking_ranges = thinking_ranges
        else:
            component_stream = []
            self._harmony_thinking_ranges = {}

        if add_generation_prompt:
            result = encoding.render_conversation_for_completion(
                prompt.conversation, Role.ASSISTANT, config
            )
        else:
            result = encoding.render_conversation(prompt.conversation, config)
        result = [int(token_id) for token_id in result]
        if self._preserve_harmony_thinking and result[: len(component_stream)] != component_stream:
            raise RuntimeError(
                "Harmony thinking retention changed component token boundaries; "
                "cannot construct exact thinking provenance."
            )
        return result

    def _render_harmony_message_drop(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> tuple[List[int], List[int], int]:
        """Render once and recover message owners from Harmony protocol boundaries."""

        from openai_harmony import RenderConversationConfig, Role

        prompt = self._build_harmony_prompt(
            messages,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        encoding = self._get_harmony_encoding()
        input_ids = [
            int(token_id)
            for token_id in encoding.render_conversation_for_completion(
                prompt.conversation,
                Role.ASSISTANT,
                RenderConversationConfig(auto_drop_analysis=False),
            )
        ]

        special_names = {
            token_id: encoding.decode([token_id])
            for token_id in set(input_ids)
            if encoding.is_special_token(token_id)
        }
        starts = [
            position
            for position, token_id in enumerate(input_ids)
            if special_names.get(token_id) == "<|start|>"
        ]
        if not starts:
            raise RuntimeError("Harmony render contains no message boundary tokens.")

        complete_ranges = list(zip(starts[:-1], starts[1:]))
        generation_start = starts[-1]
        if any(
            special_names.get(input_ids[position]) in {"<|end|>", "<|call|>", "<|return|>"}
            for position in range(generation_start, len(input_ids))
        ):
            raise RuntimeError("Harmony completion render has no generation prompt.")
        expected: list[_HarmonyComponentOwnership] = []
        component_id = 0
        for start, end in complete_ranges:
            header = encoding.decode(input_ids[start:end]).split("<|message|>", 1)[0]
            is_analysis = "<|channel|>analysis" in header
            while (
                component_id < len(prompt.ownership)
                and prompt.ownership[component_id].is_analysis
                and not is_analysis
            ):
                component_id += 1
            if (
                component_id >= len(prompt.ownership)
                or prompt.ownership[component_id].is_analysis != is_analysis
            ):
                raise RuntimeError(
                    "Harmony analysis filtering changed the native message stream; "
                    "cannot align message ownership."
                )
            expected.append(prompt.ownership[component_id])
            component_id += 1
        if any(not item.is_analysis for item in prompt.ownership[component_id:]):
            raise RuntimeError(
                "Harmony analysis filtering changed the native message stream; "
                "cannot align message ownership."
            )

        owners = [-1] * len(input_ids)
        decode_bytes = getattr(getattr(encoding, "_inner", None), "decode_bytes", None)
        for (start, end), component in zip(complete_ranges, expected):
            owners[start:end] = [component.owner] * (end - start)
            if len(component.sources) < 2:
                continue
            if decode_bytes is None:
                raise RuntimeError(
                    "Harmony byte decoding is required to split merged system messages."
                )

            token_bytes = [bytes(decode_bytes([token_id])) for token_id in input_ids[start:end]]
            offsets = [0]
            for value in token_bytes:
                offsets.append(offsets[-1] + len(value))
            rendered = b"".join(token_bytes)
            byte_owners = [component.owner] * len(rendered)
            cursor = 0
            previous_owner = component.owner
            for source_owner, source in component.sources:
                needle = source.encode("utf-8")
                source_start = rendered.find(needle, cursor)
                if source_start < 0:
                    raise RuntimeError(
                        "Harmony changed system/developer message text; "
                        "cannot construct exact ownership."
                    )
                source_end = source_start + len(needle)
                byte_owners[cursor:source_start] = [previous_owner] * (source_start - cursor)
                byte_owners[source_start:source_end] = [source_owner] * len(needle)
                cursor = source_end
                previous_owner = source_owner
            byte_owners[cursor:] = [previous_owner] * (len(rendered) - cursor)

            previous_token_owner = component.owner
            for local_id, (byte_start, byte_end) in enumerate(zip(offsets[:-1], offsets[1:])):
                if byte_start == byte_end:
                    token_owner = previous_token_owner
                else:
                    token_owner = byte_owners[byte_start]
                owners[start + local_id] = token_owner
                previous_token_owner = token_owner

        owners[generation_start:] = [len(messages)] * (len(input_ids) - generation_start)
        return input_ids, owners, generation_start

    def _build_harmony_provenance(
        self,
        input_ids: List[int],
        owners: List[int],
    ) -> TemplateTokenProvenance:
        """Recover character offsets from Harmony bytes without retokenizing."""

        encoding = self._get_harmony_encoding()
        decode_bytes = getattr(getattr(encoding, "_inner", None), "decode_bytes", None)
        if decode_bytes is None:
            raise RuntimeError("Harmony byte decoding is required for keep_text_drop.")
        token_bytes = [bytes(decode_bytes([token_id])) for token_id in input_ids]
        byte_offsets = [0]
        byte_owners: List[int] = []
        for owner, value in zip(owners, token_bytes, strict=True):
            byte_offsets.append(byte_offsets[-1] + len(value))
            byte_owners.extend([owner] * len(value))
        rendered_bytes = b"".join(token_bytes)
        rendered_text = rendered_bytes.decode("utf-8")
        char_byte_offsets = [0]
        char_owners: List[int] = []
        cross_owner_tokens = 0
        byte_pos = 0
        for char in rendered_text:
            encoded = char.encode("utf-8")
            char_owners.append(byte_owners[byte_pos] if encoded else 0)
            byte_pos += len(encoded)
            char_byte_offsets.append(byte_pos)

        offsets: List[tuple[int, int]] = []
        for token_id, (byte_start, byte_end) in enumerate(
            zip(byte_offsets[:-1], byte_offsets[1:], strict=True)
        ):
            char_start = max(bisect_right(char_byte_offsets, byte_start) - 1, 0)
            char_end = bisect_left(char_byte_offsets, byte_end)
            offsets.append((char_start, char_end))
            if byte_end > byte_start and any(
                owner != owners[token_id] for owner in byte_owners[byte_start:byte_end]
            ):
                cross_owner_tokens += 1
        return TemplateTokenProvenance(
            input_ids=list(input_ids),
            owners=list(owners),
            offsets=offsets,
            rendered_text=rendered_text,
            char_owners=char_owners,
            cross_owner_tokens=cross_owner_tokens,
        )

    def _apply_chat_template(
        self,
        messages: List[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> List[int]:
        result = self._call_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        if isinstance(result, torch.Tensor):
            result = result.view(-1).tolist()
        return [int(token_id) for token_id in result]

    def _render_chat_template(
        self,
        messages: List[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> str:
        result = self._call_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
        )
        if not isinstance(result, str):
            raise RuntimeError("Chat template did not return text with tokenize=False.")
        return result

    def _build_template_provenance(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
        expected_input_ids: List[int] | None,
    ) -> TemplateTokenProvenance:
        def render(add_generation_prompt: bool) -> str:
            return self._render_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=effective_enable_thinking,
                tools=effective_tools,
            )

        effective_tools = self._effective_template_tools(tools)
        effective_enable_thinking = enable_thinking
        canonical_no_gen = render(False)
        canonical_with_gen = render(True)
        supported_thinking = enable_thinking if self._supports_enable_thinking else None
        if supported_thinking != effective_enable_thinking:
            effective_enable_thinking = supported_thinking
            canonical_no_gen = render(False)
            canonical_with_gen = render(True)
        effective_tools = self._effective_template_tools(tools)

        return build_template_token_provenance(
            self.tokenizer,
            messages,
            canonical_text=canonical_with_gen,
            canonical_no_generation_text=canonical_no_gen,
            expected_input_ids=expected_input_ids,
            tools=effective_tools,
            add_generation_prompt=True,
            enable_thinking=effective_enable_thinking,
            chat_template=self._chat_template_override,
            template_kwargs=self._chat_template_kwargs,
        )

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )

    def _normalize_message_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Message content parts must be objects.")
                part_type = part.get("type")
                if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part_type == "thinking" and isinstance(part.get("thinking"), str):
                    parts.append(part["thinking"])
                else:
                    raise ValueError(
                        "MiniSGL currently supports only text content parts in chat messages."
                    )
            return "".join(parts)
        return self._json_dumps(content)

    def _normalize_tool_calls_for_template(self, tool_calls: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            call = dict(item)
            fn = call.get("function")
            if isinstance(fn, dict):
                fn = dict(fn)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                call["function"] = fn
            normalized.append(call)
        return normalized

    @staticmethod
    def _extract_tool_name(tool: Any) -> str | None:
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

    def _normalize_tool_choice(self, tool_choice: Any) -> tuple[str, str | None]:
        if tool_choice is None:
            return "auto", None
        if isinstance(tool_choice, str):
            text = tool_choice.strip()
            lowered = text.lower()
            if lowered in {"auto", "required", "none"}:
                return lowered, None
            return "function", text
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                if isinstance(name, str) and len(name.strip()) > 0:
                    return "function", name.strip()
        return "auto", None

    def _select_tools_for_choice(
        self,
        tools: List[Dict[str, Any]] | None,
        *,
        mode: str,
        forced_tool_name: str | None,
    ) -> List[Dict[str, Any]] | None:
        if not tools or mode == "none":
            return None
        if mode != "function" or forced_tool_name is None:
            return tools
        selected = [tool for tool in tools if self._extract_tool_name(tool) == forced_tool_name]
        # Keep all tools as fallback if we cannot find an exact match.
        return selected if len(selected) > 0 else tools

    @staticmethod
    def _flatten_tools(
        tools: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]] | None:
        if not tools:
            return None
        return [
            (
                dict(tool["function"])
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                else dict(tool)
            )
            for tool in tools
        ]

    def _effective_template_tools(
        self,
        tools: List[Dict[str, Any]] | None,
    ) -> List[Dict[str, Any]] | None:
        if self._template_requires_bare_tools:
            return self._flatten_tools(tools)
        return tools

    def _build_template_messages(
        self,
        raw_messages: List[Dict[str, Any]],
        *,
        safe_mode: bool,
    ) -> tuple[List[dict[str, Any]], int]:
        messages: List[dict[str, Any]] = []
        for raw in raw_messages:
            role = str(raw.get("role", "user")).lower()
            if role == "function":
                role = "tool"

            content = self._normalize_message_content(raw.get("content"))
            message: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant" and isinstance(raw.get("reasoning_content"), str):
                message["reasoning_content"] = raw["reasoning_content"]

            if safe_mode:
                if role == "assistant" and isinstance(raw.get("tool_calls"), list):
                    suffix = "tool_calls: " + self._json_dumps(raw.get("tool_calls"))
                    message["content"] = (content + "\n" + suffix) if content else suffix
                elif role == "tool":
                    prefix = f"[tool:{raw.get('name') or ''}]"
                    if raw.get("tool_call_id") is not None:
                        prefix += f"[tool_call_id:{raw.get('tool_call_id')}]"
                    payload = content
                    message = {
                        "role": "user",
                        "content": (prefix + "\n" + payload) if payload else prefix,
                    }
                elif role not in {"system", "user", "assistant"}:
                    message["role"] = "user"
                messages.append(message)
                continue

            if role == "assistant" and isinstance(raw.get("tool_calls"), list):
                message["tool_calls"] = self._normalize_tool_calls_for_template(
                    raw.get("tool_calls")
                )

            if role == "tool":
                if raw.get("tool_call_id") is not None:
                    message["tool_call_id"] = str(raw.get("tool_call_id"))
                if raw.get("name") is not None:
                    message["name"] = str(raw.get("name"))

            messages.append(message)

        # Tool definitions are rendered exactly once by the model's chat
        # template.  They must never be duplicated into a synthetic system
        # message because that changes both model semantics and Drop ownership.
        return messages, 0

    def _normalize_drop_message(
        self, drop_message: dict[int, List[int]] | None
    ) -> dict[int, List[int]]:
        if not drop_message:
            return {}
        normalized: dict[int, List[int]] = {}
        for k, value in drop_message.items():
            n = int(k)
            if n < 0:
                raise ValueError(f"drop_message key must be non-negative: {n}")
            if self.radix_drop_key_mode == "bitmask" and n >= 32:
                raise ValueError(f"drop_message key out of range [0, 31]: {n}")
            if self.radix_drop_key_mode in {"symbol", "delta-marker"} and n >= (1 << 63):
                raise ValueError(f"drop_message key out of int64 range: {n}")
            ids = [int(v) for v in value]
            for msg_id in ids:
                if msg_id < 0:
                    raise ValueError(f"drop_message id must be non-negative: {msg_id}")
                if self.radix_drop_key_mode == "bitmask" and msg_id >= 32:
                    raise ValueError(f"drop_message id out of range [0, 31]: {msg_id}")
                if self.radix_drop_key_mode in {"symbol", "delta-marker"} and msg_id >= (1 << 63):
                    raise ValueError(f"drop_message id out of int64 range: {msg_id}")
            normalized[n] = ids
        return normalized

    @staticmethod
    def _shift_and_validate_drop_message(
        drop_message: dict[int, List[int]],
        *,
        target_offset: int,
        normalized_message_count: int,
    ) -> dict[int, List[int]]:
        shifted: dict[int, List[int]] = {}
        for raw_n, raw_ids in drop_message.items():
            n = raw_n + target_offset
            ids = [msg_id + target_offset for msg_id in raw_ids]
            for raw_id, msg_id in zip(raw_ids, ids, strict=True):
                if n < normalized_message_count and msg_id >= normalized_message_count:
                    raise ValueError(
                        f"drop_message id {raw_id} refers to a message outside the conversation."
                    )
                if msg_id > n:
                    raise ValueError(
                        f"drop_message event {raw_n} cannot drop future message {raw_id}."
                    )
            if n >= normalized_message_count:
                # Staged warmup tokenizes message prefixes while carrying the full
                # request schedule. Events beyond this prefix have not happened yet.
                continue
            shifted[n] = ids
        return shifted

    def _build_drop_set(self, drop_message: dict[int, List[int]], upper_n: int) -> set[int]:
        dropped: set[int] = set()
        for n, ids in drop_message.items():
            if n < upper_n:
                dropped.update(ids)
        return dropped

    def _build_drop_mask(self, drop_message: dict[int, List[int]], msg_id: int) -> int:
        mask = 0
        for dropped_id in self._build_drop_set(drop_message, msg_id):
            mask |= 1 << dropped_id
        return mask

    @staticmethod
    def _build_owner_position_ranges(
        owners: List[int],
    ) -> dict[int, List[tuple[int, int]]]:
        """Return exact full-token ranges for every provenance owner."""

        ranges: dict[int, List[tuple[int, int]]] = {}
        if not owners:
            return ranges
        start = 0
        owner = int(owners[0])
        for pos in range(1, len(owners) + 1):
            next_owner = int(owners[pos]) if pos < len(owners) else None
            if next_owner == owner:
                continue
            ranges.setdefault(owner, []).append((start, pos))
            if pos < len(owners):
                start = pos
                owner = int(next_owner)
        return ranges

    @staticmethod
    def _canonicalize_position_ranges(
        ranges: List[tuple[int, int]],
    ) -> List[tuple[int, int]]:
        if not ranges:
            return []
        normalized = sorted((int(start), int(end)) for start, end in ranges)
        merged: List[tuple[int, int]] = []
        for start, end in normalized:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid token-position Drop range: [{start}, {end})")
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @classmethod
    def _ranges_for_messages(
        cls,
        owner_ranges: dict[int, List[tuple[int, int]]],
        message_ids: set[int],
    ) -> List[tuple[int, int]]:
        ranges: List[tuple[int, int]] = []
        for msg_id in sorted(message_ids):
            ranges.extend(owner_ranges.get(msg_id, ()))
        return cls._canonicalize_position_ranges(ranges)

    @classmethod
    def _build_position_drop_plan(
        cls,
        drop_message: dict[int, List[int]],
        query_epoch: List[int],
        owner_ranges: dict[int, List[tuple[int, int]]],
    ) -> PositionDropPlan:
        """Compile message selectors into absolute half-open position deltas."""

        delta_by_pos: dict[int, List[tuple[int, int]]] = {}
        effective_messages: set[int] = set()
        for event_n in sorted(drop_message):
            newly_effective = set(drop_message[event_n]) - effective_messages
            effective_messages.update(drop_message[event_n])
            delta_ranges = cls._ranges_for_messages(owner_ranges, newly_effective)
            if not delta_ranges:
                continue
            insertion_pos = bisect_right(query_epoch, event_n)
            if any(end > insertion_pos for _, end in delta_ranges):
                raise ValueError(
                    "A Drop event cannot hide a token before that token has been computed: "
                    f"event={event_n}, insertion_pos={insertion_pos}, ranges={delta_ranges}"
                )
            delta_by_pos.setdefault(insertion_pos, []).extend(delta_ranges)

        event_positions: List[int] = []
        range_offsets: List[int] = [0]
        flat_ranges: List[tuple[int, int]] = []
        visible_until = torch.full(
            (len(query_epoch),),
            torch.iinfo(torch.int32).max,
            dtype=torch.int32,
            device="cpu",
        )
        for insertion_pos in sorted(delta_by_pos):
            canonical = cls._canonicalize_position_ranges(delta_by_pos[insertion_pos])
            if not canonical:
                continue
            event_positions.append(insertion_pos)
            flat_ranges.extend(canonical)
            range_offsets.append(len(flat_ranges))
            for start, end in canonical:
                visible_until[start:end] = torch.minimum(
                    visible_until[start:end],
                    torch.tensor(insertion_pos, dtype=torch.int32, device="cpu"),
                )

        position_ranges = torch.tensor(flat_ranges, dtype=torch.int32, device="cpu").reshape(-1)
        return PositionDropPlan(
            event_positions=torch.tensor(event_positions, dtype=torch.int32, device="cpu"),
            range_offsets=torch.tensor(range_offsets, dtype=torch.int32, device="cpu"),
            position_ranges=position_ranges,
            full_token_visible_until=visible_until,
            effective_event_count=len(event_positions),
        )

    @classmethod
    def _build_position_range_drop_plan(
        cls,
        event_ranges: dict[int, List[tuple[int, int]]],
        query_epoch: List[int],
        target_msg_id: int,
    ) -> TokenDropEvents:
        """Compile rule-produced absolute ranges into the generic delta wire."""

        delta_by_pos: dict[int, List[tuple[int, int]]] = {}
        effective_by_pos: dict[int, bool | None] = {}
        covered_ranges: List[tuple[int, int]] = []
        for event_n in sorted(event_ranges):
            canonical = cls._canonicalize_position_ranges(event_ranges[event_n])
            newly_effective = cls._subtract_position_ranges(canonical, covered_ranges)
            covered_ranges = cls._canonicalize_position_ranges(covered_ranges + canonical)
            if not newly_effective:
                continue
            insertion_pos = bisect_right(query_epoch, event_n)
            if any(end > insertion_pos for _, end in newly_effective):
                raise ValueError(
                    "A Drop event cannot hide a token before that token has been computed: "
                    f"event={event_n}, insertion_pos={insertion_pos}, ranges={newly_effective}"
                )
            delta_by_pos.setdefault(insertion_pos, []).extend(newly_effective)
            is_effective = event_n < target_msg_id
            previous = effective_by_pos.setdefault(insertion_pos, is_effective)
            if previous != is_effective:
                effective_by_pos[insertion_pos] = None

        event_positions: List[int] = []
        range_offsets: List[int] = [0]
        flat_ranges: List[tuple[int, int]] = []
        effective_flags: List[bool | None] = []
        visible_until = torch.full(
            (len(query_epoch),),
            torch.iinfo(torch.int32).max,
            dtype=torch.int32,
            device="cpu",
        )
        for insertion_pos in sorted(delta_by_pos):
            canonical = cls._canonicalize_position_ranges(delta_by_pos[insertion_pos])
            if not canonical:
                continue
            event_positions.append(insertion_pos)
            effective_flags.append(effective_by_pos[insertion_pos])
            flat_ranges.extend(canonical)
            range_offsets.append(len(flat_ranges))
            for start, end in canonical:
                visible_until[start:end] = torch.minimum(
                    visible_until[start:end],
                    torch.tensor(insertion_pos, dtype=torch.int32, device="cpu"),
                )
        bool_flags = [flag for flag in effective_flags if flag is not None]
        effective_event_count = sum(bool_flags)
        if len(bool_flags) != len(effective_flags) or any(bool_flags[effective_event_count:]):
            # A template without a distinguishable target boundary can merge a
            # current and future event. Keep the exact-mask reference available
            # instead of guessing which ranges are effective.
            effective_event_count = -1
        current_ranges = cls._canonicalize_position_ranges(
            [
                item
                for event, ranges in event_ranges.items()
                if event < target_msg_id
                for item in ranges
            ]
        )
        return TokenDropEvents(
            event_insert_offsets=torch.tensor(event_positions, dtype=torch.int32, device="cpu"),
            range_offsets=torch.tensor(range_offsets, dtype=torch.int32, device="cpu"),
            raw_ranges=torch.tensor(flat_ranges, dtype=torch.int32, device="cpu").reshape(-1),
            full_token_visible_until=visible_until,
            effective_event_count=effective_event_count,
            effective_ranges=tuple(current_ranges),
        )

    @staticmethod
    def _subtract_position_ranges(
        ranges: List[tuple[int, int]],
        covered: List[tuple[int, int]],
    ) -> List[tuple[int, int]]:
        result: List[tuple[int, int]] = []
        for start, end in ranges:
            fragments = [(start, end)]
            for cover_start, cover_end in covered:
                next_fragments: List[tuple[int, int]] = []
                for frag_start, frag_end in fragments:
                    if cover_end <= frag_start or cover_start >= frag_end:
                        next_fragments.append((frag_start, frag_end))
                        continue
                    if frag_start < cover_start:
                        next_fragments.append((frag_start, cover_start))
                    if cover_end < frag_end:
                        next_fragments.append((cover_end, frag_end))
                fragments = next_fragments
            result.extend(fragments)
        return result

    @staticmethod
    def _rendered_source_start(
        provenance: TemplateTokenProvenance,
        *,
        owner: int,
        source: str,
        field: str,
        prefer_latest: bool = False,
    ) -> int:
        candidates = []
        for start, end in find_all(provenance.rendered_text, [source])[0]:
            if all(candidate == owner for candidate in provenance.char_owners[start:end]):
                candidates.append(start)
        if prefer_latest and candidates:
            return candidates[-1]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot map {field} for messages[{owner}] uniquely into the "
                "canonical chat template"
            )
        return candidates[0]

    @classmethod
    def _token_ranges_for_char_spans(
        cls,
        provenance: TemplateTokenProvenance,
        *,
        owner: int,
        spans: List[tuple[int, int]],
        field: str,
        boundary_mode: str = "contained",
        allow_empty: bool = False,
    ) -> List[tuple[int, int]]:
        if boundary_mode not in {"contained", "overlap"}:
            raise ValueError(f"Unsupported token boundary mode: {boundary_mode}")
        selected: List[int] = []
        for token_id, (start, end) in enumerate(provenance.offsets):
            if provenance.owners[token_id] != owner or start == end:
                continue
            if boundary_mode == "contained":
                matched = any(
                    start >= span_start and end <= span_end for span_start, span_end in spans
                )
            else:
                matched = any(
                    start < span_end and end > span_start for span_start, span_end in spans
                )
            if matched:
                selected.append(token_id)
        if not selected:
            if allow_empty:
                return []
            raise ValueError(
                f"{field} for messages[{owner}] contains no complete token; "
                "boundary-crossing tokens are kept"
            )
        ranges: List[tuple[int, int]] = []
        start = previous = selected[0]
        for token_id in selected[1:]:
            if token_id == previous + 1:
                previous = token_id
                continue
            ranges.append((start, previous + 1))
            start = previous = token_id
        ranges.append((start, previous + 1))
        return cls._canonicalize_position_ranges(ranges)

    @classmethod
    def _position_ranges_from_ids(cls, token_ids: List[int]) -> List[tuple[int, int]]:
        if not token_ids:
            return []
        token_ids = sorted(set(token_ids))
        ranges: List[tuple[int, int]] = []
        start = previous = token_ids[0]
        for token_id in token_ids[1:]:
            if token_id == previous + 1:
                previous = token_id
                continue
            ranges.append((start, previous + 1))
            start = previous = token_id
        ranges.append((start, previous + 1))
        return cls._canonicalize_position_ranges(ranges)

    @staticmethod
    def _find_owned_token_subsequence(
        full_ids: List[int],
        owners: List[int],
        needle: List[int],
        *,
        owner: int,
        field: str,
    ) -> tuple[int, int]:
        if not needle:
            raise ValueError(f"{field} for messages[{owner}] tokenizes to an empty sequence")
        # KMP keeps the GPT-OSS Harmony fallback linear in the full token stream.
        prefix = [0] * len(needle)
        j = 0
        for i in range(1, len(needle)):
            while j and needle[i] != needle[j]:
                j = prefix[j - 1]
            if needle[i] == needle[j]:
                j += 1
                prefix[i] = j
        matches: List[tuple[int, int]] = []
        j = 0
        for i, token in enumerate(full_ids):
            while j and token != needle[j]:
                j = prefix[j - 1]
            if token == needle[j]:
                j += 1
                if j == len(needle):
                    start = i + 1 - len(needle)
                    end = i + 1
                    if all(candidate == owner for candidate in owners[start:end]):
                        matches.append((start, end))
                    j = prefix[j - 1]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot map {field} for messages[{owner}] uniquely into the Harmony token stream"
            )
        return matches[0]

    @staticmethod
    def _encode_radix_key(token_id: int, drop_mask: int) -> int:
        # Keep token id in low 32 bits and drop bitset in high 32 bits.
        # Normalize to signed int64 range before writing into torch.int64 tensor.
        encoded = ((token_id & 0xFFFFFFFF) | (drop_mask << 32)) & ((1 << 64) - 1)
        if encoded >= (1 << 63):
            encoded -= 1 << 64
        return encoded

    @staticmethod
    def _resolve_target_msg_id(
        msg: TokenizeMsg,
        normalized_msg_count: int,
        *,
        target_offset: int,
    ) -> int:
        if msg.target_msg_id is None:
            target_msg_id = (
                max(normalized_msg_count - 1, 0) if msg.is_warmup else normalized_msg_count
            )
        else:
            target_msg_id = int(msg.target_msg_id) + target_offset
        if target_msg_id < 0 or target_msg_id > normalized_msg_count:
            raise ValueError(
                f"target_msg_id out of range [0, {normalized_msg_count}]: {target_msg_id}"
            )
        return target_msg_id

    def _build_stop_token_seqs(self, stop: List[str] | None) -> List[List[int]] | None:
        dedup: set[tuple[int, ...]] = set()
        result: List[List[int]] = []
        for raw in stop or []:
            text = str(raw)
            if len(text) == 0:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if isinstance(ids, torch.Tensor):
                ids = ids.view(-1).tolist()
            token_ids = [int(v) for v in ids]
            if len(token_ids) == 0:
                continue
            key = tuple(token_ids)
            if key in dedup:
                continue
            dedup.add(key)
            result.append(token_ids)
        if self.is_gpt_oss:
            for token_id in get_gpt_oss_terminal_stop_token_ids():
                key = (int(token_id),)
                if key not in dedup:
                    dedup.add(key)
                    result.append([int(token_id)])
        return result if len(result) > 0 else None

    @staticmethod
    def _common_prefix_len(x: List[int], y: List[int]) -> int:
        limit = min(len(x), len(y))
        idx = 0
        while idx < limit and x[idx] == y[idx]:
            idx += 1
        return idx

    @staticmethod
    def _common_suffix_len(x: List[int], y: List[int], lcp: int) -> int:
        max_suffix = min(len(x) - lcp, len(y) - lcp)
        idx = 0
        while idx < max_suffix and x[len(x) - 1 - idx] == y[len(y) - 1 - idx]:
            idx += 1
        return idx

    def _merge_owner_track(
        self,
        prev_ids: List[int],
        prev_owner: List[int],
        curr_ids: List[int],
        *,
        new_owner: int,
    ) -> tuple[List[int], int, int, bool]:
        # Conservative attribution policy:
        # - keep only exact stable prefix/suffix owners from previous round
        # - attribute rewritten middle to the newly appended message owner
        # This avoids owner drift when chat templates rewrite delimiters.
        lcp = self._common_prefix_len(prev_ids, curr_ids)
        lcsuf = self._common_suffix_len(prev_ids, curr_ids, lcp)
        unstable = lcp < len(prev_ids)

        curr_owner = [new_owner] * len(curr_ids)

        safe_lcp = min(lcp, len(prev_owner), len(curr_owner))
        if safe_lcp > 0:
            curr_owner[:safe_lcp] = prev_owner[:safe_lcp]

        if lcsuf > 0 and len(prev_owner) >= lcsuf and len(curr_owner) >= lcsuf:
            curr_owner[len(curr_owner) - lcsuf :] = prev_owner[len(prev_owner) - lcsuf :]

        return curr_owner, lcp, lcsuf, unstable

    @staticmethod
    def _merge_query_epoch_track(
        prev_epoch: List[int],
        curr_len: int,
        *,
        stable_prefix_len: int,
        new_epoch: int,
    ) -> List[int]:
        """Keep only the stable prefix epoch; all rewritten/suffix tokens use the new epoch."""

        curr_epoch = [new_epoch] * curr_len
        safe_lcp = min(stable_prefix_len, len(prev_epoch), curr_len)
        if safe_lcp > 0:
            curr_epoch[:safe_lcp] = prev_epoch[:safe_lcp]
        if any(curr_epoch[i] > curr_epoch[i + 1] for i in range(len(curr_epoch) - 1)):
            raise RuntimeError("Query epoch construction produced a non-monotonic token sequence.")
        return curr_epoch

    def _round_by_round_no_gen(
        self,
        messages: List[dict[str, Any]],
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> tuple[List[int], List[int], List[int], int]:
        assembled: List[int] = []
        owner: List[int] = []
        query_epoch: List[int] = []
        unstable_rounds = 0

        if self.is_gpt_oss:
            # Harmony always injects model system metadata (and may inject a
            # tool-bearing developer message). Keep that exact token prefix
            # outside user message ownership so Drop never removes it.
            assembled = self._apply_chat_template(
                [],
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            owner = [-1] * len(assembled)
            query_epoch = [0] * len(assembled)

        for i in range(len(messages)):
            curr = self._apply_chat_template(
                messages[: i + 1],
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            if i == 0 and not assembled:
                assembled = curr
                owner = [0] * len(curr)
                query_epoch = [0] * len(curr)
                continue

            owner, lcp, _, unstable = self._merge_owner_track(assembled, owner, curr, new_owner=i)
            query_epoch = self._merge_query_epoch_track(
                query_epoch,
                len(curr),
                stable_prefix_len=lcp,
                new_epoch=i,
            )
            unstable_rounds += int(unstable)
            assembled = curr

        return assembled, owner, query_epoch, unstable_rounds

    @staticmethod
    def _map_no_gen_pos_to_with_gen(
        pos: int,
        no_gen_len: int,
        with_gen_len: int,
        lcp: int,
        lcsuf: int,
    ) -> int:
        if pos <= lcp:
            return pos
        old_suffix_start = no_gen_len - lcsuf
        new_suffix_start = with_gen_len - lcsuf
        if pos >= old_suffix_start:
            return new_suffix_start + (pos - old_suffix_start)
        # Position is in rewritten middle; map to rewritten block start.
        return lcp

    def _drop_compile_context(
        self,
        *,
        raw_messages: List[Dict[str, Any]],
        owner_ranges: dict[int, List[tuple[int, int]]],
        provenance: TemplateTokenProvenance | None,
        full_input_ids: List[int],
        owners: List[int],
        target_offset: int,
        normalized_message_count: int,
        compile_events: Callable[[dict[int, List[tuple[int, int]]]], TokenDropEvents],
    ) -> DropCompileContext:
        def encode_text(text: str) -> list[int]:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.view(-1).tolist()
            return [int(token_id) for token_id in encoded]

        return DropCompileContext(
            raw_messages=raw_messages,
            owner_ranges=owner_ranges,
            provenance=provenance,
            full_input_ids=full_input_ids,
            owners=owners,
            target_offset=target_offset,
            normalized_message_count=normalized_message_count,
            is_gpt_oss=self.is_gpt_oss,
            harmony_thinking_ranges=self._harmony_thinking_ranges,
            normalize_content=self._normalize_message_content,
            rendered_source_start=self._rendered_source_start,
            token_ranges_for_char_spans=self._token_ranges_for_char_spans,
            canonicalize_ranges=self._canonicalize_position_ranges,
            position_ranges_from_ids=self._position_ranges_from_ids,
            find_owned_subsequence=self._find_owned_token_subsequence,
            encode_text=encode_text,
            compile_events=compile_events,
        )

    def _compile_rule_position_events(
        self,
        rule: MessageDropRule | TextDropRule | KeepTextDropRule | ThinkingDropRule,
        **context_args: Any,
    ) -> dict[int, List[tuple[int, int]]]:
        """Compatibility helper for callers that inspect pre-CSR event ranges."""

        context = self._drop_compile_context(
            **context_args,
            compile_events=lambda events: self._build_position_range_drop_plan(
                events,
                self._query_epochs_from_owners(
                    context_args["owners"], context_args["normalized_message_count"]
                ),
                context_args["normalized_message_count"],
            ),
        )
        return rule.position_events(context)

    @staticmethod
    def _query_epochs_from_owners(owners: List[int], message_count: int) -> List[int]:
        epochs: List[int] = []
        previous = 0
        for owner in owners:
            epoch = 0 if owner < 0 else min(owner, message_count)
            if epoch < previous:
                raise RuntimeError(
                    "Chat template reordered messages; cannot construct monotonic Drop events."
                )
            epochs.append(epoch)
            previous = epoch
        return epochs

    def _chat_tokenize(self, msg: TokenizeMsg) -> TokenizedResult:
        assert isinstance(msg.text, list)
        self._reasoning_effort = msg.reasoning_effort
        self._preserve_harmony_thinking = False
        self._harmony_thinking_ranges = {}
        self._chat_template_override = None
        self._chat_template_kwargs = {}
        drop_rule = parse_drop_rule(
            getattr(msg, "drop_rule", None),
            msg.text,
            legacy_drop_message=msg.drop_message,
            allow_internal=True,
        )
        has_reposition = bool(msg.reposition)
        if (drop_rule is not None or has_reposition) and self.radix_drop_key_mode != "delta-marker":
            raise ValueError(
                "Token-position Drop/Reposition requires radix_drop_key_mode='delta-marker'; "
                f"got {self.radix_drop_key_mode!r}."
            )
        tool_choice_mode, forced_tool_name = self._normalize_tool_choice(msg.tool_choice)
        selected_tools = self._select_tools_for_choice(
            msg.tools,
            mode=tool_choice_mode,
            forced_tool_name=forced_tool_name,
        )
        template_tools = selected_tools
        thinking_template_capability: str | None = None
        if isinstance(drop_rule, ThinkingDropRule):
            if self.is_gpt_oss:
                self._preserve_harmony_thinking = True
                thinking_template_capability = "harmony_auto_drop_disabled"
            else:
                thinking_plan = prepare_thinking_template(self.tokenizer, tools=template_tools)
                self._chat_template_override = thinking_plan.chat_template
                self._chat_template_kwargs = thinking_plan.template_kwargs
                thinking_template_capability = thinking_plan.capability

        if self.is_gpt_oss:
            messages = [dict(message) for message in msg.text]
            target_offset = 0
        else:
            messages, target_offset = self._build_template_messages(
                msg.text,
                safe_mode=False,
            )
        safe_mode = False
        effective_tools = template_tools
        provenance: TemplateTokenProvenance | None = None
        cross_owner_tokens = 0
        if (
            has_reposition
            or isinstance(drop_rule, (MessageDropRule, KeepTextDropRule))
            or (self.is_gpt_oss and isinstance(drop_rule, TextDropRule))
        ):
            if self.is_gpt_oss:
                full_with_gen, owner_with_gen, gen_prompt_start = self._render_harmony_message_drop(
                    messages,
                    enable_thinking=msg.enable_thinking,
                    tools=selected_tools,
                )
                if isinstance(drop_rule, (TextDropRule, KeepTextDropRule)):
                    provenance = self._build_harmony_provenance(full_with_gen, owner_with_gen)
                    cross_owner_tokens = provenance.cross_owner_tokens
            else:
                try:
                    provenance = self._build_template_provenance(
                        messages,
                        enable_thinking=msg.enable_thinking,
                        tools=template_tools,
                        expected_input_ids=None,
                    )
                except Exception:
                    messages, target_offset = self._build_template_messages(
                        msg.text,
                        safe_mode=True,
                    )
                    safe_mode = True
                    effective_tools = None
                    provenance = self._build_template_provenance(
                        messages,
                        enable_thinking=msg.enable_thinking,
                        tools=None,
                        expected_input_ids=None,
                    )
                full_with_gen = provenance.input_ids
                owner_with_gen = provenance.owners
                cross_owner_tokens = provenance.cross_owner_tokens
                try:
                    gen_prompt_start = owner_with_gen.index(len(messages))
                except ValueError:
                    gen_prompt_start = len(full_with_gen)

            full_no_gen = full_with_gen[:gen_prompt_start]
            query_epoch_with_gen = self._query_epochs_from_owners(owner_with_gen, len(messages))
            unstable_rounds = 0
            no_gen_with_gen_lcp = gen_prompt_start
            no_gen_with_gen_lcsuf = 0
            unstable_with_gen = False
        else:
            try:
                full_no_gen, no_gen_owner, no_gen_query_epoch, unstable_rounds = (
                    self._round_by_round_no_gen(messages, msg.enable_thinking, template_tools)
                )
                if len(messages) == 0:
                    no_gen_owner = []
                    no_gen_query_epoch = []
                full_with_gen = self._apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=msg.enable_thinking,
                    tools=template_tools,
                )
            except Exception:
                if self.is_gpt_oss or isinstance(drop_rule, ThinkingDropRule):
                    raise
                messages, target_offset = self._build_template_messages(
                    msg.text,
                    safe_mode=True,
                )
                safe_mode = True
                effective_tools = None
                full_no_gen, no_gen_owner, no_gen_query_epoch, unstable_rounds = (
                    self._round_by_round_no_gen(messages, msg.enable_thinking, None)
                )
                if len(messages) == 0:
                    no_gen_owner = []
                    no_gen_query_epoch = []
                full_with_gen = self._apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=msg.enable_thinking,
                    tools=None,
                )

            next_assistant_id = len(messages)
            owner_with_gen, no_gen_with_gen_lcp, no_gen_with_gen_lcsuf, unstable_with_gen = (
                self._merge_owner_track(
                    full_no_gen,
                    no_gen_owner,
                    full_with_gen,
                    new_owner=next_assistant_id,
                )
                if len(full_no_gen) > 0
                else ([next_assistant_id] * len(full_with_gen), 0, 0, False)
            )
            query_epoch_with_gen = (
                self._merge_query_epoch_track(
                    no_gen_query_epoch,
                    len(full_with_gen),
                    stable_prefix_len=no_gen_with_gen_lcp,
                    new_epoch=next_assistant_id,
                )
                if len(full_no_gen) > 0
                else [next_assistant_id] * len(full_with_gen)
            )

            if drop_rule is not None and not self.is_gpt_oss:
                provenance = self._build_template_provenance(
                    messages,
                    enable_thinking=msg.enable_thinking,
                    tools=effective_tools,
                    expected_input_ids=full_with_gen,
                )
                owner_with_gen = provenance.owners
                cross_owner_tokens = provenance.cross_owner_tokens

        full_no_gen_tensor = torch.tensor(full_no_gen, dtype=torch.int32, device="cpu")
        full_with_gen_tensor = torch.tensor(full_with_gen, dtype=torch.int32, device="cpu")
        next_assistant_id = len(messages)

        owner_ranges = self._build_owner_position_ranges(owner_with_gen)
        target_msg_id = self._resolve_target_msg_id(
            msg,
            len(messages),
            target_offset=target_offset,
        )
        if msg.reposition is None:
            reposition_raw_boundaries = None
            reposition_insert_offsets = None
        else:
            reposition_events = resolve_reposition_token_boundaries(
                msg.reposition,
                owner_ranges,
                {
                    public_id: public_id + target_offset
                    for public_id in range(len(msg.text))
                    if public_id + target_offset < len(messages)
                },
            )
            reposition_raw_boundaries = reposition_events.raw_boundaries
            reposition_insert_offsets = reposition_events.insert_offsets
        token_drop_events = None
        if drop_rule is not None:
            drop_context = self._drop_compile_context(
                raw_messages=msg.text,
                owner_ranges=owner_ranges,
                provenance=provenance,
                full_input_ids=full_with_gen,
                owners=owner_with_gen,
                target_offset=target_offset,
                normalized_message_count=len(messages),
                compile_events=lambda events: self._build_position_range_drop_plan(
                    events,
                    query_epoch_with_gen,
                    target_msg_id,
                ),
            )
            compiled_drop_events = drop_rule.compile_token_drop_events(drop_context)
            if len(compiled_drop_events.event_insert_offsets) > 0:
                token_drop_events = compiled_drop_events
        warmup_commit_token_len = (
            bisect_right(query_epoch_with_gen, len(messages) - 1)
            if msg.is_warmup and not msg.use_context_mask
            else None
        )

        keep_mask = torch.ones(len(full_with_gen_tensor), dtype=torch.bool, device="cpu")
        if token_drop_events is not None:
            for start, end in token_drop_events.effective_ranges:
                keep_mask[start:end] = False

        input_ids = full_with_gen_tensor[keep_mask].contiguous()
        true_positions = torch.arange(len(full_with_gen_tensor), dtype=torch.int32)[keep_mask]
        radix_match_ids = full_with_gen_tensor.to(torch.int64, copy=True)
        prefix_keep_mask = (
            keep_mask[: max(len(radix_match_ids) - 1, 0)].to(dtype=torch.int32).contiguous()
        )

        def compact_pos(raw_pos: int) -> int:
            if raw_pos == 0:
                return 0
            return int(keep_mask[:raw_pos].sum().item())

        # Logical owner starts are diagnostic only; Radix state starts follow query epochs.
        owner_starts: List[dict[str, Any]] = []
        for msg_id in range(len(messages) + 1):
            try:
                start = owner_with_gen.index(msg_id)
            except ValueError:
                continue
            owner_starts.append(
                {
                    "msg_id": msg_id,
                    "raw_start": start,
                    "compact_start": compact_pos(start),
                }
            )

        try:
            gen_prompt_start = owner_with_gen.index(next_assistant_id)
        except ValueError:
            gen_prompt_start = len(full_no_gen)

        radix_state_starts: List[dict[str, Any]] = []
        previous_epoch: int | None = None
        for start, epoch in enumerate(query_epoch_with_gen):
            if epoch == previous_epoch:
                continue
            previous_epoch = epoch
            if start >= len(radix_match_ids):
                continue
            dropped_ids = (
                sorted(
                    self._build_drop_set(
                        {
                            trigger + target_offset: [
                                message_id + target_offset for message_id in message_ids
                            ]
                            for trigger, message_ids in drop_rule.drop_messages.items()
                        },
                        epoch,
                    )
                )
                if isinstance(drop_rule, MessageDropRule)
                else []
            )
            start_meta: dict[str, Any] = {
                "msg_id": epoch,
                "epoch": epoch,
                "raw_start": start,
                "compact_start": compact_pos(start),
            }
            if self.radix_drop_key_mode == "bitmask":
                drop_mask = sum(1 << message_id for message_id in dropped_ids)
                token_id = int(radix_match_ids[start].item())
                radix_match_ids[start] = self._encode_radix_key(token_id, drop_mask)
                start_meta["drop_mask"] = drop_mask
            elif self.radix_drop_key_mode == "symbol":
                start_meta["dropped_ids"] = dropped_ids
            radix_state_starts.append(start_meta)

        # Backward-compatible alias for legacy consumers.
        message_starts = radix_state_starts

        layout = self._compile_delta_layout(
            full_with_gen_tensor,
            token_drop_events,
            keep_mask,
            reposition_raw_boundaries,
            reposition_insert_offsets,
        )
        if layout is None:
            radix_input_ids = radix_match_ids[keep_mask].contiguous()
            radix_commit_key_len = None
        else:
            kept_raw = torch.nonzero(keep_mask, as_tuple=False).view(-1)
            radix_match_ids = layout.records
            radix_input_ids = layout.records[
                layout.token_to_key[kept_raw.to(torch.int64)]
            ].contiguous()
            radix_commit_key_len = (
                None
                if warmup_commit_token_len is None
                else (
                    len(layout.records)
                    if warmup_commit_token_len == len(layout.token_to_key)
                    else int(layout.token_to_key[warmup_commit_token_len])
                )
            )

        return TokenizedResult(
            input_ids=input_ids,
            true_positions=true_positions,
            raw_positions=true_positions,
            radix_input_ids=radix_input_ids,
            radix_match_ids=radix_match_ids,
            prefix_keep_mask=prefix_keep_mask,
            prompt_tokens=len(full_with_gen_tensor),
            full_input_ids=(full_with_gen_tensor if token_drop_events is not None else None),
            full_token_visible_until=(
                token_drop_events.full_token_visible_until
                if token_drop_events is not None
                else None
            ),
            full_keep_mask=(
                keep_mask.to(dtype=torch.int32).contiguous()
                if token_drop_events is not None
                else None
            ),
            drop_event_positions=(
                token_drop_events.event_insert_offsets if token_drop_events is not None else None
            ),
            drop_range_offsets=(
                token_drop_events.range_offsets if token_drop_events is not None else None
            ),
            drop_position_ranges=(
                token_drop_events.raw_ranges if token_drop_events is not None else None
            ),
            drop_effective_event_count=(
                token_drop_events.effective_event_count if token_drop_events is not None else 0
            ),
            reposition_raw_boundaries=reposition_raw_boundaries,
            reposition_insert_offsets=reposition_insert_offsets,
            reposition_input_ids=(full_with_gen_tensor if msg.reposition is not None else None),
            radix_commit_token_len=warmup_commit_token_len,
            radix_commit_key_len=radix_commit_key_len,
            radix_key_virtual_mask=(layout.virtual_mask if layout is not None else None),
            radix_key_to_token=(layout.key_to_token if layout is not None else None),
            radix_token_to_key=(layout.token_to_key if layout is not None else None),
            radix_positions=(layout.positions if layout is not None else None),
            radix_repos_info=(layout.repos_info if layout is not None else None),
            radix_next_position=(layout.next_position if layout is not None else None),
            radix_current_reposition=(
                layout.current_reposition if layout is not None else -1
            ),
            reposition_layout=layout,
            stop_token_seqs=self._build_stop_token_seqs(msg.stop),
            message_meta={
                "raw_len_with_gen": len(full_with_gen_tensor),
                "raw_len_no_gen": len(full_no_gen_tensor),
                "message_starts": message_starts,
                "owner_starts": owner_starts,
                "radix_state_starts": radix_state_starts,
                "cross_owner_tokens": cross_owner_tokens,
                "unstable_rounds": unstable_rounds,
                "no_gen_with_gen_unstable": int(unstable_with_gen),
                "no_gen_with_gen_lcp": no_gen_with_gen_lcp,
                "no_gen_with_gen_lcsuf": no_gen_with_gen_lcsuf,
                "gen_prompt_start": gen_prompt_start,
                "normalized_messages": len(messages),
                "target_offset": target_offset,
                "warmup_commit_token_len": warmup_commit_token_len,
                "safe_mode": int(safe_mode),
                "radix_drop_key_mode": self.radix_drop_key_mode,
                "drop_rule_type": drop_rule.type if drop_rule is not None else None,
                "thinking_template_capability": thinking_template_capability,
            },
        )

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[TokenizedResult]:
        results: List[TokenizedResult] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                results.append(self._chat_tokenize(msg))
            else:
                prompt = msg.text
                input_ids: torch.Tensor = (  # type: ignore
                    self.tokenizer.encode(prompt, return_tensors="pt")
                )
                input_ids = input_ids.view(-1).to(torch.int32)
                layout = self._compile_delta_layout(
                    input_ids,
                    None,
                    torch.ones(len(input_ids), dtype=torch.bool, device="cpu"),
                    None,
                    None,
                )
                radix_match_ids = (
                    layout.records if layout is not None else input_ids.to(torch.int64)
                )
                radix_input_ids = (
                    layout.records[layout.token_to_key]
                    if layout is not None
                    else input_ids.to(torch.int64)
                )
                results.append(
                    TokenizedResult(
                        input_ids=input_ids,
                        true_positions=torch.arange(len(input_ids), dtype=torch.int32),
                        raw_positions=torch.arange(len(input_ids), dtype=torch.int32),
                        radix_input_ids=radix_input_ids,
                        radix_match_ids=radix_match_ids,
                        prefix_keep_mask=torch.ones(
                            max(len(input_ids) - 1, 0), dtype=torch.int32, device="cpu"
                        ),
                        prompt_tokens=len(input_ids),
                        radix_key_virtual_mask=(
                            layout.virtual_mask if layout is not None else None
                        ),
                        radix_key_to_token=(layout.key_to_token if layout is not None else None),
                        radix_token_to_key=(layout.token_to_key if layout is not None else None),
                        radix_positions=(layout.positions if layout is not None else None),
                        radix_repos_info=(layout.repos_info if layout is not None else None),
                        radix_next_position=(layout.next_position if layout is not None else None),
                        radix_current_reposition=(
                            layout.current_reposition if layout is not None else -1
                        ),
                        reposition_layout=layout,
                        stop_token_seqs=self._build_stop_token_seqs(msg.stop),
                        message_meta=None,
                    )
                )
        return results
