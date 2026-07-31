from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.template_provenance import build_template_token_provenance
from transformers import PreTrainedTokenizerBase


@dataclass
class TokenizedResult:
    input_ids: torch.Tensor
    true_positions: torch.Tensor
    radix_input_ids: torch.Tensor
    radix_match_ids: torch.Tensor | None
    prefix_keep_mask: torch.Tensor
    full_input_ids: torch.Tensor | None = None
    full_token_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    drop_event_positions: torch.Tensor | None = None
    drop_range_offsets: torch.Tensor | None = None
    drop_position_ranges: torch.Tensor | None = None
    radix_commit_token_len: int | None = None
    stop_token_seqs: List[List[int]] | None = None
    message_meta: dict | None = None


@dataclass(frozen=True)
class PositionDropPlan:
    event_positions: torch.Tensor
    range_offsets: torch.Tensor
    position_ranges: torch.Tensor
    full_token_visible_until: torch.Tensor


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
        # Be optimistic: many tokenizers accept extra kwargs via **kwargs even
        # when the explicit signature does not list `enable_thinking`.
        self._supports_enable_thinking = True
        # Many recent chat templates support `tools`; gracefully downgrade when unsupported.
        self._supports_tools = True

    def _call_chat_template(
        self,
        messages: List[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        tools: List[Dict[str, Any]] | None,
    ) -> Any:
        kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None and self._supports_enable_thinking:
            kwargs["enable_thinking"] = enable_thinking
        if tools is not None and self._supports_tools:
            kwargs["tools"] = tools
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
                if "tools" in kwargs and "tools" in str(exc):
                    kwargs.pop("tools")
                    self._supports_tools = False
                    continue
                raise

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

    def _serialize_tools_for_system(
        self,
        tools: List[Dict[str, Any]],
        *,
        mode: str,
        forced_tool_name: str | None,
    ) -> str:
        chunks: List[str] = []
        if mode in {"required", "function"}:
            if mode == "function" and forced_tool_name is not None:
                chunks.append(
                    "Tool policy: you must call exactly this function first: "
                    f"{forced_tool_name}."
                )
            else:
                chunks.append("Tool policy: you must call at least one tool before final answer.")
            chunks.append(
                "When calling a tool, output strictly in this format only:\n"
                '<tool_call>{"name":"<tool_name>","arguments":{...}}</tool_call>'
            )
            chunks.append("Do not provide final answer before tool responses are received.")
        chunks.append("Available tools:\n" + self._json_dumps(tools))
        return "\n\n".join(chunks)

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

    def _build_template_messages(
        self,
        raw_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        *,
        tool_choice_mode: str,
        forced_tool_name: str | None,
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

        target_offset = 0
        if tools:
            tool_text = self._serialize_tools_for_system(
                tools,
                mode=tool_choice_mode,
                forced_tool_name=forced_tool_name,
            )
            if messages and str(messages[0].get("role", "")).lower() == "system":
                base = self._normalize_message_content(messages[0].get("content"))
                messages[0]["content"] = (base + "\n\n" + tool_text) if base else tool_text
            else:
                messages.insert(0, {"role": "system", "content": tool_text})
                target_offset = 1

        return messages, target_offset

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
        )

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
        if not stop:
            return None
        dedup: set[tuple[int, ...]] = set()
        result: List[List[int]] = []
        for raw in stop:
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

        for i in range(len(messages)):
            curr = self._apply_chat_template(
                messages[: i + 1],
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
            )
            if i == 0:
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

    def _chat_tokenize(self, msg: TokenizeMsg) -> TokenizedResult:
        assert isinstance(msg.text, list)
        drop_message = self._normalize_drop_message(msg.drop_message)
        if drop_message and self.radix_drop_key_mode != "delta-marker":
            raise ValueError(
                "Token-position Drop requires radix_drop_key_mode='delta-marker'; "
                f"got {self.radix_drop_key_mode!r}."
            )
        tool_choice_mode, forced_tool_name = self._normalize_tool_choice(msg.tool_choice)
        selected_tools = self._select_tools_for_choice(
            msg.tools,
            mode=tool_choice_mode,
            forced_tool_name=forced_tool_name,
        )

        messages, target_offset = self._build_template_messages(
            msg.text,
            selected_tools,
            tool_choice_mode=tool_choice_mode,
            forced_tool_name=forced_tool_name,
            safe_mode=False,
        )
        safe_mode = False
        effective_tools = selected_tools
        try:
            full_no_gen, no_gen_owner, no_gen_query_epoch, unstable_rounds = (
                self._round_by_round_no_gen(messages, msg.enable_thinking, selected_tools)
            )
            if len(messages) == 0:
                no_gen_owner = []
                no_gen_query_epoch = []
            full_with_gen = self._apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=msg.enable_thinking,
                tools=selected_tools,
            )
        except Exception:
            messages, target_offset = self._build_template_messages(
                msg.text,
                selected_tools,
                tool_choice_mode=tool_choice_mode,
                forced_tool_name=forced_tool_name,
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
        if not self._supports_tools:
            effective_tools = None
        effective_enable_thinking = msg.enable_thinking if self._supports_enable_thinking else None
        drop_message = self._shift_and_validate_drop_message(
            drop_message,
            target_offset=target_offset,
            normalized_message_count=len(messages),
        )
        full_no_gen_tensor = torch.tensor(full_no_gen, dtype=torch.int32, device="cpu")
        full_with_gen_tensor = torch.tensor(full_with_gen, dtype=torch.int32, device="cpu")
        next_assistant_id = len(messages)
        generation_prompt_epoch = next_assistant_id
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
                new_epoch=generation_prompt_epoch,
            )
            if len(full_no_gen) > 0
            else [generation_prompt_epoch] * len(full_with_gen)
        )

        cross_owner_tokens = 0
        if drop_message:
            canonical_no_gen = self._render_chat_template(
                messages,
                add_generation_prompt=False,
                enable_thinking=effective_enable_thinking,
                tools=effective_tools,
            )
            canonical_with_gen = self._render_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=effective_enable_thinking,
                tools=effective_tools,
            )
            provenance = build_template_token_provenance(
                self.tokenizer,
                messages,
                canonical_text=canonical_with_gen,
                canonical_no_generation_text=canonical_no_gen,
                expected_input_ids=full_with_gen,
                tools=effective_tools,
                add_generation_prompt=True,
                enable_thinking=effective_enable_thinking,
            )
            owner_with_gen = provenance.owners
            cross_owner_tokens = provenance.cross_owner_tokens

        owner_ranges = self._build_owner_position_ranges(owner_with_gen)

        target_msg_id = self._resolve_target_msg_id(
            msg,
            len(messages),
            target_offset=target_offset,
        )
        warmup_commit_token_len = (
            bisect_right(query_epoch_with_gen, target_msg_id)
            if msg.is_warmup and not msg.use_context_mask
            else None
        )
        dropped_for_target = self._build_drop_set(drop_message, target_msg_id)
        position_drop_plan = (
            self._build_position_drop_plan(
                drop_message,
                query_epoch_with_gen,
                owner_ranges,
            )
            if drop_message
            else None
        )

        keep_mask = torch.ones(len(full_with_gen_tensor), dtype=torch.bool, device="cpu")
        for start, end in self._ranges_for_messages(owner_ranges, dropped_for_target):
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
            dropped_ids = sorted(self._build_drop_set(drop_message, epoch))
            start_meta: dict[str, Any] = {
                "msg_id": epoch,
                "epoch": epoch,
                "raw_start": start,
                "compact_start": compact_pos(start),
            }
            if self.radix_drop_key_mode == "bitmask":
                drop_mask = self._build_drop_mask(drop_message, epoch)
                token_id = int(radix_match_ids[start].item())
                radix_match_ids[start] = self._encode_radix_key(token_id, drop_mask)
                start_meta["drop_mask"] = drop_mask
            elif self.radix_drop_key_mode == "symbol":
                start_meta["dropped_ids"] = dropped_ids
            radix_state_starts.append(start_meta)

        # Backward-compatible alias for legacy consumers.
        message_starts = radix_state_starts

        radix_input_ids = radix_match_ids[keep_mask].contiguous()

        return TokenizedResult(
            input_ids=input_ids,
            true_positions=true_positions,
            radix_input_ids=radix_input_ids,
            radix_match_ids=radix_match_ids,
            prefix_keep_mask=prefix_keep_mask,
            full_input_ids=(full_with_gen_tensor if drop_message else None),
            full_token_visible_until=(
                position_drop_plan.full_token_visible_until
                if position_drop_plan is not None
                else None
            ),
            full_keep_mask=(keep_mask.to(dtype=torch.int32).contiguous() if drop_message else None),
            drop_event_positions=(
                position_drop_plan.event_positions if position_drop_plan is not None else None
            ),
            drop_range_offsets=(
                position_drop_plan.range_offsets if position_drop_plan is not None else None
            ),
            drop_position_ranges=(
                position_drop_plan.position_ranges if position_drop_plan is not None else None
            ),
            radix_commit_token_len=warmup_commit_token_len,
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
                results.append(
                    TokenizedResult(
                        input_ids=input_ids,
                        true_positions=torch.arange(len(input_ids), dtype=torch.int32),
                        radix_input_ids=input_ids.to(torch.int64),
                        radix_match_ids=input_ids.to(torch.int64),
                        prefix_keep_mask=torch.ones(
                            max(len(input_ids) - 1, 0), dtype=torch.int32, device="cpu"
                        ),
                        stop_token_seqs=self._build_stop_token_seqs(msg.stop),
                        message_meta=None,
                    )
                )
        return results
