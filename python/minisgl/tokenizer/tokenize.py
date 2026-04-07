from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.message import TokenizeMsg

if TYPE_CHECKING:
    from transformers import LlamaTokenizer


@dataclass
class TokenizedResult:
    input_ids: torch.Tensor
    true_positions: torch.Tensor
    radix_input_ids: torch.Tensor
    radix_match_ids: torch.Tensor | None
    prefix_keep_mask: torch.Tensor
    message_meta: dict | None = None


class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer
        # Be optimistic: many tokenizers accept extra kwargs via **kwargs even
        # when the explicit signature does not list `enable_thinking`.
        self._supports_enable_thinking = True

    def _apply_chat_template(
        self,
        messages: List[dict[str, str]],
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
    ) -> List[int]:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None and self._supports_enable_thinking:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            # Only downgrade when the failure is specifically due to the unknown
            # `enable_thinking` kwarg. Keep other TypeErrors visible.
            if "enable_thinking" in kwargs and "enable_thinking" in str(exc):
                kwargs.pop("enable_thinking")
                self._supports_enable_thinking = False
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            raise

    def _normalize_drop_message(self, drop_message: dict[int, List[int]] | None) -> dict[int, List[int]]:
        if not drop_message:
            return {}
        normalized: dict[int, List[int]] = {}
        for k, value in drop_message.items():
            n = int(k)
            if n < 0 or n >= 32:
                raise ValueError(f"drop_message key out of range [0, 31]: {n}")
            ids = [int(v) for v in value]
            for msg_id in ids:
                if msg_id < 0 or msg_id >= 32:
                    raise ValueError(f"drop_message id out of range [0, 31]: {msg_id}")
            normalized[n] = ids
        return normalized

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
    def _encode_radix_key(token_id: int, drop_mask: int) -> int:
        # Keep token id in low 32 bits and drop bitset in high 32 bits.
        # Normalize to signed int64 range before writing into torch.int64 tensor.
        encoded = ((token_id & 0xFFFFFFFF) | (drop_mask << 32)) & ((1 << 64) - 1)
        if encoded >= (1 << 63):
            encoded -= 1 << 64
        return encoded

    @staticmethod
    def _resolve_target_msg_id(msg: TokenizeMsg) -> int:
        if msg.target_msg_id is None:
            target_msg_id = len(msg.text) - 1 if msg.is_warmup else len(msg.text)
        else:
            target_msg_id = int(msg.target_msg_id)
        if target_msg_id < 0 or target_msg_id > len(msg.text):
            raise ValueError(
                f"target_msg_id out of range [0, {len(msg.text)}]: {target_msg_id}"
            )
        return target_msg_id

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
        lcp = self._common_prefix_len(prev_ids, curr_ids)
        lcsuf = self._common_suffix_len(prev_ids, curr_ids, lcp)
        unstable = lcp < len(prev_ids)

        old_mid_start = lcp
        old_mid_end = len(prev_ids) - lcsuf
        new_mid_start = lcp
        new_mid_end = len(curr_ids) - lcsuf

        old_mid_owner = prev_owner[old_mid_start:old_mid_end]
        new_mid_len = new_mid_end - new_mid_start
        reuse_len = min(len(old_mid_owner), new_mid_len)

        curr_owner = [-1] * len(curr_ids)
        if lcp > 0:
            curr_owner[:lcp] = prev_owner[:lcp]
        if reuse_len > 0:
            curr_owner[new_mid_start : new_mid_start + reuse_len] = old_mid_owner[:reuse_len]
        if new_mid_len > reuse_len:
            curr_owner[new_mid_start + reuse_len : new_mid_end] = [new_owner] * (
                new_mid_len - reuse_len
            )
        if lcsuf > 0:
            curr_owner[new_mid_end:] = prev_owner[len(prev_ids) - lcsuf :]

        for idx in range(len(curr_owner)):
            if curr_owner[idx] < 0:
                curr_owner[idx] = new_owner
        return curr_owner, lcp, lcsuf, unstable

    def _round_by_round_no_gen(
        self, messages: List[dict[str, str]], enable_thinking: bool | None
    ) -> tuple[List[int], List[int], int]:
        assembled: List[int] = []
        owner: List[int] = []
        unstable_rounds = 0

        for i in range(len(messages)):
            curr = self._apply_chat_template(
                messages[: i + 1],
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            if i == 0:
                assembled = curr
                owner = [0] * len(curr)
                continue

            owner, _, _, unstable = self._merge_owner_track(
                assembled, owner, curr, new_owner=i
            )
            unstable_rounds += int(unstable)
            assembled = curr

        return assembled, owner, unstable_rounds

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

        full_no_gen, no_gen_owner, unstable_rounds = self._round_by_round_no_gen(
            msg.text, msg.enable_thinking
        )
        if len(msg.text) == 0:
            no_gen_owner = []
        full_with_gen = self._apply_chat_template(
            msg.text, add_generation_prompt=True, enable_thinking=msg.enable_thinking
        )
        full_no_gen_tensor = torch.tensor(full_no_gen, dtype=torch.int32, device="cpu")
        full_with_gen_tensor = torch.tensor(full_with_gen, dtype=torch.int32, device="cpu")
        next_assistant_id = len(msg.text)
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

        target_msg_id = self._resolve_target_msg_id(msg)
        dropped_for_target = self._build_drop_set(drop_message, target_msg_id)
        owner_tensor = torch.tensor(owner_with_gen, dtype=torch.int32, device="cpu")

        keep_mask = torch.ones(len(full_with_gen_tensor), dtype=torch.bool, device="cpu")
        for msg_id in dropped_for_target:
            keep_mask &= owner_tensor != msg_id

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

        # Apply drop mask encoding to first token of every message in the full stream.
        message_starts: List[dict[str, int]] = []
        for msg_id in range(len(msg.text)):
            try:
                start = owner_with_gen.index(msg_id)
            except ValueError:
                continue
            if start >= len(radix_match_ids):
                continue
            encoded_pos = compact_pos(start)
            drop_mask = self._build_drop_mask(drop_message, msg_id)
            token_id = int(radix_match_ids[start].item())
            radix_match_ids[start] = self._encode_radix_key(token_id, drop_mask)
            message_starts.append(
                {
                    "msg_id": msg_id,
                    "raw_start": start,
                    "compact_start": encoded_pos,
                    "drop_mask": drop_mask,
                }
            )

        # Also encode the generation-prompt start as the "next assistant message" start.
        try:
            gen_prompt_start = owner_with_gen.index(next_assistant_id)
        except ValueError:
            gen_prompt_start = len(full_no_gen)
        if gen_prompt_start < len(radix_match_ids):
            encoded_pos = compact_pos(gen_prompt_start)
            drop_mask = self._build_drop_mask(drop_message, next_assistant_id)
            token_id = int(radix_match_ids[gen_prompt_start].item())
            radix_match_ids[gen_prompt_start] = self._encode_radix_key(token_id, drop_mask)
            message_starts.append(
                {
                    "msg_id": next_assistant_id,
                    "raw_start": gen_prompt_start,
                    "compact_start": encoded_pos,
                    "drop_mask": drop_mask,
                }
            )

        radix_input_ids = radix_match_ids[keep_mask].contiguous()

        return TokenizedResult(
            input_ids=input_ids,
            true_positions=true_positions,
            radix_input_ids=radix_input_ids,
            radix_match_ids=radix_match_ids,
            prefix_keep_mask=prefix_keep_mask,
            message_meta={
                "raw_len_with_gen": len(full_with_gen_tensor),
                "raw_len_no_gen": len(full_no_gen_tensor),
                "message_starts": message_starts,
                "unstable_rounds": unstable_rounds,
                "no_gen_with_gen_unstable": int(unstable_with_gen),
                "no_gen_with_gen_lcp": no_gen_with_gen_lcp,
                "no_gen_with_gen_lcsuf": no_gen_with_gen_lcsuf,
                "gen_prompt_start": gen_prompt_start,
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
                        message_meta=None,
                    )
                )
        return results
