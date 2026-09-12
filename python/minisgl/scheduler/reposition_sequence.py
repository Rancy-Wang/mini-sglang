from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from minisgl.kernel.radix_reposition import TOKEN_KIND
from minisgl.message import (
    RepositionOpenMsg,
    RepositionStepMsg,
    StagedRepositionInit,
    UserMsg,
)


@dataclass
class SchedulerRepositionSequence:
    """Scheduler-owned mirror of one staged Reposition program.

    The tokenizer transfers the immutable program once.  Later turns only move
    a cursor and, when necessary, the position delta for one Reposition event.
    """

    uid: int
    init: StagedRepositionInit
    current_records: torch.Tensor
    current_positions: torch.Tensor
    current_repos: torch.Tensor
    active_raw: torch.Tensor
    raw_cursor: int = 0
    drop_cursor: int = 0
    current_reposition: int = -1
    final_dispatched: bool = False

    @classmethod
    def from_open(cls, message: RepositionOpenMsg) -> SchedulerRepositionSequence:
        init = message.init
        cls._validate_init(init)
        return cls(
            uid=message.uid,
            init=init,
            current_records=init.radix_records,
            current_positions=init.initial_positions,
            current_repos=init.initial_repos,
            active_raw=torch.empty(0, dtype=torch.int32, device="cpu"),
        )

    @staticmethod
    def _validate_init(init: StagedRepositionInit) -> None:
        tensors = {
            "input_ids": init.input_ids,
            "radix_records": init.radix_records,
            "radix_key_virtual_mask": init.radix_key_virtual_mask,
            "radix_key_to_token": init.radix_key_to_token,
            "radix_token_to_key": init.radix_token_to_key,
            "initial_positions": init.initial_positions,
            "initial_repos": init.initial_repos,
            "drop_event_positions": init.drop_event_positions,
            "drop_range_offsets": init.drop_range_offsets,
            "drop_position_ranges": init.drop_position_ranges,
        }
        if init.full_token_visible_until is not None:
            tensors["full_token_visible_until"] = init.full_token_visible_until
        for name, tensor in tensors.items():
            if tensor.device.type != "cpu":
                raise ValueError(f"Staged Reposition {name} must be a CPU tensor.")

        token_count = len(init.input_ids)
        record_count = len(init.radix_records)
        if init.input_ids.ndim != 1 or init.input_ids.dtype != torch.int32:
            raise ValueError("Staged Reposition input_ids must be CPU int32 [N].")
        if init.radix_records.ndim != 2 or init.radix_records.shape[1] != 4:
            raise ValueError("Staged Reposition radix_records must have shape [K, 4].")
        if init.radix_records.dtype != torch.int32:
            raise ValueError("Staged Reposition radix_records must use int32.")
        if (
            init.radix_key_virtual_mask.ndim != 1
            or init.radix_key_virtual_mask.dtype != torch.bool
            or len(init.radix_key_virtual_mask) != record_count
        ):
            raise ValueError("Staged Reposition virtual-mask shape or dtype is invalid.")
        if (
            init.radix_key_to_token.ndim != 1
            or len(init.radix_key_to_token) != record_count
            or init.radix_token_to_key.ndim != 1
            or len(init.radix_token_to_key) != token_count
        ):
            raise ValueError("Staged Reposition Radix mappings have inconsistent lengths.")
        if (
            init.initial_positions.ndim != 1
            or len(init.initial_positions) != token_count
            or init.initial_repos.ndim != 1
            or len(init.initial_repos) != token_count
        ):
            raise ValueError("Staged Reposition initial position arrays are incomplete.")
        if token_count and bool(
            torch.any(
                (init.radix_token_to_key < 0) | (init.radix_token_to_key >= record_count)
            ).item()
        ):
            raise ValueError("Staged Reposition token-to-key mapping is out of range.")
        if (
            init.drop_event_positions.ndim != 1
            or init.drop_range_offsets.ndim != 1
            or len(init.drop_range_offsets) != len(init.drop_event_positions) + 1
            or init.drop_position_ranges.ndim != 1
            or len(init.drop_position_ranges) % 2 != 0
        ):
            raise ValueError("Staged Reposition Drop CSR metadata is malformed.")
        if int(init.drop_range_offsets[0]) != 0 or int(init.drop_range_offsets[-1]) * 2 != len(
            init.drop_position_ranges
        ):
            raise ValueError("Staged Reposition Drop CSR offsets are inconsistent.")
        if len(init.drop_range_offsets) > 1 and bool(
            torch.any(init.drop_range_offsets[1:] < init.drop_range_offsets[:-1]).item()
        ):
            raise ValueError("Staged Reposition Drop CSR offsets must be monotonic.")
        if len(init.drop_event_positions) > 1 and bool(
            torch.any(init.drop_event_positions[1:] < init.drop_event_positions[:-1]).item()
        ):
            raise ValueError("Staged Reposition Drop events must be ordered.")
        if (
            init.full_token_visible_until is not None
            and len(init.full_token_visible_until) != token_count
        ):
            raise ValueError("Staged Reposition visibility metadata has the wrong length.")

    def _drop_count_before(self, boundary: int) -> int:
        return int(
            torch.searchsorted(
                self.init.drop_event_positions,
                boundary,
                side="right",
            ).item()
        )

    def _drop_inside(self, end: int) -> bool:
        if self.drop_cursor >= len(self.init.drop_event_positions):
            return False
        positions = self.init.drop_event_positions[self.drop_cursor :]
        return bool(torch.any((positions > self.raw_cursor) & (positions <= end)).item())

    def _active_after_events(self, boundary: int) -> torch.Tensor:
        active = torch.cat(
            (
                self.active_raw,
                torch.arange(self.raw_cursor, boundary, dtype=torch.int32, device="cpu"),
            )
        )
        active = torch.unique(active, sorted=True)
        cursor = self.drop_cursor
        while cursor < len(self.init.drop_event_positions):
            insertion = int(self.init.drop_event_positions[cursor])
            if insertion > boundary:
                break
            begin = int(self.init.drop_range_offsets[cursor])
            finish = int(self.init.drop_range_offsets[cursor + 1])
            keep = torch.ones(len(active), dtype=torch.bool, device="cpu")
            for range_index in range(begin, finish):
                start = int(self.init.drop_position_ranges[2 * range_index])
                end = int(self.init.drop_position_ranges[2 * range_index + 1])
                keep &= ~((active >= start) & (active < end))
            active = active[keep]
            cursor += 1
        mask = torch.zeros(boundary, dtype=torch.int32, device="cpu")
        mask[active.to(torch.int64)] = 1
        return mask

    def _apply_transition(self, message: RepositionStepMsg) -> bool:
        values = (
            message.transition_raw_tokens,
            message.transition_new_positions,
            message.transition_boundary,
        )
        present = tuple(value is not None for value in values)
        if any(present) and not all(present):
            raise ValueError("Staged Reposition transition delta is incomplete.")
        if not all(present):
            return False

        assert message.transition_raw_tokens is not None
        assert message.transition_new_positions is not None
        assert message.transition_boundary is not None
        raw_tokens = message.transition_raw_tokens
        new_positions = message.transition_new_positions
        if (
            raw_tokens.device.type != "cpu"
            or raw_tokens.ndim != 1
            or raw_tokens.dtype != torch.int32
            or new_positions.device.type != "cpu"
            or new_positions.ndim != 1
            or new_positions.dtype != torch.int32
            or len(raw_tokens) != len(new_positions)
        ):
            raise ValueError("Staged Reposition transition tensors are invalid.")
        if len(raw_tokens) and bool(
            torch.any((raw_tokens < 0) | (raw_tokens >= self.raw_cursor)).item()
        ):
            raise ValueError("Staged Reposition transition refers to unseen raw tokens.")
        if message.transition_boundary <= self.current_reposition:
            raise ValueError("Staged Reposition boundaries must increase monotonically.")

        raw_index = raw_tokens.to(torch.int64)
        self.current_positions[raw_index] = new_positions
        self.current_repos[raw_index] = message.transition_boundary
        token_rows = self.init.radix_token_to_key[raw_index]
        self.current_records[token_rows, 0] = TOKEN_KIND
        self.current_records[token_rows, 2] = message.transition_boundary
        self.current_records[token_rows, 3] = new_positions
        self.current_reposition = message.transition_boundary
        return True

    def materialize(self, message: RepositionStepMsg) -> UserMsg:
        if message.uid != self.uid:
            raise ValueError("Staged Reposition step UID does not match its open session.")
        if self.final_dispatched:
            raise ValueError("Staged Reposition received a step after final dispatch.")
        token_count = len(self.init.input_ids)
        if message.end < self.raw_cursor or message.end > token_count:
            raise ValueError("Staged Reposition step cursor is out of order or range.")

        transitioned = self._apply_transition(message)
        if message.end == self.raw_cursor and not (message.is_final and transitioned):
            raise ValueError("Staged Reposition step made no raw-token progress.")
        if message.is_final and message.end != token_count:
            raise ValueError("Final staged Reposition step must cover the raw token stream.")
        if message.is_final != (message.radix_commit_key_len is None):
            raise ValueError("Staged Reposition commit length does not match final state.")
        if message.radix_current_reposition != (
            self.init.radix_final_reposition if message.is_final else self.current_reposition
        ):
            raise ValueError("Staged Reposition current boundary is inconsistent.")

        commit_key_len = (
            len(self.current_records) if message.is_final else int(message.radix_commit_key_len)
        )
        if commit_key_len < 0 or commit_key_len > len(self.current_records):
            raise ValueError("Staged Reposition commit length is out of range.")

        new_raw = torch.arange(
            self.raw_cursor,
            message.end,
            dtype=torch.int32,
            device="cpu",
        )
        execution_raw = torch.unique(torch.cat((self.active_raw, new_raw)), sorted=True)
        raw_index = execution_raw.to(torch.int64)
        token_to_key = self.init.radix_token_to_key[: message.end].contiguous()
        if len(raw_index) and bool(
            torch.any(self.init.radix_token_to_key[raw_index] >= commit_key_len).item()
        ):
            raise ValueError("Staged Reposition commit boundary omits an execution token.")

        use_context_mask = self._drop_inside(message.end)
        execution_mask = torch.zeros(message.end, dtype=torch.int32, device="cpu")
        execution_mask[raw_index] = 1
        drop_count = self._drop_count_before(message.end)
        drop_range_count = int(self.init.drop_range_offsets[drop_count])
        active_after = self._active_after_events(message.end)
        post_prefill_keep = active_after if message.is_final and use_context_mask else None
        sampling_params = (
            self.init.sampling_params
            if message.is_final
            else replace(self.init.sampling_params, max_tokens=1, ignore_eos=True)
        )

        result = UserMsg(
            uid=self.uid,
            input_ids=self.init.input_ids[raw_index].contiguous(),
            true_positions=self.current_positions[raw_index].contiguous(),
            raw_positions=execution_raw,
            radix_input_ids=self.current_records[
                self.init.radix_token_to_key[raw_index]
            ].contiguous(),
            radix_match_ids=self.current_records[:commit_key_len],
            sampling_params=sampling_params,
            prompt_tokens=self.init.prompt_tokens,
            radix_key_virtual_mask=self.init.radix_key_virtual_mask[:commit_key_len].contiguous(),
            radix_key_to_token=self.init.radix_key_to_token[:commit_key_len].contiguous(),
            radix_token_to_key=token_to_key,
            radix_commit_key_len=None if message.is_final else commit_key_len,
            drop_event_positions=self.init.drop_event_positions[:drop_count],
            drop_range_offsets=self.init.drop_range_offsets[: drop_count + 1],
            drop_position_ranges=self.init.drop_position_ranges[: 2 * drop_range_count],
            drop_effective_event_count=drop_count,
            radix_positions=self.current_positions[: message.end].contiguous(),
            radix_repos_info=self.current_repos[: message.end].contiguous(),
            radix_next_position=(self.init.radix_next_position if message.is_final else None),
            radix_current_reposition=message.radix_current_reposition,
            enable_thinking=self.init.enable_thinking,
            stop=self.init.stop,
            stop_token_seqs=self.init.stop_token_seqs,
            message_meta=self.init.message_meta,
            is_warmup=not message.is_final or self.init.request_is_warmup,
            internal_uid=self.init.internal_uid,
            prefix_keep_mask=execution_mask,
            full_input_ids=(
                self.init.input_ids[: message.end].contiguous() if use_context_mask else None
            ),
            full_token_visible_until=(
                self.init.full_token_visible_until[: message.end].contiguous()
                if use_context_mask and self.init.full_token_visible_until is not None
                else None
            ),
            full_keep_mask=execution_mask if use_context_mask else None,
            use_context_mask=use_context_mask,
            context_compact_stream=use_context_mask,
            context_post_prefill_keep_mask=post_prefill_keep,
            request_received_ns=self.init.request_received_ns,
            tokenize_invocations=self.init.tokenize_invocations,
            chat_template_invocations=self.init.chat_template_invocations,
            context_stage_count=message.context_stage_count,
            radix_compile_ns=self.init.radix_compile_ns,
            radix_match_ns=message.radix_match_ns,
            retry_plan_ns=message.retry_plan_ns,
            reposition_transition_count=message.reposition_transition_count,
            reposition_h2d_bytes=message.reposition_h2d_bytes,
            reposition_d2h_bytes=message.reposition_d2h_bytes,
            reposition_ipc_tensor_bytes=message.reposition_ipc_tensor_bytes,
        )

        self.active_raw = torch.nonzero(active_after, as_tuple=False).view(-1).to(torch.int32)
        while (
            self.drop_cursor < len(self.init.drop_event_positions)
            and int(self.init.drop_event_positions[self.drop_cursor]) <= message.end
        ):
            self.drop_cursor += 1
        self.raw_cursor = message.end
        self.final_dispatched = message.is_final
        return result


__all__ = ["SchedulerRepositionSequence"]
