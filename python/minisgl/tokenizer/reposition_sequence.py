from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from minisgl.kernel.radix_reposition import (
    DELTA_KIND,
    REPOSITION_KIND,
    TOKEN_KIND,
    RadixRepositionLayout,
    compile_radix_reposition_layout,
)
from minisgl.message import RepositionOpenMsg, TokenizeMsg, UserMsg, WarmupAckMsg

from .tokenize import TokenizedResult


def _effective_drop_wire(
    result: TokenizedResult,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if result.drop_event_positions is None:
        return (
            torch.empty(0, dtype=torch.int32, device="cpu"),
            torch.zeros(1, dtype=torch.int32, device="cpu"),
            torch.empty(0, dtype=torch.int32, device="cpu"),
        )
    assert result.drop_range_offsets is not None
    assert result.drop_position_ranges is not None
    count = min(result.drop_effective_event_count, len(result.drop_event_positions))
    range_count = int(result.drop_range_offsets[count])
    return (
        result.drop_event_positions[:count].contiguous(),
        result.drop_range_offsets[: count + 1].contiguous(),
        result.drop_position_ranges[: 2 * range_count].contiguous(),
    )


@dataclass
class RepositionSequenceState:
    """Tokenizer-owned cursor over one precompiled Drop/Reposition program.

    The large event timeline lives only in this object.  Scheduler turns receive
    immutable tensor views and return a small acknowledgement; they never parse
    the interface, tokenize text, or run the full Radix compiler again.
    """

    request: TokenizeMsg
    tokenized: TokenizedResult
    drop_event_positions: torch.Tensor
    drop_range_offsets: torch.Tensor
    drop_position_ranges: torch.Tensor
    layout: RadixRepositionLayout | None = None
    marker_ids: tuple[int, ...] = ()
    step_token_budget: int = 0
    raw_cursor: int = 0
    drop_cursor: int = 0
    current_stage: int = 0
    current_reposition: int = -1
    active_raw: torch.Tensor | None = None
    current_positions: torch.Tensor | None = None
    current_repos: torch.Tensor | None = None
    current_records: torch.Tensor | None = None
    in_flight_end: int = 0
    in_flight_final: bool = False
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    transition_count: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0

    @classmethod
    def pending(cls, request: TokenizeMsg, tokenized: TokenizedResult) -> RepositionSequenceState:
        if tokenized.reposition_input_ids is None:
            raise ValueError("Reposition sequence requires the immutable raw token stream.")
        if tokenized.reposition_raw_boundaries is None:
            raise ValueError("Reposition sequence requires raw Reposition boundaries.")
        if tokenized.reposition_insert_offsets is None:
            raise ValueError("Reposition sequence requires Reposition insertion offsets.")
        event_positions, range_offsets, position_ranges = _effective_drop_wire(tokenized)
        return cls(
            request=request,
            tokenized=tokenized,
            drop_event_positions=event_positions,
            drop_range_offsets=range_offsets,
            drop_position_ranges=position_ranges,
        )

    def open_msg(self) -> RepositionOpenMsg:
        assert self.tokenized.reposition_input_ids is not None
        return RepositionOpenMsg(
            uid=self.request.uid,
            full_token_count=len(self.tokenized.reposition_input_ids),
            drop_event_positions=self.drop_event_positions,
            drop_range_offsets=self.drop_range_offsets,
            drop_position_ranges=self.drop_position_ranges,
        )

    def compile(self, marker_ids: list[int], *, step_token_budget: int) -> None:
        if self.layout is not None:
            raise RuntimeError("A Reposition sequence was compiled more than once.")
        if step_token_budget <= 0:
            raise ValueError("Reposition step token budget must be positive.")
        if len(marker_ids) != len(self.drop_event_positions):
            raise ValueError("Scheduler marker lease does not cover the effective Drop events.")
        assert self.tokenized.reposition_input_ids is not None
        assert self.tokenized.reposition_raw_boundaries is not None
        assert self.tokenized.reposition_insert_offsets is not None
        self.marker_ids = tuple(int(marker) for marker in marker_ids)
        self.step_token_budget = step_token_budget
        self.layout = compile_radix_reposition_layout(
            self.tokenized.reposition_input_ids,
            self.drop_event_positions,
            self.drop_range_offsets,
            self.drop_position_ranges,
            torch.tensor(self.marker_ids, dtype=torch.int32, device="cpu"),
            self.tokenized.reposition_raw_boundaries,
            self.tokenized.reposition_insert_offsets,
        )
        self.active_raw = torch.empty(0, dtype=torch.int32, device="cpu")
        self.current_positions = self.layout.birth_positions.clone()

        stage_boundaries = torch.full(
            (len(self.layout.transition_offsets),), -1, dtype=torch.int32, device="cpu"
        )
        effective = self.layout.effective_reposition_stages
        for event, stage in enumerate(effective.tolist()):
            if stage > 0:
                stage_boundaries[stage] = self.tokenized.reposition_raw_boundaries[event]
        self.current_repos = stage_boundaries[self.layout.birth_stages.to(torch.int64)]
        self.current_records = self.layout.records.clone()
        token_rows = self.layout.token_to_key
        self.current_records[token_rows, 0] = TOKEN_KIND
        self.current_records[token_rows, 2] = self.current_repos
        self.current_records[token_rows, 3] = self.current_positions

    @property
    def is_compiled(self) -> bool:
        return self.layout is not None

    def _next_effective_reposition(self) -> tuple[int, int] | None:
        assert self.layout is not None
        assert self.tokenized.reposition_insert_offsets is not None
        for event, stage in enumerate(self.layout.effective_reposition_stages.tolist()):
            insertion = int(self.tokenized.reposition_insert_offsets[event])
            if stage > self.current_stage and insertion > self.raw_cursor:
                return insertion, stage
        return None

    def _event_key_position(self, *, insertion: int, kind: int) -> int | None:
        assert self.layout is not None
        assert self.current_records is not None
        if kind == REPOSITION_KIND:
            assert self.tokenized.reposition_raw_boundaries is not None
            assert self.tokenized.reposition_insert_offsets is not None
            candidates = [
                int(self.tokenized.reposition_raw_boundaries[event])
                for event, offset in enumerate(self.tokenized.reposition_insert_offsets.tolist())
                if int(offset) == insertion
                and int(self.layout.effective_reposition_stages[event]) > 0
            ]
            if not candidates:
                return None
            rows = torch.nonzero(
                (self.current_records[:, 0] == REPOSITION_KIND)
                & (self.current_records[:, 1] == candidates[0]),
                as_tuple=False,
            ).view(-1)
        else:
            candidates = [
                self.marker_ids[event]
                for event, offset in enumerate(self.drop_event_positions.tolist())
                if int(offset) == insertion
            ]
            if not candidates:
                return None
            rows = torch.nonzero(
                (self.current_records[:, 0] == DELTA_KIND)
                & (self.current_records[:, 1] == candidates[0]),
                as_tuple=False,
            ).view(-1)
        return int(rows[0]) if len(rows) > 0 else None

    def _commit_key_len(self, end: int) -> int:
        assert self.layout is not None
        raw_count = len(self.layout.token_to_key)
        if end == raw_count:
            cap = len(self.layout.records)
        else:
            cap = int(self.layout.token_to_key[end])
        for kind in (DELTA_KIND, REPOSITION_KIND):
            event_pos = self._event_key_position(insertion=end, kind=kind)
            if event_pos is not None:
                cap = min(cap, event_pos)
        return cap

    def _drop_count_before(self, boundary: int) -> int:
        return int(torch.searchsorted(self.drop_event_positions, boundary, side="left").item())

    def _drop_inside(self, end: int) -> bool:
        if self.drop_cursor >= len(self.drop_event_positions):
            return False
        positions = self.drop_event_positions[self.drop_cursor :]
        return bool(torch.any((positions > self.raw_cursor) & (positions < end)).item())

    def _drop_wire_before(
        self, boundary: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        count = self._drop_count_before(boundary)
        range_count = int(self.drop_range_offsets[count])
        return (
            self.drop_event_positions[:count],
            self.drop_range_offsets[: count + 1],
            self.drop_position_ranges[: 2 * range_count],
            count,
        )

    def build_next_msg(self) -> UserMsg:
        if self.layout is None or self.current_records is None:
            raise RuntimeError("Reposition sequence must be compiled before dispatch.")
        if self.in_flight_end > self.raw_cursor:
            raise RuntimeError("A Reposition step is already awaiting Scheduler acknowledgement.")
        assert self.active_raw is not None
        assert self.current_positions is not None
        assert self.current_repos is not None
        assert self.tokenized.reposition_input_ids is not None

        raw_count = len(self.tokenized.reposition_input_ids)
        next_reposition = self._next_effective_reposition()
        if len(self.layout.transition_offsets) == 1:
            # ``reposition=[]`` is a structured Retry source, not a staged
            # request. It retains the ordinary single-Prefill mask/Extend path.
            end = raw_count
        else:
            end = min(raw_count, self.raw_cursor + self.step_token_budget)
        if next_reposition is not None:
            end = min(end, next_reposition[0])
        if end <= self.raw_cursor:
            raise RuntimeError("Reposition sequence did not make raw-token progress.")

        new_raw = torch.arange(self.raw_cursor, end, dtype=torch.int32, device="cpu")
        execution_raw = torch.cat((self.active_raw, new_raw))
        execution_raw = torch.unique(execution_raw, sorted=True)
        raw_index = execution_raw.to(torch.int64)
        commit_key_len = self._commit_key_len(end)
        records = self.current_records[:commit_key_len].contiguous()
        virtual_mask = self.layout.virtual_mask[:commit_key_len].contiguous()
        key_to_token = self.layout.key_to_token[:commit_key_len].contiguous()
        token_to_key = self.layout.token_to_key[:end].contiguous()
        marker_rows = (
            (records[:, 0] == DELTA_KIND)
            if len(records) > 0
            else torch.empty(0, dtype=torch.bool, device="cpu")
        )
        step_markers = tuple(int(value) for value in records[marker_rows, 1].tolist())

        use_context_mask = self._drop_inside(end)
        execution_mask = torch.zeros(end, dtype=torch.bool, device="cpu")
        execution_mask[raw_index] = True
        drop_positions, drop_offsets, drop_ranges, drop_count = self._drop_wire_before(end)
        is_final = end == raw_count and next_reposition is None
        post_prefill_keep = None
        if is_final and use_context_mask:
            post_prefill_keep = self._active_after_events(end)

        sampling_params = (
            self.request.sampling_params
            if is_final
            else replace(self.request.sampling_params, max_tokens=1, ignore_eos=True)
        )
        self.in_flight_end = end
        self.in_flight_final = is_final
        return UserMsg(
            uid=self.request.uid,
            input_ids=self.tokenized.reposition_input_ids[raw_index].contiguous(),
            true_positions=self.current_positions[raw_index].contiguous(),
            raw_positions=execution_raw,
            radix_input_ids=records[token_to_key[raw_index]].contiguous(),
            radix_match_ids=records,
            sampling_params=sampling_params,
            prompt_tokens=self.tokenized.prompt_tokens,
            radix_key_virtual_mask=virtual_mask,
            radix_key_to_token=key_to_token,
            radix_token_to_key=token_to_key,
            radix_commit_key_len=commit_key_len,
            radix_marker_ids=list(step_markers),
            drop_event_positions=drop_positions,
            drop_range_offsets=drop_offsets,
            drop_position_ranges=drop_ranges,
            drop_effective_event_count=drop_count,
            radix_positions=self.current_positions[:end].contiguous(),
            radix_repos_info=self.current_repos[:end].contiguous(),
            radix_next_position=self.layout.next_position if is_final else None,
            radix_current_reposition=(
                self.layout.current_reposition if is_final else self.current_reposition
            ),
            enable_thinking=self.request.enable_thinking,
            stop=self.request.stop,
            stop_token_seqs=self.tokenized.stop_token_seqs,
            message_meta=self.tokenized.message_meta,
            is_warmup=not is_final or self.request.is_warmup,
            internal_uid=self.request.internal_uid,
            prefix_keep_mask=execution_mask.to(torch.int32),
            full_input_ids=(
                self.tokenized.reposition_input_ids[:end].contiguous() if use_context_mask else None
            ),
            full_token_visible_until=(
                self.tokenized.full_token_visible_until[:end].contiguous()
                if use_context_mask and self.tokenized.full_token_visible_until is not None
                else None
            ),
            full_keep_mask=(execution_mask.to(torch.int32) if use_context_mask else None),
            use_context_mask=use_context_mask,
            context_compact_stream=use_context_mask,
            context_post_prefill_keep_mask=post_prefill_keep,
            request_received_ns=self.request.request_received_ns,
            tokenize_invocations=self.tokenized.tokenize_invocations,
            context_stage_count=len(self.layout.transition_offsets),
            radix_compile_ns=self.layout.compile_ns,
            radix_match_ns=self.radix_match_ns,
            retry_plan_ns=self.retry_plan_ns,
            reposition_transition_count=self.transition_count,
            reposition_h2d_bytes=self.h2d_bytes,
            reposition_d2h_bytes=self.d2h_bytes,
        )

    def _active_after_events(self, boundary: int) -> torch.Tensor:
        assert self.active_raw is not None
        active = torch.cat(
            (
                self.active_raw,
                torch.arange(self.raw_cursor, boundary, dtype=torch.int32, device="cpu"),
            )
        )
        active = torch.unique(active, sorted=True)
        cursor = self.drop_cursor
        while cursor < len(self.drop_event_positions):
            insertion = int(self.drop_event_positions[cursor])
            if insertion > boundary:
                break
            begin = int(self.drop_range_offsets[cursor])
            finish = int(self.drop_range_offsets[cursor + 1])
            keep = torch.ones(len(active), dtype=torch.bool, device="cpu")
            for range_index in range(begin, finish):
                start = int(self.drop_position_ranges[2 * range_index])
                end = int(self.drop_position_ranges[2 * range_index + 1])
                keep &= ~((active >= start) & (active < end))
            active = active[keep]
            cursor += 1
        mask = torch.zeros(boundary, dtype=torch.int32, device="cpu")
        mask[active.to(torch.int64)] = 1
        return mask

    def accept_ack(self, ack: WarmupAckMsg) -> None:
        if ack.uid != self.request.uid or self.in_flight_end <= self.raw_cursor:
            raise RuntimeError("Warmup acknowledgement does not match the Reposition step.")
        if not ack.finished:
            raise RuntimeError("An internal Reposition step unexpectedly entered Decode.")
        self.radix_match_ns += ack.radix_match_ns
        self.retry_plan_ns += ack.retry_plan_ns
        self.transition_count += ack.reposition_transition_count
        self.h2d_bytes += ack.reposition_h2d_bytes
        self.d2h_bytes += ack.reposition_d2h_bytes

        assert self.layout is not None
        assert self.current_positions is not None
        assert self.current_repos is not None
        assert self.current_records is not None
        self.active_raw = (
            torch.nonzero(self._active_after_events(self.in_flight_end), as_tuple=False)
            .view(-1)
            .to(torch.int32)
        )
        while (
            self.drop_cursor < len(self.drop_event_positions)
            and int(self.drop_event_positions[self.drop_cursor]) <= self.in_flight_end
        ):
            self.drop_cursor += 1

        next_reposition = self._next_effective_reposition()
        if next_reposition is not None and next_reposition[0] == self.in_flight_end:
            _, stage = next_reposition
            begin = int(self.layout.transition_offsets[stage - 1])
            end = int(self.layout.transition_offsets[stage])
            raw_tokens = self.layout.transition_raw_tokens[begin:end].to(torch.int64)
            new_positions = self.layout.transition_new_positions[begin:end]
            self.current_positions[raw_tokens] = new_positions
            assert self.tokenized.reposition_raw_boundaries is not None
            event = int(
                torch.nonzero(self.layout.effective_reposition_stages == stage, as_tuple=False)[0]
            )
            boundary = int(self.tokenized.reposition_raw_boundaries[event])
            self.current_repos[raw_tokens] = boundary
            token_rows = self.layout.token_to_key[raw_tokens]
            self.current_records[token_rows, 2] = boundary
            self.current_records[token_rows, 3] = new_positions
            self.current_stage = stage
            self.current_reposition = boundary
        self.raw_cursor = self.in_flight_end
        self.in_flight_end = 0


__all__ = ["RepositionSequenceState"]
