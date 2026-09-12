from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.kernel.radix_reposition import (
    DELTA_KIND,
    REPOSITION_KIND,
    TOKEN_KIND,
    RadixRepositionLayout,
)
from minisgl.message import (
    RepositionOpenMsg,
    RepositionStepMsg,
    StagedRepositionInit,
    TokenizeMsg,
    WarmupAckMsg,
)

from .tokenize import TokenizedResult


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
    transition_dispatch_pending: bool = False
    pending_transition_raw_tokens: torch.Tensor | None = None
    pending_transition_new_positions: torch.Tensor | None = None
    pending_transition_boundary: int | None = None
    open_ipc_counted: bool = False
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    transition_count: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    dispatch_count: int = 0
    ipc_tensor_bytes: int = 0

    @classmethod
    def pending(cls, request: TokenizeMsg, tokenized: TokenizedResult) -> RepositionSequenceState:
        if tokenized.reposition_input_ids is None:
            raise ValueError("Reposition sequence requires the immutable raw token stream.")
        if tokenized.reposition_raw_boundaries is None:
            raise ValueError("Reposition sequence requires raw Reposition boundaries.")
        if tokenized.reposition_insert_offsets is None:
            raise ValueError("Reposition sequence requires Reposition insertion offsets.")
        if tokenized.reposition_layout is None:
            raise ValueError("Reposition sequence requires a precompiled Radix layout.")
        layout = tokenized.reposition_layout
        if len(layout.transition_offsets) <= 1:
            raise ValueError("Reposition sequence requires at least one effective transition.")
        return cls(
            request=request,
            tokenized=tokenized,
            drop_event_positions=layout.drop_insert_offsets,
            drop_range_offsets=layout.drop_range_offsets,
            drop_position_ranges=layout.drop_ranges,
            layout=layout,
        )

    def open_msg(self) -> RepositionOpenMsg:
        self._initialize_current_state()
        assert self.layout is not None
        assert self.tokenized.reposition_input_ids is not None
        assert self.current_records is not None
        assert self.current_positions is not None
        assert self.current_repos is not None
        init = StagedRepositionInit(
            input_ids=self.tokenized.reposition_input_ids,
            radix_records=self.current_records,
            radix_key_virtual_mask=self.layout.virtual_mask,
            radix_key_to_token=self.layout.key_to_token,
            radix_token_to_key=self.layout.token_to_key,
            initial_positions=self.current_positions,
            initial_repos=self.current_repos,
            drop_event_positions=self.drop_event_positions,
            drop_range_offsets=self.drop_range_offsets,
            drop_position_ranges=self.drop_position_ranges,
            full_token_visible_until=self.tokenized.full_token_visible_until,
            sampling_params=self.request.sampling_params,
            prompt_tokens=self.tokenized.prompt_tokens,
            radix_next_position=self.layout.next_position,
            radix_final_reposition=self.layout.current_reposition,
            enable_thinking=self.request.enable_thinking,
            stop=self.request.stop,
            stop_token_seqs=self.tokenized.stop_token_seqs,
            message_meta=self.tokenized.message_meta,
            request_is_warmup=self.request.is_warmup,
            internal_uid=self.request.internal_uid,
            request_received_ns=self.request.request_received_ns,
            tokenize_invocations=self.tokenized.tokenize_invocations,
            chat_template_invocations=self.tokenized.chat_template_invocations,
            radix_compile_ns=self.layout.compile_ns,
        )
        if self.open_ipc_counted:
            raise RuntimeError("Reposition sequence initialization was already dispatched.")
        self.ipc_tensor_bytes += sum(
            value.numel() * value.element_size()
            for value in vars(init).values()
            if isinstance(value, torch.Tensor)
        )
        self.open_ipc_counted = True
        return RepositionOpenMsg(uid=self.request.uid, init=init)

    def _initialize_current_state(self) -> None:
        if self.current_records is not None:
            return
        assert self.tokenized.reposition_raw_boundaries is not None
        assert self.layout is not None
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

    def activate(self, *, step_token_budget: int) -> None:
        if step_token_budget <= 0:
            raise ValueError("Reposition step token budget must be positive.")
        assert self.tokenized.reposition_input_ids is not None
        assert self.tokenized.reposition_raw_boundaries is not None
        assert self.tokenized.reposition_insert_offsets is not None
        assert self.layout is not None
        self.step_token_budget = step_token_budget
        self.active_raw = torch.empty(0, dtype=torch.int32, device="cpu")
        self._initialize_current_state()

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
                and int(self.layout.effective_reposition_stages[event]) > self.current_stage
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
                int(self.layout.drop_event_to_key[event])
                for event, offset in enumerate(self.drop_event_positions.tolist())
                if event >= self.drop_cursor and int(offset) == insertion
            ]
            if not candidates:
                return None
            return candidates[0]
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
        return int(torch.searchsorted(self.drop_event_positions, boundary, side="right").item())

    def _drop_inside(self, end: int) -> bool:
        if self.drop_cursor >= len(self.drop_event_positions):
            return False
        positions = self.drop_event_positions[self.drop_cursor :]
        return bool(torch.any((positions > self.raw_cursor) & (positions <= end)).item())

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

    def build_next_msg(self) -> RepositionStepMsg:
        if self.layout is None or self.current_records is None:
            raise RuntimeError("Reposition sequence must be compiled before dispatch.")
        if self.in_flight_end != 0 or self.in_flight_final:
            raise RuntimeError("A Reposition step is already awaiting Scheduler acknowledgement.")
        assert self.active_raw is not None
        assert self.current_positions is not None
        assert self.current_repos is not None
        assert self.tokenized.reposition_input_ids is not None

        raw_count = len(self.tokenized.reposition_input_ids)
        next_reposition = self._next_effective_reposition()
        end = min(raw_count, self.raw_cursor + self.step_token_budget)
        if next_reposition is not None:
            end = min(end, next_reposition[0])
        transition_only = end == self.raw_cursor and self.transition_dispatch_pending
        if end < self.raw_cursor or (end == self.raw_cursor and not transition_only):
            raise RuntimeError("Reposition sequence did not make raw-token progress.")

        is_final = end == raw_count and next_reposition is None
        commit_key_len = len(self.layout.records) if is_final else self._commit_key_len(end)
        self.in_flight_end = end
        self.in_flight_final = is_final
        self.transition_dispatch_pending = False
        self.dispatch_count += 1
        message = RepositionStepMsg(
            uid=self.request.uid,
            end=end,
            is_final=is_final,
            radix_commit_key_len=None if is_final else commit_key_len,
            radix_current_reposition=(
                self.layout.current_reposition if is_final else self.current_reposition
            ),
            transition_raw_tokens=self.pending_transition_raw_tokens,
            transition_new_positions=self.pending_transition_new_positions,
            transition_boundary=self.pending_transition_boundary,
            context_stage_count=self.dispatch_count,
            radix_match_ns=self.radix_match_ns,
            retry_plan_ns=self.retry_plan_ns,
            reposition_transition_count=self.transition_count,
            reposition_h2d_bytes=self.h2d_bytes,
            reposition_d2h_bytes=self.d2h_bytes,
        )
        self.pending_transition_raw_tokens = None
        self.pending_transition_new_positions = None
        self.pending_transition_boundary = None
        self.ipc_tensor_bytes += sum(
            value.numel() * value.element_size()
            for value in vars(message).values()
            if isinstance(value, torch.Tensor)
        )
        message.reposition_ipc_tensor_bytes = self.ipc_tensor_bytes
        return message

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
        # Every Scheduler turn starts from the cumulative counters carried by
        # ``build_next_msg`` and returns a new cumulative snapshot.  Merging by
        # maximum preserves monotonicity without counting the prior stages a
        # second time.
        self.radix_match_ns = max(self.radix_match_ns, ack.radix_match_ns)
        self.retry_plan_ns = max(self.retry_plan_ns, ack.retry_plan_ns)
        self.transition_count = max(self.transition_count, ack.reposition_transition_count)
        self.h2d_bytes = max(self.h2d_bytes, ack.reposition_h2d_bytes)
        self.d2h_bytes = max(self.d2h_bytes, ack.reposition_d2h_bytes)

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
            self.transition_dispatch_pending = True
            self.pending_transition_raw_tokens = raw_tokens.to(torch.int32)
            self.pending_transition_new_positions = new_positions
            self.pending_transition_boundary = boundary
        self.raw_cursor = self.in_flight_end
        self.in_flight_end = 0
        self.in_flight_final = False


__all__ = ["RepositionSequenceState"]
