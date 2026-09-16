from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ServerMetrics:
    """Terminal serving metrics measured with the server's monotonic clock."""

    request_received_ns: int
    first_token_generated_ns: int
    request_finished_ns: int
    prompt_tokens: int
    active_prompt_tokens: int
    generated_tokens: int
    completion_tokens: int
    tokenize_invocations: int = 1
    chat_template_invocations: int = 0
    context_stage_count: int = 0
    radix_compile_ns: int = 0
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    reposition_transition_count: int = 0
    reposition_h2d_bytes: int = 0
    reposition_d2h_bytes: int = 0
    reposition_ipc_tensor_bytes: int = 0
    drop_skipped_tokens: int = 0
    prefill_compute_tokens: int | None = None
    decode_compute_tokens: int | None = None
    # Scheduler-observed intervals, including hidden/special sampled tokens.
    # None means recording was disabled; an empty tuple means one token.
    token_intervals_ns: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        timestamps = (
            self.request_received_ns,
            self.first_token_generated_ns,
            self.request_finished_ns,
        )
        if not 0 <= timestamps[0] <= timestamps[1] <= timestamps[2]:
            raise ValueError("Server metric timestamps must be non-negative and monotonic.")
        if not 0 <= self.active_prompt_tokens <= self.prompt_tokens:
            raise ValueError("active_prompt_tokens must be between zero and prompt_tokens.")
        if not 0 <= self.completion_tokens <= self.generated_tokens:
            raise ValueError("completion_tokens must be between zero and generated_tokens.")
        if self.generated_tokens == 0:
            raise ValueError("A terminal generation must contain at least one sampled token.")
        if self.token_intervals_ns is not None:
            if len(self.token_intervals_ns) != self.generated_tokens - 1:
                raise ValueError("Token interval count must equal generated_tokens - 1.")
            if any(x < 0 for x in self.token_intervals_ns):
                raise ValueError("Token intervals must be non-negative.")
            if sum(self.token_intervals_ns) > self.request_finished_ns - self.first_token_generated_ns:
                raise ValueError("Token intervals exceed request duration.")
        if any(value is not None and value < 0 for value in (
            self.prefill_compute_tokens, self.decode_compute_tokens
        )):
            raise ValueError("Forward token counters must be non-negative.")
        counters = (
            self.tokenize_invocations,
            self.chat_template_invocations,
            self.context_stage_count,
            self.radix_compile_ns,
            self.radix_match_ns,
            self.retry_plan_ns,
            self.reposition_transition_count,
            self.reposition_h2d_bytes,
            self.reposition_d2h_bytes,
            self.reposition_ipc_tensor_bytes,
            self.drop_skipped_tokens,
        )
        if self.tokenize_invocations < 1 or any(value < 0 for value in counters[1:]):
            raise ValueError("Serving performance counters must be non-negative.")

    def as_api_dict(self) -> Dict[str, Any]:
        return {
            "request_received_ns": self.request_received_ns,
            "first_token_generated_ns": self.first_token_generated_ns,
            "request_finished_ns": self.request_finished_ns,
            "prompt_tokens": self.prompt_tokens,
            "active_prompt_tokens": self.active_prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "completion_tokens": self.completion_tokens,
            "tokenize_invocations": self.tokenize_invocations,
            "chat_template_invocations": self.chat_template_invocations,
            "context_stage_count": self.context_stage_count,
            "radix_compile_ns": self.radix_compile_ns,
            "radix_match_ns": self.radix_match_ns,
            "retry_plan_ns": self.retry_plan_ns,
            "reposition_transition_count": self.reposition_transition_count,
            "reposition_h2d_bytes": self.reposition_h2d_bytes,
            "reposition_d2h_bytes": self.reposition_d2h_bytes,
            "reposition_ipc_tensor_bytes": self.reposition_ipc_tensor_bytes,
            "drop_skipped_tokens": self.drop_skipped_tokens,
            "prefill_compute_tokens": self.prefill_compute_tokens,
            "decode_compute_tokens": self.decode_compute_tokens,
            "token_intervals_ns": (list(self.token_intervals_ns)
                                   if self.token_intervals_ns is not None else None),
        }


@dataclass
class RequestMetricsState:
    """Mutable scheduler-owned state used to build one terminal ServerMetrics."""

    request_received_ns: int
    prompt_tokens: int
    active_prompt_tokens: int
    first_token_generated_ns: int | None = None
    last_token_generated_ns: int | None = None
    generated_tokens: int = 0
    completion_tokens: int = 0
    tokenize_invocations: int = 1
    chat_template_invocations: int = 0
    context_stage_count: int = 0
    radix_compile_ns: int = 0
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    reposition_transition_count: int = 0
    reposition_h2d_bytes: int = 0
    reposition_d2h_bytes: int = 0
    reposition_ipc_tensor_bytes: int = 0
    drop_skipped_tokens: int = 0
    prefill_compute_tokens: int = 0
    decode_compute_tokens: int = 0
    token_intervals_ns: list[int] | None = field(default_factory=lambda: (
        [] if os.environ.get("MINISGL_RECORD_TOKEN_TIMINGS") == "1" else None
    ))

    def observe_reposition(
        self,
        *,
        radix_match_ns: int,
        retry_plan_ns: int,
        transition_count: int,
        h2d_bytes: int,
        d2h_bytes: int = 0,
    ) -> None:
        self.radix_match_ns = max(self.radix_match_ns, radix_match_ns)
        self.retry_plan_ns = max(self.retry_plan_ns, retry_plan_ns)
        self.reposition_transition_count = max(self.reposition_transition_count, transition_count)
        self.reposition_h2d_bytes = max(self.reposition_h2d_bytes, h2d_bytes)
        self.reposition_d2h_bytes = max(self.reposition_d2h_bytes, d2h_bytes)

    def observe_token(self, generated_ns: int, *, visible: bool) -> None:
        if generated_ns < self.request_received_ns:
            raise ValueError("A token timestamp cannot precede request receipt.")
        if self.last_token_generated_ns is not None and generated_ns < self.last_token_generated_ns:
            raise ValueError("Generated token timestamps must be monotonic.")
        if self.first_token_generated_ns is None:
            self.first_token_generated_ns = generated_ns
        if self.token_intervals_ns is not None and self.last_token_generated_ns is not None:
            self.token_intervals_ns.append(generated_ns - self.last_token_generated_ns)
        self.last_token_generated_ns = generated_ns
        self.generated_tokens += 1
        if visible:
            self.completion_tokens += 1

    def finish(self, finished_ns: int) -> ServerMetrics:
        if self.first_token_generated_ns is None:
            raise ValueError("Cannot finish metrics before observing a generated token.")
        return ServerMetrics(
            request_received_ns=self.request_received_ns,
            first_token_generated_ns=self.first_token_generated_ns,
            request_finished_ns=finished_ns,
            prompt_tokens=self.prompt_tokens,
            active_prompt_tokens=self.active_prompt_tokens,
            generated_tokens=self.generated_tokens,
            completion_tokens=self.completion_tokens,
            tokenize_invocations=self.tokenize_invocations,
            chat_template_invocations=self.chat_template_invocations,
            context_stage_count=self.context_stage_count,
            radix_compile_ns=self.radix_compile_ns,
            radix_match_ns=self.radix_match_ns,
            retry_plan_ns=self.retry_plan_ns,
            reposition_transition_count=self.reposition_transition_count,
            reposition_h2d_bytes=self.reposition_h2d_bytes,
            reposition_d2h_bytes=self.reposition_d2h_bytes,
            reposition_ipc_tensor_bytes=self.reposition_ipc_tensor_bytes,
            drop_skipped_tokens=self.drop_skipped_tokens,
            prefill_compute_tokens=self.prefill_compute_tokens,
            decode_compute_tokens=self.decode_compute_tokens,
            token_intervals_ns=(tuple(self.token_intervals_ns)
                                if self.token_intervals_ns is not None else None),
        )
