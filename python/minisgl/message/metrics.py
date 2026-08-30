from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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

    def as_api_dict(self) -> Dict[str, int]:
        return {
            "request_received_ns": self.request_received_ns,
            "first_token_generated_ns": self.first_token_generated_ns,
            "request_finished_ns": self.request_finished_ns,
            "prompt_tokens": self.prompt_tokens,
            "active_prompt_tokens": self.active_prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "completion_tokens": self.completion_tokens,
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

    def observe_token(self, generated_ns: int, *, visible: bool) -> None:
        if generated_ns < self.request_received_ns:
            raise ValueError("A token timestamp cannot precede request receipt.")
        if (
            self.last_token_generated_ns is not None
            and generated_ns < self.last_token_generated_ns
        ):
            raise ValueError("Generated token timestamps must be monotonic.")
        if self.first_token_generated_ns is None:
            self.first_token_generated_ns = generated_ns
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
        )
