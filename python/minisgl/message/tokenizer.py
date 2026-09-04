from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Dict, List

from minisgl.core import SamplingParams

from .metrics import ServerMetrics
from .utils import deserialize_type, serialize_type


@cache
def get_gpt_oss_terminal_stop_token_ids() -> tuple[int, ...]:
    """Return terminal Harmony stops while preserving message continuation."""
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    except ImportError as exc:
        raise RuntimeError("GPT-OSS stop handling requires openai-harmony>=0.0.8.") from exc

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    message_end = "".join(("<", "|", "end", "|", ">"))
    return tuple(
        int(token_id)
        for token_id in encoding.stop_tokens()
        if encoding.decode([int(token_id)]) != message_end
    )


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
    finish_reason: str | None = None
    matched_stop: str | None = None
    cached_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    server_metrics: ServerMetrics | None = None


@dataclass
class WarmupAckMsg(BaseTokenizerMsg):
    uid: int
    hit_ratio: float
    cached_tokens: int
    drop_skipped_tokens: int
    finished: bool
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    reposition_transition_count: int = 0
    reposition_h2d_bytes: int = 0
    reposition_d2h_bytes: int = 0


@dataclass
class RepositionOpenAckMsg(BaseTokenizerMsg):
    """Scheduler acknowledgement carrying the per-turn Prefill quantum."""

    uid: int
    step_token_budget: int


@dataclass
class RequestRejectMsg(BaseTokenizerMsg):
    """Terminal request rejection sent from the scheduler to the frontend."""

    uid: int
    status_code: int
    error_code: str
    detail: str


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, Any]]
    sampling_params: SamplingParams
    target_msg_id: int | None = None
    drop_message: Dict[int, List[int]] | None = None
    drop_rule: Dict[str, Any] | None = None
    reposition: List[int] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: str | Dict[str, Any] | None = None
    stop: List[str] | None = None
    message_meta: Dict | None = None
    is_warmup: bool = False
    internal_uid: int | None = None
    use_context_mask: bool = False
    request_received_ns: int | None = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
