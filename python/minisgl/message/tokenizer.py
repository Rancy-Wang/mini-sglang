from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


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


@dataclass
class WarmupAckMsg(BaseTokenizerMsg):
    uid: int
    hit_ratio: float
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, Any]]
    sampling_params: SamplingParams
    target_msg_id: int | None = None
    drop_message: Dict[int, List[int]] | None = None
    enable_thinking: bool | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: str | Dict[str, Any] | None = None
    stop: List[str] | None = None
    message_meta: Dict | None = None
    is_warmup: bool = False
    internal_uid: int | None = None


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
