from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .metrics import ServerMetrics
from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
    finish_reason: str | None = None
    matched_stop: str | None = None
    cache_hit_ratio: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    server_metrics: ServerMetrics | None = None


@dataclass
class WarmupReply(BaseFrontendMsg):
    uid: int
    hit_ratio: float
    finished: bool


@dataclass
class RequestErrorReply(BaseFrontendMsg):
    """Terminal request error propagated back to the HTTP frontend."""

    uid: int
    status_code: int
    error_code: str
    detail: str
    finished: bool = True
