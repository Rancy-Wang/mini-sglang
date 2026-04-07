from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    true_positions: torch.Tensor  # CPU 1D int32 tensor (absolute positions)
    radix_input_ids: torch.Tensor  # CPU 1D int64 tensor for radix match
    sampling_params: SamplingParams
    radix_match_ids: torch.Tensor | None = None  # CPU 1D int64 tensor for radix query
    target_msg_id: int | None = None
    drop_message: Dict[int, List[int]] | None = None
    enable_thinking: bool | None = None
    message_meta: Dict | None = None
    is_warmup: bool = False
    internal_uid: int | None = None
    prefix_keep_mask: torch.Tensor | None = None  # CPU 1D bool tensor for prefix filtering
