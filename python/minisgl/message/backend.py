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
    prompt_tokens: int | None = None
    radix_match_ids: torch.Tensor | None = None  # CPU 1D int64 tensor for radix query
    radix_key_virtual_mask: torch.Tensor | None = None  # CPU 1D bool, key-axis virtual markers
    radix_key_to_token: torch.Tensor | None = None  # CPU 1D int64, -1 for virtual markers
    radix_token_to_key: torch.Tensor | None = None  # CPU 1D int64, full-token to key axis
    radix_commit_key_len: int | None = None  # internal upper bound for Radix match/commit
    radix_marker_ids: List[int] | None = None  # scheduler-owned request leases
    drop_event_positions: torch.Tensor | None = None  # CPU 1D int32 absolute boundaries
    drop_range_offsets: torch.Tensor | None = None  # CPU 1D int32 CSR offsets
    drop_position_ranges: torch.Tensor | None = None  # CPU 1D int32 flattened [start, end, ...]
    radix_commit_token_len: int | None = None  # full-token warmup commit boundary
    enable_thinking: bool | None = None
    stop: List[str] | None = None
    stop_token_seqs: List[List[int]] | None = None
    message_meta: Dict | None = None
    is_warmup: bool = False
    internal_uid: int | None = None
    prefix_keep_mask: torch.Tensor | None = None  # CPU 1D bool tensor for prefix filtering
    full_input_ids: torch.Tensor | None = None  # CPU 1D int32 full token stream
    full_token_visible_until: torch.Tensor | None = None  # CPU 1D int32 first hidden query pos
    full_keep_mask: torch.Tensor | None = None  # CPU 1D int32 final full-to-active mask
    use_context_mask: bool = False  # internal warmup: Prefill the full stream with a custom mask
    request_received_ns: int | None = None  # frontend monotonic clock, public requests only


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
