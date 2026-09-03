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
class RepositionOpenMsg(BaseBackendMsg):
    """Open a staged Reposition sequence and obtain its Prefill quantum."""

    uid: int


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    true_positions: torch.Tensor  # CPU 1D int32 tensor (current KV positions)
    raw_positions: torch.Tensor  # CPU 1D int32 tensor (immutable full-token positions)
    radix_input_ids: torch.Tensor  # CPU 1D int64 tensor for radix match
    sampling_params: SamplingParams
    prompt_tokens: int | None = None
    radix_match_ids: torch.Tensor | None = None  # CPU 1D int64 tensor for radix query
    radix_key_virtual_mask: torch.Tensor | None = None  # CPU 1D bool, key-axis virtual records
    radix_key_to_token: torch.Tensor | None = None  # CPU 1D int64, -1 for virtual records
    radix_token_to_key: torch.Tensor | None = None  # CPU 1D int64, full-token to key axis
    radix_commit_key_len: int | None = None  # internal upper bound for Radix match/commit
    drop_event_positions: torch.Tensor | None = None  # CPU 1D int32 absolute boundaries
    drop_range_offsets: torch.Tensor | None = None  # CPU 1D int32 CSR offsets
    drop_position_ranges: torch.Tensor | None = None  # CPU 1D int32 flattened [start, end, ...]
    drop_effective_event_count: int = 0  # target-effective prefix of drop events
    radix_positions: torch.Tensor | None = None  # CPU 1D int32, full-token final KV position
    radix_repos_info: torch.Tensor | None = None  # CPU 1D int32, last effective R boundary
    radix_next_position: int | None = None
    radix_current_reposition: int = -1
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
    context_compact_stream: bool = False  # mask metadata accompanies an already compact stream
    context_post_prefill_keep_mask: torch.Tensor | None = None  # final raw keep-set
    request_received_ns: int | None = None  # frontend monotonic clock, public requests only
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
    prior_drop_skipped_tokens: int = 0


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
