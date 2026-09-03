from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    true_positions: torch.Tensor
    raw_positions: torch.Tensor
    radix_input_ids: torch.Tensor
    radix_match_ids: torch.Tensor | None
    sampling_params: SamplingParams
    prompt_tokens: int = 0
    stop: List[str] | None = None
    stop_token_seqs: List[List[int]] | None = None
    initial_full_match_indices: torch.Tensor | None = None
    initial_active_cached_len: int = 0
    is_warmup: bool = False
    internal_uid: int | None = None
    prefix_keep_mask: torch.Tensor | None = None  # bool mask for prefix cache filtering
    full_input_ids: torch.Tensor | None = None
    full_token_visible_until: torch.Tensor | None = None
    full_keep_mask: torch.Tensor | None = None
    drop_event_positions: torch.Tensor | None = None
    drop_range_offsets: torch.Tensor | None = None
    drop_position_ranges: torch.Tensor | None = None
    drop_effective_event_count: int = 0
    use_context_mask: bool = False
    context_compact_stream: bool = False
    context_post_prefill_keep_mask: torch.Tensor | None = None
    radix_key_virtual_mask: torch.Tensor | None = None
    radix_key_to_token: torch.Tensor | None = None
    radix_token_to_key: torch.Tensor | None = None
    radix_commit_key_len: int | None = None
    radix_marker_ids: tuple[int, ...] = ()
    radix_positions: torch.Tensor | None = None
    radix_repos_info: torch.Tensor | None = None
    radix_next_position: int | None = None
    radix_current_reposition: int = -1
    chunked_req: ChunkedReq | None = None
    tokenize_invocations: int = 1
    radix_compile_ns: int = 0
    radix_match_ns: int = 0
    retry_plan_ns: int = 0
    reposition_transition_count: int = 0
    reposition_h2d_bytes: int = 0
    reposition_d2h_bytes: int = 0

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
