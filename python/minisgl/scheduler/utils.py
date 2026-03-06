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
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    table_idx: int | None = None
    boundaries: List[(int, int, str, int)] | None = None
    message_id: int | None = None
    drop_ids: List[int] | None = None
    new_drop_ids: List[int] | None = None
    true_seq_len: int | None = None
    is_table_reuse: bool = False  # Set by frontend for throwaway context

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
