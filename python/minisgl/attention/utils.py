from dataclasses import dataclass
from typing import Sequence

import torch
from minisgl.utils import last_page_len, page_count


@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )


def _page_starts_for_len(num_tokens: int, page_size: int) -> slice:
    return slice(0, page_count(num_tokens, page_size) * page_size, page_size)


def make_backend_page_table(
    page_table: torch.Tensor,
    reqs: Sequence,
    *,
    max_seqlen_k: int,
    page_size: int,
) -> torch.Tensor:
    """Convert token-addressed scheduler rows to backend page-id rows."""

    max_pages = page_count(max_seqlen_k, page_size)
    rows = []
    for req in reqs:
        row = page_table[req.table_idx, : max_pages * page_size : page_size].clone()
        if page_size > 1:
            row.div_(page_size, rounding_mode="floor")
        rows.append(row)
    return torch.stack(rows)


def make_paged_kv_indices(
    page_table: torch.Tensor,
    reqs: Sequence,
    *,
    page_size: int,
) -> torch.Tensor:
    """Build FlashInfer ragged page-id indices from token-addressed rows."""

    rows = []
    for req in reqs:
        row = page_table[req.table_idx, _page_starts_for_len(req.device_len, page_size)].clone()
        if page_size > 1:
            row.div_(page_size, rounding_mode="floor")
        rows.append(row)
    if not rows:
        return torch.empty(0, dtype=page_table.dtype, device=page_table.device)
    return torch.cat(rows)


def make_page_indptr_cpu(
    seq_lens: Sequence[int],
    page_size: int,
    *,
    pin_memory: bool = True,
) -> torch.Tensor:
    kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": pin_memory}
    counts = [0] + [page_count(length, page_size) for length in seq_lens]
    return torch.tensor(counts, **kwargs).cumsum_(dim=0)


def make_last_page_len_cpu(
    seq_lens: Sequence[int],
    page_size: int,
    *,
    pin_memory: bool = True,
) -> torch.Tensor:
    kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": pin_memory}
    return torch.tensor([last_page_len(length, page_size) for length in seq_lens], **kwargs)
