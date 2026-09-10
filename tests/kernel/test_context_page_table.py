from __future__ import annotations

import pytest
import torch

pytest.importorskip("tvm_ffi")

from minisgl.kernel.context_page_table import compile_context_page_table_aot


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Context page-table AOT tests.")
    return torch.device("cuda:0")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block_size", [128, 256, 512])
@pytest.mark.parametrize("direct", [False, True])
def test_context_page_table_aot_layouts(
    dtype: torch.dtype,
    block_size: int,
    direct: bool,
) -> None:
    device = _cuda_or_skip()
    key_positions = torch.tensor([0, 2, 1, 3, 4, 6], dtype=torch.int32, device=device)
    key_offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int32, device=device)
    table_indices = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
    page_table = torch.arange(2 * 8, dtype=dtype, device=device).view(2, 8) + 100
    direct_pages = torch.arange(8, dtype=dtype, device=device) + 700
    source = direct_pages if direct else page_table
    owners = None if direct else table_indices

    expected_rows = []
    offsets_cpu = key_offsets.cpu().tolist()
    keys_cpu = key_positions.cpu().tolist()
    for segment in range(3):
        local_keys = keys_cpu[offsets_cpu[segment] : offsets_cpu[segment + 1]]
        if direct:
            expected_rows.append([700 + key for key in local_keys])
        else:
            table = int(table_indices[segment])
            expected_rows.append([100 + table * 8 + key for key in local_keys])
    expected_flat = torch.tensor(
        [page for row in expected_rows for page in row],
        dtype=dtype,
        device=device,
    )
    expected_padded = torch.zeros((3, 3), dtype=dtype, device=device)
    for segment, row in enumerate(expected_rows):
        expected_padded[segment, : len(row)] = torch.tensor(row, dtype=dtype, device=device)

    for layout in ("flat", "padded", "both"):
        flat = torch.empty_like(expected_flat) if layout in {"flat", "both"} else None
        padded = torch.empty_like(expected_padded) if layout in {"padded", "both"} else None
        compile_context_page_table_aot(
            source,
            owners,
            key_positions,
            key_offsets,
            max_seqlen_k=3,
            flat_indices=flat,
            padded_page_table=padded,
            direct=direct,
            block_size=block_size,
        )
        torch.cuda.synchronize(device)
        if flat is not None:
            assert torch.equal(flat, expected_flat)
        if padded is not None:
            assert torch.equal(padded, expected_padded)
