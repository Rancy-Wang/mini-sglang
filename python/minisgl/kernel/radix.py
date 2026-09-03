from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)


def fast_compare_radix_key(
    x: torch.Tensor,
    y: torch.Tensor,
    x_virtual_mask: torch.Tensor,
    y_virtual_mask: torch.Tensor,
) -> int:
    """Return the first key or virtual-kind mismatch without Python temporaries."""

    return _load_radix_module().fast_compare_radix_key(x, y, x_virtual_mask, y_virtual_mask)


def fast_compare_radix_records(x: torch.Tensor, y: torch.Tensor) -> int:
    """Return the first exact structured-record mismatch."""

    return _load_radix_module().fast_compare_radix_records(x, y)


def fast_compare_retry_radix_records(cached: torch.Tensor, target: torch.Tensor) -> int:
    """Return the first Retry mismatch, ignoring position fields on real tokens."""

    return _load_radix_module().fast_compare_retry_radix_records(cached, target)


def fast_compare_retry_radix_records_plan(
    cached: torch.Tensor,
    target: torch.Tensor,
    cached_key_to_token: torch.Tensor,
    target_key_to_token: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Compare a Retry path and emit changed token mappings and positions."""

    capacity = min(len(cached), len(target))
    output = torch.empty((capacity, 4), dtype=torch.int32, device="cpu")
    status = torch.empty(2, dtype=torch.int64, device="cpu")
    _load_radix_module().fast_compare_retry_radix_records_plan(
        cached,
        target,
        cached_key_to_token,
        target_key_to_token,
        output,
        status,
    )
    return int(status[0]), output[: int(status[1])]


def radix_record_compare_backend() -> str:
    """Return the selected structured-record comparison implementation."""

    backend = int(_load_radix_module().radix_record_compare_backend())
    return ("portable", "neon", "avx2", "avx512")[backend]


def radix_record_edge_hash(records: torch.Tensor) -> int:
    return int(_load_radix_module().radix_record_edge_hash(records))


def radix_record_edge_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool(_load_radix_module().radix_record_edge_equal(x, y))


def radix_record_retry_token(records: torch.Tensor) -> int:
    return int(_load_radix_module().radix_record_retry_token(records))
