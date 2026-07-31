from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
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
