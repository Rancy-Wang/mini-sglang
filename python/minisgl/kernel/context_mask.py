from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


DEFAULT_CONTEXT_MASK_KERNEL_CONFIG = KernelConfig(
    num_threads=256,
    max_occupancy=1,
    use_pdl=False,
)


@lru_cache(maxsize=None)
def _jit_context_mask_module(
    config: KernelConfig = DEFAULT_CONTEXT_MASK_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(*config)
    return load_jit(
        "context_mask",
        *args,
        cuda_files=["context_mask.cu"],
        cuda_wrappers=[
            ("launch", f"ContextMaskKernel<{args}>::run"),
            ("launch_unpacked", f"ContextMaskUnpackedKernel<{args}>::run"),
        ],
    )


def build_context_visibility_mask(
    full_kv_owner: torch.Tensor,
    full_query_epoch: torch.Tensor,
    drop_visible_until: torch.Tensor,
    *,
    query_start: int,
    query_length: int,
    key_length: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build FlashInfer's flattened binary visibility mask on GPU."""

    if query_start < 0 or query_length <= 0 or key_length <= 0:
        raise ValueError("Context-mask query/key lengths must be positive and in range.")
    if query_start + query_length > len(full_query_epoch):
        raise ValueError("Context-mask query range exceeds full_query_epoch.")
    if key_length > len(full_kv_owner):
        raise ValueError("Context-mask key range exceeds full_kv_owner.")
    mask_length = query_length * key_length
    if output is None:
        # FlashInfer accepts a binary uint8 custom_mask and performs its own
        # segment-aware little-endian packing on GPU.
        output = full_kv_owner.new_empty(mask_length, dtype=torch.uint8)

    module = _jit_context_mask_module()
    module.launch_unpacked(
        full_kv_owner,
        full_query_epoch,
        drop_visible_until,
        output,
        query_start,
        query_length,
        key_length,
    )
    return output


def build_context_visibility_mask_packed(
    full_kv_owner: torch.Tensor,
    full_query_epoch: torch.Tensor,
    drop_visible_until: torch.Tensor,
    *,
    query_start: int,
    query_length: int,
    key_length: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build FlashInfer's little-endian packed causal/drop visibility mask on GPU."""

    if query_start < 0 or query_length <= 0 or key_length <= 0:
        raise ValueError("Context-mask query/key lengths must be positive and in range.")
    if query_start + query_length > len(full_query_epoch):
        raise ValueError("Context-mask query range exceeds full_query_epoch.")
    if key_length > len(full_kv_owner):
        raise ValueError("Context-mask key range exceeds full_kv_owner.")
    packed_length = (query_length * key_length + 7) // 8
    if output is None:
        output = full_kv_owner.new_empty(packed_length, dtype=torch.uint8)

    module = _jit_context_mask_module()
    module.launch(
        full_kv_owner,
        full_query_epoch,
        drop_visible_until,
        output,
        query_start,
        query_length,
        key_length,
    )
    return output
