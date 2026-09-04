from __future__ import annotations

import torch


def retry_reposition_kv(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    position_pairs: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    """Copy V and rotate cached K from old RoPE positions to new positions."""

    import triton

    from .triton.retry_reposition import retry_reposition_kv_kernel

    tensors = (
        k_buffer,
        v_buffer,
        source_slots,
        destination_slots,
        position_pairs,
        cos_sin_cache,
    )
    if any(tensor.device != k_buffer.device for tensor in tensors):
        raise ValueError("Retry Reposition tensors must be on one CUDA device.")
    if not k_buffer.is_cuda or not cos_sin_cache.is_cuda:
        raise ValueError("Retry Reposition requires CUDA KV and RoPE tensors.")
    if k_buffer.ndim != 4 or v_buffer.shape != k_buffer.shape:
        raise ValueError("Retry Reposition expects [layers, slots, heads, head_dim] K/V buffers.")
    if k_buffer.stride(-1) != 1 or v_buffer.stride(-1) != 1:
        raise ValueError("Retry Reposition requires contiguous K/V head dimensions.")
    if source_slots.ndim != 1 or destination_slots.ndim != 1:
        raise ValueError("Retry Reposition page slots must be one-dimensional.")
    if position_pairs.ndim != 2 or position_pairs.shape[1] != 2:
        raise ValueError("Retry Reposition positions must use an [N, 2] old/new matrix.")
    count = len(source_slots)
    if len(destination_slots) != count or len(position_pairs) != count:
        raise ValueError("Retry Reposition metadata lengths differ.")
    if count == 0:
        return
    head_dim = k_buffer.shape[-1]
    if head_dim % 2 != 0 or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != head_dim:
        raise ValueError(
            "Retry Reposition requires an even head dimension and matching RoPE cache."
        )
    if source_slots.dtype != torch.int32 or destination_slots.dtype != torch.int32:
        raise ValueError("Retry Reposition page slots must use int32.")
    if position_pairs.dtype != torch.int32:
        raise ValueError("Retry Reposition positions must use int32.")
    if position_pairs.stride(1) != 1:
        raise ValueError("Retry Reposition position pairs require a contiguous last dimension.")
    if cos_sin_cache.stride(1) != 1:
        raise ValueError("Retry Reposition requires a contiguous RoPE cache row.")

    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    grid = (count, k_buffer.shape[0], k_buffer.shape[2])
    retry_reposition_kv_kernel[grid](
        k_buffer,
        v_buffer,
        source_slots,
        destination_slots,
        position_pairs,
        cos_sin_cache,
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        position_pairs.stride(0),
        cos_sin_cache.stride(0),
        head_dim=head_dim,
        half_dim=half_dim,
        BLOCK_HALF=block_half,
        num_warps=4,
    )


__all__ = ["retry_reposition_kv"]
