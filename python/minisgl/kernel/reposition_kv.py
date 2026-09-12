from __future__ import annotations

import torch


def reposition_kv_with_rope_delta(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    position_pairs: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    """Copy V and rotate cached K from old RoPE positions to new positions."""

    import triton

    from .triton.reposition_kv import reposition_kv_with_rope_delta_kernel

    tensors = (
        k_buffer,
        v_buffer,
        source_slots,
        destination_slots,
        position_pairs,
        cos_sin_cache,
    )
    if any(tensor.device != k_buffer.device for tensor in tensors):
        raise ValueError("KV reposition tensors must be on one CUDA device.")
    if not k_buffer.is_cuda or not cos_sin_cache.is_cuda:
        raise ValueError("KV reposition requires CUDA KV and RoPE tensors.")
    if k_buffer.ndim != 4 or v_buffer.shape != k_buffer.shape:
        raise ValueError("KV reposition expects [layers, slots, heads, head_dim] K/V buffers.")
    if k_buffer.stride(-1) != 1 or v_buffer.stride(-1) != 1:
        raise ValueError("KV reposition requires contiguous K/V head dimensions.")
    if source_slots.ndim != 1 or destination_slots.ndim != 1:
        raise ValueError("KV reposition page slots must be one-dimensional.")
    if position_pairs.ndim != 2 or position_pairs.shape[1] != 2:
        raise ValueError("KV reposition positions must use an [N, 2] old/new matrix.")
    count = len(source_slots)
    if len(destination_slots) != count or len(position_pairs) != count:
        raise ValueError("KV reposition metadata lengths differ.")
    if count == 0:
        return
    head_dim = k_buffer.shape[-1]
    if head_dim % 2 != 0 or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != head_dim:
        raise ValueError("KV reposition requires an even head dimension and matching RoPE cache.")
    if source_slots.dtype != torch.int32 or destination_slots.dtype != torch.int32:
        raise ValueError("KV reposition page slots must use int32.")
    if position_pairs.dtype != torch.int32:
        raise ValueError("KV reposition positions must use int32.")
    if position_pairs.stride(1) != 1:
        raise ValueError("KV reposition position pairs require a contiguous last dimension.")
    if cos_sin_cache.stride(1) != 1:
        raise ValueError("KV reposition requires a contiguous RoPE cache row.")

    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    grid = (count, k_buffer.shape[0], k_buffer.shape[2])
    reposition_kv_with_rope_delta_kernel[grid](
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


def prewarm_reposition_kv_with_rope_delta(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    """Compile and launch the production KV transform before serving requests."""

    if len(cos_sin_cache) < 2:
        raise ValueError("KV reposition warmup requires at least two RoPE positions.")
    k_buffer = torch.empty((1, 2, num_heads, head_dim), device=device, dtype=dtype)
    v_buffer = torch.empty_like(k_buffer)
    source_slots = torch.tensor([0], dtype=torch.int32, device=device)
    destination_slots = torch.tensor([1], dtype=torch.int32, device=device)
    position_pairs = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    reposition_kv_with_rope_delta(
        k_buffer,
        v_buffer,
        source_slots,
        destination_slots,
        position_pairs,
        cos_sin_cache,
    )


__all__ = [
    "prewarm_reposition_kv_with_rope_delta",
    "reposition_kv_with_rope_delta",
]
