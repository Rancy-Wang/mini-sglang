import triton
import triton.language as tl


@triton.jit
def retry_reposition_kv_kernel(
    k_buffer,
    v_buffer,
    source_slots,
    destination_slots,
    position_pairs,
    cos_sin_cache,
    k_stride_layer,
    k_stride_slot,
    k_stride_head,
    v_stride_layer,
    v_stride_slot,
    v_stride_head,
    position_stride_token,
    rope_stride_position,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    # Production KV buffers can exceed 2**31 elements even when every individual
    # stride fits in int32.  Promote the grid coordinates before multiplying by
    # those strides so high-layer addresses cannot wrap around.
    token = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1).to(tl.int64)
    head = tl.program_id(2).to(tl.int64)

    source = tl.load(source_slots + token).to(tl.int64)
    destination = tl.load(destination_slots + token).to(tl.int64)
    position_row = position_pairs + token * position_stride_token
    old_position = tl.load(position_row).to(tl.int64)
    new_position = tl.load(position_row + 1).to(tl.int64)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim

    source_k = k_buffer + layer * k_stride_layer + source * k_stride_slot + head * k_stride_head
    destination_k = (
        k_buffer + layer * k_stride_layer + destination * k_stride_slot + head * k_stride_head
    )
    first = tl.load(source_k + offsets, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(source_k + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    old_rope = cos_sin_cache + old_position * rope_stride_position
    new_rope = cos_sin_cache + new_position * rope_stride_position
    old_cos = tl.load(old_rope + offsets, mask=mask, other=0.0).to(tl.float32)
    old_sin = tl.load(old_rope + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    new_cos = tl.load(new_rope + offsets, mask=mask, other=0.0).to(tl.float32)
    new_sin = tl.load(new_rope + half_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    scale_squared = old_cos * old_cos + old_sin * old_sin
    delta_cos = (new_cos * old_cos + new_sin * old_sin) / scale_squared
    delta_sin = (new_sin * old_cos - new_cos * old_sin) / scale_squared

    tl.store(destination_k + offsets, first * delta_cos - second * delta_sin, mask=mask)
    tl.store(destination_k + half_dim + offsets, second * delta_cos + first * delta_sin, mask=mask)

    source_v = v_buffer + layer * v_stride_layer + source * v_stride_slot + head * v_stride_head
    destination_v = (
        v_buffer + layer * v_stride_layer + destination * v_stride_slot + head * v_stride_head
    )
    value_first = tl.load(source_v + offsets, mask=mask, other=0.0)
    value_second = tl.load(source_v + half_dim + offsets, mask=mask, other=0.0)
    tl.store(destination_v + offsets, value_first, mask=mask)
    tl.store(destination_v + half_dim + offsets, value_second, mask=mask)
