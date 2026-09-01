from __future__ import annotations

import pytest
import torch

from minisgl.kernel.retry_reposition import retry_reposition_kv


def _rotate(vector: torch.Tensor, cache: torch.Tensor, position: int) -> torch.Tensor:
    half = vector.shape[-1] // 2
    cosine = cache[position, :half]
    sine = cache[position, half:]
    first = vector[..., :half]
    second = vector[..., half:]
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine),
        dim=-1,
    )


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 3e-5), (torch.bfloat16, 3e-2)],
)
def test_retry_reposition_rotates_k_and_copies_v(dtype: torch.dtype, atol: float) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA Triton")

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260902)
    layers, slots, heads, head_dim = 2, 8, 3, 64
    half = head_dim // 2
    inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(16, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    attention_factor = 1.17
    rope_cache = torch.cat(
        (frequencies.cos() * attention_factor, frequencies.sin() * attention_factor),
        dim=-1,
    )
    assert rope_cache.shape[1] == 2 * half

    k_buffer = torch.randn(
        (layers, slots, heads, head_dim), device=device, dtype=dtype, generator=generator
    )
    v_buffer = torch.randn(
        (layers, slots, heads, head_dim), device=device, dtype=dtype, generator=generator
    )
    source_slots = torch.tensor([1, 4], dtype=torch.int32, device=device)
    destination_slots = torch.tensor([6, 7], dtype=torch.int32, device=device)
    position_pairs = torch.tensor([[2, 7], [9, 3]], dtype=torch.int32, device=device)
    unrotated = torch.randn(
        (2, layers, heads, head_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    source_values = torch.randn(
        (2, layers, heads, head_dim), device=device, dtype=dtype, generator=generator
    )
    expected_k: list[torch.Tensor] = []
    for token in range(2):
        old_position, new_position = map(int, position_pairs[token])
        old_k = _rotate(unrotated[token], rope_cache, old_position).to(dtype)
        k_buffer[:, int(source_slots[token])] = old_k
        v_buffer[:, int(source_slots[token])] = source_values[token]
        expected_k.append(_rotate(unrotated[token], rope_cache, new_position).to(dtype))

    retry_reposition_kv(
        k_buffer,
        v_buffer,
        source_slots,
        destination_slots,
        position_pairs,
        rope_cache,
    )
    torch.cuda.synchronize()

    for token in range(2):
        destination = int(destination_slots[token])
        torch.testing.assert_close(
            k_buffer[:, destination], expected_k[token], atol=atol, rtol=atol
        )
        torch.testing.assert_close(v_buffer[:, destination], source_values[token], atol=0, rtol=0)
