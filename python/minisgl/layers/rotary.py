from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        attention_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key

    @property
    def cos_sin_cache(self) -> torch.Tensor:
        return self._cos_sin_cache


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            inv_freq, attention_factor = _get_yarn_parameters(
                rotary_dim, base, rope_scaling
            )
            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                lambda _: inv_freq,
                attention_factor,
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


def _get_yarn_parameters(
    rotary_dim: int,
    base: float,
    rope_scaling: Dict[str, Any],
) -> tuple[torch.Tensor, float]:
    """Match the YaRN frequencies and attention scaling used by Transformers."""

    factor = float(rope_scaling["factor"])
    beta_fast = float(rope_scaling.get("beta_fast", 32.0))
    beta_slow = float(rope_scaling.get("beta_slow", 1.0))
    orig_max_pos = int(rope_scaling["original_max_position_embeddings"])

    def correction_dim(num_rotations: float) -> float:
        return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    low = correction_dim(beta_fast)
    high = correction_dim(beta_slow)
    if rope_scaling.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rotary_dim - 1)
    if low == high:
        high += 0.001

    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    ramp = torch.clamp(
        (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / (high - low),
        0,
        1,
    )
    extrapolation = (1 - ramp) * float(rope_scaling.get("extrapolation_factor", 1.0))
    inv_freq = (inv_freq / factor) * (1 - extrapolation) + inv_freq * extrapolation

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is None:
        mscale = rope_scaling.get("mscale")
        mscale_all_dim = rope_scaling.get("mscale_all_dim")

        def get_mscale(multiplier: float = 1.0) -> float:
            return 1.0 if factor <= 1 else 0.1 * multiplier * math.log(factor) + 1.0

        if mscale is not None and mscale_all_dim is not None:
            attention_factor = get_mscale(float(mscale)) / get_mscale(
                float(mscale_all_dim)
            )
        else:
            yarn_scale = (
                get_mscale() if rope_scaling.get("apply_yarn_scaling", True) else 1.0
            )
            attention_factor = yarn_scale * float(rope_scaling.get("attn_factor", 1.0))
    return inv_freq, float(attention_factor)


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
