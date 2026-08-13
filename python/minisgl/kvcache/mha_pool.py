from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    def copy_slots(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        if len(src) != len(dst):
            raise ValueError("src and dst must have the same length.")
        if len(src) == 0:
            return
        src = src.to(device=self._device, dtype=torch.int64, non_blocking=True)
        dst = dst.to(device=self._device, dtype=torch.int64, non_blocking=True)
        k_flat = self._k_buffer.view((self._num_layers,) + self._storage_shape)
        v_flat = self._v_buffer.view((self._num_layers,) + self._storage_shape)
        for layer_id in range(self._num_layers):
            k_values = k_flat[layer_id].index_select(0, src)
            v_values = v_flat[layer_id].index_select(0, src)
            k_flat[layer_id].index_copy_(0, dst, k_values)
            v_flat[layer_id].index_copy_(0, dst, v_values)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
