from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    KVCacheLayout,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device, table_manager=None) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache(
    model_config: ModelConfig,
    num_pages: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.LayerFirst,
) -> BaseKVCache:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        kv_layout=cache_layout,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache_manager(device: torch.device, table_manager=None):
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device, table_manager=table_manager)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache_manager(device: torch.device, table_manager=None):
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_cache_manager(device: torch.device, type: str, table_manager=None) -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device, table_manager)


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "BaseKVCache",
    "KVCacheLayout",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
]
