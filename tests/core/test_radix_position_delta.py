from __future__ import annotations

import torch

import minisgl.core as core
from minisgl.kvcache.radix_cache import RadixPrefixCache
from minisgl.scheduler.radix_delta import (
    DeltaMarkerRegistry,
    canonicalize_delta,
    canonicalize_delta_ranges,
    inject_delta_markers,
)


def _wire(events, offsets, ranges):
    return (
        torch.tensor(events, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(ranges, dtype=torch.int32).reshape(-1),
    )


def test_position_ranges_are_exact_canonical_half_open_blocks():
    assert canonicalize_delta([5, 3, 4, 3, 9]) == ((3, 6), (9, 10))
    assert canonicalize_delta_ranges([(9, 10), (3, 5), (5, 7), (4, 6)]) == (
        (3, 7),
        (9, 10),
    )


def test_multiple_position_ranges_use_one_virtual_marker_per_event():
    registry = DeltaMarkerRegistry()
    full_ids = torch.arange(10, 20, dtype=torch.int64)
    layout = inject_delta_markers(
        full_ids,
        *_wire([8], [0, 2], [[2, 4], [6, 8]]),
        registry,
    )

    assert layout is not None
    assert layout.virtual_mask.nonzero().view(-1).tolist() == [8]
    assert layout.key_to_token.tolist().count(-1) == 1
    assert len(layout.marker_ids) == 1
    marker = layout.marker_ids[0]
    assert registry.canonical_for(marker) == ((2, 4), (6, 8))
    assert registry.request_ref_count == 1

    registry.release_request_refs(layout.marker_ids)
    assert registry.size == 0


def test_different_position_deltas_branch_at_the_marker_scalar():
    registry = DeltaMarkerRegistry()
    full_ids = torch.arange(100, 110, dtype=torch.int64)
    first = inject_delta_markers(
        full_ids,
        *_wire([8], [0, 2], [[2, 4], [6, 8]]),
        registry,
    )
    second = inject_delta_markers(
        full_ids,
        *_wire([8], [0, 2], [[2, 4], [7, 8]]),
        registry,
    )

    assert first is not None and second is not None
    assert first.keys[:8].tolist() == second.keys[:8].tolist()
    assert first.keys[8].item() != second.keys[8].item()
    registry.release_request_refs(first.marker_ids)
    registry.release_request_refs(second.marker_ids)
    assert registry.size == 0


def test_registry_entries_follow_request_and_radix_tree_lifetimes():
    previous_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = core.Context(page_size=1)
    try:
        registry = DeltaMarkerRegistry()
        full_ids = torch.arange(10, 18, dtype=torch.int64)
        layout = inject_delta_markers(
            full_ids,
            *_wire([6], [0, 1], [[2, 4]]),
            registry,
        )
        assert layout is not None

        cache = RadixPrefixCache(torch.device("cpu"))
        cache.bind_delta_marker_registry(registry)
        values = torch.arange(len(layout.keys), dtype=torch.int32)
        values[layout.virtual_mask] = -1
        cache.insert_prefix(layout.keys, values, layout.virtual_mask)

        assert registry.request_ref_count == 1
        assert registry.tree_ref_count == 1
        registry.release_request_refs(layout.marker_ids)
        assert registry.size == 1
        cache.check_integrity()

        evicted = cache.evict(cache.size_info.evictable_size)
        assert len(evicted) == len(full_ids)
        assert registry.request_ref_count == 0
        assert registry.tree_ref_count == 0
        assert registry.size == 0
        cache.check_integrity()
    finally:
        core._GLOBAL_CTX = previous_ctx
