# Scheme 1: Request-Local Active KV Copy

## Goal

Support `page_size > 1` while preserving the current delta-marker Radix
semantics. Radix remains the owner of canonical full-token KV slots. When a Drop
request matches a sparse active prefix that cannot be represented as ordinary
paged KV, the scheduler copies the kept KV slots into request-local compact pages
before forward.

## Current Baseline

- Scheduler rejects non-unit pages for Drop-aware eviction and delta-marker mode
  in `python/minisgl/scheduler/scheduler.py`.
- Cache commit paths reject non-unit pages for drop-aware delta, drop-aware
  linear, ordinary delta, sparse Drop, and full-stream context-mask commits in
  `python/minisgl/scheduler/cache.py`.
- Radix rejects Drop-aware eviction and virtual marker keys with non-unit pages
  in `python/minisgl/kvcache/radix_cache.py`.
- The global `page_table` is token-addressed: each logical token offset maps to
  one physical KV slot, while the allocator/free list is page-aligned.

## Design

The scheduler keeps token-addressed `page_table` rows. Attention backends receive
page IDs derived from page-start token slots.

For:

```text
page_size = 4
full page0: [0, 1, 2, 3]
full page1: [4, 5, 6, 7]
full page2: [8, 9, 10, 11]
keep:       [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
```

The active cached slots are:

```text
[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
```

This cannot be represented as ordinary pages because `[4, 7]` has holes inside
page1. Scheme 1 allocates new request-local pages and copies kept KV in active
order:

```text
new page D: [KV0, KV1, KV2, KV3]
new page E: [KV4, KV7, KV8, KV9]
new page F: [KV10, KV11, padding, padding]
```

The compacted request uses normal paged KV. `true_positions` remain absolute:

```text
[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
```

## Implementation Plan

- Add page helper functions: page count and final-page effective length.
- Add attention metadata helpers for backend page table, FlashInfer ragged page
  indices, page indptr, and last-page length.
- Add `BaseKVCachePool.copy_slots(src, dst)` and implement it for MHA KV cache.
- Add `ContextMatchResult.requires_compaction` and `Req.compact_cached_prefix`.
- Detect whether active matched slots can be represented as ordinary pages.
- Compact request-local cached prefixes before normal page allocation.
- Commit only complete pages into Radix; release unadopted partial pages.

## Expected Behavior

- No-Drop baseline remains unchanged.
- Delta-marker key streams still distinguish different Drop histories.
- Dropped active KV never changes surviving token positions.
- Mixed-page sparse matches remain reusable, but copy cost can be high.
- Drop-aware eviction does not rewrite mixed pages in this scheme.

## Tests

- Page helper conversion from token-addressed table to page IDs.
- Mixed sparse match reports compaction required.
- Compaction copies kept KV slots in active order.
- Finished sparse/full commits only insert complete pages.
- Microbenchmark reports copied tokens, allocated pages, and elapsed copy time.

## Risks

- Long fragmented prefixes can copy a large amount of KV.
- Compaction can temporarily duplicate old Radix-owned KV and new request-local
  KV.
- This is semantically conservative but may be slower than page-pack compaction.
- FA3/FlashInfer segmented context-mask prefill still needs a page-aware segment
  compiler; Scheme 1 currently focuses on ordinary paged metadata and sparse
  prefix reuse.
