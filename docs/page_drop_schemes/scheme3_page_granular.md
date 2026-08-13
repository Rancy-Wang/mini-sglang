# Scheme 3: Page-Granular Drop KV

## Goal

Support `page_size > 1` with the simplest page ownership model: KV is dropped and
cached only at complete-page granularity. Mixed pages are not compacted or
partially reclaimed.

## Design

Radix matching still happens on the full token and delta-marker key axis, but the
usable active KV prefix is truncated to page-safe boundaries.

For:

```text
page_size = 4
full page0: [0, 1, 2, 3]
full page1: [4, 5, 6, 7]
full page2: [8, 9, 10, 11]
keep:       [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
```

Page1 is mixed. Scheme 3 reuses only:

```text
[0, 1, 2, 3]
```

The request recomputes from the next active token. If a full page is dropped and
safe to evict:

```text
[4, 5, 6, 7] all dropped -> [-1, -1, -1, -1]
```

the page start can return to the free list.

## Implementation Plan

- Keep free list and cache ownership strictly page-aligned.
- Remove non-unit page guards only where page-granular safety is enforced.
- During match, stop reuse at the first mixed page, real-token hole, or
  non-page-representable active view.
- During commit, insert only complete pages into Radix.
- Drop-aware eviction reclaims only all-dropped, unpinned full pages.

## Expected Behavior

- Minimal KV movement.
- Lowest ownership risk.
- Lower hit ratio when Drop boundaries cut through pages.
- No special backend support beyond ordinary paged KV metadata.

## Tests

- Mixed page truncates active cached prefix to previous safe boundary.
- Fully dropped page becomes hole and returns its page start to free list.
- Partial final page is not adopted by Radix.
- No-Drop `page_size=4` remains cacheable.
- Benchmark reports lower copy time but lower hit ratio than Scheme 1/2.

## Risks

- Can recompute large suffixes when Drop boundaries are not page-aligned.
- Poor cache reuse for short messages or frequent small drops.
- This is a conservative baseline, not maximum reuse.
