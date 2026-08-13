# Scheme 3: Page-Granular Drop KV

## Goal

Support `page_size > 1` with the simplest page ownership model: Drop itself is
rounded to complete-page granularity. A partial-page Drop is ignored for that
page; only a fully dropped page is removed from the active stream.

## Design

Radix matching still happens on the full token and delta-marker key axis, but the
active token stream is page-normalized before scheduling.

For:

```text
page_size = 4
full page0: [0, 1, 2, 3]
full page1: [4, 5, 6, 7]
full page2: [8, 9, 10, 11]
keep:       [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
```

The requested Drop only cuts through page1. Scheme 3 does not create a mixed
page; it keeps page1 unchanged:

```text
[0, 1, 2, 3] [4, 5, 6, 7] [8, 9, 10, 11]
```

If a full page is dropped and safe to evict:

```text
[4, 5, 6, 7] all dropped -> [-1, -1, -1, -1]
```

the page start can return to the free list.

## Implementation Plan

- Keep free list and cache ownership strictly page-aligned.
- Remove non-unit page guards only where page-granular safety is enforced.
- Before scheduling, rewrite Drop masks so partial-page drops are ignored and
  only all-dropped pages remain dropped.
- During match, ordinary paged KV metadata remains page-representable because
  mixed pages are never produced by this scheme.
- During commit, insert only complete pages into Radix.
- Drop-aware eviction reclaims only all-dropped, unpinned full pages.

## Expected Behavior

- Minimal KV movement.
- Lowest ownership risk.
- Coarser Drop semantics when Drop boundaries cut through pages.
- No special backend support beyond ordinary paged KV metadata.

## Tests

- Partial-page Drop is ignored.
- Fully dropped page is removed from the active stream.
- Fully dropped page becomes hole and returns its page start to free list.
- Partial final page is not adopted by Radix.
- No-Drop `page_size=4` remains cacheable.
- Benchmark reports lower copy time but lower hit ratio than Scheme 1/2.

## Risks

- Drop semantics are approximate: a token/message inside a partially kept page
  remains visible.
- Poor deletion precision for short messages or frequent small drops.
- This is a conservative baseline, not maximum reuse.
