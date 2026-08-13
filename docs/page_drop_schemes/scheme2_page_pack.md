# Scheme 2: Page-Pack KV Compaction

## Goal

Support `page_size > 1` with lower copy overhead than Scheme 1 by packing kept KV
inside pages or across pages, then rewriting Radix values and page ownership so
attention backends still consume ordinary paged KV.

## Core Idea

In-page pack:

```text
before: [4, _, _, 7]
after:  [4, 7, _, _]
```

Cross-page pack:

```text
before page A: [4, _, _, _]
before page B: [_, _, 9, 10]
after  page A: [4, 9, 10, _]
```

The active order must be preserved. The backend must never see arbitrary page
holes; after packing, holes may appear only as final-page padding.

## Radix Interaction

Radix maps real full-token keys to token slots and virtual marker keys to `-1`.
After packing, Radix values must be rewritten:

```text
token4  -> pageA + 0
token7  -> pageA + 1
token9  -> pageA + 2
token10 -> pageA + 3
```

Drop-aware metadata must remain consistent:

- resident page owner index,
- KV pin counts,
- kept-leaf need counts,
- evictable/protected size accounting,
- marker tree references.

## Safety Rules

Page-pack is allowed only when affected pages and Radix nodes are safe:

- no pinned source or destination page,
- no running request depends on the old slot layout,
- no shared branch would observe an unexpected canonical value rewrite,
- all ownership metadata can be updated atomically after KV copy succeeds.

If any rule fails, Scheme 2 falls back to Scheme 1 request-local copy or to
recompute from a page-safe prefix.

## Implementation Plan

- Add a page-pack planner that scans kept full-token positions in active order.
- Prefer in-page pack when a page is safe and uniquely owned.
- Use cross-page pack only when it frees a whole page or materially reduces copy.
- Rewrite Radix values after KV copy succeeds.
- Update page owner and pin accounting together with the rewrite.
- Extend integrity checks for stale owner, duplicate owner, and free/resident
  overlap.

## Expected Behavior

- Mixed pages can become backend-compatible without full active-prefix copy.
- Cross-page pack can release pages after holes are eliminated.
- Shared or pinned layouts fall back without corrupting canonical Radix state.
- Absolute positions are unchanged; only physical KV locations change.

## Tests

- In-page pack `[4, _, _, 7] -> [4, 7, _, _]`.
- Cross-page pack `[4, _, _, _] + [_, _, 9, 10] -> [4, 9, 10, _]`.
- Pinned source page falls back.
- Shared branch falls back or copy-on-write preserves the other branch.
- Integrity catches stale/duplicate owner and free/resident overlap.
- Benchmark compares copied tokens, pages freed, and latency against Scheme 1.

## Risks

- Highest correctness risk of the three schemes.
- Radix value rewrite bugs can corrupt cache for other branches.
- Atomicity around KV copy plus metadata rewrite is critical.
- Benefit depends on fragmentation pattern.
