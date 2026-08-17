---
name: context-system-documentation
description: Create or update Chinese technical documentation for the verified mini-sglang Context System implementation, including message token provenance, Drop/Radix encoding, full-active KV state, absolute positions, fallback prefill, integrity checks, examples, limitations, and exact source citations.
---

# Context System Documentation

Use `$context-system-code-audit` before writing implementation claims and
`$context-system-change-gate` before modifying documentation.

`SUBAGENT_POLICY_ONE_HELPER_MAX`: documentation does not justify a new agent. The main agent writes
and reviews it unless the task's one existing helper performs one of those phases. Reuse that
helper, do not create another reviewer, and do not allow the helper to spawn agents.

## Write from verified behavior

1. Define repository-specific terms at first use.
2. Explain the end-to-end data flow before isolated details.
3. Cover relevant edge cases: BOS/EOS, generation prompt, empty messages, special tokens,
   repeated text, staged drops, cache reuse, decode, and fallback paths.
4. State the real formula, threshold source, comparator, bit width, capacity, ownership, and
   cleanup timing where applicable.
5. Cite every key claim as `relative/path:start-end` plus the owning symbol.
6. Distinguish verified behavior, inference, known limitation, environment blocker, and suggested
   test. Never fill a documentation gap by guessing.
7. Recalculate all line references from the final working tree after editing.

## Review the document

Check that a reader unfamiliar with the patch can follow inputs, transitions, invariants, outputs,
and failure modes. Cross-check citations against current code and use
`$context-system-validation` before checkpointing.
