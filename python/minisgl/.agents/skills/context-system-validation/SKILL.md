---
name: context-system-validation
description: Independently validate mini-sglang Context System code, tests, documentation, citations, invariants, and approved scope. Use after any approved change and before declaring a checkpoint passed or publishing the final result.
---

# Context System Validation

Do not edit source files while acting as verifier. Report failures to the main agent or approved
implementer.

## Validate scope and evidence

1. Compare the diff with `APPROVED_PLAN_ID`, `APPROVED_ROUND`, and `ALLOWED_FILES`.
2. Confirm pre-existing local and remote changes were preserved.
3. Reject unapproved generated files, dependency changes, formatting churn, or semantic expansion.
4. Recheck documentation paths, line numbers, symbols, and cross-file call relationships.
5. Review the Context System invariants in `python/minisgl/AGENTS.md`.

## Validate behavior

Choose the smallest sufficient set and record exact commands:

- syntax, imports, lint, and focused CPU unit tests;
- no-drop baseline, one drop, staged drops, and repeated text with different drop history;
- high and low Radix match paths;
- absolute positions after active-KV compaction;
- table/index/page ownership and integrity;
- complete-template message token provenance;
- GPU kernels and end-to-end serving on `InfiniAI-BUS` when affected.

Classify every non-pass as one of:

- `NEW_REGRESSION`
- `UNSUPPORTED_CLAIM`
- `CITATION_DRIFT`
- `PRE_EXISTING_FAILURE`
- `ENVIRONMENT_BLOCKER`
- `NOT_COVERED_BY_APPROVED_SCOPE`

Return `PASS` only when every applicable gate passes. Include command, result, evidence, coverage
gap, and whether another approved implementation round is required.
