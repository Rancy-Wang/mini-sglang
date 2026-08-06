---
name: context-system-code-audit
description: Audit mini-sglang Context System behavior from current source code with end-to-end call tracing and exact file, line, and symbol evidence. Use for implementation questions, bug diagnosis, change planning, and verification of user claims about Drop Message, Radix/KV cache, message token boundaries, positions, or prefill.
---

# Context System Code Audit

Remain read-only unless an approved plan explicitly assigns implementation to another phase.

`SUBAGENT_POLICY_ONE_HELPER_MAX`: the main agent traces the full scope by default. If one helper is
needed, assign it one bounded evidence chain and reuse the same helper for any follow-up. Do not
split the evidence chains across multiple agents, and do not allow the helper to spawn agents.

## Trace the implementation

1. Record repository root, branch, HEAD, remotes, and dirty files.
2. Translate each claim into a code question. Treat the claim as a search hint, not a fact.
3. Find entry points, then follow concrete calls and data transformations through producers,
   consumers, state mutations, ownership transfer, cleanup, and tests.
4. Inspect all relevant branches, including no-drop behavior, cache hit/miss, prefill, decode,
   error paths, and configuration overrides.
5. Re-open the final working tree and calculate exact line references only after investigation.

For a full Context System audit, cover these evidence chains independently when scope requires:

- messages -> chat template -> tokenizer -> message boundary/provenance;
- drop parsing -> state encoding -> Radix lookup/insert -> table/cache page release;
- full/active token and KV state -> absolute positions -> integrity checks;
- match ratio -> threshold/comparator -> per-message prefill -> state commit.

## Report evidence

For every material conclusion, provide:

- `repository/relative/path:start-end`;
- class, function, method, or variable name;
- what the code proves;
- uncertainty or untested assumptions.

Separate verified facts, inferences, conflicts with the user's description, and missing coverage.
Do not propose a broad fix before locating the exact failure path and affected invariant.
