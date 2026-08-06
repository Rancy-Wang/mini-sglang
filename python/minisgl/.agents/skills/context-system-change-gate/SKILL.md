---
name: context-system-change-gate
description: Enforce read-only discovery, explicit Plan ID approval, bounded implementation, and round limits for any mini-sglang Context System, Throwaway Context, Drop Message, Radix/KV cache, chat-template boundary, position, or prefill change.
---

# Context System Change Gate

Apply the repository `AGENTS.md` files before using this workflow.

`SUBAGENT_POLICY_ONE_HELPER_MAX`: the main agent owns this workflow. Use no helper for simple
work and at most one distinct helper for a user task or implementation round. Reuse that helper
for later phases, never create a sequence of role agents, and never allow the helper to spawn an
agent. Only explicit user approval may raise this budget.

## Gate a change

1. Set the state to `READ_ONLY_DISCOVERY`.
2. Inspect the branch, HEAD, remotes, dirty files, relevant code, tests, and documentation without
   changing repository files.
3. Use `$context-system-code-audit` to establish the current behavior and exact evidence.
4. Write a plan with ID `PLAN-CS-YYYYMMDD-RN` and state `PLAN_PENDING_APPROVAL`.
5. Include exact allowed files and symbols, intended behavior, non-goals, invariants, risks,
   rollback, local checks, remote tests, and pass criteria.
6. End with a request to approve or modify that exact Plan ID, then stop.

Treat only a later user message that explicitly approves the Plan ID and scope as approval. Do not
treat the initial request, silence, a tool permission, or general encouragement as approval.

## Implement an approved plan

1. Record `APPROVED_PLAN_ID`, `APPROVED_ROUND`, and `ALLOWED_FILES`.
2. Reject or re-plan any edit outside those files or outside the approved behavior.
3. Use one writer. The main agent is the default writer and records the approval metadata before
   editing. If the single allowed helper is the writer, require it to repeat the metadata first.
4. Preserve pre-existing user changes and avoid destructive Git operations.
5. Run `$context-system-validation`, then complete a separate read-only review phase. The main
   agent performs any phase not assigned to the one existing helper; do not spawn another verifier
   or reviewer.
6. Count implementation plus complete validation plus review as one round. Stop after round 5 if
   the task still cannot pass.
7. On success, use `$system-test-checkpoint`.

If validation reveals a fix inside the approved files and semantics, correct it and repeat the
round. If the required change expands scope, return to `PLAN_PENDING_APPROVAL`.
