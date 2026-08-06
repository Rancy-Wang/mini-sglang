---
name: system-test-checkpoint
description: Finish any mini-sglang conversation that changed repository files by committing only the task changes on System-test, pushing origin, fast-forward pulling on InfiniAI-BUS, running proportional remote tests, and reporting a durable checkpoint. Also use to diagnose local, GitHub, and remote synchronization.
---

# System-test Checkpoint

Treat the local repository as the only edit source. Never repair a failure by editing the remote
working tree.

`SUBAGENT_POLICY_ONE_HELPER_MAX`: the main agent performs checkpoint, push, remote synchronization,
and reporting. Do not spawn an agent for checkpoint work; if the task already has one helper, it
must not spawn agents and does not authorize a second helper.

## Prepare the checkpoint

1. Verify the local repository, `System-test` branch, configured `origin`, HEAD, and dirty state.
2. Review `git diff` and `git diff --cached`. Preserve unrelated and pre-existing changes.
3. Run available local static or lightweight checks.
4. Stage only explicit task paths. Do not use `git add -A`.
5. Create a concise checkpoint commit. Do not create an empty commit for a read-only turn.
6. Record the commit hash and title.

Prefer one logical checkpoint per conversation. Do not rewrite already-pushed shared
`System-test` history merely to squash follow-up fixes.

## Push, pull, and test

From the repository root, run:

```bash
python/minisgl/.agents/skills/system-test-checkpoint/scripts/sync_remote_checkpoint.sh \
  "<remote test command>"
```

The script verifies the branch and origin, pushes `System-test`, performs `git pull --ff-only` in
`/share/wangruoxi/repo/mini-sglang`, checks that remote HEAD matches the pushed commit, and runs the
provided test command.

Choose tests proportional to the diff. Use
`validate_project_customizations.py` for agent/skill-only changes. Use focused pytest targets for
Python logic and add GPU/end-to-end checks when kernels, scheduling, cache ownership, serving, or
model behavior changes.

If pull is blocked by remote changes, stop. Do not stash, reset, clean, overwrite, or edit them.
Report the exact status and conflicting paths. If a remote test fails, fix locally, commit, push,
pull, and retest.

## Report the checkpoint

Include:

- `CHECKPOINT` or `NO-CHANGE checkpoint`;
- branch, commit hash, and commit title;
- push and remote fast-forward status;
- remote HEAD equality;
- exact local and remote test commands with outcomes;
- pre-existing failures, environment blockers, and remaining risks.
