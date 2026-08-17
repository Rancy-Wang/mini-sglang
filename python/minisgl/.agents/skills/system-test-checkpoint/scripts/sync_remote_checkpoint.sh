#!/usr/bin/env bash
set -euo pipefail

expected_origin_ssh="git@github.com:Rancy-Wang/mini-sglang.git"
expected_origin_short="git@github.com:Rancy-Wang/mini-sglang"
branch="System-test"
remote_host="InfiniAI-BUS"
remote_repo="/share/wangruoxi/repo/mini-sglang"
test_command="${1:-}"
test_command_b64="$(printf '%s' "$test_command" | base64 | tr -d '\n')"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$branch" ]]; then
  echo "ERROR: expected branch $branch, found $current_branch" >&2
  exit 2
fi

origin_url="$(git remote get-url origin)"
if [[ "$origin_url" != "$expected_origin_ssh" && "$origin_url" != "$expected_origin_short" ]]; then
  echo "ERROR: unexpected origin URL: $origin_url" >&2
  exit 2
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: tracked local changes remain; they will not be included in the pushed HEAD." >&2
  git status --short --branch
fi

expected_head="$(git rev-parse HEAD)"
git push origin "$branch"

ssh "$remote_host" bash -s -- "$remote_repo" "$branch" "$expected_head" "$test_command_b64" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_repo="$1"
branch="$2"
expected_head="$3"
test_command_b64="$4"
test_command="$(printf '%s' "$test_command_b64" | base64 --decode)"

cd "$remote_repo"
current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$branch" ]]; then
  echo "ERROR: remote expected branch $branch, found $current_branch" >&2
  exit 3
fi

echo "Remote status before pull:"
git status --short --branch
git pull --ff-only origin "$branch"

remote_head="$(git rev-parse HEAD)"
if [[ "$remote_head" != "$expected_head" ]]; then
  echo "ERROR: remote HEAD $remote_head does not match pushed HEAD $expected_head" >&2
  exit 4
fi

echo "Remote HEAD verified: $remote_head"
if [[ -n "$test_command" ]]; then
  echo "Remote test: $test_command"
  bash -lc "$test_command"
else
  echo "WARNING: no remote test command was supplied." >&2
fi
REMOTE_SCRIPT
