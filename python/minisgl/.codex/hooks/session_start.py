#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


TARGET_SOURCES = {"startup", "resume", "compact"}
DOC_RELATIVE_PATHS = (
    Path("docs/codex/PROJECT_CONTEXT.md"),
    Path("docs/codex/CURRENT_STATE.md"),
    Path("docs/codex/HANDOFF_019fafc9-22dc-7c80-90a0-746297fe72eb.md"),
)


def emit(additional_context: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": additional_context,
                }
            },
            ensure_ascii=False,
        )
    )


def bounded(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head_size = (limit * 2) // 3
    tail_size = limit - head_size
    return (
        text[:head_size].rstrip()
        + "\n... [bounded excerpt omitted] ...\n"
        + text[-tail_size:].lstrip()
    )


def run_git(repo_root: Path, *args: str, limit: int) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return f"git {' '.join(args)} failed with exit {result.returncode}"
    return bounded(result.stdout or "<empty>", limit)


def read_doc(repo_root: Path, relative_path: Path) -> str:
    path = repo_root / relative_path
    try:
        return bounded(path.read_text(encoding="utf-8"), 3000)
    except OSError as error:
        return f"Unable to read {relative_path}: {type(error).__name__}"


def load_source() -> tuple[str, str | None]:
    raw = sys.stdin.read().strip()
    if not raw:
        return "startup", None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "startup", "stdin was not valid JSON; using startup defaults"
    if not isinstance(payload, dict):
        return "startup", "stdin JSON was not an object; using startup defaults"
    return str(payload.get("source") or "startup"), None


def main() -> int:
    source, input_warning = load_source()
    if source not in TARGET_SOURCES:
        emit(f"mini-sglang SessionStart source={source!r}; no project snapshot injected.")
        return 0

    try:
        script_path = Path(__file__).resolve()
        repo_root = script_path.parents[4]
        workspace_root = repo_root.parent
        if not (repo_root / ".git").exists():
            emit("mini-sglang context loader could not locate the repository from its script path.")
            return 0

        sections = [
            "mini-sglang automatic project context",
            f"SessionStart source: {source}",
            f"Codex project root: {workspace_root}",
            f"Repository: {repo_root}",
            (
                "Authority order: current working tree > CURRENT_STATE snapshot > handoff > "
                "transcript/old pasted explanations. Re-read applicable AGENTS.md and never infer "
                "Plan approval from this injected context."
            ),
        ]
        if input_warning:
            sections.append(f"Hook input warning: {input_warning}")

        sections.extend(
            [
                "\n## Git branch/status\n"
                + run_git(repo_root, "status", "--short", "--branch", limit=5000),
                "\n## Git HEAD\n"
                + run_git(repo_root, "rev-parse", "--short", "HEAD", limit=200),
                "\n## Git diff stat (no full diff)\n"
                + run_git(repo_root, "diff", "--stat", "--no-ext-diff", limit=3000),
                "\n## Recent log\n"
                + run_git(repo_root, "log", "--oneline", "-8", limit=1600),
            ]
        )
        for relative_path in DOC_RELATIVE_PATHS:
            sections.append(f"\n## {relative_path}\n{read_doc(repo_root, relative_path)}")
        emit("\n".join(sections))
    except (IndexError, OSError, subprocess.SubprocessError) as error:
        emit(f"mini-sglang context loader failed safely: {type(error).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
