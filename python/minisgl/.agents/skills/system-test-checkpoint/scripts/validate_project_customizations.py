#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import re
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on InfiniAI-BUS
    tomllib = None


REQUIRED_AGENTS = {
    "context-code-mapper.toml": "context_code_mapper",
    "context-architect.toml": "context_architect",
    "context-implementer.toml": "context_implementer",
    "context-verifier.toml": "context_verifier",
    "context-reviewer.toml": "context_reviewer",
}
REQUIRED_SKILLS = {
    "context-system-change-gate",
    "context-system-code-audit",
    "context-system-documentation",
    "context-system-validation",
    "system-test-checkpoint",
}
REQUIRED_DOCS = {
    "PROJECT_CONTEXT.md": ("delta-marker", "staged", "pasted-text"),
    "CURRENT_STATE.md": ("b43e5c8", "17", "git status --short --branch"),
    "HANDOFF_019fafc9-22dc-7c80-90a0-746297fe72eb.md": (
        "PLAN-CS-20260804-R3",
        "compileall",
        "pytest",
        "FA3",
    ),
}


def fail(message: str) -> None:
    raise AssertionError(message)


def read_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if "TODO" in text:
        fail(f"{path}: unresolved TODO")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        fail(f"{path}: missing YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError:
        fail(f"{path}: unterminated YAML frontmatter")
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator:
            fail(f"{path}: malformed frontmatter line {line!r}")
        metadata[key.strip()] = value.strip()
    if set(metadata) != {"name", "description"}:
        fail(f"{path}: frontmatter must contain only name and description")
    return metadata


def read_agent_toml(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)

    data: dict[str, str] = {}
    for field in ("name", "description"):
        match = re.search(rf'(?m)^{field}\s*=\s*"([^"]+)"\s*$', text)
        if match:
            data[field] = match.group(1)
    instructions = re.search(
        r'(?ms)^developer_instructions\s*=\s*"""(.*?)"""\s*$',
        text,
    )
    if instructions:
        data["developer_instructions"] = instructions.group(1).strip()
    return data


def read_project_config(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text), text

    data: dict[str, object] = {}
    for field in (
        "model_auto_compact_token_limit",
        "tool_output_token_limit",
    ):
        match = re.search(rf"(?m)^{field}\s*=\s*(\d+)\s*$", text)
        if match:
            data[field] = int(match.group(1))
    for field in (
        "model_auto_compact_token_limit_scope",
        "experimental_compact_prompt_file",
    ):
        match = re.search(rf'(?m)^{field}\s*=\s*"([^"]+)"\s*$', text)
        if match:
            data[field] = match.group(1)
    return data, text


def validate_session_hook(hook_path: Path, workspace_root: Path) -> None:
    compile(hook_path.read_text(encoding="utf-8"), str(hook_path), "exec")
    for source in ("startup", "resume", "compact"):
        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=json.dumps({"source": source}),
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            fail(f"{hook_path}: {source} smoke exited {result.returncode}")
        try:
            payload = json.loads(result.stdout)
            output = payload["hookSpecificOutput"]
            context = output["additionalContext"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            fail(f"{hook_path}: invalid {source} JSON output: {error}")
        if output.get("hookEventName") != "SessionStart":
            fail(f"{hook_path}: wrong hookEventName for {source}")
        for marker in (source, "Git branch/status", "PROJECT_CONTEXT.md", "CURRENT_STATE.md"):
            if marker not in context:
                fail(f"{hook_path}: {source} output missing {marker!r}")

    empty_input = subprocess.run(
        [sys.executable, str(hook_path)],
        input="",
        cwd=workspace_root,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if empty_input.returncode != 0:
        fail(f"{hook_path}: empty-input smoke exited {empty_input.returncode}")
    json.loads(empty_input.stdout)


def main() -> int:
    script_path = Path(__file__).resolve()
    minisgl_dir = script_path.parents[4]
    repo_root = Path(
        subprocess.check_output(
            ["git", "-C", str(minisgl_dir), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )

    for agents_file in (repo_root / "AGENTS.md", minisgl_dir / "AGENTS.md"):
        if not agents_file.is_file():
            fail(f"missing {agents_file}")

    docs_dir = repo_root / "docs" / "codex"
    for filename, markers in REQUIRED_DOCS.items():
        path = docs_dir / filename
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                fail(f"{path}: missing {marker!r}")

    config_path = minisgl_dir / ".codex" / "config.toml"
    config, config_text = read_project_config(config_path)
    expected_config = {
        "model_auto_compact_token_limit": 150000,
        "model_auto_compact_token_limit_scope": "total",
        "tool_output_token_limit": 8000,
        "experimental_compact_prompt_file": (
            "../mini-sglang/python/minisgl/.codex/compact_prompt.txt"
        ),
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            fail(f"{config_path}: expected {field}={expected!r}")
    for marker in (
        "[[hooks.SessionStart]]",
        'matcher = "^(startup|resume|compact)$"',
        "[[hooks.SessionStart.hooks]]",
        "additionalContextLimit = 5000",
        'python3 "mini-sglang/python/minisgl/.codex/hooks/session_start.py"',
    ):
        if marker not in config_text:
            fail(f"{config_path}: missing {marker!r}")

    compact_prompt = minisgl_dir / ".codex" / "compact_prompt.txt"
    prompt_text = compact_prompt.read_text(encoding="utf-8")
    for marker in (
        "APPROVED_PLAN_ID",
        "ALLOWED_FILES",
        "git status",
        "不能推断修改授权",
        "HANDOFF_019fafc9-22dc-7c80-90a0-746297fe72eb.md",
    ):
        if marker not in prompt_text:
            fail(f"{compact_prompt}: missing {marker!r}")
    configured_prompt = (
        repo_root.parent / ".codex" / expected_config["experimental_compact_prompt_file"]
    ).resolve()
    if configured_prompt != compact_prompt.resolve():
        fail(f"{config_path}: compact prompt path is not valid from the workspace .codex base")

    hook_path = minisgl_dir / ".codex" / "hooks" / "session_start.py"
    validate_session_hook(hook_path, repo_root.parent)

    agent_dir = minisgl_dir / ".codex" / "agents"
    for filename, expected_name in REQUIRED_AGENTS.items():
        path = agent_dir / filename
        data = read_agent_toml(path)
        for field in ("name", "description", "developer_instructions"):
            if not data.get(field):
                fail(f"{path}: missing {field}")
        if data["name"] != expected_name:
            fail(f"{path}: expected name {expected_name!r}")

    skill_dir = minisgl_dir / ".agents" / "skills"
    discovered = {path.parent.name for path in skill_dir.glob("*/SKILL.md")}
    if discovered != REQUIRED_SKILLS:
        fail(f"skills mismatch: expected {sorted(REQUIRED_SKILLS)}, found {sorted(discovered)}")

    for skill_name in sorted(REQUIRED_SKILLS):
        path = skill_dir / skill_name / "SKILL.md"
        metadata = read_frontmatter(path)
        if metadata["name"] != skill_name:
            fail(f"{path}: name does not match directory")
        ui_path = path.parent / "agents" / "openai.yaml"
        ui_text = ui_path.read_text(encoding="utf-8")
        for marker in ("display_name:", "short_description:", "default_prompt:"):
            if marker not in ui_text:
                fail(f"{ui_path}: missing {marker}")
        if f"${skill_name}" not in ui_text:
            fail(f"{ui_path}: default_prompt must mention ${skill_name}")

    sync_script = skill_dir / "system-test-checkpoint" / "scripts" / "sync_remote_checkpoint.sh"
    subprocess.run(["bash", "-n", str(sync_script)], check=True)
    print(
        f"PASS: {len(REQUIRED_AGENTS)} agents, {len(REQUIRED_SKILLS)} skills, "
        f"{len(REQUIRED_DOCS)} context docs, compact config/prompt, SessionStart hook, "
        "AGENTS.md files, and checkpoint script are valid"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
