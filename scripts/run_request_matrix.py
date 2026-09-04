from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REQUEST_SUFFIXES = {".json", ".jsonl"}
OUTCOMES = ("success", "failure")
EXPECTED_SHUTDOWN_CODES = {0, -signal.SIGTERM, 128 + signal.SIGTERM}


@dataclass(frozen=True)
class ServerConfig:
    config_id: str
    name: str
    source_path: Path
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    readiness_url: str
    endpoint_url: str


@dataclass(frozen=True)
class ConversationCase:
    case_id: str
    source_path: Path
    requests: tuple[dict[str, Any], ...]


class ServerStartupError(RuntimeError):
    def __init__(self, cause: BaseException, server_returncode: int | None) -> None:
        super().__init__(str(cause))
        self.error_type = type(cause).__name__
        self.server_returncode = server_returncode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def discover_request_files(input_path: Path) -> list[Path]:
    path = input_path.expanduser()
    if not path.exists():
        raise ValueError(f"Request input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in REQUEST_SUFFIXES:
            raise ValueError(f"Request file must be .json or .jsonl: {path}")
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"Request input is neither a file nor directory: {path}")

    files = sorted(
        (
            item.resolve()
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in REQUEST_SUFFIXES
        ),
        key=lambda item: item.as_posix(),
    )
    if not files:
        raise ValueError(f"No .json or .jsonl request files found in {path}.")
    return files


def _server_urls(host: str, port: int) -> tuple[str, str]:
    connect_host = {"0.0.0.0": "127.0.0.1", "::": "::1", "[::]": "::1"}.get(host, host)
    if ":" in connect_host and not connect_host.startswith("["):
        connect_host = f"[{connect_host}]"
    base_url = f"http://{connect_host}:{port}/v1"
    return f"{base_url}/models", f"{base_url}/chat/completions"


def load_server_configs(path: Path) -> list[ServerConfig]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".json":
        raise ValueError(f"Server config must be one .json file: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"configurations"}:
        raise ValueError(f"Config manifest must contain only 'configurations': {path}")
    entries = manifest["configurations"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"'configurations' must be a non-empty array: {path}")

    configs: list[ServerConfig] = []
    names: set[str] = set()
    required = {"name", "command", "args", "cwd", "host", "port"}
    for index, entry in enumerate(entries, start=1):
        location = f"{path}:configurations[{index - 1}]"
        if not isinstance(entry, dict):
            raise ValueError(f"Configuration must be an object: {location}")
        missing = sorted(required - set(entry))
        unknown = sorted(set(entry) - required - {"env"})
        if missing or unknown:
            raise ValueError(f"Invalid fields at {location}; missing={missing}, unknown={unknown}")

        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or name.strip() in names:
            raise ValueError(f"Configuration name must be unique and non-empty: {location}")
        name = name.strip()
        names.add(name)
        command = entry["command"]
        args = entry["args"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) for item in command)
        ):
            raise ValueError(f"'command' must be a non-empty string array: {location}")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError(f"'args' must be a string array: {location}")
        if any(
            arg in {"--host", "--port"} or arg.startswith(("--host=", "--port=")) for arg in args
        ):
            raise ValueError(f"Set host/port as fields, not in args: {location}")

        cwd_value = entry["cwd"]
        if not isinstance(cwd_value, str) or not cwd_value:
            raise ValueError(f"'cwd' must be a non-empty string: {location}")
        candidate = Path(cwd_value).expanduser()
        cwd = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate
        if not cwd.is_dir():
            raise ValueError(f"cwd is not a directory: {cwd}")

        host = entry["host"]
        port = entry["port"]
        if not isinstance(host, str) or not host:
            raise ValueError(f"'host' must be a non-empty string: {location}")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"'port' must be an integer between 1 and 65535: {location}")
        readiness_url, endpoint_url = _server_urls(host, port)

        env = entry.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            raise ValueError(f"'env' must contain only string keys and values: {location}")

        configs.append(
            ServerConfig(
                config_id=f"config-{index:03d}",
                name=name,
                source_path=path,
                argv=tuple(command + args + ["--host", host, "--port", str(port)]),
                cwd=cwd,
                env=env,
                readiness_url=readiness_url,
                endpoint_url=endpoint_url,
            )
        )
    return configs


def _unwrap_request(record: Any, location: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Request must be a JSON object: {location}")
    request = record.get("request", record)
    if not isinstance(request, dict):
        raise ValueError(f"Capture request must be a JSON object: {location}")
    return request


def _load_requests(path: Path) -> tuple[dict[str, Any], ...]:
    if path.suffix.lower() == ".jsonl":
        records: list[tuple[Any, str]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append((json.loads(line), f"{path}:{line_number}"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        items = data if isinstance(data, list) else [data]
        records = [(item, f"{path}:item {index}") for index, item in enumerate(items, 1)]

    requests = tuple(_unwrap_request(record, location) for record, location in records)
    if not requests:
        raise ValueError(f"Conversation case contains no requests: {path}")
    return requests


def load_conversation_cases(paths: Sequence[Path]) -> list[ConversationCase]:
    return [
        ConversationCase(f"request-{index:06d}", path, _load_requests(path))
        for index, path in enumerate(paths, 1)
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()


def _prepare_output(
    output_dir: Path,
    configs: Sequence[ServerConfig],
    cases: Sequence[ConversationCase],
    timeouts: dict[str, float],
) -> dict[str, Any]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "server-logs"
    log_dir.mkdir()

    shutil.copyfile(configs[0].source_path, output_dir / "config.json")
    for case in cases:
        case_dir = output_dir / case.case_id
        case_dir.mkdir()
        for request in case.requests:
            _append_jsonl(case_dir / "requests.jsonl", request)
        (case_dir / "results.jsonl").touch()

    return {
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "matrix": {
            "config_count": len(configs),
            "case_count": len(cases),
            "request_count": sum(len(case.requests) for case in cases),
            "expected_cells": len(configs) * len(cases),
            "completed_cells": 0,
        },
        "timeouts_seconds": timeouts,
        "counts": dict.fromkeys(OUTCOMES, 0),
        "configs": [
            {
                "config_id": config.config_id,
                "name": config.name,
                "source_path": str(config.source_path),
                "log_path": f"server-logs/{config.config_id}.log",
            }
            for config in configs
        ],
        "cases": [
            {
                "case_id": case.case_id,
                "source_path": str(case.source_path),
                "request_count": len(case.requests),
            }
            for case in cases
        ],
    }


def _wait_ready(
    process: subprocess.Popen[bytes],
    config: ServerConfig,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    with httpx.Client(trust_env=False) as client:
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(f"Server exited before readiness with code {returncode}.")
            try:
                response = client.get(config.readiness_url, timeout=2.0)
                if response.is_success:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
    raise TimeoutError(f"Readiness timed out after {timeout:g}s; last result: {last_error}")


def _stop_server(process: subprocess.Popen[bytes], timeout: float) -> tuple[int | None, bool]:
    if process.poll() is not None:
        return process.returncode, False
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
        return process.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return process.returncode, True


@contextmanager
def running_server(
    config: ServerConfig,
    log_path: Path,
    startup_timeout: float,
    shutdown_timeout: float,
) -> Iterator[subprocess.Popen[bytes]]:
    process: subprocess.Popen[bytes] | None = None
    ready = False
    with log_path.open("wb") as log:
        try:
            try:
                environment = os.environ.copy()
                environment.update(config.env)
                process = subprocess.Popen(
                    config.argv,
                    cwd=config.cwd,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_ready(process, config, startup_timeout)
                ready = True
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                returncode = process.poll() if process else None
                if process:
                    returncode, _ = _stop_server(process, shutdown_timeout)
                    process = None
                raise ServerStartupError(exc, returncode) from exc
            yield process
        finally:
            if process:
                returncode, forced = _stop_server(process, shutdown_timeout)
                if ready and forced:
                    print(f"[{config.name}] Server required SIGKILL.", file=sys.stderr)
                elif ready and returncode not in EXPECTED_SHUTDOWN_CODES:
                    print(
                        f"[{config.name}] Server stopped with code {returncode}.", file=sys.stderr
                    )


def _error(error_type: str, message: str, returncode: int | None = None) -> dict[str, Any]:
    return {
        "type": error_type,
        "message": message,
        "server_returncode": returncode,
    }


def _send_request(
    client: httpx.Client,
    config: ServerConfig,
    request: dict[str, Any],
    request_index: int,
) -> dict[str, Any]:
    try:
        response = client.post(config.endpoint_url, json=request)
    except Exception as exc:
        return {
            "request_index": request_index,
            "outcome": "failure",
            "response": None,
            "error": _error(type(exc).__name__, str(exc)),
        }

    try:
        json_body: Any = response.json()
    except ValueError:
        json_body = None
    success = response.is_success
    return {
        "request_index": request_index,
        "outcome": "success" if success else "failure",
        "response": {
            "status_code": response.status_code,
            "headers": list(response.headers.multi_items()),
            "body_text": response.text,
            "json_body": json_body,
        },
        "error": (
            None
            if success
            else _error(
                "HTTPStatusError",
                f"HTTP {response.status_code} returned by {config.endpoint_url}",
            )
        ),
    }


def _run_case(
    client: httpx.Client,
    process: subprocess.Popen[bytes],
    config: ServerConfig,
    case: ConversationCase,
) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.monotonic()
    chain: list[dict[str, Any]] = []

    for index, request in enumerate(case.requests, 1):
        returncode = process.poll()
        if returncode is None:
            result = _send_request(client, config, request, index)
            returncode = process.poll()
            if result["error"] and returncode is not None:
                result["error"]["server_returncode"] = returncode
        else:
            result = {
                "request_index": index,
                "outcome": "failure",
                "response": None,
                "error": _error(
                    "RuntimeError",
                    f"Server exited with code {returncode} before request.",
                    returncode,
                ),
            }
        chain.append(result)

    first_error = next((result["error"] for result in chain if result["error"]), None)
    return {
        "config_id": config.config_id,
        "config_name": config.name,
        "case_id": case.case_id,
        "outcome": "failure" if first_error else "success",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "response_chain": chain,
        "error": first_error,
    }


def _startup_failure(
    config: ServerConfig,
    case: ConversationCase,
    exc: ServerStartupError,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "config_id": config.config_id,
        "config_name": config.name,
        "case_id": case.case_id,
        "outcome": "failure",
        "started_at": now,
        "finished_at": now,
        "duration_seconds": 0.0,
        "response_chain": [],
        "error": _error(exc.error_type, str(exc), exc.server_returncode),
    }


def _record(output_dir: Path, state: dict[str, Any], result: dict[str, Any]) -> None:
    _append_jsonl(output_dir / result["case_id"] / "results.jsonl", result)
    state["counts"][result["outcome"]] += 1
    state["matrix"]["completed_cells"] += 1


def _finish(
    output_dir: Path, state: dict[str, Any], status: str, best_effort: bool = False
) -> None:
    state["status"] = status
    state["finished_at"] = _utc_now()
    state["failed_cells"] = state["counts"]["failure"]
    state["incomplete_cells"] = (
        state["matrix"]["expected_cells"] - state["matrix"]["completed_cells"]
    )
    try:
        _write_json(output_dir / "run.json", state)
    except OSError as exc:
        if not best_effort:
            raise
        print(f"Could not write final run.json: {exc}", file=sys.stderr)


def run_matrix(
    configs: Sequence[ServerConfig],
    cases: Sequence[ConversationCase],
    *,
    output_dir: Path,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
) -> dict[str, Any]:
    if not configs or not cases:
        raise ValueError("At least one server configuration and conversation case are required.")
    timeouts = {
        "startup": startup_timeout,
        "request": request_timeout,
        "shutdown": shutdown_timeout,
    }
    if any(value <= 0 for value in timeouts.values()):
        raise ValueError("Timeouts must be positive.")

    output_dir = output_dir.expanduser().resolve()
    state = _prepare_output(output_dir, configs, cases, timeouts)
    try:
        for config_number, config in enumerate(configs, 1):
            print(f"[{config_number}/{len(configs)}] Starting {config.name}")
            try:
                with running_server(
                    config,
                    output_dir / "server-logs" / f"{config.config_id}.log",
                    startup_timeout,
                    shutdown_timeout,
                ) as process:
                    print(f"[{config.name}] Ready at {config.endpoint_url}")
                    with httpx.Client(timeout=request_timeout, trust_env=False) as client:
                        for case_number, case in enumerate(cases, 1):
                            print(
                                f"[{config.name}] Case {case_number}/{len(cases)}: "
                                f"{case.case_id} ({len(case.requests)} requests)"
                            )
                            _record(output_dir, state, _run_case(client, process, config, case))
            except ServerStartupError as exc:
                print(f"[{config.name}] Startup failed: {exc}", file=sys.stderr)
                for case in cases:
                    _record(output_dir, state, _startup_failure(config, case, exc))
    except KeyboardInterrupt:
        _finish(output_dir, state, "interrupted", best_effort=True)
        raise
    except BaseException:
        _finish(output_dir, state, "failed", best_effort=True)
        raise

    status = "success" if not state["counts"]["failure"] else "failed"
    _finish(output_dir, state, status)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run server configurations against conversation request cases."
    )
    parser.add_argument("--configs", required=True, type=Path, help="One JSON config manifest.")
    parser.add_argument(
        "--requests",
        required=True,
        type=Path,
        help="One JSON/JSONL conversation file or directory.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        configs = load_server_configs(args.configs)
        cases = load_conversation_cases(discover_request_files(args.requests))
        state = run_matrix(
            configs,
            cases,
            output_dir=args.output_dir,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            shutdown_timeout=args.shutdown_timeout,
        )
    except KeyboardInterrupt:
        print("Interrupted; the active server was stopped.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Matrix {state['status']}: {state['counts']['success']}/"
        f"{state['matrix']['expected_cells']} cells succeeded."
    )
    return 0 if state["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
