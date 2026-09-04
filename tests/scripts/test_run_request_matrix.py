from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

import scripts.run_request_matrix as matrix
from scripts.run_request_matrix import (
    EXPECTED_SHUTDOWN_CODES,
    ServerConfig,
    _send_request,
    _stop_server,
    build_parser,
    discover_request_files,
    load_conversation_cases,
    load_server_configs,
    run_matrix,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _server_config(
    name: str,
    command: list[str],
    *,
    args: list[str] | None = None,
    cwd: str = ".",
    host: str = "127.0.0.1",
    port: int | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    return {
        "name": name,
        "command": command,
        "args": args or [],
        "cwd": cwd,
        "host": host,
        "port": port or _free_port(),
        "env": env or {},
    }


def _write_manifest(path: Path, configurations: list[dict]) -> None:
    _write_json(path, {"configurations": configurations})


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_accepts_one_config_manifest_and_one_request_path() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--configs",
            "servers.json",
            "--requests",
            "case.jsonl",
            "--output-dir",
            "output",
        ]
    )
    assert args.configs == Path("servers.json")
    assert args.requests == Path("case.jsonl")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--configs",
                "server-a.json",
                "server-b.json",
                "--requests",
                "case.jsonl",
                "--output-dir",
                "output",
            ]
        )


def test_each_request_file_becomes_one_conversation_case(tmp_path: Path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    _write_json(
        request_dir / "a.json",
        [{"model": "raw"}, {"request": {"model": "captured"}}],
    )
    (request_dir / "b.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"model": "round-1"}),
                json.dumps({"request": {"model": "round-2"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_conversation_cases(discover_request_files(request_dir))

    assert [case.case_id for case in cases] == ["request-000001", "request-000002"]
    assert [[request["model"] for request in case.requests] for case in cases] == [
        ["raw", "captured"],
        ["round-1", "round-2"],
    ]


def test_load_config_manifest_resolves_cwd_host_port_and_argv(tmp_path: Path) -> None:
    config_path = tmp_path / "servers.json"
    _write_manifest(
        config_path,
        [
            _server_config(
                "server",
                [sys.executable, "-m", "minisgl"],
                args=["--label", "alpha"],
                host="0.0.0.0",
                port=31234,
                env={"FLAG": "true"},
            )
        ],
    )

    config = load_server_configs(config_path)[0]

    assert config.config_id == "config-001"
    assert config.cwd == tmp_path
    assert config.argv[-6:] == (
        "--label",
        "alpha",
        "--host",
        "0.0.0.0",
        "--port",
        "31234",
    )
    assert config.endpoint_url == "http://127.0.0.1:31234/v1/chat/completions"
    assert config.env == {"FLAG": "true"}


@pytest.mark.parametrize(
    "manifest",
    [
        {"version": 1, "configurations": []},
        {
            "configurations": [
                {
                    "name": "missing-fields",
                }
            ]
        },
        {
            "configurations": [
                _server_config(
                    "host-in-args",
                    [sys.executable],
                    args=["--host", "127.0.0.1"],
                )
            ]
        },
        {
            "configurations": [
                _server_config(
                    "non-string-env",
                    [sys.executable],
                    env={"FLAG": "true"} | {"COUNT": 1},
                )
            ]
        },
        {"configurations": [_server_config("string-port", [sys.executable]) | {"port": "30000"}]},
    ],
)
def test_config_manifest_rejects_invalid_schema(tmp_path: Path, manifest: dict) -> None:
    path = tmp_path / "invalid.json"
    _write_json(path, manifest)
    with pytest.raises(ValueError):
        load_server_configs(path)


def test_stop_server_terminates_parent_then_kills_group_only_on_timeout(monkeypatch) -> None:
    process = Mock(pid=1234, returncode=143)
    process.poll.return_value = None
    killpg = Mock()
    monkeypatch.setattr(matrix.os, "killpg", killpg)

    assert _stop_server(process, 5) == (143, False)
    process.terminate.assert_called_once_with()
    killpg.assert_not_called()
    assert 143 in EXPECTED_SHUTDOWN_CODES

    process.reset_mock()
    process.returncode = -signal.SIGKILL
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 5), None]

    assert _stop_server(process, 5) == (-signal.SIGKILL, True)
    process.terminate.assert_called_once_with()
    killpg.assert_called_once_with(1234, signal.SIGKILL)


def test_run_matrix_records_one_cell_per_config_and_case(tmp_path: Path, monkeypatch) -> None:
    server_script = tmp_path / "fake_server.py"
    server_script.write_text(
        """
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
request_number = 0

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = b'{"data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global request_number
        request_number += 1
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        body = json.dumps({
            "label": args.label,
            "request_number": request_number,
            "request": request,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer((args.host, args.port), Handler).serve_forever()
""".strip() + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "servers.json"
    _write_manifest(
        config_path,
        [
            _server_config(
                label,
                [sys.executable, str(server_script)],
                args=["--label", label],
            )
            for label in ("alpha", "beta")
        ],
    )

    request_dir = tmp_path / "requests-in"
    request_dir.mkdir()
    (request_dir / "case-a.jsonl").write_text(
        '{"model":"round-1"}\n{"request":{"model":"round-2"}}\n',
        encoding="utf-8",
    )
    _write_json(request_dir / "case-b.json", {"model": "round-3"})

    configs = load_server_configs(config_path)
    cases = load_conversation_cases(discover_request_files(request_dir))
    output_dir = tmp_path / "output"
    run_json_writes = 0
    original_write_json = matrix._write_json

    def count_run_json_write(path: Path, payload: object) -> None:
        nonlocal run_json_writes
        if path.name == "run.json":
            run_json_writes += 1
        original_write_json(path, payload)

    monkeypatch.setattr(matrix, "_write_json", count_run_json_write)
    state = run_matrix(
        configs,
        cases,
        output_dir=output_dir,
        startup_timeout=10,
        request_timeout=10,
        shutdown_timeout=5,
    )

    assert state["status"] == "success"
    assert state["matrix"] == {
        "config_count": 2,
        "case_count": 2,
        "request_count": 3,
        "expected_cells": 4,
        "completed_cells": 4,
    }
    assert state["counts"] == {"success": 4, "failure": 0}
    assert run_json_writes == 1
    assert sorted(path.name for path in output_dir.glob("request-*")) == [
        "request-000001",
        "request-000002",
    ]

    first_case_results = _read_jsonl(output_dir / "request-000001" / "results.jsonl")
    assert [result["config_name"] for result in first_case_results] == ["alpha", "beta"]
    assert [len(result["response_chain"]) for result in first_case_results] == [2, 2]
    for result in first_case_results:
        bodies = [
            round_result["response"]["json_body"] for round_result in result["response_chain"]
        ]
        assert [body["request"]["model"] for body in bodies] == ["round-1", "round-2"]
        assert [body["request_number"] for body in bodies] == [1, 2]

    stored_requests = _read_jsonl(output_dir / "request-000001" / "requests.jsonl")
    assert [request["model"] for request in stored_requests] == ["round-1", "round-2"]
    assert json.loads((output_dir / "config.json").read_text()) == json.loads(
        config_path.read_text()
    )
    assert len(list((output_dir / "server-logs").glob("*.log"))) == 2


def test_startup_failure_writes_one_failure_for_the_whole_case(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.json"
    _write_manifest(
        config_path,
        [
            _server_config(
                "broken",
                [sys.executable, "-c", "import sys; sys.exit(7)"],
            )
        ],
    )
    request_path = tmp_path / "case.jsonl"
    request_path.write_text('{"model":"one"}\n{"model":"two"}\n', encoding="utf-8")
    configs = load_server_configs(config_path)
    cases = load_conversation_cases([request_path.resolve()])
    output_dir = tmp_path / "output"

    state = run_matrix(
        configs,
        cases,
        output_dir=output_dir,
        startup_timeout=5,
        request_timeout=5,
        shutdown_timeout=2,
    )

    assert state["status"] == "failed"
    assert state["counts"] == {"success": 0, "failure": 1}
    [result] = _read_jsonl(output_dir / "request-000001" / "results.jsonl")
    assert result["outcome"] == "failure"
    assert result["response_chain"] == []
    assert result["error"]["type"] == "RuntimeError"
    assert result["error"]["server_returncode"] == 7


def test_send_preserves_failure_body_and_streaming_sse(tmp_path: Path) -> None:
    config = ServerConfig(
        config_id="config-001",
        name="test",
        source_path=tmp_path / "test.json",
        argv=(sys.executable, "--host", "127.0.0.1", "--port", "30001"),
        cwd=tmp_path,
        env={},
        readiness_url="http://test/v1/models",
        endpoint_url="http://test/v1/chat/completions",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("stream") is True:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text='data: {"choices":[]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(503, json={"error": "not ready"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        failure = _send_request(client, config, {"model": "test"}, 1)
        stream = _send_request(client, config, {"model": "test", "stream": True}, 2)

    assert failure["outcome"] == "failure"
    assert failure["response"]["status_code"] == 503
    assert failure["response"]["json_body"] == {"error": "not ready"}
    assert failure["error"]["type"] == "HTTPStatusError"
    assert stream["outcome"] == "success"
    assert stream["response"]["json_body"] is None
    assert stream["response"]["body_text"].endswith("data: [DONE]\n\n")
