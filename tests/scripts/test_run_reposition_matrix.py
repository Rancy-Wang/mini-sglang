from __future__ import annotations

import asyncio
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from minisgl.benchmark.reposition_bcp import CapturedRequest, ReplayTask

import scripts.run_reposition_matrix as matrix


def _messages(turns: int) -> list[dict]:
    result: list[dict] = [{"role": "user", "content": "task"}]
    for turn in range(turns):
        call_id = f"call-{turn}"
        result.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "result"},
            ]
        )
    return result


def _task() -> ReplayTask:
    return ReplayTask(
        case_id="231",
        requests=tuple(
            CapturedRequest(
                case_id="231",
                physical_id=index,
                logical_id=index,
                retry_id=0,
                path=Path(f"/capture/{index}"),
                request={"model": "model", "messages": _messages(index), "stream": False},
            )
            for index in range(6)
        ),
    )


def test_replay_writes_raw_full_trajectory_and_rolling_plan(tmp_path, monkeypatch) -> None:
    async def fake_post(client, *, endpoint, request):
        del client, endpoint
        payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "complete answer"},
                    "finish_reason": "stop",
                }
            ]
        }
        raw = json.dumps(payload).encode()
        return raw, {
            "ok": True,
            "status_code": 200,
            "headers": [],
            "client_started_ns": 1,
            "client_first_byte_ns": 2,
            "client_finished_ns": 3,
            "raw_response_text": raw.decode(),
            "message": payload["choices"][0]["message"],
            "finish_reason": "stop",
            "server_metrics": None,
            "usage": None,
        }

    monkeypatch.setattr(matrix, "_post_capture", fake_post)
    records = asyncio.run(
        matrix.replay_tasks(
            [_task()],
            endpoints=["http://server/v1"],
            mode="rolling",
            concurrency=1,
            repetitions=1,
            output_dir=tmp_path / "result",
            request_overrides={"max_tokens": 32, "temperature": 0},
            request_timeout=10,
        )
    )

    assert len(records) == 6
    assert records[-1]["request"]["drop_rule"]["drop_messages"] == {"10": [1, 2]}
    assert records[-1]["request"]["reposition"] == [10]
    assert records[-1]["request"]["max_tokens"] == 32
    assert records[0]["audit"]["issues"] == [
        "tool_calls:count_mismatch",
        "tool_calls:oracle_name_mismatch",
    ]
    assert records[-1]["audit"]["issues"] == []
    assert (tmp_path / "result" / "trajectory.txt").read_text().count("complete answer") == 6
    with gzip.open(tmp_path / "result" / "result.jsonl.gz", "rt") as stream:
        assert len([json.loads(line) for line in stream]) == 6
    assert len(list((tmp_path / "result" / "raw").rglob("*.request.json.gz"))) == 6
    assert len(list((tmp_path / "result" / "raw").rglob("*.response.bin.gz"))) == 6


def test_rolling_replay_requires_retry_transition_and_h2d_metrics(tmp_path, monkeypatch) -> None:
    async def fake_post(client, *, endpoint, request):
        del client, endpoint, request
        response = {
            "ok": True,
            "status_code": 200,
            "headers": [],
            "client_started_ns": 1,
            "client_first_byte_ns": 2,
            "client_finished_ns": 3,
            "raw_response_text": "{}",
            "message": {"role": "assistant", "content": "complete answer"},
            "finish_reason": "stop",
            "server_metrics": {
                "request_received_ns": 1,
                "first_token_generated_ns": 2,
                "request_finished_ns": 3,
                "prompt_tokens": 100,
                "active_prompt_tokens": 20,
                "generated_tokens": 2,
                "completion_tokens": 2,
                "tokenize_invocations": 1,
                "context_stage_count": 3,
                "reposition_transition_count": 0,
                "reposition_h2d_bytes": 0,
                "reposition_d2h_bytes": 0,
            },
            "usage": {"completion_tokens": 2},
        }
        return b"{}", response

    monkeypatch.setattr(matrix, "_post_capture", fake_post)
    records = asyncio.run(
        matrix.replay_tasks(
            [_task()],
            endpoints=["http://server/v1"],
            mode="rolling",
            concurrency=1,
            repetitions=1,
            output_dir=tmp_path / "result",
            request_overrides={},
            request_timeout=10,
            request_selection="last",
            require_server_metrics=True,
        )
    )

    assert records[0]["audit"]["issues"] == [
        "system:cold_retry_h2d_missing",
        "system:cold_retry_transition_missing",
    ]


def test_proxy_metrics_audit_requires_retry_activity_only_when_enabled() -> None:
    request = {
        "messages": [{"role": "user", "content": "task"}],
        "reposition": [10],
    }
    parsed = {
        "message": {"role": "assistant", "content": "complete answer"},
        "finish_reason": "stop",
        "server_metrics": {
            "request_received_ns": 1,
            "first_token_generated_ns": 2,
            "request_finished_ns": 3,
            "prompt_tokens": 100,
            "active_prompt_tokens": 20,
            "generated_tokens": 2,
            "completion_tokens": 2,
            "tokenize_invocations": 1,
            "context_stage_count": 3,
            "reposition_transition_count": 0,
            "reposition_h2d_bytes": 0,
            "reposition_d2h_bytes": 0,
        },
    }

    optional = matrix._audit_proxy_response(
        parsed,
        request=request,
        require_server_metrics=False,
    )
    required = matrix._audit_proxy_response(
        parsed,
        request=request,
        require_server_metrics=True,
    )

    assert optional["issues"] == []
    assert required["issues"] == [
        "system:cold_retry_h2d_missing",
        "system:cold_retry_transition_missing",
    ]


def test_proxy_parser_exposes_server_metrics_requirement() -> None:
    parser = matrix.build_parser()
    args = parser.parse_args(
        [
            "proxy",
            "--port",
            "32100",
            "--upstream",
            "http://127.0.0.1:32000",
            "--mode",
            "rolling",
            "--audit-output",
            "audit.jsonl.gz",
            "--require-server-metrics",
        ]
    )

    assert args.require_server_metrics is True


def test_manifest_requires_exactly_one_endpoint_source(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "capture_roots": ["captures"],
                "cells": [
                    {
                        "name": "bad",
                        "mode": "full",
                        "concurrency": 1,
                        "endpoints": ["http://server/v1"],
                        "server_config": "servers.json",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="exactly one"):
        matrix._load_manifest(manifest)


def test_summary_keeps_latency_ttft_and_issue_counts() -> None:
    summary = matrix.summarize_records(
        [
            {
                "response": {
                    "ok": True,
                    "client_started_ns": 1_000_000,
                    "client_first_byte_ns": 3_000_000,
                    "client_finished_ns": 6_000_000,
                },
                "audit": {"issues": ["content:replacement"]},
            },
            {
                "response": {
                    "ok": False,
                    "client_started_ns": 10_000_000,
                    "client_finished_ns": 20_000_000,
                },
                "audit": {"issues": ["transport:request_or_parse_failure"]},
            },
        ]
    )

    assert summary["requests"] == 2
    assert summary["successful"] == 1
    assert summary["latency_ms"]["p50"] in {5.0, 10.0}
    assert summary["ttft_ms"]["p50"] == 2.0
    assert summary["issues"] == {
        "content:replacement": 1,
        "transport:request_or_parse_failure": 1,
    }


def test_request_matrix_audit_counts_success_and_generated_tokens(tmp_path: Path) -> None:
    case_dir = tmp_path / "request-000001"
    case_dir.mkdir()
    request = {
        "messages": [{"role": "user", "content": "task"}],
        "stream": False,
    }
    response_body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "complete answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 2},
    }
    (case_dir / "requests.jsonl").write_text(json.dumps(request) + "\n")
    (case_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config_name": "minisgl",
                "case_id": "request-000001",
                "response_chain": [
                    {
                        "request_index": 1,
                        "outcome": "success",
                        "response": {
                            "status_code": 200,
                            "body_text": json.dumps(response_body),
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    report = matrix.audit_request_matrix(tmp_path)

    assert report["summary"]["requests"] == 1
    assert report["summary"]["successful"] == 1
    assert report["summary"]["generated_tokens"] == 2
    assert report["summary"]["issues"] == {}
    assert report["records"][0]["response"]["ok"] is True

    metrics_report = matrix.audit_request_matrix(tmp_path, require_server_metrics=True)
    assert metrics_report["summary"]["issues"] == {"system:missing_server_metrics": 1}


def _write_fixed_matrix_case(
    root: Path,
    *,
    case_number: int,
    config_name: str,
    transition_count: int,
    h2d_bytes: int,
    stream: bool = False,
) -> None:
    case_dir = root / f"request-{case_number:06d}"
    case_dir.mkdir()
    request = {
        "messages": [{"role": "user", "content": "task"}],
        "reposition": [10],
        "stream": stream,
    }
    metrics = {
        "request_received_ns": 1,
        "first_token_generated_ns": 2,
        "request_finished_ns": 3,
        "prompt_tokens": 100,
        "active_prompt_tokens": 20,
        "generated_tokens": 2,
        "completion_tokens": 2,
        "tokenize_invocations": 1,
        "context_stage_count": 3,
        "reposition_transition_count": transition_count,
        "reposition_h2d_bytes": h2d_bytes,
        "reposition_d2h_bytes": 0,
    }
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "complete answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 2},
        "server_metrics": metrics,
    }
    body_text = json.dumps(payload)
    if stream:
        event = {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "complete answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 2},
            "server_metrics": metrics,
        }
        body_text = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    (case_dir / "requests.jsonl").write_text(json.dumps(request) + "\n")
    (case_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config_name": config_name,
                "case_id": case_dir.name,
                "response_chain": [
                    {
                        "request_index": 1,
                        "outcome": "success",
                        "response": {
                            "status_code": 200,
                            "body_text": body_text,
                        },
                    }
                ],
            }
        )
        + "\n"
    )


def test_fixed_matrix_accepts_zero_retry_for_exact_rehit_or_noop_reposition(
    tmp_path: Path,
) -> None:
    _write_fixed_matrix_case(
        tmp_path,
        case_number=1,
        config_name="minisgl",
        transition_count=2,
        h2d_bytes=128,
    )
    _write_fixed_matrix_case(
        tmp_path,
        case_number=2,
        config_name="minisgl",
        transition_count=0,
        h2d_bytes=0,
        stream=True,
    )

    report = matrix.audit_request_matrix(tmp_path, require_server_metrics=True)

    assert report["summary"]["issues"] == {}


def test_fixed_matrix_requires_retry_activity_per_server_config(tmp_path: Path) -> None:
    _write_fixed_matrix_case(
        tmp_path,
        case_number=1,
        config_name="minisgl-a",
        transition_count=0,
        h2d_bytes=0,
    )
    _write_fixed_matrix_case(
        tmp_path,
        case_number=2,
        config_name="minisgl-b",
        transition_count=1,
        h2d_bytes=64,
    )

    report = matrix.audit_request_matrix(tmp_path, require_server_metrics=True)

    assert report["summary"]["issues"] == {"system:retry_activity_missing": 1}


def test_fixed_matrix_rejects_retry_transition_h2d_mismatch(tmp_path: Path) -> None:
    _write_fixed_matrix_case(
        tmp_path,
        case_number=1,
        config_name="minisgl",
        transition_count=1,
        h2d_bytes=0,
    )

    report = matrix.audit_request_matrix(tmp_path, require_server_metrics=True)

    assert report["summary"]["issues"] == {
        "system:retry_activity_missing": 1,
        "system:retry_transition_h2d_inconsistent": 1,
    }


def test_request_matrix_audit_cli_writes_readable_trajectory(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    _write_fixed_matrix_case(
        matrix_dir,
        case_number=1,
        config_name="minisgl",
        transition_count=1,
        h2d_bytes=64,
    )
    output = tmp_path / "audit.json"

    assert (
        matrix.main(
            [
                "audit-request-matrix",
                "--matrix-dir",
                str(matrix_dir),
                "--output",
                str(output),
                "--require-server-metrics",
            ]
        )
        == 0
    )

    trajectory = output.with_suffix(".trajectory.txt").read_text(encoding="utf-8")
    assert "request-000001 / nonstream / 1" in trajectory
    assert "[content]\ncomplete answer" in trajectory
    assert "[issues] none" in trajectory


def test_cell_replay_forwards_selection_warmup_and_server_lifecycle(tmp_path, monkeypatch) -> None:
    seen: dict = {}

    async def fake_replay(tasks, **kwargs):
        seen["tasks"] = tasks
        seen.update(kwargs)
        return []

    monkeypatch.setattr(matrix, "replay_tasks", fake_replay)
    cell = matrix.ExperimentCell(
        name="full",
        mode="full",
        concurrency=4,
        repetitions=3,
        endpoints=("http://server/v1",),
        server_config=None,
        request_overrides={"max_tokens": 32},
        baseline_cell=None,
        framework="sglang",
        profile_on_slowdown=False,
        nsys=False,
        telemetry=False,
        request_selection="last",
        warmup=True,
    )

    records = matrix._run_cell_replay(
        cell,
        tasks=[_task()],
        output_dir=tmp_path / "cell",
        startup_timeout=1,
        request_timeout=2,
        shutdown_timeout=3,
    )

    assert records == []
    assert seen["request_selection"] == "last"
    assert seen["warmup"] is True
    assert seen["concurrency"] == 4
    assert seen["repetitions"] == 3


def test_managed_server_restarts_and_warms_up_for_every_repetition(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []
    lifecycle: list[tuple[str, str]] = []
    config = SimpleNamespace(config_id="server-1", endpoint_url="http://server/v1")

    @contextmanager
    def fake_running_server(config, log_path, startup_timeout, shutdown_timeout):
        del log_path, startup_timeout, shutdown_timeout
        lifecycle.append(("start", config.config_id))
        yield
        lifecycle.append(("stop", config.config_id))

    async def fake_replay(tasks, **kwargs):
        del tasks
        calls.append(kwargs)
        repetition = kwargs["repetition_start"]
        return [
            {
                "case_id": "231",
                "mode": "rolling",
                "endpoint_index": 0,
                "repetition": repetition,
                "turn": 6,
                "response": {"message": {"role": "assistant", "content": "ok"}},
                "audit": {"issues": []},
            }
        ]

    monkeypatch.setattr(matrix, "load_server_configs", lambda _: [config])
    monkeypatch.setattr(matrix, "running_server", fake_running_server)
    monkeypatch.setattr(matrix, "replay_tasks", fake_replay)
    cell = matrix.ExperimentCell(
        name="rolling",
        mode="rolling",
        concurrency=4,
        repetitions=3,
        endpoints=(),
        server_config=tmp_path / "servers.json",
        request_overrides={"max_tokens": 32},
        baseline_cell=None,
        framework="minisgl",
        profile_on_slowdown=False,
        nsys=False,
        telemetry=False,
        request_selection="last",
        warmup=True,
    )

    records = matrix._run_cell_replay(
        cell,
        tasks=[_task()],
        output_dir=tmp_path / "managed",
        startup_timeout=1,
        request_timeout=2,
        shutdown_timeout=3,
    )

    assert lifecycle == [
        ("start", "server-1"),
        ("stop", "server-1"),
        ("start", "server-1"),
        ("stop", "server-1"),
        ("start", "server-1"),
        ("stop", "server-1"),
    ]
    assert [call["repetition_start"] for call in calls] == [1, 2, 3]
    assert all(call["repetitions"] == 1 for call in calls)
    assert all(call["warmup"] is True for call in calls)
    assert all(call["require_server_metrics"] is True for call in calls)
    assert [record["repetition"] for record in records] == [1, 2, 3]
    with gzip.open(tmp_path / "managed" / "result.jsonl.gz", "rt") as stream:
        assert [json.loads(line)["repetition"] for line in stream] == [1, 2, 3]
