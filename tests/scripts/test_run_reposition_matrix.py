from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path

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
