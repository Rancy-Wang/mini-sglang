import asyncio
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "rolling_replay", Path(__file__).with_name("run_rolling_tool_drop.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_individual_tool_boundary_and_immutable_history():
    # Two tool replies per assistant: count replies, not assistant/tool groups.
    messages = [{"role": "user", "content": "question"}]
    for i in range(7):
        messages.extend(
            [
                {"role": "assistant", "reasoning_content": "retain", "tool_calls": []},
                {"role": "tool", "content": f"first {i}"},
                {"role": "tool", "content": f"second {i}"},
            ]
        )
    turn = {
        "request": {
            "messages": messages,
            "max_completion_tokens": 16,
            "extra_body": {"reposition": [1], "drop_rule": {"old": True}},
            "drop_message": {"9": [0]},
        }
    }
    original = copy.deepcopy(turn)
    result = runner.make_request({}, turn, "rolling_drop", "model", 16384)
    assert result["drop_message"] == {"20": [2], "21": [3]}
    assert result["messages"] == messages
    assert turn == original
    assert result["max_completion_tokens"] == result["max_tokens"] == 16384
    assert not {"reposition", "drop_rule", "extra_body"} & result.keys()
    baseline = runner.make_request({}, turn, "no_drop", "model", 16384)
    assert not {"drop_message", "drop_rule", "reposition"} & baseline.keys()
    turn["request"]["messages"] = messages[:19]
    assert "drop_message" not in runner.make_request({}, turn, "rolling_drop", "m", 100)


def test_reconstruct_snapshot_without_inventing_retries(tmp_path):
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "reasoning_content": "reason", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "document", "tool_call_id": "x"},
        {"role": "assistant", "content": "answer"},
    ]
    (tmp_path / "trajectories.jsonl").write_text(
        json.dumps({"case_id": "231", "trial": 0, "trajectory": messages}) + "\n"
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "benchmark_config": {"generation": {"temperature": 1, "enable_thinking": None}}
                }
            }
        )
    )
    tools = tmp_path / "tools.json"
    tools.write_text('[{"type":"function","function":{"name":"search"}}]')
    bundle = runner.load_inputs(tmp_path, tools)
    assert bundle["provenance"] == "trajectory_reconstructed"
    case = bundle["cases"][0]
    assert len(case["turns"]) == 2
    assert all(turn["physical"] is None and turn["retry"] is None for turn in case["turns"])
    assert (
        runner.make_request(case, case["turns"][0], "no_drop", "m", 50)["messages"] == messages[:1]
    )
    body = runner.make_request(case, case["turns"][1], "no_drop", "m", 50)
    assert body["messages"] == messages[:3]
    assert "enable_thinking" not in body
    (tmp_path / "rollouts.jsonl").write_text(
        json.dumps({"case_id": "231", "trial": 0, "metadata": {"model_calls": [{}]}})
    )
    with pytest.raises(ValueError, match="count mismatch"):
        runner.load_inputs(tmp_path, tools)


def test_raw_gzip_preserves_physical_retries(tmp_path):
    for physical, retry in [(46, 1), (45, 0)]:
        path = (
            tmp_path
            / f"episodes/case_231/trial_000/attempts/attempt_001/backend_calls/{physical:06}_agent_043_retry_{retry:02}/request.json.gz"
        )
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as stream:
            json.dump({"request": {"messages": [{"role": "user", "content": "hello"}]}}, stream)
    bundle = runner.load_inputs(tmp_path)
    assert bundle["provenance"] == "raw_capture"
    assert [turn["physical"] for turn in bundle["cases"][0]["turns"]] == [45, 46]
    assert [turn["retry"] for turn in bundle["cases"][0]["turns"]] == [0, 1]


def sse(chunks, done=True):
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n" if done else ""
    )


@pytest.mark.parametrize(
    "delta",
    [
        {"content": "answer"},
        {"reasoning_content": "reason"},
        {"tool_calls": [{"function": {"arguments": "{}"}}]},
    ],
)
def test_sse_ttft_ignores_empty_role_and_uses_usage(delta, monkeypatch):
    times = iter([100.0, 102.0, 108.0])
    # Avoid modifying asyncio's global time.perf_counter.
    monkeypatch.setattr(runner, "time", type("Clock", (), {"perf_counter": lambda: next(times)}))
    data = sse(
        [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": delta}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"usage": {"completion_tokens": 4, "prompt_tokens": 12}},
        ]
    )

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text=data))
        ) as client:
            return await runner.send_request(client, "http://test/v1/chat/completions", {})

    row = asyncio.run(check())
    assert row["ok"]
    assert row["client"] == {
        "ttft_s": 2,
        "e2e_s": 8,
        "tpot_s": 2,
        "decode_s": 6,
        "decode_intervals": 3,
    }
    assert len(row["deltas"]) == 1


@pytest.mark.parametrize(
    "status,data",
    [(400, "context limit"), (200, sse([], done=False)), (200, sse([{"error": "failure"}]))],
)
def test_failures_are_retained(status, data):
    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(status, text=data))
        ) as client:
            return await runner.send_request(client, "http://test", {})

    row = asyncio.run(check())
    assert not row["ok"] and row["error"]
    assert row["client"]["e2e_s"] >= 0


def test_server_clock_and_one_token():
    assert runner.timings(0, 1, 4, 1)["tpot_s"] is None
    assert runner.timings(0, None, 4, 10)["tpot_s"] is None
    measured = runner.server_timings(
        {
            "request_received_ns": 10**18,
            "first_token_generated_ns": 10**18 + 2 * 10**9,
            "request_finished_ns": 10**18 + 8 * 10**9,
            "generated_tokens": 4,
        }
    )
    assert measured["ttft_s"] == 2 and measured["tpot_s"] == 2
    assert runner.server_timings({}) is None


def test_aggregate_turns_retries_weighting_and_failures():
    rows = []
    for case, turn, first, end, tokens in [("a", 0, 1, 5, 3), ("a", 0, 2, 8, 4), ("b", 1, 1, 2, 1)]:
        rows.append(
            {
                "case_id": case,
                "trial": 0,
                "logical_turn": turn,
                "ok": True,
                "client": runner.timings(0, first, end, tokens),
                "server": None,
                "usage": {"completion_tokens": tokens},
            }
        )
    rows.append(dict(rows[0], ok=False, client=runner.timings(0, None, 20, None)))
    result = runner.summarize(rows, 10)
    assert result["global_stats"]["errors"] == 1
    assert result["global_stats"]["client"]["ttft_s"] == {
        "count": 3,
        "sum": 4,
        "mean": 4 / 3,
        "case_turn_count": 2,
        "mean_case_turn_sum": 2,
    }
    # A retried case contributes its summed logical-turn cost once to the case mean.
    assert result["per_turn"][0]["client"]["e2e_s"]["mean_case_turn_sum"] == 13
    assert result["global_stats"]["client"]["weighted_tpot_s"] == 2
    assert result["global_stats"]["server"]["ttft_s"]["sum"] is None
    assert result["per_case_turn"][0]["client"]["e2e_s"]["sum"] == 13
    assert result["failed_client_e2e_s_sum"] == 20
    assert result["completion_tokens_per_s"] == 0.8


def test_four_workers_keep_case_turn_order():
    cases = [
        {
            "case_id": str(i),
            "trial": 0,
            "turns": [{"logical_turn": j, "physical": None, "retry": None} for j in range(3)],
        }
        for i in range(8)
    ]
    active = set()
    peak = 0
    results = []

    async def execute(case, turn):
        nonlocal peak
        assert case["case_id"] not in active
        active.add(case["case_id"])
        peak = max(peak, len(active))
        await asyncio.sleep(0)
        active.remove(case["case_id"])
        return {}

    asyncio.run(runner.replay(cases, execute, results.append))
    assert peak == 4
    assert len(results) == 24
    for case in cases:
        assert [row["logical_turn"] for row in results if row["case_id"] == case["case_id"]] == [
            0,
            1,
            2,
        ]
