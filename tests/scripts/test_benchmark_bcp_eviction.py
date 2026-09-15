import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_bcp_eviction.py"
    spec = importlib.util.spec_from_file_location("bcp_eviction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_requires_distinct_cases_and_strict_full_context_threshold():
    m = load()
    rows = [
        {"case_id": "1", "full_tokens": 131072},
        {"case_id": "2", "full_tokens": 131073, "trial": "first"},
        {"case_id": "2", "full_tokens": 150000, "trial": "second"},
        {"case_id": "3", "full_tokens": 140000},
    ]
    assert m.select_cases(rows, 2) == [rows[1], rows[3]]
    with pytest.raises(ValueError, match="distinct"):
        m.select_cases(rows, 3)


def test_rolling_counts_responses_with_independent_message_ids():
    m = load()
    for count in (12, 13, 22):
        messages = [{"role": "system"}, {"role": "user"}]
        for n in range(count):
            messages.extend([{"role": "assistant"}, {"role": "tool"}])
            if n % 3 == 0:
                messages.append({"role": "user"})
        schedule = m.rolling_interface(messages)
        tools = [i for i, x in enumerate(messages) if x["role"] == "tool"]
        assert schedule["reposition"] == tools[12:]
        assert schedule["drop_message"] == {
            str(tools[i]): [tools[i - 12]] for i in range(12, count)
        }
        for event, removed in schedule["drop_message"].items():
            assert removed[0] < int(event)
            assert messages[removed[0]]["role"] == "tool"


def test_pressure_queries_keep_canonical_trigger_and_seed_separate():
    m = load()
    messages = [{"role": "user"}]
    for n in range(16):
        messages.extend([{"role": "assistant"}, {"role": "tool"},
                         {"role": "assistant"}, {"role": "assistant"}])
    ends = m.pressure_turn_ends(messages)
    for tr, end in zip(range(12, 16), ends, strict=True):
        prefix = messages[:end]
        assert sum(x["role"] == "tool" for x in prefix) == tr
        assert prefix[-1]["role"] == "tool"
        assert len(m.rolling_interface(prefix)["reposition"]) == tr - 12
    assert len(ends[1:]) * 8 == 24
    with pytest.raises(ValueError):
        m.pressure_turn_ends(messages[:ends[0]])


def test_common_turns_preserve_prefix_and_reserve_full_output_budget():
    m = load()
    turns = [{"full_tokens": size} for size in (1000, 126976, 126977, 100)]
    assert m.common_turns(turns, 4096) == turns[:2]
    # Later turns cannot be resurrected by a shorter template or truncated transcript.
    assert m.common_turns(turns, 131000) == []


def test_metrics_include_real_generated_tokens_and_single_token_tpot_is_missing():
    m = load()
    metrics = {
        "request_received_ns": 1_000_000_000,
        "first_token_generated_ns": 3_000_000_000,
        "request_finished_ns": 5_000_000_000,
        "generated_tokens": 5,
    }
    assert m.metric_values(metrics) == {"ttft_s": 2, "tpot_s": 0.5}
    assert m.metric_values(dict(metrics, generated_tokens=1))["tpot_s"] is None
    with pytest.raises(ValueError):
        m.metric_values(dict(metrics, generated_tokens=0))
    with pytest.raises(ValueError):
        m.metric_values(dict(metrics, request_finished_ns=1))


def test_throughput_retains_failed_and_unattempted_turns_in_denominator_and_status():
    m = load()
    records = [
        {
            "metrics": {"generated_tokens": 20, "prompt_tokens": 150000},
            "ttft_s": 2,
            "tpot_s": 0.1,
            "finish_reason": "length",
        },
        {"error": "worker failed"},
    ]
    result = m.summarize(records, 10, 5)
    assert result["output_tokens_per_s"] == 2
    assert result["turns_per_s"] == 0.1
    assert result["uncompleted_turns"] == 4
    assert result["failed_turns"] == 1
    assert result["over_128k_turns"] == 1


@pytest.mark.parametrize("stress_count", [16, 32])
def test_complete_matrix_has_paired_cells_and_requested_concurrency_counts(stress_count):
    m = load()
    cells = m.matrix_cells(stress_count)
    names = [m.cell_name(c) for c in cells]
    assert len(names) == len(set(names)) == 50
    stress = [c for c in cells if c["suite"] == f"stress{stress_count}"]
    assert len(stress) == 2
    assert all(c["count"] == stress_count and c["concurrency"] == 8 for c in stress)
    assert all(c["tp"] == 2 and c["phase"] == "full" for c in stress)
    assert sum(c["phase"] == "common" for c in cells) == 32
    assert sum(c["phase"] == "full" and c["suite"] == "scaling" for c in cells) == 16
    for cell in cells:
        if cell["suite"] == "scaling":
            assert cell["count"] == cell["concurrency"]
        if cell["eviction"] == "drop-aware":
            assert m.cell_name(cell).replace("drop-aware", "ordinary") in names
        if cell["workload"] == "no_drop":
            assert cell["phase"] == "common"


def test_experiment_outputs_cannot_land_in_repository():
    m = load()
    with pytest.raises(ValueError):
        m.external(m.REPO / "results")


@pytest.mark.parametrize("fixed_output", [False, True])
def test_replay_uses_recorded_prefixes_and_waits_for_each_turn(tmp_path, fixed_output):
    m = load()
    trajectory = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "recorded answer"},
        {"role": "user", "content": "next question"},
    ]
    calls = []

    class Client:
        async def post(self, url, json):
            calls.append(json)
            index = len(calls)
            metrics = {
                "request_received_ns": 0,
                "first_token_generated_ns": 1,
                "request_finished_ns": 2,
                "generated_tokens": 2,
                "prompt_tokens": index,
                "drop_skipped_tokens": 0,
            }
            return SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {
                    "server_metrics": metrics,
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "new generated answer"},
                        }
                    ],
                },
            )

    turns = [
        {"turn": i, "end": end, "full_tokens": i + 1, "messages_sha256": m.digest(trajectory[:end])}
        for i, end in enumerate((1, 3))
    ]
    cell = {
        "tp": 2,
        "concurrency": 1,
        "count": 1,
        "workload": "no_drop",
        "phase": "common",
        "eviction": "ordinary",
        "suite": "scaling",
    }
    result = asyncio.run(
        m.replay(
            SimpleNamespace(model="test"),
            cell,
            {"tools": [], "max_tokens": 2, "fixed_output": fixed_output},
            [{"case_id": "1", "trajectory": trajectory, "selected_turns": turns}],
            tmp_path,
            Client(),
            "http://test",
        )
    )
    assert result["completed_turns"] == 2
    assert calls[0]["messages"] == trajectory[:1]
    assert calls[1]["messages"] == trajectory
    assert all("drop_message" not in payload for payload in calls)
    assert all(payload.get("ignore_eos", False) == fixed_output for payload in calls)
