import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/validate_drop_aware_eviction.py"
    spec = importlib.util.spec_from_file_location("drop_eviction_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rolling_schedule_counts_tool_responses_per_request():
    module = load()
    for case, count in [(0, 12), (1, 13), (2, 18)]:
        messages = module.make_messages(case, count, 1)
        schedule = module.rolling_interface(messages)
        tools = [i for i, msg in enumerate(messages) if msg["role"] == "tool"]
        assert schedule["reposition"] == tools[12:]
        assert schedule["drop_message"] == {
            str(tools[n]): [tools[n - 12]] for n in range(12, count)
        }


def test_workload_histories_are_distinct_and_tool_calls_pair():
    module = load()
    histories = [module.make_messages(case, 34 + case % 6, 1) for case in range(32)]
    assert len({str(messages) for messages in histories}) == 32
    for messages in histories:
        for offset in range(2, len(messages), 2):
            assert messages[offset]["tool_calls"][0]["id"] == messages[offset + 1]["tool_call_id"]


def test_control_workloads_preserve_drop_schedule_without_reposition():
    module = load()
    messages = module.make_messages(0, 15, 1)
    assert module.workload_interface(messages, 12, "no-drop") == {}
    dropped = module.workload_interface(messages, 12, "rolling-drop")
    assert dropped == {"drop_message": module.rolling_interface(messages)["drop_message"]}
    assert module.workload_interface(messages, 12, "rolling-reposition") == module.rolling_interface(messages)


def test_capacity_profile_varies_active_history_without_changing_rolling_timing():
    module = load()
    for case in range(32):
        rounds = module.workload_rounds(case, 8, [96, 96, 96, 46])
        assert rounds == (96 if case < 24 else 46) + case % 6
        messages = module.make_messages(case, rounds, 1)
        schedule = module.rolling_interface(messages)
        assert len(schedule["reposition"]) == rounds - 12
        assert schedule["reposition"][0] == 27
        assert module.workload_rounds(case, 8) == 34 + case % 6
