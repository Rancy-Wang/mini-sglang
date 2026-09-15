import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load("profile_bcp_ttft_concurrency")
hooks = load("bcp_ttft_profile_hooks")


def trajectory():
    result = [{"role":"user", "content":"question"}]
    for i in range(15):
        result.extend([{"role":"assistant", "content":str(i)},
                       {"role":"tool", "content":str(i)}])
    return result


@pytest.mark.parametrize("turns", [0,13,-1,100])
def test_turn_limit(turns):
    with pytest.raises(ValueError):
        profile.bounded_turns(turns)


def test_k8_trigger_and_per_conversation_numbering():
    messages = trajectory()
    assert profile.rolling_interface(messages[:17]) == {}
    assert profile.rolling_interface(messages[:19]) == {"drop_message":{"18":[2]}, "reposition":[18]}
    plan = profile.rolling_interface(messages[:23])
    assert plan["drop_message"] == {"18":[2], "20":[4], "22":[6]}
    assert profile.rolling_interface(messages[:19]) == {"drop_message":{"18":[2]}, "reposition":[18]}


def test_only_one_nested_cohort_and_first_twelve_turns():
    cases = profile.choose_cases([dict(case_id=i, trajectory=trajectory()) for i in range(8)],12)
    assert len(cases) == 8
    for concurrency in (1,2,4,8):
        cohort = cases[:concurrency]
        assert len(cohort) == concurrency
        assert all(len(case["ends"]) == 12 and case["ends"][0] == 1 for case in cohort)


def test_gate_requires_unique_uids_and_full_wave():
    gate = hooks.WaveGate()
    gate.arrive(("x",0),2,10)
    gate.arrive(("x",0),2,10)
    assert not gate.ready()
    gate.arrive(("x",0),2,11)
    assert gate.ready()
    gate.released = True
    gate.arrive(("x",1),2,12)
    assert not gate.ready()
    with pytest.raises(RuntimeError):
        gate.arrive(("x",2),2,13)


def test_metric_definition_is_server_first_sample_not_full_response():
    value = profile.metrics_values(dict(first_token_generated_ns=20_000_000,
                 request_received_ns=10_000_000, request_finished_ns=50_000_000, generated_tokens=4))
    assert value == dict(ttft_ms=10,tpot_ms=10)


def test_artifacts_not_written_into_git():
    with pytest.raises(ValueError):
        profile.external(ROOT/"results")


def test_digest_preserves_message_content():
    assert profile.digest([{"a":1,"b":2}]) == profile.digest([{"b":2,"a":1}])
    assert profile.digest([{"a":1}]) != profile.digest([{"a":2}])


def test_hook_preserves_static_class_and_instance_binding():
    class Sample:
        @staticmethod
        def static(a):
            return a + 1

        @classmethod
        def class_method(cls, a):
            return cls.static(a)

        def instance(self, a):
            return self.static(a)

    for name in ("static", "class_method", "instance"):
        hooks.replace_callable(Sample, name, lambda fn: lambda *a, **kw: fn(*a, **kw))
    assert Sample.static(1) == 2
    assert Sample.class_method(2) == 3
    assert Sample().instance(3) == 4


def test_uid_clock_partition_does_not_sum_tp_ranks():
    row = dict(cell="baseline-c1-no_drop", uid=7, turn=0, case_id="x", mode="baseline",
               concurrency=1, workload="no_drop", ttft_ms=.000010,
               response={"server_metrics":{"request_received_ns":0,"first_token_generated_ns":10}})
    common = dict(cell=row["cell"],turn=0)
    events = [dict(common,kind="tokenizer",uid=7,start_ns=1,end_ns=3)]
    for pid in (100,200):
        events.extend([dict(common,kind="arrival",uid=7,pid=pid,time_ns=4),
                       dict(common,kind="gate_release",pid=pid,end_ns=5),
                       dict(common,kind="batch",pid=pid,uids=[7],phase="prefill",start_ns=6,tokens=[42])])
    parts, _ = profile.correlate([row],events)
    assert len(parts)==1 and parts[0]["partition_valid"]
    assert parts[0]["sum_ms"]==pytest.approx(.000010)
