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


def test_overlap_is_natural_and_bounded():
    for turn in range(12):
        control = dict(turn=turn, **profile.mode_control("overlap", turn))
        assert not control["barrier"] and not control["detail"]
        assert hooks.overlap_enabled(control) == (turn in (9,10,11))
        assert not hooks.overlap_enabled(dict(turn=turn, **profile.mode_control("natural", turn)))
    assert profile.mode_control("detail", 9)["barrier"]


def test_overlap_partition_closes_and_rejects_reversed_clocks():
    parts = profile.overlap_partition(10, 30, 31, 32, 39, 40)
    assert parts == dict(post_engine=20, control_gap=1, collect_before_sync=1,
                         copy_wait=7, record_gap=1)
    assert sum(parts.values()) == 40-10
    with pytest.raises(ValueError):
        profile.overlap_partition(10, 9, 31, 32, 39, 40)


def test_result_identity_uses_previous_batch_not_current_batch():
    from types import SimpleNamespace as NS
    previous = (NS(batch=NS(reqs=[NS(uid=3),NS(uid=5)])), object())
    assert hooks.result_uids(previous) == [3,5]
    assert hooks.result_uids(None) == []


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


def test_ttft_function_window_excludes_decode_and_does_not_prorate_cpu():
    rows = [dict(cell="x", turn=0, uid=1, response={"server_metrics":
             dict(request_received_ns=10, first_token_generated_ns=20)})]
    def event(pid, start, end):
        return dict(kind="function", cell="x", turn=0, pid=pid, name="f", phase="decode",
                    start_ns=start, end_ns=end, wall_ns=end-start, cpu_ns=2,
                    self_ns=3, self_cpu_ns=1)
    totals = profile.ttft_function_totals(rows, [event(1,12,17),event(1,18,23),
                                                event(1,23,30),event(2,12,17)])
    assert len(totals)==2
    first = next(t for t in totals if t["pid"]==1)
    assert first["calls"]==2 and first["clipped_calls"]==1
    assert first["inclusive_overlap_ms"]==pytest.approx(7/1e6)
    assert first["contained_cpu_ms"]==pytest.approx(2/1e6)


def test_gpu_ranges_only_on_existing_long_and_last_turn_prefill():
    assert hooks.gpu_detail_enabled(dict(gpu_detail=True,turn=11),"prefill")
    assert hooks.gpu_detail_enabled(dict(gpu_detail=True,turn=6),"prefill")
    assert not hooks.gpu_detail_enabled(dict(gpu_detail=True,turn=10),"prefill")
    assert not hooks.gpu_detail_enabled(dict(gpu_detail=True,turn=11),"decode")
    assert not hooks.gpu_detail_enabled(dict(detail=True,turn=11),"prefill")


def test_communication_timeout_override_preserves_original_config():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Config:
        distributed_timeout: float = 60
        model_path: str = "unchanged"

    original = Config()
    effective = hooks.communication_config(original, 600)
    assert original.distributed_timeout == 60
    assert effective.distributed_timeout == 600
    assert effective.model_path == original.model_path
    with pytest.raises(ValueError):
        hooks.communication_config(original, 0)


def test_physical_batch_audit_rejects_split_prefill_and_missing_rank():
    rows = [dict(cell="x", turn=0, uid=u, concurrency=2) for u in (7,8)]
    def batch(pid, uids):
        return dict(kind="batch", cell="x", turn=0, pid=pid, phase="prefill",
                    size=len(uids), uids=uids)
    complete = [batch(1,[7,8]),batch(2,[8,7])]
    assert profile.audit_prefill_batches(rows, complete)[0]["passed"]
    assert not profile.audit_prefill_batches(rows, complete[:1])[0]["passed"]
    assert not profile.audit_prefill_batches(rows, complete+[batch(1,[8])])[0]["passed"]


def test_output_gate_excludes_unused_overlap_but_rejects_missing_generated_tokens():
    rows, events = [], []
    for mode, extra in (("baseline", 50), ("detail", 60)):
        rows.append(dict(cell=mode, uid=7, turn=0, case_id="x", mode=mode,
                         concurrency=1, workload="no_drop", ttft_ms=10/1e6,
                         response={"server_metrics":dict(request_received_ns=0,
                                   first_token_generated_ns=10, generated_tokens=1)}))
        events.extend([dict(kind="tokenizer",cell=mode,uid=7,start_ns=1,end_ns=3),
                       dict(kind="arrival",cell=mode,uid=7,pid=1,time_ns=4),
                       dict(kind="batch",cell=mode,pid=1,uids=[7],phase="prefill",
                            start_ns=6,tokens=[42]),
                       dict(kind="batch",cell=mode,pid=1,uids=[7],phase="decode",
                            start_ns=9,tokens=[extra])])
    _, comparison = profile.correlate(rows, events)
    assert comparison[0]["equal"]
    assert comparison[0]["detail_tokens"] == [42]
    rows[-1]["response"]["server_metrics"]["generated_tokens"] = 3
    _, comparison = profile.correlate(rows, events)
    assert not comparison[0]["equal"]
