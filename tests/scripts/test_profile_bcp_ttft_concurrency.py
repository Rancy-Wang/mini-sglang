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


def test_short_validation_selects_real_tr8_tr9_tr10_only():
    cases = [dict(turns=[dict(turn=i, tool_responses=i) for i in range(12)]) for _ in range(8)]
    assert profile.short_validation_turns(cases) == [8, 9, 10]
    cases[-1]["turns"][9]["tool_responses"] = 11
    with pytest.raises(ValueError, match="exactly 9"):
        profile.short_validation_turns(cases)


def test_short_performance_keeps_first_decode_wait_separate():
    request = dict(uid=4, case_id="210", turn=9, workload="rolling",
                   ttft_ms=10, tpot_ms=51, e2e_ms=112,
                   response=dict(server_metrics=dict(generated_tokens=3)))
    times = {"rank0": [dict(uid=4, count=1, time_ns=t) for t in (0, 100_000_000, 102_000_000)]}
    rows = profile.short_performance_rows([dict(request, turn=8), request], times)
    assert len(rows) == 1 and rows[0]["group"] == "P1"
    assert rows[0]["tpot_ms"] == 51
    measured = rows[0]["token_intervals"][0]
    assert measured["count_valid"]
    assert measured["first_to_second_ms"] == 100
    assert measured["subsequent_mean_ms"] == 2


def test_committed_comparison_rejects_missing_rank_and_late_token_difference():
    from copy import deepcopy
    request = dict(cell="fixed-c8-rolling", uid=1, mode="fixed", concurrency=8,
                   workload="rolling", case_id="210", turn=9, request_sha256="same",
                   response=dict(server_metrics=dict(generated_tokens=2),
                                 choices=[dict(message=dict(content="same"))], usage=dict(cached_tokens=4)))
    events = [dict(kind="committed_token", cell=request["cell"], uid=1, pid=pid, tokens=[token])
              for pid in (10, 11) for token in (3, 4)]
    assert profile.compare_committed([request], events, [request], events)[0]["passed"]
    assert not profile.compare_committed([request], events[:2], [request], events)[0]["passed"]
    changed = deepcopy(events)
    changed[1]["tokens"] = changed[3]["tokens"] = [9]
    result = profile.compare_committed([request], events, [request], changed)[0]
    assert result["commits_valid"] and result["choices_equal"] and not result["tokens_equal"]
    assert not result["passed"]


def test_batch_signatures_keep_row_order_and_real_shapes_not_uid_numbers():
    requests = [dict(cell="x", uid=10, case_id="A"), dict(cell="x", uid=20, case_id="B")]
    events = [dict(kind="batch", cell="x", turn=9, pid=5, start_ns=1, phase="prefill",
                   uids=[20,10], extend=[3,4], cached=[6,7], graph=False)]
    signature = profile.physical_batch_signatures(requests, events)[("x",9)][0][0]
    assert signature == dict(phase="prefill", graph=False, cases=["B","A"], extend=[3,4], cached=[6,7])


def test_output_comparison_excludes_only_server_random_tool_id():
    from copy import deepcopy
    choices = [dict(message=dict(content="text", reasoning_content="reason",
        tool_calls=[dict(id="call_"+"a"*24, index=0, type="function",
                         function=dict(name="search", arguments='{"q":"x"}'))]))]
    changed = deepcopy(choices)
    changed[0]["message"]["tool_calls"][0]["id"] = "call_"+"b"*24
    assert choices != changed
    assert profile.canonical_choices(choices) == profile.canonical_choices(changed)
    assert choices[0]["message"]["tool_calls"][0]["id"] == "call_"+"a"*24
    request = dict(cell="x", uid=1, mode="fixed", concurrency=8, workload="rolling",
        case_id="210", turn=9, request_sha256="same", response=dict(
            server_metrics=dict(generated_tokens=1), choices=choices, usage={}))
    other = deepcopy(request)
    other["response"]["choices"] = changed
    events = [dict(kind="committed_token", cell="x", uid=1, pid=p, tokens=[3]) for p in (10,11)]
    result = profile.compare_committed([request], events, [other], events)[0]
    assert result["passed"] and result["choices_equal"] and not result["raw_choices_equal"]
    for key in ("content", "reasoning_content"):
        bad = deepcopy(changed)
        bad[0]["message"][key] += "changed"
        assert profile.canonical_choices(choices) != profile.canonical_choices(bad)
    for key in ("name", "arguments"):
        bad = deepcopy(changed)
        bad[0]["message"]["tool_calls"][0]["function"][key] += "changed"
        assert profile.canonical_choices(choices) != profile.canonical_choices(bad)
    changed[0]["message"]["tool_calls"][0]["id"] = "model-provided-id"
    assert profile.canonical_choices(choices) != profile.canonical_choices(changed)


def test_fixed_cohort_admits_one_then_seven_without_changing_requests():
    from types import SimpleNamespace as NS
    requests = [NS(uid=i, prompt_tokens=100+i) for i in range(8)]
    selected, deferred = hooks.fixed_pending_partition(requests[::-1], range(100,108), True)
    assert selected == requests[:1] and deferred == requests[1:]
    selected, deferred = hooks.fixed_pending_partition(deferred, range(100,108), False)
    assert selected == requests[1:] and not deferred
    with pytest.raises(ValueError):
        hooks.fixed_pending_partition(requests, [100,100], True)


def test_fixed_trace_has_explicit_barrier_and_bounded_capture():
    for turn in range(12):
        value = profile.mode_control("fixed-overlap", turn)
        assert value["fixed_split"] and value["barrier"]
        assert value["overlap"] == (turn in (9,10,11))
    assert not profile.mode_control("natural", 9)["fixed_split"]


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


def test_nsys_correlation_is_process_local_and_uses_device_times(tmp_path):
    import json
    import sqlite3
    path = tmp_path/"trace.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE StringIds(id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,
            correlationId INTEGER,nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,
            correlationId INTEGER,demangledName INTEGER);
        """)
        db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1,"cudaLaunchKernel"),(2,"kernel")])
        for pid, uid in [(10,3),(11,7)]:
            tid = (pid << 24) | pid
            label = json.dumps(dict(r3="engine.forward",uids=[uid],host_ns=1))
            db.execute("INSERT INTO NVTX_EVENTS VALUES (1,10,?,?)", (label,tid))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (2,3,?,9,1)",(tid,))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (20,50,?,9,2)",(pid<<24,))
    result = profile.nsys_rows(path)
    assert [(g["pid"],g["uids"]) for g in result["gpu"]] == [(10,[3]),(11,[7])]
    assert all(g["start"] == 20 and g["end"] == 50 and g["api_end"] == 3 for g in result["gpu"])


def test_clock_alignment_retains_bracket_uncertainty():
    clocks = [dict(pid=4, r3_clock=1000, start=100)]
    host = [dict(kind="trace_clock", pid=4, before_ns=1000, after_ns=1004, cell="x",turn=9)]
    value = profile.trace_clock_offset(clocks, host)
    assert value["offset_ns"] == -902 and value["uncertainty_ns"] == 2
    with pytest.raises(ValueError):
        profile.trace_clock_offset([],host)
    clocks.append(dict(pid=5,r3_clock=1000,start=200))
    host.append(dict(kind="trace_clock",pid=5,before_ns=1000,after_ns=1004,cell="x",turn=9))
    with pytest.raises(ValueError):
        profile.trace_clock_offset(clocks,host)


def test_execution_union_clips_and_does_not_double_count_stream_overlap():
    rows = [dict(start=0,end=20),dict(start=10,end=30),dict(start=40,end=60)]
    assert profile.execution_union_ns(rows,5,50) == 35
    assert profile.execution_union_ns(rows,70,80) == 0


def test_blocking_api_attribution_excludes_other_threads_and_ranks():
    op = dict(pid=1,tid=2,r3="compact.op.to",start=10,end=50)
    api = dict(pid=1,tid=2,start=12,end=48,correlationId=7)
    trace = dict(ranges=[op],apis=[api,dict(api,tid=3,start=10,end=50)],gpu=[
        dict(pid=1,deviceId=0,correlationId=7,start=46,end=49,kind="memcpy"),
        dict(pid=2,deviceId=1,correlationId=7,start=0,end=100,kind="memcpy")])
    result = profile.blocked_copy_evidence(trace)[0]
    assert result["api"] == api and len(result["correlated_gpu"]) == 1
    assert result["execution"][0]["active_union_ns"] == 2


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


@pytest.mark.parametrize("modes", [("baseline", "detail"), ("natural", "overlap")])
def test_output_gate_excludes_unused_overlap_but_rejects_missing_generated_tokens(modes):
    rows, events = [], []
    for mode, extra in zip(modes, (50, 60)):
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
    assert comparison[0]["reference_mode"] == modes[0]
    assert comparison[0]["measured_mode"] == modes[1]
    assert comparison[0]["detail_tokens"] == [42]
    rows[-1]["response"]["server_metrics"]["generated_tokens"] = 3
    _, comparison = profile.correlate(rows, events)
    assert not comparison[0]["equal"]
