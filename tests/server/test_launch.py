import queue

import pytest

from minisgl.server.launch import _stop_worker_processes, _wait_for_worker_acks


class FakeProcess:
    def __init__(self, name, pid, exitcode=None, *, survive_terminate=False):
        self.name = name
        self.pid = pid
        self.exitcode = exitcode
        self.survive_terminate = survive_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = []

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.terminate_calls += 1
        if not self.survive_terminate:
            self.exitcode = -15

    def kill(self):
        self.kill_calls += 1
        self.exitcode = -9

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class EmptyQueue:
    def get(self, timeout):
        raise queue.Empty


class MessageQueue:
    def __init__(self, messages):
        self.messages = iter(messages)

    def get(self, timeout):
        return next(self.messages)


def test_wait_for_worker_acks_reports_early_worker_exit():
    failed = FakeProcess("minisgl-TP2-scheduler", 1234, exitcode=1)

    with pytest.raises(
        RuntimeError,
        match=r"minisgl-TP2-scheduler \(pid=1234, exitcode=1\)",
    ):
        _wait_for_worker_acks([failed], EmptyQueue(), 1, poll_interval_s=0)


def test_wait_for_worker_acks_accepts_expected_messages():
    running = FakeProcess("minisgl-TP0-scheduler", 1234)

    _wait_for_worker_acks(
        [running],
        MessageQueue(["scheduler ready", "tokenizer ready"]),
        2,
        poll_interval_s=0,
    )


def test_stop_worker_processes_terminates_then_kills_survivors():
    cooperative = FakeProcess("cooperative", 1)
    stubborn = FakeProcess("stubborn", 2, survive_terminate=True)
    already_exited = FakeProcess("exited", 3, exitcode=0)

    _stop_worker_processes(
        [cooperative, stubborn, already_exited],
        terminate_timeout_s=0,
    )

    assert cooperative.terminate_calls == 1
    assert cooperative.kill_calls == 0
    assert stubborn.terminate_calls == 1
    assert stubborn.kill_calls == 1
    assert already_exited.terminate_calls == 0
    assert already_exited.kill_calls == 0
