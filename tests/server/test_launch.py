import gc
import queue
import socket
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisgl.server.launch import (
    _check_public_port,
    _find_available_internal_port,
    _stop_worker_processes,
    _wait_for_worker_acks,
    _worker_cleanup,
)


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


class ClosableQueue:
    def __init__(self):
        self.close_calls = 0
        self.join_calls = 0

    def close(self):
        self.close_calls += 1

    def join_thread(self):
        self.join_calls += 1


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


def test_internal_port_does_not_depend_on_public_port_plus_one():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        blocked = int(occupied.getsockname()[1])
        selected = _find_available_internal_port(exclude={blocked})

    assert selected != blocked


def test_public_port_failure_is_reported_before_workers_start():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = int(occupied.getsockname()[1])
        with pytest.raises(RuntimeError, match=f"127.0.0.1:{port} is unavailable"):
            _check_public_port("127.0.0.1", port)


def test_public_port_check_uses_server_restart_socket_semantics():
    with patch("minisgl.server.launch.socket.socket") as socket_factory:
        sock = socket_factory.return_value.__enter__.return_value

        _check_public_port("127.0.0.1", 8000)

    sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind.assert_called_once_with(("127.0.0.1", 8000))


def test_worker_cleanup_is_idempotent_and_closes_ack_queue():
    process = FakeProcess("worker", 1)
    ack_queue = ClosableQueue()
    queue_ref = weakref.ref(ack_queue)
    stop = _worker_cleanup([process], ack_queue)

    stop()
    stop()

    assert process.terminate_calls == 1
    assert ack_queue.close_calls == 1
    assert ack_queue.join_calls == 1
    del ack_queue
    gc.collect()
    assert queue_ref() is None


def test_api_server_return_stops_frontend_and_backend(monkeypatch):
    from minisgl.server import api_server

    queues = []

    class FakeZmqQueue:
        def __init__(self, *args, **kwargs):
            self.stop_calls = 0
            queues.append(self)

        def stop(self):
            self.stop_calls += 1

    backend_stop_calls = []
    monkeypatch.setattr(api_server, "ZmqAsyncPullQueue", FakeZmqQueue)
    monkeypatch.setattr(api_server, "ZmqAsyncPushQueue", FakeZmqQueue)
    monkeypatch.setattr(api_server.uvicorn, "run", lambda *args, **kwargs: None)
    config = SimpleNamespace(
        server_host="127.0.0.1",
        server_port=8000,
        use_dummy_weight=False,
        zmq_frontend_addr="ipc:///tmp/test_frontend",
        zmq_tokenizer_addr="ipc:///tmp/test_tokenizer",
        frontend_create_tokenizer_link=True,
    )

    api_server.run_api_server(
        config,
        lambda: lambda: backend_stop_calls.append(True),
        run_shell=False,
    )

    assert backend_stop_calls == [True]
    assert [item.stop_calls for item in queues] == [1, 1]
    assert api_server._GLOBAL_STATE is None
