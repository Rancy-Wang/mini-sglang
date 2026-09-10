import threading
from types import SimpleNamespace

from minisgl.server.launch import _start_worker_watchdog


def test_runtime_worker_loss_notifies_once():
    worker = SimpleNamespace(exitcode=None)
    failure = threading.Event()
    calls = []

    def failed():
        calls.append(True)
        failure.set()

    cancel = _start_worker_watchdog([worker], failed, poll_interval_s=0.01)
    try:
        worker.exitcode = 1
        assert failure.wait(1)
    finally:
        cancel()
    assert calls == [True]


def test_normal_shutdown_cancels_watchdog_before_workers_stop():
    worker = SimpleNamespace(exitcode=None)
    calls = []
    cancel = _start_worker_watchdog([worker], lambda: calls.append(True))
    cancel()
    worker.exitcode = 0
    assert calls == []
