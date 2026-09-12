from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Sequence

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


logger = init_logger(__name__, "initializer")


@contextmanager
def _backend_failure_interrupt():
    """Unwind the API even when a dead backend leaves an HTTP request pending.

    Uvicorn owns SIGINT/SIGTERM and may wait indefinitely for open transports
    even after its second SIGINT. A separate, scoped signal interrupts its event
    loop instead: asyncio cancels requests, then run_api_server's finally closes
    frontend resources and stops this instance's remaining worker processes.
    This is only for actual worker exit, never a request timeout or normal stop.
    """
    previous = signal.getsignal(signal.SIGUSR1)

    def interrupt(_signum, _frame):
        raise SystemExit(1)

    signal.signal(signal.SIGUSR1, interrupt)
    try:
        yield lambda: os.kill(os.getpid(), signal.SIGUSR1)
    finally:
        signal.signal(signal.SIGUSR1, previous)


def _check_public_port(host: str, port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as exc:
        raise RuntimeError(
            f"Public server port {host}:{port} is unavailable: {exc}"
        ) from exc


def _find_available_internal_port(*, exclude: set[int] | None = None) -> int:
    excluded = exclude or set()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in excluded:
            return port


def _exited_workers(processes: Sequence[mp.Process]) -> list[mp.Process]:
    return [process for process in processes if process.exitcode is not None]


def _start_worker_watchdog(
    processes: Sequence[mp.Process], on_failure: Callable[[], None], *, poll_interval_s: float = 0.2
) -> Callable[[], None]:
    """Detect runtime worker loss; cancellation must precede normal worker shutdown."""
    cancelled = threading.Event()

    def watch() -> None:
        while not cancelled.wait(poll_interval_s):
            if _exited_workers(processes):
                on_failure()
                return

    thread = threading.Thread(target=watch, name="minisgl-worker-watchdog", daemon=True)
    thread.start()

    def cancel() -> None:
        cancelled.set()
        if threading.current_thread() is not thread:
            thread.join(timeout=1.0)

    return cancel


def _wait_for_worker_acks(
    processes: Sequence[mp.Process],
    ack_queue: mp.Queue[str],
    expected_acks: int,
    *,
    poll_interval_s: float = 0.1,
) -> None:
    received = 0
    while received < expected_acks:
        exited = _exited_workers(processes)
        if exited:
            details = ", ".join(
                f"{process.name} (pid={process.pid}, exitcode={process.exitcode})"
                for process in exited
            )
            raise RuntimeError(f"Backend worker exited before server startup completed: {details}")

        try:
            message = ack_queue.get(timeout=poll_interval_s)
        except queue.Empty:
            continue
        logger.info(message)
        received += 1

    exited = _exited_workers(processes)
    if exited:
        details = ", ".join(
            f"{process.name} (pid={process.pid}, exitcode={process.exitcode})"
            for process in exited
        )
        raise RuntimeError(f"Backend worker exited during server startup: {details}")


def _stop_worker_processes(
    processes: Sequence[mp.Process], *, terminate_timeout_s: float = 5.0
) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.monotonic() + terminate_timeout_s
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    survivors = [process for process in processes if process.is_alive()]
    for process in survivors:
        process.kill()
    for process in survivors:
        process.join()


def _worker_cleanup(
    processes: Sequence[mp.Process], ack_queue: mp.Queue[str]
) -> Callable[[], None]:
    stopped = False
    queue_ref: mp.Queue[str] | None = ack_queue

    def stop() -> None:
        nonlocal queue_ref, stopped
        if stopped:
            return
        stopped = True
        _stop_worker_processes(processes)
        assert queue_ref is not None
        queue_ref.close()
        queue_ref.join_thread()
        # Do not keep the Queue's named semaphore alive through interpreter
        # shutdown via this returned closure.
        queue_ref = None

    return stop


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def stop_scheduler(*_) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_scheduler)
        try:
            if args.tp_info.is_primary():
                ack_queue.put("Scheduler is ready")

            if args.silent_output:
                logging.disable(logging.INFO)

            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    _check_public_port(server_args.server_host, server_args.server_port)
    server_args = replace(
        server_args,
        distributed_port=_find_available_internal_port(
            exclude={server_args.server_port}
        ),
    )

    def start_subprocess() -> Callable[[], None]:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()
        processes: list[mp.Process] = []
        stop = _worker_cleanup(processes, ack_queue)

        try:
            for i in range(world_size):
                new_args = replace(
                    server_args,
                    tp_info=DistributedInfo(i, world_size),
                )
                process = mp.Process(
                    target=_run_scheduler,
                    args=(new_args, ack_queue),
                    daemon=False,
                    name=f"minisgl-TP{i}-scheduler",
                )
                process.start()
                processes.append(process)

            num_tokenizers = server_args.num_tokenizer
            # DeTokenizer, only 1
            process = mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_detokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "radix_drop_key_mode": server_args.radix_drop_key_mode,
                    "reposition_execution_mode": server_args.reposition_execution_mode,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": num_tokenizers,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="minisgl-detokenizer-0",
            )
            process.start()
            processes.append(process)
            for i in range(num_tokenizers):
                process = mp.Process(
                    target=tokenize_worker,
                    kwargs={
                        "tokenizer_path": server_args.model_path,
                        "addr": server_args.zmq_tokenizer_addr,
                        "backend_addr": server_args.zmq_backend_addr,
                        "frontend_addr": server_args.zmq_frontend_addr,
                        "local_bs": 1,
                        "radix_drop_key_mode": server_args.radix_drop_key_mode,
                        "reposition_execution_mode": server_args.reposition_execution_mode,
                        "create": server_args.tokenizer_create_addr,
                        "tokenizer_id": i,
                        "ack_queue": ack_queue,
                    },
                    daemon=False,
                    name=f"minisgl-tokenizer-{i}",
                )
                process.start()
                processes.append(process)

            # Only the primary scheduler sends an acknowledgment after all TP ranks sync.
            _wait_for_worker_acks(processes, ack_queue, num_tokenizers + 2)
            def failed() -> None:
                logger.error("Backend worker exited; stopping this server instance.")
                interrupt_backend_failure()

            cancel_watchdog = _start_worker_watchdog(processes, failed)

            def stop_monitored() -> None:
                cancel_watchdog()
                stop()

            return stop_monitored
        except BaseException:
            stop()
            raise

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_startup(signum, _) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupt_startup)
    try:
        with _backend_failure_interrupt() as interrupt_backend_failure:
            run_api_server(server_args, start_subprocess, run_shell=run_shell)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    launch_server()
