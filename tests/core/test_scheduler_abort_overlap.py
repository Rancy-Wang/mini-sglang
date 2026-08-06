from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from minisgl.message import AbortBackendMsg
from minisgl.scheduler.scheduler import Scheduler


def test_abort_tombstone_discards_stale_prefill_result(monkeypatch):
    scheduler = object.__new__(Scheduler)
    request = object()
    scheduler.prefill_manager = MagicMock()
    scheduler.prefill_manager.abort_req.return_value = None
    scheduler.decode_manager = MagicMock()
    scheduler.decode_manager.abort_req.return_value = request
    scheduler.cache_manager = MagicMock()
    scheduler.cache_manager.lazy_free_region.return_value = nullcontext()
    scheduler._free_req_resources = MagicMock()
    scheduler.send_result = MagicMock()
    scheduler.finished_reqs = set()
    monkeypatch.setattr("minisgl.scheduler.scheduler.logger.debug_rank0", MagicMock())

    scheduler._process_one_msg(AbortBackendMsg(uid=17))

    scheduler._free_req_resources.assert_called_once_with(request)
    assert scheduler.finished_reqs == {request}

    copy_done = MagicMock()
    batch = SimpleNamespace(reqs=[request], is_prefill=True)
    last_data = (SimpleNamespace(batch=batch), (None, [], copy_done))
    scheduler._process_last_data(last_data)

    copy_done.synchronize.assert_called_once_with()
    scheduler.cache_manager.cache_req.assert_not_called()
    scheduler._free_req_resources.assert_called_once_with(request)
    assert scheduler.finished_reqs == {request}
