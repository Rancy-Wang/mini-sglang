from unittest.mock import MagicMock, patch

import torch
from minisgl.engine.graph import GraphRunner, _determine_cuda_graph_bs
from minisgl.server.args import parse_args


class _Batch:
    def __init__(self, size: int, *, decode: bool = True):
        self.reqs = [object()] * size
        self.padded_reqs = list(self.reqs)
        self.is_decode = decode

    @property
    def size(self):
        return len(self.reqs)

    @property
    def padded_size(self):
        return len(self.padded_reqs)


def _graph_runner(targets=(1, 2, 4)):
    backend = MagicMock()
    model = MagicMock()
    with patch.object(GraphRunner, "_capture_graphs") as capture_graphs:
        runner = GraphRunner(
            stream=object(),
            device=torch.device("cpu"),
            model=model,
            attn_backend=backend,
            cuda_graph_bs=list(targets),
            cuda_graph_max_bs=None,
            free_memory=1,
            max_seq_len=32,
            vocab_size=16,
            dummy_req=object(),
        )
    capture_graphs.assert_called_once_with(list(targets), model, None)
    return runner, backend


def test_default_auto_enables_and_zero_disables_cuda_graph():
    args, _ = parse_args(["--model", "unused", "--dtype", "float32"])
    default_sizes = _determine_cuda_graph_bs(None, None, 1)

    assert args.cuda_graph_max_bs is None
    assert default_sizes[:3] == [1, 2, 4]
    assert default_sizes[-1] == 160
    assert _determine_cuda_graph_bs(None, None, 81 << 30)[-1] == 256
    assert _determine_cuda_graph_bs(None, 1, 1) == [1]
    assert _determine_cuda_graph_bs(None, 0, 1) == []
    assert _determine_cuda_graph_bs([0], None, 1) == []
    assert _determine_cuda_graph_bs([4, 1, 4], None, 1) == [1, 4]


def test_constructor_initializes_fixed_whole_model_capture():
    runner, backend = _graph_runner()

    backend.init_capture_graph.assert_called_once_with(max_seq_len=32, bs_list=[1, 2, 4])
    assert runner.graph_map == {}
    assert runner.buffer.input_ids.shape == (4,)


def test_padding_and_eligibility_use_only_successfully_captured_sizes():
    runner, _ = _graph_runner()
    runner.graph_map = {2: object(), 4: object()}
    batch = _Batch(1)

    runner.pad_batch(batch)

    assert batch.padded_size == 2
    assert runner.can_use_cuda_graph(batch)

    multi_request_batch = _Batch(3)
    runner.pad_batch(multi_request_batch)
    assert multi_request_batch.padded_size == 4
    assert runner.can_use_cuda_graph(multi_request_batch)

    runner.graph_map.clear()
    batch = _Batch(1)
    runner.pad_batch(batch)
    assert batch.padded_size == 1
    assert not runner.can_use_cuda_graph(batch)
    prefill = _Batch(1, decode=False)
    runner.graph_map = {2: object()}
    runner.pad_batch(prefill)
    assert prefill.padded_size == 1
    assert not runner.can_use_cuda_graph(prefill)


def test_capture_failure_falls_back_for_only_that_shape(monkeypatch):
    runner, _ = _graph_runner((1, 2))
    model = MagicMock()

    def capture_one(bs, model):
        if bs == 2:
            raise RuntimeError("boom")
        return object()

    runner._capture_one = MagicMock(side_effect=capture_one)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr("minisgl.engine.graph.logger.info_rank0", MagicMock())
    monkeypatch.setattr("minisgl.engine.graph.logger.warning_rank0", MagicMock())
    monkeypatch.setattr("minisgl.engine.graph.logger.exception", MagicMock())

    runner._capture_graphs([1, 2], model, None)

    assert [entry.args for entry in runner._capture_one.call_args_list] == [(2, model), (1, model)]
    assert runner.graph_bs_list == [1]


def test_tp_failure_keeps_shape_out_of_graph_map(monkeypatch):
    runner, _ = _graph_runner((2,))
    model = MagicMock()
    graph = object()
    runner._capture_one = MagicMock(return_value=graph)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def mark_remote_failure(success, **kwargs):
        success.zero_()

    all_reduce = MagicMock(side_effect=mark_remote_failure)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr("minisgl.engine.graph.logger.info_rank0", MagicMock())
    monkeypatch.setattr("minisgl.engine.graph.logger.warning_rank0", MagicMock())

    tp_cpu_group = object()
    runner._capture_graphs([2], model, tp_cpu_group)

    all_reduce.assert_called_once()
    assert all_reduce.call_args.kwargs["group"] is tp_cpu_group
    assert runner.graph_map == {}


def test_disable_cuda_graph_cli_alias():
    args, _ = parse_args(["--model", "unused", "--dtype", "float32", "--disable-cuda-graph"])

    assert args.cuda_graph_max_bs == 0
    shell_args, _ = parse_args(
        ["--model", "unused", "--dtype", "float32", "--disable-cuda-graph"],
        run_shell=True,
    )
    assert shell_args.cuda_graph_max_bs == 0
