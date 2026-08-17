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
    with patch.object(GraphRunner, "_capture_graphs") as capture_graphs:
        runner = GraphRunner(
            stream=object(),
            device=torch.device("cpu"),
            model=MagicMock(),
            attn_backend=backend,
            cuda_graph_bs=list(targets),
            cuda_graph_max_bs=None,
            free_memory=1,
            max_seq_len=32,
            vocab_size=16,
            dummy_req=object(),
        )
    capture_graphs.assert_called_once_with()
    return runner, backend


def test_gpt_oss_defaults_and_zero_disable():
    assert _determine_cuda_graph_bs(None, None, 1, is_gpt_oss=True) == []
    assert _determine_cuda_graph_bs(None, 1, 1, is_gpt_oss=True) == [1]
    assert _determine_cuda_graph_bs(None, 0, 1, is_gpt_oss=True) == []
    assert _determine_cuda_graph_bs([0], None, 1, is_gpt_oss=True) == []
    assert _determine_cuda_graph_bs([4, 1, 4], None, 1, is_gpt_oss=True) == [1, 4]


def test_constructor_initializes_fixed_whole_model_capture():
    runner, backend = _graph_runner()

    backend.init_capture_graph.assert_called_once_with(max_seq_len=32, bs_list=[1, 2, 4])
    assert runner.target_bs_list == [1, 2, 4]
    assert runner.graph_map == {}
    assert runner.max_graph_bs == 0
    assert runner.buffer.input_ids.shape == (4,)


def test_padding_and_eligibility_use_only_successfully_captured_sizes():
    runner, _ = _graph_runner()
    runner.graph_map = {2: object(), 4: object()}
    batch = _Batch(1)

    runner.pad_batch(batch)

    assert batch.padded_size == 2
    assert runner.can_use_cuda_graph(batch)
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


def test_capture_failure_marks_shape_eager_and_continues(monkeypatch):
    runner, _ = _graph_runner((1, 2))
    runner._capture_one = MagicMock(
        side_effect=lambda bs: (_ for _ in ()).throw(RuntimeError("boom")) if bs == 2 else None
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr("minisgl.engine.graph.logger.info_rank0", MagicMock())
    monkeypatch.setattr("minisgl.engine.graph.logger.warning_rank0", MagicMock())
    monkeypatch.setattr("minisgl.engine.graph.logger.exception", MagicMock())

    runner._capture_graphs()

    assert runner._capture_one.call_args_list[0].args == (2,)
    assert runner.failed_bs == {2}
    assert not hasattr(runner, "capture_next")


def test_disable_cuda_graph_cli_alias():
    args, _ = parse_args(["--model", "unused", "--dtype", "float32", "--disable-cuda-graph"])

    assert args.cuda_graph_max_bs == 0
    shell_args, _ = parse_args(
        ["--model", "unused", "--dtype", "float32", "--disable-cuda-graph"],
        run_shell=True,
    )
    assert shell_args.cuda_graph_max_bs == 0
