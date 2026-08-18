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
    capture_graphs.assert_called_once_with(32, 16, model)
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

    backend.init_capture_graph.assert_not_called()
    assert runner.max_graph_bs == 4
    assert runner.graph_bs_list == [1, 2, 4]


def test_padding_and_eligibility_match_main_for_multi_request_batch():
    runner, _ = _graph_runner()
    batch = _Batch(1)

    runner.pad_batch(batch)

    assert batch.padded_size == 1
    assert runner.can_use_cuda_graph(batch)

    multi_request_batch = _Batch(3)
    runner.pad_batch(multi_request_batch)
    assert multi_request_batch.padded_size == 4
    assert runner.can_use_cuda_graph(multi_request_batch)

    oversized_batch = _Batch(5)
    runner.pad_batch(oversized_batch)
    assert oversized_batch.padded_size == 5
    assert not runner.can_use_cuda_graph(oversized_batch)

    prefill = _Batch(1, decode=False)
    runner.pad_batch(prefill)
    assert prefill.padded_size == 1
    assert not runner.can_use_cuda_graph(prefill)


def test_disable_cuda_graph_cli_alias():
    args, _ = parse_args(["--model", "unused", "--dtype", "float32", "--disable-cuda-graph"])

    assert args.cuda_graph_max_bs == 0
    shell_args, _ = parse_args(
        ["--model", "unused", "--dtype", "float32", "--disable-cuda-graph"],
        run_shell=True,
    )
    assert shell_args.cuda_graph_max_bs == 0
