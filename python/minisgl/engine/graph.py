from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
    is_gpt_oss: bool = False,
) -> List[int]:
    if cuda_graph_bs is not None:
        return sorted({bs for bs in cuda_graph_bs if bs > 0})

    if cuda_graph_max_bs is None and is_gpt_oss:
        # GPT-OSS stays eager unless the operator explicitly requests a fixed
        # whole-model graph size. This avoids any startup-time auto tuning.
        return []

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        tp_cpu_group: torch.distributed.ProcessGroup | None = None,
        is_gpt_oss: bool = False,
    ) -> None:
        target_bs_list = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
            is_gpt_oss=is_gpt_oss,
        )
        self.attn_backend = attn_backend
        self.target_bs_list = target_bs_list
        self.failed_bs: set[int] = set()
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.model = model
        self.tp_cpu_group = tp_cpu_group
        self._pool = None
        self.buffer: GraphCaptureBuffer | None = None
        if not target_bs_list:
            logger.info_rank0("CUDA graph is disabled.")
            return
        self.attn_backend.init_capture_graph(
            max_seq_len=max_seq_len,
            bs_list=target_bs_list,
        )
        self.buffer = GraphCaptureBuffer.init(max(target_bs_list), vocab_size, self.device)
        self._capture_graphs()

    @property
    def graph_bs_list(self) -> List[int]:
        return sorted(self.graph_map)

    @property
    def max_graph_bs(self) -> int:
        return max(self.graph_map, default=0)

    def _capture_graphs(self) -> None:
        logger.info_rank0(
            f"Capturing fixed whole-model CUDA graphs with sizes: {self.target_bs_list}"
        )
        for bs in sorted(self.target_bs_list, reverse=True):
            local_success = True
            try:
                self._capture_one(bs)
            except Exception:
                local_success = False
                self.graph_map.pop(bs, None)
                logger.exception(f"CUDA graph capture failed for bs={bs}; using eager fallback.")

            success = torch.tensor(int(local_success), dtype=torch.int32, device="cpu")
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.all_reduce(
                    success,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_cpu_group,
                )
            if not bool(success.item()):
                self.mark_capture_failed(bs)
                logger.warning_rank0(
                    f"CUDA graph bs={bs} failed on at least one TP rank; using eager fallback."
                )

        logger.info_rank0(f"Usable CUDA graph sizes: {self.graph_bs_list}")

    def mark_capture_failed(self, bs: int) -> None:
        self.graph_map.pop(bs, None)
        self.failed_bs.add(bs)

    def _capture_one(self, bs: int) -> None:
        assert self.buffer is not None
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.padded_reqs = batch.reqs
        self.attn_backend.prepare_for_capture(batch)
        self.buffer.set_batch(batch)
        with get_global_ctx().forward_batch(batch):
            self.buffer.logits[:bs] = self.model.forward()
            with torch.cuda.graph(graph, pool=self._pool, stream=self.stream):
                self.buffer.logits[:bs] = self.model.forward()
        if self._pool is None:
            self._pool = graph.pool()
        self.graph_map[bs] = graph
        logger.info_rank0(f"Captured whole-model CUDA graph bs={bs}.")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.padded_size in self.graph_map

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        assert self.buffer is not None
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (
            next((bs for bs in self.graph_bs_list if bs >= batch.size), batch.size)
            if batch.is_decode
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        self.graph_map.clear()
        self._pool = None
        gc.collect()
