from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata, build_context_attention_segments
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIContextSegmentMetadata:
    query_start: int
    query_end: int
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    cu_seqlens_q_gpu: torch.Tensor
    indices: torch.Tensor
    last_page_len_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    initialized_wrappers: set[int] = field(default_factory=set)
    wrappers: Dict[int, Any] = field(default_factory=dict)


@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    is_decode:          bool
    context_segments:  tuple[FIContextSegmentMetadata, ...] | None = None
    graph_bs:           int | None = None
    initialized_wrappers: set[int] = field(default_factory=set)
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )
    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self._prefill_wrapper_cls = BatchPrefillWithPagedKVCacheWrapper
        self._decode_wrapper_cls = BatchDecodeWithPagedKVCacheWrapper
        self.prefill_wrappers: Dict[int, BatchPrefillWithPagedKVCacheWrapper] = {
            -1: self._new_prefill_wrapper()
        }
        self.decode_wrappers: Dict[int, BatchDecodeWithPagedKVCacheWrapper] = {
            -1: self._new_decode_wrapper()
        }
        if config.sliding_window is not None:
            window_left = config.sliding_window - 1
            self.prefill_wrappers[window_left] = self._new_prefill_wrapper()
            self.decode_wrappers[window_left] = self._new_decode_wrapper()

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[
            tuple[int, int], CUDAGraphBatchDecodeWithPagedKVCacheWrapper
        ] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _new_prefill_wrapper(self):
        return self._prefill_wrapper_cls(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # FlashInfer FA3 is slow; use its FA2 backend.
        )

    def _new_decode_wrapper(self):
        return self._decode_wrapper_cls(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # FlashInfer FA3 is slow; use its FA2 backend.
        )

    @staticmethod
    def _window_left(sliding_window: int | None) -> int:
        return -1 if sliding_window is None else sliding_window

    def _initialize_metadata_once(
        self,
        metadata: FIMetadata | FIContextSegmentMetadata,
        wrapper,
        *,
        is_decode: bool,
        window_left: int,
    ) -> None:
        wrapper_id = id(wrapper)
        if wrapper_id in metadata.initialized_wrappers:
            return
        num_qo_heads = getattr(metadata, "num_qo_heads", self.qo_head_local)
        num_kv_heads = getattr(metadata, "num_kv_heads", self.kv_head_local)
        head_dim = getattr(metadata, "head_dim", self.config.head_dim)
        page_size = getattr(metadata, "page_size", 1)
        pos_encoding_mode = getattr(metadata, "pos_encoding_mode", "NONE")
        dtype = getattr(metadata, "dtype", self.kvcache.dtype)
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if is_decode:
            wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pos_encoding_mode=pos_encoding_mode,
                window_left=window_left,
                seq_lens=metadata.seq_lens_cpu,
                data_type=dtype,
                q_data_type=dtype,
                kv_data_type=dtype,
                non_blocking=True,
            )
        else:
            qo_indptr = metadata.cu_seqlens_q_cpu
            kv_indptr = metadata.cu_seqlens_k_cpu
            last_page_len = metadata.last_page_len_cpu
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                pos_encoding_mode=pos_encoding_mode,
                window_left=window_left,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=dtype,
                kv_data_type=dtype,
                non_blocking=True,
                causal=True,
            )
        metadata.initialized_wrappers.add(wrapper_id)
        self.last_event.record()

    def _ordinary_wrapper(self, metadata: FIMetadata, window_left: int):
        if metadata.graph_bs is not None:
            return self.graph_wrappers[(metadata.graph_bs, window_left)]
        wrappers = self.decode_wrappers if metadata.is_decode else self.prefill_wrappers
        if window_left not in wrappers:
            wrappers[window_left] = (
                self._new_decode_wrapper() if metadata.is_decode else self._new_prefill_wrapper()
            )
        return wrappers[window_left]

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        sinks: torch.Tensor | None = None,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        window_left = self._window_left(sliding_window)
        run_kwargs = {"window_left": window_left}
        if sinks is not None:
            run_kwargs["sinks"] = sinks

        if metadata.context_segments is not None:
            outputs = []
            for segment in metadata.context_segments:
                wrapper = segment.wrappers.get(window_left)
                if wrapper is None:
                    wrapper = self._new_prefill_wrapper()
                    segment.wrappers[window_left] = wrapper
                self._initialize_metadata_once(
                    segment,
                    wrapper,
                    is_decode=False,
                    window_left=window_left,
                )
                outputs.append(
                    wrapper.run(
                        q=q[segment.query_start : segment.query_end],
                        paged_kv_cache=kv_cache,
                        **run_kwargs,
                    )
                )
            return torch.cat(outputs, dim=0)

        wrapper = self._ordinary_wrapper(metadata, window_left)
        self._initialize_metadata_once(
            metadata,
            wrapper,
            is_decode=metadata.is_decode,
            window_left=window_left,
        )
        return wrapper.run(q=q, paged_kv_cache=kv_cache, **run_kwargs)

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        masked_reqs = [req for req in reqs if req.use_context_mask]
        if masked_reqs:
            if not batch.is_prefill or batch.size != 1 or len(reqs) != 1:
                raise RuntimeError(
                    "FlashInfer context-mask Prefill must be an unpadded, single-request batch."
                )
            if len(masked_reqs) != 1:
                raise RuntimeError("Mixed masked and ordinary FlashInfer requests are unsupported.")

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        context_segments = None
        if masked_reqs:
            req = masked_reqs[0]
            if req.full_token_visible_until is None:
                raise RuntimeError("Context-mask Prefill request is missing visibility metadata.")

        page_table = get_global_ctx().page_table
        flat_indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        if masked_reqs:
            req = masked_reqs[0]
            assert req.full_token_visible_until is not None
            base_indices = page_table[req.table_idx, : req.device_len]
            context_segments = tuple(
                FIContextSegmentMetadata(
                    query_start=segment.query_start,
                    query_end=segment.query_end,
                    cu_seqlens_q_cpu=torch.tensor(
                        [0, segment.query_end - segment.query_start], **CPU_KWARGS
                    ),
                    cu_seqlens_k_cpu=torch.tensor(
                        [0, len(segment.key_positions)], **CPU_KWARGS
                    ),
                    cu_seqlens_q_gpu=torch.tensor(
                        [0, segment.query_end - segment.query_start],
                        device=device,
                        dtype=torch.int32,
                    ),
                    indices=base_indices.index_select(
                        0,
                        segment.key_positions.to(
                            device=device, dtype=torch.int64, non_blocking=True
                        ),
                    ),
                    last_page_len_cpu=self._get_ones_cpu(1),
                    seq_lens_cpu=torch.tensor([len(segment.key_positions)], **CPU_KWARGS),
                )
                for segment in build_context_attention_segments(
                    req.full_token_visible_until,
                    query_start=req.cached_len,
                    query_length=req.extend_len,
                    key_length=req.device_len,
                )
            )
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=flat_indices,
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            is_decode=batch.is_decode,
            context_segments=context_segments,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and self.capture
        capture = self.capture
        windows = {-1}
        if self.config.sliding_window is not None:
            windows.add(self.config.sliding_window - 1)
        for window_left in windows:
            key = (bs, window_left)
            assert key not in self.graph_wrappers
            wrapper = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                self.float_workspace_buffer,
                kv_layout="NHD",
                use_tensor_cores=self.use_tensor_cores,
                indptr_buffer=capture.cu_seqlens_k[: bs + 1],
                indices_buffer=capture.indices,
                last_page_len_buffer=capture.one_tensor[:bs],
            )
            wrapper._backend = "fa2"
            self.graph_wrappers[key] = wrapper
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.graph_bs = bs
        for window_left in windows:
            self._initialize_metadata_once(
                metadata,
                self.graph_wrappers[(bs, window_left)],
                is_decode=True,
                window_left=window_left,
            )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata)
        assert self.capture is not None and bs in self.capture_bs
        metadata.graph_bs = bs
        windows = {-1}
        if self.config.sliding_window is not None:
            windows.add(self.config.sliding_window - 1)
        for window_left in windows:
            self._initialize_metadata_once(
                metadata,
                self.graph_wrappers[(bs, window_left)],
                is_decode=True,
                window_left=window_left,
            )
