from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    RequestRejectMsg,
    UserMsg,
    WarmupAckMsg,
)
from minisgl.message.tokenizer import get_gpt_oss_terminal_stop_token_ids
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .radix_delta import (
    DeltaMarkerRegistry,
    inject_delta_markers,
    key_prefix_len_for_token_boundary,
)
from .radix_symbol import RadixSymbolRegistry, inject_radix_symbols
from .table import TableManager
from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        if config.drop_aware_eviction and config.cache_type != "radix":
            raise ValueError("--enable-drop-aware-eviction requires --cache-type radix.")
        if config.drop_aware_eviction and config.radix_drop_key_mode != "delta-marker":
            raise ValueError(
                "--enable-drop-aware-eviction requires --radix-drop-key-mode delta-marker."
            )
        if config.drop_aware_eviction and config.page_size != 1:
            raise ValueError("--enable-drop-aware-eviction requires --page-size 1.")
        if config.radix_drop_key_mode == "delta-marker" and config.page_size != 1:
            raise ValueError("Delta-marker Radix mode requires --page-size 1.")
        if config.radix_drop_key_mode == "delta-marker" and "trtllm" in config.attention_backend:
            raise ValueError(
                "Delta-marker Radix mode is incompatible with TRTLLM because "
                "TRTLLM requires a non-unit page size."
            )
        from minisgl.engine import Engine

        self.engine = Engine(config)
        if config.radix_drop_key_mode == "delta-marker" and config.page_size != 1:
            final_page_size = config.page_size
            try:
                self.engine.shutdown()
            except Exception:
                logger.exception(
                    "Failed to shut down Engine after an incompatible page-size adjustment."
                )
            raise ValueError(
                "Delta-marker Radix mode requires final --page-size 1, but Engine "
                f"selected {final_page_size}."
            )

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.radix_drop_key_mode = config.radix_drop_key_mode
        self.radix_symbol_registry = (
            RadixSymbolRegistry() if self.radix_drop_key_mode == "symbol" else None
        )
        self.delta_marker_registry = (
            DeltaMarkerRegistry() if self.radix_drop_key_mode == "delta-marker" else None
        )
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type,
            drop_aware_eviction=config.drop_aware_eviction,
        )
        if self.delta_marker_registry is not None:
            self.cache_manager.bind_delta_marker_registry(self.delta_marker_registry)
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        if config.contextual_prefill_mode not in {"staged", "mask"}:
            raise ValueError(
                "contextual_prefill_mode must be 'mask' or 'staged', got "
                f"{config.contextual_prefill_mode!r}."
            )
        if config.contextual_prefill_mode == "mask":
            if config.page_size != 1:
                raise ValueError("Context-mask Prefill currently requires --page-size 1.")
            self.engine.attn_backend.validate_context_mask_prefill(self.device)

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        eos_values = (
            self.eos_token_id
            if isinstance(self.eos_token_id, (list, tuple, set))
            else [self.eos_token_id]
        )
        self.eos_token_ids = {int(token_id) for token_id in eos_values if token_id is not None}
        if config.model_config.is_gpt_oss:
            self.eos_token_ids.update(get_gpt_oss_terminal_stop_token_ids())
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg | WarmupAckMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                if req in self.finished_reqs:
                    # An abort or an earlier overlapping batch already released
                    # this request. Keep the tombstone until the stale GPU result
                    # has drained, and never commit or unlock its cache twice.
                    new_finished_reqs.add(req)
                    continue

                if req.is_warmup:
                    finished = not req.can_decode
                    reply.append(
                        WarmupAckMsg(
                            uid=req.uid,
                            hit_ratio=req.cache_hit_ratio,
                            finished=finished,
                        )
                    )
                else:
                    next_token_tensor = next_tokens_cpu[i]
                    req.append_host(next_token_tensor.unsqueeze(0))
                    next_token = int(next_token_tensor.item())
                    finished = not req.can_decode
                    finish_reason = "length" if finished else None
                    matched_stop: str | None = None
                    if not req.sampling_params.ignore_eos and next_token in self.eos_token_ids:
                        finished = True
                        finish_reason = "stop"
                    stop_matched, matched_stop = req.match_stop()
                    if stop_matched:
                        finished = True
                        finish_reason = "stop"
                    reply.append(
                        DetokenizeMsg(
                            uid=req.uid,
                            next_token=next_token,
                            finished=finished,
                            finish_reason=finish_reason if finished else None,
                            matched_stop=matched_stop,
                        )
                    )

                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            true_input_len = (
                len(msg.full_input_ids)
                if msg.use_context_mask and msg.full_input_ids is not None
                else int(msg.true_positions[-1].item()) + 1 if len(msg.true_positions) > 0 else 0
            )
            max_seq_len = self.engine.max_seq_len
            max_output_len = max_seq_len - true_input_len
            if max_output_len <= 0:
                detail = (
                    f"Input true sequence length {true_input_len} exceeds the usable "
                    f"context length {max_seq_len - 1}; at least one output token is required."
                )
                logger.warning_rank0("Rejecting request %s: %s", msg.uid, detail)
                self.send_result(
                    [
                        RequestRejectMsg(
                            uid=msg.uid,
                            status_code=413,
                            error_code="context_length_exceeded",
                            detail=detail,
                        )
                    ]
                )
                return

            if self.radix_symbol_registry is not None and msg.message_meta is not None:
                state_starts = msg.message_meta.get(
                    "radix_state_starts", msg.message_meta.get("message_starts", [])
                )
                if not isinstance(state_starts, list):
                    raise ValueError("message_meta.radix_state_starts must be a list.")
                if msg.radix_match_ids is None:
                    raise ValueError("Symbol Radix mode requires full radix_match_ids.")
                msg.radix_match_ids, msg.radix_input_ids = inject_radix_symbols(
                    msg.radix_match_ids,
                    msg.true_positions,
                    state_starts,
                    self.radix_symbol_registry,
                )
            elif self.delta_marker_registry is not None:
                if msg.radix_match_ids is None:
                    raise ValueError("Delta-marker Radix mode requires full radix_match_ids.")
                drop_wire = (
                    msg.drop_event_positions,
                    msg.drop_range_offsets,
                    msg.drop_position_ranges,
                )
                if any(tensor is not None for tensor in drop_wire):
                    if not all(tensor is not None for tensor in drop_wire):
                        raise ValueError(
                            "Token-position Drop metadata must be provided as one complete set."
                        )
                    event_positions, range_offsets, position_ranges = drop_wire
                    assert event_positions is not None
                    assert range_offsets is not None
                    assert position_ranges is not None
                    marker_ids: tuple[int, ...] = ()
                    try:
                        layout = inject_delta_markers(
                            msg.radix_match_ids,
                            event_positions,
                            range_offsets,
                            position_ranges,
                            self.delta_marker_registry,
                        )
                        if layout is not None:
                            marker_ids = layout.marker_ids
                            msg.radix_match_ids = layout.keys
                            msg.radix_key_virtual_mask = layout.virtual_mask
                            msg.radix_key_to_token = layout.key_to_token
                            msg.radix_token_to_key = layout.token_to_key
                            msg.radix_marker_ids = list(marker_ids)
                        if layout is not None and msg.radix_commit_token_len is not None:
                            msg.radix_commit_key_len = key_prefix_len_for_token_boundary(
                                layout, msg.radix_commit_token_len
                            )
                    except Exception:
                        if marker_ids:
                            self.delta_marker_registry.release_request_refs(marker_ids)
                            msg.radix_marker_ids = None
                        raise

            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            try:
                self.prefill_manager.add_one_req(msg)
            except Exception:
                if self.delta_marker_registry is not None and msg.radix_marker_ids:
                    self.delta_marker_registry.release_request_refs(msg.radix_marker_ids)
                    msg.radix_marker_ids = None
                raise
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if isinstance(req_to_free, PendingReq):
                if self.delta_marker_registry is not None and req_to_free.radix_marker_ids:
                    self.delta_marker_registry.release_request_refs(req_to_free.radix_marker_ids)
                    req_to_free.radix_marker_ids = ()
            elif req_to_free is not None:
                self._free_req_resources(req_to_free)
                # The request may still be present in an overlapping GPU batch.
                # _process_last_data uses this tombstone to discard that stale result.
                self.finished_reqs.add(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        try:
            self.cache_manager.cache_req(req, finished=True)
        finally:
            try:
                self.table_manager.free(req.table_idx)
            finally:
                if self.delta_marker_registry is not None and req.radix_marker_ids:
                    self.delta_marker_registry.release_request_refs(req.radix_marker_ids)
                    req.radix_marker_ids = ()

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        indices_host[offset : offset + length].copy_(
            req.true_positions[req.cached_len : req.device_len]
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offsets_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        torch.arange(
            req.cached_len,
            req.device_len,
            out=offsets_host[offset : offset + length],
        )
        offset += length
    return mapping_host.to(device, non_blocking=True), offsets_host.to(device, non_blocking=True)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
