from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.kernel.context_plan import preload_context_plan_kernel
from minisgl.layers import get_rope
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    RepositionOpenAckMsg,
    RepositionOpenMsg,
    RequestMetricsState,
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
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            config.cache_type,
            drop_aware_eviction=config.drop_aware_eviction,
        )
        self.decode_manager = DecodeManager(config.page_size)
        rotary_config = config.model_config.rotary_config
        retry_rope = get_rope(
            head_dim=rotary_config.head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=(tuple(rotary_config.scaling.items()) if rotary_config.scaling else None),
        )
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            has_sliding_window=config.model_config.sliding_window is not None,
            enable_mask_free_context_prefill=config.mask_free_context_prefill,
            kv_cache=self.engine.kv_cache,
            retry_rope_cache=retry_rope.cos_sin_cache,
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
            if config.mask_free_context_prefill:
                try:
                    preload_context_plan_kernel()
                except Exception:
                    logger.warning(
                        "Could not preload the sparse Context planner kernel; "
                        "the O(N) reference remains available.",
                        exc_info=True,
                    )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.request_metrics: Dict[int, RequestMetricsState] = {}
        self.context_sequence_uids: Set[int] = set()
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
        generated_ns = time.perf_counter_ns()
        reply: List[DetokenizeMsg | WarmupAckMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    self.prefill_manager.complete_chunk(req)
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
                            hit_ratio=req.cache_reuse_ratio,
                            cached_tokens=req.reported_cached_tokens,
                            drop_skipped_tokens=req.drop_skipped_tokens,
                            finished=finished,
                            radix_match_ns=req.radix_match_ns,
                            retry_plan_ns=req.retry_plan_ns,
                            reposition_transition_count=req.reposition_transition_count,
                            reposition_h2d_bytes=req.reposition_h2d_bytes,
                            reposition_d2h_bytes=req.reposition_d2h_bytes,
                        )
                    )
                else:
                    if req.context_post_prefill_keep_mask is not None:
                        self._compact_context_after_prefill(req)
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
                    server_metrics = None
                    metrics_state = self.request_metrics.get(req.uid)
                    if metrics_state is not None:
                        metrics_state.observe_reposition(
                            radix_match_ns=req.radix_match_ns,
                            retry_plan_ns=req.retry_plan_ns,
                            transition_count=req.reposition_transition_count,
                            h2d_bytes=req.reposition_h2d_bytes,
                            d2h_bytes=req.reposition_d2h_bytes,
                        )
                        visible = not (finished and next_token in self.eos_token_ids)
                        metrics_state.observe_token(generated_ns, visible=visible)
                        if finished:
                            server_metrics = metrics_state.finish(generated_ns)
                            self.request_metrics.pop(req.uid, None)
                    reply.append(
                        DetokenizeMsg(
                            uid=req.uid,
                            next_token=next_token,
                            finished=finished,
                            finish_reason=finish_reason if finished else None,
                            matched_stop=matched_stop,
                            cached_tokens=(req.reported_cached_tokens if finished else None),
                            prompt_tokens=req.prompt_tokens if finished else None,
                            completion_tokens=req.completion_tokens if finished else None,
                            server_metrics=server_metrics,
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

    def _compact_context_after_prefill(self, req: Req) -> None:
        """Switch the final masked Reposition Prefill to its active Decode view."""

        keep_mask = req.context_post_prefill_keep_mask
        if keep_mask is None:
            return
        prompt_len = req.cached_len
        if prompt_len != len(req.input_ids) or req.device_len != prompt_len + 1:
            raise RuntimeError("Post-Prefill compaction must run immediately after prompt Prefill.")
        prompt_raw = req.raw_positions[:prompt_len].to(dtype=torch.int64, device="cpu")
        if len(prompt_raw) == 0 or int(prompt_raw[-1]) >= len(keep_mask):
            raise RuntimeError("Post-Prefill keep mask does not cover the prompt raw positions.")
        keep = (keep_mask[prompt_raw] != 0).to(dtype=torch.bool, device="cpu")
        if not bool(torch.any(keep).item()):
            raise RuntimeError("Reposition cannot Drop every prompt token before generation.")

        pages = self.table_manager.page_table[req.table_idx, :prompt_len].clone()
        keep_device = keep.to(device=pages.device, non_blocking=True)
        active_slots = torch.arange(prompt_len, dtype=torch.int64, device="cpu")
        owned = active_slots >= req.initial_active_cached_len
        if req.retry_transformed_mask is not None:
            owned[: len(req.retry_transformed_mask)] |= req.retry_transformed_mask
        dropped_owned = (~keep) & owned
        if bool(torch.any(dropped_owned).item()):
            dropped_device = dropped_owned.to(device=pages.device, non_blocking=True)
            dropped_positions = prompt_raw[dropped_owned]
            dropped_pages = pages[dropped_device]
            if req.inactive_cached_positions is None:
                req.inactive_cached_positions = dropped_positions
                req.inactive_cached_pages = dropped_pages
            else:
                assert req.inactive_cached_pages is not None
                req.inactive_cached_positions = torch.cat(
                    (req.inactive_cached_positions, dropped_positions)
                )
                req.inactive_cached_pages = torch.cat((req.inactive_cached_pages, dropped_pages))

        kept_count = int(torch.count_nonzero(keep).item())
        self.table_manager.page_table[req.table_idx, :kept_count].copy_(pages[keep_device])
        self.table_manager.token_pool[req.table_idx, :kept_count].copy_(
            self.table_manager.token_pool[req.table_idx, :prompt_len][keep_device]
        )
        self.table_manager.token_pool[req.table_idx, kept_count].copy_(
            self.table_manager.token_pool[req.table_idx, prompt_len]
        )

        queued_true_position = req.true_positions[prompt_len:].clone()
        queued_raw_position = req.raw_positions[prompt_len:].clone()
        req.input_ids = req.input_ids[keep].contiguous()
        req.true_positions = torch.cat(
            (req.true_positions[:prompt_len][keep].contiguous(), queued_true_position)
        )
        req.raw_positions = torch.cat(
            (req.raw_positions[:prompt_len][keep].contiguous(), queued_raw_position)
        )
        req.radix_input_ids = req.radix_input_ids[keep].contiguous()

        initial_keep = keep[: req.initial_active_cached_len]
        req.initial_active_cached_len = int(torch.count_nonzero(initial_keep).item())
        if req.retry_transformed_mask is not None:
            req.retry_transformed_mask = req.retry_transformed_mask[initial_keep].contiguous()
        removed = prompt_len - kept_count
        req.cached_len = kept_count
        req.device_len = kept_count + 1
        req.max_device_len -= removed
        req.use_context_mask = False
        req.context_compact_stream = False
        req.context_post_prefill_keep_mask = None
        req.full_input_ids = None
        req.full_token_visible_until = None
        req.full_keep_mask = None

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for item in msg.data:
                self._process_one_msg(item)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, RepositionOpenMsg):
            try:
                if self.radix_drop_key_mode != "delta-marker":
                    raise ValueError("Reposition requires --radix-drop-key-mode delta-marker.")
                if self.cache_manager.drop_aware_eviction:
                    raise ValueError(
                        "Reposition currently supports ordinary Radix eviction only; "
                        "disable Drop-aware eviction."
                    )
                if msg.uid in self.context_sequence_uids:
                    raise ValueError(f"Duplicate Reposition sequence UID: {msg.uid}")
                self.context_sequence_uids.add(msg.uid)
                self.send_result(
                    [
                        RepositionOpenAckMsg(
                            uid=msg.uid,
                            step_token_budget=self.prefill_budget,
                        )
                    ]
                )
            except ValueError as exc:
                self._close_context_sequence(msg.uid)
                self.send_result(
                    [
                        RequestRejectMsg(
                            uid=msg.uid,
                            status_code=400,
                            error_code="invalid_context_events",
                            detail=str(exc),
                        )
                    ]
                )
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
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
            elif self.radix_drop_key_mode == "delta-marker":
                if msg.radix_match_ids is None:
                    raise ValueError("Delta-marker Radix mode requires full radix_match_ids.")
                if msg.radix_match_ids.ndim != 2 or msg.radix_match_ids.shape[1] != 4:
                    raise ValueError(
                        "Delta-marker Radix keys must be precompiled CPU int32 [N, 4] records."
                    )

            if msg.radix_next_position is not None:
                true_input_len = msg.radix_next_position
            elif len(msg.true_positions) > 0:
                true_input_len = int(msg.true_positions[-1].item()) + 1
            else:
                true_input_len = 0
            max_seq_len = self.engine.max_seq_len
            if true_input_len > max_seq_len:
                detail = (
                    f"A Reposition materialization step needs sequence length {true_input_len}, "
                    f"which exceeds the model context length {max_seq_len}."
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
                self._close_context_sequence(msg.uid)
                return
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
                self._close_context_sequence(msg.uid)
                return

            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            if not msg.is_warmup:
                request_received_ns = msg.request_received_ns
                if request_received_ns is None:
                    request_received_ns = time.perf_counter_ns()
                prompt_tokens = (
                    msg.prompt_tokens if msg.prompt_tokens is not None else true_input_len
                )
                self.request_metrics[msg.uid] = RequestMetricsState(
                    request_received_ns=request_received_ns,
                    prompt_tokens=prompt_tokens,
                    active_prompt_tokens=len(msg.input_ids),
                    tokenize_invocations=msg.tokenize_invocations,
                    chat_template_invocations=msg.chat_template_invocations,
                    context_stage_count=msg.context_stage_count,
                    radix_compile_ns=msg.radix_compile_ns,
                    reposition_ipc_tensor_bytes=msg.reposition_ipc_tensor_bytes,
                )
                self.request_metrics[msg.uid].observe_reposition(
                    radix_match_ns=msg.radix_match_ns,
                    retry_plan_ns=msg.retry_plan_ns,
                    transition_count=msg.reposition_transition_count,
                    h2d_bytes=msg.reposition_h2d_bytes,
                    d2h_bytes=msg.reposition_d2h_bytes,
                )
            try:
                self.prefill_manager.add_one_req(msg)
            except Exception:
                self.request_metrics.pop(msg.uid, None)
                self._close_context_sequence(msg.uid)
                raise
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            self.request_metrics.pop(msg.uid, None)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if isinstance(req_to_free, PendingReq):
                self._close_context_sequence(msg.uid)
            elif req_to_free is not None:
                self._free_req_resources(req_to_free)
                # The request may still be present in an overlapping GPU batch.
                # _process_last_data uses this tombstone to discard that stale result.
                self.finished_reqs.add(req_to_free)
            else:
                self._close_context_sequence(msg.uid)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _close_context_sequence(self, uid: int) -> bool:
        if uid not in self.context_sequence_uids:
            return False
        self.context_sequence_uids.remove(uid)
        return True

    def _free_req_resources(self, req: Req) -> None:
        try:
            self.cache_manager.cache_req(req, finished=True)
        finally:
            try:
                self.table_manager.free(req.table_idx)
            finally:
                if (
                    req.is_warmup
                    and req.radix_next_position is None
                    and req.uid in self.context_sequence_uids
                ):
                    return
                self._close_context_sequence(req.uid)

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
