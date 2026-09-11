from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    RepositionOpenAckMsg,
    RequestErrorReply,
    RequestRejectMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
    WarmupAckMsg,
    WarmupReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def _tokenize_individually(tokenize_manager, msgs: List[TokenizeMsg], logger):
    """Tokenize requests independently so one bad prompt cannot kill its peers."""

    tokenized_pairs: List[tuple[TokenizeMsg, Any]] = []
    error_replies: List[RequestErrorReply] = []
    for msg in msgs:
        try:
            tokenized_pairs.append((msg, tokenize_manager.tokenize([msg])[0]))
        except ValueError as exc:
            logger.warning("Rejecting invalid request %s: %s", msg.uid, exc)
            error_replies.append(
                RequestErrorReply(
                    uid=msg.uid,
                    status_code=400,
                    error_code="invalid_request",
                    detail=str(exc),
                )
            )
        except Exception as exc:
            logger.exception("Tokenization failed for request %s", msg.uid)
            error_replies.append(
                RequestErrorReply(
                    uid=msg.uid,
                    status_code=500,
                    error_code="tokenization_failed",
                    detail=f"Tokenization failed: {exc}",
                )
            )
    return tokenized_pairs, error_replies


def _build_user_msg(msg: TokenizeMsg, t: Any) -> UserMsg:
    return UserMsg(
        uid=msg.uid,
        input_ids=t.input_ids,
        true_positions=t.true_positions,
        raw_positions=t.raw_positions,
        radix_input_ids=t.radix_input_ids,
        radix_match_ids=t.radix_match_ids,
        sampling_params=msg.sampling_params,
        prompt_tokens=t.prompt_tokens,
        radix_key_virtual_mask=t.radix_key_virtual_mask,
        radix_key_to_token=t.radix_key_to_token,
        radix_token_to_key=t.radix_token_to_key,
        radix_positions=t.radix_positions,
        radix_repos_info=t.radix_repos_info,
        radix_next_position=t.radix_next_position,
        radix_current_reposition=t.radix_current_reposition,
        drop_event_positions=t.drop_event_positions,
        drop_range_offsets=t.drop_range_offsets,
        drop_position_ranges=t.drop_position_ranges,
        drop_effective_event_count=t.drop_effective_event_count,
        radix_commit_key_len=t.radix_commit_key_len,
        enable_thinking=msg.enable_thinking,
        stop=msg.stop,
        stop_token_seqs=t.stop_token_seqs,
        message_meta=t.message_meta,
        is_warmup=msg.is_warmup,
        internal_uid=msg.internal_uid,
        prefix_keep_mask=t.prefix_keep_mask,
        full_input_ids=t.full_input_ids,
        full_token_visible_until=t.full_token_visible_until,
        full_keep_mask=t.full_keep_mask,
        use_context_mask=msg.use_context_mask and t.full_input_ids is not None,
        context_post_prefill_keep_mask=(
            t.full_keep_mask if msg.use_context_mask and not msg.is_warmup else None
        ),
        request_received_ns=msg.request_received_ns,
        tokenize_invocations=t.tokenize_invocations,
        chat_template_invocations=t.chat_template_invocations,
    )


def _build_occurrence_radix_records(t: Any) -> torch.Tensor:
    """Freeze real-token Radix rows at the KV position where each token was born."""

    layout = t.reposition_layout
    raw_boundaries = t.reposition_raw_boundaries
    if layout is None or raw_boundaries is None:
        raise ValueError("Paged-occurrence Reposition requires compiled event boundaries.")
    if len(raw_boundaries) != len(layout.effective_reposition_stages):
        raise ValueError("Reposition boundaries and effective stages have different lengths.")

    stage_boundaries = torch.full(
        (len(layout.transition_offsets),), -1, dtype=torch.int32, device="cpu"
    )
    effective_stages = layout.effective_reposition_stages.to(torch.int64)
    effective = effective_stages > 0
    if bool(torch.any(effective).item()):
        stage_boundaries[effective_stages[effective]] = raw_boundaries[effective]

    birth_stages = layout.birth_stages.to(torch.int64)
    if bool(torch.any(birth_stages < 0).item()) or bool(
        torch.any(birth_stages >= len(stage_boundaries)).item()
    ):
        raise ValueError("Paged-occurrence token birth stage is outside the compiled layout.")
    token_boundaries = stage_boundaries[birth_stages]
    if bool(torch.any((birth_stages > 0) & (token_boundaries < 0)).item()):
        raise ValueError("Paged-occurrence token birth stage has no Reposition boundary.")

    records = layout.records.clone()
    token_rows = layout.token_to_key
    records[token_rows, 2] = token_boundaries
    records[token_rows, 3] = layout.birth_positions
    return records


def _build_occurrence_user_msg(msg: TokenizeMsg, t: Any) -> UserMsg:
    from .reposition_occurrence import compile_reposition_occurrence_plan

    if t.reposition_layout is None or t.reposition_input_ids is None:
        raise ValueError("Paged-occurrence Reposition requires a precompiled layout.")
    plan = compile_reposition_occurrence_plan(
        t.reposition_layout,
        t.full_token_visible_until,
    )
    token_count = len(t.reposition_input_ids)
    visible_until = t.full_token_visible_until
    if visible_until is None:
        visible_until = torch.full((token_count,), token_count + 1, dtype=torch.int32, device="cpu")
    keep_mask = t.reposition_layout.keep_mask.to(dtype=torch.int32).contiguous()
    raw_positions = torch.arange(token_count, dtype=torch.int32, device="cpu")
    radix_records = _build_occurrence_radix_records(t)
    message = UserMsg(
        uid=msg.uid,
        input_ids=t.reposition_input_ids,
        true_positions=t.reposition_layout.birth_positions,
        raw_positions=raw_positions,
        radix_input_ids=radix_records[t.reposition_layout.token_to_key].contiguous(),
        radix_match_ids=radix_records,
        sampling_params=msg.sampling_params,
        prompt_tokens=t.prompt_tokens,
        radix_key_virtual_mask=t.reposition_layout.virtual_mask,
        radix_key_to_token=t.reposition_layout.key_to_token,
        radix_token_to_key=t.reposition_layout.token_to_key,
        radix_positions=t.reposition_layout.positions,
        radix_repos_info=t.reposition_layout.repos_info,
        radix_next_position=t.reposition_layout.next_position,
        radix_current_reposition=t.reposition_layout.current_reposition,
        drop_event_positions=t.drop_event_positions,
        drop_range_offsets=t.drop_range_offsets,
        drop_position_ranges=t.drop_position_ranges,
        drop_effective_event_count=t.drop_effective_event_count,
        radix_commit_key_len=t.radix_commit_key_len,
        enable_thinking=msg.enable_thinking,
        stop=msg.stop,
        stop_token_seqs=t.stop_token_seqs,
        message_meta=t.message_meta,
        is_warmup=msg.is_warmup,
        internal_uid=msg.internal_uid,
        prefix_keep_mask=torch.ones(token_count, dtype=torch.int32, device="cpu"),
        full_input_ids=t.reposition_input_ids,
        full_token_visible_until=visible_until,
        full_keep_mask=keep_mask,
        use_context_mask=True,
        context_compact_stream=False,
        context_post_prefill_keep_mask=keep_mask,
        request_received_ns=msg.request_received_ns,
        tokenize_invocations=t.tokenize_invocations,
        chat_template_invocations=t.chat_template_invocations,
        context_stage_count=1,
        radix_compile_ns=t.reposition_layout.compile_ns,
        reposition_transition_count=len(t.reposition_layout.transition_raw_tokens),
        reposition_execution_mode="paged-occurrence",
        occurrence_raw_tokens=plan.occurrence_raw_tokens,
        occurrence_positions=plan.occurrence_positions,
        occurrence_birth_indices=plan.birth_occurrences,
        occurrence_terminal_indices=plan.terminal_occurrences,
        occurrence_segment_query_starts=plan.segment_query_starts,
        occurrence_segment_query_ends=plan.segment_query_ends,
        occurrence_segment_key_offsets=plan.segment_key_offsets,
        occurrence_segment_key_indices=plan.segment_key_occurrences,
    )
    message.reposition_ipc_tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in vars(message).values()
        if isinstance(value, torch.Tensor)
    )
    return message


def _prewarm_tokenizer_worker(tokenizer: Any, *, radix_drop_key_mode: str) -> None:
    """Pay tokenizer and structured Radix first-use costs before ready."""

    tokenizer.encode("")
    tokenizer.decode([])
    if radix_drop_key_mode == "delta-marker":
        from minisgl.kernel.radix_reposition import prewarm_radix_reposition_layout_kernel

        prewarm_radix_reposition_layout_kernel()


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    radix_drop_key_mode: str = "delta-marker",
    reposition_execution_mode: str = "paged-occurrence",
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    if reposition_execution_mode not in {"staged", "paged-occurrence"}:
        raise ValueError("reposition_execution_mode must be 'staged' or 'paged-occurrence'.")
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .reposition_sequence import RepositionSequenceState
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer, radix_drop_key_mode=radix_drop_key_mode)
    detokenize_manager = DetokenizeManager(tokenizer)
    reposition_sequences: dict[int, RepositionSequenceState] = {}
    prewarm_started_ns = time.perf_counter_ns()
    _prewarm_tokenizer_worker(tokenizer, radix_drop_key_mode=radix_drop_key_mode)
    logger.info(
        "Tokenizer/Radix prewarm completed in %.2f ms.",
        (time.perf_counter_ns() - prewarm_started_ns) / 1e6,
    )

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            warmup_msg = [m for m in pending_msg if isinstance(m, WarmupAckMsg)]
            reposition_open_msg = [m for m in pending_msg if isinstance(m, RepositionOpenAckMsg)]
            reject_msg = [m for m in pending_msg if isinstance(m, RequestRejectMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(warmup_msg) + len(
                reposition_open_msg
            ) + len(reject_msg) + len(abort_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                            incremental_token_ids=[msg.next_token],
                            finish_reason=msg.finish_reason,
                            matched_stop=msg.matched_stop,
                            cached_tokens=msg.cached_tokens,
                            repos_tokens=msg.repos_tokens,
                            drop_skipped_tokens=msg.drop_skipped_tokens,
                            prompt_tokens=msg.prompt_tokens,
                            completion_tokens=msg.completion_tokens,
                            server_metrics=msg.server_metrics,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(warmup_msg) > 0:
                frontend_warmups = []
                continuation_msgs: List[BaseBackendMsg] = []
                for msg in warmup_msg:
                    state = reposition_sequences.get(msg.uid)
                    if state is None:
                        frontend_warmups.append(msg)
                        continue
                    try:
                        state.accept_ack(msg)
                        next_msg = state.build_next_msg()
                        continuation_msgs.append(next_msg)
                        if state.in_flight_final:
                            reposition_sequences.pop(msg.uid, None)
                    except Exception as exc:
                        reposition_sequences.pop(msg.uid, None)
                        continuation_msgs.append(AbortBackendMsg(uid=msg.uid))
                        logger.exception("Reposition continuation failed for request %s", msg.uid)
                        send_frontend.put(
                            RequestErrorReply(
                                uid=msg.uid,
                                status_code=500,
                                error_code="reposition_sequence_failed",
                                detail=f"Reposition continuation failed: {exc}",
                            )
                        )
                if frontend_warmups:
                    batch_output = BatchFrontendMsg(
                        data=[
                            WarmupReply(
                                uid=msg.uid,
                                hit_ratio=msg.hit_ratio,
                                cached_tokens=msg.cached_tokens,
                                repos_tokens=msg.repos_tokens,
                                drop_skipped_tokens=msg.drop_skipped_tokens,
                                finished=msg.finished,
                            )
                            for msg in frontend_warmups
                        ]
                    )
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_frontend.put(batch_output)
                if continuation_msgs:
                    backend_output: BaseBackendMsg = BatchBackendMsg(data=continuation_msgs)
                    if len(continuation_msgs) == 1:
                        backend_output = continuation_msgs[0]
                    send_backend.put(backend_output)

            if len(reposition_open_msg) > 0:
                backend_msgs: List[BaseBackendMsg] = []
                for ack in reposition_open_msg:
                    state = reposition_sequences.get(ack.uid)
                    if state is None:
                        backend_msgs.append(AbortBackendMsg(uid=ack.uid))
                        continue
                    try:
                        state.activate(step_token_budget=ack.step_token_budget)
                        next_msg = state.build_next_msg()
                        backend_msgs.append(next_msg)
                        if state.in_flight_final:
                            reposition_sequences.pop(ack.uid, None)
                    except Exception as exc:
                        reposition_sequences.pop(ack.uid, None)
                        backend_msgs.append(AbortBackendMsg(uid=ack.uid))
                        logger.exception("Reposition compile failed for request %s", ack.uid)
                        send_frontend.put(
                            RequestErrorReply(
                                uid=ack.uid,
                                status_code=400 if isinstance(exc, ValueError) else 500,
                                error_code="invalid_context_events",
                                detail=str(exc),
                            )
                        )
                if backend_msgs:
                    backend_output = BatchBackendMsg(data=backend_msgs)
                    if len(backend_msgs) == 1:
                        backend_output = backend_msgs[0]
                    send_backend.put(backend_output)

            if len(reject_msg) > 0:
                for msg in reject_msg:
                    reposition_sequences.pop(msg.uid, None)
                batch_output = BatchFrontendMsg(
                    data=[
                        RequestErrorReply(
                            uid=msg.uid,
                            status_code=msg.status_code,
                            error_code=msg.error_code,
                            detail=msg.detail,
                        )
                        for msg in reject_msg
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                tokenized_pairs, error_replies = _tokenize_individually(
                    tokenize_manager, tokenize_msg, logger
                )

                if error_replies:
                    error_output = BatchFrontendMsg(data=error_replies)
                    if len(error_output.data) == 1:
                        error_output = error_output.data[0]
                    send_frontend.put(error_output)

                if tokenized_pairs:
                    backend_msgs: List[BaseBackendMsg] = []
                    for msg, tokenized in tokenized_pairs:
                        if (
                            radix_drop_key_mode == "delta-marker"
                            and tokenized.reposition_input_ids is not None
                            and reposition_execution_mode == "staged"
                        ):
                            if msg.uid in reposition_sequences:
                                raise RuntimeError(f"Duplicate Reposition sequence UID: {msg.uid}")
                            state = RepositionSequenceState.pending(msg, tokenized)
                            reposition_sequences[msg.uid] = state
                            backend_msgs.append(state.open_msg())
                        elif tokenized.reposition_input_ids is not None:
                            backend_msgs.append(_build_occurrence_user_msg(msg, tokenized))
                        else:
                            backend_msgs.append(_build_user_msg(msg, tokenized))
                    batch_output = BatchBackendMsg(data=backend_msgs)
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_backend.put(batch_output)
            if len(abort_msg) > 0:
                for msg in abort_msg:
                    reposition_sequences.pop(msg.uid, None)
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
