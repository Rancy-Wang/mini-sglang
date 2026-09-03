from __future__ import annotations

import multiprocessing as mp
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
    RequestErrorReply,
    RequestRejectMsg,
    RepositionOpenAckMsg,
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
        radix_commit_token_len=t.radix_commit_token_len,
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
        use_context_mask=msg.use_context_mask,
        request_received_ns=msg.request_received_ns,
        tokenize_invocations=t.tokenize_invocations,
    )


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
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
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
                        ):
                            if msg.uid in reposition_sequences:
                                raise RuntimeError(f"Duplicate Reposition sequence UID: {msg.uid}")
                            state = RepositionSequenceState.pending(msg, tokenized)
                            reposition_sequences[msg.uid] = state
                            backend_msgs.append(state.open_msg())
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
