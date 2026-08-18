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
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer, radix_drop_key_mode=radix_drop_key_mode)
    detokenize_manager = DetokenizeManager(tokenizer)

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
            reject_msg = [m for m in pending_msg if isinstance(m, RequestRejectMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(warmup_msg) + len(
                reject_msg
            ) + len(abort_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                            finish_reason=msg.finish_reason,
                            matched_stop=msg.matched_stop,
                            cache_hit_ratio=msg.cache_hit_ratio,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(warmup_msg) > 0:
                batch_output = BatchFrontendMsg(
                    data=[
                        WarmupReply(uid=msg.uid, hit_ratio=msg.hit_ratio, finished=msg.finished)
                        for msg in warmup_msg
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(reject_msg) > 0:
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
                    batch_output = BatchBackendMsg(
                        data=[
                            UserMsg(
                                uid=msg.uid,
                                input_ids=t.input_ids,
                                true_positions=t.true_positions,
                                radix_input_ids=t.radix_input_ids,
                                radix_match_ids=t.radix_match_ids,
                                sampling_params=msg.sampling_params,
                                drop_event_positions=t.drop_event_positions,
                                drop_range_offsets=t.drop_range_offsets,
                                drop_position_ranges=t.drop_position_ranges,
                                radix_commit_token_len=t.radix_commit_token_len,
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
                            )
                            for msg, t in tokenized_pairs
                        ]
                    )
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_backend.put(batch_output)
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
