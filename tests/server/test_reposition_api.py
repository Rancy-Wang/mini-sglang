from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg, WarmupReply
from minisgl.server.api_server import (
    FrontendManager,
    OpenAICompletionRequest,
    _validate_reposition,
)
from minisgl.tokenizer.tokenize import TokenizeManager
from pydantic import ValidationError


def _request(reposition) -> OpenAICompletionRequest:
    return OpenAICompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        reposition=reposition,
    )


@pytest.mark.parametrize("value", [[0.0], [True], ["0"]])
def test_reposition_requires_strict_integer_message_ids(value) -> None:
    with pytest.raises(ValidationError):
        _request(value)


def test_reposition_accepts_strictly_increasing_message_ids() -> None:
    request = OpenAICompletionRequest(
        model="test-model",
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
        ],
        reposition=[0, 1],
    )

    _validate_reposition(
        request.reposition,
        message_count=len(request.messages or []),
        radix_drop_key_mode="delta-marker",
    )


@pytest.mark.parametrize("value", [[1, 0], [0, 0]])
def test_reposition_rejects_unsorted_or_duplicate_message_ids(value) -> None:
    with pytest.raises(HTTPException, match="strictly increasing unique"):
        _validate_reposition(
            value,
            message_count=2,
            radix_drop_key_mode="delta-marker",
        )


@pytest.mark.parametrize("value", [[-1], [2]])
def test_reposition_rejects_out_of_range_message_ids(value) -> None:
    with pytest.raises(HTTPException, match="outside the current message"):
        _validate_reposition(
            value,
            message_count=2,
            radix_drop_key_mode="delta-marker",
        )


def test_reposition_requires_delta_marker_mode() -> None:
    with pytest.raises(HTTPException, match="delta-marker"):
        _validate_reposition(
            [0],
            message_count=1,
            radix_drop_key_mode="symbol",
        )


def test_empty_reposition_preserves_ordinary_mode_compatibility() -> None:
    _validate_reposition(
        [],
        message_count=1,
        radix_drop_key_mode="symbol",
    )


def test_tokenizer_preserves_none_vs_empty_reposition() -> None:
    manager = TokenizeManager(SimpleNamespace(name_or_path="fake"))
    manager._build_template_messages = lambda messages, safe_mode: (messages, 0)
    manager._round_by_round_no_gen = lambda messages, enable_thinking, tools: (
        [10, 11],
        [0, 0],
        [0, 0],
        0,
    )
    manager._apply_chat_template = lambda *args, **kwargs: [10, 11, 12]

    def tokenize(reposition, **kwargs):
        return manager._chat_tokenize(
            TokenizeMsg(
                uid=1,
                text=[{"role": "user", "content": "hello"}],
                sampling_params=SamplingParams(max_tokens=1),
                reposition=reposition,
                **kwargs,
            )
        )

    ordinary = tokenize(None)
    retry_source = tokenize([])
    boundary_warmup = tokenize([], is_warmup=True, target_msg_id=1)

    assert ordinary.reposition_raw_boundaries is None
    assert ordinary.reposition_insert_offsets is None
    assert retry_source.reposition_raw_boundaries is not None
    assert retry_source.reposition_raw_boundaries.numel() == 0
    assert retry_source.reposition_insert_offsets is not None
    assert retry_source.reposition_insert_offsets.numel() == 0
    assert retry_source.reposition_input_ids is not None
    assert retry_source.reposition_input_ids.tolist() == [10, 11, 12]
    assert retry_source.tokenize_invocations == 1
    assert boundary_warmup.radix_commit_token_len == 2


def test_reposition_uses_one_public_tokenization_without_frontend_warmups() -> None:
    manager = FrontendManager(
        config=SimpleNamespace(contextual_prefill_mode="mask"),
        send_tokenizer=None,
        recv_tokenizer=None,
    )
    sent = []

    async def send_one(msg) -> None:
        sent.append(msg)

    manager.send_one = send_one
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    drop_rule = {
        "type": "message_drop",
        "drop_messages": {"1": [0]},
    }

    report = asyncio.run(
        manager.run_contextual_warmup(
            messages,
            drop_rule,
            [1],
            None,
            None,
            None,
            None,
        )
    )

    assert report is None
    assert sent == []


def test_drop_only_staged_warmup_preserves_legacy_key_mode() -> None:
    manager = FrontendManager(
        config=SimpleNamespace(contextual_prefill_mode="staged"),
        send_tokenizer=None,
        recv_tokenizer=None,
    )
    sent = []

    async def send_one(msg) -> None:
        sent.append(msg)

    async def wait_for_warmup(uid: int) -> WarmupReply:
        return WarmupReply(
            uid=uid,
            hit_ratio=0.0,
            cached_tokens=0,
            drop_skipped_tokens=0,
            finished=True,
        )

    manager.send_one = send_one
    manager.wait_for_warmup = wait_for_warmup
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    drop_rule = {
        "type": "message_drop",
        "drop_messages": {"1": [0]},
    }

    asyncio.run(
        manager.run_contextual_warmup(
            messages,
            drop_rule,
            None,
            None,
            None,
            None,
            None,
        )
    )

    assert len(sent) == 3
    assert all(msg.reposition is None for msg in sent)
