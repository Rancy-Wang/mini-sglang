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
from minisgl.tokenizer.template_provenance import TemplateTokenProvenance
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


def test_tokenizer_treats_empty_reposition_as_ordinary_with_warning(caplog) -> None:
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

    assert ordinary.reposition_raw_boundaries is None
    assert ordinary.reposition_insert_offsets is None
    assert retry_source.reposition_raw_boundaries is not None
    assert retry_source.reposition_raw_boundaries.numel() == 0
    assert retry_source.reposition_insert_offsets is not None
    assert retry_source.reposition_insert_offsets.numel() == 0
    assert retry_source.reposition_input_ids is None
    assert "Ignoring empty Reposition list" in caplog.text


def test_tokenizer_warns_and_skips_all_noop_reposition(caplog) -> None:
    manager = TokenizeManager(SimpleNamespace(name_or_path="fake"))
    manager._build_template_messages = lambda messages, safe_mode: (messages, 0)
    manager._build_template_provenance = lambda *args, **kwargs: TemplateTokenProvenance(
        input_ids=[10, 11, 12],
        owners=[0, 0, 1],
        offsets=[(0, 1), (1, 2), (2, 3)],
        rendered_text="abc",
        char_owners=[0, 0, 1],
        cross_owner_tokens=0,
    )
    result = manager._chat_tokenize(
        TokenizeMsg(
            uid=19,
            text=[{"role": "user", "content": "hello"}],
            sampling_params=SamplingParams(max_tokens=1),
            reposition=[0],
        )
    )

    assert result.reposition_input_ids is None
    assert result.message_meta["ignored_reposition_boundaries"] == [1]
    assert "request 19" in caplog.text
    assert "[1]" in caplog.text


def test_tokenizer_warns_for_ignored_prefix_but_stages_effective_reposition(caplog) -> None:
    manager = TokenizeManager(SimpleNamespace(name_or_path="fake"))
    manager._build_template_messages = lambda messages, safe_mode: (messages, 0)
    manager._build_template_provenance = lambda *args, **kwargs: TemplateTokenProvenance(
        input_ids=[10, 11, 12, 13, 14],
        owners=[0, 0, 1, 1, 2],
        offsets=[(index, index + 1) for index in range(5)],
        rendered_text="abcde",
        char_owners=[0, 0, 1, 1, 2],
        cross_owner_tokens=0,
    )

    result = manager._chat_tokenize(
        TokenizeMsg(
            uid=23,
            text=[
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "answer"},
            ],
            sampling_params=SamplingParams(max_tokens=1),
            drop_message={1: [0]},
            reposition=[0, 1],
        )
    )

    assert result.reposition_input_ids is not None
    assert result.reposition_layout.effective_reposition_stages.tolist() == [0, 1]
    assert result.message_meta["ignored_reposition_boundaries"] == [1]
    assert "request 23" in caplog.text


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
