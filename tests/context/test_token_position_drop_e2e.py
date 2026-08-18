from __future__ import annotations

from pathlib import Path

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.template_provenance import build_template_token_provenance
from minisgl.tokenizer.tokenize import TokenizeManager
from transformers import AutoTokenizer

QWEN3_17B = next(
    (
        path
        for path in (
            Path("/share/public/public_models/Qwen3-1.7B"),
            Path(
                "/share/wangruoxi/.cache/huggingface/hub/"
                "models--Qwen--Qwen3-1.7B/snapshots/"
                "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
            ),
        )
        if path.exists()
    ),
    Path("/share/public/public_models/Qwen3-1.7B"),
)


def _tokenize(manager, messages, drop_message):
    return manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=8),
                target_msg_id=len(messages),
                drop_message=drop_message,
            )
        ]
    )[0]


def _legacy_message_keep_mask(tokenizer, messages, full_ids, dropped_messages):
    canonical_no_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    canonical_with_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    provenance = build_template_token_provenance(
        tokenizer,
        messages,
        canonical_text=canonical_with_gen,
        canonical_no_generation_text=canonical_no_gen,
        expected_input_ids=full_ids,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=None,
    )
    keep = torch.tensor(
        [owner not in dropped_messages for owner in provenance.owners],
        dtype=torch.bool,
    )
    return provenance, keep


@pytest.mark.skipif(not QWEN3_17B.exists(), reason="Qwen3-1.7B tokenizer is unavailable")
def test_long_agent_chat_message_drop_and_position_drop_are_text_identical():
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_17B, local_files_only=True)
    manager = TokenizeManager(tokenizer)

    def visible(label, char):
        return f"{label}_BEGIN_" + char * 1450 + f"_{label}_END"

    messages = [
        {"role": "system", "content": visible("SYSTEM", "s")},
        {"role": "user", "content": visible("USER_ONE", "u")},
        {
            "role": "assistant",
            "reasoning_content": "FIRST_PRIVATE_REASON_" + "r" * 3000,
            "content": visible("ASSISTANT_ONE", "a"),
        },
        {"role": "user", "content": visible("USER_TWO", "v")},
        {
            "role": "assistant",
            "reasoning_content": "SECOND_PRIVATE_REASON_" + "q" * 3000,
            "content": visible("ASSISTANT_TWO", "b"),
        },
        {"role": "user", "content": visible("USER_THREE", "w")},
    ]

    result = _tokenize(manager, messages, {3: [2]})
    no_drop = _tokenize(manager, messages, None)
    full_ids = result.full_input_ids.tolist()
    provenance, legacy_keep = _legacy_message_keep_mask(tokenizer, messages, full_ids, {2})
    legacy_positions = torch.arange(len(full_ids), dtype=torch.int32)[legacy_keep]
    legacy_input_ids = result.full_input_ids[legacy_keep]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert len(rendered) >= 8000
    assert "FIRST_PRIVATE_REASON_" not in rendered
    assert "SECOND_PRIVATE_REASON_" not in rendered
    assert no_drop.input_ids.tolist() == full_ids

    assert torch.equal(result.full_keep_mask.to(torch.bool), legacy_keep)
    assert torch.equal(result.true_positions, legacy_positions)
    assert torch.equal(result.input_ids, legacy_input_ids)
    assert tokenizer.decode(result.input_ids) == tokenizer.decode(legacy_input_ids)

    dropped_positions = {
        position
        for start, end in result.drop_position_ranges.view(-1, 2).tolist()
        for position in range(start, end)
    }
    expected_dropped_positions = {
        position for position, owner in enumerate(provenance.owners) if owner == 2
    }
    assert dropped_positions == expected_dropped_positions
    assert result.drop_event_positions.numel() == 1
    event_position = int(result.drop_event_positions[0].item())
    sentinel = torch.iinfo(torch.int32).max
    for position, owner in enumerate(provenance.owners):
        expected = event_position if owner == 2 else sentinel
        assert int(result.full_token_visible_until[position].item()) == expected


@pytest.mark.skipif(not QWEN3_17B.exists(), reason="Qwen3-1.7B tokenizer is unavailable")
def test_multiple_message_drop_union_equals_position_range_union():
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_17B, local_files_only=True)
    manager = TokenizeManager(tokenizer)
    messages = [
        {"role": "system", "content": "SYSTEM_EQUIVALENCE"},
        {"role": "user", "content": "USER_ONE_EQUIVALENCE"},
        {"role": "assistant", "content": "ANSWER_ONE_EQUIVALENCE"},
        {"role": "user", "content": "USER_TWO_EQUIVALENCE"},
        {"role": "assistant", "content": "ANSWER_TWO_EQUIVALENCE"},
        {"role": "user", "content": "USER_THREE_EQUIVALENCE"},
    ]
    result = _tokenize(manager, messages, {4: [1, 2], 5: [1, 2, 3]})
    provenance, legacy_keep = _legacy_message_keep_mask(
        tokenizer, messages, result.full_input_ids.tolist(), {1, 2, 3}
    )

    assert torch.equal(result.full_keep_mask.to(torch.bool), legacy_keep)
    assert torch.equal(
        result.input_ids,
        result.full_input_ids[legacy_keep],
    )
    assert len(result.drop_event_positions) == 2
    assert {
        position
        for start, end in result.drop_position_ranges.view(-1, 2).tolist()
        for position in range(start, end)
    } == {position for position, owner in enumerate(provenance.owners) if owner in {1, 2, 3}}
