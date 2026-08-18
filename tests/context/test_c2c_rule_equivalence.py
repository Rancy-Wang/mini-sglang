from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.template_provenance import build_template_token_provenance
from minisgl.tokenizer.tokenize import TokenizeManager
from transformers import AutoTokenizer


C2C_EXAMPLES = Path("/share/wangruoxi/repo/C2C/script/context/examples.yaml")
QWEN3_17B = Path("/share/public/public_models/Qwen3-1.7B")
ASSISTANT_FIXTURES = {
    "math_followup": ["15 + 27 = 42."],
    "memory_test": [
        "I will remember that your favorite color is blue and your lucky number is 7.",
        "Your lucky number is 7.",
    ],
}


def _load_examples():
    yaml = pytest.importorskip("yaml")
    if not C2C_EXAMPLES.exists():
        pytest.skip("C2C examples.yaml is unavailable")
    data = yaml.safe_load(C2C_EXAMPLES.read_text())
    return data["examples"]


def _history_at_trigger(example, trigger):
    messages = []
    if example.get("system_prompt") is not None:
        messages.append({"role": "system", "content": example["system_prompt"]})
    fixtures = ASSISTANT_FIXTURES.get(example["name"], [])
    for round_id, user_content in enumerate(example["user_messages"]):
        messages.append({"role": "user", "content": user_content})
        if len(messages) - 1 == trigger:
            return messages
        if round_id >= len(fixtures):
            pytest.fail(
                f"missing deterministic assistant fixture for {example['name']} "
                f"round {round_id}"
            )
        messages.append({"role": "assistant", "content": fixtures[round_id]})
    pytest.fail(f"trigger {trigger} is not a user message in {example['name']}")


def _tokenize(manager, messages, drop_rule):
    return manager.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=messages,
                sampling_params=SamplingParams(max_tokens=1),
                target_msg_id=len(messages),
                drop_rule=drop_rule,
            )
        ]
    )[0]


def _oracle(tokenizer, messages, full_ids, dropped_ids):
    no_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    with_gen = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    provenance = build_template_token_provenance(
        tokenizer,
        messages,
        canonical_text=with_gen,
        canonical_no_generation_text=no_gen,
        expected_input_ids=full_ids,
        tools=None,
        add_generation_prompt=True,
        enable_thinking=None,
    )
    owner_ranges = TokenizeManager._build_owner_position_ranges(provenance.owners)
    ranges = TokenizeManager._ranges_for_messages(owner_ranges, set(dropped_ids))
    generation_start = provenance.owners.index(len(messages))
    keep = torch.ones(len(full_ids), dtype=torch.bool)
    for start, end in ranges:
        keep[start:end] = False
    return provenance, ranges, generation_start, keep


@pytest.mark.skipif(not QWEN3_17B.exists(), reason="Qwen3-1.7B tokenizer is unavailable")
def test_every_active_c2c_example_has_exact_message_text_and_oracle_equivalence():
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_17B, local_files_only=True)
    manager = TokenizeManager(tokenizer)

    for example in _load_examples():
        for raw_trigger, dropped_ids in example.get("drop_messages", {}).items():
            trigger = int(raw_trigger)
            dropped_ids = [int(message_id) for message_id in dropped_ids]
            messages = _history_at_trigger(example, trigger)
            assert trigger == len(messages) - 1

            message_rule = {
                "type": "message_drop",
                "drop_messages": {str(trigger): dropped_ids},
            }
            text_rule = {
                "type": "text_drop",
                "drop_messages": [
                    {
                        "role": message["role"],
                        "content": message["content"] if message_id in dropped_ids else None,
                    }
                    for message_id, message in enumerate(messages)
                ],
            }
            message_result = _tokenize(manager, messages, message_rule)
            text_result = _tokenize(manager, messages, text_rule)
            provenance, oracle_ranges, event_pos, oracle_keep = _oracle(
                tokenizer,
                messages,
                message_result.full_input_ids.tolist(),
                dropped_ids,
            )
            flat_oracle = [value for pair in oracle_ranges for value in pair]

            assert message_result.full_input_ids.tolist() == text_result.full_input_ids.tolist()
            assert message_result.drop_event_positions.tolist() == [event_pos]
            assert text_result.drop_event_positions.tolist() == [event_pos]
            assert message_result.drop_range_offsets.tolist() == [0, len(oracle_ranges)]
            assert text_result.drop_range_offsets.tolist() == [0, len(oracle_ranges)]
            assert message_result.drop_position_ranges.tolist() == flat_oracle
            assert text_result.drop_position_ranges.tolist() == flat_oracle
            assert message_result.full_keep_mask.to(torch.bool).tolist() == oracle_keep.tolist()
            assert text_result.full_keep_mask.to(torch.bool).tolist() == oracle_keep.tolist()
            oracle_positions = torch.arange(
                len(provenance.input_ids), dtype=torch.int32
            )[oracle_keep]
            assert torch.equal(message_result.true_positions, oracle_positions)
            assert torch.equal(text_result.true_positions, oracle_positions)
