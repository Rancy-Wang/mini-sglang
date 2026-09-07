from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from reference_runtime_adapter import (
    ReferenceCase,
    compact_dynamic_cache,
    render_case,
    short_reference_cases,
    validate_case_plan,
)
from repetition_cause_runner import diagnose_reference


class PrefixTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize and not enable_thinking
        tokens = []
        for index, _ in enumerate(messages):
            tokens.extend((100 + index, 200 + index))
        if add_generation_prompt:
            tokens.append(999)
        return tokens


def test_reference_cases_have_independent_canonical_boundaries():
    tokenizer = PrefixTokenizer()
    for case in short_reference_cases():
        ids, boundaries = render_case(tokenizer, case)
        trigger, removed = validate_case_plan(case, ids, boundaries)
        assert boundaries == [(2 * i, 2 * i + 2) for i in range(len(case.messages))]
        if case.drop_after_message is None:
            assert trigger == len(ids) and removed == ()
        else:
            assert trigger == boundaries[case.drop_after_message][1]
            assert removed == tuple(
                token
                for message in case.drop_messages
                for token in range(*boundaries[message])
            )


def test_reference_plan_rejects_future_and_noncanonical_ranges():
    case = short_reference_cases()[1]
    ids, boundaries = render_case(PrefixTokenizer(), case)
    with pytest.raises(ValueError, match="preceding"):
        validate_case_plan(replace(case, drop_messages=(2,)), ids, boundaries)
    with pytest.raises(ValueError, match="contiguous"):
        validate_case_plan(case, ids, ((0, 2), (3, 4), (4, 6)))
    with pytest.raises(ValueError, match="requires"):
        validate_case_plan(replace(case, drop_after_message=None), ids, boundaries)


def test_render_rejects_prefix_template_drift():
    class DriftingTokenizer(PrefixTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if not kwargs["add_generation_prompt"] and len(messages) == 2:
                result[0] = -1
            return result

    with pytest.raises(RuntimeError, match="canonical full prefix"):
        render_case(DriftingTokenizer(), short_reference_cases()[0])


def test_dynamic_cache_compaction_is_layer_consistent_and_fresh():
    layers = []
    for offset in (0, 100):
        keys = torch.arange(offset, offset + 12).reshape(1, 1, 4, 3)
        values = keys + 1000
        layers.append(SimpleNamespace(is_initialized=True, keys=keys, values=values))
    cache = SimpleNamespace(layers=layers)
    compact_dynamic_cache(cache, torch.tensor([1, 3], dtype=torch.int64))
    for layer, offset in zip(cache.layers, (0, 100), strict=True):
        assert layer.keys.flatten().tolist() == list(range(offset + 3, offset + 6)) + list(
            range(offset + 9, offset + 12)
        )
        assert layer.values.flatten().tolist() == [x + 1000 for x in layer.keys.flatten()]
        assert layer.keys.is_contiguous() and layer.values.is_contiguous()


def test_dynamic_cache_compaction_rejects_bad_indices_and_lengths():
    layer = SimpleNamespace(
        is_initialized=True,
        keys=torch.zeros(1, 1, 4, 3),
        values=torch.zeros(1, 1, 3, 3),
    )
    cache = SimpleNamespace(layers=[layer])
    with pytest.raises(ValueError, match="one-dimensional int64"):
        compact_dynamic_cache(cache, torch.tensor([1], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="lengths disagree"):
        compact_dynamic_cache(cache, torch.tensor([1], dtype=torch.int64))


def test_case_schema_rejects_drop_without_trigger():
    case = ReferenceCase(
        "invalid",
        ({"role": "user", "content": "x"},),
        drop_messages=(0,),
    )
    ids, boundaries = render_case(PrefixTokenizer(), case)
    with pytest.raises(ValueError, match="requires"):
        validate_case_plan(case, ids, boundaries)


def test_retained_drop_rows_must_be_unique_members(monkeypatch):
    import reference_runtime_adapter as adapter

    class EmptyCache:
        def __init__(self, config=None):
            self.layers = []

    monkeypatch.setattr(
        "transformers.cache_utils.DynamicCache",
        EmptyCache,
    )
    case = short_reference_cases()[1]
    model = SimpleNamespace(config=SimpleNamespace())
    tokenizer = PrefixTokenizer()
    with pytest.raises(ValueError, match="duplicates"):
        adapter.generate_case(
            model,
            tokenizer,
            case,
            max_tokens=1,
            retained_removed_raw=(0, 0),
        )
    with pytest.raises(ValueError, match="subset"):
        adapter.generate_case(
            model,
            tokenizer,
            case,
            max_tokens=1,
            retained_removed_raw=(999,),
        )


def test_diagnosis_requires_copy_failure_and_rebuild_recovery():
    normal = [10, 11, 12]
    repeated = [20] * 8

    def row(name, tokens, *, clone=False, rebuild=False, sha=None):
        return {
            "case": {"name": name},
            "tokens": tokens,
            "canonical_sha256": sha or name,
            "clone_survivors": clone,
            "rebuild_survivors": rebuild,
        }

    payload = {
        "status": "completed",
        "fingerprint": {"model_manifest": {"config.json": "frozen"}},
        "results": [
            row("original_no_drop", normal, sha="same"),
            row("original_drop_first_user", repeated, sha="same"),
            row("original_drop_first_user", repeated, clone=True, sha="same"),
            row("original_drop_first_user", normal, rebuild=True, sha="same"),
            row("two_turn_drop_first_user", repeated),
            row("two_turn_drop_first_user", repeated, clone=True),
            row("two_turn_drop_first_user", normal, rebuild=True),
            row("two_turn_drop_second_user", normal),
            row("two_turn_drop_second_user", normal, rebuild=True),
            row("system_kept_drop_first_user", normal),
            row("system_kept_drop_first_user", normal, rebuild=True),
            row("orphaned_assistant_no_drop", normal),
            row("question_only_no_drop", [30, 31, 32]),
        ],
    }
    diagnosis = diagnose_reference(payload)
    assert diagnosis["verdict"] == "stale_survivor_kv_after_drop"

    payload["results"][2]["tokens"] = normal
    with pytest.raises(AssertionError, match="fresh_allocation_preserves_failure"):
        diagnose_reference(payload)
