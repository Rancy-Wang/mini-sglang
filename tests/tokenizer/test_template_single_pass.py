from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.server import _build_occurrence_user_msg
from minisgl.tokenizer.tokenize import TokenizeManager


class _SinglePassTokenizer:
    name_or_path = "single-pass"
    is_fast = True
    special_tokens_map: dict[str, str] = {}

    def __init__(self) -> None:
        self.encode_calls = 0
        self.apply_calls = 0
        self.chat_template = (
            "{% for message in messages %}"
            "{{ '<' + message['role'] + '>' + message['content'] }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
        )

    def get_chat_template(self, *, tools=None):
        del tools
        return self.chat_template

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        self.encode_calls += 1
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        from jinja2 import Template

        self.apply_calls += 1
        text = Template(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )
        if tokenize:
            self.encode_calls += 1
            return self.encode(text)
        return text

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


def test_ordinary_chat_renders_and_encodes_exactly_once(monkeypatch) -> None:
    tokenizer = _SinglePassTokenizer()
    manager = TokenizeManager(tokenizer)

    def forbidden(*args, **kwargs):
        raise AssertionError("No-event inference must bypass context compilers.")

    monkeypatch.setattr(manager, "_build_template_provenance", forbidden)
    monkeypatch.setattr(manager, "_compile_delta_layout", forbidden)
    result = manager._chat_tokenize(
        TokenizeMsg(
            uid=30,
            text=[
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "new"},
            ],
            sampling_params=SamplingParams(max_tokens=1),
        )
    )

    expected = "<user>old<assistant>answer<user>new<assistant>"
    assert result.input_ids.tolist() == [ord(char) for char in expected]
    assert result.tokenize_invocations == 1
    assert result.chat_template_invocations == 1
    assert tokenizer.encode_calls == 1
    assert tokenizer.apply_calls == 1
    assert result.reposition_layout is None
    assert result.radix_key_to_token is None
    assert result.radix_match_ids[:, 1].tolist() == result.input_ids.tolist()
    for removed in (
        "message_starts",
        "owner_starts",
        "unstable_rounds",
        "no_gen_with_gen_unstable",
        "no_gen_with_gen_lcp",
        "no_gen_with_gen_lcsuf",
    ):
        assert removed not in result.message_meta
    assert not hasattr(result, "radix_commit_token_len")


def test_structured_drop_and_reposition_render_and_encode_exactly_once() -> None:
    tokenizer = _SinglePassTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=31,
        text=[
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "new"},
        ],
        sampling_params=SamplingParams(max_tokens=1),
        drop_message={1: [0]},
        reposition=[1],
    )
    result = manager._chat_tokenize(msg)

    expected = "<user>old<assistant>answer<user>new<assistant>"
    assert result.reposition_input_ids is not None
    assert result.reposition_input_ids.tolist() == [ord(char) for char in expected]
    assert result.tokenize_invocations == 1
    assert result.chat_template_invocations == 1
    assert tokenizer.encode_calls == 1
    assert tokenizer.apply_calls == 0
    assert result.message_meta["gen_prompt_start"] == len(expected) - len("<assistant>")
    assert torch.any(result.reposition_layout.records[:, 0] == 1)
    backend = _build_occurrence_user_msg(msg, result)
    assert torch.equal(
        backend.occurrence_layout_transition_raw_tokens,
        result.reposition_layout.transition_raw_tokens,
    )
    assert backend.occurrence_raw_tokens is None
    assert backend.occurrence_segment_key_indices is None


def test_empty_and_future_drop_use_ordinary_path(monkeypatch) -> None:
    manager = TokenizeManager(_SinglePassTokenizer())

    def forbidden(*args, **kwargs):
        raise AssertionError("Inactive events must not compile context metadata.")

    monkeypatch.setattr(manager, "_build_template_provenance", forbidden)
    monkeypatch.setattr(manager, "_compile_delta_layout", forbidden)
    for drops in ({}, {0: []}, {9: [0]}):
        result = manager._chat_tokenize(
            TokenizeMsg(
                uid=32,
                text=[{"role": "user", "content": "hello"}],
                sampling_params=SamplingParams(max_tokens=2),
                drop_message=drops,
                use_context_mask=True,
            )
        )
        from minisgl.tokenizer.server import _build_user_msg

        message = _build_user_msg(
            TokenizeMsg(
                uid=32,
                text="unused",
                sampling_params=SamplingParams(max_tokens=2),
                use_context_mask=True,
            ),
            result,
        )
        assert not message.use_context_mask
        assert message.context_post_prefill_keep_mask is None
