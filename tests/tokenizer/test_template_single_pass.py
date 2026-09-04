from __future__ import annotations

import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
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

    def apply_chat_template(self, *args, **kwargs):
        self.apply_calls += 1
        raise AssertionError("The structured path must use the single traced render.")

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


def test_structured_drop_and_reposition_render_and_encode_exactly_once() -> None:
    tokenizer = _SinglePassTokenizer()
    manager = TokenizeManager(tokenizer)
    result = manager._chat_tokenize(
        TokenizeMsg(
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
    )

    expected = "<user>old<assistant>answer<user>new<assistant>"
    assert result.reposition_input_ids is not None
    assert result.reposition_input_ids.tolist() == [ord(char) for char in expected]
    assert result.tokenize_invocations == 1
    assert result.chat_template_invocations == 1
    assert tokenizer.encode_calls == 1
    assert tokenizer.apply_calls == 0
    assert result.message_meta["gen_prompt_start"] == len(expected) - len("<assistant>")
    assert torch.any(result.reposition_layout.records[:, 0] == 1)
