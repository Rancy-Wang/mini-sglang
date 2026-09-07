from dataclasses import replace

import pytest
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager
from test_template_single_pass import _SinglePassTokenizer


def test_reference_uses_one_full_template_even_when_prefix_is_not_concatenable(monkeypatch):
    tok = _SinglePassTokenizer()
    # Every rendering depends on the TOTAL message count, so re-rendering prefixes
    # cannot reproduce the canonical token stream.
    tok.chat_template = '{{ messages|length }}' + tok.chat_template
    manager = TokenizeManager(tok)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference must not compile Radix keys')
    monkeypatch.setattr(manager, '_compile_delta_layout', forbidden)
    messages = [{'role':'user','content':'same'}, {'role':'assistant','content':'same'},
                {'role':'user','content':'last'}]
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams(max_tokens=2),
                      target_msg_id=3, drop_message={1:[0]}, staged_reference=True)
    result = manager.tokenize([msg])[0]
    assert result.staged_reference
    assert result.input_ids.tolist() == list(map(ord, '3<user>same<assistant>same<user>last<assistant>'))
    assert result.raw_positions.tolist() == list(range(len(result.input_ids)))
    assert tok.apply_calls == tok.encode_calls == 1
    assert result.radix_match_ids is None and result.reposition_layout is None
    assert result.full_token_visible_until is None
    other = manager.tokenize([replace(msg, drop_message={1:[1]})])[0]
    assert other.input_ids.tolist() == result.input_ids.tolist()
    assert other.drop_position_ranges.tolist() != result.drop_position_ranges.tolist()


@pytest.mark.parametrize('drop', [None, {}, {99:[0]}])
def test_no_effective_drop_keeps_ordinary_fast_path(monkeypatch, drop):
    tok = _SinglePassTokenizer()
    manager = TokenizeManager(tok)
    msg = TokenizeMsg(uid=1, text=[{'role':'user','content':'hello'}],
                      sampling_params=SamplingParams(), drop_message=drop, staged_reference=True)
    result = manager.tokenize([msg])[0]
    assert not result.staged_reference
    assert result.input_ids.tolist() == list(map(ord, '<user>hello<assistant>'))
    assert tok.encode_calls == tok.apply_calls == 1
