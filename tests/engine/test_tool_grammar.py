from __future__ import annotations

from types import SimpleNamespace

import torch
from minisgl.engine.tool_grammar import ToolGrammarManager


class _Tokenizer:
    name_or_path = "AgenticQwen-test"


class _Compiler:
    def __init__(self, tokenizer_info):
        self.tokenizer_info = tokenizer_info

    def compile_structural_tag(self, structural_tag):
        return structural_tag


class _Matcher:
    def __init__(self, compiled, **kwargs):
        self.sequence = list(compiled["sequence"])
        self.kwargs = kwargs
        self.offset = 0

    def fill_next_token_bitmask(self, bitmask):
        bitmask.zero_()
        token = self.sequence[self.offset]
        bitmask[0, token // 32] = 1 << (token % 32)

    def accept_token(self, token):
        if token != self.sequence[self.offset]:
            return False
        self.offset += 1
        return True

    def is_terminated(self):
        return self.offset == len(self.sequence)


class _FakeXGrammar:
    GrammarCompiler = _Compiler
    GrammarMatcher = _Matcher

    class TokenizerInfo:
        calls = []

        @classmethod
        def from_huggingface(cls, tokenizer, **kwargs):
            cls.calls.append((tokenizer, kwargs))
            return (tokenizer, kwargs)

    def __init__(self):
        self.structural_calls = []
        self.matcher_count = 0

        outer = self

        class Matcher(_Matcher):
            def __init__(self, compiled, **kwargs):
                outer.matcher_count += 1
                super().__init__(compiled, **kwargs)

        self.GrammarMatcher = Matcher

    @staticmethod
    def allocate_token_bitmask(batch_size, vocab_size):
        return torch.empty((batch_size, (vocab_size + 31) // 32), dtype=torch.int32)

    def get_model_structural_tag(self, model, *, tools, tool_choice, reasoning):
        self.structural_calls.append(
            {
                "model": model,
                "tools": tools,
                "tool_choice": tool_choice,
                "reasoning": reasoning,
            }
        )
        if isinstance(tool_choice, dict):
            sequence = (5, 6, 0)
        elif tool_choice == "required":
            sequence = (3, 4, 0)
        else:
            sequence = (1, 2, 0)
        return {"sequence": sequence}

    @staticmethod
    def apply_token_bitmask_inplace(logits, bitmask, *, vocab_size, indices):
        for row in indices:
            for token in range(vocab_size):
                word = int(bitmask[row, token // 32].item()) & 0xFFFFFFFF
                if not (word & (1 << (token % 32))):
                    logits[row, token] = -torch.inf


class _Event:
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


def _descriptor(tool_choice="auto"):
    return {
        "version": 1,
        "model": "qwen_3",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
        "reasoning": False,
    }


class _Req:
    pass


def _req(uid, *, descriptor=None, committed=True):
    req = _Req()
    req.uid = uid
    req.sampling_params = SimpleNamespace(tool_grammar=descriptor)
    req.sample_is_committed = committed
    return req


def _sample(manager, reqs):
    prepared = manager.prepare(reqs)
    assert prepared is not None
    logits = torch.zeros((len(reqs), manager.vocab_size))
    logits[:, -1] = 100
    manager.apply(logits, prepared)
    return logits.argmax(dim=-1)


def test_no_tools_path_does_not_initialize_matcher_cache_executor_or_sync():
    class _ExplodingXGrammar:
        def __getattribute__(self, name):
            raise AssertionError(f"no-tools path accessed XGrammar.{name}")

    manager = ToolGrammarManager(_Tokenizer(), 8, {0}, xgrammar_module=_ExplodingXGrammar())
    req = _req(1)
    assert manager.prepare([req]) is None
    event = _Event()
    manager.accept_sampled_tokens([req], torch.tensor([7]), event)
    manager.discard(req)
    assert event.synchronizations == 0
    assert manager.compile_count == 0
    assert manager._executor is None
    assert manager._states == {}


def test_compiled_grammar_is_shared_but_matchers_and_reordered_rows_are_isolated():
    xgrammar = _FakeXGrammar()
    manager = ToolGrammarManager(_Tokenizer(), 8, {0}, xgrammar_module=xgrammar)
    first = _req(7, descriptor=_descriptor())
    plain = _req(8)
    second = _req(9, descriptor=_descriptor())

    tokens = _sample(manager, [first, plain, second])
    assert tokens.tolist() == [1, 7, 1]
    event = _Event()
    manager.accept_sampled_tokens([first, plain, second], tokens, event)

    tokens = _sample(manager, [second, first, plain])
    assert tokens.tolist() == [2, 2, 7]
    assert manager.compile_count == 1
    assert xgrammar.matcher_count == 2
    assert manager._states[first].matcher is not manager._states[second].matcher
    assert event.synchronizations == 2
    manager.shutdown()


def test_discard_and_uid_reuse_start_a_fresh_request_matcher():
    xgrammar = _FakeXGrammar()
    manager = ToolGrammarManager(_Tokenizer(), 8, {0}, xgrammar_module=xgrammar)
    old = _req(42, descriptor=_descriptor("required"))
    tokens = _sample(manager, [old])
    manager.accept_sampled_tokens([old], tokens, _Event())
    manager.discard(old)

    replacement = _req(42, descriptor=_descriptor("required"))
    assert _sample(manager, [replacement]).tolist() == [3]
    assert manager.compile_count == 1
    assert xgrammar.matcher_count == 2
    manager.shutdown()


def test_uncommitted_staged_sample_does_not_create_or_advance_grammar_state():
    xgrammar = _FakeXGrammar()
    manager = ToolGrammarManager(_Tokenizer(), 8, {0}, xgrammar_module=xgrammar)
    req = _req(1, descriptor=_descriptor(), committed=False)
    assert manager.prepare([req]) is None
    manager.accept_sampled_tokens([req], torch.tensor([7]), _Event())
    assert manager._states == {}

    req.sample_is_committed = True
    assert _sample(manager, [req]).tolist() == [1]
    manager.shutdown()


def test_required_and_forced_modes_are_forwarded_without_token_blacklists():
    xgrammar = _FakeXGrammar()
    manager = ToolGrammarManager(_Tokenizer(), 8, {0}, xgrammar_module=xgrammar)
    required = _req(1, descriptor=_descriptor("required"))
    forced_choice = {"type": "function", "function": {"name": "search"}}
    forced = _req(2, descriptor=_descriptor(forced_choice))

    assert _sample(manager, [required, forced]).tolist() == [3, 5]
    choices = [call["tool_choice"] for call in xgrammar.structural_calls]
    assert "required" in choices and forced_choice in choices
    assert all(call["model"] == "qwen_3" for call in xgrammar.structural_calls)
    manager.shutdown()
