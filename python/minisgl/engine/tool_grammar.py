from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import torch

if TYPE_CHECKING:
    from minisgl.core import Req
    from transformers import PreTrainedTokenizerBase


@dataclass
class _GrammarState:
    key: str
    matcher: Any | None = None
    future: Future[torch.Tensor] | None = None
    last_mask: torch.Tensor | None = None
    terminated: bool = False


class ToolGrammarManager:
    """Per-request XGrammar state with compiled-schema reuse and CPU/GPU overlap."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int,
        stop_token_ids: Iterable[int],
        *,
        xgrammar_module: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.stop_token_ids = sorted({int(token) for token in stop_token_ids})
        self._xgrammar = xgrammar_module
        self._executor: ThreadPoolExecutor | None = None
        self._runtime_lock = threading.Lock()
        self._compiler: Any | None = None
        self._compiled: dict[str, Any] = {}
        self._states: dict[Req, _GrammarState] = {}
        self.compile_count = 0

    @staticmethod
    def _descriptor(req: Req) -> dict[str, Any] | None:
        params = req.sampling_params
        descriptor = None if params is None else params.tool_grammar
        return descriptor if isinstance(descriptor, dict) else None

    def _key(self, descriptor: dict[str, Any]) -> str:
        tokenizer_name = str(getattr(self.tokenizer, "name_or_path", ""))
        return json.dumps(
            {
                "tokenizer": tokenizer_name,
                "vocab_size": self.vocab_size,
                "stop_token_ids": self.stop_token_ids,
                "grammar": descriptor,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _get_xgrammar(self) -> Any:
        if self._xgrammar is None:
            import xgrammar

            self._xgrammar = xgrammar
        return self._xgrammar

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            workers = max(2, min(8, os.cpu_count() or 2))
            self._executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="minisgl-tool-grammar",
            )
        return self._executor

    def _compile(self, key: str, descriptor: dict[str, Any]) -> Any:
        with self._runtime_lock:
            compiled = self._compiled.get(key)
            if compiled is not None:
                return compiled
            xgrammar = self._get_xgrammar()
            if self._compiler is None:
                tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
                    self.tokenizer,
                    vocab_size=self.vocab_size,
                    stop_token_ids=self.stop_token_ids or None,
                )
                self._compiler = xgrammar.GrammarCompiler(tokenizer_info)
            structural_tag = xgrammar.get_model_structural_tag(
                descriptor["model"],
                tools=descriptor["tools"],
                tool_choice=descriptor["tool_choice"],
                reasoning=bool(descriptor["reasoning"]),
            )
            compiled = self._compiler.compile_structural_tag(structural_tag)
            self._compiled[key] = compiled
            self.compile_count += 1
            return compiled

    def _new_mask(self) -> torch.Tensor:
        xgrammar = self._get_xgrammar()
        # XGrammar fills CPU masks. The batched copy in ``apply`` is pinned on
        # CUDA hosts; keeping this mask pageable supports CPU-only validation.
        return xgrammar.allocate_token_bitmask(1, self.vocab_size)

    def _initialize(self, state: _GrammarState, descriptor: dict[str, Any]) -> torch.Tensor:
        xgrammar = self._get_xgrammar()
        compiled = self._compile(state.key, descriptor)
        state.matcher = xgrammar.GrammarMatcher(
            compiled,
            override_stop_tokens=self.stop_token_ids or None,
            terminate_without_stop_token=False,
        )
        bitmask = self._new_mask()
        state.matcher.fill_next_token_bitmask(bitmask)
        state.last_mask = bitmask
        return bitmask

    def _ensure(self, req: Req, descriptor: dict[str, Any]) -> _GrammarState:
        key = self._key(descriptor)
        state = self._states.get(req)
        if state is not None and state.key == key:
            return state
        if state is not None:
            self._states.pop(req, None)
        state = _GrammarState(key=key)
        state.future = self._get_executor().submit(self._initialize, state, descriptor)
        self._states[req] = state
        return state

    def prepare(self, reqs: Sequence[Req]) -> tuple[Req, ...] | None:
        constrained: list[Req] = []
        for req in reqs:
            descriptor = self._descriptor(req)
            if descriptor is None or not req.sample_is_committed:
                continue
            self._ensure(req, descriptor)
            constrained.append(req)
        return tuple(reqs) if constrained else None

    def apply(self, logits: torch.Tensor, reqs: Sequence[Req]) -> None:
        rows: list[tuple[int, torch.Tensor]] = []
        for index, req in enumerate(reqs):
            state = self._states.get(req)
            if state is None:
                continue
            if state.future is None:
                raise RuntimeError("Tool grammar state has no pending token mask.")
            rows.append((index, state.future.result()))
        if not rows:
            return

        width = rows[0][1].shape[1]
        bitmask = torch.full(
            (len(reqs), width),
            -1,
            dtype=torch.int32,
            pin_memory=logits.device.type == "cuda",
        )
        indices: list[int] = []
        for index, row in rows:
            bitmask[index].copy_(row[0])
            indices.append(index)
        device_mask = bitmask.to(logits.device, non_blocking=True)
        self._get_xgrammar().apply_token_bitmask_inplace(
            logits,
            device_mask,
            vocab_size=self.vocab_size,
            indices=indices,
        )

    def _accept(
        self,
        state: _GrammarState,
        previous: Future[torch.Tensor],
        token: int,
        ready_event: Any | None,
    ) -> torch.Tensor:
        previous.result()
        if ready_event is not None:
            ready_event.synchronize()
        if state.terminated:
            assert state.last_mask is not None
            return state.last_mask
        assert state.matcher is not None
        if not state.matcher.accept_token(token):
            raise RuntimeError(f"XGrammar rejected a sampled token that passed its mask: {token}.")
        if state.matcher.is_terminated():
            state.terminated = True
            assert state.last_mask is not None
            return state.last_mask
        bitmask = self._new_mask()
        state.matcher.fill_next_token_bitmask(bitmask)
        state.last_mask = bitmask
        return bitmask

    def accept_sampled_tokens(
        self,
        reqs: Sequence[Req],
        tokens: torch.Tensor,
        ready_event: Any | None,
    ) -> None:
        for index, req in enumerate(reqs):
            state = self._states.get(req)
            if state is None or not req.sample_is_committed:
                continue
            if state.future is None:
                raise RuntimeError("Tool grammar state advanced without a token mask.")
            previous = state.future

            def accept_one(
                *,
                grammar_state: _GrammarState = state,
                prior: Future[torch.Tensor] = previous,
                token_index: int = index,
            ) -> torch.Tensor:
                if ready_event is not None:
                    ready_event.synchronize()
                token = int(tokens[token_index].item())
                return self._accept(grammar_state, prior, token, None)

            state.future = self._get_executor().submit(accept_one)

    def discard(self, req: Req) -> None:
        self._states.pop(req, None)

    def shutdown(self) -> None:
        states, self._states = self._states, {}
        for state in states.values():
            if state.future is not None:
                state.future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
