from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from minisgl.core import SamplingParams
from minisgl.kvcache.base import BaseCacheHandle, SizeInfo
from minisgl.message import TokenizeMsg


def _load_module(name: str, rel_path: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / rel_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_cache_mod = _load_module("_agentic_scheduler_cache", "scheduler/cache.py")
_sched_utils_mod = _load_module("_agentic_scheduler_utils", "scheduler/utils.py")
_tokenize_mod = _load_module("_agentic_tokenize", "tokenizer/tokenize.py")
CacheManager = _cache_mod.CacheManager
PendingReq = _sched_utils_mod.PendingReq
TokenizeManager = _tokenize_mod.TokenizeManager


@dataclass(frozen=True)
class _FakeHandle(BaseCacheHandle):
    pass


class _FakePrefixManager:
    def __init__(self, full_match_indices: torch.Tensor):
        self.full_match_indices = full_match_indices
        self.last_insert_ids: torch.Tensor | None = None
        self.last_insert_indices: torch.Tensor | None = None
        self.size_info = SizeInfo(evictable_size=1024, protected_size=0)

    def match_prefix(self, input_ids: torch.Tensor):
        _ = input_ids
        return _FakeHandle(cached_len=len(self.full_match_indices)), self.full_match_indices.clone()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False):
        _ = handle, unlock

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        self.last_insert_ids = input_ids.clone()
        self.last_insert_indices = indices.clone()
        return 0

    def evict(self, size: int) -> torch.Tensor:
        _ = size
        return torch.empty(0, dtype=torch.int32)

    def check_integrity(self) -> None:
        return


class DummyTokenizer:
    def encode(self, text: str, return_tensors: str | None = None, add_special_tokens: bool = True):
        _ = add_special_tokens
        ids = [ord(ch) for ch in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.int32)
        return ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ):
        _ = kwargs
        assert tokenize
        chunks = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if isinstance(msg.get("tool_calls"), list):
                content += "|tool_calls=" + json.dumps(msg["tool_calls"], sort_keys=True)
            if msg.get("tool_call_id") is not None:
                content += f"|tool_call_id={msg['tool_call_id']}"
            chunks.append(f"[{role}]{content}[/]")
        if add_generation_prompt:
            chunks.append("[assistant]")
        return self.encode("".join(chunks), add_special_tokens=False)


def _decode_radix_key(value: int) -> tuple[int, int]:
    unsigned = value if value >= 0 else value + (1 << 64)
    token_id = unsigned & 0xFFFFFFFF
    drop_mask = unsigned >> 32
    return token_id, drop_mask


def test_agentic_message_boundaries_and_drop_mask_encoding():
    tokenizer = DummyTokenizer()
    manager = TokenizeManager(tokenizer)  # type: ignore[arg-type]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather by city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are a tool assistant."},
        {"role": "user", "content": "What is weather in Beijing?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'},
                }
            ],
        },
        {
            "role": "tool",
            "name": "get_weather",
            "tool_call_id": "call_1",
            "content": '{"temp":26}',
        },
    ]
    req = TokenizeMsg(
        uid=1,
        text=messages,
        sampling_params=SamplingParams(max_tokens=16),
        target_msg_id=4,
        drop_message={3: [3]},
        tools=tools,
        stop=["</tool_call>"],
    )

    result = manager.tokenize([req])[0]
    meta = result.message_meta or {}
    starts = {item["msg_id"]: item for item in meta.get("message_starts", [])}

    # system/user/assistant/tool + next assistant generation-prompt start
    for msg_id in [0, 1, 2, 3, 4]:
        assert msg_id in starts

    # The dropped tool message (id=3) should not appear in active true positions for target 4.
    assert starts[3]["raw_start"] not in set(result.true_positions.tolist())

    # Ensure composite radix key encoding is still applied to assistant/tool boundaries.
    for msg_id in [2, 3, 4]:
        start = starts[msg_id]["raw_start"]
        encoded = int(result.radix_match_ids[start].item())
        _, drop_mask = _decode_radix_key(encoded)
        assert drop_mask == starts[msg_id]["drop_mask"]

    # target=4 with drop_message={3:[3]} means generation prompt has bit-3 set.
    assert starts[4]["drop_mask"] & (1 << 3)
    assert result.stop_token_seqs is not None and len(result.stop_token_seqs) >= 1


def test_cache_full_active_reconstruction_with_drop_mask():
    device = torch.device("cpu")
    cache = CacheManager(device=device, num_pages=256, type="naive")
    fake_manager = _FakePrefixManager(torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device))
    cache.manager = fake_manager

    pending = PendingReq(
        uid=0,
        input_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),
        true_positions=torch.arange(5, dtype=torch.int32),
        radix_input_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        radix_match_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64),
        sampling_params=SamplingParams(max_tokens=1),
        prefix_keep_mask=torch.tensor([1, 0, 1, 1], dtype=torch.int32),
    )
    match = cache.match_req(pending)
    assert match.full_match_indices.tolist() == [1, 2, 3, 4]
    assert match.active_match_indices.tolist() == [1, 3, 4]

    full_radix_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    active_indices = torch.tensor([1, 3, 4, 9], dtype=torch.int32, device=device)
    active_true_positions = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=device)
    cache.free_and_cache_finished_req(
        old_handle=match.handle,
        full_radix_ids=full_radix_ids,
        active_indices=active_indices,
        active_true_positions=active_true_positions,
        initial_full_match_indices=match.full_match_indices.clone(),
        initial_active_cached_len=match.active_cached_len,
        prefix_keep_mask=pending.prefix_keep_mask,
    )
    assert fake_manager.last_insert_ids is not None
    assert fake_manager.last_insert_indices is not None
    assert fake_manager.last_insert_ids.tolist() == [10, 20, 30, 40, 50]
    assert fake_manager.last_insert_indices.tolist() == [1, 2, 3, 4, 9]
