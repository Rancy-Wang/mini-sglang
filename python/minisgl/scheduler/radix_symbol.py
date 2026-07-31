from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

DropState = tuple[int, ...]
RadixSymbolKey = tuple[int, DropState]


def canonicalize_drop_state(dropped_ids: Iterable[int]) -> DropState:
    state = tuple(sorted({int(msg_id) for msg_id in dropped_ids}))
    if any(msg_id < 0 for msg_id in state):
        raise ValueError(f"Drop-state message IDs must be non-negative: {state}")
    return state


@dataclass
class RadixSymbolRegistry:
    """Intern exact (token_id, cumulative drop state) pairs into signed int64 symbols."""

    _symbols: dict[RadixSymbolKey, int] = field(default_factory=dict)
    _next_symbol: int = -1

    def intern(self, token_id: int, dropped_ids: Iterable[int]) -> int:
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError(f"Raw token ID must be non-negative before symbol injection: {token_id}")

        state = canonicalize_drop_state(dropped_ids)
        if len(state) == 0:
            return token_id

        key = (token_id, state)
        existing = self._symbols.get(key)
        if existing is not None:
            return existing

        if self._next_symbol < -(1 << 63):
            raise RuntimeError("Exhausted signed int64 Radix symbol namespace.")
        symbol = self._next_symbol
        self._next_symbol -= 1
        self._symbols[key] = symbol
        return symbol

    @property
    def size(self) -> int:
        return len(self._symbols)


def inject_radix_symbols(
    full_radix_ids: torch.Tensor,
    true_positions: torch.Tensor,
    state_starts: list[dict],
    registry: RadixSymbolRegistry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact full/active Radix keys while keeping one int64 key per real token."""

    if full_radix_ids.device.type != "cpu" or full_radix_ids.dtype != torch.int64:
        raise ValueError("full_radix_ids must be a CPU int64 tensor.")
    if full_radix_ids.ndim != 1 or true_positions.ndim != 1:
        raise ValueError("Radix IDs and true positions must be one-dimensional.")

    encoded = full_radix_ids.clone()
    seen_starts: set[int] = set()
    for start_meta in state_starts:
        raw_start = int(start_meta["raw_start"])
        if raw_start in seen_starts:
            raise ValueError(f"Duplicate Radix state-start position: {raw_start}")
        seen_starts.add(raw_start)
        if raw_start < 0 or raw_start >= len(encoded):
            raise ValueError(
                f"Radix state-start position {raw_start} is outside key length {len(encoded)}."
            )

        token_id = int(encoded[raw_start].item())
        encoded[raw_start] = registry.intern(token_id, start_meta.get("dropped_ids", ()))

    active_positions = true_positions.to(dtype=torch.int64, device="cpu")
    if len(active_positions) > 0:
        if bool(torch.any(active_positions < 0).item()) or bool(
            torch.any(active_positions >= len(encoded)).item()
        ):
            raise ValueError("true_positions contains an out-of-range full-token position.")
    active = encoded[active_positions].contiguous()
    return encoded, active
