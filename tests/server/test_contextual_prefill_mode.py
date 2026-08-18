import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from minisgl.server.api_server import FrontendManager
from minisgl.server.args import parse_args


def _parse_mode(*extra_args):
    args, _ = parse_args(
        ["--model", "unused", "--dtype", "float32", *extra_args]
    )
    return args.contextual_prefill_mode


def _frontend(mode, hit_ratios):
    manager = object.__new__(FrontendManager)
    manager.config = SimpleNamespace(contextual_prefill_mode=mode)
    manager.new_user = MagicMock(side_effect=iter(range(100, 200)))
    manager.send_one = AsyncMock()
    manager.wait_for_warmup = AsyncMock(
        side_effect=[SimpleNamespace(hit_ratio=ratio) for ratio in hit_ratios]
    )
    return manager


def _run_warmup(manager, drop_rule=None):
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    asyncio.run(
        manager.run_contextual_warmup(
            messages,
            drop_rule
            or {"type": "message_drop", "drop_messages": {"1": [0]}},
            enable_thinking=None,
            reasoning_effort=None,
            tools=None,
            tool_choice=None,
        )
    )


def test_contextual_prefill_defaults_to_backend_neutral_mask():
    assert _parse_mode() == "mask"
    assert _parse_mode("--contextual-prefill-mode", "mask") == "mask"
    assert _parse_mode("--contextual-prefill-mode", "staged") == "staged"


@pytest.mark.parametrize("legacy_mode", ["flashinfer-mask", "flashattention-mask"])
def test_backend_specific_mask_modes_are_deprecated_aliases(legacy_mode):
    with pytest.warns(FutureWarning, match="deprecated"):
        assert _parse_mode("--contextual-prefill-mode", legacy_mode) == "mask"


def test_mask_mode_sends_one_full_stream_context_warmup():
    manager = _frontend("mask", [0.0])

    _run_warmup(manager)

    manager.send_one.assert_awaited_once()
    manager.wait_for_warmup.assert_awaited_once_with(100)
    msg = manager.send_one.await_args.args[0]
    assert msg.use_context_mask
    assert msg.is_warmup
    assert msg.target_msg_id == 3
    assert len(msg.text) == 3


def test_staged_mode_keeps_high_hit_short_circuit():
    manager = _frontend("staged", [0.95])

    _run_warmup(manager)

    manager.send_one.assert_awaited_once()
    msg = manager.send_one.await_args.args[0]
    assert not msg.use_context_mask


def test_staged_mode_keeps_strictly_low_hit_message_fallback():
    manager = _frontend("staged", [0.949, 1.0, 1.0])

    _run_warmup(manager)

    assert manager.send_one.await_count == 3
    sent = [call.args[0] for call in manager.send_one.await_args_list]
    assert [len(msg.text) for msg in sent] == [3, 1, 2]
    assert all(not msg.use_context_mask for msg in sent)
    assert [msg.target_msg_id for msg in sent] == [2, 0, 1]


def test_future_only_drop_schedule_skips_contextual_warmup():
    manager = _frontend("mask", [])

    _run_warmup(
        manager,
        {"type": "message_drop", "drop_messages": {"3": [0]}},
    )

    manager.new_user.assert_not_called()
    manager.send_one.assert_not_awaited()
    manager.wait_for_warmup.assert_not_awaited()
