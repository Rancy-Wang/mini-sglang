from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.engine import _validate_runtime_config
from minisgl.models.config import ModelConfig
from minisgl.scheduler.scheduler import Scheduler
from minisgl.server.args import parse_args


def _gpt_oss_120b_hf_config():
    return SimpleNamespace(
        num_hidden_layers=36,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        hidden_size=2880,
        vocab_size=201088,
        intermediate_size=2880,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
        num_local_experts=128,
        num_experts_per_tok=4,
        norm_topk_prob=False,
        model_type="gpt_oss",
        architectures=["GptOssForCausalLM"],
        layer_types=["sliding_attention", "full_attention"] * 18,
        sliding_window=128,
        attention_bias=True,
        swiglu_limit=7.0,
        hidden_act_alpha=1.702,
        quantization_config={"quant_method": "mxfp4"},
        rope_scaling=None,
        rope_theta=150000.0,
        dtype=None,
        torch_dtype=None,
    )


def test_exact_tp4_cli_resolves_gpt_oss_dtype_before_engine_startup():
    hf_config = _gpt_oss_120b_hf_config()
    with patch("minisgl.utils.cached_load_hf_config", return_value=hf_config):
        args, run_shell = parse_args(
            [
                "--model",
                "/share/wangruoxi/models/gpt-oss-120b",
                "--tp",
                "4",
                "--port",
                "8000",
                "--attention-backend",
                "fi",
            ]
        )

    assert not run_shell
    assert args.dtype is torch.bfloat16
    assert args.tp_info == DistributedInfo(0, 4)
    assert args.server_port == 8000
    assert args.attention_backend == "fi"
    assert not args.drop_aware_eviction


def test_drop_aware_eviction_is_explicit_and_default_remains_disabled():
    hf_config = _gpt_oss_120b_hf_config()
    with patch("minisgl.utils.cached_load_hf_config", return_value=hf_config):
        default_args, _ = parse_args(["--model", "unused"])
        enabled_args, _ = parse_args(
            ["--model", "unused", "--enable-drop-aware-eviction"]
        )

    assert not default_args.drop_aware_eviction
    assert enabled_args.drop_aware_eviction


def test_drop_aware_eviction_rejects_incompatible_cache_key_mode_and_page_size():
    common = {
        "drop_aware_eviction": True,
        "cache_type": "radix",
        "radix_drop_key_mode": "delta-marker",
        "page_size": 1,
    }
    invalid = [
        ({**common, "cache_type": "chunk"}, "requires --cache-type radix"),
        ({**common, "radix_drop_key_mode": "symbol"}, "requires --radix-drop-key-mode"),
        ({**common, "page_size": 8}, "requires --page-size 1"),
    ]
    for values, message in invalid:
        with pytest.raises(ValueError, match=message):
            Scheduler(SimpleNamespace(**values))


def test_gpt_oss_120b_tp4_passes_capability_validation():
    runtime_config = SimpleNamespace(
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(0, 4),
        model_config=ModelConfig.from_hf(_gpt_oss_120b_hf_config()),
    )

    _validate_runtime_config(runtime_config)


def test_unresolved_dtype_is_rejected_before_communication_setup():
    runtime_config = SimpleNamespace(
        dtype=None,
        tp_info=DistributedInfo(0, 4),
        model_config=ModelConfig.from_hf(_gpt_oss_120b_hf_config()),
    )

    try:
        _validate_runtime_config(runtime_config)
    except TypeError as exc:
        assert "resolved torch.dtype" in str(exc)
    else:
        raise AssertionError("An unresolved dtype must not reach engine communication setup")
