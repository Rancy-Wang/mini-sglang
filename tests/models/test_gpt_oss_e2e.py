from types import SimpleNamespace
from unittest.mock import MagicMock

import minisgl.engine.engine as engine_module
import minisgl.models.gpt_oss as gpt_oss
import pytest
import torch
from minisgl.layers.embedding import ParallelLMHead


def _attention_config(layer_type):
    return SimpleNamespace(
        hidden_size=64,
        head_dim=8,
        num_qo_heads=8,
        num_kv_heads=2,
        attention_bias=True,
        layer_types=(layer_type,),
        sliding_window=128,
        rotary_config=object(),
    )


def test_attention_uses_local_sinks_and_alternating_window(monkeypatch):
    monkeypatch.setattr(gpt_oss, "get_tp_info", lambda: SimpleNamespace(size=2))
    monkeypatch.setattr(gpt_oss, "LinearQKVMerged", MagicMock())
    monkeypatch.setattr(gpt_oss, "LinearOProj", MagicMock())
    attention_layer = MagicMock()
    monkeypatch.setattr(gpt_oss, "AttentionLayer", attention_layer)

    sliding = gpt_oss.GptOssAttention(_attention_config("sliding_attention"), 0)
    full = gpt_oss.GptOssAttention(_attention_config("full_attention"), 0)

    assert sliding.sinks.shape == (4,)
    assert sliding.sinks.dtype == torch.bfloat16
    assert attention_layer.call_args_list[0].kwargs["sinks"] is sliding.sinks
    assert attention_layer.call_args_list[0].kwargs["sliding_window"] == 127
    assert attention_layer.call_args_list[1].kwargs["sinks"] is full.sinks
    assert attention_layer.call_args_list[1].kwargs["sliding_window"] is None


def test_attention_post_load_rebinds_replaced_sink_tensor(monkeypatch):
    monkeypatch.setattr(gpt_oss, "get_tp_info", lambda: SimpleNamespace(size=2))
    monkeypatch.setattr(gpt_oss, "LinearQKVMerged", MagicMock())
    monkeypatch.setattr(gpt_oss, "LinearOProj", MagicMock())
    backend_layer = SimpleNamespace(sinks=None)
    monkeypatch.setattr(gpt_oss, "AttentionLayer", lambda **kwargs: backend_layer)
    attention = gpt_oss.GptOssAttention(_attention_config("sliding_attention"), 0)
    replacement = torch.empty(4, dtype=torch.bfloat16)

    attention.sinks = replacement
    attention.post_load()

    assert attention.attn.sinks is replacement


def test_model_rejects_non_mxfp4_before_allocating_layers():
    config = SimpleNamespace(
        quant_method="bf16",
        layer_types=("sliding_attention",),
        num_layers=1,
        sliding_window=128,
    )

    with pytest.raises(ValueError, match="mxfp4"):
        gpt_oss.GptOssForCausalLM(config)


def test_engine_dummy_and_loaded_weights_follow_template_dtype(monkeypatch):
    engine = object.__new__(engine_module.Engine)
    engine.device = torch.device("cpu")
    engine.model = SimpleNamespace(
        state_dict=lambda: {
            "packed": torch.empty(4, dtype=torch.uint8),
            "bias": torch.empty(4, dtype=torch.bfloat16),
        }
    )
    dummy = engine._load_weight_state_dict(
        SimpleNamespace(use_dummy_weight=True, model_path="unused")
    )

    assert dummy["packed"].dtype == torch.uint8
    assert torch.count_nonzero(dummy["packed"]) == 0
    assert dummy["bias"].dtype == torch.bfloat16

    monkeypatch.setattr(
        engine_module,
        "load_weight",
        lambda model_path, device: iter(
            (
                ("packed", torch.ones(4, dtype=torch.uint8)),
                ("bias", torch.ones(4)),
                ("lm_head.weight", torch.ones(4)),
            )
        ),
    )
    loaded = engine._load_weight_state_dict(
        SimpleNamespace(use_dummy_weight=False, model_path="checkpoint")
    )

    assert loaded["packed"].dtype == torch.uint8
    assert loaded["bias"].dtype == torch.bfloat16
    assert loaded["lm_head.weight"].dtype == torch.float32


def test_tied_lm_head_consumes_redundant_checkpoint_weight():
    lm_head = object.__new__(ParallelLMHead)
    lm_head.tied_embedding = object()
    state_dict = {
        "lm_head.weight": torch.ones(4),
        "unexpected.weight": torch.ones(1),
    }

    lm_head.load_state_dict(state_dict, prefix="lm_head", _internal=True)

    assert "lm_head.weight" not in state_dict
    assert list(state_dict) == ["unexpected.weight"]
