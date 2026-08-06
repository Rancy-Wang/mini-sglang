from types import SimpleNamespace

from minisgl.models.config import ModelConfig


def _gpt_oss_hf_config(**overrides):
    values = {
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2880,
        "vocab_size": 201088,
        "intermediate_size": 2880,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": False,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "norm_topk_prob": False,
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "layer_types": ["sliding_attention", "full_attention"] * 12,
        "sliding_window": 128,
        "attention_bias": True,
        "swiglu_limit": 7.0,
        "hidden_act_alpha": 1.702,
        "quantization_config": {"quant_method": "mxfp4"},
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        "rope_theta": 150000.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gpt_oss_config_fields_and_moe_detection():
    config = ModelConfig.from_hf(_gpt_oss_hf_config())

    assert config.is_gpt_oss
    assert config.is_moe
    assert config.num_experts == 32
    assert config.moe_intermediate_size == 2880
    assert config.layer_types == ("sliding_attention", "full_attention") * 12
    assert config.sliding_window == 128
    assert config.attention_bias is True
    assert config.swiglu_limit == 7.0
    assert config.hidden_act_alpha == 1.702
    assert config.quant_method == "mxfp4"
    assert config.rotary_config.base == 150000.0
    assert config.rotary_config.scaling["rope_type"] == "yarn"
    assert config.rotary_config.scaling["original_max_position_embeddings"] == 4096


def test_gpt_oss_v5_rope_parameters_use_model_default_theta():
    config = ModelConfig.from_hf(
        _gpt_oss_hf_config(
            rope_scaling=None,
            rope_theta=None,
            default_theta=150000.0,
            rope_parameters={
                "rope_type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": 4096,
            },
        )
    )

    assert config.rotary_config.base == 150000.0
    assert config.rotary_config.scaling["rope_theta"] == 150000.0
    assert config.rotary_config.scaling["factor"] == 32.0


def test_model_type_name_does_not_make_dense_model_moe():
    config = ModelConfig.from_hf(
        _gpt_oss_hf_config(
            model_type="dense_model_with_moe_in_name",
            architectures=["DenseForCausalLM"],
            num_local_experts=0,
        )
    )

    assert not config.is_moe
    assert not config.is_gpt_oss
