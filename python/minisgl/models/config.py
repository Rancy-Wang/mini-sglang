from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from transformers import PretrainedConfig


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_model_dtype(
    config: PretrainedConfig | dict[str, Any],
    requested_dtype: str | torch.dtype,
) -> torch.dtype:
    """Resolve a CLI/config dtype before any engine or communication setup."""
    text_config = _config_value(config, "text_config") or config
    architectures = list(
        _config_value(config, "architectures")
        or _config_value(text_config, "architectures")
        or ()
    )
    model_type = _config_value(text_config, "model_type", "")
    quantization_config = _config_value(config, "quantization_config") or _config_value(
        text_config, "quantization_config"
    )
    quant_method = (
        quantization_config.get("quant_method")
        if isinstance(quantization_config, dict)
        else None
    )

    is_gpt_oss = model_type == "gpt_oss" or "GptOssForCausalLM" in architectures
    if is_gpt_oss and quant_method == "mxfp4":
        # Match SGLang's model-specific override: the packed MXFP4 kernels use
        # BF16 activations even though the official GPT-OSS config omits dtype.
        return torch.bfloat16

    config_dtype = _config_value(text_config, "dtype") or _config_value(
        text_config, "torch_dtype"
    )
    if config_dtype is None and text_config is not config:
        config_dtype = _config_value(config, "dtype") or _config_value(
            config, "torch_dtype"
        )
    if isinstance(config_dtype, str):
        config_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(config_dtype.lower())
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(requested_dtype, torch.dtype):
        return requested_dtype
    if not isinstance(requested_dtype, str):
        raise ValueError(f"Unknown dtype: {requested_dtype!r}")

    requested_dtype = requested_dtype.lower()
    if requested_dtype == "auto":
        return torch.float16 if config_dtype == torch.float32 else config_dtype
    try:
        return _STR_DTYPE_TO_TORCH_DTYPE[requested_dtype]
    except KeyError as exc:
        raise ValueError(f"Unknown dtype: {requested_dtype}") from exc


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    attention_bias: bool = False
    swiglu_limit: float = 0.0
    hidden_act_alpha: float = 1.702
    quant_method: str | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def is_gpt_oss(self) -> bool:
        return self.model_type == "gpt_oss" or "GptOssForCausalLM" in self.architectures

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(
            config,
            "moe_intermediate_size",
            config.intermediate_size if num_experts else 0,
        )
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = list(getattr(config, "architectures", None) or ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_scaling is None and isinstance(rope_parameters, dict):
            rope_scaling = dict(rope_parameters)
        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
            if "rope_type" not in rope_scaling and "type" in rope_scaling:
                rope_scaling["rope_type"] = rope_scaling["type"]
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(config, "default_theta", 10000.0)
        if isinstance(rope_scaling, dict):
            rope_scaling.setdefault("rope_theta", rope_theta)

        layer_types = tuple(getattr(config, "layer_types", ()) or ())
        quantization_config = getattr(config, "quantization_config", None)
        quant_method = (
            quantization_config.get("quant_method")
            if isinstance(quantization_config, dict)
            else None
        )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            layer_types=layer_types,
            sliding_window=getattr(config, "sliding_window", None),
            attention_bias=getattr(config, "attention_bias", False),
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0)),
            hidden_act_alpha=float(getattr(config, "hidden_act_alpha", 1.702)),
            quant_method=quant_method,
        )
