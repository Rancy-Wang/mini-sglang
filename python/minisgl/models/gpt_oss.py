from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.moe.mxfp4 import GptOssMxfp4Experts
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


class GptOssAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=config.attention_bias,
        )
        local_q_heads = config.num_qo_heads // get_tp_info().size
        self.sinks = torch.empty(local_q_heads, dtype=torch.bfloat16)
        layer_type = config.layer_types[layer_id]
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported GPT-OSS layer type: {layer_type}")
        window = config.sliding_window - 1 if layer_type == "sliding_attention" else None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            sinks=self.sinks,
            sliding_window=window,
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * config.head_dim,
            config.hidden_size,
            has_bias=config.attention_bias,
        )

    @nvtx_annotate("GPT-OSS Attention")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj.forward(self.attn.forward(self.qkv_proj.forward(x)))

    def post_load(self) -> None:
        # BaseOP loading replaces tensors, while AttentionLayer is stateless and
        # otherwise retains the original meta tensor passed during construction.
        self.attn.sinks = self.sinks


class GptOssSparseMoeBlock(BaseOP):
    def __init__(self, config: ModelConfig):
        self.router = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=True,
        )
        self.experts = GptOssMxfp4Experts(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            top_k=config.num_experts_per_tok,
            alpha=config.hidden_act_alpha,
            limit=config.swiglu_limit,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.experts.forward(x, self.router.forward(x))


class GptOssDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = GptOssAttention(config, layer_id)
        self.mlp = GptOssSparseMoeBlock(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class GptOssModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [GptOssDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class GptOssForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        if config.quant_method != "mxfp4":
            raise ValueError("GPT-OSS requires quantization_config.quant_method='mxfp4'.")
        if len(config.layer_types) != config.num_layers:
            raise ValueError("GPT-OSS layer_types must contain one entry per decoder layer.")
        if config.sliding_window is None or config.sliding_window < 1:
            raise ValueError("GPT-OSS requires a positive sliding_window.")
        self.model = GptOssModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def post_load(self) -> None:
        for layer in self.model.layers.op_list:
            layer.self_attn.post_load()
            layer.mlp.experts.post_load()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["GptOssForCausalLM"]
