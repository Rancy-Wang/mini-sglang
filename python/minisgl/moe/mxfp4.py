from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers.base import BaseOP


def _load_triton_kernels_abi() -> SimpleNamespace:
    try:
        from triton_kernels.matmul_ogs import (
            FlexCtx,
            FnSpecs,
            FusedActivation,
            PrecisionConfig,
            matmul_ogs,
        )
        from triton_kernels.numerics import InFlexData
        from triton_kernels.routing import routing
        from triton_kernels.swiglu import swiglu_fn
        from triton_kernels.tensor import (
            FP4,
            convert_layout,
            wrap_torch_tensor,
        )
        from triton_kernels.tensor_details import layout
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "GPT-OSS MXFP4 requires the triton_kernels 3.6 ABI "
            "(matmul_ogs, topk, FP4 layout conversion, and PrecisionConfig)."
        ) from exc

    return SimpleNamespace(
        FlexCtx=FlexCtx,
        FnSpecs=FnSpecs,
        FusedActivation=FusedActivation,
        PrecisionConfig=PrecisionConfig,
        matmul_ogs=matmul_ogs,
        InFlexData=InFlexData,
        routing=routing,
        swiglu_fn=swiglu_fn,
        FP4=FP4,
        convert_layout=convert_layout,
        wrap_torch_tensor=wrap_torch_tensor,
        layout=layout,
    )


def _swizzle_mxfp4(weight: torch.Tensor, scale: torch.Tensor, abi: SimpleNamespace):
    value_layout, value_options = abi.layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout, scale_options = abi.layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1,
        num_warps=8,
    )
    weight = abi.convert_layout(
        abi.wrap_torch_tensor(weight.transpose(-2, -1), dtype=abi.FP4),
        value_layout,
        **value_options,
    )
    scale = abi.convert_layout(
        abi.wrap_torch_tensor(scale.transpose(-2, -1)),
        scale_layout,
        **scale_options,
    )
    return weight, abi.InFlexData(), scale


def _route(router_logits: torch.Tensor, top_k: int, abi: SimpleNamespace):
    return abi.routing(
        router_logits,
        top_k,
        sm_first=False,
        expt_indx=None,
        simulated_ep=1,
        n_rows=None,
    )


class GptOssMxfp4Experts(BaseOP):
    """GPT-OSS packed MXFP4 experts for TP with EP fixed to one."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        alpha: float = 1.702,
        limit: float = 7.0,
    ) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("GPT-OSS MXFP4 requires CUDA compute capability SM80 or newer.")
        self._abi = _load_triton_kernels_abi()
        tp_info = get_tp_info()
        blocks_per_rank = math.ceil(intermediate_size / 32 / tp_info.size)
        self.local_intermediate_size = blocks_per_rank * 32
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.alpha = alpha
        self.limit = limit
        self.tp_size = tp_info.size
        self._comm = DistributedCommunicator()

        local_size = self.local_intermediate_size
        self.w13_weight = torch.empty(
            num_experts,
            2 * local_size,
            hidden_size // 2,
            dtype=torch.uint8,
        )
        self.w13_weight_scale = torch.empty(
            num_experts,
            2 * local_size,
            hidden_size // 32,
            dtype=torch.uint8,
        )
        self.w13_weight_bias = torch.empty(
            num_experts,
            2 * local_size,
            dtype=torch.bfloat16,
        )
        self.w2_weight = torch.empty(
            num_experts,
            hidden_size,
            local_size // 2,
            dtype=torch.uint8,
        )
        self.w2_weight_scale = torch.empty(
            num_experts,
            hidden_size,
            local_size // 32,
            dtype=torch.uint8,
        )
        self.w2_weight_bias = torch.empty(
            num_experts,
            hidden_size,
            dtype=torch.bfloat16,
        )
        self._loaded = False

    def post_load(self) -> None:
        if self._loaded:
            return
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            if getattr(self, name).dtype != torch.uint8:
                raise TypeError(f"{name} must remain packed uint8 through checkpoint loading")

        self.w13_weight_bias = self.w13_weight_bias.float()
        self.w2_weight_bias = self.w2_weight_bias.float()
        w13, w13_flex, w13_scale = _swizzle_mxfp4(
            self.w13_weight,
            self.w13_weight_scale,
            self._abi,
        )
        w2, w2_flex, w2_scale = _swizzle_mxfp4(
            self.w2_weight,
            self.w2_weight_scale,
            self._abi,
        )
        self._w13 = w13
        self._w2 = w2
        self._w13_precision = self._abi.PrecisionConfig(
            weight_scale=w13_scale,
            flex_ctx=self._abi.FlexCtx(rhs_data=w13_flex),
        )
        self._w2_precision = self._abi.PrecisionConfig(
            weight_scale=w2_scale,
            flex_ctx=self._abi.FlexCtx(rhs_data=w2_flex),
        )
        del self.w13_weight
        del self.w2_weight
        self._loaded = True

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("GPT-OSS MXFP4 experts must be post-loaded before execution")
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError("GPT-OSS MXFP4 activations must be bfloat16")

        routing_data, gather_indices, scatter_indices = _route(
            router_logits,
            self.top_k,
            self._abi,
        )
        tokens = hidden_states.shape[0]
        intermediate = torch.empty(
            (1, tokens * self.top_k, self.local_intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output = torch.empty(
            (1, tokens, self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        activation = self._abi.FusedActivation(
            self._abi.FnSpecs(
                "swiglu",
                self._abi.swiglu_fn,
                ("alpha", "limit"),
            ),
            (self.alpha, self.limit),
            2,
        )
        self._abi.matmul_ogs(
            hidden_states,
            self._w13,
            self.w13_weight_bias,
            routing_data,
            gather_indx=gather_indices,
            precision_config=self._w13_precision,
            fused_activation=activation,
            y=intermediate,
        )
        self._abi.matmul_ogs(
            intermediate.view(tokens * self.top_k, self.local_intermediate_size),
            self._w2,
            self.w2_weight_bias,
            routing_data,
            scatter_indx=scatter_indices,
            precision_config=self._w2_precision,
            gammas=routing_data.gate_scal,
            y=output,
        )
        output = output.view(tokens, self.hidden_size)
        return self._comm.all_reduce(output) if self.tp_size > 1 else output


__all__ = ["GptOssMxfp4Experts"]
