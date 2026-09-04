from types import SimpleNamespace
from unittest.mock import MagicMock

import minisgl.moe.mxfp4 as mxfp4
import torch
from minisgl.distributed import DistributedInfo
from minisgl.moe.mxfp4 import GptOssMxfp4Experts, _route, _swizzle_mxfp4


def test_route_uses_official_normalized_routing():
    result = (SimpleNamespace(gate_scal=torch.tensor([0.75, 0.25])), "gather", "scatter")
    abi = SimpleNamespace(
        routing=MagicMock(return_value=result),
    )

    routing, gather, scatter = _route(torch.empty(1, 4), 2, abi)

    assert (routing, gather, scatter) == result
    assert routing.gate_scal.sum().item() == 1.0
    abi.routing.assert_called_once()
    assert abi.routing.call_args.args[1] == 2
    assert abi.routing.call_args.kwargs == {
        "sm_first": False,
        "expt_indx": None,
        "simulated_ep": 1,
        "n_rows": None,
    }


def test_swizzle_uses_mxfp4_axis_one_layouts():
    value_layout = object()
    scale_layout = object()
    wrapped_weight = object()
    wrapped_scale = object()
    converted_weight = object()
    converted_scale = object()
    abi = SimpleNamespace(
        layout=SimpleNamespace(
            make_default_matmul_mxfp4_w_layout=MagicMock(return_value=(value_layout, {})),
            make_default_matmul_mxfp4_w_scale_layout=MagicMock(
                return_value=(scale_layout, {})
            ),
        ),
        wrap_torch_tensor=MagicMock(side_effect=[wrapped_weight, wrapped_scale]),
        convert_layout=MagicMock(side_effect=[converted_weight, converted_scale]),
        FP4=object(),
        InFlexData=MagicMock(return_value="flex"),
    )
    weight = torch.empty(1, 8, 4, dtype=torch.uint8)
    scale = torch.empty(1, 8, 1, dtype=torch.uint8)

    result = _swizzle_mxfp4(weight, scale, abi)

    assert result == (converted_weight, "flex", converted_scale)
    abi.layout.make_default_matmul_mxfp4_w_layout.assert_called_once_with(mx_axis=1)
    abi.layout.make_default_matmul_mxfp4_w_scale_layout.assert_called_once_with(
        mx_axis=1,
        num_warps=8,
    )
    assert abi.wrap_torch_tensor.call_args_list[0].kwargs["dtype"] is abi.FP4


def test_forward_uses_swiglu_and_two_matmul_ogs_calls(monkeypatch):
    routing = SimpleNamespace(gate_scal=torch.ones(2))
    monkeypatch.setattr(mxfp4, "_route", lambda logits, top_k, abi: (routing, "g", "s"))
    abi = SimpleNamespace(
        FusedActivation=MagicMock(return_value="activation"),
        FnSpecs=MagicMock(return_value="spec"),
        swiglu_fn=object(),
        matmul_ogs=MagicMock(),
    )
    experts = object.__new__(GptOssMxfp4Experts)
    experts._loaded = True
    experts._abi = abi
    experts._w13 = object()
    experts._w2 = object()
    experts._w13_precision = object()
    experts._w2_precision = object()
    experts.w13_weight_bias = torch.empty(2, 8, dtype=torch.float32)
    experts.w2_weight_bias = torch.empty(2, 4, dtype=torch.float32)
    experts.top_k = 2
    experts.local_intermediate_size = 4
    experts.hidden_size = 4
    experts.alpha = 1.702
    experts.limit = 7.0
    experts.tp_size = 1
    experts._comm = MagicMock()

    output = experts.forward(torch.empty(1, 4, dtype=torch.bfloat16), torch.empty(1, 2))

    assert output.shape == (1, 4)
    assert abi.matmul_ogs.call_count == 2
    first, second = abi.matmul_ogs.call_args_list
    assert first.kwargs["gather_indx"] == "g"
    assert first.kwargs["fused_activation"] == "activation"
    assert second.kwargs["scatter_indx"] == "s"
    assert second.kwargs["gammas"] is routing.gate_scal
    abi.FnSpecs.assert_called_once_with(
        "swiglu",
        abi.swiglu_fn,
        ("alpha", "limit"),
    )
    abi.FusedActivation.assert_called_once_with("spec", (1.702, 7.0), 2)


def test_constructor_fails_fast_below_sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))

    try:
        GptOssMxfp4Experts(2, 64, 64, 1)
    except RuntimeError as exc:
        assert "SM80" in str(exc)
    else:
        raise AssertionError("SM75 must be rejected before allocating packed weights")


def test_tp4_constructor_allocates_23_mxfp4_blocks_per_rank(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(mxfp4, "_load_triton_kernels_abi", lambda: SimpleNamespace())
    monkeypatch.setattr(mxfp4, "get_tp_info", lambda: DistributedInfo(0, 4))
    monkeypatch.setattr(mxfp4, "DistributedCommunicator", MagicMock)

    experts = GptOssMxfp4Experts(1, 64, 2880, 1)

    assert experts.local_intermediate_size == 736
    assert experts.w13_weight.shape[1] == 1472
    assert experts.w2_weight_scale.shape[2] == 23
