import torch
from minisgl.models.weight import (
    _map_gpt_oss_name,
    _shard_gpt_oss_dense,
    _shard_gpt_oss_mxfp4,
)


def test_gpt_oss_weight_name_mapping():
    assert _map_gpt_oss_name("embedding.weight") == "model.embed_tokens.weight"
    assert _map_gpt_oss_name("block.2.attn.out.bias") == (
        "model.layers.2.self_attn.o_proj.bias"
    )
    assert _map_gpt_oss_name("block.2.mlp.gate_up_proj_blocks") == (
        "model.layers.2.mlp.experts.gate_up_proj_blocks"
    )


def test_tp8_rank7_w13_uses_interleaved_contiguous_rows_and_tail_padding():
    intermediate = 2880
    packed = torch.arange(2 * intermediate, dtype=torch.int32).view(1, -1, 1, 1).to(torch.uint8)

    name, shard = _shard_gpt_oss_mxfp4(
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        packed,
        rank=7,
        tp_size=8,
        intermediate_size=intermediate,
    )

    assert name.endswith(".w13_weight")
    assert shard.dtype == torch.uint8
    assert shard.shape == (1, 768, 1)
    assert torch.equal(shard[:, :384], packed.flatten(start_dim=2)[:, 5376:5760])
    assert torch.count_nonzero(shard[:, 384:]) == 0


def test_tp8_rank7_down_blocks_and_scales_pad_to_384_intermediate():
    blocks = torch.ones(1, 4, 90, 16, dtype=torch.uint8)
    scales = torch.ones(1, 4, 90, dtype=torch.uint8)

    _, block_shard = _shard_gpt_oss_mxfp4(
        "model.layers.0.mlp.experts.down_proj_blocks",
        blocks,
        rank=7,
        tp_size=8,
        intermediate_size=2880,
    )
    _, scale_shard = _shard_gpt_oss_mxfp4(
        "model.layers.0.mlp.experts.down_proj_scales",
        scales,
        rank=7,
        tp_size=8,
        intermediate_size=2880,
    )

    assert block_shard.shape == (1, 4, 192)
    assert scale_shard.shape == (1, 4, 12)
    assert torch.count_nonzero(block_shard[:, :, 96:]) == 0
    assert torch.count_nonzero(scale_shard[:, :, 6:]) == 0


def test_tp4_rank3_mxfp4_shards_use_23_blocks_and_tail_padding():
    intermediate = 2880
    packed = (
        torch.arange(2 * intermediate, dtype=torch.int32)
        .view(1, -1, 1, 1)
        .to(torch.uint8)
    )
    blocks = torch.ones(1, 4, 90, 16, dtype=torch.uint8)
    scales = torch.ones(1, 4, 90, dtype=torch.uint8)

    _, gate_up = _shard_gpt_oss_mxfp4(
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        packed,
        rank=3,
        tp_size=4,
        intermediate_size=intermediate,
    )
    _, down = _shard_gpt_oss_mxfp4(
        "model.layers.0.mlp.experts.down_proj_blocks",
        blocks,
        rank=3,
        tp_size=4,
        intermediate_size=intermediate,
    )
    _, down_scales = _shard_gpt_oss_mxfp4(
        "model.layers.0.mlp.experts.down_proj_scales",
        scales,
        rank=3,
        tp_size=4,
        intermediate_size=intermediate,
    )

    assert gate_up.shape == (1, 1472, 1)
    assert torch.equal(gate_up[:, :1344], packed.flatten(start_dim=2)[:, 4416:5760])
    assert torch.count_nonzero(gate_up[:, 1344:]) == 0
    assert down.shape == (1, 4, 368)
    assert torch.count_nonzero(down[:, :, :336]) == 1 * 4 * 336
    assert torch.count_nonzero(down[:, :, 336:]) == 0
    assert down_scales.shape == (1, 4, 23)
    assert torch.count_nonzero(down_scales[:, :, :21]) == 1 * 4 * 21
    assert torch.count_nonzero(down_scales[:, :, 21:]) == 0


def test_down_bias_is_kept_only_on_tp_rank_zero():
    bias = torch.ones(2, 4, dtype=torch.bfloat16)
    key = "model.layers.0.mlp.experts.down_proj_bias"

    _, primary = _shard_gpt_oss_mxfp4(key, bias, 0, 8, 2880)
    _, secondary = _shard_gpt_oss_mxfp4(key, bias, 3, 8, 2880)

    assert torch.equal(primary, bias)
    assert torch.count_nonzero(secondary) == 0


def test_attention_sinks_are_sharded_but_o_proj_bias_is_replicated():
    sinks = torch.arange(64)
    bias = torch.arange(8)

    sink_shard = _shard_gpt_oss_dense("x.self_attn.sinks", sinks, 3, 8)
    bias_shard = _shard_gpt_oss_dense("x.self_attn.o_proj.bias", bias, 3, 8)

    assert torch.equal(sink_shard, sinks[24:32])
    assert bias_shard is bias
