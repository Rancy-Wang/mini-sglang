from __future__ import annotations

import glob
import math
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _pad_tensor(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    if tensor.shape[dim] == size:
        return tensor.contiguous()
    shape = list(tensor.shape)
    shape[dim] = size - tensor.shape[dim]
    padding = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=dim).contiguous()


def _shard_gpt_oss_mxfp4(
    name: str,
    value: torch.Tensor,
    rank: int,
    tp_size: int,
    intermediate_size: int,
) -> tuple[str, torch.Tensor]:
    block_size = 32
    if intermediate_size % block_size:
        raise ValueError("GPT-OSS intermediate size must be divisible by the MXFP4 block size")
    blocks_per_rank = math.ceil(intermediate_size / block_size / tp_size)
    local_intermediate = blocks_per_rank * block_size
    start = rank * local_intermediate
    end = min(start + local_intermediate, intermediate_size)

    replacements = {
        "gate_up_proj_blocks": "w13_weight",
        "gate_up_proj_scales": "w13_weight_scale",
        "gate_up_proj_bias": "w13_weight_bias",
        "down_proj_blocks": "w2_weight",
        "down_proj_scales": "w2_weight_scale",
        "down_proj_bias": "w2_weight_bias",
    }
    for source, target in replacements.items():
        if source in name:
            output_name = name.replace(source, target)
            break
    else:
        raise ValueError(f"Unknown GPT-OSS MXFP4 tensor: {name}")

    if "gate_up_proj_blocks" in name:
        value = value.flatten(start_dim=2)
        shard = value[:, 2 * start : 2 * end]
        return output_name, _pad_tensor(shard, 1, 2 * local_intermediate)
    if "gate_up_proj_scales" in name or "gate_up_proj_bias" in name:
        shard = value[:, 2 * start : 2 * end]
        return output_name, _pad_tensor(shard, 1, 2 * local_intermediate)
    if "down_proj_blocks" in name:
        value = value.flatten(start_dim=2)
        block_start = start // 2
        block_end = end // 2
        return output_name, _pad_tensor(
            value[:, :, block_start:block_end],
            2,
            local_intermediate // 2,
        )
    if "down_proj_scales" in name:
        block_start = start // block_size
        block_end = end // block_size
        return output_name, _pad_tensor(
            value[:, :, block_start:block_end],
            2,
            blocks_per_rank,
        )
    if "down_proj_bias" in name:
        return output_name, value if rank == 0 else torch.zeros_like(value)
    raise AssertionError("unreachable")


def _map_gpt_oss_name(name: str) -> str:
    if name == "embedding.weight":
        return "model.embed_tokens.weight"
    if name == "unembedding.weight":
        return "lm_head.weight"
    if name == "norm.scale":
        return "model.norm.weight"
    match = re.match(r"block\.(\d+)\.(.+)", name)
    if match is None:
        return name
    layer, suffix = match.groups()
    prefix = f"model.layers.{layer}."
    if suffix.startswith("mlp.gate_up_proj"):
        return prefix + "mlp.experts." + suffix.removeprefix("mlp.")
    if suffix.startswith("mlp.down_proj"):
        return prefix + "mlp.experts." + suffix.removeprefix("mlp.")
    replacements = {
        "attn.q_proj": "self_attn.q_proj",
        "attn.k_proj": "self_attn.k_proj",
        "attn.v_proj": "self_attn.v_proj",
        "attn.out": "self_attn.o_proj",
        "attn.sinks": "self_attn.sinks",
        "attn.norm.scale": "input_layernorm.weight",
        "mlp.gate": "mlp.router",
        "mlp.norm.scale": "post_attention_layernorm.weight",
        "mlp.experts.": "mlp.experts.",
    }
    for source, target in replacements.items():
        if suffix == source or suffix.startswith(source + "."):
            suffix = target + suffix[len(source) :]
            break
    return prefix + suffix


def _shard_gpt_oss_dense(
    name: str,
    value: torch.Tensor,
    rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    if name.endswith(".self_attn.sinks"):
        if value.shape[0] % tp_size:
            raise ValueError("GPT-OSS attention sinks must divide evenly across TP ranks")
        return value.chunk(tp_size, dim=0)[rank].clone()
    if name.endswith(".self_attn.o_proj.bias"):
        return value
    return None


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                if config.is_gpt_oss:
                    name = _map_gpt_oss_name(name)
                    if any(
                        marker in name
                        for marker in (
                            "gate_up_proj_blocks",
                            "gate_up_proj_scales",
                            "gate_up_proj_bias",
                            "down_proj_blocks",
                            "down_proj_scales",
                            "down_proj_bias",
                        )
                    ):
                        yield _shard_gpt_oss_mxfp4(
                            name,
                            raw,
                            tp_info.rank,
                            tp_info.size,
                            config.intermediate_size,
                        )
                        continue
                    if (
                        dense_shard := _shard_gpt_oss_dense(
                            name,
                            raw,
                            tp_info.rank,
                            tp_info.size,
                        )
                    ) is not None:
                        yield name, dense_shard
                        continue
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
