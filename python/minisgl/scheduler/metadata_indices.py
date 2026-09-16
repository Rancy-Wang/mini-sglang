"""CPU-known metadata selection, without device-side variable-length indexing.

No shared scratch, KV ownership changes, or new stream dependencies. Each upload
owns its pinned source; PyTorch's pinned allocator tracks the asynchronous copy.
"""

import torch


def pack_cpu_metadata(parts: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    if not parts:
        return []
    dtype = parts[0].dtype
    if any(t.device.type != "cpu" or t.dtype != dtype for t in parts):
        raise ValueError("Metadata packing requires CPU tensors with one dtype.")
    device = torch.device(device)
    if device.type == "cpu":
        return parts
    host = torch.empty(sum(t.numel() for t in parts), dtype=dtype, pin_memory=True)
    offset = 0
    for part in parts:
        host[offset:offset + part.numel()].copy_(part.reshape(-1))
        offset += part.numel()
    uploaded = host.to(device, non_blocking=True)
    result = []
    offset = 0
    for part in parts:
        view = uploaded[offset:offset + part.numel()].view(part.shape)
        # Retain an explicit owner as well as the allocator's copy-event tracking.
        view._metadata_host = host
        result.append(view)
        offset += part.numel()
    return result


def cpu_mask_indices(mask: torch.Tensor) -> torch.Tensor:
    if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.ndim != 1:
        raise ValueError("Selection requires a one-dimensional CPU bool mask.")
    return torch.nonzero(mask, as_tuple=False).view(-1)


def select_cpu_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices = cpu_mask_indices(mask)
    if len(mask) != len(values):
        raise ValueError("Selection mask must cover the source rows.")
    return values.index_select(0, pack_cpu_metadata([indices], values.device)[0])
