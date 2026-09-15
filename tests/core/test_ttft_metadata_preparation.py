import gc
import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).parents[2] / "python/minisgl/scheduler/metadata_indices.py"
_spec = importlib.util.spec_from_file_location("metadata_indices", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
cpu_mask_indices = _module.cpu_mask_indices
pack_cpu_metadata = _module.pack_cpu_metadata
select_cpu_mask = _module.select_cpu_mask


@pytest.mark.parametrize("mask", [[], [False, False], [True, True], [False, True, False, True]])
def test_cpu_mask_preserves_values_and_order(mask):
    mask = torch.tensor(mask, dtype=torch.bool)
    values = torch.arange(len(mask) * 2, dtype=torch.int32).view(-1, 2)
    assert torch.equal(select_cpu_mask(values, mask), values[mask])
    assert torch.equal(cpu_mask_indices(mask), torch.arange(len(mask))[mask])


def test_pack_shapes_empty_and_validation():
    parts = [torch.arange(6).view(2, 3).t(), torch.empty(0, dtype=torch.int64)]
    assert pack_cpu_metadata([], torch.device("cpu")) == []
    packed = pack_cpu_metadata(parts, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(parts, packed))
    with pytest.raises(ValueError):
        pack_cpu_metadata([parts[0], torch.ones(1)], torch.device("cpu"))
    with pytest.raises(ValueError):
        cpu_mask_indices(torch.ones(2))
    with pytest.raises(ValueError):
        select_cpu_mask(torch.ones(2), torch.ones(1, dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_pack_selection_and_copy_lifetime():
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    expected = [torch.arange(6000).view(2000, 3).t(), torch.empty(0, dtype=torch.int64)]
    with torch.cuda.stream(stream):
        # Enqueue before releasing the caller's CPU references. Exercise real
        # non-contiguous inputs and the same stream used by preparation.
        parts = [t.clone() for t in expected]
        uploaded = pack_cpu_metadata(parts, device)
        del parts
        gc.collect()
        copies = [t.clone() for t in uploaded]
        del uploaded
        mask = torch.tensor([True, False, True, False])
        values = torch.arange(4, device=device)
        selected = select_cpu_mask(values, mask)
    stream.synchronize()
    assert all(torch.equal(a.cpu(), b) for a, b in zip(copies, expected))
    assert selected.cpu().tolist() == [0, 2]
