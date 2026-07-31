from __future__ import annotations

import pytest
import torch

from minisgl.kernel.radix import fast_compare_radix_key


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fused_radix_compare_detects_key_and_virtual_mismatches(dtype):
    left = torch.tensor([10, -1, 11, 12], dtype=dtype)
    same = left.clone()
    changed_key = torch.tensor([10, -1, 99, 12], dtype=dtype)
    mask = torch.tensor([False, True, False, False], dtype=torch.bool)
    changed_mask = torch.tensor([False, False, False, False], dtype=torch.bool)

    assert fast_compare_radix_key(left, same, mask, mask.clone()) == 4
    assert fast_compare_radix_key(left, changed_key, mask, mask) == 2
    assert fast_compare_radix_key(left, same, mask, changed_mask) == 1
    assert fast_compare_radix_key(left[:3], same, mask[:3], changed_mask) == 1


def test_fused_radix_compare_validates_mask_lengths():
    keys = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(Exception, match="virtual mask length"):
        fast_compare_radix_key(
            keys,
            keys,
            torch.tensor([False], dtype=torch.bool),
            torch.tensor([False, False], dtype=torch.bool),
        )
