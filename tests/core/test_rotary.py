import math

import pytest
import torch

from minisgl.layers.rotary import _get_yarn_parameters


def test_gpt_oss_yarn_parameters_match_transformers_formula():
    scaling = {
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
    }
    inv_freq, attention_factor = _get_yarn_parameters(64, 150000.0, scaling)

    def correction_dim(rotations):
        return 64 * math.log(4096 / (rotations * 2 * math.pi)) / (
            2 * math.log(150000.0)
        )

    low, high = correction_dim(32), correction_dim(1)
    base_inv_freq = 1.0 / (
        150000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    )
    ramp = torch.clamp((torch.arange(32, dtype=torch.float32) - low) / (high - low), 0, 1)
    expected = (base_inv_freq / 32.0) * ramp + base_inv_freq * (1 - ramp)

    torch.testing.assert_close(inv_freq, expected)
    assert attention_factor == pytest.approx(1 + 0.1 * math.log(32.0))


def test_yarn_truncate_flag_changes_fractional_correction_range():
    scaling = {
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 4096,
        "truncate": False,
    }
    exact, _ = _get_yarn_parameters(64, 150000.0, scaling)
    truncated, _ = _get_yarn_parameters(64, 150000.0, {**scaling, "truncate": True})

    assert not torch.equal(exact, truncated)


def test_yarn_explicit_attention_factor_takes_precedence():
    _, attention_factor = _get_yarn_parameters(
        64,
        150000.0,
        {
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "attention_factor": 1.125,
        },
    )

    assert attention_factor == 1.125


def test_yarn_vllm_attention_multiplier_is_composed():
    _, attention_factor = _get_yarn_parameters(
        64,
        150000.0,
        {
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "attn_factor": 0.75,
        },
    )

    assert attention_factor == pytest.approx(0.75 * (1 + 0.1 * math.log(32.0)))
