from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minisgl.server.args import parse_args


MODEL_FAMILIES = [
    (Path("/share/public/public_models/Qwen3-0.6B"), "qwen3", True),
    (Path("/share/public/public_models/Qwen3-8B"), "qwen3", False),
    (Path("/share/public/public_models/Qwen3-30B-A3B"), "qwen3_moe", False),
    (Path("/share/wangruoxi/models/gpt-oss-20b"), "gpt_oss", False),
]


@pytest.mark.parametrize("model_path,model_type,tied", MODEL_FAMILIES)
def test_default_cli_builds_each_supported_model_family(model_path, model_type, tied):
    if not model_path.exists():
        pytest.skip(f"model config is unavailable: {model_path}")

    args, run_shell = parse_args(["--model", str(model_path), "--port", "8000"])
    args = replace(args, distributed_port=29501)

    assert not run_shell
    assert args.model_config.model_type == model_type
    assert args.model_config.tie_word_embeddings is tied
    assert args.cuda_graph_max_bs is None
    assert args.distributed_addr == "tcp://127.0.0.1:29501"
