"""Independent Transformers reference for token-position KV deletion.

This module intentionally does not import the mini-sglang scheduler, Radix cache,
attention backends, or Drop compiler.  It is used by the R9 diagnostic runner to
check whether the same frozen model degenerates when the prescribed KV entries
are removed in a second runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    messages: tuple[dict[str, str], ...]
    drop_after_message: int | None = None
    drop_messages: tuple[int, ...] = ()


def short_reference_cases() -> tuple[ReferenceCase, ...]:
    base = (
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "I remember 42."},
        {"role": "user", "content": "What number was mentioned?"},
    )
    twice = base[:2] + base
    return (
        ReferenceCase("original_no_drop", base),
        ReferenceCase("original_drop_first_user", base, 1, (0,)),
        ReferenceCase("two_turn_drop_first_user", twice, 3, (0,)),
        ReferenceCase("two_turn_drop_second_user", twice, 3, (2,)),
        ReferenceCase(
            "system_kept_drop_first_user",
            ({"role": "system", "content": "You are a helpful assistant."},) + base,
            2,
            (1,),
        ),
        ReferenceCase("orphaned_assistant_no_drop", base[1:]),
        ReferenceCase("question_only_no_drop", base[2:]),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(model_path: Path) -> dict[str, str]:
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    }
    return {
        path.name: sha256_file(path)
        for path in sorted(model_path.iterdir())
        if path.is_file() and (path.name in names or path.name.endswith(".safetensors.index.json"))
    }


def render_case(tokenizer: Any, case: ReferenceCase) -> tuple[list[int], list[tuple[int, int]]]:
    """Render once and independently prove each message boundary against prefixes."""

    messages = [dict(message) for message in case.messages]
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    boundaries: list[tuple[int, int]] = []
    start = 0
    for end in range(1, len(messages) + 1):
        prefix = tokenizer.apply_chat_template(
            messages[:end],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if list(full_ids[: len(prefix)]) != list(prefix):
            raise RuntimeError(
                f"Chat template prefix for message {end - 1} is not a canonical full prefix."
            )
        boundaries.append((start, len(prefix)))
        start = len(prefix)
    if not 0 < start < len(full_ids):
        raise RuntimeError("Generation prompt must follow at least one rendered message token.")
    return [int(token) for token in full_ids], boundaries


def validate_case_plan(
    case: ReferenceCase,
    full_ids: Sequence[int],
    boundaries: Sequence[tuple[int, int]],
) -> tuple[int, tuple[int, ...]]:
    if len(boundaries) != len(case.messages):
        raise ValueError("Message boundary count does not match the case.")
    if not full_ids:
        raise ValueError("Reference case cannot have an empty token stream.")
    previous = 0
    for start, end in boundaries:
        if start != previous or not start < end <= len(full_ids):
            raise ValueError("Message boundaries must be contiguous, increasing token ranges.")
        previous = end
    if case.drop_after_message is None:
        if case.drop_messages:
            raise ValueError("drop_messages requires drop_after_message.")
        return len(full_ids), ()
    if not 0 <= case.drop_after_message < len(boundaries):
        raise ValueError("drop_after_message is outside the conversation.")
    if any(message < 0 or message >= case.drop_after_message for message in case.drop_messages):
        raise ValueError("A Drop may only target a message preceding its trigger.")
    trigger = boundaries[case.drop_after_message][1]
    removed = tuple(
        token
        for message in case.drop_messages
        for token in range(boundaries[message][0], boundaries[message][1])
    )
    if len(set(removed)) != len(removed):
        raise ValueError("Drop token ranges overlap.")
    return trigger, removed


def compact_dynamic_cache(cache: Any, keep_indices: torch.Tensor) -> None:
    """Compact every initialized Transformers cache layer to the same KV rows."""

    if keep_indices.ndim != 1 or keep_indices.dtype != torch.int64:
        raise ValueError("keep_indices must be a one-dimensional int64 tensor.")
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        if layer.keys.shape[-2] != layer.values.shape[-2]:
            raise RuntimeError("Reference K/V cache lengths disagree.")
        if keep_indices.numel() and int(keep_indices[-1]) >= layer.keys.shape[-2]:
            raise ValueError("keep_indices exceeds the reference cache length.")
        indices = keep_indices.to(layer.keys.device)
        layer.keys = layer.keys.index_select(-2, indices).contiguous()
        layer.values = layer.values.index_select(-2, indices).contiguous()


def clone_dynamic_cache(cache: Any) -> Any:
    """Copy survivor K/V into fresh allocations without using mini-sglang pages."""

    from transformers.cache_utils import DynamicCache

    copied = [
        (layer.keys.clone(), layer.values.clone())
        for layer in cache.layers
        if layer.is_initialized
    ]
    return DynamicCache(ddp_cache_data=copied)


def _cache_length(cache: Any) -> int:
    lengths = {
        int(layer.keys.shape[-2])
        for layer in cache.layers
        if layer.is_initialized
    }
    if len(lengths) != 1:
        raise RuntimeError(f"Reference cache layers have inconsistent lengths: {lengths}")
    return lengths.pop()


def _forward_segment(
    model: Any,
    cache: Any,
    token_ids: Sequence[int],
    raw_positions: Sequence[int],
) -> torch.Tensor:
    if len(token_ids) != len(raw_positions) or not token_ids:
        raise ValueError("A reference segment needs equal non-empty token and position streams.")
    device = next(model.parameters()).device
    physical_start = _cache_length(cache) if any(
        layer.is_initialized for layer in cache.layers
    ) else 0
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    position_ids = torch.tensor([raw_positions], dtype=torch.long, device=device)
    cache_position = torch.arange(
        physical_start,
        physical_start + len(token_ids),
        dtype=torch.long,
        device=device,
    )
    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return output.logits[0, -1].float()


def generate_case(
    model: Any,
    tokenizer: Any,
    case: ReferenceCase,
    *,
    max_tokens: int,
    clone_survivors: bool = False,
    rebuild_survivors: bool = False,
    retained_removed_raw: Sequence[int] = (),
) -> dict[str, Any]:
    from transformers.cache_utils import DynamicCache

    if clone_survivors and rebuild_survivors:
        raise ValueError("Survivor cache cloning and rebuilding are mutually exclusive.")
    full_ids, boundaries = render_case(tokenizer, case)
    trigger, removed = validate_case_plan(case, full_ids, boundaries)
    retained = tuple(int(raw) for raw in retained_removed_raw)
    if len(set(retained)) != len(retained):
        raise ValueError("retained_removed_raw cannot contain duplicates.")
    if any(raw not in removed for raw in retained):
        raise ValueError("retained_removed_raw must be a subset of the Drop range.")
    retained_set = set(retained)
    effective_removed = tuple(raw for raw in removed if raw not in retained_set)
    cache = DynamicCache(config=model.config)
    active_raw: list[int] = []
    segment_records: list[dict[str, Any]] = []

    first_end = trigger if removed else len(full_ids)
    logits = _forward_segment(model, cache, full_ids[:first_end], range(first_end))
    active_raw.extend(range(first_end))
    segment_records.append(
        {"raw": list(range(first_end)), "cache_before_drop": _cache_length(cache)}
    )
    if effective_removed:
        removed_set = set(effective_removed)
        keep_physical = torch.tensor(
            [i for i, raw in enumerate(active_raw) if raw not in removed_set],
            dtype=torch.int64,
        )
        compact_dynamic_cache(cache, keep_physical)
        active_raw = [raw for raw in active_raw if raw not in removed_set]
        if rebuild_survivors:
            cache = DynamicCache(config=model.config)
            if active_raw:
                logits = _forward_segment(
                    model,
                    cache,
                    [full_ids[raw] for raw in active_raw],
                    active_raw,
                )
        elif clone_survivors:
            cache = clone_dynamic_cache(cache)
        if trigger < len(full_ids):
            logits = _forward_segment(
                model,
                cache,
                full_ids[trigger:],
                range(trigger, len(full_ids)),
            )
            active_raw.extend(range(trigger, len(full_ids)))
            segment_records.append(
                {
                    "raw": list(range(trigger, len(full_ids))),
                    "cache_after_segment": _cache_length(cache),
                }
            )

    generated: list[int] = []
    top10: list[dict[str, Any]] = []
    eos_values = tokenizer.eos_token_id
    eos_ids = {
        int(token)
        for token in (
            eos_values if isinstance(eos_values, (list, tuple, set)) else [eos_values]
        )
        if token is not None
    }
    for offset in range(max_tokens):
        values, ids = logits.topk(10)
        token = int(ids[0])
        generated.append(token)
        top10.append(
            {"ids": [int(item) for item in ids], "logits": [float(item) for item in values]}
        )
        if token in eos_ids:
            break
        raw_position = len(full_ids) + offset
        logits = _forward_segment(model, cache, [token], [raw_position])
        active_raw.append(raw_position)

    return {
        "case": asdict(case),
        "canonical_token_ids": full_ids,
        "canonical_sha256": hashlib.sha256(
            torch.tensor(full_ids, dtype=torch.int32).numpy().tobytes()
        ).hexdigest(),
        "boundaries": [list(item) for item in boundaries],
        "trigger": trigger,
        "removed_raw": list(removed),
        "effective_removed_raw": list(effective_removed),
        "retained_removed_raw": list(retained),
        "active_prompt_raw": [raw for raw in active_raw if raw < len(full_ids)],
        "segments": segment_records,
        "clone_survivors": clone_survivors,
        "rebuild_survivors": rebuild_survivors,
        "tokens": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=False),
        "top10": top10,
        "finish_reason": "stop" if generated and generated[-1] in eos_ids else "length",
    }


def runtime_fingerprint(model_path: Path) -> dict[str, Any]:
    import transformers

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": None
        if gpu is None
        else {"name": gpu.name, "total_memory": gpu.total_memory},
        "model_path": str(model_path),
        "model_manifest": model_manifest(model_path),
    }


@torch.inference_mode()
def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The independent reference requires CUDA for this model.")
    model_path = Path(args.model).resolve()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prepared = []
    for case in short_reference_cases():
        full_ids, boundaries = render_case(tokenizer, case)
        trigger, removed = validate_case_plan(case, full_ids, boundaries)
        prepared.append(
            {
                "name": case.name,
                "canonical_token_ids": full_ids,
                "boundaries": [list(item) for item in boundaries],
                "trigger": trigger,
                "removed_raw": list(removed),
            }
        )
    fingerprint = runtime_fingerprint(model_path)
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attention,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    loaded = time.monotonic()
    results = []
    for case in short_reference_cases():
        results.append(
            generate_case(model, tokenizer, case, max_tokens=args.max_tokens)
        )
        if case.drop_messages:
            results.append(
                generate_case(
                    model,
                    tokenizer,
                    case,
                    max_tokens=args.max_tokens,
                    clone_survivors=True,
                )
            )
            results.append(
                generate_case(
                    model,
                    tokenizer,
                    case,
                    max_tokens=args.max_tokens,
                    rebuild_survivors=True,
                )
            )
    intervention_case = short_reference_cases()[1]
    intervention_ids, intervention_boundaries = render_case(tokenizer, intervention_case)
    _, intervention_removed = validate_case_plan(
        intervention_case,
        intervention_ids,
        intervention_boundaries,
    )
    interventions = []
    if not args.skip_retention_interventions:
        for raw in intervention_removed:
            interventions.append(
                generate_case(
                    model,
                    tokenizer,
                    intervention_case,
                    max_tokens=args.intervention_max_tokens,
                    clone_survivors=True,
                    retained_removed_raw=(raw,),
                )
            )
        for length in range(1, len(intervention_removed) + 1):
            interventions.append(
                generate_case(
                    model,
                    tokenizer,
                    intervention_case,
                    max_tokens=args.intervention_max_tokens,
                    clone_survivors=True,
                    retained_removed_raw=intervention_removed[:length],
                )
            )
    return {
        "status": "completed",
        "adapter": "transformers_dynamic_cache",
        "attention": args.attention,
        "fingerprint": fingerprint,
        "preflight": prepared,
        "results": results,
        "minimal_retention_interventions": interventions,
        "timing": {
            "preflight_seconds": load_started - started,
            "load_seconds": loaded - load_started,
            "experiment_seconds": time.monotonic() - loaded,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--intervention-max-tokens", type=int, default=8)
    parser.add_argument("--skip-retention-interventions", action="store_true")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_reference(args)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
