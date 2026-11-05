#!/usr/bin/env python3
"""Synchronize one OpenPI training config into this inference repository."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re


BEGIN = "    # BEGIN AUTO-SYNCED TRAINING CONFIGS\n"
END = "    # END AUTO-SYNCED TRAINING CONFIGS\n"
TARGETS = (
    "rtc_stage/openpi/training/config.py",
    "docker_source/src/openpi/training/config.py",
)


def _literal_keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.literal_eval(keyword.value)
    raise ValueError(f"Missing {name!r} in training config")


def _find_config(source: str, config_name: str) -> tuple[str, ast.Call]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if func_name != "TrainConfig":
            continue
        try:
            name = _literal_keyword(node, "name")
        except (ValueError, TypeError):
            continue
        if name == config_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise ValueError(f"Cannot extract source for {config_name}")
            return segment, node
    raise ValueError(f"Config {config_name!r} was not found")


def _model_values(call: ast.Call) -> dict:
    model_call = None
    for keyword in call.keywords:
        if keyword.arg == "model" and isinstance(keyword.value, ast.Call):
            model_call = keyword.value
            break
    if model_call is None:
        raise ValueError("Config has no literal model constructor")
    values = {}
    for keyword in model_call.keywords:
        if keyword.arg in {
            "pi05", "action_dim", "action_horizon", "max_token_len",
            "discrete_state_input", "paligemma_variant", "action_expert_variant",
        }:
            values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def _indent_call(segment: str) -> str:
    return "\n".join("    " + line for line in segment.splitlines()) + ",\n"


def _make_fused(segment: str, config_name: str) -> tuple[str, str]:
    fused_name = f"{config_name}_fused"
    fused = segment.replace(f'name="{config_name}"', f'name="{fused_name}"', 1)
    fused = re.sub(
        r'((?:paligemma|action_expert)_variant\s*=\s*"[^"]+)_lora"',
        r'\1"',
        fused,
    )
    return fused_name, fused


def _replace_managed_block(path: Path, block: str) -> None:
    text = path.read_text()
    if BEGIN in text or END in text:
        if text.count(BEGIN) != 1 or text.count(END) != 1:
            raise ValueError(f"Malformed managed markers in {path}")
        start = text.index(BEGIN)
        finish = text.index(END, start) + len(END)
        updated = text[:start] + block + text[finish:]
    else:
        anchor = "    #\n    # Inference Aloha configs.\n"
        if anchor not in text:
            raise ValueError(f"Insertion anchor not found in {path}")
        updated = text.replace(anchor, block + anchor, 1)
    path.write_text(updated)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--jax-checkpoint", type=Path)
    parser.add_argument("--pytorch-checkpoint", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    source_path = args.training_root / "src/openpi/training/config.py"
    source = source_path.read_text()
    segment, call = _find_config(source, args.config)
    model = _model_values(call)
    is_lora = any(str(value).endswith("_lora") for value in model.values())
    pytorch_config = args.config
    calls = [_indent_call(segment)]
    if is_lora:
        pytorch_config, fused = _make_fused(segment, args.config)
        calls.append(_indent_call(fused))

    block = BEGIN + "".join(calls) + END
    for relative in TARGETS:
        _replace_managed_block(args.repo_root / relative, block)

    manifest = {
        "schema_version": 1,
        "source_config": args.config,
        "jax_config": args.config,
        "pytorch_config": pytorch_config,
        "model": model,
    }
    manifest_path = args.repo_root / "configs/current.json"
    _write_manifest(manifest_path, manifest)
    for checkpoint in (args.jax_checkpoint, args.pytorch_checkpoint):
        if checkpoint is not None:
            if not checkpoint.is_dir():
                raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
            _write_manifest(checkpoint / "inference_config.json", manifest)

    print(f"Synchronized {args.config} from {source_path}")
    print(f"JAX config: {args.config}")
    print(f"PyTorch config: {pytorch_config}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
