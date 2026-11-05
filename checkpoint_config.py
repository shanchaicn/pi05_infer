"""Resolve an OpenPI runtime config from checkpoint metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re


_SAFE_CONFIG = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _validate(value: object, source: Path | str) -> str:
    if not isinstance(value, str) or not _SAFE_CONFIG.fullmatch(value):
        raise ValueError(f"Invalid OpenPI config in {source}: {value!r}")
    return value


def resolve_checkpoint_config(
    checkpoint: str | Path,
    backend: str,
    requested: str | None = None,
    default_manifest: str | Path | None = None,
) -> str:
    """Return a config name, preferring explicit input then checkpoint metadata."""
    if requested and requested != "auto":
        return _validate(requested, "command line")
    if backend not in {"jax", "pytorch"}:
        raise ValueError(f"Unsupported backend: {backend}")

    checkpoint = Path(checkpoint).expanduser().resolve()
    local_manifest = checkpoint / "inference_config.json"
    manifest = _read_json(local_manifest)
    key = f"{backend}_config"
    if key in manifest:
        return _validate(manifest[key], local_manifest)

    conversion_path = checkpoint / "conversion_metadata.json"
    conversion = _read_json(conversion_path)
    source_config = conversion.get("source_config")
    conversion_kind = str(conversion.get("conversion", "")).lower()
    if backend == "pytorch" and source_config and "fused" in conversion_kind:
        return _validate(f"{source_config}_fused", conversion_path)
    if source_config:
        return _validate(source_config, conversion_path)

    selected_default = default_manifest or os.environ.get("OPENPI_INFERENCE_CONFIG")
    if selected_default:
        selected_default = Path(selected_default)
        defaults = _read_json(selected_default)
        if key in defaults:
            return _validate(defaults[key], selected_default)

    raise ValueError(
        f"Cannot auto-detect the {backend} config for {checkpoint}. "
        "Run tools/sync_training_config.py with --jax-checkpoint/"
        "--pytorch-checkpoint, or pass an explicit config."
    )


def replace_auto_config_argv(argv: list[str], backend: str) -> list[str]:
    """Resolve Tyro-style policy config/dir arguments before OpenPI starts."""
    result = list(argv)
    requested = "auto"
    checkpoint = None
    for index, value in enumerate(result):
        if value.startswith("--policy.config="):
            requested = value.split("=", 1)[1]
        elif value == "--policy.config" and index + 1 < len(result):
            requested = result[index + 1]
        elif value.startswith("--policy.dir="):
            checkpoint = value.split("=", 1)[1]
        elif value == "--policy.dir" and index + 1 < len(result):
            checkpoint = result[index + 1]
    if requested != "auto":
        return result
    if checkpoint is None:
        raise ValueError("--policy.dir is required for automatic config discovery")
    resolved = resolve_checkpoint_config(checkpoint, backend, requested)
    for index, value in enumerate(result):
        if value == "--policy.config" and index + 1 < len(result):
            result[index + 1] = resolved
        elif value.startswith("--policy.config="):
            result[index] = f"--policy.config={resolved}"
    return result
