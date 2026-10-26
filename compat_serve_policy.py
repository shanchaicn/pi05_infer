"""Start OpenPI while accepting one verified tied-weight checkpoint alias.

The TB6-R5 training checkpoint stores the PaliGemma token embedding in
addition to the identical tied LM-head tensor.  The runtime model exposes only
the LM-head name, so safetensors reports the embedding name as unexpected.
All other missing or unexpected keys remain fatal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import runpy
import struct

import safetensors.torch


_TIED_EMBEDDING = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
_TIED_LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
_original_load_model = safetensors.torch.load_model


def _verify_identical_tensors(filename: str | Path) -> None:
    path = Path(filename)
    with path.open("rb") as checkpoint:
        header_size = struct.unpack("<Q", checkpoint.read(8))[0]
        header = json.loads(checkpoint.read(header_size))
        data_start = 8 + header_size

        embedding = header[_TIED_EMBEDDING]
        lm_head = header[_TIED_LM_HEAD]
        if embedding["dtype"] != lm_head["dtype"] or embedding["shape"] != lm_head["shape"]:
            raise RuntimeError("Tied embedding and LM-head tensor metadata differ")

        embedding_start, embedding_end = embedding["data_offsets"]
        lm_head_start, lm_head_end = lm_head["data_offsets"]
        if embedding_end - embedding_start != lm_head_end - lm_head_start:
            raise RuntimeError("Tied embedding and LM-head tensor sizes differ")

        offset = 0
        tensor_size = embedding_end - embedding_start
        while offset < tensor_size:
            chunk_size = min(16 * 1024 * 1024, tensor_size - offset)
            checkpoint.seek(data_start + embedding_start + offset)
            embedding_chunk = checkpoint.read(chunk_size)
            checkpoint.seek(data_start + lm_head_start + offset)
            lm_head_chunk = checkpoint.read(chunk_size)
            if embedding_chunk != lm_head_chunk:
                raise RuntimeError("Tied embedding and LM-head tensor values differ")
            offset += chunk_size


def _load_model_compat(model, filename, strict=True, device="cpu", *, backend="mmap"):
    missing, unexpected = _original_load_model(
        model,
        filename,
        strict=False,
        device=device,
        backend=backend,
    )
    ignored = [key for key in unexpected if key == _TIED_EMBEDDING]
    remaining_unexpected = [key for key in unexpected if key != _TIED_EMBEDDING]

    if strict and ignored:
        _verify_identical_tensors(filename)
        logging.warning(
            "Accepted duplicated tied PaliGemma embedding from checkpoint: %s",
            _TIED_EMBEDDING,
        )

    if strict and (missing or remaining_unexpected):
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if remaining_unexpected:
            details.append(f"unexpected keys: {sorted(remaining_unexpected)}")
        raise RuntimeError("Checkpoint is incompatible: " + "; ".join(details))

    return missing, remaining_unexpected


safetensors.torch.load_model = _load_model_compat
runpy.run_path("/app/scripts/serve_policy.py", run_name="__main__")
