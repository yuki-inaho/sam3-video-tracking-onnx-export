"""Shared helpers for loading the SAM3 equiv-source and finalising ONNX exports.

These utilities factor out logic that was previously duplicated across
``image_encoder.py``, ``memory_attention.py``, ``memory_encoder.py`` and
``decode_head.py``:

  * ``evict_sam3_cache`` — drop cached ``sam3.*`` modules so a fresh import
    resolves to the equiv-source.
  * ``equiv_sam3_on_path`` — context manager that prepends the equiv-source root
    to ``sys.path`` (evicting the cache on entry) and restores it on exit.
  * ``load_checkpoint`` — read ``models/sam3.pt`` and unwrap the optional
    ``{"model": ...}`` container.
  * ``extract_prefixed_state`` — slice a checkpoint state dict by key prefix.
  * ``patch_output_dims`` — replace symbolic output dims in an ONNX ModelProto
    with concrete values and re-save.

KISS: these are thin, single-purpose helpers; no speculative configuration is
added beyond what the four export modules already share.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import onnx

log = logging.getLogger(__name__)


def evict_sam3_cache() -> None:
    """Remove all ``sam3`` / ``sam3.*`` entries from ``sys.modules``.

    Forces the next ``import sam3...`` to resolve against whatever is currently
    first on ``sys.path`` (the equiv-source when used with ``equiv_sam3_on_path``).
    """
    to_del = [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]
    for k in to_del:
        del sys.modules[k]


@contextmanager
def equiv_sam3_on_path(equiv_source_root: Path) -> Iterator[None]:
    """Temporarily make ``sam3.*`` resolve to the equiv-source copy.

    On entry the resolved ``equiv_source_root`` is prepended to ``sys.path`` (if
    not already present) and the ``sam3.*`` module cache is evicted.  On exit the
    injected path entry is removed so subsequent imports are unaffected.

    Args:
        equiv_source_root: Path to ``outputs/sam3_equiv_source``.

    Raises:
        FileNotFoundError: if ``equiv_source_root`` does not exist.
    """
    if not equiv_source_root.exists():
        raise FileNotFoundError(f"Equiv source not found: {equiv_source_root}")

    equiv_root_str = str(equiv_source_root.resolve())
    inserted = equiv_root_str not in sys.path
    if inserted:
        sys.path.insert(0, equiv_root_str)
    evict_sam3_cache()
    try:
        yield
    finally:
        if inserted and equiv_root_str in sys.path:
            sys.path.remove(equiv_root_str)


def load_checkpoint(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    """Load a SAM3 checkpoint and unwrap the optional ``{"model": ...}`` container.

    Args:
        checkpoint_path: Path to ``models/sam3.pt``.

    Returns:
        The flat state dict mapping parameter names to tensors.

    Raises:
        FileNotFoundError: if ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return ckpt


def extract_prefixed_state(
    ckpt: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Return the sub-state-dict whose keys start with ``prefix`` (prefix stripped).

    Args:
        ckpt: Full checkpoint state dict.
        prefix: Key prefix to select and strip (e.g. ``"tracker."``).

    Returns:
        New dict mapping ``key[len(prefix):]`` -> tensor for matching keys.
    """
    return {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}


def patch_output_dims(
    model_proto: onnx.ModelProto,
    output_path: Path,
    known_shapes: Mapping[str, list[int]],
) -> int:
    """Replace symbolic dim_params in graph output ValueInfo with concrete values.

    This is a metadata-only change: the ONNX computation nodes are unmodified.
    ``onnx.checker`` is re-run on the patched proto and the model is saved only if
    at least one symbolic dim was replaced.

    Args:
        model_proto:  Loaded ONNX ModelProto (modified in-place).
        output_path:  Path where the patched model is saved.
        known_shapes: Mapping of output tensor name -> concrete int shape.

    Returns:
        Number of symbolic dims that were replaced.
    """
    import onnx  # noqa: PLC0415

    patched = 0
    for out in model_proto.graph.output:
        if out.name not in known_shapes:
            continue
        shape = known_shapes[out.name]
        for i, d in enumerate(out.type.tensor_type.shape.dim):
            if d.HasField("dim_param"):
                d.ClearField("dim_param")
                d.dim_value = shape[i]
                patched += 1

    if patched:
        log.info("patch_output_dims: patched %d symbolic dims → concrete values.", patched)
        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, str(output_path))
        log.info("Patched model saved to %s.", output_path)
    else:
        log.info("patch_output_dims: all output dims already concrete — no patch needed.")
    return patched
