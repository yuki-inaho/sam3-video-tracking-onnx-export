"""Build the official SAM 3.1 multiplex tracker from its merged checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from sam3_onnx_equiv.export._equiv_loader import equiv_sam3_on_path, load_checkpoint
from sam3_onnx_equiv.path_config import model_paths, sam31_cpu_source_root


def _extract_tracker_weights(checkpoint: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    weights = {
        key[len("tracker.model.") :]: value
        for key, value in checkpoint.items()
        if key.startswith("tracker.model.")
    }
    weights.update(
        {
            key[len("detector.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("detector.backbone.")
        }
    )
    return weights


def _derived_rope_buffers(expected: set[str]) -> set[str]:
    # Real-RoPE buffers are derived from checkpointed freqs_cis, not learned.
    return {
        key
        for key in expected
        if key.endswith(".freqs_cis_real") or key.endswith(".freqs_cis_imag")
    }


def _load_tracker_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    weights = _extract_tracker_weights(load_checkpoint(checkpoint_path))
    expected = set(model.state_dict())
    derived = _derived_rope_buffers(expected)
    missing = expected - weights.keys() - derived
    if missing:
        raise ValueError(
            f"SAM 3.1 tracker checkpoint lacks {len(missing)} keys: {sorted(missing)[:5]}"
        )
    result = model.load_state_dict(
        {key: weights[key] for key in expected & weights.keys()}, strict=False
    )
    if set(result.missing_keys) != derived or result.unexpected_keys:
        raise ValueError("SAM 3.1 checkpoint loading produced unexpected missing/extra keys")


def build_sam31_tracker(
    *,
    source_root: Path | None = None,
    checkpoint_path: Path | None = None,
    use_rope_real: bool = False,
) -> torch.nn.Module:
    """Return a CPU tracker with all 931 parameters loaded from facebook/sam3.1.

    The official merged checkpoint stores tracker weights under ``tracker.model``
    and the shared TriHead backbone under ``detector.backbone``. The standalone
    tracker requires both groups. Requiring a complete key match prevents an
    accidentally mixed SAM 3/SAM 3.1 model.
    """
    paths = model_paths("sam31")
    source_root = source_root or sam31_cpu_source_root()
    checkpoint_path = checkpoint_path or paths.checkpoint_path
    with equiv_sam3_on_path(source_root):
        from sam3.model_builder import build_sam3_multiplex_video_model

        model = build_sam3_multiplex_video_model(
            checkpoint_path=None,
            load_from_HF=False,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=use_rope_real,
            device="cpu",
            compile=False,
        )

    _load_tracker_state(model, checkpoint_path)
    model.eval()
    return model
