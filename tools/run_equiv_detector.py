#!/usr/bin/env python3
"""Run SAM3 detector inference using the use_rope_real=True equivalent source.

This script explicitly loads model_builder.py from the patched equiv source tree
(outputs/sam3_equiv_source) via importlib so that the original sam3 package is
NOT shadowed — only model_builder is replaced.

Design note (intentional equiv-source variant loading):
    The official SAM3 checkout configured by the environment provides the full sam3 package.
    Only sam3.model_builder is patched (use_rope_real=True wiring).
    We load the patched model_builder.py file directly with importlib so that:
      - All other sam3.* submodules come from the installed package (no PYTHONPATH hack).
      - The patch is explicit and auditable (not a silent path override).

Usage:
    uv run python tools/run_equiv_detector.py \\
        --equiv-source outputs/sam3_equiv_source \\
        --checkpoint models/sam3.pt \\
        --output outputs/reference/equiv_detector.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
IMAGE_RESOLUTION = 1008
TEXT_PROMPT = "red circle"
CONFIDENCE_THRESHOLD = 0.1


def _make_synthetic_image_1008() -> Image.Image:
    """Black-background red-circle image at 1008x1008 (matches D2 oracle exactly)."""
    size = IMAGE_RESOLUTION
    radius = size // 10
    center = (size // 2, size // 2)
    image = Image.new("RGB", (size, size), (0, 0, 0))
    bbox = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    ImageDraw.Draw(image).ellipse(bbox, fill=(255, 0, 0))
    return image


def _load_equiv_model_builder(equiv_source_root: Path) -> Any:
    """Load model_builder from equiv source via importlib.

    Intentional: loads the patched file directly so only model_builder
    uses the use_rope_real=True variant; all other sam3.* come from
    the installed editable package.
    """
    builder_path = equiv_source_root / "sam3" / "model_builder.py"
    if not builder_path.exists():
        raise FileNotFoundError(
            f"Patched model_builder not found: {builder_path}. "
            "Run tools/create_equivalent_sam3_source.py first."
        )
    mod_key = "sam3_equiv.model_builder"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, builder_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    log.info("Loaded equiv model_builder from %s", builder_path)
    return module


def run_equiv_detector(
    equiv_source_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    """Build equiv detector (use_rope_real=True), run inference, return outputs."""
    equiv_builder = _load_equiv_model_builder(equiv_source_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Building equiv SAM3 image model (use_rope_real=True) on %s", device)
    model = equiv_builder.build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        use_rope_real=True,
    )

    # Verify all RoPEAttention modules use real path
    from sam3.sam.transformer import RoPEAttention

    rope_modules = [m for m in model.modules() if isinstance(m, RoPEAttention)]
    log.info("Found %d RoPEAttention modules", len(rope_modules))
    for m in rope_modules:
        if not m.use_rope_real:
            raise RuntimeError(f"RoPEAttention module has use_rope_real=False after patching: {m}")
    log.info(
        "All %d RoPEAttention modules use real RoPE (complex path bypassed)",
        len(rope_modules),
    )

    from sam3.model.sam3_image_processor import Sam3Processor

    image = _make_synthetic_image_1008()
    processor = Sam3Processor(
        model,
        resolution=IMAGE_RESOLUTION,
        device=device,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    log.info("Running set_image...")
    state = processor.set_image(image)
    log.info("Running set_text_prompt: '%s'", TEXT_PROMPT)
    output = processor.set_text_prompt(prompt=TEXT_PROMPT, state=state)

    masks = output["masks"].cpu().numpy().astype(np.float32)
    scores = output["scores"].cpu().numpy().astype(np.float32)
    boxes = output["boxes"].cpu().numpy().astype(np.float32)

    log.info("Prediction: masks=%s scores=%s", masks.shape, scores.tolist()[:5])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), masks=masks, scores=scores, boxes=boxes)
    log.info("Saved equiv detector output to %s", output_path)

    return {"masks": masks, "scores": scores, "boxes": boxes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3 detector with use_rope_real=True equiv source"
    )
    parser.add_argument(
        "--equiv-source",
        type=Path,
        default=SANDBOX_ROOT / "outputs" / "sam3_equiv_source",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SANDBOX_ROOT / "models" / "sam3.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SANDBOX_ROOT / "outputs" / "reference" / "equiv_detector.npz",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_equiv_detector(args.equiv_source, args.checkpoint, args.output)
    print(f"scores={result['scores']}")
    print(f"masks_shape={result['masks'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
