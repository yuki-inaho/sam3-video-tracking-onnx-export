"""Test that the use_rope_real=True equivalent detector matches the D2 baseline.

Baseline: outputs/reference/baseline_detector.npz (D2 oracle, complex RoPE).
Equiv: run detector via equivalent source (use_rope_real=True, cos/sin RoPE).

Expected: masks IoU >= 0.99, scores relative diff <= 1e-2.
Also asserts that the patched forward does NOT use complex RoPE in the tracker
attention modules that are present (RoPEAttention.use_rope_real == True).

Design note — equiv-source model_builder loading (intentional):
    The equiv source model_builder.py is loaded via importlib directly so that
    only model_builder is replaced and all other sam3.* submodules come from the
    installed editable package. This is not a PYTHONPATH override.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
BASELINE_NPZ = SANDBOX_ROOT / "outputs" / "reference" / "baseline_detector.npz"
EQUIV_SOURCE_ROOT = SANDBOX_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = SANDBOX_ROOT / "models" / "sam3.pt"

IMAGE_RESOLUTION = 1008
TEXT_PROMPT = "red circle"
CONFIDENCE_THRESHOLD = 0.1


def _make_synthetic_image_1008() -> Image.Image:
    """Black-background red-circle 1008x1008 — identical to D2 oracle fixture."""
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


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _load_equiv_model_builder(equiv_source_root: Path) -> Any:
    """Load patched model_builder via importlib (intentional equiv-source variant)."""
    builder_path = equiv_source_root / "sam3" / "model_builder.py"
    assert builder_path.exists(), (
        f"Patched model_builder not found: {builder_path}. "
        "Run tools/create_equivalent_sam3_source.py first."
    )
    mod_key = "sam3_equiv_test.model_builder"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, builder_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_NPZ.exists(), f"D2 baseline not found: {BASELINE_NPZ}"
    data = np.load(str(BASELINE_NPZ))
    return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def equiv_detector_output() -> dict:
    """Load equiv-source model (use_rope_real=True) and run detector inference."""
    assert EQUIV_SOURCE_ROOT.exists(), (
        f"Equivalent source not generated yet: {EQUIV_SOURCE_ROOT}. "
        "Run tools/create_equivalent_sam3_source.py first."
    )
    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"

    equiv_builder = _load_equiv_model_builder(EQUIV_SOURCE_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = equiv_builder.build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=str(CHECKPOINT_PATH),
        load_from_HF=False,
        use_rope_real=True,
    )

    from sam3.model.sam3_image_processor import Sam3Processor

    image = _make_synthetic_image_1008()
    processor = Sam3Processor(
        model,
        resolution=IMAGE_RESOLUTION,
        device=device,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=TEXT_PROMPT, state=state)

    masks = output["masks"].cpu().numpy().astype(np.float32)
    scores = output["scores"].cpu().numpy().astype(np.float32)
    boxes = output["boxes"].cpu().numpy().astype(np.float32)
    return {"masks": masks, "scores": scores, "boxes": boxes}


class TestEquivDetectorMatchesBaseline:
    """Equiv-source (use_rope_real=True) detector must match D2 baseline output."""

    def test_masks_iou_above_threshold(
        self, baseline: dict, equiv_detector_output: dict
    ) -> None:
        baseline_mask = baseline["masks"][0, 0]
        equiv_mask = equiv_detector_output["masks"][0, 0]
        iou = _compute_iou(baseline_mask > 0.5, equiv_mask > 0.5)
        assert iou >= 0.99, (
            f"Mask IoU {iou:.4f} < 0.99. "
            "use_rope_real=True path may not produce equivalent output."
        )

    def test_scores_relative_diff_below_threshold(
        self, baseline: dict, equiv_detector_output: dict
    ) -> None:
        baseline_score = float(baseline["scores"][0])
        equiv_score = float(equiv_detector_output["scores"][0])
        rel_diff = abs(baseline_score - equiv_score) / (abs(baseline_score) + 1e-8)
        assert rel_diff <= 1e-2, (
            f"Score relative diff {rel_diff:.4e} > 1e-2. "
            f"baseline={baseline_score:.4f}, equiv={equiv_score:.4f}"
        )


class TestEquivDetectorNoComplexRoPE:
    """Verify patched model uses real RoPE, not complex RoPE.

    Note: build_sam3_image_model with enable_inst_interactivity=True includes
    the tracker (which uses RoPEAttention). We verify these modules have
    use_rope_real=True after patching.
    """

    def test_all_rope_attention_modules_use_real_path(self) -> None:
        """RoPEAttention modules in equiv-source model must have use_rope_real=True."""
        assert EQUIV_SOURCE_ROOT.exists(), f"Equiv source not found: {EQUIV_SOURCE_ROOT}"

        equiv_builder = _load_equiv_model_builder(EQUIV_SOURCE_ROOT)
        from sam3.sam.transformer import RoPEAttention

        # enable_inst_interactivity=True includes tracker (RoPEAttention modules)
        model = equiv_builder.build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            use_rope_real=True,
            enable_inst_interactivity=True,
        )
        rope_modules = [m for m in model.modules() if isinstance(m, RoPEAttention)]
        assert len(rope_modules) > 0, (
            "No RoPEAttention modules found even with enable_inst_interactivity=True"
        )
        failing = [m for m in rope_modules if not m.use_rope_real]
        assert len(failing) == 0, (
            f"{len(failing)}/{len(rope_modules)} RoPEAttention modules "
            "have use_rope_real=False after patching"
        )
