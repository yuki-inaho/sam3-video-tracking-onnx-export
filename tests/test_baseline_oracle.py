"""Baseline oracle test: verify that D2 reference artifacts exist and are non-empty.

Run after tools/run_pytorch_detector.py to confirm the oracle outputs are usable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REFERENCE_DIR = Path(__file__).parent.parent / "outputs" / "reference"
_NPZ_PATH = _REFERENCE_DIR / "baseline_detector.npz"


def test_reference_npz_exists_and_nonempty() -> None:
    """The baseline npz must exist and contain at least one non-empty mask."""
    assert _NPZ_PATH.exists(), f"baseline_detector.npz not found at {_NPZ_PATH}"

    data = np.load(_NPZ_PATH)
    assert "masks" in data, "npz must contain 'masks' array"

    masks = data["masks"]
    assert masks.ndim >= 3, f"masks must be ≥3-D, got shape {masks.shape}"
    assert masks.shape[0] >= 1, f"At least one mask expected, got {masks.shape[0]}"

    # At least one mask must contain foreground pixels.
    assert masks.any(), "All masks are empty — detector returned no foreground pixels"
