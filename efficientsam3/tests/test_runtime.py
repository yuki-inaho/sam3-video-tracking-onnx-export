import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from contract import CONTEXT, MODEL_SHA256, RESOLUTION, SOURCE_REVISION, VARIANT
from runtime import preprocess, save_annotation, validate_manifest


def valid_manifest():
    return {
        "format_version": 1,
        "variant": VARIANT,
        "resolution": RESOLUTION,
        "context": CONTEXT,
        "checkpoint_sha256": MODEL_SHA256,
        "source_revision": SOURCE_REVISION,
        "sequence_mode": "independent_detection",
        "graphs": ["vision.onnx", "text.onnx", "grounding.onnx"],
    }


@pytest.mark.parametrize("key", ["context", "variant", "sequence_mode", "checkpoint_sha256"])
def test_wrong_manifest_rejected(key):
    manifest = valid_manifest()
    manifest[key] = "wrong"
    with pytest.raises(ValueError, match=key):
        validate_manifest(manifest)


def test_runtime_import_does_not_load_torch():
    p = subprocess.run(
        [sys.executable, "-c", "import runtime, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert p.returncode == 0, p.stderr


def test_preprocessing_range_and_layout():
    black = preprocess(Image.new("RGB", (17, 11), (0, 0, 0)))
    white = preprocess(Image.new("RGB", (19, 15), (255, 255, 255)))
    assert black.shape == (1, 3, RESOLUTION, RESOLUTION)
    assert black.dtype == np.float32
    assert np.all(black == -1) and np.all(white == 1)


def test_saved_mask_and_box(tmp_path):
    raw = (
        np.array([[[0.5, 0.5, 0.5, 1.0]]]),
        np.array([[[10.0]]]),
        np.array([[10.0]]),
        np.array([[[[-1, 1], [-1, 1]]]], dtype=np.float32),
    )
    result = save_annotation(raw, Image.new("RGB", (4, 2)), tmp_path, 0.5)
    assert result[0]["box_xyxy"] == [1, 0, 3, 2]
    assert result[0]["foreground_pixels"] == 4
    assert np.asarray(Image.open(tmp_path / result[0]["mask"])).tolist() == [[0, 0, 255, 255]] * 2
    json.dumps(result, allow_nan=False)


def test_invalid_threshold_rejected(tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        save_annotation(None, Image.new("RGB", (1, 1)), tmp_path, -1)
