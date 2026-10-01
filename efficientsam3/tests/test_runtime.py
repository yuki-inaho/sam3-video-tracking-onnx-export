import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import runtime
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


@pytest.mark.parametrize(
    "key", ["context", "variant", "sequence_mode", "checkpoint_sha256", "resolution", "graphs"]
)
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


@pytest.mark.parametrize("size", [(0, 11), (17, 0)])
def test_empty_image_rejected(size):
    with pytest.raises(ValueError, match="Empty image"):
        preprocess(Image.new("RGB", size))


@pytest.mark.parametrize("threads", [0, -1])
def test_invalid_threads_rejected_before_model_loading(tmp_path, threads):
    with pytest.raises(ValueError, match="threads must be positive"):
        runtime.Runtime(tmp_path, threads)


def test_cli_empty_image_list_rejected_before_model_loading(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runtime", "--images", "--text", "dog"])
    with pytest.raises(SystemExit) as error:
        runtime.main()
    assert error.value.code == 2
    assert "expected at least one argument" in capsys.readouterr().err


def test_cli_preserves_explicit_frame_order_and_writes_paired_masks(tmp_path, monkeypatch):
    # Reverse lexical order and different image sizes catch sorting and index mixups.
    first, second = tmp_path / "z.png", tmp_path / "a.png"
    Image.new("RGB", (6, 4), (20, 0, 0)).save(first)
    Image.new("RGB", (4, 6), (80, 0, 0)).save(second)
    calls = []

    class FakeRuntime:
        def __init__(self, directory, threads):
            assert threads == 4

        def raw(self, image, prompt):
            calls.append((image.size, image.getpixel((0, 0))[0], prompt))
            masks = np.array([[[[-1, 1], [-1, 1]]]], dtype=np.float32)
            if image.getpixel((0, 0))[0] == 80:
                masks = -masks
            return (
                np.array([[[0.5, 0.5, 1.0, 1.0]]]),
                np.array([[[10.0]]]),
                np.array([[10.0]]),
                masks,
            ), {"vision_ms": 1.0, "text_ms": 0.0, "grounding_ms": 1.0}

    output = tmp_path / "annotations"
    monkeypatch.setattr(runtime, "Runtime", FakeRuntime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime",
            "--images",
            str(first),
            str(second),
            "--text",
            "dog",
            "--output",
            str(output),
            "--threads",
            "4",
        ],
    )
    runtime.main()
    report = json.loads((output / "annotations.json").read_text())
    assert calls == [((6, 4), 20, "dog"), ((4, 6), 80, "dog")]
    assert [f["frame_index"] for f in report["frames"]] == [0, 1]
    for index, size in enumerate([(6, 4), (4, 6)]):
        frame = report["frames"][index]
        assert (frame["width"], frame["height"]) == size
        detection = frame["detections"][0]
        with Image.open(output / f"frame_{index:06d}" / detection["mask"]) as image:
            assert image.size == size
            assert (image.getpixel((size[0] - 1, 0)) > 0) == (index == 0)
            assert (image.getpixel((0, 0)) > 0) == (index == 1)
