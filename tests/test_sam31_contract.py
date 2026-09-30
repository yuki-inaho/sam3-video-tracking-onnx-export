"""Contracts pinned to the official SAM 3.1 Object Multiplex checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sam3_onnx_equiv import path_config
from sam3_onnx_equiv.sam31_model import build_sam31_tracker
from sam3_onnx_equiv.sam31_source_patcher import create_sam31_cpu_source_copy


@pytest.fixture(scope="module")
def sam31_state() -> dict[str, torch.Tensor]:
    path = path_config.repo_root() / "models" / "sam3.1_multiplex.pt"
    assert path.is_file(), f"Download facebook/sam3.1/sam3.1_multiplex.pt to {path}"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("model", checkpoint)


def test_version_paths_select_official_sam31() -> None:
    paths = path_config.model_paths("sam31")
    assert paths.source_root == path_config.repo_root() / "sam31"
    expected_checkpoint = path_config.repo_root() / "models/sam3.1_multiplex.pt"
    assert paths.checkpoint_path == expected_checkpoint.resolve()
    assert paths.onnx_dir == path_config.repo_root() / "outputs/onnx_sam31"
    assert (paths.source_root / "RELEASE_SAM3p1.md").is_file()
    assert (paths.source_root / "sam3/model/video_tracking_multiplex.py").is_file()


def test_default_paths_keep_sam3() -> None:
    paths = path_config.model_paths()
    assert paths.source_root == path_config.sam3_source_root()
    assert paths.checkpoint_path == path_config.checkpoint_path()
    assert paths.onnx_dir == path_config.onnx_dir()


def test_multiplex_checkpoint_contract(sam31_state: dict[str, torch.Tensor]) -> None:
    assert len(sam31_state) == 1623
    assert sam31_state[
        "tracker.model.maskmem_backbone.mask_downsampler.encoder.0.weight"
    ].shape == (16, 32, 3, 3)
    assert sam31_state["tracker.model.sam_mask_decoder.mask_tokens.weight"].shape == (48, 256)
    assert sam31_state["tracker.model.sam_mask_decoder.iou_token.weight"].shape == (16, 256)
    assert any(
        key.startswith("detector.backbone.vision_backbone.propagation_convs") for key in sam31_state
    )


def test_sam31_builder_is_distinct() -> None:
    source = path_config.model_paths("sam31").source_root / "sam3/model_builder.py"
    assert source.is_file()
    text = Path(source).read_text(encoding="utf-8")
    assert "def build_sam3_multiplex_video_model(" in text
    assert "def build_sam3_multiplex_video_predictor(" in text


def test_sam31_tracker_loads_every_official_weight() -> None:
    model = build_sam31_tracker()
    assert len(model.state_dict()) == 931
    assert model.multiplex_count == 16
    assert next(model.parameters()).device.type == "cpu"


def test_source_copy_rejects_overlapping_directories() -> None:
    source = path_config.model_paths("sam31").source_root
    with pytest.raises(ValueError, match="separate directories"):
        create_sam31_cpu_source_copy(source, source)
