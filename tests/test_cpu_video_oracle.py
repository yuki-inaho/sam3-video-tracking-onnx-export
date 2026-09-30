"""CPU contracts needed by the PyTorch video oracle without checkpoint loading."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

from sam3_onnx_equiv.export._equiv_loader import equiv_sam3_on_path
from sam3_onnx_equiv.path_config import equiv_source_root


def _predictor_class():
    with equiv_sam3_on_path(equiv_source_root()):
        module = importlib.import_module("sam3.model.sam3_tracking_predictor")
    return module.Sam3TrackerPredictor


def test_init_state_stores_frames_on_cpu() -> None:
    predictor = _predictor_class()
    stub = SimpleNamespace(
        device=torch.device("cpu"),
        clear_all_points_in_video=lambda _state: None,
    )

    state = predictor.init_state(stub, video_height=8, video_width=8, num_frames=1)

    assert state["device"] == torch.device("cpu")
    assert state["storage_device"] == torch.device("cpu")


def test_image_feature_uses_tracker_device() -> None:
    predictor = _predictor_class()
    features = {
        "backbone_fpn": [torch.zeros(1, 1, 2, 2)],
        "vision_pos_enc": [torch.zeros(1, 1, 2, 2)],
    }
    stub = SimpleNamespace(
        device=torch.device("cpu"),
        backbone=object(),
        forward_image=lambda _image: features,
        _prepare_backbone_features=lambda _features: (),
    )
    state = {"images": torch.zeros(1, 3, 8, 8), "cached_features": {}}

    image = predictor._get_image_feature(stub, state, frame_idx=0, batch_size=1)[0]

    assert image.device.type == "cpu"
