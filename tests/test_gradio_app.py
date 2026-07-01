from __future__ import annotations

import sys

import numpy as np

from sam3_onnx_equiv.gradio_app import (
    BBox,
    bbox_to_point_prompt,
    extract_boxes_from_annotator,
    parse_frame_choice,
    select_directional_windows,
)


def test_extract_boxes_from_annotator_skips_partial_entries() -> None:
    payload = {
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "boxes": [
            {"xmin": 1, "ymin": 2, "xmax": 5, "ymax": 8, "label": "object"},
            {"xmin": 0, "ymin": 0, "xmax": 3},
        ],
    }

    assert extract_boxes_from_annotator(payload) == [BBox(1.0, 2.0, 5.0, 8.0)]


def test_bbox_to_point_prompt_uses_clamped_center() -> None:
    coords, labels = bbox_to_point_prompt(BBox(-10, 20, 110, 60), width=100, height=80)

    np.testing.assert_allclose(coords, np.array([[[49.5 / 100.0, 40.0 / 80.0]]], dtype=np.float32))
    np.testing.assert_array_equal(labels, np.array([[1]], dtype=np.int32))


def test_select_directional_windows_are_capped_and_bidirectional() -> None:
    frames = [f"{i:06d}.png" for i in range(20)]

    forward, reverse = select_directional_windows(frames, base_index=10, max_direction_frames=7)

    assert forward == [10, 11, 12, 13, 14, 15, 16]
    assert reverse == [10, 9, 8, 7, 6, 5, 4]


def test_parse_frame_choice_accepts_dropdown_label() -> None:
    assert parse_frame_choice("12: frame.png") == 12
    assert parse_frame_choice(3) == 3
    assert parse_frame_choice(None) == 0


def test_cuda_execution_provider_order_prefers_cuda(monkeypatch) -> None:
    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    import sam3_onnx_equiv.gradio_app as gradio_app

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)

    assert gradio_app._cuda_execution_providers() == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
