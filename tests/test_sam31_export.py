"""SAM 3.1 ONNX artifacts must implement the four multiplex tensor boundaries."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from sam3_onnx_equiv.export.sam31 import (
    MultiplexMaskHead,
    MultiplexMemoryAttention,
    MultiplexMemoryEncoder,
    TriHeadEncoder,
)
from sam3_onnx_equiv.path_config import model_paths
from sam3_onnx_equiv.sam31_model import build_sam31_tracker


def _attention_case(
    filename: str, memory_frames: int, pointer_frames: int
) -> tuple[str, list[tuple[int, ...]], list[str], float]:
    memory_seq = memory_frames * 5184
    pointer_tokens = pointer_frames * 16
    return (
        filename,
        [
            (5184, 1, 256),
            (5184, 1, 256),
            (memory_seq, 1, 256),
            (memory_seq + pointer_tokens, 1, 256),
            (5184, 1, 256),
            (5184, 1, 256),
            (memory_seq, 1, 256),
            (memory_seq + pointer_tokens, 1, 256),
        ],
        [
            "image",
            "src",
            "memory_image",
            "memory",
            "image_pos",
            "src_pos",
            "memory_image_pos",
            "memory_pos",
        ],
        3e-4,
    )


@pytest.mark.parametrize(
    "name",
    [
        "trihead_image_encoder.onnx",
        "multiplex_memory_attention.onnx",
        "multiplex_memory_attention_m1_p2.onnx",
        "multiplex_memory_attention_m2_p1.onnx",
        "multiplex_memory_attention_m2_p2.onnx",
        "multiplex_mask_decoder.onnx",
        "multiplex_memory_encoder.onnx",
    ],
)
def test_sam31_onnx_exists_and_checks(name: str) -> None:
    path = model_paths("sam31").onnx_dir / name
    assert path.is_file(), f"Missing SAM 3.1 export: {path}"
    onnx.checker.check_model(str(path))


@pytest.fixture(scope="module")
def tracker() -> torch.nn.Module:
    return build_sam31_tracker(use_rope_real=True)


@pytest.mark.parametrize(
    ("name", "input_shapes", "input_names", "atol"),
    [
        (
            "multiplex_memory_encoder.onnx",
            [(1, 256, 72, 72), (1, 32, 1008, 1008)],
            ["pix_feat", "bucket_masks"],
            1e-3,
        ),
        (
            "multiplex_mask_decoder.onnx",
            [
                (1, 256, 72, 72),
                (1, 256, 72, 72),
                (1, 32, 288, 288),
                (1, 64, 144, 144),
                (1, 16, 256),
            ],
            ["image_embeddings", "image_pe", "high_res_0", "high_res_1", "slot_embeddings"],
            3e-4,
        ),
        _attention_case("multiplex_memory_attention.onnx", 1, 1),
        _attention_case("multiplex_memory_attention_m1_p2.onnx", 1, 2),
        _attention_case("multiplex_memory_attention_m2_p1.onnx", 2, 1),
        _attention_case("multiplex_memory_attention_m2_p2.onnx", 2, 2),
        (
            "trihead_image_encoder.onnx",
            [(1, 3, 1008, 1008)],
            ["pixels"],
            4e-4,
        ),
    ],
)
def test_sam31_ort_matches_official_pytorch(
    tracker: torch.nn.Module,
    name: str,
    input_shapes: list[tuple[int, ...]],
    input_names: list[str],
    atol: float,
) -> None:
    """Compare live CPU ORT outputs against the official checkpointed modules."""
    generator = torch.Generator().manual_seed(31)
    inputs = tuple(torch.randn(shape, generator=generator) * 0.1 for shape in input_shapes)
    if name == "multiplex_memory_encoder.onnx":
        module = MultiplexMemoryEncoder(tracker.maskmem_backbone)
        # The 16 active mask channels and 16 conditioning channels are distinct.
        inputs[1][:, 2:16] = 0
        inputs[1][:, 18:] = 0
    elif name == "multiplex_mask_decoder.onnx":
        module = MultiplexMaskHead(tracker.sam_mask_decoder)
    elif name.startswith("multiplex_memory_attention"):
        pointer_tokens = inputs[3].shape[0] - inputs[2].shape[0]
        module = MultiplexMemoryAttention(
            tracker.transformer.encoder, pointer_tokens=pointer_tokens
        )
    else:
        module = TriHeadEncoder(tracker)

    with torch.inference_mode():
        expected = module(*inputs)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_paths("sam31").onnx_dir / name),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    feed = {key: value.numpy() for key, value in zip(input_names, inputs, strict=True)}
    feed = {entry.name: feed[entry.name] for entry in session.get_inputs()}
    actual = session.run(None, feed)
    assert len(actual) == len(expected)
    for info, got, want in zip(session.get_outputs(), actual, expected, strict=True):
        want_array = want.numpy()
        assert got.shape == want_array.shape, info.name
        max_abs = float(np.max(np.abs(got - want_array)))
        print(f"{name} {info.name}: max_abs={max_abs:.6g}")
        np.testing.assert_allclose(got, want_array, atol=atol, rtol=atol, err_msg=info.name)
