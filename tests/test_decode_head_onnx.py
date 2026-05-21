"""Tests: SAM decode head ONNX export (B-3).

Covers prompt_encoder + mask_decoder + obj_ptr_proj (object pointer MLP).
This is the per-frame mask/score/obj_ptr generator used in video tracking.

TDD: write tests first (red), then implement until green.

RoPE finding: sam_mask_decoder uses TwoWayTransformer (standard Attention,
no RoPEAttention) — zero RoPE involvement. No replace_rope_freqs needed.

Fixed-shape I/O contract (B-3 spec):
  Inputs:
    image_embeddings  : float32 (1, 256, 72, 72)   post-memory-attention features
    high_res_feat0    : float32 (1, 32, 288, 288)  conv_s0-projected FPN level 0
    high_res_feat1    : float32 (1, 64, 144, 144)  conv_s1-projected FPN level 1
    point_coords      : float32 (1, 1, 2)          absolute pixel coords (x,y)
    point_labels      : int32   (1, 1)             1=pos,0=neg,-1=pad
    mask_input        : float32 (1, 1, 288, 288)   prior mask logits (4x embedding)
    has_mask_input    : float32 (1,)               1.0 if mask_input is valid

  Outputs:
    low_res_masks         : float32 (1, 1, 288, 288)
    iou_scores            : float32 (1, 1)
    object_score_logits   : float32 (1, 1)
    obj_ptr               : float32 (1, 256)

Notes:
  - multimask_output=False baked in (single-mask output).
  - is_obj_appearing = object_score_logits > 0  (data-dependent torch.where — OK).
  - obj_ptr blended: is_obj * proj(token) + (1-is_obj) * no_obj_ptr.
  - high_res_features already conv_s0/s1 projected (not raw backbone_fpn).
  - No RoPE in decode head → no replace_rope_freqs needed.
  - opset_version=18, dynamo=False.

Run:
  uv run pytest tests/test_decode_head_onnx.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SANDBOX_ROOT = Path("/home/inaho-omen/Project/sam3_onnx_sandbox")
EQUIV_SOURCE_ROOT = SANDBOX_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = SANDBOX_ROOT / "models" / "sam3.pt"
ONNX_PATH = SANDBOX_ROOT / "outputs" / "onnx" / "decode_head.onnx"

# ---------------------------------------------------------------------------
# Fixed shape constants (B-3 spec)
# ---------------------------------------------------------------------------
B = 1
EMB_DIM = 256  # image_embeddings channel dim
EMB_H = EMB_W = 72  # spatial resolution (1008 / 14)
HRF0_C = 32  # high_res_feat0 channels (conv_s0: 256→32)
HRF0_H = HRF0_W = 288  # high_res_feat0 spatial (72 * 4)
HRF1_C = 64  # high_res_feat1 channels (conv_s1: 256→64)
HRF1_H = HRF1_W = 144  # high_res_feat1 spatial (72 * 2)
N_POINTS = 1  # number of points (fixed)
MASK_IN_H = MASK_IN_W = 288  # mask_input spatial (4 * 72)
LOW_RES_H = LOW_RES_W = 288  # low_res_masks spatial (4 * 72)

# Parity tolerances (float32, CPU)
RTOL = 1e-3
ATOL = 1e-3
N_REPEATS = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _requires_checkpoint() -> None:
    """Skip if checkpoint or equiv source are absent."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not EQUIV_SOURCE_ROOT.exists():
        pytest.skip(f"Equiv source not found: {EQUIV_SOURCE_ROOT}")


def _export_decode_head() -> None:
    """Run the export if the ONNX file is absent or invalid."""
    from sam3_onnx_equiv.export.decode_head import export_decode_head  # noqa: PLC0415

    export_decode_head(
        equiv_source_root=EQUIV_SOURCE_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=ONNX_PATH,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exports() -> None:
    """decode_head.onnx is generated and passes onnx.checker.

    RED: ImportError (module not written) or FileNotFoundError.
    GREEN: ONNX file exists and checker passes.
    """
    _requires_checkpoint()
    _export_decode_head()

    assert ONNX_PATH.exists(), f"ONNX file not generated at {ONNX_PATH}"
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)


def test_no_complex_ops() -> None:
    """ONNX graph must not contain Complex / polar / ViewAsComplex / ViewAsReal / If ops.

    decode_head uses TwoWayTransformer (standard Attention, no RoPE) so
    complex ops must be absent.  'If' would indicate unresolved Python
    control flow (multimask_output branch or dynamic_multimask_via_stability).

    RED: ONNX absent or forbidden ops found.
    GREEN: op_type set does not intersect forbidden set.
    """
    _requires_checkpoint()
    _export_decode_head()

    model = onnx.load(str(ONNX_PATH))
    op_types = {node.op_type for node in model.graph.node}
    print(f"op_types in decode_head graph: {sorted(op_types)}")

    forbidden = {
        "ComplexFloat",
        "Complex",
        "Polar",
        "ViewAsComplex",
        "ViewAsReal",
        "If",
    }
    found = {op for op in op_types if op.lower() in {f.lower() for f in forbidden}}

    assert not found, (
        f"Forbidden ops found in decode_head ONNX graph: {found}\n"
        f"Full op_type set: {sorted(op_types)}\n"
        "Check multimask_output baking and dynamic_multimask_via_stability=False."
    )


def test_ort_parity() -> None:
    """ORT output matches PyTorch (equiv-source, float32) for N_REPEATS seeds.

    All four outputs are checked:
      - low_res_masks        : (1, 1, 288, 288)
      - iou_scores           : (1, 1)
      - object_score_logits  : (1, 1)
      - obj_ptr              : (1, 256)

    RED: any mismatch or NaN.
    GREEN: all repeats within rtol/atol=1e-3.

    ORT provider: CPUExecutionProvider (explicit; no auto-selection).
    """
    _requires_checkpoint()
    _export_decode_head()

    from sam3_onnx_equiv.export.decode_head import (  # noqa: PLC0415
        build_decode_head_module,
    )

    wrapper = build_decode_head_module(
        equiv_source_root=EQUIV_SOURCE_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    wrapper.eval()

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"ORT inputs:  {input_names}")
    print(f"ORT outputs: {output_names}")

    expected_inputs = {
        "image_embeddings",
        "high_res_feat0",
        "high_res_feat1",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
    }
    expected_outputs = {
        "low_res_masks",
        "iou_scores",
        "object_score_logits",
        "obj_ptr",
    }
    assert set(input_names) == expected_inputs, f"Unexpected input names: {input_names}"
    assert set(output_names) == expected_outputs, f"Unexpected output names: {output_names}"

    for repeat_idx in range(N_REPEATS):
        seed = 42 + repeat_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        image_embeddings = torch.randn(B, EMB_DIM, EMB_H, EMB_W, dtype=torch.float32)
        high_res_feat0 = torch.randn(B, HRF0_C, HRF0_H, HRF0_W, dtype=torch.float32)
        high_res_feat1 = torch.randn(B, HRF1_C, HRF1_H, HRF1_W, dtype=torch.float32)
        point_coords = torch.rand(B, N_POINTS, 2, dtype=torch.float32) * 1008.0
        point_labels = torch.ones(B, N_POINTS, dtype=torch.int32)
        mask_input = torch.randn(B, 1, MASK_IN_H, MASK_IN_W, dtype=torch.float32)
        has_mask_input = torch.ones(1, dtype=torch.float32)

        with torch.no_grad():
            pt_masks, pt_iou, pt_obj_logits, pt_obj_ptr = wrapper(
                image_embeddings,
                high_res_feat0,
                high_res_feat1,
                point_coords,
                point_labels,
                mask_input,
                has_mask_input,
            )

        ort_inputs = {
            "image_embeddings": image_embeddings.numpy(),
            "high_res_feat0": high_res_feat0.numpy(),
            "high_res_feat1": high_res_feat1.numpy(),
            "point_coords": point_coords.numpy(),
            "point_labels": point_labels.numpy(),
            "mask_input": mask_input.numpy(),
            "has_mask_input": has_mask_input.numpy(),
        }
        ort_outs = sess.run(None, ort_inputs)
        ort_out_map = dict(zip(output_names, ort_outs))

        outputs_to_check = [
            ("low_res_masks", pt_masks),
            ("iou_scores", pt_iou),
            ("object_score_logits", pt_obj_logits),
            ("obj_ptr", pt_obj_ptr),
        ]
        for out_name, pt_tensor in outputs_to_check:
            pt_np = pt_tensor.cpu().numpy()
            ort_np = ort_out_map[out_name]

            # NaN checks (strict).
            assert not np.any(np.isnan(ort_np)), (
                f"[repeat {repeat_idx}, seed={seed}] ORT {out_name} contains NaN."
            )
            assert not np.any(np.isnan(pt_np)), (
                f"[repeat {repeat_idx}, seed={seed}] PyTorch {out_name} contains NaN."
            )

            # Numerical parity.
            np.testing.assert_allclose(
                pt_np,
                ort_np,
                rtol=RTOL,
                atol=ATOL,
                err_msg=(
                    f"[repeat {repeat_idx}, seed={seed}] {out_name} mismatch "
                    "between PyTorch (equiv-source) and ORT."
                ),
            )

            max_diff = float(np.abs(pt_np - ort_np).max())
            print(f"[repeat {repeat_idx}, seed={seed}] {out_name} max_abs_diff={max_diff:.3e}")
