"""Tests: memory_encoder ONNX export (B-2).

The module under test is SimpleMaskEncoder (mask_downsampler + CXBlock fuser +
out_proj + PositionEmbeddingSine), extracted from the equiv-source tracker.

TDD: write tests first (red), then implement until green.

Test cases:
  test_exports         - ONNX file is generated and passes onnx.checker.
  test_no_complex_ops  - ONNX graph has no Complex / polar / ViewAsComplex /
                         ViewAsReal / If ops.
  test_ort_parity      - ORT (CPUExecutionProvider) output matches PyTorch
                         (equiv-source, float32) at rtol/atol=1e-3 for
                         N_REPEATS seed-fixed inputs (both maskmem_features and
                         maskmem_pos_enc outputs verified).

Fixed-shape I/O contract (B-2 spec, workdoc §9.9):
  Input:
    pix_feat      : float32 (B, 256, 72, 72)   = (1, 256, 72, 72)
    mask_for_mem  : float32 (B, 1, 1008, 1008) = (1, 1, 1008, 1008)
                    (pre-sigmoid; wrapper applies sigmoid internally)

  Output:
    maskmem_features : float32 (B, 64, 72, 72) = (1, 64, 72, 72)
    maskmem_pos_enc  : float32 (B, 64, 72, 72) = (1, 64, 72, 72)
                       (the [0] element of the list returned by SimpleMaskEncoder)

Notes:
  - B=1, H=W=72 (1008 / 14), mask input H=W=1008.
  - The equiv-source SimpleMaskDownSampler uses antialias=False (patched) so
    that aten::_upsample_bilinear2d_aa is not emitted (unsupported in opset 18).
    Numerical difference vs antialias=True for upsample 1008→1152: < 5e-7.
  - PositionEmbeddingSine cache is pre-populated for (72,72) before tracing,
    eliminating the torch.arange dynamic-shape path.
  - skip_mask_sigmoid=True is baked into the wrapper forward (production call
    site in sam3_tracker_base.py:835 always passes skip_mask_sigmoid=True).

Run:
  uv run pytest tests/test_memory_encoder_onnx.py -q
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
ONNX_PATH = SANDBOX_ROOT / "outputs" / "onnx" / "memory_encoder.onnx"

# ---------------------------------------------------------------------------
# Fixed shape constants (B-2 spec)
# ---------------------------------------------------------------------------
B = 1
IN_DIM = 256  # pix_feat channel dim
OUT_DIM = 64  # maskmem_features / pos_enc channel dim
FEAT_H = FEAT_W = 72  # spatial resolution (1008 / 14)
MASK_H = MASK_W = 1008  # high-res mask input resolution

# Parity tolerances
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


def _export_memory_encoder() -> None:
    """Run the export if the ONNX file is absent or invalid."""
    from sam3_onnx_equiv.export.memory_encoder import (  # noqa: PLC0415
        export_memory_encoder,
    )

    export_memory_encoder(
        equiv_source_root=EQUIV_SOURCE_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=ONNX_PATH,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exports() -> None:
    """memory_encoder.onnx is generated and passes onnx.checker.

    RED: ImportError (module not written) or FileNotFoundError.
    GREEN: ONNX file exists and checker passes.
    """
    _requires_checkpoint()
    _export_memory_encoder()

    assert ONNX_PATH.exists(), f"ONNX file not generated at {ONNX_PATH}"
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)


def test_no_complex_ops() -> None:
    """ONNX graph must not contain Complex / polar / ViewAsComplex / ViewAsReal / If ops.

    SimpleMaskEncoder is conv-dominant (no RoPE), so complex ops should not appear.
    'If' ops would indicate un-resolved dynamic control flow (e.g., get_abs_pos or
    cache-miss path in PositionEmbeddingSine).

    RED: ONNX file absent or forbidden ops present.
    GREEN: op_type set does not intersect forbidden set.
    """
    _requires_checkpoint()
    _export_memory_encoder()

    model = onnx.load(str(ONNX_PATH))
    op_types = {node.op_type for node in model.graph.node}
    print(f"op_types in memory_encoder graph: {sorted(op_types)}")

    forbidden = {
        "ComplexFloat",
        "Complex",
        "Polar",
        "ViewAsComplex",
        "ViewAsReal",
        "If",
    }
    forbidden_lower = {o.lower() for o in forbidden}
    found = {op for op in op_types if op.lower() in forbidden_lower}

    assert not found, (
        f"Forbidden ops found in memory_encoder ONNX graph: {found}\n"
        f"Full op_type set: {sorted(op_types)}\n"
        "Check antialias patch and PositionEmbeddingSine cache pre-population."
    )


def test_ort_parity() -> None:
    """ORT output matches PyTorch (equiv-source, float32) for N_REPEATS seeds.

    Both outputs are checked:
      - maskmem_features : (1, 64, 72, 72)
      - maskmem_pos_enc  : (1, 64, 72, 72)

    RED: any mismatch or NaN.
    GREEN: all repeats within rtol/atol=1e-3.

    ORT provider: CPUExecutionProvider (explicit; no auto-selection).
    """
    _requires_checkpoint()
    _export_memory_encoder()

    from sam3_onnx_equiv.export.memory_encoder import (  # noqa: PLC0415
        build_memory_encoder_module,
    )

    wrapper = build_memory_encoder_module(
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
    assert set(input_names) == {"pix_feat", "mask_for_mem"}, (
        f"Unexpected input names: {input_names}"
    )
    assert set(output_names) == {"maskmem_features", "maskmem_pos_enc"}, (
        f"Unexpected output names: {output_names}"
    )

    for repeat_idx in range(N_REPEATS):
        seed = 42 + repeat_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        pix_feat = torch.randn(B, IN_DIM, FEAT_H, FEAT_W, dtype=torch.float32)
        mask_for_mem = torch.randn(B, 1, MASK_H, MASK_W, dtype=torch.float32)

        with torch.no_grad():
            pt_features, pt_pos_enc = wrapper(pix_feat, mask_for_mem)

        pt_features_np = pt_features.cpu().numpy()
        pt_pos_enc_np = pt_pos_enc.cpu().numpy()

        ort_inputs = {
            "pix_feat": pix_feat.numpy(),
            "mask_for_mem": mask_for_mem.numpy(),
        }
        ort_outs = sess.run(None, ort_inputs)
        # ORT returns outputs in the order declared during export.
        # We map by name to avoid ordering assumptions.
        ort_out_map = dict(zip(output_names, ort_outs))
        ort_features_np = ort_out_map["maskmem_features"]
        ort_pos_enc_np = ort_out_map["maskmem_pos_enc"]

        # NaN checks (strict).
        assert not np.any(np.isnan(ort_features_np)), (
            f"[repeat {repeat_idx}, seed={seed}] ORT maskmem_features contains NaN."
        )
        assert not np.any(np.isnan(pt_features_np)), (
            f"[repeat {repeat_idx}, seed={seed}] PyTorch maskmem_features contains NaN."
        )
        assert not np.any(np.isnan(ort_pos_enc_np)), (
            f"[repeat {repeat_idx}, seed={seed}] ORT maskmem_pos_enc contains NaN."
        )
        assert not np.any(np.isnan(pt_pos_enc_np)), (
            f"[repeat {repeat_idx}, seed={seed}] PyTorch maskmem_pos_enc contains NaN."
        )

        # Numerical parity for maskmem_features.
        np.testing.assert_allclose(
            pt_features_np,
            ort_features_np,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"[repeat {repeat_idx}, seed={seed}] maskmem_features mismatch "
                "between PyTorch and ORT"
            ),
        )

        # Numerical parity for maskmem_pos_enc.
        np.testing.assert_allclose(
            pt_pos_enc_np,
            ort_pos_enc_np,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"[repeat {repeat_idx}, seed={seed}] maskmem_pos_enc mismatch "
                "between PyTorch and ORT"
            ),
        )

        print(
            f"[repeat {repeat_idx}, seed={seed}] "
            f"features max_abs_diff={np.abs(pt_features_np - ort_features_np).max():.3e}, "
            f"pos_enc max_abs_diff={np.abs(pt_pos_enc_np - ort_pos_enc_np).max():.3e}"
        )
