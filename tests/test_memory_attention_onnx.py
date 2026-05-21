"""Tests: memory_attention ONNX export (B-1).

The module under test is TransformerEncoderCrossAttention (4-layer self+cross-attn
with cos/sin RoPE), extracted from the equiv-source tracker.

TDD: write tests first (red), then implement until green.

Test cases:
  test_exports         - ONNX file is generated and passes onnx.checker.
  test_no_complex_ops  - ONNX graph has no Complex / polar / ViewAsComplex / ViewAsReal ops.
  test_ort_parity      - ORT (CPUExecutionProvider) output matches PyTorch (equiv-source,
                         float32) at rtol/atol=1e-3 for N_REPEATS seed-fixed inputs.

Fixed-shape contract (see B-1 spec in workdoc §9.9):
  src          : (HW, B, 256)  = (5184, 1, 256)  current-frame vision features
  src_pos      : (HW, B, 256)  = (5184, 1, 256)  positional encoding for src
  prompt       : (M,  B, 64)   = (M_FIXED, 1, 64) memory tokens (padded to M_FIXED)
  prompt_pos   : (M,  B, 64)   = (M_FIXED, 1, 64) pos encoding for prompt
  num_k_exclude_rope: int       scalar — number of obj_ptr tokens excluded from RoPE

Output:
  memory       : (HW, B, 256)  = (5184, 1, 256)  updated features

Notes:
  - B=1 (single object, single frame — production batch size for eval).
  - M_FIXED uses TWO_FRAME_MEM_LEN for the parity/export test (fast, ~10K tokens).
    The full 7-frame config (M=36304) is too heavy for CPU ORT parity; its export
    feasibility is tested separately and reported.
  - All args are float32; num_k_exclude_rope is embedded as a constant in the
    MemoryAttentionWrapper (fixed-shape contract).
  - The equiv-source tracker is loaded with use_rope_real=True via build_tracker().

Run:
  uv run pytest tests/test_memory_attention_onnx.py -q
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from sam3_onnx_equiv.path_config import checkpoint_path, equiv_source_root, onnx_dir, repo_root

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SANDBOX_ROOT = repo_root()
EQUIV_SOURCE_ROOT = equiv_source_root()
CHECKPOINT_PATH = checkpoint_path()
ONNX_PATH = onnx_dir() / "memory_attention.onnx"

# ---------------------------------------------------------------------------
# Fixed shape constants (B-1 spec)
# ---------------------------------------------------------------------------
HW = 72 * 72  # 5184 spatial tokens per frame
B = 1
D_MODEL = 256  # current-frame feature dim
MEM_DIM = 64  # memory feature dim (kv_in_dim for cross-attn)

# Number of frames for fast export/parity test (2 frames, no obj ptrs).
# 2 * HW = 10368 memory tokens. Full config: 7 * HW + 16 = 36304.
TWO_FRAME_MEM_LEN = 2 * HW  # 10368

# Production memory length: 7 * HW frames + up to 16 obj_ptr tokens.
# Each obj_ptr is expanded by C // mem_dim = 256 // 64 = 4 tokens → max 64.
FULL_MEM_LEN = 7 * HW + 16 * (D_MODEL // MEM_DIM)  # 5184*7 + 64 = 36352
# NOTE: workdoc §9.9 says 36304 (obj_ptr≤16 tokens directly, not expanded).
# The expanded form is 16 * 4 = 64. We use 36352 for full export test.

NUM_K_EXCLUDE_ROPE = 0  # no obj_ptr tokens in the test (simplest case)

OPSET_VERSION = 18

# Parity tolerances
RTOL = 1e-3
ATOL = 1e-3
N_REPEATS = 3  # fewer than image encoder because CPU ORT is slow for 10K tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _requires_checkpoint() -> None:
    """Skip if checkpoint or equiv source are absent."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not EQUIV_SOURCE_ROOT.exists():
        pytest.skip(f"Equiv source not found: {EQUIV_SOURCE_ROOT}")


def _export_memory_attention(mem_len: int = TWO_FRAME_MEM_LEN) -> None:
    """Run the export if the ONNX file is absent."""
    from sam3_onnx_equiv.export.memory_attention import (  # noqa: PLC0415
        export_memory_attention,
    )

    export_memory_attention(
        equiv_source_root=EQUIV_SOURCE_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=ONNX_PATH,
        mem_len=mem_len,
        num_k_exclude_rope=NUM_K_EXCLUDE_ROPE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exports() -> None:
    """memory_attention.onnx is generated and passes onnx.checker.

    RED: ImportError (module not written) or FileNotFoundError.
    GREEN: ONNX file exists and checker passes.
    """
    _requires_checkpoint()
    _export_memory_attention()

    assert ONNX_PATH.exists(), f"ONNX file not generated at {ONNX_PATH}"
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)


def test_no_complex_ops() -> None:
    """ONNX graph must not contain Complex / polar / ViewAsComplex / ViewAsReal ops.

    RED: ONNX file absent or complex ops present.
    GREEN: op_type set does not intersect forbidden set.
    """
    _requires_checkpoint()
    _export_memory_attention()

    model = onnx.load(str(ONNX_PATH))
    op_types = {node.op_type for node in model.graph.node}
    print(f"op_types in memory_attention graph: {sorted(op_types)}")

    forbidden = {
        "ComplexFloat",
        "Complex",
        "Polar",
        "ViewAsComplex",
        "ViewAsReal",
    }
    forbidden_lower = {o.lower() for o in forbidden}
    found = {op for op in op_types if op.lower() in forbidden_lower}

    assert not found, (
        f"Complex ops found in memory_attention ONNX graph: {found}\n"
        f"Full op_type set: {sorted(op_types)}\n"
        "Check that tracker RoPEAttention has use_rope_real=True in equiv source."
    )


def test_ort_parity() -> None:
    """ORT output matches PyTorch (equiv-source, float32) for N_REPEATS seeds.

    RED: any mismatch or NaN.
    GREEN: all repeats within rtol/atol=1e-3.

    ORT provider: CPUExecutionProvider (explicit; no auto-selection).
    """
    _requires_checkpoint()
    _export_memory_attention()

    from sam3_onnx_equiv.export.memory_attention import (  # noqa: PLC0415
        build_memory_attention_module,
    )

    encoder = build_memory_attention_module(
        equiv_source_root=EQUIV_SOURCE_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        mem_len=TWO_FRAME_MEM_LEN,
        num_k_exclude_rope=NUM_K_EXCLUDE_ROPE,
    )
    encoder.eval()

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"ORT inputs:  {input_names}")
    print(f"ORT outputs: {output_names}")

    for repeat_idx in range(N_REPEATS):
        seed = 42 + repeat_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        src = torch.randn(HW, B, D_MODEL, dtype=torch.float32)
        src_pos = torch.randn(HW, B, D_MODEL, dtype=torch.float32)
        prompt = torch.randn(TWO_FRAME_MEM_LEN, B, MEM_DIM, dtype=torch.float32)
        prompt_pos = torch.randn(TWO_FRAME_MEM_LEN, B, MEM_DIM, dtype=torch.float32)

        with torch.no_grad():
            pt_out = encoder(src, src_pos, prompt, prompt_pos)

        pt_np = pt_out.cpu().numpy()

        ort_inputs = {
            "src": src.numpy(),
            "src_pos": src_pos.numpy(),
            "prompt": prompt.numpy(),
            "prompt_pos": prompt_pos.numpy(),
        }
        ort_out = sess.run(None, ort_inputs)
        ort_np = ort_out[0]

        assert not np.any(np.isnan(ort_np)), (
            f"[repeat {repeat_idx}, seed={seed}] ORT memory output contains NaN."
        )
        assert not np.any(np.isnan(pt_np)), (
            f"[repeat {repeat_idx}, seed={seed}] PyTorch memory output contains NaN."
        )

        np.testing.assert_allclose(
            pt_np,
            ort_np,
            rtol=RTOL,
            atol=ATOL,
            err_msg=(
                f"[repeat {repeat_idx}, seed={seed}] memory output mismatch between PyTorch and ORT"
            ),
        )
