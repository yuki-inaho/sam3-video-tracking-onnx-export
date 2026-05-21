"""Tests: official SAM3 apply_rotary_enc_real (cos/sin) equals apply_rotary_enc (complex).

Covers:
    (a) Self-attention case (repeat_freqs_k=False): real == complex, ViT-style input.
    (b) Cross-attention/memory case (repeat_freqs_k=True): key seq-len is integer multiple
        of query seq-len — verifying the repeat_k path is equivalent.
    (c) PoC apply_rotary_enc_real_safe vs official apply_rotary_enc (complex reference):
        the lifted PoC implementation agrees with the official complex formulation.

All tests use compute_axial_cis / apply_rotary_enc / apply_rotary_enc_real from the
official SAM3 source configured by SAM3_SRC (read-only). The PoC real-valued helpers are
imported from sam3_onnx_equiv.rope_equivalent (promoted from temp/).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from sam3_onnx_equiv.path_config import sam3_source_root


def _load_rope_module(sam3_path: Path) -> types.ModuleType:
    """Load ``sam3.sam.rope`` directly without triggering ``sam3/__init__.py``.

    Adding the submodule root to ``sys.path`` and importing ``sam3.sam.rope``
    triggers ``sam3/__init__.py`` which transitively imports training-only code
    (``decord``, etc.).  Loading via ``importlib.util`` avoids that chain.
    """
    rope_file = sam3_path / "sam3" / "sam" / "rope.py"
    spec = importlib.util.spec_from_file_location("sam3.sam.rope", rope_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sam3.sam.rope", mod)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Official SAM3 imports (read-only source)
# ---------------------------------------------------------------------------
SAM3_PATH = sam3_source_root()
if not (SAM3_PATH / "sam3" / "sam" / "rope.py").exists():
    pytest.skip(f"Official SAM3 rope.py not found: {SAM3_PATH}", allow_module_level=True)

_rope_mod = _load_rope_module(SAM3_PATH)
apply_rotary_enc = _rope_mod.apply_rotary_enc
apply_rotary_enc_real = _rope_mod.apply_rotary_enc_real
compute_axial_cis = _rope_mod.compute_axial_cis

# PoC helpers promoted to src/
from sam3_onnx_equiv.rope_equivalent import (  # noqa: E402
    apply_rotary_enc_real_safe,
    compute_axial_cis_real,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------
BATCH = 2
HEADS = 4
END_X = 8
END_Y = 8
SEQ_LEN = END_X * END_Y  # 64
HEAD_DIM = 32  # must be divisible by 4 for axial RoPE
ATOL = 1e-4
RTOL = 1e-4

torch.manual_seed(42)


def _make_qk(
    seq_len_q: int, seq_len_k: int, head_dim: int = HEAD_DIM
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(BATCH, HEADS, seq_len_q, head_dim)
    k = torch.randn(BATCH, HEADS, seq_len_k, head_dim)
    return q, k


def _axial_cis(device: torch.device | None = None) -> torch.Tensor:
    """Complex freqs shaped [SEQ_LEN, HEAD_DIM//2]."""
    return compute_axial_cis(dim=HEAD_DIM, end_x=END_X, end_y=END_Y, device=device)


def _axial_cis_real(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) pair each shaped [SEQ_LEN, HEAD_DIM//2]."""
    return compute_axial_cis_real(dim=HEAD_DIM, end_x=END_X, end_y=END_Y, device=device)


# ---------------------------------------------------------------------------
# (a) Official real vs official complex — self-attention (no repeat_k)
# ---------------------------------------------------------------------------


def test_official_real_matches_complex_self_attn():
    """apply_rotary_enc_real == apply_rotary_enc for self-attention (repeat_freqs_k=False)."""
    q, k = _make_qk(SEQ_LEN, SEQ_LEN)
    freqs_cis = _axial_cis()
    freqs_real = freqs_cis.real
    freqs_imag = freqs_cis.imag

    q_complex, k_complex = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=False)
    q_real_out, k_real_out = apply_rotary_enc_real(
        q, k, freqs_real, freqs_imag, repeat_freqs_k=False
    )

    assert torch.allclose(q_complex, q_real_out, rtol=RTOL, atol=ATOL), (
        f"Q mismatch: max_abs_err={(q_complex - q_real_out).abs().max().item():.6e}"
    )
    assert torch.allclose(k_complex, k_real_out, rtol=RTOL, atol=ATOL), (
        f"K mismatch: max_abs_err={(k_complex - k_real_out).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# (b) Official real vs official complex — cross-attention (repeat_freqs_k=True)
# ---------------------------------------------------------------------------


def test_official_real_matches_complex_cross_attn_repeat_k():
    """apply_rotary_enc_real == apply_rotary_enc for cross-attention (repeat_freqs_k=True).

    Key seq-len is 2x query seq-len, matching memory cross-attention in SAM3 tracker.
    """
    q, k = _make_qk(SEQ_LEN, SEQ_LEN * 2)
    freqs_cis = _axial_cis()
    freqs_real = freqs_cis.real
    freqs_imag = freqs_cis.imag

    q_complex, k_complex = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=True)
    q_real_out, k_real_out = apply_rotary_enc_real(
        q, k, freqs_real, freqs_imag, repeat_freqs_k=True
    )

    assert torch.allclose(q_complex, q_real_out, rtol=RTOL, atol=ATOL), (
        f"Q mismatch (repeat_k): max_abs_err={(q_complex - q_real_out).abs().max().item():.6e}"
    )
    assert torch.allclose(k_complex, k_real_out, rtol=RTOL, atol=ATOL), (
        f"K mismatch (repeat_k): max_abs_err={(k_complex - k_real_out).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# (c) PoC apply_rotary_enc_real_safe vs official complex — self-attention
# ---------------------------------------------------------------------------


def test_poc_real_safe_matches_official_complex_self_attn():
    """PoC apply_rotary_enc_real_safe agrees with official apply_rotary_enc (complex)."""
    q, k = _make_qk(SEQ_LEN, SEQ_LEN)
    freqs_cis = _axial_cis()
    cos, sin = _axial_cis_real()

    q_complex, k_complex = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=False)
    q_poc, k_poc = apply_rotary_enc_real_safe(q, k, cos, sin, repeat_freqs_k=False)

    assert torch.allclose(q_complex, q_poc, rtol=RTOL, atol=ATOL), (
        f"PoC Q mismatch: max_abs_err={(q_complex - q_poc).abs().max().item():.6e}"
    )
    assert torch.allclose(k_complex, k_poc, rtol=RTOL, atol=ATOL), (
        f"PoC K mismatch: max_abs_err={(k_complex - k_poc).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# (c-ext) PoC real_safe vs official complex — cross-attention (repeat_k)
# ---------------------------------------------------------------------------


def test_poc_real_safe_matches_official_complex_cross_attn_repeat_k():
    """PoC apply_rotary_enc_real_safe agrees with official complex for repeat_k=True."""
    q, k = _make_qk(SEQ_LEN, SEQ_LEN * 3)
    freqs_cis = _axial_cis()
    cos, sin = _axial_cis_real()

    q_complex, k_complex = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=True)
    q_poc, k_poc = apply_rotary_enc_real_safe(q, k, cos, sin, repeat_freqs_k=True)

    assert torch.allclose(q_complex, q_poc, rtol=RTOL, atol=ATOL), (
        f"PoC Q mismatch (repeat_k=3): max_abs_err={(q_complex - q_poc).abs().max().item():.6e}"
    )
    assert torch.allclose(k_complex, k_poc, rtol=RTOL, atol=ATOL), (
        f"PoC K mismatch (repeat_k=3): max_abs_err={(k_complex - k_poc).abs().max().item():.6e}"
    )
