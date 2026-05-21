"""Tests: vitdet apply_rotary_enc2 (cos/sin) equals apply_rotary_enc (complex).

This test covers D5-1: the wkentaro-recipe ViT RoPE cos/sin equivalent.

apply_rotary_enc2 is the function added to the equiv-source vitdet.py.
It must:
  (a) produce numerically identical output to apply_rotary_enc (complex) for ViT self-attn.
  (b) use only real-valued arithmetic (no torch.polar / view_as_complex / view_as_real).
  (c) derive freqs_cos, freqs_sin from freqs_cis.real / freqs_cis.imag correctly.

The test imports apply_rotary_enc2 from the generated equiv-source at
outputs/sam3_equiv_source/sam3/model/vitdet.py, and compares it against the
official complex apply_rotary_enc from the SAM3_SRC checkout.

If outputs/sam3_equiv_source does not yet contain apply_rotary_enc2, these tests
will fail with ImportError or AttributeError (expected red state before D5-1 impl).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from sam3_onnx_equiv.path_config import equiv_source_root, sam3_source_root


def _load_official_vitdet(sam3_official: Path) -> types.ModuleType:
    """Load ``sam3.model.vitdet`` directly without triggering ``sam3/__init__.py``.

    Adding the submodule root to ``sys.path`` and importing ``sam3.model.vitdet``
    triggers ``sam3/__init__.py`` which transitively imports training-only code
    (``decord``, etc.).  Loading via ``importlib.util`` avoids that chain.
    """
    vitdet_file = sam3_official / "sam3" / "model" / "vitdet.py"
    spec = importlib.util.spec_from_file_location("sam3.model.vitdet", vitdet_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sam3.model.vitdet", mod)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Official SAM3 imports (read-only source) -- complex reference
# ---------------------------------------------------------------------------
SAM3_OFFICIAL = sam3_source_root()
if not (SAM3_OFFICIAL / "sam3" / "model" / "vitdet.py").exists():
    pytest.skip(f"Official SAM3 vitdet.py not found: {SAM3_OFFICIAL}", allow_module_level=True)

_vitdet_mod = _load_official_vitdet(SAM3_OFFICIAL)
apply_rotary_enc = _vitdet_mod.apply_rotary_enc
compute_axial_cis = _vitdet_mod.compute_axial_cis

# ---------------------------------------------------------------------------
# Equiv-source imports -- cos/sin target (loaded from outputs/sam3_equiv_source)
# ---------------------------------------------------------------------------
EQUIV_VITDET = equiv_source_root() / "sam3" / "model" / "vitdet.py"


def _load_equiv_vitdet():
    """Dynamically load the equiv-source vitdet module.

    The equiv-source vitdet.py has a single relative import (``from .model_misc
    import LayerScale``).  We load ``model_misc`` first via
    ``importlib.util.spec_from_file_location``, then load ``vitdet`` the same
    way.  This avoids adding the equiv-source root to ``sys.path`` and
    triggering ``sam3/__init__.py`` → ``model_builder`` → ``decord``.

    Returns the module object so individual functions can be accessed.
    Raises ImportError with a diagnostic if the file is missing or apply_rotary_enc2
    is absent (expected failure before D5-1 implementation).
    """
    if not EQUIV_VITDET.exists():
        raise ImportError(
            f"Equiv-source vitdet not found at {EQUIV_VITDET}. "
            "Run: uv run python tools/create_equivalent_sam3_source.py "
            '--source-root "$SAM3_SRC" --output-root outputs/sam3_equiv_source'
        )
    equiv_model_dir = EQUIV_VITDET.parent  # outputs/sam3_equiv_source/sam3/model

    # Step 1: load model_misc so that the relative import in vitdet.py resolves.
    misc_file = equiv_model_dir / "model_misc.py"
    if "sam3.model.model_misc" not in sys.modules:
        spec_misc = importlib.util.spec_from_file_location("sam3.model.model_misc", misc_file)
        assert spec_misc is not None and spec_misc.loader is not None
        mod_misc = importlib.util.module_from_spec(spec_misc)
        mod_misc.__package__ = "sam3.model"
        sys.modules["sam3.model.model_misc"] = mod_misc
        spec_misc.loader.exec_module(mod_misc)  # type: ignore[union-attr]

    # Step 2: load vitdet directly (bypasses __init__.py chain).
    cache_key = "sam3.model.vitdet_equiv"  # distinct key avoids collision with official mod
    if cache_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(cache_key, EQUIV_VITDET)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "sam3.model"
        sys.modules[cache_key] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = sys.modules[cache_key]  # type: ignore[assignment]

    if not hasattr(mod, "apply_rotary_enc2"):
        raise ImportError(
            "apply_rotary_enc2 not found in equiv-source vitdet.py. "
            "D5-1 patcher has not been applied yet."
        )
    return mod


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
BATCH = 2
HEADS = 4
END_X = 8
END_Y = 8
SEQ_LEN = END_X * END_Y  # 64
HEAD_DIM = 32  # divisible by 4 for axial RoPE
ATOL = 1e-4
RTOL = 1e-4

torch.manual_seed(0)


def _make_qk(seq_len_q: int, seq_len_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(BATCH, HEADS, seq_len_q, HEAD_DIM), torch.randn(
        BATCH, HEADS, seq_len_k, HEAD_DIM
    )


def _axial_cis() -> torch.Tensor:
    """Complex freqs [SEQ_LEN, HEAD_DIM//2]."""
    return compute_axial_cis(dim=HEAD_DIM, end_x=END_X, end_y=END_Y)


# ---------------------------------------------------------------------------
# (a) apply_rotary_enc2 self-attention: cos/sin vs complex
# ---------------------------------------------------------------------------


def test_apply_rotary_enc2_matches_complex_self_attn():
    """apply_rotary_enc2 (cos/sin) == apply_rotary_enc (complex) for self-attn."""
    mod = _load_equiv_vitdet()
    apply_rotary_enc2 = mod.apply_rotary_enc2

    q, k = _make_qk(SEQ_LEN, SEQ_LEN)
    freqs_cis = _axial_cis()
    freqs_cos = freqs_cis.real
    freqs_sin = freqs_cis.imag

    q_ref, k_ref = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=False)
    q_cos, k_cos = apply_rotary_enc2(q, k, freqs_cos, freqs_sin, repeat_freqs_k=False)

    max_q = (q_ref - q_cos).abs().max().item()
    max_k = (k_ref - k_cos).abs().max().item()
    assert torch.allclose(q_ref, q_cos, rtol=RTOL, atol=ATOL), (
        f"Q mismatch: max_abs_err={max_q:.6e}"
    )
    assert torch.allclose(k_ref, k_cos, rtol=RTOL, atol=ATOL), (
        f"K mismatch: max_abs_err={max_k:.6e}"
    )


# ---------------------------------------------------------------------------
# (b) freqs_cos = freqs_cis.real, freqs_sin = freqs_cis.imag correspondence
# ---------------------------------------------------------------------------


def test_freqs_cos_sin_derived_from_freqs_cis():
    """freqs_cos=freqs_cis.real and freqs_sin=freqs_cis.imag give correct results.

    Verifies that the derivation freqs_cos = freqs_cis.real, freqs_sin = freqs_cis.imag
    is the correct mapping for apply_rotary_enc2.
    """
    mod = _load_equiv_vitdet()
    apply_rotary_enc2 = mod.apply_rotary_enc2

    q, k = _make_qk(SEQ_LEN, SEQ_LEN)
    freqs_cis = _axial_cis()

    # Correct derivation
    q_ref, k_ref = apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=False)
    q_cos, k_cos = apply_rotary_enc2(q, k, freqs_cis.real, freqs_cis.imag, repeat_freqs_k=False)

    assert torch.allclose(q_ref, q_cos, rtol=RTOL, atol=ATOL), (
        "freqs_cos=.real / freqs_sin=.imag derivation is wrong for Q"
    )
    assert torch.allclose(k_ref, k_cos, rtol=RTOL, atol=ATOL), (
        "freqs_cos=.real / freqs_sin=.imag derivation is wrong for K"
    )

    # Incorrect derivation (swapped) must differ -- ensures we are not testing a no-op
    q_swapped, _ = apply_rotary_enc2(q, k, freqs_cis.imag, freqs_cis.real, repeat_freqs_k=False)
    assert not torch.allclose(q_ref, q_swapped, atol=1e-6), (
        "Swapped cos/sin should differ but matched -- test is degenerate"
    )


# ---------------------------------------------------------------------------
# (c) No complex ops in apply_rotary_enc2 source code
# ---------------------------------------------------------------------------


def test_apply_rotary_enc2_has_no_complex_ops():
    """apply_rotary_enc2 code lines must not call torch.polar, view_as_complex, view_as_real.

    Only actual code lines are checked; docstring mentions are ignored.
    """
    import inspect

    mod = _load_equiv_vitdet()
    src = inspect.getsource(mod.apply_rotary_enc2)

    # Strip the docstring: keep only lines that are not inside triple-quoted strings.
    # Simple approach: remove the docstring block (text between first pair of triple quotes).
    import re

    # Remove docstring (first triple-quoted block after the def/signature)
    code_lines = re.sub(r'""".*?"""', "", src, count=1, flags=re.DOTALL)

    forbidden = ("torch.polar", "view_as_complex", "view_as_real")
    for op in forbidden:
        assert op not in code_lines, (
            f"apply_rotary_enc2 code (excluding docstring) must not use '{op}' "
            "(ONNX-incompatible complex op)"
        )


# ---------------------------------------------------------------------------
# (d) _apply_rope branches on freqs_cis attribute presence
# ---------------------------------------------------------------------------


def test_attention_apply_rope_uses_cos_sin_when_freqs_cis_absent():
    """Attention._apply_rope uses apply_rotary_enc2 when freqs_cis buffer is absent.

    Builds a small Attention from the equiv-source and exercises the cos/sin branch
    by deleting freqs_cis and registering freqs_cos / freqs_sin instead.
    """
    mod = _load_equiv_vitdet()
    if not hasattr(mod, "Attention"):
        pytest.skip("Attention class not found in equiv-source vitdet")

    dim = 32
    num_heads = 4
    input_size = (END_X, END_Y)
    attn = mod.Attention(
        dim=dim,
        num_heads=num_heads,
        input_size=input_size,
        use_rope=True,
    )
    attn.eval()

    # Verify freqs_cis exists first (complex path)
    assert hasattr(attn, "freqs_cis"), "Attention should have freqs_cis after __init__"

    # Reference forward: complex path
    x = torch.randn(1, END_X, END_Y, dim)
    with torch.no_grad():
        out_complex = attn(x)

    # Switch to cos/sin path by replacing freqs_cis with freqs_cos / freqs_sin
    freqs_cos = attn.freqs_cis.real.float()
    freqs_sin = attn.freqs_cis.imag.float()
    # Remove complex buffer; register real buffers
    del attn.freqs_cis
    attn.register_buffer("freqs_cos", freqs_cos)
    attn.register_buffer("freqs_sin", freqs_sin)

    assert not hasattr(attn, "freqs_cis"), "freqs_cis should be removed"
    assert hasattr(attn, "freqs_cos"), "freqs_cos should be registered"

    with torch.no_grad():
        out_cossin = attn(x)

    max_err = (out_complex - out_cossin).abs().max().item()
    assert torch.allclose(out_complex, out_cossin, rtol=RTOL, atol=ATOL), (
        f"Attention output mismatch between complex and cos/sin path: max_abs_err={max_err:.6e}"
    )
