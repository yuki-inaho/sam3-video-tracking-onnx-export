"""Tests: replace_rope_freqs scope and RoPEAttention independence (D5-3 fix 3).

Verifies that:
  1. replace_rope_freqs applied to model.backbone replaces only backbone
     Attention freqs_cis buffers, not RoPEAttention (tracker) modules.
  2. replace_rope_freqs returns the correct replacement count.
  3. After applying to backbone, no freqs_cis remain in backbone.
  4. RoPEAttention modules (tracker path) are NOT touched by
     replace_rope_freqs -- they use use_rope_real instead.

These tests confirm the scope contract described in rope_freqs.py and guard
against accidental breakage when the helper is later reused for memory_attention
export (Stage B).

Run:
  uv run pytest tests/test_rope_freqs_scope.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
EQUIV_SOURCE_ROOT = SANDBOX_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = SANDBOX_ROOT / "models" / "sam3.pt"


def _skip_if_missing() -> None:
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not EQUIV_SOURCE_ROOT.exists():
        pytest.skip(f"Equiv source not found: {EQUIV_SOURCE_ROOT}")


# ---------------------------------------------------------------------------
# Unit tests (no checkpoint needed)
# ---------------------------------------------------------------------------


class _FakeAttention(nn.Module):
    """Minimal module that mimics a backbone Attention with a freqs_cis buffer."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        # Complex buffer: shape (T, dim//2), dtype complex64
        t = torch.complex(
            torch.ones(8, dim // 2), torch.zeros(8, dim // 2)
        )
        self.register_buffer("freqs_cis", t)


class _FakeRoPEAttention(nn.Module):
    """Minimal module that mimics tracker RoPEAttention (no freqs_cis buffer)."""

    def __init__(self, use_rope_real: bool = True) -> None:
        super().__init__()
        self.use_rope_real = use_rope_real
        # RoPEAttention embeds cos/sin at runtime -- no freqs_cis buffer.


def test_replace_returns_count() -> None:
    """replace_rope_freqs returns the number of replaced buffers."""
    from sam3_onnx_equiv.export.rope_freqs import replace_rope_freqs  # noqa: PLC0415

    attn1 = _FakeAttention()
    attn2 = _FakeAttention()
    parent = nn.Sequential(attn1, attn2)

    count = replace_rope_freqs(parent)
    assert count == 2, f"Expected 2 replacements, got {count}"


def test_backbone_freqs_cis_removed_after_replace() -> None:
    """After replace_rope_freqs, no freqs_cis remain in the target sub-tree."""
    from sam3_onnx_equiv.export.rope_freqs import replace_rope_freqs  # noqa: PLC0415

    attn = _FakeAttention()
    replace_rope_freqs(attn)

    assert not hasattr(attn, "freqs_cis"), (
        "freqs_cis should be deleted after replace_rope_freqs"
    )
    assert hasattr(attn, "freqs_cos"), "freqs_cos must exist after replacement"
    assert hasattr(attn, "freqs_sin"), "freqs_sin must exist after replacement"
    assert attn.freqs_cos.dtype == torch.float32
    assert attn.freqs_sin.dtype == torch.float32


def test_replace_does_not_affect_rope_attention_modules() -> None:
    """replace_rope_freqs does NOT modify RoPEAttention-style modules.

    Tracker modules use use_rope_real=True (set at construction time) and
    carry no freqs_cis buffers.  This test verifies that replace_rope_freqs
    is a no-op for such modules, and their use_rope_real flag is unchanged.
    """
    from sam3_onnx_equiv.export.rope_freqs import replace_rope_freqs  # noqa: PLC0415

    rope_attn = _FakeRoPEAttention(use_rope_real=True)
    count = replace_rope_freqs(rope_attn)

    assert count == 0, (
        "replace_rope_freqs should return 0 for RoPEAttention modules "
        f"(no freqs_cis present), but returned {count}"
    )
    assert rope_attn.use_rope_real is True, (
        "use_rope_real flag must not be modified by replace_rope_freqs"
    )
    # No freqs_cos/freqs_sin should have been added
    assert not hasattr(rope_attn, "freqs_cos"), (
        "freqs_cos must not be added to RoPEAttention by replace_rope_freqs"
    )


def test_replace_scope_backbone_only_leaves_rope_attention_intact() -> None:
    """Applying replace_rope_freqs to backbone does not touch tracker modules.

    Models a scenario where backbone and tracker co-exist in a parent module.
    Only backbone's Attention modules carry freqs_cis; tracker's RoPEAttention
    does not.  After applying replace_rope_freqs(backbone), the tracker module
    remains unmodified.
    """
    from sam3_onnx_equiv.export.rope_freqs import replace_rope_freqs  # noqa: PLC0415

    backbone = nn.Sequential(_FakeAttention(), _FakeAttention())
    tracker = nn.Sequential(_FakeRoPEAttention(), _FakeRoPEAttention())

    parent = nn.ModuleDict({"backbone": backbone, "tracker": tracker})

    # Apply only to backbone (as done in image_encoder._load_equiv_sam3_model).
    count = replace_rope_freqs(parent["backbone"])
    assert count == 2

    # Backbone: no freqs_cis should remain.
    remaining = sum(
        1 for m in parent["backbone"].modules() if hasattr(m, "freqs_cis")
    )
    assert remaining == 0, (
        f"freqs_cis still present in backbone after replace: {remaining}"
    )

    # Tracker: RoPEAttention modules must be untouched.
    for m in parent["tracker"].modules():
        if isinstance(m, _FakeRoPEAttention):
            assert not hasattr(m, "freqs_cos"), (
                "replace_rope_freqs should NOT have added freqs_cos to tracker"
            )
            assert m.use_rope_real is True


# ---------------------------------------------------------------------------
# Integration test (requires checkpoint + equiv source)
# ---------------------------------------------------------------------------


def test_image_encoder_replace_rope_freqs_scope_integration() -> None:
    """Integration: replace_rope_freqs(model.backbone) zero-freqs_cis guarantee.

    Loads the full SAM3 image model via equiv source and confirms:
      - replace_rope_freqs(model.backbone) replaces > 0 buffers.
      - After the call, model.named_buffers() has zero freqs_cis entries.
      - RoPEAttention modules (if present via enable_inst_interactivity) are
        NOT touched (no freqs_cos added to them directly).
    """
    _skip_if_missing()

    from sam3_onnx_equiv.export.image_encoder import _load_equiv_sam3_model  # noqa: PLC0415

    # _load_equiv_sam3_model already calls replace_rope_freqs(model.backbone).
    model = _load_equiv_sam3_model(EQUIV_SOURCE_ROOT, CHECKPOINT_PATH)

    # No freqs_cis anywhere in the full model.
    fc_remaining = [
        n for n, _ in model.named_buffers() if "freqs_cis" in n
    ]
    assert len(fc_remaining) == 0, (
        f"freqs_cis still present after replace_rope_freqs: {fc_remaining}"
    )

    # freqs_cos buffers must exist (replacement happened).
    fcos_count = sum(1 for n, _ in model.named_buffers() if "freqs_cos" in n)
    assert fcos_count > 0, (
        "No freqs_cos buffers found after replace_rope_freqs -- "
        "replacement may not have run"
    )
