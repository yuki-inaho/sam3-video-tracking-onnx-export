"""Tests for sam3_source_patcher: verify exact-once text replacement.

These tests run against the official SAM3 source configured by SAM3_SRC,
so they confirm that MODEL_BUILDER_REPLACEMENTS are consistent with the
current version of model_builder.py (fail fast if the upstream code changes).
"""

from __future__ import annotations

import pytest

from sam3_onnx_equiv.path_config import sam3_source_root
from sam3_onnx_equiv.sam3_source_patcher import (
    MODEL_BUILDER_REPLACEMENTS,
    TextReplacement,
    _replace_once,
    patch_model_builder_text,
)

SAM3_SOURCE_ROOT = sam3_source_root()
MODEL_BUILDER_PATH = SAM3_SOURCE_ROOT / "sam3" / "model_builder.py"


@pytest.fixture(scope="module")
def model_builder_text() -> str:
    if not MODEL_BUILDER_PATH.exists():
        pytest.skip(f"Official SAM3 model_builder.py not found: {MODEL_BUILDER_PATH}")
    return MODEL_BUILDER_PATH.read_text(encoding="utf-8")


class TestReplaceOnce:
    """Unit tests for _replace_once helper."""

    def test_replaces_exactly_once(self) -> None:
        result = _replace_once("aXb", TextReplacement(old="X", new="Y", label="t"))
        assert result == "aYb"

    def test_raises_on_zero_occurrences(self) -> None:
        with pytest.raises(RuntimeError, match="Expected exactly one occurrence"):
            _replace_once("abc", TextReplacement(old="Z", new="W", label="missing"))

    def test_raises_on_two_occurrences(self) -> None:
        with pytest.raises(RuntimeError, match="Expected exactly one occurrence"):
            _replace_once("aXaX", TextReplacement(old="X", new="Y", label="dup"))


class TestModelBuilderReplacements:
    """Verify each entry in MODEL_BUILDER_REPLACEMENTS occurs exactly once in model_builder.py."""

    @pytest.mark.parametrize("replacement", MODEL_BUILDER_REPLACEMENTS, ids=lambda r: r.label)
    def test_each_replacement_occurs_exactly_once(
        self, model_builder_text: str, replacement: TextReplacement
    ) -> None:
        count = model_builder_text.count(replacement.old)
        assert count == 1, (
            f"Replacement '{replacement.label}' found {count} times "
            f"(expected 1) in {MODEL_BUILDER_PATH}"
        )


class TestPatchModelBuilderText:
    """Integration test: apply all patches to model_builder.py text."""

    def test_patched_text_contains_use_rope_real_true(self, model_builder_text: str) -> None:
        patched = patch_model_builder_text(model_builder_text)
        # build_sam3_image_model should accept use_rope_real
        assert "use_rope_real: bool = True" in patched

    def test_patched_text_no_longer_contains_original_false_defaults(
        self, model_builder_text: str
    ) -> None:
        patched = patch_model_builder_text(model_builder_text)
        # None of the original use_rope_real=False in RoPEAttention construction remain
        # (they are replaced with use_rope_real=use_rope_real)
        assert "use_rope_real=False" not in patched

    def test_patch_is_idempotent_in_structure(self, model_builder_text: str) -> None:
        """Patched text should still be valid Python (no syntax-breaking substitution)."""
        import ast

        patched = patch_model_builder_text(model_builder_text)
        # Should parse without SyntaxError
        ast.parse(patched)
