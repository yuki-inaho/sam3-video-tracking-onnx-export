"""Smoke test: verify that core dependencies can be imported in the uv environment.

This test is intentionally minimal — it imports the four critical packages
and asserts nothing more. The goal is a fast red→green signal during D1 setup.
"""

from __future__ import annotations


def test_imports() -> None:
    """All four packages must be importable without raising ImportError."""
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import sam3  # noqa: F401
    import torch  # noqa: F401
