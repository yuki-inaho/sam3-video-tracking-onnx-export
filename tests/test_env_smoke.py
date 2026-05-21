"""Smoke test: verify that core dependencies can be imported in the uv environment.

This test is intentionally minimal — it imports the four critical packages
and asserts nothing more. The goal is a fast red→green signal during D1 setup.
"""

from __future__ import annotations

from sam3_onnx_equiv.path_config import checkpoint_path, onnx_dir, repo_root, sam3_source_root


def test_imports() -> None:
    """All four packages must be importable without raising ImportError.

    ``sam3`` is installed as an editable package (submodule ``sam3/`` at the
    repo root), so no manual sys.path manipulation is needed or desired.
    Adding the submodule root to sys.path would shadow the editable install
    and cause deep train-only imports (e.g. ``decord``) to be triggered.
    """
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    # Verify that sam3_source_root() resolves to the root submodule.
    sam3_src = sam3_source_root()
    assert sam3_src.name == "sam3", f"Expected root submodule path, got: {sam3_src}"

    import torch  # noqa: F401

    import sam3  # noqa: F401


def test_relative_env_paths_resolve_from_repo_root(monkeypatch) -> None:
    """Relative env paths from example.env are repository-root relative."""
    root = repo_root()
    monkeypatch.setenv("SAM3_ONNX_REPO", str(root))
    monkeypatch.setenv("SAM3_SRC", "sam3")
    monkeypatch.setenv("SAM3_ONNX_DIR", "outputs/onnx")
    monkeypatch.setenv("SAM3_CHECKPOINT", "models/sam3.pt")
    monkeypatch.chdir(root / "notebooks")

    assert sam3_source_root() == root / "sam3"
    assert onnx_dir() == root / "outputs" / "onnx"
    assert checkpoint_path() == root / "models" / "sam3.pt"
