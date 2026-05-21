"""Repository path configuration resolved from environment variables.

Defaults are repository-relative conventions only.  Machine-specific absolute
paths belong in a local ``.env`` file loaded by direnv.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve(raw_path: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def repo_root() -> Path:
    """Return the repository root, overridable with SAM3_ONNX_REPO."""
    raw = os.environ.get("SAM3_ONNX_REPO")
    if raw:
        return _resolve(raw)
    return Path(__file__).resolve().parents[2]


def _repo_relative_env(env_names: tuple[str, ...], default: str) -> Path:
    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw:
            return _resolve(raw, base=repo_root())
    return (repo_root() / default).resolve()


def sam3_source_root() -> Path:
    """Return the external official SAM3 source root.

    ``SAM3_SRC`` is the primary configuration point.  The default resolves to
    the ``sam3/`` Git submodule at the repository root and contains no
    user-specific path.
    """
    raw = os.environ.get("SAM3_SRC")
    if raw:
        return _resolve(raw, base=repo_root())
    return (repo_root() / "sam3").resolve()


def equiv_source_root() -> Path:
    """Return the generated equivalent SAM3 source root."""
    return _repo_relative_env(
        ("SAM3_EQUIV_SOURCE", "EQUIV_SOURCE"),
        "outputs/sam3_equiv_source",
    )


def checkpoint_path() -> Path:
    """Return the SAM3 checkpoint path."""
    return _repo_relative_env(("SAM3_CHECKPOINT", "CHECKPOINT"), "models/sam3.pt")


def onnx_dir() -> Path:
    """Return the ONNX artifact directory."""
    return _repo_relative_env(("SAM3_ONNX_DIR", "ONNX_DIR"), "outputs/onnx")


def constants_dir() -> Path:
    """Return the exported Python-side constants directory."""
    return _repo_relative_env(
        ("SAM3_CONSTANTS_DIR", "CONSTANTS_DIR"),
        "outputs/reference/constants",
    )


def reference_dir() -> Path:
    """Return the reference output directory."""
    return _repo_relative_env(("SAM3_REFERENCE_DIR", "REFERENCE_DIR"), "outputs/reference")
