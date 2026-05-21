"""TDD tests for Stage C-1: PyTorch video tracking oracle.

Verifies that:
- outputs/reference/video_oracle_*.npz files exist and contain valid per-frame data
- The oracle tracks a consistent object id across multiple frames
- Per-frame masks are non-empty
- outputs/reference/constants/ manifest.json exists with required constant names

Run only this file (do NOT run the full suite during C-1):
    uv run pytest tests/test_video_oracle.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
ORACLE_DIR = REPO_ROOT / "outputs" / "reference"
CONSTANTS_DIR = ORACLE_DIR / "constants"
VIS_DIR = ORACLE_DIR / "video_oracle_vis"

# Required oracle files (at least one npz must exist)
ORACLE_GLOB = "video_oracle_*.npz"

# Required constant names per §9.10
REQUIRED_CONSTANTS = [
    "maskmem_tpos_enc",
    "no_mem_embed",
    "no_mem_pos_enc",
    "no_obj_embed_spatial",
    "conv_s0_weight",
    "conv_s0_bias",
    "conv_s1_weight",
    "conv_s1_bias",
]


class TestOracleTracks:
    """Verify that the oracle npz files represent coherent multi-frame tracking."""

    def test_oracle_npz_files_exist(self) -> None:
        """At least one oracle npz file must have been produced."""
        files = sorted(ORACLE_DIR.glob(ORACLE_GLOB))
        assert len(files) > 0, (
            f"No oracle files matching '{ORACLE_GLOB}' found under {ORACLE_DIR}. "
            "Run tools/run_pytorch_video.py first."
        )

    def test_oracle_tracks_across_frames(self) -> None:
        """Main oracle test: frames > 1, same object id across frames, mask non-empty.

        Loads the aggregated oracle npz and checks per-frame consistency.
        The oracle script saves one npz per frame named video_oracle_frame_{idx:04d}.npz
        or a single combined video_oracle_all.npz.
        """
        combined = ORACLE_DIR / "video_oracle_all.npz"
        if combined.exists():
            data = np.load(combined, allow_pickle=True)
            frame_indices = data["frame_indices"]
            obj_ids_per_frame = data["obj_ids_per_frame"]  # shape: (F, N_obj)
            masks_per_frame = data["masks_per_frame"]  # shape: (F, N_obj, H, W)
        else:
            # Fall back to per-frame files
            files = sorted(ORACLE_DIR.glob(ORACLE_GLOB))
            assert len(files) > 0, (
                "No oracle npz files found. Run tools/run_pytorch_video.py first."
            )
            frame_indices = []
            obj_ids_list = []
            masks_list = []
            for f in files:
                d = np.load(f, allow_pickle=True)
                frame_indices.append(int(d["frame_idx"]))
                obj_ids_list.append(d["obj_ids"])
                masks_list.append(d["masks"])
            frame_indices = np.array(frame_indices)
            obj_ids_per_frame = np.array(obj_ids_list, dtype=object)
            masks_per_frame = np.array(masks_list, dtype=object)

        # 1) More than one frame tracked
        n_frames = len(frame_indices)
        assert n_frames > 1, (
            f"Oracle only has {n_frames} frame(s); expected >1 for tracking validation."
        )

        # 2) At least one object id is consistent across frames
        # Collect all obj_ids that appear in every frame
        if hasattr(obj_ids_per_frame, "dtype") and obj_ids_per_frame.dtype == object:
            all_sets = [set(ids.tolist()) for ids in obj_ids_per_frame]
        else:
            all_sets = [set(row.tolist()) for row in obj_ids_per_frame]
        common_ids = all_sets[0].intersection(*all_sets[1:])
        assert len(common_ids) > 0, (
            f"No object id is consistent across all {n_frames} frames. "
            f"Per-frame ids: {[sorted(s) for s in all_sets]}"
        )

        # 3) For each consistent id, mask is non-empty in every frame
        for frame_i, masks in enumerate(masks_per_frame):
            # masks shape: (N_obj, H, W) or variable
            if hasattr(masks, "shape") and masks.ndim == 3:
                assert masks.any(), (
                    f"Frame {frame_indices[frame_i]}: all masks are empty (shape={masks.shape})"
                )
            # object-array case — at least one element is non-zero
            elif hasattr(masks, "dtype") and masks.dtype == object:
                assert any(m.any() for m in masks), (
                    f"Frame {frame_indices[frame_i]}: all masks are empty."
                )


class TestConstantsExtracted:
    """Verify that Python-side constants were extracted and saved."""

    def test_constants_dir_exists(self) -> None:
        assert CONSTANTS_DIR.exists(), (
            f"{CONSTANTS_DIR} does not exist. Run tools/run_pytorch_video.py first."
        )

    def test_manifest_exists(self) -> None:
        manifest = CONSTANTS_DIR / "manifest.json"
        assert manifest.exists(), (
            f"manifest.json not found in {CONSTANTS_DIR}. Run tools/run_pytorch_video.py first."
        )

    def test_required_constants_in_manifest(self) -> None:
        manifest = CONSTANTS_DIR / "manifest.json"
        assert manifest.exists(), "manifest.json missing"
        with manifest.open() as f:
            info = json.load(f)
        recorded = {entry["name"] for entry in info}
        missing = set(REQUIRED_CONSTANTS) - recorded
        assert not missing, f"Missing constants in manifest: {sorted(missing)}"

    def test_constant_npy_files_loadable(self) -> None:
        manifest = CONSTANTS_DIR / "manifest.json"
        if not manifest.exists():
            pytest.skip("manifest.json not present yet")
        with manifest.open() as f:
            info = json.load(f)
        for entry in info:
            npy_path = CONSTANTS_DIR / entry["file"]
            assert npy_path.exists(), f"npy file missing: {npy_path}"
            arr = np.load(npy_path)
            assert arr is not None
            # Shape and dtype must match manifest
            assert list(arr.shape) == entry["shape"], (
                f"{entry['name']}: shape mismatch {arr.shape} vs {entry['shape']}"
            )
            assert str(arr.dtype) == entry["dtype"], (
                f"{entry['name']}: dtype mismatch {arr.dtype} vs {entry['dtype']}"
            )


class TestVisualization:
    """Verify visualization PNGs were produced."""

    def test_vis_dir_exists(self) -> None:
        assert VIS_DIR.exists(), (
            f"Visualization directory {VIS_DIR} does not exist. "
            "Run tools/run_pytorch_video.py first."
        )

    def test_vis_pngs_exist(self) -> None:
        pngs = sorted(VIS_DIR.glob("*.png"))
        assert len(pngs) > 0, (
            f"No PNG files found in {VIS_DIR}. Run tools/run_pytorch_video.py first."
        )
