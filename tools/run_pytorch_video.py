"""Stage C-1: PyTorch full video tracking oracle using Sam3TrackerPredictor.

Uses build_tracker(with_backbone=True) + Sam3TrackerPredictor directly.
Sam3TrackerPredictor.__init__ enters bfloat16 autocast for the entire model
lifetime (sam3_tracking_predictor.py:49-50), avoiding the OOM that occurs
when using build_sam3_video_model on GTX 1070 (8GB VRAM).

§9.9 OOM record:
  - build_sam3_video_model (float32) with bfloat16 autocast fails: decoder.py:281
    hardcodes device="cuda" regardless of model.device. Cannot run on CPU.
  - GPU forward pass OOM: float32 ViT-H attention requests 1.60 GiB with 1.08 GiB free.
  - Resolution: use Sam3TrackerPredictor directly (autocast enters at construction;
    tracker-only ViT is ~2x smaller than full video model).

Inputs
------
Synthetic moving red circle video frames (N=6 frames, 128x128 synthetic)
saved to a temp directory and loaded via video_path=<dir>.
Point prompt at circle center on frame 0.

Outputs
-------
outputs/reference/video_oracle_all.npz       — aggregated oracle (frame_indices,
                                                obj_ids_per_frame, masks_per_frame,
                                                probs_per_frame, boxes_per_frame)
outputs/reference/video_oracle_vis/          — mask overlay PNGs per frame
outputs/reference/constants/*.npy            — Python-side constants
outputs/reference/constants/manifest.json   — shape/dtype metadata
logs/video_oracle.log                        — execution log

Usage
-----
    uv run python tools/run_pytorch_video.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Repository layout constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
CHECKPOINT_PATH = REPO_ROOT / "models" / "sam3.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "reference"
ORACLE_NPZ = OUTPUT_DIR / "video_oracle_all.npz"
VIS_DIR = OUTPUT_DIR / "video_oracle_vis"
CONSTANTS_DIR = OUTPUT_DIR / "constants"
LOG_PATH = REPO_ROOT / "logs" / "video_oracle.log"

# Video parameters — explicitly declared per §9.9 OOM policy
# §9.9 OOM record: GPU inference uses Sam3TrackerPredictor with bfloat16 autocast
# which enters at __init__ (sam3_tracking_predictor.py:49-50).
N_FRAMES = 6            # number of synthetic frames (>=4 per task spec)
IMAGE_SIZE = 128        # synthetic frame resolution (saved as JPEG, SAM3 resamples to 1008)
CIRCLE_RADIUS = 16      # circle radius in synthetic frame
STEP = 12               # pixels right per frame
JPEG_QUALITY = 95       # JPEG quality for frame serialization

# Frame 0 prompt: point at the circle center on frame 0
_FRAME0_CX = IMAGE_SIZE // 6 + 0 * STEP   # = IMAGE_SIZE//6 = 21 px
_FRAME0_CY = IMAGE_SIZE // 2              # = 64 px

# Python-side constants to extract (§9.10 audit specification, tracker attributes)
_CONSTANT_SPECS: list[dict[str, str]] = [
    {"name": "maskmem_tpos_enc",       "attr": "maskmem_tpos_enc"},
    {"name": "no_mem_embed",           "attr": "no_mem_embed"},
    {"name": "no_mem_pos_enc",         "attr": "no_mem_pos_enc"},
    {"name": "no_obj_embed_spatial",   "attr": "no_obj_embed_spatial"},
]
_CONV_SPECS: list[dict[str, str]] = [
    {"name": "conv_s0_weight", "attr": "conv_s0", "param": "weight"},
    {"name": "conv_s0_bias",   "attr": "conv_s0", "param": "bias"},
    {"name": "conv_s1_weight", "attr": "conv_s1", "param": "weight"},
    {"name": "conv_s1_bias",   "attr": "conv_s1", "param": "bias"},
]

NO_OBJ_SCORE: float = -1024.0  # sam3_tracker_base.py:24


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("video_oracle")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Synthetic video frame generation
# ---------------------------------------------------------------------------

def _make_synthetic_frames(
    n_frames: int, size: int, radius: int, step: int
) -> list[Image.Image]:
    """Generate N frames of a red circle moving right on a black background."""
    frames: list[Image.Image] = []
    start_x = size // 6
    for i in range(n_frames):
        cx = start_x + i * step
        cy = size // 2
        img = Image.new("RGB", (size, size), (0, 0, 0))
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        ImageDraw.Draw(img).ellipse(bbox, fill=(255, 0, 0))
        frames.append(img)
    return frames


def _save_frames_to_dir(frames: list[Image.Image], tmp_dir: Path, quality: int) -> Path:
    """Save PIL frames as JPEG files suitable for load_video_frames(video_path=dir)."""
    video_dir = tmp_dir / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(video_dir / f"{i:06d}.jpg", quality=quality)
    return video_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_tracker(checkpoint_path: Path, logger: logging.Logger) -> Any:
    """Build Sam3TrackerPredictor with ViT backbone and load checkpoint.

    Sam3TrackerPredictor enters bfloat16 autocast in __init__ (line 49-50),
    so ViT forward passes run in bfloat16 → avoids OOM on GTX 1070 (8GB).

    Checkpoint key mapping:
      sam3.pt has 'tracker.*' keys.
      Sam3TrackerPredictor state_dict uses those names directly.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Copy models/sam3.pt from ~/Project/sam3/models/sam3.pt."
        )

    logger.info("Building Sam3TrackerPredictor with ViT backbone ...")
    t0 = time.time()
    from sam3.model_builder import build_tracker  # type: ignore[import]

    tracker = build_tracker(
        apply_temporal_disambiguation=False,  # simple tracking, no heuristics
        with_backbone=True,                   # include ViT backbone for image encoding
    )
    logger.info("Tracker structure built in %.1f s", time.time() - t0)

    # Load checkpoint: extract 'tracker.*' keys → strip 'tracker.' prefix.
    # Backbone weights are stored under 'detector.backbone.*' in sam3.pt (shared backbone
    # in the full video model); map these to 'backbone.*' for the standalone tracker.
    logger.info("Loading checkpoint from %s ...", checkpoint_path)
    t1 = time.time()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    # Tracker-specific weights (strips 'tracker.' prefix)
    tracker_ckpt: dict[str, torch.Tensor] = {
        k[len("tracker."):]: v
        for k, v in ckpt.items()
        if k.startswith("tracker.")
    }
    # Backbone weights: checkpoint stores them under 'detector.backbone.*';
    # standalone tracker expects them under 'backbone.*'.
    # (In the full video model the backbone is shared via the detector submodule.)
    backbone_ckpt: dict[str, torch.Tensor] = {
        k[len("detector."):]: v
        for k, v in ckpt.items()
        if k.startswith("detector.backbone.")
    }
    tracker_ckpt.update(backbone_ckpt)
    logger.info(
        "Mapped %d tracker keys + %d backbone keys from checkpoint",
        len(tracker_ckpt) - len(backbone_ckpt),
        len(backbone_ckpt),
    )

    missing, unexpected = tracker.load_state_dict(tracker_ckpt, strict=False)
    if missing:
        logger.warning("Missing keys in tracker ckpt: %d keys (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys in tracker ckpt: %d keys (first 5: %s)", len(unexpected), unexpected[:5])
    logger.info(
        "Checkpoint loaded in %.1f s (missing=%d, unexpected=%d)",
        time.time() - t1,
        len(missing),
        len(unexpected),
    )

    # Move to CUDA: convert weights to bfloat16 first to halve VRAM from ~5.9 GB → ~3.0 GB.
    # Sam3TrackerPredictor.__init__ enters bfloat16 autocast globally (line 49-50), so
    # compute is already bfloat16. Converting weights to bfloat16 here ensures inputs and
    # weights share the same dtype → no implicit up-cast overhead.
    # §9.9 OOM record: float32 weights (5.92 GiB) + ViT-H attention (412 MiB) → OOM on
    # GTX 1070 (8GB). Fix: .bfloat16() before .cuda() reduces weight footprint to ~3.0 GiB.
    if torch.cuda.is_available():
        logger.info("Converting tracker weights to bfloat16 to reduce VRAM footprint ...")
        t2 = time.time()
        tracker = tracker.bfloat16()
        logger.info("Moving tracker to CUDA ...")
        tracker = tracker.cuda()
        logger.info("Tracker on CUDA in %.1f s (GPU mem: %.1f GiB used)",
                    time.time() - t2,
                    torch.cuda.memory_allocated() / 1e9)
    else:
        logger.warning("CUDA not available — using CPU (slow).")

    tracker.eval()
    logger.info("Tracker ready (device=%s) in %.1f s total", tracker.device, time.time() - t0)
    return tracker


# ---------------------------------------------------------------------------
# Oracle tracking
# ---------------------------------------------------------------------------

def _run_oracle_tracking(
    tracker: Any,
    frames: list[Image.Image],
    logger: logging.Logger,
    tmp_dir: Path,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run memory-bank tracking via Sam3TrackerPredictor and collect per-frame outputs.

    Uses point prompt at circle center on frame 0.

    Returns
    -------
    frame_indices  : list of int
    obj_ids_list   : list of np.ndarray
    masks_list     : list of np.ndarray (low_res bool masks shape N_obj x H x W)
    probs_list     : list of np.ndarray (object score logits)
    boxes_list     : list of np.ndarray (placeholder; tracker API returns masks only)
    """
    # Save frames to temp dir for load_video_frames
    video_dir = _save_frames_to_dir(frames, tmp_dir, JPEG_QUALITY)
    logger.info("Saved %d frames to %s", len(frames), video_dir)

    video_h, video_w = IMAGE_SIZE, IMAGE_SIZE

    with torch.inference_mode():
        logger.info("Initializing tracker inference state ...")
        inference_state = tracker.init_state(
            video_path=str(video_dir),
            video_height=video_h,
            video_width=video_w,
            offload_video_to_cpu=True,    # save GPU memory; tracker streams frames per step
        )
        logger.info(
            "Inference state initialized: num_frames=%d video_size=%dx%d",
            inference_state["num_frames"],
            video_h,
            video_w,
        )

        # Add point prompt at circle center on frame 0
        # Coordinates: (x, y) normalized [0, 1] relative to image size
        cx_norm = _FRAME0_CX / IMAGE_SIZE
        cy_norm = _FRAME0_CY / IMAGE_SIZE
        logger.info(
            "Adding point prompt at frame 0: center=(%d, %d) normalized=(%.3f, %.3f)",
            _FRAME0_CX, _FRAME0_CY, cx_norm, cy_norm,
        )
        # Returns (frame_idx, obj_ids, low_res_masks, video_res_masks)
        _, out_obj_ids, _, out_video_masks = tracker.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=[[cx_norm, cy_norm]],   # (x, y) normalized
            labels=[1],                     # 1=foreground
        )
        logger.info(
            "add_new_points_or_box on frame 0: obj_ids=%s video_masks_shape=%s",
            out_obj_ids,
            out_video_masks.shape if hasattr(out_video_masks, "shape") else "N/A",
        )

        # Propagate forward
        logger.info("Propagating tracking forward through %d frames ...", inference_state["num_frames"])
        frame_indices: list[int] = []
        obj_ids_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        probs_list: list[np.ndarray] = []
        boxes_list: list[np.ndarray] = []

        t0 = time.time()
        for (
            frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores
        ) in tracker.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=inference_state["num_frames"],
            reverse=False,
            propagate_preflight=True,  # consolidates temp_output_dict_per_obj → cond_frame_outputs
        ):
            frame_indices.append(int(frame_idx))
            ids_arr = np.array(obj_ids)
            # low_res_masks: (N_obj, 1, H, W) → binary via >0
            lrm = low_res_masks.cpu().float()
            masks_bin = (lrm > 0).squeeze(1).numpy()   # (N_obj, H, W)
            scores_arr = obj_scores.cpu().float().numpy()  # (N_obj, 1) or (N_obj,)
            if scores_arr.ndim > 1:
                scores_arr = scores_arr.squeeze(1)

            obj_ids_list.append(ids_arr)
            masks_list.append(masks_bin)
            probs_list.append(scores_arr)
            boxes_list.append(np.zeros((len(ids_arr), 4), dtype=np.float32))  # placeholder

            logger.info(
                "frame %d: obj_ids=%s scores=%s mask_shape=%s mask_nonzero=%s",
                frame_idx,
                ids_arr.tolist(),
                [f"{s:.3f}" for s in scores_arr.tolist()],
                masks_bin.shape,
                [int(masks_bin[j].sum()) for j in range(masks_bin.shape[0])],
            )

        elapsed = time.time() - t0
        logger.info(
            "Propagation done in %.1f s; %d frames tracked",
            elapsed,
            len(frame_indices),
        )

    return frame_indices, obj_ids_list, masks_list, probs_list, boxes_list


# ---------------------------------------------------------------------------
# Oracle saving
# ---------------------------------------------------------------------------

def _save_oracle(
    frame_indices: list[int],
    obj_ids_list: list[np.ndarray],
    masks_list: list[np.ndarray],
    probs_list: list[np.ndarray],
    boxes_list: list[np.ndarray],
    logger: logging.Logger,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ORACLE_NPZ,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        obj_ids_per_frame=np.array(obj_ids_list, dtype=object),
        masks_per_frame=np.array(masks_list, dtype=object),
        probs_per_frame=np.array(probs_list, dtype=object),
        boxes_per_frame=np.array(boxes_list, dtype=object),
    )
    logger.info("Oracle saved → %s", ORACLE_NPZ)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_visualizations(
    frames: list[Image.Image],
    frame_indices: list[int],
    masks_list: list[np.ndarray],
    obj_ids_list: list[np.ndarray],
    logger: logging.Logger,
) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    colours = [(0, 200, 200, 120), (200, 100, 0, 120), (100, 200, 0, 120)]
    for i, frame_idx in enumerate(frame_indices):
        src = frame_idx if frame_idx < len(frames) else len(frames) - 1
        img = frames[src].convert("RGBA")
        masks = masks_list[i]   # (N_obj, H, W)
        obj_ids = obj_ids_list[i]
        for j, _ in enumerate(obj_ids):
            if j >= len(masks):
                break
            mask = masks[j]  # (H, W)
            ov_arr = np.zeros((*mask.shape, 4), dtype=np.uint8)
            ov_arr[mask] = colours[j % len(colours)]
            if ov_arr.shape[:2] != (img.size[1], img.size[0]):
                ov_img = Image.fromarray(ov_arr, "RGBA").resize(img.size, Image.NEAREST)
            else:
                ov_img = Image.fromarray(ov_arr, "RGBA")
            img = Image.alpha_composite(img, ov_img)
        out_path = VIS_DIR / f"frame_{frame_idx:04d}.png"
        img.convert("RGB").save(out_path)
        logger.info("Visualization → %s", out_path)


# ---------------------------------------------------------------------------
# Constants / weights extraction
# ---------------------------------------------------------------------------

def _extract_constants(tracker: Any, logger: logging.Logger) -> None:
    """Extract Python-side constants and projection weights.

    Sources (public sam3 file:line):
    - maskmem_tpos_enc     : sam3_tracker_base.py:104  shape=(num_maskmem,1,1,mem_dim)
    - no_mem_embed         : sam3_tracker_base.py:110  shape=(1,1,hidden_dim)
    - no_mem_pos_enc       : sam3_tracker_base.py:111  shape=(1,1,hidden_dim)
    - no_obj_embed_spatial : sam3_tracker_base.py:142  shape=(1,mem_dim)
    - conv_s0_{weight,bias}: sam3_tracker_base.py:450  sam_mask_decoder.conv_s0
    - conv_s1_{weight,bias}: sam3_tracker_base.py:453  sam_mask_decoder.conv_s1
    - NO_OBJ_SCORE         : sam3_tracker_base.py:24   Python constant = -1024.0
    - sigmoid_scale_for_mem_enc : sam3_tracker_base.py:116 (default=20.0)
    - sigmoid_bias_for_mem_enc  : sam3_tracker_base.py:117 (default=-10.0)
    """
    CONSTANTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    def _save(name: str, tensor: torch.Tensor, source: str) -> None:
        arr = tensor.detach().cpu().float().numpy()
        fname = f"{name}.npy"
        np.save(CONSTANTS_DIR / fname, arr)
        manifest.append({"name": name, "file": fname,
                         "shape": list(arr.shape), "dtype": str(arr.dtype),
                         "source_attr": source})
        logger.info("Extracted %-30s shape=%-20s dtype=%s", name, str(tuple(arr.shape)), arr.dtype)

    # Tracker-level nn.Parameters
    for spec in _CONSTANT_SPECS:
        param = getattr(tracker, spec["attr"])
        data = param.data if isinstance(param, torch.nn.Parameter) else param
        _save(spec["name"], data, f"tracker.{spec['attr']} (sam3_tracker_base.py)")

    # conv_s0 / conv_s1 on sam_mask_decoder
    decoder = tracker.sam_mask_decoder
    for spec in _CONV_SPECS:
        conv = getattr(decoder, spec["attr"])
        pt = getattr(conv, spec["param"])
        data = pt.data if isinstance(pt, torch.nn.Parameter) else pt
        _save(spec["name"], data, f"tracker.sam_mask_decoder.{spec['attr']}.{spec['param']}")

    # Scalar constants
    for name, val, source in [
        ("NO_OBJ_SCORE", NO_OBJ_SCORE, "sam3_tracker_base.py:24 (= -1024.0)"),
        ("sigmoid_scale_for_mem_enc", tracker.sigmoid_scale_for_mem_enc,
         "sam3_tracker_base.py:116 (default=20.0)"),
        ("sigmoid_bias_for_mem_enc", tracker.sigmoid_bias_for_mem_enc,
         "sam3_tracker_base.py:117 (default=-10.0)"),
    ]:
        arr = np.array(val, dtype=np.float32)
        fname = f"{name}.npy"
        np.save(CONSTANTS_DIR / fname, arr)
        manifest.append({"name": name, "file": fname,
                         "shape": [], "dtype": "float32", "source_attr": source})
        logger.info("Extracted scalar %-30s = %s", name, val)

    manifest_path = CONSTANTS_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Constants manifest → %s (%d entries)", manifest_path, len(manifest))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = _setup_logging()
    logger.info("=== Stage C-1: SAM3 Video Tracking Oracle (Sam3TrackerPredictor) ===")
    logger.info("N_FRAMES=%d IMAGE_SIZE=%d CIRCLE_RADIUS=%d STEP=%d",
                N_FRAMES, IMAGE_SIZE, CIRCLE_RADIUS, STEP)
    logger.info("GPU available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s (%.1f GiB total)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Step 1: load tracker
    tracker = _load_tracker(CHECKPOINT_PATH, logger)

    # Step 2: generate synthetic frames
    logger.info("Generating %d synthetic frames (%dx%d) ...", N_FRAMES, IMAGE_SIZE, IMAGE_SIZE)
    frames = _make_synthetic_frames(N_FRAMES, IMAGE_SIZE, CIRCLE_RADIUS, STEP)

    # Step 3: extract constants (model in clean state before inference)
    logger.info("Extracting Python-side constants ...")
    _extract_constants(tracker, logger)

    # Step 4: run oracle tracking
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        frame_indices, obj_ids_list, masks_list, probs_list, boxes_list = _run_oracle_tracking(
            tracker, frames, logger, tmp_dir
        )

    if len(frame_indices) == 0:
        logger.error(
            "No frames yielded from propagate_in_video. "
            "HARD ERROR — no implicit fallback (§9.9)."
        )
        sys.exit(1)

    # Step 5: save oracle
    _save_oracle(frame_indices, obj_ids_list, masks_list, probs_list, boxes_list, logger)

    # Step 6: visualize
    _save_visualizations(frames, frame_indices, masks_list, obj_ids_list, logger)

    # Step 7: summary
    logger.info("=== Oracle Summary ===")
    logger.info("Frames tracked: %d", len(frame_indices))
    all_ids: set[int] = set()
    for ids in obj_ids_list:
        all_ids.update(ids.tolist())
    logger.info("Unique object IDs: %s", sorted(all_ids))
    for i, fidx in enumerate(frame_indices):
        ids = obj_ids_list[i]
        masks = masks_list[i]
        probs = probs_list[i]
        px_counts = [int(masks[j].sum()) if j < len(masks) else 0 for j in range(len(ids))]
        logger.info(
            "  frame %d: obj_ids=%s scores=%s mask_px=%s",
            fidx,
            ids.tolist(),
            [f"{p:.3f}" for p in probs.tolist()],
            px_counts,
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
