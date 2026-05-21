"""Stage C-2 CLI: run the ONNX video orchestrator and compare with the C-1 oracle.

Runs the full memory-bank tracking loop (image_encoder + memory_attention_dynamic +
decode_head + memory_encoder + Python memory bank) on the same synthetic 6-frame
clip used by tools/run_pytorch_video.py, then prints per-frame mask IoU / score /
mask-pixel comparisons against outputs/reference/video_oracle_all.npz.

Usage:
    uv run python tools/run_onnx_video.py
    uv run python tools/run_onnx_video.py --max-frames 2   # frame0+frame1 only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

ONNX_DIR = REPO_ROOT / "outputs" / "onnx"
CONSTANTS_DIR = REPO_ROOT / "outputs" / "reference" / "constants"
ORACLE_NPZ = REPO_ROOT / "outputs" / "reference" / "video_oracle_all.npz"
LOG_PATH = REPO_ROOT / "logs" / "onnx_video.log"

IMAGE_SIZE = 128


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("onnx_video")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (debug). Default: all 6.")
    parser.add_argument("--emulate-bf16", action="store_true",
                        help="Diagnostic: round stored memory to bf16 (emulate oracle).")
    args = parser.parse_args()

    logger = _setup_logging()
    from sam3_onnx_equiv.video_orchestrator import VideoOrchestrator, make_oracle_frames

    orch = VideoOrchestrator(
        onnx_dir=ONNX_DIR,
        constants_dir=CONSTANTS_DIR,
        providers=["CPUExecutionProvider"],
    )

    frames = make_oracle_frames()
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    cx = (IMAGE_SIZE // 6) / IMAGE_SIZE
    cy = (IMAGE_SIZE // 2) / IMAGE_SIZE
    point_coords_norm = np.array([[[cx, cy]]], dtype=np.float32)
    point_labels = np.array([[1]], dtype=np.int32)

    result = orch.run_clip(
        frames_pil=frames,
        frame0_point_coords_norm=point_coords_norm,
        frame0_point_labels=point_labels,
        use_memory=True,
        emulate_oracle_bf16=args.emulate_bf16,
    )

    logger.info("memory_attention invocations: %d", result["memory_attention_invoke_count"])

    if not ORACLE_NPZ.exists():
        logger.warning("Oracle not found at %s — skipping comparison.", ORACLE_NPZ)
        return

    data = np.load(str(ORACLE_NPZ), allow_pickle=True)
    oracle_masks = data["masks_per_frame"]
    oracle_scores = data["probs_per_frame"]

    logger.info("=== ONNX orchestrator vs oracle ===")
    for i in range(len(result["masks"])):
        orch_mask = result["masks"][i].astype(bool)
        oracle_mask = oracle_masks[i][0].astype(bool)
        if oracle_mask.shape != orch_mask.shape:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(oracle_mask.astype(np.uint8) * 255, mode="L")
            pil = pil.resize((orch_mask.shape[1], orch_mask.shape[0]), PILImage.NEAREST)
            oracle_mask = np.array(pil) > 127
        iou = _mask_iou(orch_mask, oracle_mask)
        os = float(oracle_scores[i][0])
        cs = float(result["scores"][i])
        rel = abs(cs - os) / max(abs(os), 1e-8)
        logger.info(
            "  frame %d: IoU=%.4f | score onnx=%.4f oracle=%.4f rel=%.4f | px onnx=%d oracle=%d",
            i, iou, cs, os, rel, int(orch_mask.sum()), int(oracle_mask.sum()),
        )


if __name__ == "__main__":
    main()
