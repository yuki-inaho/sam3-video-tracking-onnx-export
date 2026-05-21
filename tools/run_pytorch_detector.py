"""PyTorch baseline (oracle) detector for SAM3.

Loads models/sam3.pt via build_sam3_image_model, runs inference on a
synthetic black-background red-circle image with a text prompt, and saves
masks/boxes/scores to outputs/reference/baseline_detector.npz together with
a visualization PNG.

Usage
-----
    uv run python tools/run_pytorch_detector.py

Output artefacts (relative to repository root)
-----------------------------------------------
    outputs/reference/baseline_detector.npz  — masks, boxes, scores arrays
    outputs/reference/baseline_detector_vis.png  — PIL visualization overlay
    logs/baseline_pytorch.log  — run log
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Repository layout constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
CHECKPOINT_PATH = REPO_ROOT / "models" / "sam3.pt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "reference"
LOG_PATH = REPO_ROOT / "logs" / "baseline_pytorch.log"

# Inference constants matching F-11 I/O contract
IMAGE_RESOLUTION = 1008
CONFIDENCE_THRESHOLD = 0.1
TEXT_PROMPT = "red circle"


def _setup_logging() -> logging.Logger:
    """Configure logging to both stderr and the log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("baseline_pytorch")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    # File handler
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _make_synthetic_image(size: int = IMAGE_RESOLUTION) -> Image.Image:
    """Return a black-background image with a centred red circle.

    The circle radius is ~10 % of the image to ensure visible foreground area.
    """
    radius = size // 10
    center = (size // 2, size // 2)
    image = Image.new("RGB", (size, size), (0, 0, 0))
    bbox = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    ImageDraw.Draw(image).ellipse(bbox, fill=(255, 0, 0))
    return image


def _detect_device(logger: logging.Logger) -> str:
    """Select inference device; NEVER silently fall back — always log the choice."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA available — using GPU: %s", gpu_name)
    else:
        device = "cpu"
        logger.warning(
            "CUDA NOT available — running on CPU. "
            "This is slower but produces equivalent results. "
            "If GPU was expected, check CUDA installation."
        )
    return device


def _save_visualization(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
) -> None:
    """Overlay the top-scored mask and bounding box on the source image."""
    vis = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if masks.shape[0] > 0:
        # Visualize the mask with the highest score.
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]  # shape: (1, H, W) or (H, W) depending on squeeze
        if mask.ndim == 3:
            mask = mask[0]
        # Resize mask to original image resolution if needed.
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        if mask_img.size != vis.size:
            mask_img = mask_img.resize(vis.size, Image.NEAREST)
        mask_arr = np.asarray(mask_img) > 127

        # Teal overlay for the foreground mask region.
        overlay_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
        overlay_arr[mask_arr] = [0, 200, 200, 120]
        overlay = Image.fromarray(overlay_arr, "RGBA")

        # Draw best bounding box.
        box = boxes[best_idx]
        draw = ImageDraw.Draw(vis)
        draw.rectangle(box.tolist(), outline=(0, 255, 255), width=3)

    vis = Image.alpha_composite(vis, overlay)
    vis.convert("RGB").save(out_path)


def run_detector() -> dict:
    """Run PyTorch detector baseline and return result dict."""
    logger = _setup_logging()
    logger.info("=== SAM3 PyTorch Detector Baseline ===")
    logger.info("Checkpoint: %s", CHECKPOINT_PATH)
    logger.info("Output dir: %s", OUTPUT_DIR)

    if not CHECKPOINT_PATH.exists():
        msg = (
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Copy models/sam3.pt from ~/Project/sam3/models/sam3.pt first."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    device = _detect_device(logger)

    # --- Load model (offline, no HuggingFace download) ---
    logger.info("Loading model from checkpoint (load_from_HF=False)...")
    t0 = time.time()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
        eval_mode=True,
        load_from_HF=False,
    )
    logger.info("Model loaded in %.1f s", time.time() - t0)

    processor = Sam3Processor(
        model,
        resolution=IMAGE_RESOLUTION,
        device=device,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # --- Build synthetic input ---
    logger.info("Generating synthetic image (black background, red circle, %dx%d)", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    image = _make_synthetic_image(IMAGE_RESOLUTION)

    # --- Inference ---
    logger.info("Running set_image...")
    t1 = time.time()
    state = processor.set_image(image)
    logger.info("set_image done in %.2f s", time.time() - t1)

    logger.info("Running set_text_prompt: '%s'...", TEXT_PROMPT)
    t2 = time.time()
    output = processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)
    logger.info("set_text_prompt done in %.2f s", time.time() - t2)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    logger.info("Detected %d object(s) above threshold=%.2f", len(scores), CONFIDENCE_THRESHOLD)
    logger.info("Scores: %s", scores.cpu().numpy().tolist()[:10])
    logger.info("Masks shape: %s", tuple(masks.shape))
    logger.info("Boxes shape: %s", tuple(boxes.shape))

    if len(scores) == 0:
        logger.warning(
            "No objects detected! The model returned zero detections above "
            "confidence_threshold=%.2f. Consider lowering the threshold.",
            CONFIDENCE_THRESHOLD,
        )

    # Convert to numpy for saving
    masks_np = masks.cpu().numpy().astype(np.float32)
    boxes_np = boxes.cpu().numpy().astype(np.float32)
    scores_np = scores.cpu().numpy().astype(np.float32)

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "baseline_detector.npz"
    np.savez(npz_path, masks=masks_np, boxes=boxes_np, scores=scores_np)
    logger.info("Saved NPZ: %s", npz_path)

    vis_path = OUTPUT_DIR / "baseline_detector_vis.png"
    _save_visualization(image, masks_np, boxes_np, scores_np, vis_path)
    logger.info("Saved visualization: %s", vis_path)

    logger.info("=== Baseline run complete ===")
    return {"masks": masks_np, "boxes": boxes_np, "scores": scores_np}


if __name__ == "__main__":
    run_detector()
