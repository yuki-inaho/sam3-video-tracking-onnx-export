"""Export the SAM3 *tracker* image encoder (SAM2 neck) to ONNX.

The detector image_encoder.onnx (build_sam3_image_model, add_sam2_neck=False)
returns the sam3 FPN, but the video tracker consumes the SAM2 neck features
(sam2_backbone_out).  This tool exports image_encoder_tracker.onnx with the
correct FPN for the video orchestrator (Stage C-2).

Usage:
    uv run python tools/export_image_encoder_tracker.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

EQUIV_SOURCE_ROOT = REPO_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = REPO_ROOT / "models" / "sam3.pt"
OUTPUT_PATH = REPO_ROOT / "outputs" / "onnx" / "image_encoder_tracker.onnx"


def main() -> None:
    from sam3_onnx_equiv.export.image_encoder import export_tracker_image_encoder

    log.info("=== SAM3 tracker image encoder ONNX export ===")
    log.info("equiv_source_root : %s", EQUIV_SOURCE_ROOT)
    log.info("checkpoint_path   : %s", CHECKPOINT_PATH)
    log.info("output_path       : %s", OUTPUT_PATH)
    export_tracker_image_encoder(EQUIV_SOURCE_ROOT, CHECKPOINT_PATH, OUTPUT_PATH)
    log.info("Done.")


if __name__ == "__main__":
    main()
