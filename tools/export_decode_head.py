"""CLI tool: export SAM3 decode head to ONNX (B-3).

Exports prompt_encoder + mask_decoder + obj_ptr_proj with fixed shapes
and multimask_output=False baked in.

Usage:
  uv run python tools/export_decode_head.py
  uv run python tools/export_decode_head.py --output /path/to/decode_head.onnx

Fixed-shape I/O:
  Inputs:
    image_embeddings  : (1, 256, 72, 72)
    high_res_feat0    : (1, 32, 288, 288)   conv_s0-projected FPN level 0
    high_res_feat1    : (1, 64, 144, 144)   conv_s1-projected FPN level 1
    point_coords      : (1, 1, 2)
    point_labels      : (1, 1)
    mask_input        : (1, 1, 288, 288)
    has_mask_input    : (1,)

  Outputs:
    low_res_masks         : (1, 1, 288, 288)
    iou_scores            : (1, 1)
    object_score_logits   : (1, 1)
    obj_ptr               : (1, 256)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = SANDBOX_ROOT / "outputs" / "onnx" / "decode_head.onnx"
EQUIV_SOURCE_ROOT = SANDBOX_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = SANDBOX_ROOT / "models" / "sam3.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination ONNX path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--equiv-source",
        type=Path,
        default=EQUIV_SOURCE_ROOT,
        help=f"Equiv-source root (default: {EQUIV_SOURCE_ROOT})",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"SAM3 checkpoint path (default: {CHECKPOINT_PATH})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from sam3_onnx_equiv.export.decode_head import export_decode_head  # noqa: PLC0415

    log.info("=== SAM3 decode head ONNX export (B-3) ===")
    log.info("equiv_source_root : %s", args.equiv_source)
    log.info("checkpoint_path   : %s", args.checkpoint)
    log.info("output_path       : %s", args.output)

    export_decode_head(
        equiv_source_root=args.equiv_source,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    log.info("Done. ONNX written to %s", args.output)


if __name__ == "__main__":
    main()
