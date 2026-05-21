"""CLI tool: export SAM3 memory_encoder (SimpleMaskEncoder) to ONNX (B-2).

Usage:
  uv run python tools/export_memory_encoder.py
  uv run python tools/export_memory_encoder.py --output /path/to/output.onnx

The script exports the memory_encoder using fixed input shapes:
  pix_feat     : (1, 256, 72, 72)
  mask_for_mem : (1,   1, 1008, 1008)

Outputs:
  maskmem_features : (1, 64, 72, 72)
  maskmem_pos_enc  : (1, 64, 72, 72)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = SANDBOX_ROOT / "outputs" / "onnx" / "memory_encoder.onnx"
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

    from sam3_onnx_equiv.export.memory_encoder import export_memory_encoder  # noqa: PLC0415

    log.info("=== SAM3 memory_encoder ONNX export (B-2) ===")
    log.info("equiv_source_root : %s", args.equiv_source)
    log.info("checkpoint_path   : %s", args.checkpoint)
    log.info("output_path       : %s", args.output)

    export_memory_encoder(
        equiv_source_root=args.equiv_source,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    log.info("Done. ONNX written to %s", args.output)


if __name__ == "__main__":
    main()
