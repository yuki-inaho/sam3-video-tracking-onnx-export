#!/usr/bin/env python3
"""CLI: export the SAM3 image encoder to ONNX (D5-2).

Usage:
    uv run python tools/export_image_encoder.py
    uv run python tools/export_image_encoder.py \\
        --equiv-source outputs/sam3_equiv_source \\
        --checkpoint models/sam3.pt \\
        --output outputs/onnx/image_encoder.onnx

After export, lists the unique op_types in the graph so complex ops can be
confirmed absent.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SANDBOX_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export SAM3 image encoder to ONNX")
    p.add_argument(
        "--equiv-source",
        type=Path,
        default=SANDBOX_ROOT / "outputs" / "sam3_equiv_source",
        help="Path to equiv source root (outputs/sam3_equiv_source)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=SANDBOX_ROOT / "models" / "sam3.pt",
        help="Path to models/sam3.pt",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=SANDBOX_ROOT / "outputs" / "onnx" / "image_encoder.onnx",
        help="Destination ONNX file path",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    from sam3_onnx_equiv.export.image_encoder import export_image_encoder  # noqa: PLC0415

    export_image_encoder(
        equiv_source_root=args.equiv_source,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )

    # Report op_type summary
    model_proto = onnx.load(str(args.output))
    op_types = sorted({node.op_type for node in model_proto.graph.node})
    print(f"[OK] ONNX file: {args.output}")
    print(f"[OK] Op types ({len(op_types)}): {op_types}")

    forbidden = {"ComplexFloat", "Complex", "Polar", "ViewAsComplex", "ViewAsReal"}
    found = {op for op in op_types if op in forbidden}
    if found:
        print(f"[FAIL] Complex ops in graph: {found}")
        return 1
    print("[OK] No complex ops in graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
