"""CLI tool: export SAM3 memory_attention (TransformerEncoderCrossAttention) to ONNX.

Usage:
    uv run python tools/export_memory_attention.py [--mem-len N] [--num-k-exclude-rope K]

Exports outputs/onnx/memory_attention.onnx with the fixed-shape contract defined
in src/sam3_onnx_equiv/export/memory_attention.py.

Default mem_len=10368 (2-frame test config, fast).  Use --mem-len 36352 for the
full 7-frame production config (slow on CPU ORT).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from sam3_onnx_equiv.export.memory_attention import export_memory_attention  # noqa: E402

_TWO_FRAME = 2 * 72 * 72       # 10368
_FULL = 7 * 72 * 72 + 16 * 4  # 36352

EQUIV_SOURCE_ROOT = SANDBOX_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = SANDBOX_ROOT / "models" / "sam3.pt"
ONNX_PATH = SANDBOX_ROOT / "outputs" / "onnx" / "memory_attention.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export memory_attention to ONNX")
    parser.add_argument(
        "--mem-len",
        type=int,
        default=_TWO_FRAME,
        help=f"Memory token length (default: {_TWO_FRAME} = 2-frame config). "
             f"Full config: {_FULL}.",
    )
    parser.add_argument(
        "--num-k-exclude-rope",
        type=int,
        default=0,
        help="Number of obj_ptr tokens excluded from cross-attn RoPE (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ONNX_PATH),
        help="Output ONNX path (default: outputs/onnx/memory_attention.onnx).",
    )
    args = parser.parse_args()

    export_memory_attention(
        equiv_source_root=EQUIV_SOURCE_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=Path(args.output),
        mem_len=args.mem_len,
        num_k_exclude_rope=args.num_k_exclude_rope,
    )

    import onnx  # noqa: PLC0415
    model = onnx.load(args.output)
    op_types = sorted({node.op_type for node in model.graph.node})
    print(f"\nOp types ({len(op_types)}): {op_types}")
    forbidden = {"ComplexFloat", "Complex", "Polar", "ViewAsComplex", "ViewAsReal"}
    found = [op for op in op_types if op in forbidden or op.lower() in {f.lower() for f in forbidden}]
    if found:
        print(f"WARNING: complex ops found: {found}")
    else:
        print("OK: no complex ops in graph.")


if __name__ == "__main__":
    main()
