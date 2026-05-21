#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sam3_onnx_equiv.sam3_source_patcher import create_equivalent_source_copy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a SAM3 source copy with explicit real-valued RoPE builder wiring"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = create_equivalent_source_copy(args.source_root, args.output_root)
    print(f"source={result.source_root}")
    print(f"output={result.output_root}")
    for path in result.modified_files:
        print(f"modified={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
