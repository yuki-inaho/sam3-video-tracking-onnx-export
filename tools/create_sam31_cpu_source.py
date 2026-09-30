"""Generate the reviewed CPU/ONNX copy of the pinned SAM 3.1 source."""

from __future__ import annotations

import argparse
from pathlib import Path

from sam3_onnx_equiv.sam31_source_patcher import create_sam31_cpu_source_copy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("sam31"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sam31_cpu_source"))
    args = parser.parse_args()
    result = create_sam31_cpu_source_copy(args.source_root, args.output_root)
    print(f"SAM 3.1 CPU source: {result.output_root} ({len(result.modified_files)} patched files)")


if __name__ == "__main__":
    main()
