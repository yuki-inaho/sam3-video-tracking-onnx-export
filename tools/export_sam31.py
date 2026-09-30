"""Export official SAM 3.1 Object Multiplex tensor modules to ONNX."""

from __future__ import annotations

import argparse

from sam3_onnx_equiv.export.sam31 import export_sam31


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=[
            "memory_encoder",
            "mask_decoder",
            "memory_attention",
            "memory_attention_m1_p2",
            "memory_attention_m2_p1",
            "memory_attention_m2_p2",
            "image_encoder",
        ],
    )
    args = parser.parse_args()
    for path in export_sam31(only=args.only):
        print(path, flush=True)


if __name__ == "__main__":
    main()
