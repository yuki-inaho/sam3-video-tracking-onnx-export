#!/usr/bin/env python3
"""Convert tracker image encoder ONNX weights to fp16 while preserving IO types."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16


def _position_encoding_node_names(model: onnx.ModelProto) -> list[str]:
    return [node.name for node in model.graph.node if "position_encoding" in node.name]


def convert_tracker_image_encoder_fp16(input_path: Path, output_path: Path) -> None:
    model = onnx.load_model(input_path, load_external_data=True)
    op_block_list = list(DEFAULT_OP_BLOCK_LIST)
    if "Range" not in op_block_list:
        op_block_list.append("Range")

    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=op_block_list,
        node_block_list=_position_encoding_node_names(model),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model_fp16, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/onnx/image_encoder_tracker.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/onnx/image_encoder_tracker_fp16.onnx"),
    )
    args = parser.parse_args()

    convert_tracker_image_encoder_fp16(args.input, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
