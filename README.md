# SAM3 Video Tracking ONNX Export

## Prerequisites

- Python `>=3.10,<3.13`
- [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management
- [`just`](https://just.systems/man/en/) for the command runner
- Git submodule support
- Official SAM3 checkpoint at `models/sam3.pt`

## Quick Start

Clone with submodules:

```bash
git clone --recursive git@github.com:yuki-inaho/sam3-video-tracking-onnx-export.git
cd sam3-video-tracking-onnx-export
```

If the repository was cloned without `--recursive`, initialize the SAM3 submodule:

```bash
git submodule update --init --recursive
```

Install dependencies:

```bash
uv sync --extra dev --group dev
```

Place the SAM3 checkpoint:

```bash
mkdir -p models
# put the official checkpoint at:
# models/sam3.pt
```

Generate the equivalent source tree and export all ONNX modules:

```bash
just build-all
```

## Minimum Commands

For a fresh checkout with `models/sam3.pt` already present:

```bash
git submodule update --init --recursive
uv sync --extra dev --group dev
just build-all
```

`just build-all` runs:

```bash
just equiv-source
just export-all
```

## ONNX Inference Demo

The notebook ONNX inference example is:

- [notebooks/sam3_onnx_video_demo.ipynb](notebooks/sam3_onnx_video_demo.ipynb)

It uses the exported ONNX modules under `outputs/onnx/` and constants under
`outputs/reference/constants/`, then compares the ONNX video-tracking path with
the PyTorch oracle.

## Gradio Web UI

Launch the bbox-prompted ONNX tracking UI:

```bash
just image-encoder-tracker-fp16
just webgui
```

The UI requires ONNX Runtime `CUDAExecutionProvider` and does not use TensorRT.
It keeps `CUDAExecutionProvider` first and permits `CPUExecutionProvider` fallback
for ORT-required ops. The `image-encoder-tracker-fp16` step creates
`outputs/onnx/image_encoder_tracker_fp16.onnx`, which is preferred automatically
when present and is intended for GTX 1070 / 8GB VRAM runs. The UI accepts
sequential image uploads, lets you select a base frame, draw one bounding box,
then generates per-frame masks and overlay visualizations under
`outputs/gradio_sessions/`.
