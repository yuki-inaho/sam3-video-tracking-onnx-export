# SAM3 Video Tracking ONNX Export

## Prerequisites

- Python `>=3.10,<3.13`
- [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management
- [`just`](https://just.systems/man/en/) for the command runner
- Git submodule support
- Official SAM 3 checkpoint at `models/sam3.pt` for the original workflow
- Official SAM 3.1 Object Multiplex checkpoint at `models/sam3.1_multiplex.pt` for the new workflow

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

## SAM 3.1 Object Multiplex on CPU

The repository pins separate official SAM 3 and SAM 3.1 source submodules. Accept
the model terms for [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)
on Hugging Face, then place its `sam3.1_multiplex.pt` checkpoint at
`models/sam3.1_multiplex.pt`. The checkpoint is kept outside Git.

```bash
git submodule update --init --recursive
uv sync --extra dev --group dev
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just build-sam31
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 just sam31-e2e
```

`just build-sam31` makes a CPU-compatible copy of the pinned official source in
`outputs/sam31_cpu_source/`, then exports the TriHead image encoder, multiplex
mask decoder, multiplex memory encoder, and four bounded shapes of decoupled
memory attention to `outputs/onnx_sam31/`. It does not edit either Git submodule.
The SAM 3.1 E2E test compares a two-object, six-frame clip against the same official PyTorch
model, checks every frame's object IDs and mask IoU, and verifies ONNX session
calls. All inference uses the CPU.
First-frame interactive prompt handling and bucket bookkeeping use the official
PyTorch/Python code; the exported ONNX modules run image features and propagation.

To run point-prompted tracking on a folder of numbered image frames:

```bash
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 \
  uv run python tools/run_sam31_video.py --video-dir /path/to/frames \
  --point 1:0.25:0.35 --point 2:0.70:0.65 \
  --output outputs/reference/my_sam31_video.npz
```

The `--point` values are `object_id:x:y` with coordinates normalized to
`[0, 1]`. The NPZ stores frame indices, object IDs, mask logits, and object
scores. The ONNX path uses 16 slots per bucket, one previous mask-memory frame,
and up to two object-pointer frames; the PyTorch reference in the E2E test uses
the same window. This bounded window keeps attention ONNX shapes stable across
any clip length. The official unmodified model uses a longer temporal window,
so its outputs may differ.
