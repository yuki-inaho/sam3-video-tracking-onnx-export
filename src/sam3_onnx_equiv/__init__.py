"""sam3_onnx_equiv — SAM3 video tracking ONNX equivalence layer.

Public API
----------
Video tracking (ONNX inference, no torch import needed):
    ``VideoOrchestrator``  — runs the full memory-bank tracking loop over ONNX modules.
    ``Constants``          — loads the Python-side SAM3 constants.
    ``PythonMemoryBank``   — per-frame memory store.
    ``ClipResult`` / ``FrameEntry`` — typed result / record containers.

ONNX export helpers (import ``sam3_onnx_equiv.export.*`` submodules; these pull torch):
    ``export.image_encoder``    — detector / tracker image encoder export.
    ``export.memory_attention`` — TransformerEncoderCrossAttention export.
    ``export.memory_encoder``   — SimpleMaskEncoder export.
    ``export.decode_head``      — prompt encoder + mask decoder + obj_ptr export.

Importing this package has no side effects and does not import torch / onnxruntime
(those are imported lazily inside the functions that need them).
"""

from __future__ import annotations

from sam3_onnx_equiv.video_orchestrator import (
    ClipResult,
    Constants,
    FrameEntry,
    PythonMemoryBank,
    VideoOrchestrator,
    make_oracle_frames,
)

__all__ = [
    "ClipResult",
    "Constants",
    "FrameEntry",
    "PythonMemoryBank",
    "VideoOrchestrator",
    "make_oracle_frames",
]
