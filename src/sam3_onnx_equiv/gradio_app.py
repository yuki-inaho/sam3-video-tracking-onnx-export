"""Gradio app for bbox-prompted SAM3 ONNX video tracking.

The exported decode head currently accepts one point prompt.  The UI accepts a
user-drawn bounding box for ergonomics, then converts the first box to its centre
point before calling :class:`sam3_onnx_equiv.video_orchestrator.VideoOrchestrator`.
"""

from __future__ import annotations

import shutil
import site
import tempfile
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sam3_onnx_equiv.path_config import constants_dir, onnx_dir, repo_root

SESSION_ROOT = repo_root() / "outputs" / "gradio_sessions"
SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_DIRECTION_FRAMES = 7
OVERLAY_RGBA = (0, 199, 199, 118)
CUDA11_LIBRARY_RELATIVE_PATHS = (
    "nvidia/cuda_runtime/lib/libcudart.so.11.0",
    "nvidia/cublas/lib/libcublas.so.11",
    "nvidia/cublas/lib/libcublasLt.so.11",
    "nvidia/cufft/lib/libcufft.so.10",
    "nvidia/curand/lib/libcurand.so.10",
)


@dataclass(frozen=True)
class BBox:
    """Absolute xyxy box in displayed image pixel coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5)

    def clamped(self, width: int, height: int) -> BBox:
        xmin = min(max(self.xmin, 0.0), float(width - 1))
        ymin = min(max(self.ymin, 0.0), float(height - 1))
        xmax = min(max(self.xmax, 0.0), float(width - 1))
        ymax = min(max(self.ymax, 0.0), float(height - 1))
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        return BBox(xmin, ymin, xmax, ymax)


def _file_path(file_data: Any) -> Path:
    if isinstance(file_data, (str, Path)):
        return Path(file_data)
    if isinstance(file_data, dict) and file_data.get("name"):
        return Path(file_data["name"])
    name = getattr(file_data, "name", None)
    if name:
        return Path(name)
    path = getattr(file_data, "path", None)
    if path:
        return Path(path)
    raise TypeError(f"Unsupported uploaded file payload: {type(file_data)!r}")


def _session_dir() -> Path:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="sam3_track_", dir=SESSION_ROOT))


def _choice_label(index: int, path: str) -> str:
    return f"{index}: {Path(path).name}"


def parse_frame_choice(choice: str | int | None) -> int:
    """Return frame index from a dropdown label like ``"3: frame.png"``."""
    if choice is None:
        return 0
    if isinstance(choice, int):
        return choice
    head = str(choice).split(":", 1)[0].strip()
    return int(head) if head else 0


def extract_boxes_from_annotator(payload: Any) -> list[BBox]:
    """Extract xyxy boxes from ``gradio_image_annotation`` output."""
    if not isinstance(payload, dict):
        return []
    raw_boxes = payload.get("boxes") or []
    boxes: list[BBox] = []
    for box in raw_boxes:
        if not isinstance(box, dict):
            continue
        if not all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
            continue
        boxes.append(
            BBox(
                xmin=float(box["xmin"]),
                ymin=float(box["ymin"]),
                xmax=float(box["xmax"]),
                ymax=float(box["ymax"]),
            )
        )
    return boxes


def bbox_to_point_prompt(box: BBox, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a bbox to the single positive point prompt expected by ONNX."""
    box = box.clamped(width, height)
    if box.width < 1 or box.height < 1:
        raise ValueError("Draw a non-empty bounding box on the base frame.")
    cx, cy = box.center
    coords = np.array([[[cx / float(width), cy / float(height)]]], dtype=np.float32)
    labels = np.array([[1]], dtype=np.int32)
    return coords, labels


def select_directional_windows(
    frame_paths: list[str],
    base_index: int,
    max_direction_frames: int,
) -> tuple[list[int], list[int]]:
    """Return original-frame indices for forward and reverse tracking runs."""
    if not frame_paths:
        raise ValueError("No frames are loaded.")
    if base_index < 0 or base_index >= len(frame_paths):
        raise IndexError(f"Base frame index out of range: {base_index}")
    max_direction_frames = max(1, min(int(max_direction_frames), MAX_DIRECTION_FRAMES))
    forward = list(range(base_index, min(len(frame_paths), base_index + max_direction_frames)))
    reverse = list(range(base_index, max(-1, base_index - max_direction_frames), -1))
    return forward, reverse


def load_uploaded_frames(files: list[Any] | None) -> tuple[dict[str, Any], list[str], Any, str]:
    """Copy uploaded image files into a session folder sorted by filename."""
    import gradio as gr  # noqa: PLC0415

    if not files:
        return (
            {},
            [],
            gr.update(choices=[], value=None),
            "Upload one or more sequential image files.",
        )

    src_paths = sorted(_file_path(f) for f in files)
    src_paths = [p for p in src_paths if p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not src_paths:
        return {}, [], gr.update(choices=[], value=None), "No supported image files found."

    session_dir = _session_dir()
    image_dir = session_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[str] = []
    for idx, src in enumerate(src_paths):
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            dst = image_dir / f"{idx:06d}_{src.stem}.png"
            rgb.save(dst)
        frame_paths.append(str(dst))

    state = {"session_dir": str(session_dir), "frame_paths": frame_paths}
    choices = [_choice_label(i, p) for i, p in enumerate(frame_paths)]
    dropdown = gr.update(choices=choices, value=choices[0] if choices else None)
    return state, frame_paths, dropdown, f"Loaded {len(frame_paths)} frame(s). Select a base frame."


def annotator_payload_for_frame(
    state: dict[str, Any] | None, choice: str | int | None
) -> dict[str, Any]:
    """Build an annotator value for the selected base frame."""
    if not state or not state.get("frame_paths"):
        return {"image": None, "boxes": []}
    frame_paths = list(state["frame_paths"])
    idx = parse_frame_choice(choice)
    idx = min(max(idx, 0), len(frame_paths) - 1)
    image = np.asarray(Image.open(frame_paths[idx]).convert("RGB"))
    return {"image": image, "boxes": []}


def _overlay_mask(
    frame: Image.Image, mask_288: np.ndarray, bbox: BBox | None, label: str
) -> Image.Image:
    frame = frame.convert("RGB")
    w, h = frame.size
    mask_img = Image.fromarray(mask_288.astype(np.uint8) * 255, mode="L").resize(
        (w, h), Image.Resampling.NEAREST
    )
    overlay = Image.new("RGBA", (w, h), OVERLAY_RGBA)
    transparent = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    blended = Image.composite(overlay, transparent, mask_img)
    out = Image.alpha_composite(frame.convert("RGBA"), blended).convert("RGB")
    draw = ImageDraw.Draw(out)
    if bbox is not None:
        box = bbox.clamped(w, h)
        draw.rectangle([box.xmin, box.ymin, box.xmax, box.ymax], outline=(255, 230, 0), width=3)
    draw.text((8, 8), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def _save_result_bundle(
    session_dir: Path,
    frame_paths: list[str],
    masks_by_index: dict[int, np.ndarray],
    scores_by_index: dict[int, float],
    bbox: BBox,
    base_index: int,
) -> tuple[list[str], str]:
    result_dir = session_dir / "tracking_results"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    mask_dir = result_dir / "masks"
    overlay_dir = result_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    overlay_paths: list[str] = []
    for idx in sorted(masks_by_index):
        frame = Image.open(frame_paths[idx]).convert("RGB")
        w, h = frame.size
        mask = masks_by_index[idx]
        mask_full = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
            (w, h), Image.Resampling.NEAREST
        )
        mask_path = mask_dir / f"mask_{idx:06d}.png"
        mask_full.save(mask_path)
        score = scores_by_index.get(idx, float("nan"))
        overlay = _overlay_mask(
            frame,
            mask,
            bbox if idx == base_index else None,
            f"{idx} score={score:.3f}",
        )
        overlay_path = overlay_dir / f"overlay_{idx:06d}.png"
        overlay.save(overlay_path)
        overlay_paths.append(str(overlay_path))

    zip_path = result_dir / "sam3_onnx_tracking_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(mask_dir.glob("*.png")) + sorted(overlay_dir.glob("*.png")):
            zf.write(path, path.relative_to(result_dir))
    return overlay_paths, str(zip_path)


def _run_one_direction(
    orch: Any,
    frame_paths: list[str],
    indices: list[int],
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    emulate_bf16: bool,
) -> dict[str, Any]:
    frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]
    return orch.run_clip(
        frames_pil=frames,
        frame0_point_coords_norm=point_coords,
        frame0_point_labels=point_labels,
        use_memory=True,
        emulate_oracle_bf16=emulate_bf16,
    )


@lru_cache(maxsize=1)
def _preload_cuda11_runtime_libraries() -> None:
    import ctypes  # noqa: PLC0415

    roots = [Path(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))

    missing: list[str] = []
    for relative in CUDA11_LIBRARY_RELATIVE_PATHS:
        path = next((root / relative for root in roots if (root / relative).exists()), None)
        if path is None:
            missing.append(relative)
            continue
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

    if missing:
        raise RuntimeError(
            "CUDA 11 runtime libraries required by onnxruntime-gpu==1.18.0 were not found: "
            + ", ".join(missing)
        )


def _cuda_execution_providers() -> list[str]:
    _preload_cuda11_runtime_libraries()

    import onnxruntime as ort  # noqa: PLC0415

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available in onnxruntime. "
            f"available providers: {available}. Install the webgui extra with "
            "`uv sync --extra webgui --extra dev --group dev` or run `just webgui`."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _tracker_image_encoder_name() -> str:
    fp16_name = "image_encoder_tracker_fp16.onnx"
    if (onnx_dir() / fp16_name).exists():
        return fp16_name
    return "image_encoder_tracker.onnx"


@lru_cache(maxsize=1)
def _get_orchestrator() -> Any:
    from sam3_onnx_equiv.video_orchestrator import VideoOrchestrator  # noqa: PLC0415

    return VideoOrchestrator(
        onnx_dir=onnx_dir(),
        constants_dir=constants_dir(),
        providers=_cuda_execution_providers(),
        image_encoder_name=_tracker_image_encoder_name(),
    )


def run_tracking(
    state: dict[str, Any] | None,
    base_choice: str | int | None,
    annotation: Any,
    max_direction_frames: int,
    run_reverse: bool,
    emulate_bf16: bool,
) -> tuple[list[str], str | None, str]:
    """Run bbox-centre prompted tracking and save masks + overlays."""
    if not state or not state.get("frame_paths") or not state.get("session_dir"):
        return [], None, "Upload and confirm sequential frames first."

    frame_paths = list(state["frame_paths"])
    base_index = parse_frame_choice(base_choice)
    base_image = Image.open(frame_paths[base_index]).convert("RGB")
    width, height = base_image.size

    boxes = extract_boxes_from_annotator(annotation)
    if not boxes:
        return [], None, "Draw one bounding box on the selected base frame."
    bbox = boxes[0].clamped(width, height)
    point_coords, point_labels = bbox_to_point_prompt(bbox, width, height)

    started = time.time()
    orch = _get_orchestrator()

    forward_indices, reverse_indices = select_directional_windows(
        frame_paths, base_index, max_direction_frames
    )
    masks_by_index: dict[int, np.ndarray] = {}
    scores_by_index: dict[int, float] = {}

    forward = _run_one_direction(
        orch, frame_paths, forward_indices, point_coords, point_labels, emulate_bf16
    )
    for original_idx, mask, score in zip(
        forward_indices, forward["masks"], forward["scores"], strict=True
    ):
        masks_by_index[original_idx] = np.asarray(mask, dtype=bool)
        scores_by_index[original_idx] = float(score)

    reverse_count = 0
    if run_reverse and len(reverse_indices) > 1:
        backward = _run_one_direction(
            orch, frame_paths, reverse_indices, point_coords, point_labels, emulate_bf16
        )
        for original_idx, mask, score in zip(
            reverse_indices, backward["masks"], backward["scores"], strict=True
        ):
            masks_by_index[original_idx] = np.asarray(mask, dtype=bool)
            scores_by_index[original_idx] = float(score)
        reverse_count = len(reverse_indices)

    overlay_paths, zip_path = _save_result_bundle(
        Path(state["session_dir"]), frame_paths, masks_by_index, scores_by_index, bbox, base_index
    )
    elapsed = time.time() - started
    summary = (
        f"Tracked {len(masks_by_index)} frame(s) in {elapsed:.1f}s. "
        f"base={base_index}, forward={len(forward_indices)}, reverse={reverse_count}. "
        "Masks and overlays were written under outputs/gradio_sessions/."
    )
    return overlay_paths, zip_path, summary


def build_demo() -> Any:
    """Create the Gradio Blocks app."""
    import gradio as gr
    from gradio_image_annotation import image_annotator

    with gr.Blocks(title="SAM3 ONNX Video Tracking") as demo:
        gr.Markdown("# SAM3 ONNX Video Tracking")
        gr.Markdown(
            "Upload sequential images, select a base frame, draw one bounding box, "
            "then run ONNX memory-bank tracking. The app saves per-frame masks and "
            "overlay visualizations."
        )

        frame_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1):
                uploads = gr.File(
                    label="Sequential images",
                    file_count="multiple",
                    file_types=["image"],
                )
                confirm = gr.Button("Confirm frames", variant="primary")
                base_frame = gr.Dropdown(label="Base frame", choices=[], interactive=True)
                max_frames = gr.Slider(
                    1,
                    MAX_DIRECTION_FRAMES,
                    value=6,
                    step=1,
                    label="Max frames per direction",
                )
                run_reverse = gr.Checkbox(label="Track backward from base frame", value=True)
                emulate_bf16 = gr.Checkbox(label="Emulate bf16 memory storage", value=False)
                run = gr.Button("Track and generate masks", variant="primary")
                status = gr.Textbox(label="Status", lines=5)
            with gr.Column(scale=1):
                preview = gr.Gallery(label="Confirmed frames", columns=4, height=320)
                annotator = image_annotator(
                    label_list=["object"],
                    label="Base-frame bbox annotation",
                )
            with gr.Column(scale=1):
                overlays = gr.Gallery(label="Mask overlays", columns=2, height=520)
                result_zip = gr.File(label="Download masks + overlays")

        confirm.click(
            fn=load_uploaded_frames,
            inputs=[uploads],
            outputs=[frame_state, preview, base_frame, status],
        ).then(
            fn=annotator_payload_for_frame,
            inputs=[frame_state, base_frame],
            outputs=[annotator],
        )
        base_frame.change(
            fn=annotator_payload_for_frame,
            inputs=[frame_state, base_frame],
            outputs=[annotator],
        )
        run.click(
            fn=run_tracking,
            inputs=[frame_state, base_frame, annotator, max_frames, run_reverse, emulate_bf16],
            outputs=[overlays, result_zip, status],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue(max_size=8).launch(show_error=True)


if __name__ == "__main__":
    main()
