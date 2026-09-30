"""Public, JSON-safe annotation helpers for SAM 3.1 video tracking.

The data helpers in this module deliberately do not import torch, onnxruntime,
or the official SAM package.  Those dependencies and the model weights are
loaded only when :class:`Sam31AnnotationEngine` opens its first frame folder.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image

MAX_OBJECTS = 16
MAX_PROMPT_POINTS = 64

_FRAME_NAME = re.compile(r"^([0-9]+)\.(?:jpe?g)$", re.IGNORECASE)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

# A fixed, high-contrast RGB palette.  Selecting by object ID makes rendering
# independent of Python hash randomization and request order.
_OVERLAY_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 188, 212),
    (255, 112, 67),
    (126, 87, 194),
    (102, 187, 106),
    (255, 202, 40),
    (66, 165, 245),
    (236, 64, 122),
    (141, 110, 99),
    (38, 198, 218),
    (255, 167, 38),
    (92, 107, 192),
    (156, 204, 101),
    (239, 83, 80),
    (41, 182, 246),
    (171, 71, 188),
    (124, 179, 66),
)


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class PromptPoint:
    """One normalized foreground (1) or background (0) point prompt."""

    x: float
    y: float
    label: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _coordinate(self.x, "x"))
        object.__setattr__(self, "y", _coordinate(self.y, "y"))
        label = _integer(self.label, "label")
        if label not in (0, 1):
            raise ValueError("label must be 0 (background) or 1 (foreground)")
        object.__setattr__(self, "label", label)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PromptPoint:
        """Parse a point from a request mapping, rejecting unknown fields."""
        if not isinstance(value, Mapping):
            raise ValueError("point must be an object")
        unknown = set(value) - {"x", "y", "label"}
        if unknown:
            raise ValueError(f"unknown point field(s): {', '.join(sorted(map(str, unknown)))}")
        missing = {"x", "y"} - set(value)
        if missing:
            raise ValueError(f"missing point field(s): {', '.join(sorted(missing))}")
        return cls(x=value["x"], y=value["y"], label=value.get("label", 1))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, float | int]:
        return {"x": self.x, "y": self.y, "label": self.label}


def validate_prompt_points(
    points: Sequence[PromptPoint | Mapping[str, object]],
) -> tuple[PromptPoint, ...]:
    """Return validated points and require at least one foreground prompt."""
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise ValueError("points must be a sequence")
    if len(points) > MAX_PROMPT_POINTS:
        raise ValueError(f"at most {MAX_PROMPT_POINTS} prompt points may be supplied")
    validated = tuple(
        point if isinstance(point, PromptPoint) else PromptPoint.from_mapping(point)
        for point in points
    )
    if not validated:
        raise ValueError("at least one point is required")
    if not any(point.label == 1 for point in validated):
        raise ValueError("at least one foreground point is required")
    return validated


def encode_uncompressed_rle(mask: object) -> dict[str, list[int]]:
    """Encode a two-dimensional binary mask as uncompressed COCO RLE.

    COCO walks pixels in column-major (Fortran) order.  Counts always begin
    with the background run, so a foreground top-left pixel produces a leading
    zero count.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    height, width = map(int, array.shape)
    if height < 1 or width < 1:
        raise ValueError("mask dimensions must be positive")
    if array.dtype.kind not in "biuf":
        raise ValueError("mask values must be binary")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("mask values must be finite and binary")
    if not np.logical_or(array == 0, array == 1).all():
        raise ValueError("mask values must be 0 or 1")

    flat = np.asarray(array, dtype=np.bool_).reshape(-1, order="F")
    counts: list[int] = []
    current = False
    run_length = 0
    for pixel in flat:
        value = bool(pixel)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            current = value
            run_length = 1
    counts.append(run_length)
    return {"size": [height, width], "counts": counts}


def _validated_rle(rle: object) -> tuple[int, int, list[int]]:
    if not isinstance(rle, Mapping):
        raise ValueError("RLE must be an object")
    if set(rle) != {"size", "counts"}:
        raise ValueError("RLE must contain exactly 'size' and 'counts'")

    size = rle["size"]
    if isinstance(size, (str, bytes)) or not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError("RLE size must be [height, width]")
    height = _integer(size[0], "RLE height", minimum=1)
    width = _integer(size[1], "RLE width", minimum=1)

    raw_counts = rle["counts"]
    if (
        isinstance(raw_counts, (str, bytes))
        or not isinstance(raw_counts, Sequence)
        or not raw_counts
    ):
        raise ValueError("RLE counts must be a non-empty integer array")
    counts = [
        _integer(count, f"RLE count {index}", minimum=0) for index, count in enumerate(raw_counts)
    ]
    if any(count == 0 for count in counts[1:]):
        raise ValueError("only the first RLE count may be zero")
    if sum(counts) != height * width:
        raise ValueError("RLE counts must sum exactly to height * width")
    return height, width, counts


def decode_uncompressed_rle(rle: object) -> np.ndarray:
    """Decode and strictly validate an uncompressed COCO RLE object."""
    height, width, counts = _validated_rle(rle)
    flat = np.empty(height * width, dtype=np.bool_)
    offset = 0
    foreground = False
    for count in counts:
        flat[offset : offset + count] = foreground
        offset += count
        foreground = not foreground
    return np.array(flat.reshape((height, width), order="F"), dtype=np.bool_, order="C")


# Explicit COCO names are useful at API boundaries while the shorter names stay
# convenient in application code.
encode_uncompressed_coco_rle = encode_uncompressed_rle
decode_uncompressed_coco_rle = decode_uncompressed_rle


def _bbox_and_area(mask: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    rows, columns = np.nonzero(mask)
    area = int(rows.size)
    if area == 0:
        return (0, 0, 0, 0), 0
    x_min = int(columns.min())
    x_max = int(columns.max())
    y_min = int(rows.min())
    y_max = int(rows.max())
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1), area


def finite_score(value: object) -> float | None:
    """Convert a scalar finite score to float; map missing/nonfinite scores to null."""
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class MaskAnnotation:
    """A per-object, per-frame mask record using absolute COCO XYWH bounds."""

    frame_index: int
    object_id: int
    score: float | None
    bbox: tuple[int, int, int, int]
    area: int
    segmentation: dict[str, list[int]]

    def __post_init__(self) -> None:
        frame_index = _integer(self.frame_index, "frame_index", minimum=0)
        object_id = _integer(self.object_id, "object_id")
        if isinstance(self.bbox, (str, bytes)) or len(self.bbox) != 4:
            raise ValueError("bbox must be (x, y, width, height)")
        bbox = tuple(
            _integer(value, f"bbox[{index}]", minimum=0) for index, value in enumerate(self.bbox)
        )
        area = _integer(self.area, "area", minimum=0)
        mask = decode_uncompressed_rle(self.segmentation)
        actual_bbox, actual_area = _bbox_and_area(mask)
        if bbox != actual_bbox or area != actual_area:
            raise ValueError("bbox and area must exactly describe segmentation")

        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "score", finite_score(self.score))
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "area", area)
        object.__setattr__(
            self,
            "segmentation",
            {"size": list(mask.shape), "counts": list(self.segmentation["counts"])},
        )

    @property
    def mask(self) -> dict[str, list[int]]:
        """Return a copy of the COCO mask record (a readable alias)."""
        return {
            "size": list(self.segmentation["size"]),
            "counts": list(self.segmentation["counts"]),
        }

    @property
    def rle(self) -> dict[str, list[int]]:
        return self.mask

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable annotation dictionary."""
        return {
            "frame_index": self.frame_index,
            "object_id": self.object_id,
            "score": self.score,
            "bbox": list(self.bbox),
            "area": self.area,
            "segmentation": self.mask,
        }


def mask_annotation(
    mask: object,
    *,
    frame_index: int,
    object_id: int,
    score: object = None,
) -> MaskAnnotation:
    """Build a validated annotation from one binary mask."""
    segmentation = encode_uncompressed_rle(mask)
    binary = decode_uncompressed_rle(segmentation)
    bbox, area = _bbox_and_area(binary)
    return MaskAnnotation(
        frame_index=frame_index,
        object_id=object_id,
        score=finite_score(score),
        bbox=bbox,
        area=area,
        segmentation=segmentation,
    )


annotation_from_mask = mask_annotation


def render_overlay(
    image: Image.Image | np.ndarray,
    annotations: Sequence[MaskAnnotation],
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Render masks and two-pixel bounding boxes with deterministic colors."""
    alpha_value = _coordinate(alpha, "alpha")
    if isinstance(image, Image.Image):
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    else:
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] not in (3, 4) or source.dtype != np.uint8:
            raise ValueError("image must be a uint8 RGB/RGBA array or a PIL image")
        pixels = source[..., :3].copy()

    height, width = pixels.shape[:2]
    ordered = sorted(annotations, key=lambda annotation: annotation.object_id)
    opacity = int(round(alpha_value * 255.0))
    inverse = 255 - opacity

    for annotation in ordered:
        mask = decode_uncompressed_rle(annotation.segmentation)
        if mask.shape != (height, width):
            raise ValueError("annotation mask size must match image size")
        color = np.asarray(
            _OVERLAY_COLORS[annotation.object_id % len(_OVERLAY_COLORS)], dtype=np.uint16
        )
        if opacity:
            base = pixels[mask].astype(np.uint16)
            pixels[mask] = ((base * inverse + color * opacity + 127) // 255).astype(np.uint8)

    # Draw bounds after all fills so their appearance does not depend on overlap.
    for annotation in ordered:
        x, y, box_width, box_height = annotation.bbox
        if box_width == 0 or box_height == 0:
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + box_width), min(height, y + box_height)
        if x0 >= x1 or y0 >= y1:
            continue
        color = np.asarray(
            _OVERLAY_COLORS[annotation.object_id % len(_OVERLAY_COLORS)], dtype=np.uint8
        )
        thickness = min(2, x1 - x0, y1 - y0)
        pixels[y0 : y0 + thickness, x0:x1] = color
        pixels[y1 - thickness : y1, x0:x1] = color
        pixels[y0:y1, x0 : x0 + thickness] = color
        pixels[y0:y1, x1 - thickness : x1] = color

    return Image.fromarray(pixels, mode="RGB")


render_annotation_overlay = render_overlay


def _as_numpy(value: object) -> np.ndarray:
    result = value
    if hasattr(result, "detach"):
        result = result.detach()  # type: ignore[union-attr]
    if hasattr(result, "cpu"):
        result = result.cpu()  # type: ignore[union-attr]
    if hasattr(result, "numpy"):
        try:
            result = result.numpy()  # type: ignore[union-attr]
        except TypeError:
            if not hasattr(result, "float"):
                raise
            result = result.float().numpy()  # type: ignore[union-attr]
    return np.asarray(result)


class Sam31AnnotationEngine:
    """Stateful CPU annotation engine backed by SAM 3.1 and the ONNX sessions."""

    def __init__(self, *, threads: int = 8, onnx_dir: str | Path | None = None) -> None:
        self._threads = _integer(threads, "threads", minimum=1)
        self._onnx_dir = Path(onnx_dir) if onnx_dir is not None else None
        self._model: Any | None = None
        self._sessions: Any | None = None
        self._state: dict[str, Any] | None = None
        self._frame_dir: Path | None = None
        self._frame_paths: tuple[Path, ...] = ()
        self._width = 0
        self._height = 0
        self._object_ids: list[int] = []
        self._results: dict[int, list[MaskAnnotation]] = {}
        self._has_propagated = False
        self._lock = RLock()

    @property
    def is_open(self) -> bool:
        return self._state is not None

    @property
    def object_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._object_ids)

    @property
    def media_info(self) -> Mapping[str, int]:
        with self._lock:
            if self._state is None:
                raise RuntimeError("open a frame directory first")
            return MappingProxyType(
                {
                    "frame_count": len(self._frame_paths),
                    "width": self._width,
                    "height": self._height,
                }
            )

    @property
    def results(self) -> Mapping[int, tuple[MaskAnnotation, ...]]:
        """Return a read-only snapshot of current per-frame annotations."""
        with self._lock:
            snapshot = {
                frame_index: tuple(self._copy_annotation(annotation) for annotation in annotations)
                for frame_index, annotations in sorted(self._results.items())
            }
            return MappingProxyType(snapshot)

    @property
    def result_frame_indices(self) -> tuple[int, ...]:
        """Return annotated frame keys without copying mask payloads."""
        with self._lock:
            return tuple(sorted(self._results))

    @staticmethod
    def _copy_annotation(annotation: MaskAnnotation) -> MaskAnnotation:
        return MaskAnnotation(
            frame_index=annotation.frame_index,
            object_id=annotation.object_id,
            score=annotation.score,
            bbox=annotation.bbox,
            area=annotation.area,
            segmentation=annotation.mask,
        )

    def _ensure_backend(self) -> None:
        if self._model is not None:
            return
        from sam3_onnx_equiv.sam31_model import build_sam31_tracker
        from sam3_onnx_equiv.sam31_onnx_video import (
            Sam31OnnxSessions,
            configure_sam31_onnx_window,
        )

        model = build_sam31_tracker(use_rope_real=True)
        configure_sam31_onnx_window(model)
        sessions = Sam31OnnxSessions(onnx_dir=self._onnx_dir, threads=self._threads)
        sessions.install(model)
        self._model = model
        self._sessions = sessions

    def _initialize_state(self, frame_dir: Path) -> dict[str, Any]:
        from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

        return VideoTrackingMultiplexDemo.init_state(
            self._model,
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

    @staticmethod
    def _inspect_frames(frame_dir: str | Path) -> tuple[Path, tuple[Path, ...], int, int]:
        directory = Path(frame_dir)
        if not directory.is_dir():
            raise ValueError("frame directory does not exist")

        numbered: list[tuple[int, Path]] = []
        for candidate in directory.iterdir():
            if not candidate.is_file():
                continue
            suffix = candidate.suffix.lower()
            if suffix in _IMAGE_SUFFIXES and suffix not in {".jpg", ".jpeg"}:
                raise ValueError("frame directory may contain only numbered JPEG images")
            if suffix not in {".jpg", ".jpeg"}:
                continue
            match = _FRAME_NAME.fullmatch(candidate.name)
            if match is None or candidate.is_symlink():
                raise ValueError("JPEG frame names must be non-negative integers")
            numbered.append((int(match.group(1)), candidate))

        if not numbered:
            raise ValueError("frame directory contains no numbered JPEG images")
        numbered.sort(key=lambda item: item[0])
        indices = [index for index, _ in numbered]
        if indices != list(range(len(numbered))):
            raise ValueError("JPEG frame numbers must be unique and contiguous from zero")

        paths = tuple(path for _, path in numbered)
        expected_size: tuple[int, int] | None = None
        for path in paths:
            try:
                with Image.open(path) as frame:
                    frame.load()
                    size = frame.size
            except Exception as exc:
                raise ValueError("a JPEG frame could not be decoded") from exc
            if size[0] < 1 or size[1] < 1:
                raise ValueError("JPEG frame dimensions must be positive")
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError("all JPEG frames must have the same dimensions")

        assert expected_size is not None
        return directory, paths, expected_size[0], expected_size[1]

    def open(self, frame_dir: str | Path) -> dict[str, int]:
        """Open a contiguous, zero-based numbered JPEG frame directory."""
        with self._lock:
            directory, paths, width, height = self._inspect_frames(frame_dir)
            self._ensure_backend()
            state = self._initialize_state(directory)
            if int(state.get("num_frames", -1)) != len(paths):
                raise RuntimeError("SAM initialized an unexpected number of frames")
            if (
                int(state.get("video_width", -1)) != width
                or int(state.get("video_height", -1)) != height
            ):
                raise RuntimeError("SAM initialized unexpected frame dimensions")

            self._state = state
            self._frame_dir = directory
            self._frame_paths = paths
            self._width = width
            self._height = height
            self._object_ids = []
            self._results = {}
            self._has_propagated = False
            return {"frame_count": len(paths), "width": width, "height": height}

    def close(self) -> None:
        """Release the current clip state while retaining the lazily loaded model."""
        with self._lock:
            self._state = None
            self._frame_dir = None
            self._frame_paths = ()
            self._width = 0
            self._height = 0
            self._object_ids = []
            self._results = {}
            self._has_propagated = False

    def _require_open(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("open a frame directory first")
        return self._state

    def _mask_array(self, masks: object, count: int) -> np.ndarray:
        array = _as_numpy(masks)
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        elif array.ndim == 2 and count == 1:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[0] != count:
            raise RuntimeError(f"SAM returned invalid mask shape {array.shape}")
        if array.shape[1:] != (self._height, self._width):
            raise RuntimeError(
                f"SAM returned mask size {array.shape[1:]}, expected {(self._height, self._width)}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError("SAM returned nonfinite mask logits")
        return array > 0

    @staticmethod
    def _score_list(scores: object | None, count: int) -> list[float | None]:
        """Convert the official object's presence logits to probabilities."""
        if scores is None:
            return [None] * count
        array = _as_numpy(scores)
        if array.size == 0:
            return [None] * count
        if array.ndim == 0:
            if count != 1:
                raise RuntimeError("SAM returned too few object scores")
            values = [array.item()]
        elif array.shape[0] == count:
            values = [array[index].reshape(-1)[0] for index in range(count)]
        elif count == 1:
            values = [array.reshape(-1)[0]]
        else:
            raise RuntimeError("SAM returned an unexpected number of object scores")
        probabilities: list[float | None] = []
        for value in values:
            logit = finite_score(value)
            if logit is None:
                probabilities.append(None)
            elif logit >= 0:
                probabilities.append(1.0 / (1.0 + math.exp(-logit)))
            else:
                exponential = math.exp(logit)
                probabilities.append(exponential / (1.0 + exponential))
        return probabilities

    def _records(
        self,
        frame_index: int,
        object_ids: Sequence[int],
        masks: object,
        scores: object | None,
    ) -> list[MaskAnnotation]:
        binary_masks = self._mask_array(masks, len(object_ids))
        score_values = self._score_list(scores, len(object_ids))
        return [
            mask_annotation(
                binary_masks[index],
                frame_index=frame_index,
                object_id=object_id,
                score=score_values[index],
            )
            for index, object_id in enumerate(object_ids)
        ]

    def _prime_refinement_feature(self, state: dict[str, Any], frame_index: int) -> None:
        """Cache a tracked frame before the official model builds a singleton state.

        SAM 3.1 refines an existing tracked object by extracting it into a new
        singleton inference state.  That state inherits ``cached_features`` but
        intentionally has no ``images`` entry.  The official cache retains only
        the most recently visited frame, so refining an earlier frame otherwise
        misses the cache and fails while trying to read the absent images entry.

        Prime the requested frame while the source state still owns the decoded
        frames.  Test doubles without the private official helper do not need
        this compatibility step.
        """
        if not state.get("tracking_has_started"):
            return
        cached_features = state.get("cached_features")
        if isinstance(cached_features, Mapping) and frame_index in cached_features:
            return
        get_image_feature = getattr(self._model, "_get_image_feature", None)
        if get_image_feature is None:
            return
        if "images" not in state:
            raise RuntimeError("SAM source state cannot restore the refinement frame")
        get_image_feature(state, frame_index, 1)
        refreshed_cache = state.get("cached_features")
        if not isinstance(refreshed_cache, Mapping) or frame_index not in refreshed_cache:
            raise RuntimeError("SAM did not cache the refinement frame")

    @staticmethod
    def _align_refinement_memory_dtype(state: dict[str, Any]) -> None:
        """Match compact tracking memories to the multiplex transition matrices.

        The official demo stores mask memories as bfloat16, while its multiplex
        state is constructed with float32 transition matrices.  CUDA autocast
        normally hides that difference.  CPU refinement performs direct matrix
        multiplies while extracting and merging singleton state, so restore the
        compact tensors to the matrix dtype before entering that path.
        """
        multiplex_state = state.get("multiplex_state")
        mux_matrix = getattr(multiplex_state, "mux_matrix", None)
        if mux_matrix is None:
            return

        target_dtype = mux_matrix.dtype
        target_device = mux_matrix.device

        def aligned(value: object) -> object:
            if value is None or not hasattr(value, "to") or not hasattr(value, "dtype"):
                return value
            if value.dtype == target_dtype and getattr(value, "device", None) == target_device:
                return value
            return value.to(device=target_device, dtype=target_dtype, non_blocking=True)

        output_dict = state.get("output_dict", {})
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for output in output_dict.get(storage_key, {}).values():
                for tensor_key in ("maskmem_features", "obj_ptr"):
                    if tensor_key in output:
                        output[tensor_key] = aligned(output[tensor_key])
                positions = output.get("maskmem_pos_enc")
                if isinstance(positions, list):
                    output["maskmem_pos_enc"] = [aligned(position) for position in positions]

        constants = state.get("constants", {})
        positions = constants.get("maskmem_pos_enc")
        if isinstance(positions, list):
            constants["maskmem_pos_enc"] = [aligned(position) for position in positions]

    @staticmethod
    def _validated_object_ids(value: object, expected: set[int]) -> list[int]:
        """Validate the official tracker's ID registry while preserving its order."""
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise RuntimeError("SAM returned an invalid object ID registry")
        try:
            object_ids = [_integer(item, "returned object_id") for item in value]
        except ValueError as exc:
            raise RuntimeError("SAM returned an invalid object ID registry") from exc
        if len(object_ids) != len(set(object_ids)) or set(object_ids) != expected:
            raise RuntimeError("SAM changed the registered object IDs")
        return object_ids

    @staticmethod
    def _state_score(state: Mapping[str, Any], frame_index: int, object_id: int) -> object | None:
        obj_index = state.get("obj_id_to_idx", {}).get(object_id)
        if obj_index is None:
            return None
        per_object = state.get("output_dict_per_obj", {}).get(obj_index, {})
        temporary = state.get("temp_output_dict_per_obj", {}).get(obj_index, {})
        for storage in (temporary, per_object):
            for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                output = storage.get(key, {}).get(frame_index)
                if output is not None and "object_score_logits" in output:
                    return output["object_score_logits"]
        return None

    @classmethod
    def _state_scores(
        cls,
        state: Mapping[str, Any],
        frame_index: int,
        object_ids: Sequence[int],
    ) -> list[object] | None:
        """Read per-object presence logits stored by the official tracker.

        The current Object Multiplex propagation generator returns four values,
        while writing ``object_score_logits`` into its per-object state.  Keep
        missing or malformed entries as NaN so the normal score conversion maps
        only those individual values to JSON ``null``.
        """
        values: list[object] = []
        found = False
        for object_id in object_ids:
            score = cls._state_score(state, frame_index, object_id)
            if score is None:
                values.append(float("nan"))
                continue
            array = _as_numpy(score)
            if array.size == 0:
                values.append(float("nan"))
                continue
            values.append(array.reshape(-1)[0])
            found = True
        return values if found else None

    def segment(
        self,
        frame_index: int,
        object_id: int,
        points: Sequence[PromptPoint | Mapping[str, object]],
    ) -> list[MaskAnnotation]:
        """Segment one object from all supplied point prompts on one frame."""
        with self._lock:
            state = self._require_open()
            frame = _integer(frame_index, "frame_index", minimum=0)
            if frame >= len(self._frame_paths):
                raise ValueError("frame_index is outside the open clip")
            stable_id = _integer(object_id, "object_id")
            prompts = validate_prompt_points(points)
            is_new = stable_id not in self._object_ids
            if is_new and len(self._object_ids) >= MAX_OBJECTS:
                raise ValueError(f"at most {MAX_OBJECTS} objects may be annotated")

            import torch

            coordinates = torch.tensor(
                [[point.x, point.y] for point in prompts], dtype=torch.float32
            )
            labels = torch.tensor([point.label for point in prompts], dtype=torch.int32)
            with torch.inference_mode():
                if stable_id in self._object_ids:
                    self._prime_refinement_feature(state, frame)
                    self._align_refinement_memory_dtype(state)
                output = self._model.add_new_points(
                    state,
                    frame,
                    stable_id,
                    coordinates,
                    labels,
                    clear_old_points=True,
                )
            if not isinstance(output, Sequence) or len(output) not in (4, 5):
                raise RuntimeError("SAM returned an invalid point-segmentation result")
            returned_frame, returned_ids, _low_resolution, masks = output[:4]
            ids = [_integer(value, "returned object_id") for value in returned_ids]
            if int(returned_frame) != frame or ids != [stable_id]:
                raise RuntimeError("SAM changed the requested frame or object ID")
            expected_ids = set(self._object_ids)
            expected_ids.add(stable_id)
            self._object_ids = self._validated_object_ids(state.get("obj_ids"), expected_ids)
            scores = output[4] if len(output) == 5 else self._state_score(state, frame, stable_id)
            records = self._records(frame, ids, masks, scores)

            if self._has_propagated:
                self._results = {}
                self._has_propagated = False
            by_id = {
                annotation.object_id: annotation for annotation in self._results.get(frame, [])
            }
            by_id[stable_id] = records[0]
            self._results[frame] = [
                by_id[current_id] for current_id in self._object_ids if current_id in by_id
            ]
            return [self._copy_annotation(annotation) for annotation in self._results[frame]]

    def propagate(self) -> dict[int, list[MaskAnnotation]]:
        """Propagate all objects sequentially across every frame in the open clip."""
        with self._lock:
            state = self._require_open()
            if not self._object_ids:
                raise RuntimeError("segment at least one object before propagation")

            collected: dict[int, list[MaskAnnotation]] = {}
            expected_frame = 0
            import torch

            with torch.inference_mode():
                self._model.propagate_in_video_preflight(state, run_mem_encoder=True)
                stream = self._model.propagate_in_video(
                    state,
                    0,
                    len(self._frame_paths) - 1,
                    False,
                    tqdm_disable=True,
                )
                for output in stream:
                    if not isinstance(output, Sequence) or len(output) not in (4, 5):
                        raise RuntimeError("SAM returned an invalid propagation result")
                    frame_index, returned_ids, _low_resolution, masks = output[:4]
                    frame = _integer(frame_index, "returned frame_index", minimum=0)
                    if frame != expected_frame:
                        raise RuntimeError("SAM did not propagate frames sequentially")
                    ids = [_integer(value, "returned object_id") for value in returned_ids]
                    ids = self._validated_object_ids(ids, set(self._object_ids))
                    scores = (
                        output[4] if len(output) == 5 else self._state_scores(state, frame, ids)
                    )
                    collected[frame] = self._records(frame, ids, masks, scores)
                    expected_frame += 1

            if expected_frame != len(self._frame_paths):
                raise RuntimeError("SAM did not return every frame")
            self._results = collected
            self._has_propagated = True
            return {
                frame_index: [self._copy_annotation(annotation) for annotation in annotations]
                for frame_index, annotations in collected.items()
            }


__all__ = [
    "MAX_OBJECTS",
    "MAX_PROMPT_POINTS",
    "MaskAnnotation",
    "PromptPoint",
    "Sam31AnnotationEngine",
    "annotation_from_mask",
    "decode_uncompressed_coco_rle",
    "decode_uncompressed_rle",
    "encode_uncompressed_coco_rle",
    "encode_uncompressed_rle",
    "finite_score",
    "mask_annotation",
    "render_annotation_overlay",
    "render_overlay",
    "validate_prompt_points",
]
