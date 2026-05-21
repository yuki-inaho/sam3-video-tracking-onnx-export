"""Synthetic image fixtures for deterministic SAM3 export validation.

These fixtures do not pretend to replace SAM3. They create simple images and exact
expected masks that can be used as a first sanity check for preprocessing,
postprocessing, and ONNX inference adapters before a gated SAM3 checkpoint is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, UInt8
from numpy.typing import NDArray
from PIL import Image, ImageDraw

PathLike: TypeAlias = str | Path
BoolMask: TypeAlias = Bool[NDArray[np.bool_], "height width"]
UInt8Mask: TypeAlias = UInt8[NDArray[np.uint8], "height width"]
BBoxXYXY: TypeAlias = tuple[int, int, int, int]
RgbColor: TypeAlias = tuple[int, int, int]

BLACK: RgbColor = (0, 0, 0)
RED: RgbColor = (255, 0, 0)


@dataclass(frozen=True)
class SyntheticCase:
    """Generated image, exact binary mask, and prompt metadata."""

    name: str
    image_path: Path
    mask_path: Path
    description: str
    prompt: str
    expected_bbox_xyxy: BBoxXYXY


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _empty_rgb_image(size: int) -> Image.Image:
    _validate_positive_int(size, "size")
    return Image.new("RGB", (size, size), BLACK)


def _bbox_from_mask(mask: BoolMask | UInt8Mask) -> BBoxXYXY:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("mask must contain at least one foreground pixel")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _circle_bbox(center: tuple[int, int], radius: int) -> BBoxXYXY:
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def _circle_mask(size: int, center: tuple[int, int], radius: int) -> BoolMask:
    yy, xx = np.ogrid[:size, :size]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius**2


def _save_case(
    name: str,
    image: Image.Image,
    mask: BoolMask | UInt8Mask,
    output_dir: Path,
    prompt: str,
    description: str,
) -> SyntheticCase:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{name}.png"
    mask_path = output_dir / f"{name}_mask.npy"
    image.save(image_path)
    np.save(mask_path, mask.astype(np.uint8))
    return SyntheticCase(name, image_path, mask_path, description, prompt, _bbox_from_mask(mask))


def make_red_circle_case(output_dir: PathLike, size: int = 128, radius: int = 24) -> SyntheticCase:
    """Create a single-frame black-background red-circle fixture."""
    _validate_positive_int(radius, "radius")
    output_dir = Path(output_dir)
    center = (size // 2, size // 2)
    image = _empty_rgb_image(size)
    ImageDraw.Draw(image).ellipse(_circle_bbox(center, radius), fill=RED)
    return _save_case(
        "black_bg_red_circle",
        image,
        _circle_mask(size, center, radius),
        output_dir,
        prompt="red circle",
        description="black background with one red circle",
    )


def make_red_rectangle_case(output_dir: PathLike, size: int = 128) -> SyntheticCase:
    """Create a single-frame black-background red-rectangle fixture."""
    output_dir = Path(output_dir)
    image = _empty_rgb_image(size)
    box: BBoxXYXY = (28, 36, 96, 88)
    ImageDraw.Draw(image).rectangle(box, fill=RED)
    mask = np.zeros((size, size), dtype=np.bool_)
    mask[box[1] : box[3] + 1, box[0] : box[2] + 1] = True
    return _save_case(
        "black_bg_red_rectangle",
        image,
        mask,
        output_dir,
        prompt="red rectangle",
        description="black background with one red rectangle",
    )


def make_moving_red_circle_sequence(
    output_dir: PathLike,
    frames: int = 3,
    size: int = 128,
    radius: int = 16,
) -> list[SyntheticCase]:
    """Create deterministic moving-circle frames for a tiny tracking fixture."""
    _validate_positive_int(frames, "frames")
    _validate_positive_int(radius, "radius")
    output_dir = Path(output_dir)
    cases: list[SyntheticCase] = []
    for idx in range(frames):
        center = (32 + idx * 24, 64)
        image = _empty_rgb_image(size)
        ImageDraw.Draw(image).ellipse(_circle_bbox(center, radius), fill=RED)
        cases.append(
            _save_case(
                f"moving_red_circle_{idx:02d}",
                image,
                _circle_mask(size, center, radius),
                output_dir,
                prompt="red circle",
                description=f"frame {idx}: black background with one translated red circle",
            )
        )
    return cases


def generate_all_synthetic_cases(output_dir: PathLike) -> list[SyntheticCase]:
    """Generate all deterministic image and mini-video fixtures."""
    output_dir = Path(output_dir)
    cases = [make_red_circle_case(output_dir), make_red_rectangle_case(output_dir)]
    cases.extend(make_moving_red_circle_sequence(output_dir / "video_frames"))
    return cases


def red_threshold_mask(image: Image.Image) -> UInt8Mask:
    """Strict deterministic red-object mask used only to validate synthetic fixtures."""
    arr = np.asarray(image.convert("RGB"))
    return ((arr[..., 0] >= 200) & (arr[..., 1] <= 30) & (arr[..., 2] <= 30)).astype(np.uint8)


def load_mask(path: PathLike) -> UInt8Mask:
    """Load a generated uint8 mask file."""
    return np.load(Path(path)).astype(np.uint8)
