"""Local HTTP application for interactive SAM 3.1 annotation.

The browser is deliberately kept separate from the model process.  Uploaded
media is copied into a server-owned workspace, decoded to numbered JPEGs, and
then passed to the existing SAM 3.1 Object Multiplex tracker.  Client supplied
paths are never opened directly.
"""

from __future__ import annotations

import fcntl
import ipaddress
import json
import math
import secrets
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast
from urllib.parse import parse_qs, quote, unquote, urlsplit

from PIL import Image

from sam3_onnx_equiv.annotation import (
    MAX_PROMPT_POINTS,
    MaskAnnotation,
    PromptPoint,
    Sam31AnnotationEngine,
    decode_uncompressed_rle,
    render_overlay,
)
from sam3_onnx_equiv.path_config import repo_root

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_VIDEO_FRAMES = 60
MAX_LABEL_LENGTH = 120
MAX_FRAME_PIXELS = 4096 * 4096
MAX_DECODED_PIXELS = 600_000_000
PROJECT_MARKER = ".sam31-annotation-project"
PROJECT_MARKER_CONTENT = "sam31-annotation-project-v1\n"
PROJECT_HEADER = "X-SAM31-Project"
WORKSPACE_LOCK = ".sam31-annotation.lock"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class DecodedMediaTooLargeError(ValueError):
    """Raised when decoded dimensions exceed bounded annotation capacity."""


class ProjectGenerationError(Exception):
    """Raised when a request does not target the current annotation project."""


class WorkspaceInUseError(RuntimeError):
    """Raised before cleanup when another service owns the workspace."""


class AnnotationEngine(Protocol):
    """Small engine surface used by the HTTP layer and its tests."""

    def open(self, frame_dir: Path) -> dict[str, object]: ...

    def segment(
        self, frame_index: int, object_id: int, points: list[PromptPoint]
    ) -> list[MaskAnnotation]: ...

    def propagate(self) -> dict[int, list[MaskAnnotation]]: ...

    @property
    def results(self) -> Mapping[int, tuple[MaskAnnotation, ...]]: ...

    @property
    def result_frame_indices(self) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class MediaInfo:
    """Public, portable metadata for the currently opened media."""

    name: str
    kind: str
    width: int
    height: int
    frame_count: int
    fps: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps": self.fps,
        }


def _safe_filename(raw_name: str) -> str:
    """Return a display/storage basename without accepting a client path."""
    name = Path(unquote(raw_name).replace("\\", "/")).name.strip()
    if name in {"", ".", ".."}:
        raise ValueError("A media filename is required")
    cleaned = "".join(ch for ch in name if ch.isprintable() and ch not in "\r\n\0")
    if not cleaned:
        raise ValueError("The media filename is invalid")
    encoded = cleaned.encode("utf-8")[:240]
    while True:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            encoded = encoded[: exc.start]


def _normalise_max_frames(raw: str | None, configured_max: int) -> int:
    if raw is None or raw == "":
        return configured_max
    try:
        requested = int(raw)
    except ValueError as exc:
        raise ValueError("max_frames must be an integer") from exc
    if requested < 1:
        raise ValueError("max_frames must be positive")
    return min(requested, configured_max)


def _is_loopback_host(host: str) -> bool:
    candidate = host.strip().rstrip(".").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate.partition("%")[0]).is_loopback
    except ValueError:
        return False


def _require_loopback_bind_host(host: str) -> None:
    """Reject wildcard, public, and DNS-dependent bind addresses."""
    if not _is_loopback_host(host):
        raise ValueError("The annotation server may bind only to a loopback host")


def _parse_host_header(value: str) -> tuple[str, int | None]:
    if not value or any(character.isspace() for character in value):
        raise ForbiddenRequestError("The Host header is invalid")
    try:
        parsed = urlsplit(f"//{value}")
        port = parsed.port
    except ValueError as exc:
        raise ForbiddenRequestError("The Host header is invalid") from exc
    if (
        parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ForbiddenRequestError("The Host header is invalid")
    return parsed.hostname.rstrip(".").lower(), port


def _write_image_frames(source: Path, frame_dir: Path) -> tuple[int, int, int, float]:
    with Image.open(source) as image:
        width, height = image.size
        _validate_decoded_pixels(width, height, 1)
        rgb = image.convert("RGB")
        rgb.save(frame_dir / "000000.jpg", format="JPEG", quality=95)
    return width, height, 1, 0.0


def _write_video_frames(
    source: Path, frame_dir: Path, max_frames: int
) -> tuple[int, int, int, float]:
    # OpenCV is already a project dependency, but importing it lazily keeps
    # unit tests and `--help` fast and avoids initialising its native runtime.
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("The uploaded file is neither a supported image nor a readable video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = height = count = 0
    try:
        while count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if count == 0:
                height, width = frame.shape[:2]
                _validate_decoded_pixels(width, height, 1)
            elif frame.shape[1] != width or frame.shape[0] != height:
                raise ValueError("Video frames must have a constant size")
            _validate_decoded_pixels(width, height, count + 1)
            destination = frame_dir / f"{count:06d}.jpg"
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError("Could not write a decoded video frame")
            count += 1
    finally:
        capture.release()

    if count == 0:
        raise ValueError("The video contains no readable frames")
    return width, height, count, fps


def _validate_decoded_pixels(width: int, height: int, frame_count: int) -> None:
    pixels = width * height
    if width < 1 or height < 1 or pixels > MAX_FRAME_PIXELS:
        raise DecodedMediaTooLargeError(
            f"Decoded frames may contain at most {MAX_FRAME_PIXELS} pixels each"
        )
    if frame_count < 1 or pixels * frame_count > MAX_DECODED_PIXELS:
        raise DecodedMediaTooLargeError(
            f"Decoded media may contain at most {MAX_DECODED_PIXELS} pixels in total"
        )


def _decode_media(
    source: Path, frame_dir: Path, *, max_frames: int
) -> tuple[str, int, int, int, float]:
    """Decode an upload into the tracker format without trusting its suffix."""
    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        try:
            width, height, count, fps = _write_image_frames(source, frame_dir)
            return "image", width, height, count, fps
        except DecodedMediaTooLargeError:
            raise
        except (OSError, ValueError):
            # A file can have the wrong extension; let the video decoder make
            # the final determination rather than accepting an empty project.
            pass
    else:
        try:
            width, height, count, fps = _write_image_frames(source, frame_dir)
            return "image", width, height, count, fps
        except DecodedMediaTooLargeError:
            raise
        except (OSError, ValueError):
            pass

    width, height, count, fps = _write_video_frames(source, frame_dir, max_frames)
    return "video", width, height, count, fps


class AnnotationService:
    """Serialised project state shared by all local HTTP requests."""

    def __init__(
        self,
        workspace: Path,
        *,
        engine: AnnotationEngine | None = None,
        threads: int = 8,
        max_video_frames: int = MAX_VIDEO_FRAMES,
    ) -> None:
        if max_video_frames < 1:
            raise ValueError("max_video_frames must be positive")
        selected_engine = engine or Sam31AnnotationEngine(threads=threads)
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        workspace_lock_file = (self.workspace / WORKSPACE_LOCK).open("a+b")
        try:
            fcntl.flock(workspace_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            workspace_lock_file.close()
            raise WorkspaceInUseError("The annotation workspace is already in use") from exc
        self._workspace_lock_file: BinaryIO | None = workspace_lock_file
        for stale_project in self.workspace.glob("project-*"):
            marker = stale_project / PROJECT_MARKER
            try:
                owned = (
                    stale_project.is_dir()
                    and not stale_project.is_symlink()
                    and marker.read_text(encoding="utf-8") == PROJECT_MARKER_CONTENT
                )
            except (OSError, UnicodeError):
                owned = False
            if owned:
                shutil.rmtree(stale_project, ignore_errors=True)
        self._project_dir: Path | None = None
        self.frame_dir = self.workspace / "frames"
        self.upload_dir = self.workspace / "uploads"
        self.engine = selected_engine
        self.max_video_frames = max_video_frames
        self.media: MediaInfo | None = None
        self.project_id: str | None = None
        self.labels: dict[int, str] = {}
        self.prompts: dict[int, list[dict[str, object]]] = {}
        self.propagation_required = False
        self._lock = threading.RLock()

    def close(self) -> None:
        """Release this service's exclusive workspace ownership."""
        lock_file = self._workspace_lock_file
        if lock_file is None:
            return
        self._workspace_lock_file = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "model": "SAM 3.1 Object Multiplex",
            "runtime": "ONNX Runtime CPU",
            "media_loaded": self.media is not None,
            "max_video_frames": self.max_video_frames,
        }

    def project(self) -> dict[str, object]:
        with self._lock:
            return {
                "ok": True,
                "media": self.media.to_dict() if self.media else None,
                "project_id": self.project_id,
                "objects": self._objects(),
                "frames": self._frames(),
                "propagation_required": self.propagation_required,
            }

    def open_upload(self, filename: str, payload: bytes, max_frames: int) -> dict[str, object]:
        if not payload:
            raise ValueError("The uploaded media is empty")
        safe_name = _safe_filename(filename)
        with self._lock:
            project_dir = Path(tempfile.mkdtemp(prefix="project-", dir=self.workspace))
            staged_frame_dir = project_dir / "frames"
            staged_upload_dir = project_dir / "uploads"
            try:
                (project_dir / PROJECT_MARKER).write_text(PROJECT_MARKER_CONTENT, encoding="utf-8")
                staged_frame_dir.mkdir()
                staged_upload_dir.mkdir()
                source = staged_upload_dir / safe_name
                source.write_bytes(payload)
                kind, width, height, frame_count, fps = _decode_media(
                    source, staged_frame_dir, max_frames=max_frames
                )
                next_media = MediaInfo(
                    name=safe_name,
                    kind=kind,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    fps=fps,
                )
                next_project_id = secrets.token_urlsafe(24)
                self.engine.open(staged_frame_dir)
            except Exception:
                shutil.rmtree(project_dir, ignore_errors=True)
                raise

            old_project_dir = self._project_dir
            self._project_dir = project_dir
            self.frame_dir = staged_frame_dir
            self.upload_dir = staged_upload_dir
            self.media = next_media
            self.project_id = next_project_id
            self.labels.clear()
            self.prompts.clear()
            self.propagation_required = False
            if old_project_dir is not None:
                shutil.rmtree(old_project_dir, ignore_errors=True)
            return {
                "ok": True,
                "media": self.media.to_dict(),
                "project_id": self.project_id,
            }

    @contextmanager
    def guard_project(self, supplied_project_id: str | None) -> Iterator[None]:
        """Hold the project generation stable for one HTTP operation."""
        with self._lock:
            if (
                self.media is None
                or self.project_id is None
                or supplied_project_id is None
                or not secrets.compare_digest(supplied_project_id, self.project_id)
            ):
                raise ProjectGenerationError("The annotation project is missing or stale")
            yield

    @contextmanager
    def guard_upload(self, supplied_project_id: str | None) -> Iterator[None]:
        """Prevent a stale tab from replacing a newer project generation."""
        with self._lock:
            if self.project_id is None:
                if supplied_project_id is not None:
                    raise ProjectGenerationError("The annotation project is missing or stale")
            elif supplied_project_id is None or not secrets.compare_digest(
                supplied_project_id, self.project_id
            ):
                raise ProjectGenerationError("The annotation project is missing or stale")
            yield

    def segment(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            media = self._require_media()
            frame_index = _required_int(payload, "frame_index", minimum=0)
            if frame_index >= media.frame_count:
                raise ValueError("frame_index is outside the loaded media")
            object_id = _required_int(payload, "object_id", minimum=1, maximum=16)
            label = str(payload.get("label", "object")).strip() or "object"
            if len(label) > MAX_LABEL_LENGTH:
                raise ValueError(f"label must be at most {MAX_LABEL_LENGTH} characters")
            raw_points = payload.get("points")
            if not isinstance(raw_points, list):
                raise ValueError("points must be a list")
            if len(raw_points) > MAX_PROMPT_POINTS:
                raise ValueError(f"at most {MAX_PROMPT_POINTS} prompt points may be supplied")
            points = [
                PromptPoint.from_mapping(cast(Mapping[str, object], point)) for point in raw_points
            ]
            previous_frames = set(self._result_frame_indices())
            annotations = self.engine.segment(frame_index, object_id, points)
            current_frames = set(self._result_frame_indices())
            self.labels[object_id] = label
            retained_prompts = [
                prompt
                for prompt in self.prompts.get(object_id, [])
                if prompt.get("frame_index") != frame_index
            ]
            retained_prompts.extend(
                {
                    "frame_index": frame_index,
                    "x": point.x,
                    "y": point.y,
                    "label": point.label,
                }
                for point in points
            )
            self.prompts[object_id] = sorted(
                retained_prompts,
                key=lambda prompt: (
                    int(prompt["frame_index"]),
                    float(prompt["y"]),
                    float(prompt["x"]),
                    int(prompt["label"]),
                ),
            )
            invalidated_frames = sorted(previous_frames - current_frames)
            self.propagation_required = self.propagation_required or bool(invalidated_frames)
            return {
                "ok": True,
                "frame_index": frame_index,
                "annotations": [self._record_dict(item) for item in annotations],
                "objects": self._objects(),
                "invalidated_frames": invalidated_frames,
                "propagation_required": self.propagation_required,
            }

    def propagate(self) -> dict[str, object]:
        with self._lock:
            self._require_media()
            results = self.engine.propagate()
            self.propagation_required = False
            return {
                "ok": True,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "annotations": [self._record_dict(item) for item in records],
                    }
                    for frame_index, records in sorted(results.items())
                ],
                "objects": self._objects(),
                "propagation_required": self.propagation_required,
            }

    def reset(self) -> dict[str, object]:
        with self._lock:
            self._require_media()
            self.engine.open(self.frame_dir)
            self.labels.clear()
            self.prompts.clear()
            self.propagation_required = False
            return {"ok": True, "media": self.media.to_dict() if self.media else None}

    def frame_path(self, frame_index: int) -> Path:
        with self._lock:
            media = self._require_media()
            if not 0 <= frame_index < media.frame_count:
                raise ValueError("frame index is outside the loaded media")
            path = self.frame_dir / f"{frame_index:06d}.jpg"
            if not path.is_file():
                raise FileNotFoundError("frame is missing")
            return path

    def frame_bytes(self, frame_index: int) -> bytes:
        """Read one frame while preventing a concurrent media replacement."""
        with self._lock:
            return self.frame_path(frame_index).read_bytes()

    def export_document(self) -> dict[str, object]:
        with self._lock:
            media = self._require_media()
            categories: list[dict[str, object]] = []
            category_ids: dict[str, int] = {}
            for object_id in sorted(self.labels):
                label = self.labels[object_id]
                if label not in category_ids:
                    category_ids[label] = len(category_ids) + 1
                    categories.append(
                        {"id": category_ids[label], "name": label, "supercategory": ""}
                    )

            images = [
                {
                    "id": frame_index + 1,
                    "file_name": f"{frame_index:06d}.jpg",
                    "width": media.width,
                    "height": media.height,
                    "frame_index": frame_index,
                    **({"video_id": 1} if media.kind == "video" else {}),
                }
                for frame_index in range(media.frame_count)
            ]

            annotations: list[dict[str, object]] = []
            annotation_id = 1
            for frame_index, records in sorted(self.engine.results.items()):
                for record in sorted(records, key=lambda item: item.object_id):
                    label = self.labels.get(record.object_id, f"object-{record.object_id}")
                    if label not in category_ids:
                        category_ids[label] = len(category_ids) + 1
                        categories.append(
                            {"id": category_ids[label], "name": label, "supercategory": ""}
                        )
                    item = record.to_dict()
                    item.update(
                        {
                            "id": annotation_id,
                            "image_id": frame_index + 1,
                            "category_id": category_ids[label],
                            "track_id": record.object_id,
                            "iscrowd": 0,
                        }
                    )
                    item.pop("frame_index", None)
                    item.pop("object_id", None)
                    annotations.append(item)
                    annotation_id += 1

            document: dict[str, object] = {
                "info": {
                    "description": "SAM 3.1 image/video annotation",
                    "version": "1.0",
                },
                "licenses": [],
                "sam3": {
                    "version": "3.1",
                    "variant": "object-multiplex",
                    "runtime": "onnxruntime-cpu",
                    "mask_threshold": 0.0,
                },
                "media": media.to_dict(),
                "images": images,
                "annotations": annotations,
                "categories": categories,
                "objects": self._objects(),
            }
            if media.kind == "video":
                document["videos"] = [
                    {
                        "id": 1,
                        "file_name": media.name,
                        "width": media.width,
                        "height": media.height,
                        "frame_count": media.frame_count,
                        "fps": media.fps,
                    }
                ]
            return document

    def save_artifacts(self, output_dir: Path) -> dict[str, object]:
        """Write a self-contained annotation bundle for CLI and review use."""
        with self._lock:
            media = self._require_media()
            destination = output_dir.resolve()
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "preview.mp4").unlink(missing_ok=True)
            frames_dir = destination / "frames"
            masks_dir = destination / "masks"
            overlays_dir = destination / "overlays"
            media_dir = destination / "media"
            shutil.rmtree(media_dir, ignore_errors=True)
            for generated_dir in (frames_dir, masks_dir, overlays_dir):
                shutil.rmtree(generated_dir, ignore_errors=True)
                generated_dir.mkdir(parents=True)

            document = self.export_document()
            images_value = document["images"]
            if not isinstance(images_value, list):
                raise RuntimeError("Annotation export contains invalid image records")
            images = cast(list[dict[str, object]], images_value)
            for image_record in images:
                if not isinstance(image_record, dict):
                    raise RuntimeError("Annotation export contains an invalid image record")
                frame_index = image_record.get("frame_index")
                if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                    raise RuntimeError("Annotation export contains an invalid frame index")
                image_record["file_name"] = f"frames/{frame_index:06d}.jpg"

            if media.kind == "video":
                media_dir.mkdir()
                bundled_name = _safe_filename(media.name)
                shutil.copyfile(self.upload_dir / media.name, media_dir / bundled_name)
                videos_value = document.get("videos")
                if not isinstance(videos_value, list) or len(videos_value) != 1:
                    raise RuntimeError("Annotation export contains invalid video metadata")
                video_record = videos_value[0]
                if not isinstance(video_record, dict):
                    raise RuntimeError("Annotation export contains invalid video metadata")
                typed_video_record = cast(dict[str, object], video_record)
                typed_video_record["file_name"] = f"media/{bundled_name}"

            results = self.engine.results
            for frame_index in range(media.frame_count):
                source = self.frame_path(frame_index)
                shutil.copyfile(source, frames_dir / f"{frame_index:06d}.jpg")
                with Image.open(source) as image:
                    annotations = tuple(results.get(frame_index, ()))
                    overlay = render_overlay(image, annotations)
                overlay.save(overlays_dir / f"{frame_index:06d}.jpg", quality=92)
                for annotation in annotations:
                    object_dir = masks_dir / f"object_{annotation.object_id:04d}"
                    object_dir.mkdir(exist_ok=True)
                    mask = decode_uncompressed_rle(annotation.segmentation)
                    Image.fromarray(mask.astype("uint8") * 255, mode="L").save(
                        object_dir / f"{frame_index:06d}.png"
                    )

            annotation_path = destination / "annotations.json"
            annotation_path.write_text(
                json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            preview_name = None
            if media.kind == "video" and media.frame_count > 1:
                preview_name = "preview.mp4"
                _write_preview_video(
                    overlays_dir,
                    destination / preview_name,
                    frame_count=media.frame_count,
                    fps=media.fps,
                    width=media.width,
                    height=media.height,
                )
            return {
                "annotation": annotation_path.name,
                "frames": media.frame_count,
                "objects": len(self.labels),
                "preview": preview_name,
            }

    def _record_dict(self, record: MaskAnnotation) -> dict[str, object]:
        item = record.to_dict()
        item["label"] = self.labels.get(record.object_id, f"object-{record.object_id}")
        return item

    def _result_frame_indices(self) -> tuple[int, ...]:
        indices = getattr(self.engine, "result_frame_indices", None)
        if indices is not None:
            return tuple(indices)
        return tuple(self.engine.results)

    def _objects(self) -> list[dict[str, object]]:
        return [
            {
                "id": object_id,
                "label": label,
                "prompts": list(self.prompts.get(object_id, [])),
            }
            for object_id, label in sorted(self.labels.items())
        ]

    def _frames(self) -> list[dict[str, object]]:
        return [
            {
                "frame_index": frame_index,
                "annotations": [self._record_dict(item) for item in records],
            }
            for frame_index, records in sorted(self.engine.results.items())
        ]

    def _require_media(self) -> MediaInfo:
        if self.media is None:
            raise RuntimeError("Load an image or video first")
        return self.media


def _required_int(
    payload: dict[str, object], key: str, *, minimum: int, maximum: int | None = None
) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if raw < minimum or (maximum is not None and raw > maximum):
        limit = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{key} must be {limit}")
    return raw


def _write_preview_video(
    overlay_dir: Path,
    destination: Path,
    *,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    """Encode rendered overlays with OpenCV's portable MP4V writer."""
    import cv2

    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create the annotation preview video")
    try:
        for frame_index in range(frame_count):
            frame = cv2.imread(str(overlay_dir / f"{frame_index:06d}.jpg"))
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError("Could not read a rendered annotation frame")
            writer.write(frame)
    finally:
        writer.release()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    """Request handler whose service/static root are supplied by the server."""

    server: AnnotationHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        try:
            self._validate_host()
            if path == "/api/health":
                self._send_json(self.server.service.health())
                return
            if path == "/api/project":
                self._validate_sensitive_read()
                self._send_json(self.server.service.project())
                return
            if path == "/api/export":
                self._validate_sensitive_read()
                with self.server.service.guard_project(self._project_header()):
                    body = _json_bytes(self.server.service.export_document())
                self._send_bytes(
                    body,
                    "application/json; charset=utf-8",
                    headers={
                        "Content-Disposition": 'attachment; filename="sam31-annotations.json"'
                    },
                )
                return
            if path.startswith("/api/frame/"):
                self._validate_sensitive_read()
                raw_index = path.removeprefix("/api/frame/").removesuffix(".jpg")
                try:
                    frame_index = int(raw_index)
                except ValueError as exc:
                    raise ValueError("frame index must be an integer") from exc
                with self.server.service.guard_project(self._project_header()):
                    body = self.server.service.frame_bytes(frame_index)
                self._send_bytes(body, "image/jpeg")
                return
            static = STATIC_FILES.get(path)
            if static is not None:
                filename, content_type = static
                self._send_bytes((self.server.static_root / filename).read_bytes(), content_type)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Resource not found")
        except Exception as exc:  # request boundary: convert to a stable response
            self._handle_exception(exc)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        target = urlsplit(self.path)
        try:
            if target.path.startswith("/api/"):
                self._validate_mutation_request()
            if target.path == "/api/media":
                self._require_content_type("application/octet-stream")
                query = parse_qs(target.query, keep_blank_values=True)
                filename = query.get("filename", [""])[0]
                max_frames = _normalise_max_frames(
                    query.get("max_frames", [None])[0], self.server.service.max_video_frames
                )
                with self.server.upload_lock:
                    with self.server.service.guard_upload(self._project_header()):
                        payload = self._read_body()
                        response = self.server.service.open_upload(filename, payload, max_frames)
                self._send_json(response, status=HTTPStatus.CREATED)
                return
            if target.path == "/api/segment":
                self._require_content_type("application/json")
                payload = self._read_json()
                with self.server.service.guard_project(self._project_header()):
                    response = self.server.service.segment(payload)
                self._send_json(response)
                return
            if target.path == "/api/propagate":
                self._require_content_type("application/json")
                self._read_json(allow_empty=True)
                with self.server.service.guard_project(self._project_header()):
                    response = self.server.service.propagate()
                self._send_json(response)
                return
            if target.path == "/api/reset":
                self._require_content_type("application/json")
                self._read_json(allow_empty=True)
                with self.server.service.guard_project(self._project_header()):
                    response = self.server.service.reset()
                self._send_json(response)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "Resource not found")
        except Exception as exc:  # request boundary: convert to a stable response
            self._handle_exception(exc)

    def _validate_mutation_request(self) -> None:
        request_host = self._validate_host()

        origin = self.headers.get("Origin")
        if origin is not None:
            try:
                parsed_origin = urlsplit(origin)
                origin_port = parsed_origin.port
            except ValueError as exc:
                raise ForbiddenRequestError("The Origin header is invalid") from exc
            origin_host = parsed_origin.hostname
            if (
                parsed_origin.scheme.lower() != "http"
                or origin_host is None
                or parsed_origin.username is not None
                or parsed_origin.password is not None
                or parsed_origin.path not in ("", "/")
                or parsed_origin.query
                or parsed_origin.fragment
                or origin_host.rstrip(".").lower() != request_host
                or (origin_port if origin_port is not None else 80) != int(self.server.server_port)
            ):
                raise ForbiddenRequestError("The Origin header must match this server")

        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site is not None and fetch_site.strip().lower() != "same-origin":
            raise ForbiddenRequestError("Cross-site mutation requests are not accepted")

    def _validate_sensitive_read(self) -> None:
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site is not None and fetch_site.strip().lower() not in {"none", "same-origin"}:
            raise ForbiddenRequestError("Cross-site project reads are not accepted")

    def _project_header(self) -> str | None:
        values = self.headers.get_all(PROJECT_HEADER, failobj=[])
        if len(values) > 1:
            raise ProjectGenerationError("The annotation project is missing or stale")
        return values[0] if values else None

    def _validate_host(self) -> str:
        host_headers = self.headers.get_all("Host", failobj=[])
        if len(host_headers) != 1:
            raise ForbiddenRequestError("Exactly one Host header is required")
        request_host, request_port = _parse_host_header(host_headers[0])
        if not _is_loopback_host(request_host):
            raise ForbiddenRequestError("The Host header must name a loopback host")
        server_port = int(self.server.server_port)
        effective_request_port = request_port if request_port is not None else 80
        if effective_request_port != server_port:
            raise ForbiddenRequestError("The Host header port does not match this server")
        return request_host

    def _require_content_type(self, expected: str) -> None:
        if self.headers.get_content_type().lower() != expected:
            raise UnsupportedMediaTypeError(f"Content-Type must be {expected}")

    def _read_body(self, *, max_bytes: int | None = None) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length is invalid") from exc
        limit = self.server.max_upload_bytes if max_bytes is None else max_bytes
        if length < 0 or length > limit:
            raise RequestTooLargeError(f"Request body exceeds the {limit} byte limit")
        return self.rfile.read(length)

    def _read_json(self, *, allow_empty: bool = False) -> dict[str, object]:
        body = self._read_body(max_bytes=self.server.max_json_bytes)
        if allow_empty and not body:
            return {}
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Request JSON must be an object")
        return value

    def _send_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(_json_bytes(payload), "application/json; charset=utf-8", status=status)

    def _send_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json({"ok": False, "code": code, "error": message}, status=status)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self' blob: data:; "
            "media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'none'",
        )
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _handle_exception(self, exc: Exception) -> None:
        if isinstance(exc, ProjectGenerationError):
            self._send_error(HTTPStatus.CONFLICT, "stale_project", str(exc))
        elif isinstance(exc, ForbiddenRequestError):
            self._send_error(HTTPStatus.FORBIDDEN, "forbidden_request", str(exc))
        elif isinstance(exc, UnsupportedMediaTypeError):
            self._send_error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                str(exc),
            )
        elif isinstance(exc, RequestTooLargeError):
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload_too_large", str(exc))
        elif isinstance(exc, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        elif isinstance(exc, FileNotFoundError):
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "missing_artifact",
                "A required media or model artifact is unavailable",
            )
        elif isinstance(exc, RuntimeError):
            message = str(exc)
            if not message.startswith("Load an image or video"):
                self.log_error("inference failed: %s", type(exc).__name__)
                message = "SAM 3.1 inference could not complete"
            self._send_error(HTTPStatus.CONFLICT, "invalid_state", message)
        else:
            self.log_error("request failed: %s", type(exc).__name__)
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "The annotation request failed",
            )

    def log_message(self, format: str, *args: object) -> None:
        # Never place upload query strings, which contain local display names,
        # in logs. Keep only the peer, method, URL path, and response status.
        path = urlsplit(self.path).path
        status = str(args[1]) if format == '"%s" %s %s' and len(args) >= 2 else "-"
        self.server.log(f"{self.client_address[0]} - {self.command} {path} {status}")


class RequestTooLargeError(ValueError):
    """Raised before an oversized request body is read into memory."""


class ForbiddenRequestError(ValueError):
    """Raised when browser request metadata does not describe this server."""


class UnsupportedMediaTypeError(ValueError):
    """Raised when a mutation request uses an unexpected content type."""


class AnnotationHTTPServer(ThreadingHTTPServer):
    """Threading server carrying immutable application configuration."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: AnnotationService,
        *,
        static_root: Path,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        max_json_bytes: int = MAX_JSON_BYTES,
        log: Any = print,
    ) -> None:
        _require_loopback_bind_host(address[0])
        self.service = service
        self.static_root = static_root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_json_bytes = max_json_bytes
        self.upload_lock = threading.Lock()
        self.log = log
        missing = [
            name for name, _ in STATIC_FILES.values() if not (self.static_root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Annotation frontend is incomplete: {sorted(set(missing))}")
        super().__init__(address, AnnotationRequestHandler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.service.close()


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: Path | None = None,
    static_root: Path | None = None,
    threads: int = 8,
    max_video_frames: int = MAX_VIDEO_FRAMES,
    engine: AnnotationEngine | None = None,
    log: Any = print,
) -> AnnotationHTTPServer:
    """Construct the local server without starting its event loop."""
    root = repo_root()
    service = AnnotationService(
        workspace or root / "outputs" / "annotation_workspace",
        engine=engine,
        threads=threads,
        max_video_frames=max_video_frames,
    )
    try:
        return AnnotationHTTPServer(
            (host, port),
            service,
            static_root=static_root or root / "web" / "annotation",
            log=log,
        )
    except Exception:
        service.close()
        raise


def server_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else str(host)
    return f"http://{quote(display_host, safe='.:[]')}:{port}"
