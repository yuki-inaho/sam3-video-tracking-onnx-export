"""HTTP and media-boundary tests for the local SAM 3.1 annotator."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from sam3_onnx_equiv.annotation import MAX_PROMPT_POINTS, MaskAnnotation, PromptPoint
from sam3_onnx_equiv.annotation_server import (
    MAX_DECODED_PIXELS,
    MAX_FRAME_PIXELS,
    MAX_JSON_BYTES,
    PROJECT_HEADER,
    PROJECT_MARKER,
    PROJECT_MARKER_CONTENT,
    AnnotationHTTPServer,
    AnnotationService,
    DecodedMediaTooLargeError,
    MediaInfo,
    WorkspaceInUseError,
    _decode_media,
    _normalise_max_frames,
    _safe_filename,
    _validate_decoded_pixels,
)


class FakeEngine:
    """Deterministic engine that exercises the server without loading weights."""

    def __init__(self) -> None:
        self.frame_count = 0
        self._results: dict[int, list[MaskAnnotation]] = {}
        self._has_propagated = False
        self.fail_next_open = False

    def open(self, frame_dir: Path) -> dict[str, object]:
        if self.fail_next_open:
            self.fail_next_open = False
            raise RuntimeError("synthetic model initialization failure")
        frames = sorted(frame_dir.glob("*.jpg"))
        with Image.open(frames[0]) as image:
            width, height = image.size
        self.frame_count = len(frames)
        self._results = {}
        self._has_propagated = False
        return {"frame_count": len(frames), "width": width, "height": height}

    def segment(
        self, frame_index: int, object_id: int, points: list[PromptPoint]
    ) -> list[MaskAnnotation]:
        assert any(point.label == 1 for point in points)
        if self._has_propagated:
            self._results = {}
            self._has_propagated = False
        record = MaskAnnotation(
            frame_index=frame_index,
            object_id=object_id,
            score=0.875,
            bbox=(1, 1, 2, 2),
            area=4,
            segmentation={"size": [3, 4], "counts": [4, 2, 1, 2, 3]},
        )
        existing = [
            item for item in self._results.get(frame_index, []) if item.object_id != object_id
        ]
        existing.append(record)
        self._results[frame_index] = existing
        return list(existing)

    def propagate(self) -> dict[int, list[MaskAnnotation]]:
        source = next(iter(self._results.values()))
        self._results = {
            frame_index: [
                MaskAnnotation(
                    frame_index=frame_index,
                    object_id=item.object_id,
                    score=item.score,
                    bbox=item.bbox,
                    area=item.area,
                    segmentation=item.segmentation,
                )
                for item in source
            ]
            for frame_index in range(self.frame_count)
        }
        self._has_propagated = True
        return {key: list(value) for key, value in self._results.items()}

    @property
    def results(self) -> MappingProxyType[int, tuple[MaskAnnotation, ...]]:
        return MappingProxyType({key: tuple(value) for key, value in self._results.items()})

    @property
    def result_frame_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._results))


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (30, 60, 90)).save(output, format="PNG")
    return output.getvalue()


def _static_root(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<!doctype html><title>test</title>")
    (path / "app.js").write_text("'use strict';")
    (path / "styles.css").write_text("body{}")
    return path


def _request(
    base: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    request = Request(base + path, data=body, method=method)
    if body is not None:
        selected_type = content_type or (
            "application/octet-stream" if path.startswith("/api/media?") else "application/json"
        )
        request.add_header("Content-Type", selected_type)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, response.read(), dict(response.headers.items())
    except HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers.items())


def test_server_upload_segment_propagate_and_export_are_portable(tmp_path: Path) -> None:
    logs: list[str] = []
    engine = FakeEngine()
    service = AnnotationService(tmp_path / "work", engine=engine, max_video_frames=6)
    server = AnnotationHTTPServer(
        ("127.0.0.1", 0),
        service,
        static_root=_static_root(tmp_path / "static"),
        log=logs.append,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body, _ = _request(
            base,
            "/api/media?filename=..%2Fprivate.png&max_frames=5",
            method="POST",
            body=_image_bytes(),
        )
        uploaded = json.loads(body)
        assert status == 201
        project_id = uploaded["project_id"]
        assert isinstance(project_id, str) and project_id
        assert uploaded["media"] == {
            "name": "private.png",
            "kind": "image",
            "width": 4,
            "height": 3,
            "frame_count": 1,
            "fps": 0.0,
        }

        status, body, headers = _request(base, "/api/health")
        assert status == 200
        assert json.loads(body)["max_video_frames"] == 6
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert "base-uri 'none'" in headers["Content-Security-Policy"]
        assert "form-action 'none'" in headers["Content-Security-Policy"]
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"

        segment_payload = json.dumps(
            {
                "frame_index": 0,
                "object_id": 7,
                "label": "parcel",
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            }
        ).encode()
        status, body, _ = _request(
            base,
            "/api/segment",
            method="POST",
            body=segment_payload,
            headers={
                "Origin": base,
                PROJECT_HEADER: project_id,
                "Sec-Fetch-Site": "same-origin",
            },
        )
        segmented = json.loads(body)
        assert status == 200
        assert segmented["invalidated_frames"] == []
        assert segmented["propagation_required"] is False
        assert segmented["annotations"][0]["object_id"] == 7
        assert segmented["annotations"][0]["label"] == "parcel"

        status, body, _ = _request(
            base,
            "/api/propagate",
            method="POST",
            body=b"{}",
            headers={PROJECT_HEADER: project_id},
        )
        assert status == 200
        assert len(json.loads(body)["frames"]) == 1

        status, body, headers = _request(base, "/api/export", headers={PROJECT_HEADER: project_id})
        exported = json.loads(body)
        assert status == 200
        assert headers["Content-Disposition"] == 'attachment; filename="sam31-annotations.json"'
        assert exported["sam3"]["version"] == "3.1"
        assert exported["annotations"][0]["track_id"] == 7
        assert exported["categories"] == [{"id": 1, "name": "parcel", "supercategory": ""}]
        assert str(tmp_path) not in body.decode()

        status, frame, headers = _request(
            base, "/api/frame/0.jpg", headers={PROJECT_HEADER: project_id}
        )
        assert status == 200
        assert headers["Content-Type"] == "image/jpeg"
        assert frame.startswith(b"\xff\xd8")
        assert all("private.png" not in message for message in logs)
        assert any("POST /api/media 201" in message for message in logs)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_request_validation_and_safe_names(tmp_path: Path) -> None:
    assert _safe_filename("../../folder/photo.jpg") == "photo.jpg"
    assert _safe_filename(r"..\folder\photo.jpg") == "photo.jpg"
    assert _normalise_max_frames("100", 12) == 12
    unicode_name = _safe_filename("😀日本語" * 100 + ".png")
    assert unicode_name
    assert len(unicode_name.encode("utf-8")) <= 240

    service = AnnotationService(tmp_path / "work", engine=FakeEngine())
    try:
        service.segment({"frame_index": 0, "object_id": 1, "points": []})
    except RuntimeError as exc:
        assert "Load" in str(exc)
    else:
        raise AssertionError("segmenting without media must fail")

    service.open_upload("image.png", _image_bytes(), 1)
    with pytest.raises(ValueError, match=str(MAX_PROMPT_POINTS)):
        service.segment(
            {
                "frame_index": 0,
                "object_id": 1,
                "points": [{"x": 0.5, "y": 0.5}] * (MAX_PROMPT_POINTS + 1),
            }
        )


def test_media_replacement_is_transactional_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    legacy_frames = workspace / "frames"
    legacy_uploads = workspace / "uploads"
    legacy_frames.mkdir(parents=True)
    legacy_uploads.mkdir()
    (legacy_frames / "keep.txt").write_text("legacy frame data")
    (legacy_uploads / "keep.txt").write_text("legacy upload data")
    stale = workspace / "project-stale"
    stale.mkdir()
    (stale / "orphan").write_text("stale")
    (stale / PROJECT_MARKER).write_text(PROJECT_MARKER_CONTENT)
    unrelated = workspace / "project-user-data"
    unrelated.mkdir()
    (unrelated / "keep").write_text("user data")
    engine = FakeEngine()
    service = AnnotationService(workspace, engine=engine)
    assert not stale.exists()
    assert (unrelated / "keep").read_text() == "user data"

    service.open_upload("first.png", _image_bytes(), 5)
    first_project_id = service.project_id
    assert first_project_id is not None
    assert (legacy_frames / "keep.txt").read_text() == "legacy frame data"
    assert (legacy_uploads / "keep.txt").read_text() == "legacy upload data"
    service.segment(
        {
            "frame_index": 0,
            "object_id": 3,
            "label": "kept",
            "points": [{"x": 0.25, "y": 0.5, "label": 1}],
        }
    )
    old_project = service.frame_dir.parent
    old_frame = service.frame_path(0).read_bytes()
    old_project_state = service.project()

    engine.fail_next_open = True
    with pytest.raises(RuntimeError, match="synthetic"):
        service.open_upload("bad-replacement.png", _image_bytes(), 5)

    assert service.frame_dir.parent == old_project
    assert service.frame_path(0).read_bytes() == old_frame
    assert service.project() == old_project_state
    assert service.project_id == first_project_id
    assert set(workspace.glob("project-*")) == {old_project, unrelated}

    def fail_decode(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic decode failure")

    with monkeypatch.context() as patch:
        patch.setattr("sam3_onnx_equiv.annotation_server._decode_media", fail_decode)
        with pytest.raises(ValueError, match="synthetic"):
            service.open_upload("bad-decode.png", _image_bytes(), 5)

    assert service.frame_dir.parent == old_project
    assert service.frame_path(0).read_bytes() == old_frame
    assert service.project() == old_project_state
    assert service.project_id == first_project_id
    assert set(workspace.glob("project-*")) == {old_project, unrelated}

    service.open_upload("replacement.png", _image_bytes(), 5)
    assert service.media is not None and service.media.name == "replacement.png"
    assert service.project_id is not None and service.project_id != first_project_id
    assert service.frame_dir.parent != old_project
    assert not old_project.exists()
    assert service.labels == {}
    assert service.prompts == {}


def test_workspace_lock_prevents_active_project_cleanup(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    first = AnnotationService(workspace, engine=FakeEngine())
    first.open_upload("active.png", _image_bytes(), 1)
    active_frame = first.frame_path(0)
    active_bytes = active_frame.read_bytes()

    with pytest.raises(WorkspaceInUseError, match="already in use"):
        AnnotationService(workspace, engine=FakeEngine())

    assert active_frame.read_bytes() == active_bytes
    media = first.project()["media"]
    assert isinstance(media, dict)
    assert cast(dict[str, object], media)["name"] == "active.png"
    first.close()

    second = AnnotationService(workspace, engine=FakeEngine())
    try:
        assert not active_frame.parent.parent.exists()
    finally:
        second.close()


def test_save_artifacts_removes_stale_preview_for_image(tmp_path: Path) -> None:
    service = AnnotationService(tmp_path / "work", engine=FakeEngine())
    service.open_upload("image.png", _image_bytes(), 1)
    service.segment(
        {
            "frame_index": 0,
            "object_id": 1,
            "points": [{"x": 0.5, "y": 0.5, "label": 1}],
        }
    )
    output = tmp_path / "output"
    output.mkdir()
    preview = output / "preview.mp4"
    preview.write_bytes(b"stale")
    stale_media = output / "media"
    stale_media.mkdir()
    (stale_media / "old-video.mp4").write_bytes(b"stale")

    summary = service.save_artifacts(output)

    assert summary["preview"] is None
    assert not preview.exists()
    assert not stale_media.exists()


def test_save_artifacts_bundles_video_source_without_changing_browser_export(
    tmp_path: Path,
) -> None:
    payload = _image_bytes()
    service = AnnotationService(tmp_path / "work", engine=FakeEngine())
    service.open_upload("clip.mp4", payload, 1)
    service.media = MediaInfo("clip.mp4", "video", 4, 3, 1, 24.0)
    service.segment(
        {
            "frame_index": 0,
            "object_id": 1,
            "points": [{"x": 0.5, "y": 0.5, "label": 1}],
        }
    )
    assert service.export_document()["videos"] == [
        {
            "id": 1,
            "file_name": "clip.mp4",
            "width": 4,
            "height": 3,
            "frame_count": 1,
            "fps": 24.0,
        }
    ]

    output = tmp_path / "output"
    service.save_artifacts(output)

    assert (output / "media" / "clip.mp4").read_bytes() == payload
    bundled = json.loads((output / "annotations.json").read_text())
    assert bundled["videos"][0]["file_name"] == "media/clip.mp4"


def test_decoded_pixel_limits_allow_ordinary_4k() -> None:
    _validate_decoded_pixels(4096, 2160, 60)
    with pytest.raises(ValueError, match=str(MAX_FRAME_PIXELS)):
        _validate_decoded_pixels(5000, 5000, 1)
    with pytest.raises(ValueError, match=str(MAX_DECODED_PIXELS)):
        _validate_decoded_pixels(4096, 2160, 70)


def test_oversized_image_does_not_fall_back_to_video_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_called = False

    def reject_image(*args: object, **kwargs: object) -> object:
        raise DecodedMediaTooLargeError("too many decoded pixels")

    def record_video(*args: object, **kwargs: object) -> object:
        nonlocal video_called
        video_called = True
        raise AssertionError("video decoder must not receive an oversized decoded image")

    monkeypatch.setattr("sam3_onnx_equiv.annotation_server._write_image_frames", reject_image)
    monkeypatch.setattr("sam3_onnx_equiv.annotation_server._write_video_frames", record_video)
    source = tmp_path / "oversized.png"
    source.write_bytes(b"synthetic")
    frames = tmp_path / "frames"
    frames.mkdir()

    with pytest.raises(DecodedMediaTooLargeError, match="too many"):
        _decode_media(source, frames, max_frames=1)
    assert video_called is False


def test_prompts_keep_other_frames_and_correction_reports_invalidations(tmp_path: Path) -> None:
    engine = FakeEngine()
    service = AnnotationService(tmp_path / "work", engine=engine)
    service.open_upload("clip.png", _image_bytes(), 3)
    assert service.media is not None
    service.media = MediaInfo("clip.mp4", "video", 4, 3, 3, 24.0)
    engine.frame_count = 3

    service.segment(
        {
            "frame_index": 0,
            "object_id": 1,
            "label": "parcel",
            "points": [{"x": 0.1, "y": 0.2, "label": 1}],
        }
    )
    service.propagate()
    corrected = service.segment(
        {
            "frame_index": 1,
            "object_id": 1,
            "label": "parcel",
            "points": [{"x": 0.3, "y": 0.4, "label": 1}],
        }
    )
    assert corrected["invalidated_frames"] == [0, 2]
    assert corrected["propagation_required"] is True
    assert service.project()["propagation_required"] is True
    assert [prompt["frame_index"] for prompt in service.prompts[1]] == [0, 1]

    corrected_again = service.segment(
        {
            "frame_index": 1,
            "object_id": 1,
            "label": "parcel",
            "points": [
                {"x": 0.6, "y": 0.7, "label": 1},
                {"x": 0.8, "y": 0.9, "label": 0},
            ],
        }
    )
    assert corrected_again["invalidated_frames"] == []
    assert corrected_again["propagation_required"] is True
    prompts = service.prompts[1]
    assert [prompt["frame_index"] for prompt in prompts] == [0, 1, 1]
    assert [(prompt["x"], prompt["label"]) for prompt in prompts] == [
        (0.1, 1),
        (0.6, 1),
        (0.8, 0),
    ]
    propagated = service.propagate()
    assert propagated["propagation_required"] is False
    assert service.project()["propagation_required"] is False


def test_server_rejects_non_loopback_bind_and_hostile_request_metadata(tmp_path: Path) -> None:
    static_root = _static_root(tmp_path / "static")
    service = AnnotationService(tmp_path / "work", engine=FakeEngine())
    with pytest.raises(ValueError, match="loopback"):
        AnnotationHTTPServer(("0.0.0.0", 0), service, static_root=static_root)

    server = AnnotationHTTPServer(
        ("127.0.0.1", 0), service, static_root=static_root, log=lambda _: None
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body, _ = _request(
            base,
            "/api/project",
            headers={"Host": f"attacker.example:{server.server_port}"},
        )
        assert status == 403
        assert json.loads(body)["code"] == "forbidden_request"

        status, body, _ = _request(
            base,
            "/api/frame/0.jpg",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert status == 403
        assert json.loads(body)["code"] == "forbidden_request"

        status, body, _ = _request(
            base,
            "/api/media?filename=image.png",
            method="POST",
            body=_image_bytes(),
            headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
        )
        assert status == 403
        assert json.loads(body)["code"] == "forbidden_request"

        status, body, _ = _request(
            base,
            "/api/media?filename=image.png",
            method="POST",
            body=_image_bytes(),
            content_type="image/png",
        )
        assert status == 415
        assert json.loads(body)["code"] == "unsupported_media_type"

        status, body, _ = _request(
            base,
            "/api/reset",
            method="POST",
            body=b"{}",
            content_type="text/plain",
        )
        assert status == 415
        assert json.loads(body)["code"] == "unsupported_media_type"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_project_generation_rejects_stale_tabs(tmp_path: Path) -> None:
    service = AnnotationService(tmp_path / "work", engine=FakeEngine())
    server = AnnotationHTTPServer(
        ("127.0.0.1", 0),
        service,
        static_root=_static_root(tmp_path / "static"),
        log=lambda _: None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body, _ = _request(
            base,
            "/api/media?filename=tab-a.png",
            method="POST",
            body=_image_bytes(),
        )
        assert status == 201
        tab_a = json.loads(body)["project_id"]

        status, body, _ = _request(base, "/api/project")
        assert status == 200
        assert json.loads(body)["project_id"] == tab_a

        status, body, _ = _request(
            base,
            "/api/media?filename=tab-b.png",
            method="POST",
            body=_image_bytes(),
            headers={PROJECT_HEADER: tab_a},
        )
        assert status == 201
        tab_b = json.loads(body)["project_id"]
        assert tab_b != tab_a

        segment_body = json.dumps(
            {
                "frame_index": 0,
                "object_id": 1,
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            }
        ).encode()
        stale_requests = [
            ("/api/segment", "POST", segment_body),
            ("/api/reset", "POST", b"{}"),
            ("/api/frame/0.jpg", "GET", None),
            ("/api/export", "GET", None),
            ("/api/media?filename=stale.png", "POST", _image_bytes()),
        ]
        for path, method, request_body in stale_requests:
            status, body, _ = _request(
                base,
                path,
                method=method,
                body=request_body,
                headers={PROJECT_HEADER: tab_a},
            )
            assert status == 409
            assert json.loads(body) == {
                "ok": False,
                "code": "stale_project",
                "error": "The annotation project is missing or stale",
            }

        status, body, _ = _request(base, "/api/export")
        assert status == 409
        assert json.loads(body)["code"] == "stale_project"

        status, body, _ = _request(
            base,
            "/api/segment",
            method="POST",
            body=segment_body,
            headers={PROJECT_HEADER: tab_b},
        )
        assert status == 200
        assert json.loads(body)["annotations"][0]["object_id"] == 1
        status, frame, _ = _request(
            base,
            "/api/frame/0.jpg",
            headers={PROJECT_HEADER: tab_b},
        )
        assert status == 200
        assert frame.startswith(b"\xff\xd8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_json_body_cap_and_single_inflight_upload(tmp_path: Path) -> None:
    service = AnnotationService(tmp_path / "work", engine=FakeEngine())
    server = AnnotationHTTPServer(
        ("127.0.0.1", 0),
        service,
        static_root=_static_root(tmp_path / "static"),
        max_json_bytes=64,
        log=lambda _: None,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    request_done = threading.Event()
    result: list[tuple[int, bytes, dict[str, str]]] = []
    server.upload_lock.acquire()
    try:

        def upload() -> None:
            result.append(
                _request(
                    base,
                    "/api/media?filename=image.png",
                    method="POST",
                    body=_image_bytes(),
                )
            )
            request_done.set()

        upload_thread = threading.Thread(target=upload, daemon=True)
        upload_thread.start()
        assert not request_done.wait(timeout=0.2)
        server.upload_lock.release()
        assert request_done.wait(timeout=5)
        upload_thread.join(timeout=5)
        assert result[0][0] == 201
        project_id = json.loads(result[0][1])["project_id"]

        oversized_json = json.dumps({"padding": "x" * (MAX_JSON_BYTES + 1)}).encode()
        status, body, _ = _request(
            base,
            "/api/segment",
            method="POST",
            body=oversized_json,
            headers={PROJECT_HEADER: project_id},
        )
        assert status == 413
        assert json.loads(body)["code"] == "upload_too_large"
    finally:
        if server.upload_lock.locked():
            server.upload_lock.release()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
