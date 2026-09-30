"""Run the local SAM 3.1 image/video annotation application."""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import cast

from sam3_onnx_equiv.annotation import PromptPoint
from sam3_onnx_equiv.annotation_server import (
    MAX_VIDEO_FRAMES,
    AnnotationService,
    create_server,
    server_url,
)
from sam3_onnx_equiv.path_config import repo_root


def _point(raw: str) -> tuple[int, PromptPoint]:
    fields = raw.split(":")
    if len(fields) not in (3, 4):
        raise argparse.ArgumentTypeError("point must be OBJECT_ID:X:Y[:LABEL]")
    try:
        object_id = int(fields[0])
        point = PromptPoint(
            x=float(fields[1]),
            y=float(fields[2]),
            label=int(fields[3]) if len(fields) == 4 else 1,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not 1 <= object_id <= 16:
        raise argparse.ArgumentTypeError("OBJECT_ID must be in 1..16")
    return object_id, point


def _label(raw: str) -> tuple[int, str]:
    try:
        object_id_raw, label = raw.split(":", 1)
        object_id = int(object_id_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("label must be OBJECT_ID:NAME") from exc
    label = label.strip()
    if not 1 <= object_id <= 16 or not label:
        raise argparse.ArgumentTypeError("label needs an object ID in 1..16 and a name")
    return object_id, label


def _serve(args: argparse.Namespace) -> None:
    server = create_server(
        host=args.host,
        port=args.port,
        workspace=args.workspace,
        threads=args.threads,
        max_video_frames=args.max_video_frames,
    )
    print(f"SAM 3.1 Annotator ready at {server_url(server)}", flush=True)
    print("Model and ONNX sessions load lazily when media is first opened.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _annotate(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise SystemExit(f"Input media does not exist: {args.input}")
    points_by_object: dict[int, list[PromptPoint]] = defaultdict(list)
    for object_id, point in args.point:
        points_by_object[object_id].append(point)
    labels = dict(args.label or [])

    with tempfile.TemporaryDirectory(prefix="sam31-annotation-") as workspace:
        service = AnnotationService(
            Path(workspace), threads=args.threads, max_video_frames=args.max_frames
        )
        try:
            opened = service.open_upload(args.input.name, args.input.read_bytes(), args.max_frames)
            media_value = opened["media"]
            if not isinstance(media_value, dict):
                raise SystemExit("Annotation service returned invalid media metadata")
            media = cast(dict[str, object], media_value)
            if media.get("kind") != args.kind:
                actual = media.get("kind", "unknown")
                raise SystemExit(f"Expected {args.kind} input, decoded {actual}")
            for object_id, points in points_by_object.items():
                service.segment(
                    {
                        "frame_index": 0,
                        "object_id": object_id,
                        "label": labels.get(object_id, f"object-{object_id}"),
                        "points": [point.to_dict() for point in points],
                    }
                )
            if args.kind == "video":
                service.propagate()
            summary = service.save_artifacts(args.output)
        finally:
            service.close()

    print(
        f"Saved {summary['frames']} annotated frame(s), {summary['objects']} object(s) "
        f"to {args.output.resolve()}",
        flush=True,
    )
    if summary["preview"]:
        print(f"Preview: {args.output.resolve() / str(summary['preview'])}", flush=True)


def _add_annotation_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], kind: str
) -> None:
    command = subparsers.add_parser(kind, help=f"annotate one {kind} from point prompts")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument(
        "--point",
        type=_point,
        action="append",
        required=True,
        help="OBJECT_ID:X:Y[:LABEL], coordinates are normalized; LABEL is 1 or 0",
    )
    command.add_argument(
        "--label", type=_label, action="append", help="OBJECT_ID:NAME (repeat as needed)"
    )
    command.add_argument("--threads", type=int, default=8)
    command.add_argument("--max-frames", type=int, default=1 if kind == "image" else 6)
    command.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "outputs" / "annotations" / kind,
    )
    command.set_defaults(func=_annotate, kind=kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the localhost annotation UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--threads", type=int, default=8)
    serve.add_argument("--max-video-frames", type=int, default=MAX_VIDEO_FRAMES)
    serve.add_argument(
        "--workspace",
        type=Path,
        default=repo_root() / "outputs" / "annotation_workspace",
    )
    serve.set_defaults(func=_serve)
    _add_annotation_parser(subparsers, "image")
    _add_annotation_parser(subparsers, "video")
    args = parser.parse_args()
    if hasattr(args, "port") and (args.port < 0 or args.port > 65535):
        parser.error("--port must be in 0..65535")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if hasattr(args, "max_video_frames") and args.max_video_frames < 1:
        parser.error("--max-video-frames must be positive")
    if hasattr(args, "max_frames") and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
