"""Track point-prompted objects in a video folder with SAM 3.1 ONNX on CPU.

Example:
  uv run python tools/run_sam31_video.py --video-dir frames \
      --point 1:0.25:0.35 --point 2:0.70:0.65

Point coordinates are fractions of frame width and height. Images must be
named in frame order, such as 000000.jpg, 000001.jpg, and so on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sam3_onnx_equiv.path_config import repo_root
from sam3_onnx_equiv.sam31_model import build_sam31_tracker
from sam3_onnx_equiv.sam31_onnx_video import Sam31OnnxSessions, configure_sam31_onnx_window


def _point(raw: str) -> tuple[int, float, float]:
    try:
        object_id_raw, x_raw, y_raw = raw.split(":")
        object_id, x, y = int(object_id_raw), float(x_raw), float(y_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Point must be ID:X:Y, e.g. 1:0.25:0.35") from exc
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise argparse.ArgumentTypeError("Point coordinates must be normalized to [0, 1]")
    return object_id, x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--point", type=_point, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "outputs/reference/sam31_onnx_video.npz",
    )
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not args.video_dir.is_dir():
        parser.error(f"Video directory does not exist: {args.video_dir}")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    object_ids = [item[0] for item in args.point]
    if len(object_ids) != len(set(object_ids)):
        parser.error("Object IDs must be unique")

    model = build_sam31_tracker(use_rope_real=True)
    configure_sam31_onnx_window(model)
    sessions = Sam31OnnxSessions()
    sessions.install(model)

    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    with torch.inference_mode():
        state = VideoTrackingMultiplexDemo.init_state(
            model,
            video_path=str(args.video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        for object_id, x, y in args.point:
            model.add_new_points(
                state,
                0,
                object_id,
                torch.tensor([[x, y]], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int32),
                clear_old_points=True,
            )
        model.propagate_in_video_preflight(state, run_mem_encoder=True)
        frame_limit = min(args.max_frames or state["num_frames"], state["num_frames"])
        frames = []
        masks = []
        scores = []
        for frame, ids, _, video_masks, object_scores in model.propagate_in_video(
            state, 0, frame_limit - 1, False, tqdm_disable=True
        ):
            if list(ids) != object_ids:
                raise RuntimeError(f"Object IDs changed in frame {frame}: {ids}")
            frames.append(frame)
            masks.append(video_masks.detach().cpu().numpy())
            scores.append(object_scores.detach().cpu().numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frame_indices=np.asarray(frames),
        object_ids=np.asarray(object_ids),
        mask_logits=np.stack(masks),
        object_scores=np.stack(scores),
    )
    print(f"Saved {len(frames)} frames for {len(object_ids)} objects to {args.output}")
    print(f"ONNX module calls: {dict(sessions.calls)}")


if __name__ == "__main__":
    main()
