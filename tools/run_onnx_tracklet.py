"""Run one real-data tracklet with the ONNX video orchestrator.

This is a development driver for the small tracklet dataset under ``temp``.  It
requires CUDAExecutionProvider, runs memory attention with a fixed rounded K
(default: K=32), and records per-frame scores, mask bbox IoU against GT boxes,
runtime, and lightweight overlays.

Use the venv interpreter directly after installing ``onnxruntime-gpu``; ``uv run``
may resync the CPU onnxruntime wheel from the lock file.

Example:
    .venv/bin/python tools/run_onnx_tracklet.py \
      --dataset /path/to/temp/sam3_dev_tracklet --num-frames 16 --fixed-k 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ONNX_DIR = REPO / "outputs" / "onnx"
CONSTANTS_DIR = REPO / "outputs" / "reference" / "constants"


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _iou_xyxy(a: list[float] | list[int], b: list[float] | list[int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _fixed_k_obj_tokens(
    obj_tokens: np.ndarray,
    obj_pos: np.ndarray,
    fixed_k: int,
    tokens_per_ptr: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return exactly ``fixed_k`` obj_ptr tokens using pointer-group ring semantics.

    The dynamic memory attention graph bakes ``num_k_exclude_rope=fixed_k``.  The
    obj_ptr tail must therefore be exactly fixed_k tokens.  Zero padding is not
    neutral after attention projections, so warmup repeats available pointer
    groups.  When too many groups exist, keep the conditioning pointer group and
    the latest non-conditioning groups from the existing official ordering.
    """
    if fixed_k % tokens_per_ptr != 0:
        raise ValueError(f"fixed_k must be a multiple of {tokens_per_ptr}, got {fixed_k}")
    if obj_tokens.shape != obj_pos.shape:
        raise ValueError(f"obj token/pos shape mismatch: {obj_tokens.shape} vs {obj_pos.shape}")
    if obj_tokens.shape[0] % tokens_per_ptr != 0:
        raise ValueError(f"obj token count is not grouped by {tokens_per_ptr}: {obj_tokens.shape}")

    actual_k = int(obj_tokens.shape[0])
    n_groups = actual_k // tokens_per_ptr
    target_groups = fixed_k // tokens_per_ptr
    tokens_g = obj_tokens.reshape(n_groups, tokens_per_ptr, *obj_tokens.shape[1:])
    pos_g = obj_pos.reshape(n_groups, tokens_per_ptr, *obj_pos.shape[1:])

    if n_groups >= target_groups:
        indices = np.arange(target_groups, dtype=np.int64)
        policy = "trim_keep_cond_and_latest"
    else:
        indices = np.arange(target_groups, dtype=np.int64) % n_groups
        policy = "ring_repeat_existing"

    fixed_tokens = tokens_g[indices].reshape(fixed_k, *obj_tokens.shape[1:]).astype(np.float32)
    fixed_pos = pos_g[indices].reshape(fixed_k, *obj_pos.shape[1:]).astype(np.float32)
    stats = {
        "actual_k_minibatch": actual_k,
        "fixed_k": fixed_k,
        "actual_ptr_groups": n_groups,
        "fixed_ptr_groups": target_groups,
        "policy": policy,
        "group_indices": indices.tolist(),
    }
    return fixed_tokens, fixed_pos, stats


class FixedKVideoOrchestrator:
    """Thin wrapper around ``VideoOrchestrator`` that forces obj_ptr K to fixed_k."""

    def __init__(self, onnx_dir: Path, constants_dir: Path, providers: list[Any], fixed_k: int) -> None:
        import onnxruntime as ort
        from sam3_onnx_equiv.video_orchestrator import VideoOrchestrator

        self.fixed_k = fixed_k
        self._inner = VideoOrchestrator(onnx_dir, constants_dir, providers)
        fixed_graph = onnx_dir / f"memory_attention_dynamic_k{fixed_k}.onnx"
        if not fixed_graph.exists():
            raise FileNotFoundError(f"fixed-K memory attention graph not found: {fixed_graph}")
        self._inner._mem_attn[fixed_k] = ort.InferenceSession(str(fixed_graph), providers=providers)
        self._assert_cuda_sessions()
        self.fixed_k_events: list[dict[str, Any]] = []

    def _assert_cuda_sessions(self) -> None:
        sessions = {
            "image_encoder_tracker": self._inner._image_enc,
            "decode_head": self._inner._decode,
            "memory_encoder": self._inner._mem_enc,
            f"memory_attention_dynamic_k{self.fixed_k}": self._inner._mem_attn[self.fixed_k],
        }
        non_cuda = {
            name: sess.get_providers()
            for name, sess in sessions.items()
            if "CUDAExecutionProvider" not in sess.get_providers()
        }
        if non_cuda:
            raise RuntimeError(
                "CUDAExecutionProvider is required but some sessions did not use it: "
                f"{non_cuda}. If CUDA libraries are inside the venv, run with "
                "LD_LIBRARY_PATH including .venv/lib/python*/site-packages/nvidia/*/lib."
            )

    def run_clip(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        original = self._inner._conditioned_features

        def fixed_conditioned_features(
            frame_idx: int,
            num_frames: int,
            fpn2_seq: np.ndarray,
            pos2_seq: np.ndarray,
            bank: Any,
            use_mem_this_frame: bool,
        ) -> np.ndarray:
            if not use_mem_this_frame:
                return original(frame_idx, num_frames, fpn2_seq, pos2_seq, bank, use_mem_this_frame)

            from sam3_onnx_equiv.video_orchestrator import (
                D_MODEL,
                FEAT_H,
                FEAT_W,
                OBJ_PTR_TOKENS_PER_FRAME,
                _collect_maskmem,
                _collect_obj_ptrs,
            )

            maskmem_feats, maskmem_poses = _collect_maskmem(
                frame_idx, bank, self._inner._constants.maskmem_tpos_enc
            )
            if not maskmem_feats:
                raise ValueError(f"frame {frame_idx}: memory path entered without maskmem")

            obj_tokens, obj_pos, actual_k = _collect_obj_ptrs(
                frame_idx, num_frames, bank, self._inner._constants
            )
            fixed_tokens, fixed_pos, event = _fixed_k_obj_tokens(
                obj_tokens, obj_pos, self.fixed_k, OBJ_PTR_TOKENS_PER_FRAME
            )
            event["frame_idx"] = frame_idx
            event["actual_k"] = actual_k
            self.fixed_k_events.append(event)

            maskmem = np.concatenate(maskmem_feats, axis=0)
            maskmem_pos = np.concatenate(maskmem_poses, axis=0)
            prompt = np.concatenate([maskmem, fixed_tokens], axis=0).astype(np.float32)
            prompt_pos = np.concatenate([maskmem_pos, fixed_pos], axis=0).astype(np.float32)
            mem_out = self._inner._mem_attn[self.fixed_k].run(
                None,
                {
                    "src": fpn2_seq.astype(np.float32),
                    "src_pos": pos2_seq.astype(np.float32),
                    "prompt": prompt,
                    "prompt_pos": prompt_pos,
                },
            )
            memory_seq = mem_out[0]
            return memory_seq.transpose(1, 2, 0).reshape(1, D_MODEL, FEAT_H, FEAT_W)

        self._inner._conditioned_features = fixed_conditioned_features
        try:
            return self._inner.run_clip(*args, **kwargs)
        finally:
            self._inner._conditioned_features = original


def _cuda_providers(gpu_mem_limit_mb: int) -> list[Any]:
    return [
        (
            "CUDAExecutionProvider",
            {
                "gpu_mem_limit": str(gpu_mem_limit_mb * 1024 * 1024),
                "arena_extend_strategy": "kSameAsRequested",
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]


def _write_overlay(
    out_path: Path,
    frame: Image.Image,
    mask: np.ndarray,
    pred_box: list[int] | None,
    gt_box_xywh: list[float] | None,
    seed_point: list[float] | None,
    label: str,
) -> None:
    base = frame.copy()
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    ys, xs = np.where(mask)
    if xs.size:
        for yy, xx in zip(ys[::7], xs[::7]):
            od.point((int(xx), int(yy)), fill=(255, 40, 40, 160))
    comp = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(comp)
    if pred_box:
        d.rectangle(pred_box, outline=(255, 230, 0), width=2)
    if gt_box_xywh:
        x, y, w, h = gt_box_xywh
        d.rectangle([x, y, x + w, y + h], outline=(0, 220, 0), width=2)
    if seed_point:
        cx, cy = seed_point
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(0, 160, 255), width=3)
    d.text((5, 5), label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.save(out_path, quality=88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--num-frames", type=int, default=127)
    parser.add_argument("--fixed-k", type=int, default=32, choices=[32, 64])
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=8192)
    parser.add_argument("--out-subdir", default=None)
    parser.add_argument("--overlay-stride", type=int, default=3)
    parser.add_argument("--no-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDAExecutionProvider is required. Install onnxruntime-gpu and run with "
            ".venv/bin/python instead of uv run if uv resyncs the CPU wheel."
        )

    ds = args.dataset
    prompt = json.loads((ds / "prompt.json").read_text())
    meta = json.loads((ds / "meta.json").read_text())
    gt = json.loads((ds / "tracklet_boxes.json").read_text())
    gt_by_clip = {int(b["clip_idx"]): b["bbox_xywh_800x600"] for b in gt["boxes"]}
    width, height = meta["source_view_size"]

    out_subdir = args.out_subdir or f"onnx_infer_gpu_k{args.fixed_k}"
    out = ds / out_subdir
    out.mkdir(parents=True, exist_ok=True)

    providers = _cuda_providers(args.gpu_mem_limit_mb)
    print(f"[init] providers={providers}", flush=True)
    print(f"[init] available_providers={ort.get_available_providers()}", flush=True)
    t_init0 = time.perf_counter()
    orch = FixedKVideoOrchestrator(ONNX_DIR, CONSTANTS_DIR, providers, fixed_k=args.fixed_k)
    init_seconds = time.perf_counter() - t_init0
    print(f"[init] loaded sessions in {init_seconds:.1f}s", flush=True)

    n = min(args.num_frames, int(meta["clip_len"]))
    t_load0 = time.perf_counter()
    frames = [Image.open(ds / "frames" / f"{i:06d}.jpg").convert("RGB") for i in range(n)]
    load_seconds = time.perf_counter() - t_load0
    print(f"[load] {n} frames in {load_seconds:.2f}s", flush=True)

    coords = np.array(prompt["frame0_point_coords_norm"], dtype=np.float32)
    labels = np.array(prompt["frame0_point_labels"], dtype=np.int32)

    t_run0 = time.perf_counter()
    res = orch.run_clip(frames, coords, labels, use_memory=True)
    run_seconds = time.perf_counter() - t_run0
    print(
        f"[run] {n} frames in {run_seconds:.1f}s ({run_seconds / n:.3f}s/frame), "
        f"mem_attn_invocations={res['memory_attention_invoke_count']}",
        flush=True,
    )

    t_post0 = time.perf_counter()
    per_frame: list[dict[str, Any]] = []
    ious: list[float] = []
    for i in range(n):
        mask288 = res["masks"][i]
        score = float(res["scores"][i])
        resized = Image.fromarray((mask288.astype(np.uint8) * 255)).resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 127
        pred_box = _mask_bbox(mask)
        rec: dict[str, Any] = {
            "clip_idx": i,
            "obj_score": round(score, 4),
            "mask_px_288": int(mask288.sum()),
            "pred_bbox_xyxy_800x600": pred_box,
        }
        gt_box = gt_by_clip.get(i)
        if gt_box and pred_box is not None:
            x, y, w, h = gt_box
            gt_xyxy = [x, y, x + w, y + h]
            iou = _iou_xyxy(pred_box, gt_xyxy)
            rec["gt_bbox_xyxy_800x600"] = [round(float(v), 1) for v in gt_xyxy]
            rec["iou_vs_gt"] = round(iou, 4)
            ious.append(iou)
        per_frame.append(rec)

        should_overlay = (
            not args.no_overlays
            and (i % args.overlay_stride == 0 or i in gt_by_clip or i == 0 or i == n - 1)
        )
        if should_overlay:
            _write_overlay(
                out / "overlays" / f"{i:06d}.jpg",
                frames[i],
                mask,
                pred_box,
                gt_box,
                prompt["seed_point_abs_800x600"] if i == 0 else None,
                f"f{i} score={score:.2f} K={args.fixed_k}",
            )
    post_seconds = time.perf_counter() - t_post0

    fixed_k_summary: dict[str, Any] = {
        "frames_with_fixed_k": len(orch.fixed_k_events),
        "policies": sorted({e["policy"] for e in orch.fixed_k_events}),
        "actual_k_min": min((e["actual_k"] for e in orch.fixed_k_events), default=None),
        "actual_k_max": max((e["actual_k"] for e in orch.fixed_k_events), default=None),
    }
    scores = [float(s) for s in res["scores"]]
    summary = {
        "dataset_label": meta.get("dataset", "redacted-local-tracklet"),
        "num_frames_run": n,
        "fixed_k": args.fixed_k,
        "cuda_required": True,
        "gpu_mem_limit_mb": args.gpu_mem_limit_mb,
        "init_seconds": round(init_seconds, 3),
        "load_seconds": round(load_seconds, 3),
        "run_seconds": round(run_seconds, 3),
        "seconds_per_frame": round(run_seconds / n, 4),
        "post_seconds": round(post_seconds, 3),
        "mem_attn_invocations": int(res["memory_attention_invoke_count"]),
        "obj_score_min": round(float(np.min(scores)), 4),
        "obj_score_mean": round(float(np.mean(scores)), 4),
        "obj_score_max": round(float(np.max(scores)), 4),
        "frames_object_present_score_gt_0": int(sum(s > 0 for s in scores)),
        "gt_frames_evaluated": len(ious),
        "iou_mean_vs_gt": round(float(np.mean(ious)), 4) if ious else None,
        "iou_median_vs_gt": round(float(np.median(ious)), 4) if ious else None,
        "fixed_k_summary": fixed_k_summary,
    }
    (out / "results.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "fixed_k_events": orch.fixed_k_events,
                "per_frame": per_frame,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results -> {out / 'results.json'}")
    if not args.no_overlays:
        print(f"overlays -> {out / 'overlays'}")


if __name__ == "__main__":
    main()
