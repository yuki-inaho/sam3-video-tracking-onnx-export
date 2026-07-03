"""実データ1トラックレットを ONNX video orchestrator で推論する開発用ドライバ。

repo同梱の run_onnx_video.py は合成6フレーム+oracle比較専用。本スクリプトは
temp 以下の簡易データセット(連続フレーム + frame0 シード点)を読み、
VideoOrchestrator で追跡し、per-frame マスク/スコア/GTボックスIoU とオーバーレイを出す。

長いクリップは obj_ptr トークン数が 24 を超えるため、追加 export 済みの
memory_attention_dynamic_k{28..64}.onnx も orchestrator にロードする。

Usage:
    uv run python tools/run_onnx_tracklet.py --dataset <dir> --num-frames 127
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

ONNX_DIR = REPO / "outputs" / "onnx"
CONSTANTS_DIR = REPO / "outputs" / "reference" / "constants"
EXTRA_K = [28, 32, 36, 40, 44, 48, 52, 56, 60, 64]


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # xyxy


def _iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-frames", type=int, default=127)
    ap.add_argument("--out-subdir", default="onnx_infer")
    args = ap.parse_args()

    import onnxruntime as ort
    from sam3_onnx_equiv.video_orchestrator import VideoOrchestrator

    ds = Path(args.dataset)
    prompt = json.loads((ds / "prompt.json").read_text())
    meta = json.loads((ds / "meta.json").read_text())
    gt = json.loads((ds / "tracklet_boxes.json").read_text())
    gt_by_clip = {b["clip_idx"]: b["bbox_xywh_800x600"] for b in gt["boxes"]}
    W, H = meta["source_view_size"]

    providers = ["CPUExecutionProvider"]
    print(f"[init] VideoOrchestrator (providers={providers}) ...", flush=True)
    orch = VideoOrchestrator(ONNX_DIR, CONSTANTS_DIR, providers)
    # 長いクリップ用に追加 k グラフをロード
    for k in EXTRA_K:
        p = ONNX_DIR / f"memory_attention_dynamic_k{k}.onnx"
        orch._mem_attn[k] = ort.InferenceSession(str(p), providers=providers)
    print(f"[init] mem_attn graphs: {sorted(orch._mem_attn)}", flush=True)

    # フレーム読込
    n = min(args.num_frames, meta["clip_len"])
    frames = []
    for i in range(n):
        frames.append(Image.open(ds / "frames" / f"{i:06d}.jpg").convert("RGB"))
    print(f"[load] {n} frames", flush=True)

    coords = np.array(prompt["frame0_point_coords_norm"], dtype=np.float32)
    labels = np.array(prompt["frame0_point_labels"], dtype=np.int32)

    t0 = time.time()
    res = orch.run_clip(frames, coords, labels, use_memory=True)
    dt = time.time() - t0
    print(f"[run] {n} frames in {dt:.1f}s ({dt/n:.2f}s/frame), "
          f"mem_attn_invocations={res['memory_attention_invoke_count']}", flush=True)

    out = ds / args.out_subdir
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    per_frame = []
    ious = []
    for i in range(n):
        mask288 = res["masks"][i]  # (288,288) bool
        score = float(res["scores"][i])
        # 288 -> 800x600
        m = Image.fromarray((mask288.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)
        m = np.asarray(m) > 127
        pbox = _mask_bbox(m)
        rec = {"clip_idx": i, "obj_score": round(score, 4),
               "mask_px_288": int(mask288.sum()), "pred_bbox_xyxy_800x600": pbox}
        # IoU vs GT (tracklet 存在フレームのみ)
        if i in gt_by_clip and pbox is not None:
            x, y, w, h = gt_by_clip[i]
            gbox = [x, y, x + w, y + h]
            iou = _iou_xyxy(pbox, gbox)
            rec["gt_bbox_xyxy_800x600"] = [round(v, 1) for v in gbox]
            rec["iou_vs_gt"] = round(iou, 4)
            ious.append(iou)
        per_frame.append(rec)

        # overlay 出力 (先頭は間引かず、全フレーム出すと重いので 0..n を等間隔+GT期間)
        if i % 3 == 0 or i in gt_by_clip:
            base = frames[i].copy()
            ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            ys, xs = np.where(m)
            if xs.size:
                for yy, xx in zip(ys[::7], xs[::7]):
                    od.point((int(xx), int(yy)), fill=(255, 40, 40, 160))
            comp = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(comp)
            if pbox:
                d.rectangle(pbox, outline=(255, 230, 0), width=2)  # pred bbox 黄
            if i in gt_by_clip:
                x, y, w, h = gt_by_clip[i]
                d.rectangle([x, y, x + w, y + h], outline=(0, 220, 0), width=2)  # GT 緑
            if i == 0:
                cx, cy = prompt["seed_point_abs_800x600"]
                d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(0, 160, 255), width=3)
            d.text((5, 5), f"f{i} score={score:.2f}", fill=(255, 255, 255))
            comp.save(out / "overlays" / f"{i:06d}.jpg", quality=88)

    summary = {
        "dataset": meta["dataset"], "num_frames_run": n,
        "seconds_total": round(dt, 1), "seconds_per_frame": round(dt / n, 3),
        "mem_attn_invocations": res["memory_attention_invoke_count"],
        "obj_score_min": round(min(r["obj_score"] for r in per_frame), 3),
        "obj_score_mean": round(float(np.mean([r["obj_score"] for r in per_frame])), 3),
        "frames_object_present(score>0)": int(sum(r["obj_score"] > 0 for r in per_frame)),
        "gt_frames_evaluated": len(ious),
        "iou_mean_vs_gt": round(float(np.mean(ious)), 4) if ious else None,
        "iou_median_vs_gt": round(float(np.median(ious)), 4) if ious else None,
    }
    (out / "results.json").write_text(json.dumps(
        {"summary": summary, "per_frame": per_frame}, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"overlays -> {out/'overlays'}  results -> {out/'results.json'}")


if __name__ == "__main__":
    main()
