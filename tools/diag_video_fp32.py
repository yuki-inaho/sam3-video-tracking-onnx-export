"""Full-clip fp32 PyTorch HEAD reference vs ONNX orchestrator (C-3 attribution).

The ViT backbone in fp32 OOMs on an 8 GB GPU, so this diagnostic uses the ONNX
tracker image encoder (CPU) to obtain backbone features, then runs ONLY the tracker
HEAD (_prepare_memory_conditioned_features + _forward_sam_heads + _encode_new_memory)
in PyTorch float32 on CUDA (the head is tiny).  This isolates the memory/decode logic
in fp32 with no bf16 quantization.

Purpose: attribute the per-frame object_score gap (ONNX-fp32 vs the bf16 C-1 oracle)
to bf16/fp32 quantization rather than a logic bug.  If ONNX-fp32 matches PT-fp32-head
tightly, the gap to the bf16 oracle is the oracle's bf16 recurrent drift (accepted per
the briefing: oracle=bf16 / ONNX=fp32 difference is permitted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

EQUIV_SOURCE_ROOT = REPO_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = REPO_ROOT / "models" / "sam3.pt"
CONSTANTS_DIR = REPO_ROOT / "outputs" / "reference" / "constants"
ORACLE_NPZ = REPO_ROOT / "outputs" / "reference" / "video_oracle_all.npz"
ONNX_DIR = REPO_ROOT / "outputs" / "onnx"

IMAGE_SIZE = 128
SAM3_IMAGE_SIZE = 1008
HW = 72 * 72
D_MODEL = 256


def _load_tracker_head(dev):
    """build_tracker WITHOUT backbone (head only) on the given device, fp32."""
    equiv_root = str(EQUIV_SOURCE_ROOT.resolve())
    if equiv_root not in sys.path:
        sys.path.insert(0, equiv_root)
    for k in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
        del sys.modules[k]
    from sam3.model_builder import build_tracker  # type: ignore

    tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=False,
                            use_rope_real=True)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    state = {k[len("tracker."):]: v for k, v in ckpt.items() if k.startswith("tracker.")}
    tracker.load_state_dict(state, strict=False)
    tracker = tracker.to(dev).float().eval()
    # Move plain RoPE attrs (non-buffer) to device.
    for _, mod in tracker.named_modules():
        if type(mod).__name__ == "RoPEAttention":
            for attr in ("freqs_cis", "freqs_cis_real", "freqs_cis_imag"):
                v = getattr(mod, attr, None)
                if isinstance(v, torch.Tensor):
                    setattr(mod, attr, v.to(dev))
    return tracker


def _mask_iou(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def main() -> None:
    from sam3_onnx_equiv.video_orchestrator import (
        make_oracle_frames, _preprocess_frame, _conv1x1, Constants, VideoOrchestrator,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = make_oracle_frames()
    num_frames = len(frames)
    C = Constants(CONSTANTS_DIR)

    import onnxruntime as ort
    enc = ort.InferenceSession(str(ONNX_DIR / "image_encoder_tracker.onnx"),
                               providers=["CPUExecutionProvider"])

    tracker = _load_tracker_head(dev)

    cx = IMAGE_SIZE // 6
    cy = IMAGE_SIZE // 2
    coords = torch.tensor([[[cx / IMAGE_SIZE * SAM3_IMAGE_SIZE,
                             cy / IMAGE_SIZE * SAM3_IMAGE_SIZE]]], dtype=torch.float32, device=dev)
    labels = torch.tensor([[1]], dtype=torch.int32, device=dev)

    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    pt_low, pt_score = [], []
    with torch.inference_mode():
        for fidx, frame in enumerate(frames):
            px = _preprocess_frame(frame)
            outs = enc.run(None, {"pixel_values": px})
            pos2 = outs[2]; fpn0 = outs[3]; fpn1 = outs[4]; fpn2 = outs[5]
            # conv_s0/s1 applied (forward_image does this in place); head expects projected hrf.
            hrf0 = torch.from_numpy(_conv1x1(fpn0, C.conv_s0_weight, C.conv_s0_bias)).to(dev)
            hrf1 = torch.from_numpy(_conv1x1(fpn1, C.conv_s1_weight, C.conv_s1_bias)).to(dev)
            # seq-first features (HW, B, C)
            vf2 = torch.from_numpy(fpn2[0].reshape(D_MODEL, HW).T[:, None, :]).to(dev)
            vp2 = torch.from_numpy(pos2[0].reshape(D_MODEL, HW).T[:, None, :]).to(dev)
            vision_feats = [vf2]            # head uses [-1] for memory; high_res passed separately
            vision_pos = [vp2]
            feat_sizes = [(72, 72)]
            is_init = fidx == 0
            # high_res_features must be the projected hrf (track_step builds them from
            # current_vision_feats[:-1]); we pass them directly via _forward_sam_heads path.
            # Reproduce track_step's no-mask branch with our own high_res_features.
            pix_feat_with_mem = tracker._prepare_memory_conditioned_features(
                frame_idx=fidx, is_init_cond_frame=is_init,
                current_vision_feats=vision_feats, current_vision_pos_embeds=vision_pos,
                feat_sizes=feat_sizes, output_dict=output_dict, num_frames=num_frames,
            )
            sam_out = tracker._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs={"point_coords": coords, "point_labels": labels} if is_init else None,
                mask_inputs=None, high_res_features=[hrf0, hrf1],
                multimask_output=tracker._use_multimask(
                    is_init, {"point_coords": coords, "point_labels": labels} if is_init else None),
            )
            (_, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr,
             object_score_logits) = sam_out
            maskmem_features, maskmem_pos_enc = tracker._encode_new_memory(
                image=None, current_vision_feats=vision_feats, feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks, object_score_logits=object_score_logits,
                is_mask_from_pts=is_init, output_dict=output_dict, is_init_cond_frame=is_init,
            )
            cur = {"maskmem_features": maskmem_features, "maskmem_pos_enc": maskmem_pos_enc,
                   "obj_ptr": obj_ptr, "object_score_logits": object_score_logits,
                   "pred_masks": low_res_masks}
            if is_init:
                output_dict["cond_frame_outputs"][fidx] = cur
            else:
                output_dict["non_cond_frame_outputs"][fidx] = cur
            pt_low.append(low_res_masks.float().cpu().numpy())
            pt_score.append(float(object_score_logits[0, 0]))

    orch = VideoOrchestrator(ONNX_DIR, CONSTANTS_DIR, ["CPUExecutionProvider"])
    pcn = np.array([[[cx / IMAGE_SIZE, cy / IMAGE_SIZE]]], dtype=np.float32)
    res = orch.run_clip(frames, pcn, np.array([[1]], dtype=np.int32), use_memory=True)

    data = np.load(str(ORACLE_NPZ), allow_pickle=True)
    bf16_scores = data["probs_per_frame"]

    print("\n=== ONNX-fp32 vs PT-fp32-head | vs bf16 oracle ===")
    for i in range(num_frames):
        pm = pt_low[i][0, 0] > 0
        om = res["masks"][i]
        iou = _mask_iou(pm, om)
        ps = pt_score[i]; cs = float(res["scores"][i]); bs = float(bf16_scores[i][0])
        rel_pt = abs(cs - ps) / max(abs(ps), 1e-8)
        rel_bf = abs(cs - bs) / max(abs(bs), 1e-8)
        print(f"  f{i}: IoU(onnx,ptfp32)={iou:.4f} | onnx={cs:.4f} ptfp32={ps:.4f} "
              f"rel={rel_pt:.4f} || bf16={bs:.4f} rel_bf={rel_bf:.4f}")


if __name__ == "__main__":
    main()
