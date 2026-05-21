"""Frame-0 isolation diagnostic (C-2 debugging, not a deliverable test).

Runs the equiv-source tracker float32 on CPU for frame 0 (is_init, point prompt)
and compares its decode output AND backbone features against the ONNX pipeline
(image_encoder.onnx + decode_head.onnx).  This removes the bf16 confound present
in the oracle and localises any per-frame mismatch to image_encoder vs decode_head
vs coordinate handling.
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
ONNX_DIR = REPO_ROOT / "outputs" / "onnx"
CONSTANTS_DIR = REPO_ROOT / "outputs" / "reference" / "constants"

HW = 72 * 72
D_MODEL = 256
SAM3_IMAGE_SIZE = 1008
IMAGE_SIZE = 128


def _load_equiv_tracker():
    equiv_root = str(EQUIV_SOURCE_ROOT.resolve())
    if equiv_root not in sys.path:
        sys.path.insert(0, equiv_root)
    for k in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
        del sys.modules[k]
    from sam3.model_builder import build_tracker  # type: ignore

    tracker = build_tracker(
        apply_temporal_disambiguation=False, with_backbone=True, use_rope_real=True
    )
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    state = {k[len("tracker.") :]: v for k, v in ckpt.items() if k.startswith("tracker.")}
    state.update(
        {k[len("detector.") :]: v for k, v in ckpt.items() if k.startswith("detector.backbone.")}
    )
    missing, unexpected = tracker.load_state_dict(state, strict=False)
    print(f"tracker load: missing={len(missing)} unexpected={len(unexpected)}")
    return tracker.cpu().float().eval()


def main() -> None:
    from sam3_onnx_equiv.video_orchestrator import _preprocess_frame, make_oracle_frames

    frames = make_oracle_frames()
    frame0 = frames[0]
    pixel_values = _preprocess_frame(frame0)  # (1,3,1008,1008)

    tracker = _load_equiv_tracker()

    # --- PyTorch backbone (forward_image applies conv_s0/s1 to fpn_0/1) ---
    img_t = torch.from_numpy(pixel_values).float()
    with torch.inference_mode():
        backbone_out = tracker.forward_image(img_t)
        _, vision_feats, vision_pos, feat_sizes = tracker._prepare_backbone_features(backbone_out)
        # top-level feature (HW, B, C)
        pt_fpn2 = vision_feats[-1]  # (5184,1,256)
        # high res features (already conv-projected by forward_image)
        hrf = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        pt_hrf0 = hrf[0]  # (1,32,288,288)
        pt_hrf1 = hrf[1]  # (1,64,144,144)

    # --- ONNX image encoder (tracker SAM2 neck) ---
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(ONNX_DIR / "image_encoder_tracker.onnx"), providers=["CPUExecutionProvider"]
    )
    enc = sess.run(None, {"pixel_values": pixel_values})
    onnx_fpn2 = enc[5]  # (1,256,72,72)
    onnx_fpn0 = enc[3]  # (1,256,288,288)
    onnx_fpn1 = enc[4]  # (1,256,144,144)

    # Reshape PT fpn2 (HW,1,256) -> (1,256,72,72) to compare with ONNX
    pt_fpn2_bchw = pt_fpn2[:, 0, :].T.reshape(1, D_MODEL, 72, 72).numpy()
    print("\n=== image_encoder parity (PT float32 vs ONNX) ===")
    print(f"  fpn2 max_abs_diff = {np.abs(pt_fpn2_bchw - onnx_fpn2).max():.6f}")
    print(f"  fpn2 PT mean={pt_fpn2_bchw.mean():.4f} ONNX mean={onnx_fpn2.mean():.4f}")

    # conv_s0/s1 from constants applied to ONNX raw fpn? No: forward_image already
    # applied them in PT. The ONNX image_encoder returns RAW fpn (image_encoder wrapper
    # does NOT apply conv_s0/s1). Compare PT high_res (conv-projected) vs ONNX conv-projected.
    from sam3_onnx_equiv.video_orchestrator import Constants, _conv1x1

    C = Constants(CONSTANTS_DIR)
    onnx_hrf0 = _conv1x1(onnx_fpn0, C.conv_s0_weight, C.conv_s0_bias)
    onnx_hrf1 = _conv1x1(onnx_fpn1, C.conv_s1_weight, C.conv_s1_bias)
    print(f"  hrf0 max_abs_diff = {np.abs(pt_hrf0.numpy() - onnx_hrf0).max():.6f}")
    print(f"  hrf1 max_abs_diff = {np.abs(pt_hrf1.numpy() - onnx_hrf1).max():.6f}")
    # Is the ONNX fpn2 already conv-projected? Compare PT fpn2 (which is NOT conv-projected;
    # only fpn0/fpn1 get conv) -> they should match.

    # --- PyTorch decode head via track_step (frame 0, is_init, point) ---
    cx = IMAGE_SIZE // 6
    cy = IMAGE_SIZE // 2
    coords = torch.tensor(
        [[[cx / IMAGE_SIZE * SAM3_IMAGE_SIZE, cy / IMAGE_SIZE * SAM3_IMAGE_SIZE]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1]], dtype=torch.int32)
    point_inputs = {"point_coords": coords, "point_labels": labels}
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    with torch.inference_mode():
        out = tracker.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            image=img_t,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=6,
        )
    pt_low = out["pred_masks"].float().numpy()  # (1,1,288,288)
    pt_score = float(out["object_score_logits"][0, 0])
    pt_px = int((pt_low[0, 0] > 0).sum())
    print(f"\n=== PT track_step frame0: score={pt_score:.4f} px={pt_px} ===")

    # --- ONNX decode head ---
    dsess = ort.InferenceSession(
        str(ONNX_DIR / "decode_head.onnx"), providers=["CPUExecutionProvider"]
    )
    # Replicate orchestrator: pix_feat = fpn2_seq + no_mem_embed -> (1,256,72,72)
    fpn2_seq = onnx_fpn2[0].reshape(D_MODEL, HW).T[:, None, :]
    pix_seq = fpn2_seq + C.no_mem_embed
    pix_bchw = pix_seq.transpose(1, 2, 0).reshape(1, D_MODEL, 72, 72).astype(np.float32)
    dout = dsess.run(
        None,
        {
            "image_embeddings": pix_bchw,
            "high_res_feat0": onnx_hrf0,
            "high_res_feat1": onnx_hrf1,
            "point_coords": coords.numpy(),
            "point_labels": labels.numpy(),
            "mask_input": np.zeros((1, 1, 288, 288), np.float32),
            "has_mask_input": np.zeros((1,), np.float32),
        },
    )
    onnx_low = dout[0]
    onnx_score = float(dout[2][0, 0])
    onnx_px = int((onnx_low[0, 0] > 0).sum())
    print(f"=== ONNX decode frame0: score={onnx_score:.4f} px={onnx_px} ===")
    print(f"  low_res max_abs_diff = {np.abs(pt_low - onnx_low).max():.6f}")
    # IoU between PT and ONNX masks
    pm = pt_low[0, 0] > 0
    om = onnx_low[0, 0] > 0
    inter = (pm & om).sum()
    union = (pm | om).sum()
    print(f"  PT-vs-ONNX mask IoU = {inter / union if union else 1.0:.4f}")

    # Compare PT no-mem pix_feat against what track_step actually used
    # (track_step frame0 uses _prepare_memory_conditioned_features no-mem path internally)


if __name__ == "__main__":
    main()
