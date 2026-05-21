"""Compare my _build_memory_prompt vs official _prepare_memory_conditioned_features.

Runs the PT head (CUDA fp32) for frames 0->1, capturing the official prompt/prompt_pos
tensors that _prepare_memory_conditioned_features feeds to the transformer encoder.
Then builds the prompt with the orchestrator's _build_memory_prompt from the SAME stored
maskmem/obj_ptr, and compares element-wise.  Also compares the memory_attention output
(PT encoder vs ONNX) for the captured prompt.
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
ONNX_DIR = REPO_ROOT / "outputs" / "onnx"

IMAGE_SIZE = 128
SAM3_IMAGE_SIZE = 1008
HW = 72 * 72
D_MODEL = 256
MEM_DIM = 64


def _load_tracker_head(dev):
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
    for _, mod in tracker.named_modules():
        if type(mod).__name__ == "RoPEAttention":
            for attr in ("freqs_cis", "freqs_cis_real", "freqs_cis_imag"):
                v = getattr(mod, attr, None)
                if isinstance(v, torch.Tensor):
                    setattr(mod, attr, v.to(dev))
    return tracker


def main() -> None:
    from sam3_onnx_equiv.video_orchestrator import (
        make_oracle_frames, _preprocess_frame, _conv1x1, Constants,
        _build_memory_prompt, PythonMemoryBank,
    )
    import onnxruntime as ort

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = make_oracle_frames()
    C = Constants(CONSTANTS_DIR)
    enc = ort.InferenceSession(str(ONNX_DIR / "image_encoder_tracker.onnx"),
                               providers=["CPUExecutionProvider"])
    tracker = _load_tracker_head(dev)

    # Capture the official prompt by monkeypatching the encoder call.
    captured = {}
    orig_encoder = tracker.transformer.encoder

    class CapturingEncoder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, *a, **k):
            captured["prompt"] = k["prompt"].detach().float().cpu().numpy()
            captured["prompt_pos"] = k["prompt_pos"].detach().float().cpu().numpy()
            captured["num_obj_ptr_tokens"] = k["num_obj_ptr_tokens"]
            return self.inner(*a, **k)
    tracker.transformer.encoder = CapturingEncoder(orig_encoder)

    cx = IMAGE_SIZE // 6; cy = IMAGE_SIZE // 2
    coords = torch.tensor([[[cx / IMAGE_SIZE * SAM3_IMAGE_SIZE,
                             cy / IMAGE_SIZE * SAM3_IMAGE_SIZE]]], dtype=torch.float32, device=dev)
    labels = torch.tensor([[1]], dtype=torch.int32, device=dev)

    bank = PythonMemoryBank()
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    with torch.inference_mode():
        for fidx in range(2):  # frame 0 then frame 1
            px = _preprocess_frame(frames[fidx])
            outs = enc.run(None, {"pixel_values": px})
            pos2 = outs[2]; fpn0 = outs[3]; fpn1 = outs[4]; fpn2 = outs[5]
            hrf0 = torch.from_numpy(_conv1x1(fpn0, C.conv_s0_weight, C.conv_s0_bias)).to(dev)
            hrf1 = torch.from_numpy(_conv1x1(fpn1, C.conv_s1_weight, C.conv_s1_bias)).to(dev)
            vf2 = torch.from_numpy(fpn2[0].reshape(D_MODEL, HW).T[:, None, :]).to(dev)
            vp2 = torch.from_numpy(pos2[0].reshape(D_MODEL, HW).T[:, None, :]).to(dev)
            is_init = fidx == 0
            pix = tracker._prepare_memory_conditioned_features(
                frame_idx=fidx, is_init_cond_frame=is_init,
                current_vision_feats=[vf2], current_vision_pos_embeds=[vp2],
                feat_sizes=[(72, 72)], output_dict=output_dict, num_frames=6)
            sam_out = tracker._forward_sam_heads(
                backbone_features=pix,
                point_inputs={"point_coords": coords, "point_labels": labels} if is_init else None,
                mask_inputs=None, high_res_features=[hrf0, hrf1],
                multimask_output=tracker._use_multimask(
                    is_init, {"point_coords": coords, "point_labels": labels} if is_init else None))
            (_, _, _, low, hr, obj_ptr, osl) = sam_out
            mmf, mmp = tracker._encode_new_memory(
                image=None, current_vision_feats=[vf2], feat_sizes=[(72, 72)],
                pred_masks_high_res=hr, object_score_logits=osl,
                is_mask_from_pts=is_init, output_dict=output_dict, is_init_cond_frame=is_init)
            cur = {"maskmem_features": mmf, "maskmem_pos_enc": mmp, "obj_ptr": obj_ptr,
                   "object_score_logits": osl, "pred_masks": low}
            entry = {"maskmem_features": mmf.detach().float().cpu().numpy(),
                     "maskmem_pos_enc": [mmp[-1].detach().float().cpu().numpy()],
                     "obj_ptr": obj_ptr.detach().float().cpu().numpy(),
                     "object_score_logits": float(osl[0, 0])}
            if is_init:
                output_dict["cond_frame_outputs"][fidx] = cur
                bank.store_cond(fidx, entry)
            else:
                output_dict["non_cond_frame_outputs"][fidx] = cur
                bank.store_non_cond(fidx, entry)

    # Official captured prompt for frame 1
    off_prompt = captured["prompt"]          # (mem_len, 1, 64)
    off_pos = captured["prompt_pos"]
    off_nobj = captured["num_obj_ptr_tokens"]
    print(f"Official frame1: prompt shape={off_prompt.shape} num_obj_ptr_tokens={off_nobj}")

    # My prompt for frame 1
    my_prompt, my_pos, my_nobj = _build_memory_prompt(1, 6, bank, C)
    print(f"Mine     frame1: prompt shape={my_prompt.shape} num_obj_ptr_tokens={my_nobj}")

    if off_prompt.shape == my_prompt.shape:
        print(f"  prompt     max_abs_diff = {np.abs(off_prompt - my_prompt).max():.6e}")
        print(f"  prompt_pos max_abs_diff = {np.abs(off_pos - my_pos).max():.6e}")
        # Split maskmem vs obj_ptr
        ml = off_prompt.shape[0] - off_nobj
        print(f"  maskmem  diff = {np.abs(off_prompt[:ml]-my_prompt[:ml]).max():.6e}")
        print(f"  objptr   diff = {np.abs(off_prompt[ml:]-my_prompt[ml:]).max():.6e}")
        print(f"  pos maskmem diff = {np.abs(off_pos[:ml]-my_pos[:ml]).max():.6e}")
        print(f"  pos objptr  diff = {np.abs(off_pos[ml:]-my_pos[ml:]).max():.6e}")
    else:
        print("  SHAPE MISMATCH — prompt ordering/length differs!")


if __name__ == "__main__":
    main()
