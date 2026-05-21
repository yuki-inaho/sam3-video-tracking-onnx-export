"""Stage C-2: ONNX video orchestrator for SAM3 memory-bank tracking.

Implements the full tracking loop using 4 ONNX modules + a Python-side memory
bank:
  image_encoder_tracker.onnx         — ViT backbone + SAM2 neck FPN (per frame).
                                       Uses sam2_backbone_out (the tracker FPN), NOT
                                       the detector image_encoder.onnx (sam3 FPN).
  memory_attention_dynamic_k{N}.onnx — TransformerEncoderCrossAttention (frames t>=1).
                                       Variable-length maskmem (dim 0 dynamic); one
                                       graph per obj_ptr token count N (num_k_exclude_rope).
  decode_head.onnx                   — prompt_encoder + mask_decoder + obj_ptr_proj
                                       (multimask_output=True + best-IoU selection).
  memory_encoder.onnx                — SimpleMaskEncoder (maskmem feature extractor)

All inter-frame state (maskmem_features, maskmem_pos_enc, obj_ptr) is held in
Python (PythonMemoryBank).  The ONNX graphs are stateless.

The processing per frame mirrors the official tracker
(sam3/model/sam3_tracker_base.py):
  track_step(932) -> _prepare_memory_conditioned_features(572)
                  -> _forward_sam_heads(218) -> _encode_new_memory(797)

Key semantics replicated from the official code (no implicit fallback):
  * is_mask_from_pts = (point_inputs is not None)  (track_step:1038).
    Frame 0 carries a point prompt  -> is_mask_from_pts=True  -> memory mask is
      BINARY: mask_for_mem = (high_res_masks > 0).float()  (_encode_new_memory:822-823).
    Frames >= 1 carry no prompt      -> is_mask_from_pts=False -> SIGMOID:
      mask_for_mem = sigmoid(high_res_masks)               (_encode_new_memory:824-826).
    Both then apply * sigmoid_scale + sigmoid_bias.
  * memory prompt ordering (_prepare_memory_conditioned_features):
    - selected conditioning frames first (t=0 -> maskmem_tpos_enc[num_maskmem-1]),
    - then non-conditioning frames t_pos=1..num_maskmem-1
      (t -> maskmem_tpos_enc[num_maskmem-t-1]).
    Only frames that actually exist contribute (dynamic length, no zero padding
    of the maskmem section).
  * object pointers: from selected cond frames (rel_pos = frame_idx - t) and up to
    (max_obj_ptrs_in_encoder-1) non-cond frames (rel_pos = t_diff).  Each pointer
    (256-d) is split into C//mem_dim = 4 tokens of mem_dim=64.  Temporal position is
    get_1d_sine_pe(rel_pos / (max_obj_ptrs_in_encoder-1), dim=256) projected by
    obj_ptr_tpos_proj (Linear 256->64), then repeat_interleaved over the 4 tokens.
  * mask gating: low_res_masks = where(obj_score > 0, low_res_masks, NO_OBJ_SCORE).
    decode_head.onnx already applies this internally (B-3).
  * no_obj_embed_spatial added after memory_encoder (_encode_new_memory:845-848).

Device:  CPUExecutionProvider (explicit; no auto-detect).
Dtype:   float32 throughout (oracle is bf16; the bf16/fp32 gap is accepted but the
         DoD thresholds are NOT silently relaxed).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
from jaxtyping import Bool, Float, Int
from PIL import Image

log = logging.getLogger(__name__)


class FrameEntry(TypedDict):
    """Per-frame memory bank record (mirrors the official output_dict entries)."""

    maskmem_features: Float[np.ndarray, "1 64 72 72"]
    maskmem_pos_enc: list[Float[np.ndarray, "1 64 72 72"]]
    obj_ptr: Float[np.ndarray, "1 256"]
    object_score_logits: float


class ClipResult(TypedDict):
    """Per-clip tracking outputs returned by ``VideoOrchestrator.run_clip``."""

    masks: list[Bool[np.ndarray, "288 288"]]
    scores: list[float]
    obj_ids: list[int]
    low_res_mask_logits: list[Float[np.ndarray, "1 288 288"]]
    memory_attention_invoke_count: int


# Fixed spatial / model dimensions (build_tracker config, model_builder.py:449).
HW = 72 * 72  # 5184 spatial tokens (1008 / 14 = 72)
FEAT_H = FEAT_W = 72  # spatial resolution of the top-level FPN feature
D_MODEL = 256  # tracker hidden dim
MEM_DIM = 64  # maskmem / obj_ptr token channel dim
NUM_MASKMEM = 7  # total memory slots (1 selected cond + up to 6 non-cond)
OBJ_PTR_TOKENS_PER_FRAME = D_MODEL // MEM_DIM  # = 4
MAX_OBJ_PTRS_IN_ENCODER = 16  # sam3_tracker_base default
MAX_COND_FRAMES_IN_ATTN = 4  # build_tracker config

# memory_attention_dynamic_k{N}.onnx graphs bake num_k_exclude_rope=N (obj_ptr token
# count).  The cross-attention applies RoPE to the first (mem_len - N) keys; num_k_rope
# must be a multiple of HW, so N MUST equal the real obj_ptr token count (no padding).
# One graph per valid count (frame t: min(t,6) sets × 4 tokens).
DYN_NUM_K_VALUES = (4, 8, 12, 16, 20, 24)

# Synthetic frame parameters (must match tools/run_pytorch_video.py).
N_FRAMES = 6
IMAGE_SIZE = 128
CIRCLE_RADIUS = 16
STEP = 12  # pixels right per frame
JPEG_QUALITY = 95

# SAM3 image resolution after resize in the ViT backbone.
SAM3_IMAGE_SIZE = 1008
NO_OBJ_SCORE: float = -1024.0

ROPE_THETA = 10000.0  # get_1d_sine_pe temperature (sam3_tracker_utils.py:327)


# ---------------------------------------------------------------------------
# Synthetic frame generator (must match C-1 oracle)
# ---------------------------------------------------------------------------


def make_oracle_frames(jpeg_quality: int = JPEG_QUALITY) -> list[Image.Image]:
    """Generate the same 6 synthetic frames used by run_pytorch_video.py.

    The oracle saved frames as JPEG (quality=95) before loading them, so we apply
    the same JPEG round-trip to match the exact pixel values the oracle saw.

    Returns a list of PIL RGB images (128x128).
    """
    import io  # noqa: PLC0415

    from PIL import ImageDraw  # noqa: PLC0415

    frames: list[Image.Image] = []
    start_x = IMAGE_SIZE // 6
    for i in range(N_FRAMES):
        cx = start_x + i * STEP
        cy = IMAGE_SIZE // 2
        img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        r = CIRCLE_RADIUS
        ImageDraw.Draw(img).ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        img_jpeg = Image.open(buf).convert("RGB")
        img_jpeg.load()
        frames.append(img_jpeg)
    return frames


# ---------------------------------------------------------------------------
# Constants loader
# ---------------------------------------------------------------------------


class Constants:
    """Python-side SAM3 constants extracted from the checkpoint (run_pytorch_video.py)."""

    def __init__(self, constants_dir: Path) -> None:
        self.dir = constants_dir
        self.maskmem_tpos_enc = self._load("maskmem_tpos_enc")  # (7,1,1,64)
        self.no_mem_embed = self._load("no_mem_embed")  # (1,1,256)
        self.no_mem_pos_enc = self._load("no_mem_pos_enc")  # (1,1,256)
        self.no_obj_embed_spatial = self._load("no_obj_embed_spatial")  # (1,64)
        self.conv_s0_weight = self._load("conv_s0_weight")  # (32,256,1,1)
        self.conv_s0_bias = self._load("conv_s0_bias")  # (32,)
        self.conv_s1_weight = self._load("conv_s1_weight")  # (64,256,1,1)
        self.conv_s1_bias = self._load("conv_s1_bias")  # (64,)
        self.obj_ptr_tpos_proj_weight = self._load("obj_ptr_tpos_proj_weight")  # (64,256)
        self.obj_ptr_tpos_proj_bias = self._load("obj_ptr_tpos_proj_bias")  # (64,)
        self.NO_OBJ_SCORE = float(self._load("NO_OBJ_SCORE"))  # scalar
        self.sigmoid_scale = float(self._load("sigmoid_scale_for_mem_enc"))  # 20.0
        self.sigmoid_bias = float(self._load("sigmoid_bias_for_mem_enc"))  # -10.0

    def _load(self, name: str) -> np.ndarray:
        path = self.dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Constant not found: {path}")
        return np.load(str(path)).astype(np.float32)


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------


class PythonMemoryBank:
    """Stores per-frame maskmem_features, maskmem_pos_enc, obj_ptr, object_score.

    Mirrors the official output_dict structure (sam3_tracker_base.py):
      cond_frame_outputs     : conditioning frames (frame 0 here, with point prompt)
      non_cond_frame_outputs : propagated frames (1..N)

    Each entry:
      {
        "maskmem_features":     np.ndarray (1, 64, 72, 72) float32,
        "maskmem_pos_enc":      [np.ndarray (1, 64, 72, 72)] float32  (list of 1),
        "obj_ptr":              np.ndarray (1, 256) float32,
        "object_score_logits":  float,
      }
    """

    def __init__(self) -> None:
        self.cond_frame_outputs: dict[int, FrameEntry] = {}
        self.non_cond_frame_outputs: dict[int, FrameEntry] = {}

    def store_cond(self, frame_idx: int, entry: FrameEntry) -> None:
        self.cond_frame_outputs[frame_idx] = entry

    def store_non_cond(self, frame_idx: int, entry: FrameEntry) -> None:
        self.non_cond_frame_outputs[frame_idx] = entry

    def clear(self) -> None:
        self.cond_frame_outputs.clear()
        self.non_cond_frame_outputs.clear()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _preprocess_frame(pil_img: Image.Image) -> Float[np.ndarray, "1 3 1008 1008"]:
    """Resize and normalise a PIL RGB frame for image_encoder.onnx.

    Matches _load_img_as_tensor + AsyncVideoFrameLoader.__getitem__ exactly
    (sam3/model/utils/sam2_utils.py:16-89):
      - img = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        NOTE: PIL .resize() default resample is BICUBIC (resample=None -> BICUBIC).
      - img /= 255.0
      - img -= mean ; img /= std  with ImageNet stats.

    Returns float32 ndarray (1, 3, 1008, 1008).
    """
    # PIL default resample (None) == BICUBIC for RGB; this matches the oracle loader.
    img = pil_img.convert("RGB").resize((SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return arr


# ---------------------------------------------------------------------------
# conv_s0 / conv_s1 projection (1x1 conv) and mask upsampling
# ---------------------------------------------------------------------------


def _conv1x1(
    x: Float[np.ndarray, "1 c_in h w"],
    weight: Float[np.ndarray, "c_out c_in 1 1"],
    bias: Float[np.ndarray, c_out],
) -> Float[np.ndarray, "1 c_out h w"]:
    """Apply a 1x1 convolution (stride 1, no padding).

    Args:
        x:      (1, C_in, H, W) float32
        weight: (C_out, C_in, 1, 1) float32
        bias:   (C_out,) float32

    Returns:
        (1, C_out, H, W) float32
    """
    c_out, c_in, _, _ = weight.shape
    w = weight[:, :, 0, 0]  # (C_out, C_in)
    _, _, h, w_in = x.shape
    x_flat = x[0].reshape(c_in, h * w_in).T  # (H*W, C_in)
    y_flat = x_flat @ w.T + bias  # (H*W, C_out)
    return y_flat.T.reshape(1, c_out, h, w_in).astype(np.float32)


def _upsample_bilinear(
    mask: Float[np.ndarray, "1 1 h_in w_in"], target_h: int, target_w: int
) -> Float[np.ndarray, "1 1 target_h target_w"]:
    """Bilinear upsample mask (1, 1, H_in, W_in) -> (1, 1, target_h, target_w).

    Uses torch.nn.functional.interpolate(mode='bilinear', align_corners=False) to
    match _forward_sam_heads:353 exactly (same op the oracle runs).  torch is a
    project dependency; no implicit fallback is taken.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    h_in, w_in = mask.shape[2], mask.shape[3]
    if (h_in, w_in) == (target_h, target_w):
        return mask
    t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
    out = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return out.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Object-pointer temporal positional encoding
# ---------------------------------------------------------------------------


def _to_bf16_round(arr: Float[np.ndarray, ...]) -> Float[np.ndarray, ...]:
    """Round a float32 array to bfloat16 precision and back (diagnostic only).

    Used by emulate_oracle_bf16 to mimic the bf16 oracle's recurrent memory storage.
    bf16 truncation: keep the high 16 bits of the float32 representation (round to
    nearest-even via torch.bfloat16).
    """
    import torch  # noqa: PLC0415

    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return t.to(torch.bfloat16).to(torch.float32).numpy()


def _get_1d_sine_pe(
    pos_inds: Float[np.ndarray, n], dim: int, temperature: float = ROPE_THETA
) -> Float[np.ndarray, "n dim"]:
    """NumPy port of sam3_tracker_utils.get_1d_sine_pe (line 327).

    Args:
        pos_inds: (N,) float32 array of normalised positions.
        dim:      embedding dim (256 for the tracker hidden dim).

    Returns:
        (N, dim) float32 positional embedding [sin(.), cos(.)] concatenated.
    """
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds[:, None] / dim_t  # (N, pe_dim)
    return np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=-1).astype(np.float32)


def _obj_ptr_tpos(
    rel_pos_list: list[int], max_abs_pos: int, constants: Constants
) -> Float[np.ndarray, "n 64"]:
    """Compute obj_ptr temporal positional encoding (sam3_tracker_base._get_tpos_enc:162).

    pos_enc = get_1d_sine_pe(rel_pos / (max_abs_pos - 1), dim=256)
            -> obj_ptr_tpos_proj (Linear 256 -> 64).

    Args:
        rel_pos_list: list of integer temporal distances (one per obj_ptr frame).
        max_abs_pos:  max_obj_ptrs_in_encoder used as normaliser (min(num_frames, 16)).
        constants:    holds obj_ptr_tpos_proj weight/bias.

    Returns:
        (len(rel_pos_list), 64) float32 — one positional vector per pointer frame.
    """
    t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
    pos = np.asarray(rel_pos_list, dtype=np.float32) / float(t_diff_max)
    pos_enc = _get_1d_sine_pe(pos, dim=D_MODEL)  # (N, 256)
    # Linear(256 -> 64): y = x @ W^T + b
    return (
        pos_enc @ constants.obj_ptr_tpos_proj_weight.T + constants.obj_ptr_tpos_proj_bias
    ).astype(np.float32)  # (N, 64)


# ---------------------------------------------------------------------------
# Memory bank prompt builder
# ---------------------------------------------------------------------------


def _maskmem_seq(entry: FrameEntry) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one frame's maskmem feature/pos to seq-first (5184, 1, 64) tensors.

    Args:
        entry: A memory-bank frame record.

    Returns:
        ``(feats, pos)`` each float32 (5184, 1, 64).  The temporal position
        encoding is NOT added here (the caller adds it to ``pos`` / the key only).
    """
    feats = entry["maskmem_features"][0].reshape(MEM_DIM, HW).T[:, None, :]  # (5184,1,64)
    pos = entry["maskmem_pos_enc"][-1][0].reshape(MEM_DIM, HW).T[:, None, :]
    return feats, pos


def _collect_maskmem(
    frame_idx: int,
    bank: PythonMemoryBank,
    tpos_enc: Float[np.ndarray, "7 1 1 64"],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect the maskmem (value/key) sequences in the official attention order.

    Order (sam3_tracker_base.py:586-792, single-object oracle config):
      (1) selected conditioning frames (t=0, idx = num_maskmem-1);
      (2) non-conditioning frames t_pos = 1..num_maskmem-1
          (prev = frame_idx-(num_maskmem-t_pos), idx = num_maskmem-t_pos-1).

    tpos is added to the key (pos) only, never the value (feats):
    tracker_base.py:659 prompt = feats only; :676-678 prompt_pos = pos + tpos.

    Returns:
        ``(maskmem_feats, maskmem_poses)`` — parallel lists of (5184,1,64) arrays.
    """
    cond = bank.cond_frame_outputs
    non_cond = bank.non_cond_frame_outputs
    maskmem_feats: list[np.ndarray] = []
    maskmem_poses: list[np.ndarray] = []

    # (1) selected conditioning frames (t=0).  select_closest_cond_frames returns
    # ALL cond frames when len <= max_cond.
    for t_cond in sorted(cond.keys()):
        feats, pos = _maskmem_seq(cond[t_cond])
        idx = NUM_MASKMEM - 0 - 1  # = 6 (t = 0 for selected cond frame)
        maskmem_feats.append(feats)
        maskmem_poses.append(pos + tpos_enc[idx])

    # (2) non-conditioning frames t_pos = 1..num_maskmem-1.
    for t_pos in range(1, NUM_MASKMEM):
        prev_frame_idx = frame_idx - (NUM_MASKMEM - t_pos)
        if prev_frame_idx < 0:
            continue
        entry = non_cond.get(prev_frame_idx)
        if entry is None:
            # An unselected conditioning frame among the last (num_maskmem-1) frames
            # is attended to as a non-conditioning frame.  With a single cond frame
            # (frame 0) that is always selected, this is None and the slot is skipped.
            continue
        feats, pos = _maskmem_seq(entry)
        idx = NUM_MASKMEM - t_pos - 1
        maskmem_feats.append(feats)
        maskmem_poses.append(pos + tpos_enc[idx])

    return maskmem_feats, maskmem_poses


def _collect_obj_ptrs(
    frame_idx: int,
    num_frames: int,
    bank: PythonMemoryBank,
    constants: Constants,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collect the object-pointer token/pos sequences (sam3_tracker_base.py:737-761).

    max_obj_ptrs = min(num_frames, MAX_OBJ_PTRS_IN_ENCODER).
      - selected cond frames (t <= frame_idx): rel_pos = frame_idx - t.
      - non-cond frames t_diff = 1..max_obj_ptrs-1: t = frame_idx - t_diff; rel_pos = t_diff.
    Each 256-d pointer is reshaped to 4 tokens of 64-d; pos = obj_ptr_tpos (256->64)
    repeat-interleaved over the 4 tokens.  No padding.

    Returns:
        ``(obj_tokens, obj_pos, num_obj_ptr_tokens)`` where obj_tokens / obj_pos are
        float32 (n_ptr*4, 1, 64) and num_obj_ptr_tokens == n_ptr * 4.
    """
    cond = bank.cond_frame_outputs
    non_cond = bank.non_cond_frame_outputs
    max_obj_ptrs = min(num_frames, MAX_OBJ_PTRS_IN_ENCODER)

    pos_and_ptrs: list[tuple[int, np.ndarray]] = []  # (rel_pos, obj_ptr (1,256))
    for t_cond in sorted(cond.keys()):
        if t_cond <= frame_idx:
            pos_and_ptrs.append((frame_idx - t_cond, cond[t_cond]["obj_ptr"]))
    for t_diff in range(1, max_obj_ptrs):
        t = frame_idx - t_diff
        if t < 0:
            break
        entry = non_cond.get(t)
        if entry is not None:
            pos_and_ptrs.append((t_diff, entry["obj_ptr"]))

    if not pos_and_ptrs:
        raise ValueError(
            f"frame {frame_idx}: no object pointers available — the memory path always "
            "has at least the conditioning frame's pointer (no implicit fallback)."
        )

    rel_list = [rp for rp, _ in pos_and_ptrs]
    ptrs = np.stack([p[0] for _, p in pos_and_ptrs], axis=0)  # (n_ptr, 256)
    # split each pointer into C//mem_dim = 4 tokens (reshape/permute/flatten, tracker_base:759-760)
    ptr_tokens = ptrs.reshape(-1, OBJ_PTR_TOKENS_PER_FRAME, MEM_DIM)  # (n_ptr,4,64)
    obj_tokens = ptr_tokens.reshape(-1, MEM_DIM)[:, None, :]  # (n_ptr*4,1,64)
    pos_proj = _obj_ptr_tpos(rel_list, max_obj_ptrs, constants)  # (n_ptr,64)
    # repeat_interleave over the 4 tokens (tracker_base:761)
    obj_pos = np.repeat(pos_proj, OBJ_PTR_TOKENS_PER_FRAME, axis=0)[:, None, :]  # (n_ptr*4,1,64)
    return obj_tokens, obj_pos, int(obj_tokens.shape[0])


def _build_memory_prompt(
    frame_idx: int,
    num_frames: int,
    bank: PythonMemoryBank,
    constants: Constants,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build (prompt, prompt_pos, num_obj_ptr_tokens) for memory_attention_dynamic_k{N}.onnx.

    Replicates _prepare_memory_conditioned_features (sam3_tracker_base.py:586-792)
    for the single-object, single-conditioning-frame case used by the oracle
    (use_memory_selection=False, max_cond_frames_in_attn=4, keep_first_cond_frame=False):

      Maskmem section (variable length, no padding):
        1. Selected conditioning frames (all cond frames, since len<=max_cond):
           t = 0, idx = num_maskmem - 1.
        2. Non-conditioning frames t_pos = 1..num_maskmem-1:
           prev_frame_idx = frame_idx - (num_maskmem - t_pos); skip if absent.
           t = t_pos, idx = num_maskmem - t - 1.
        Each frame: feats (1,64,72,72) flatten -> (5184,1,64)  [value, no tpos];
                    pos   (1,64,72,72) flatten -> (5184,1,64) + maskmem_tpos_enc[idx]
                                                              [key, pos + tpos].
        tpos is added to the key (prompt_pos) only, matching tracker_base.py:659
        (prompt = feats only) and :676-678 (prompt_pos = pos + tpos).

      Object-pointer section (exact length = n_ptr_frames * 4, no padding):
        max_obj_ptrs = min(num_frames, MAX_OBJ_PTRS_IN_ENCODER).
        - selected cond frames (t <= frame_idx): rel_pos = frame_idx - t.
        - non-cond frames t_diff = 1..max_obj_ptrs-1: t = frame_idx - t_diff; rel_pos = t_diff.
        Each pointer (256-d) reshaped to 4 tokens of 64-d; pos = obj_ptr_tpos
        (256 -> 64) repeat-interleaved over the 4 tokens.
        NO padding: the obj_ptr section has exactly num_obj_ptr_tokens tokens, and the
        orchestrator selects memory_attention_dynamic_k{num_obj_ptr_tokens}.onnx so
        num_k_exclude_rope matches and num_k_rope = maskmem_len is a multiple of HW.

    Returns:
        prompt              : float32 (maskmem_len + num_obj_ptr_tokens, 1, 64)
        prompt_pos          : float32 (maskmem_len + num_obj_ptr_tokens, 1, 64)
        num_obj_ptr_tokens  : int (= n_ptr_frames * 4)
    """
    maskmem_feats, maskmem_poses = _collect_maskmem(frame_idx, bank, constants.maskmem_tpos_enc)
    if not maskmem_feats:
        raise ValueError(
            f"frame {frame_idx}: no maskmem frames available — memory_attention path "
            "must not be entered without past memory (no implicit fallback)."
        )

    obj_tokens, obj_pos, num_obj_ptr_tokens = _collect_obj_ptrs(
        frame_idx, num_frames, bank, constants
    )

    maskmem = np.concatenate(maskmem_feats, axis=0)  # (N*5184,1,64)
    maskmem_pos = np.concatenate(maskmem_poses, axis=0)
    prompt = np.concatenate([maskmem, obj_tokens], axis=0).astype(np.float32)
    prompt_pos = np.concatenate([maskmem_pos, obj_pos], axis=0).astype(np.float32)
    return prompt, prompt_pos, num_obj_ptr_tokens


# ---------------------------------------------------------------------------
# VideoOrchestrator
# ---------------------------------------------------------------------------


class VideoOrchestrator:
    """ONNX-based SAM3 video tracker (Stage C-2).

    Loads all 4 ONNX models once and runs the full memory-bank tracking loop per
    run_clip() call.

    Args:
        onnx_dir:      Directory containing the *.onnx files.
        constants_dir: Directory containing constant *.npy files.
        providers:     ORT execution providers (explicit; no auto-detect).
    """

    def __init__(
        self,
        onnx_dir: Path,
        constants_dir: Path,
        providers: list[str],
    ) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        if not onnx_dir.exists():
            raise FileNotFoundError(f"ONNX dir not found: {onnx_dir}")
        if not constants_dir.exists():
            raise FileNotFoundError(f"Constants dir not found: {constants_dir}")
        if not providers:
            raise ValueError("providers must be a non-empty list (no implicit auto-detect).")

        log.info("Loading ONNX sessions from %s (providers=%s) ...", onnx_dir, providers)
        self._providers = providers

        def _load(name: str) -> ort.InferenceSession:
            path = onnx_dir / name
            if not path.exists():
                raise FileNotFoundError(f"ONNX file not found: {path}")
            sess = ort.InferenceSession(str(path), providers=providers)
            log.info("  Loaded %s", name)
            return sess

        # NOTE: the tracker consumes the SAM2 neck features (sam2_backbone_out),
        # NOT the detector image_encoder.onnx (sam3 FPN).  image_encoder_tracker.onnx
        # is exported from build_tracker's backbone (add_sam2_neck=True).
        self._image_enc = _load("image_encoder_tracker.onnx")
        # One memory_attention graph per obj_ptr token count (num_k_exclude_rope).
        # Selected at runtime by the real obj_ptr count (no zero padding).
        self._mem_attn: dict[int, ort.InferenceSession] = {
            k: _load(f"memory_attention_dynamic_k{k}.onnx") for k in DYN_NUM_K_VALUES
        }
        self._decode = _load("decode_head.onnx")
        self._mem_enc = _load("memory_encoder.onnx")

        self._constants = Constants(constants_dir)
        log.info("VideoOrchestrator initialised (4 ONNX sessions + constants).")

    # ------------------------------------------------------------------ #
    # Per-frame pipeline steps (private)                                  #
    # ------------------------------------------------------------------ #

    def _conditioned_features(
        self,
        frame_idx: int,
        num_frames: int,
        fpn2_seq: Float[np.ndarray, "5184 1 256"],
        pos2_seq: Float[np.ndarray, "5184 1 256"],
        bank: PythonMemoryBank,
        use_mem_this_frame: bool,
    ) -> Float[np.ndarray, "1 256 72 72"]:
        """Compute pix_feat_with_mem (memory-conditioned features) for a frame.

        Mirrors _prepare_memory_conditioned_features (sam3_tracker_base.py:769-794):
        the no-memory path adds no_mem_embed to the top-level feature, while the
        memory path runs memory_attention_dynamic_k{N}.onnx over the prompt built
        from the bank.  Returns features in (B, C, H, W) layout.
        """
        if not use_mem_this_frame:
            # No-memory path: pix_feat = vision_feats[-1] + no_mem_embed
            # (sam3_tracker_base.py:769-772).  (HW,B,C) -> (B,C,H,W) via permute(1,2,0).
            pix_feat_seq = fpn2_seq + self._constants.no_mem_embed  # (5184,1,256)
            return pix_feat_seq.transpose(1, 2, 0).reshape(1, D_MODEL, FEAT_H, FEAT_W)

        prompt, prompt_pos, n_obj_tokens = _build_memory_prompt(
            frame_idx, num_frames, bank, self._constants
        )
        if n_obj_tokens not in self._mem_attn:
            raise ValueError(
                f"frame {frame_idx}: obj_ptr token count {n_obj_tokens} has no "
                f"matching memory_attention_dynamic_k graph "
                f"(available: {sorted(self._mem_attn)})."
            )
        mem_out = self._mem_attn[n_obj_tokens].run(
            None,
            {
                "src": fpn2_seq.astype(np.float32),
                "src_pos": pos2_seq.astype(np.float32),
                "prompt": prompt,
                "prompt_pos": prompt_pos,
            },
        )
        memory_seq = mem_out[0]  # (5184,1,256)
        # encoder_out["memory"].permute(1,2,0).view(B,C,H,W) (line 794)
        return memory_seq.transpose(1, 2, 0).reshape(1, D_MODEL, FEAT_H, FEAT_W)

    def _encode_new_memory(
        self,
        is_init: bool,
        is_obj_appearing: bool,
        backbone_fpn_2: Float[np.ndarray, "1 256 72 72"],
        low_res_masks: Float[np.ndarray, "1 1 288 288"],
        emulate_oracle_bf16: bool,
    ) -> tuple[Float[np.ndarray, "1 64 72 72"], Float[np.ndarray, "1 64 72 72"]]:
        """Run the memory encoder and post-process (mirrors _encode_new_memory:797-848).

        Steps 5-7 of the per-frame loop: upsample the mask, build mask_for_mem
        (binary for the prompted frame, sigmoid otherwise), run memory_encoder.onnx,
        then add no_obj_embed_spatial when the object is absent.
        """
        C = self._constants
        # high_res_masks = F.interpolate(low_res_masks, image_size) (_forward_sam_heads:353)
        high_res_masks = _upsample_bilinear(low_res_masks, SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE)
        # is_mask_from_pts = (point_inputs is not None) -> True only for frame 0.
        if is_init:
            # _encode_new_memory:822-823 (binary hard mask)
            mask_for_mem = (high_res_masks > 0).astype(np.float32)
        else:
            # _encode_new_memory:824-826 (sigmoid)
            z = np.clip(high_res_masks, -88, 88).astype(np.float32)
            mask_for_mem = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        mask_for_mem_1008 = (mask_for_mem * C.sigmoid_scale + C.sigmoid_bias).astype(np.float32)

        mem_enc_outs = self._mem_enc.run(
            None,
            {
                "pix_feat": backbone_fpn_2.astype(np.float32),
                "mask_for_mem": mask_for_mem_1008,
            },
        )
        maskmem_features = mem_enc_outs[0]  # (1,64,72,72)
        maskmem_pos_enc = mem_enc_outs[1]  # (1,64,72,72)

        # Add no_obj_embed_spatial when the object is absent (_encode_new_memory:845-848).
        is_obj_float = 1.0 if is_obj_appearing else 0.0
        noe = C.no_obj_embed_spatial[0, :, None, None]  # (64,1,1)
        maskmem_features = maskmem_features + (1.0 - is_obj_float) * noe[None]  # (1,64,72,72)

        if emulate_oracle_bf16:
            maskmem_features = _to_bf16_round(maskmem_features)
            maskmem_pos_enc = _to_bf16_round(maskmem_pos_enc)
        return maskmem_features, maskmem_pos_enc

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_clip(
        self,
        frames_pil: list[Image.Image],
        frame0_point_coords_norm: Float[np.ndarray, "1 n_pts 2"],
        frame0_point_labels: Int[np.ndarray, "1 n_pts"],
        use_memory: bool = True,
        emulate_oracle_bf16: bool = False,
    ) -> ClipResult:
        """Run ONNX video tracking on a clip and return per-frame outputs.

        Args:
            frames_pil:               List of PIL RGB frames.
            frame0_point_coords_norm: Frame-0 point prompt, shape (1, N_pts, 2),
                                      float32, normalised [0,1] (x_n, y_n).  Scaled
                                      to 1008-pixel space internally (matches
                                      add_new_points_or_box rel_coordinates=True).
            frame0_point_labels:      Labels (1, N_pts) int32 (1=pos, 0=neg).
            use_memory:               If False, frames >= 1 use the no-mem path
                                      (ablation; DoD-D-MUST-b).
            emulate_oracle_bf16:      Diagnostic only.  If True, round the stored
                                      maskmem_features / maskmem_pos_enc / obj_ptr to
                                      bfloat16 before reuse, emulating the bf16 oracle's
                                      recurrent memory storage.  Default False (the
                                      production orchestrator is full fp32).  Used to
                                      attribute the per-frame score gap to bf16/fp32.

        Returns dict with:
            masks:                          list[np.ndarray] (288, 288) bool
            scores:                         list[float] object_score_logits per frame
            obj_ids:                        list[int]  (always 1)
            low_res_mask_logits:            list[np.ndarray] (1, 288, 288) float32
            memory_attention_invoke_count:  int
        """
        bank = PythonMemoryBank()
        num_frames = len(frames_pil)

        masks: list[Bool[np.ndarray, "288 288"]] = []
        scores: list[float] = []
        obj_ids: list[int] = []
        logits_list: list[Float[np.ndarray, "1 288 288"]] = []
        mem_attn_count = 0

        point_coords_abs = (frame0_point_coords_norm * SAM3_IMAGE_SIZE).astype(np.float32)

        for frame_idx, pil_frame in enumerate(frames_pil):
            is_init = frame_idx == 0

            # --- Step 1: image encoder ------------------------------------ #
            pixel_values = _preprocess_frame(pil_frame)
            enc_outs = self._image_enc.run(None, {"pixel_values": pixel_values})
            # order: vision_pos_enc_0,_1,_2, backbone_fpn_0,_1,_2
            vision_pos_enc_2 = enc_outs[2]  # (1,256,72,72)
            backbone_fpn_0 = enc_outs[3]  # (1,256,288,288)
            backbone_fpn_1 = enc_outs[4]  # (1,256,144,144)
            backbone_fpn_2 = enc_outs[5]  # (1,256,72,72)

            # --- Step 2: conv_s0 / conv_s1 (forward_image:450-453) -------- #
            # high_res_feat0 -> (1,32,288,288); high_res_feat1 -> (1,64,144,144)
            high_res_feat0 = _conv1x1(
                backbone_fpn_0, self._constants.conv_s0_weight, self._constants.conv_s0_bias
            )
            high_res_feat1 = _conv1x1(
                backbone_fpn_1, self._constants.conv_s1_weight, self._constants.conv_s1_bias
            )

            # --- Step 3: memory conditioning (or no-mem path) ------------- #
            # fpn_2 / pos_2 flattened seq-first: (HW, B, C).
            fpn2_seq = backbone_fpn_2[0].reshape(D_MODEL, HW).T[:, None, :]  # (5184,1,256)
            pos2_seq = vision_pos_enc_2[0].reshape(D_MODEL, HW).T[:, None, :]  # (5184,1,256)

            use_mem_this_frame = use_memory and not is_init and bool(bank.cond_frame_outputs)
            pix_feat_with_mem = self._conditioned_features(
                frame_idx, num_frames, fpn2_seq, pos2_seq, bank, use_mem_this_frame
            )
            if use_mem_this_frame:
                mem_attn_count += 1

            # --- Step 4: decode_head -------------------------------------- #
            if is_init:
                pts = point_coords_abs
                lbls = frame0_point_labels
            else:
                pts = np.zeros((1, 1, 2), dtype=np.float32)
                lbls = np.array([[-1]], dtype=np.int32)

            decode_outs = self._decode.run(
                None,
                {
                    "image_embeddings": pix_feat_with_mem.astype(np.float32),
                    "high_res_feat0": high_res_feat0,
                    "high_res_feat1": high_res_feat1,
                    "point_coords": pts,
                    "point_labels": lbls,
                    "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
                    "has_mask_input": np.zeros((1,), dtype=np.float32),
                },
            )
            low_res_masks = decode_outs[0]  # (1,1,288,288), already obj-gated (B-3)
            object_score_logits = decode_outs[2]  # (1,1)
            obj_ptr = decode_outs[3]  # (1,256)

            obj_score = float(object_score_logits[0, 0])
            is_obj_appearing = obj_score > 0

            # --- Steps 5-7: build and post-process new memory ------------- #
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                is_init, is_obj_appearing, backbone_fpn_2, low_res_masks, emulate_oracle_bf16
            )
            if emulate_oracle_bf16:
                obj_ptr = _to_bf16_round(obj_ptr)

            # --- Step 8: store in bank ------------------------------------ #
            entry: FrameEntry = {
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": [maskmem_pos_enc],
                "obj_ptr": obj_ptr,
                "object_score_logits": obj_score,
            }
            if is_init:
                bank.store_cond(frame_idx, entry)
            else:
                bank.store_non_cond(frame_idx, entry)

            # --- Collect per-frame outputs -------------------------------- #
            binary_mask = low_res_masks[0, 0] > 0  # (288,288) bool — matches oracle (>0)
            masks.append(binary_mask)
            scores.append(obj_score)
            obj_ids.append(1)
            logits_list.append(low_res_masks[0].copy())

            log.info(
                "frame %d: obj_score=%.4f mask_px=%d mem_attn=%s",
                frame_idx,
                obj_score,
                int(binary_mask.sum()),
                use_mem_this_frame,
            )

        return {
            "masks": masks,
            "scores": scores,
            "obj_ids": obj_ids,
            "low_res_mask_logits": logits_list,
            "memory_attention_invoke_count": mem_attn_count,
        }
