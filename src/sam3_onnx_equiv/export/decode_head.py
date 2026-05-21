"""SAM3 decode head ONNX export (B-3).

Exports the per-frame mask/score/obj_ptr generator:
  prompt_encoder + mask_decoder + obj_ptr_proj (object pointer MLP).

This is the decode head used in _forward_sam_heads (sam3_tracker_base.py:218).

Architecture (sam3_tracker_base.py:177-216, sam/mask_decoder.py, sam/prompt_encoder.py):
  sam_prompt_encoder: PromptEncoder
    - PE layer for point and image encoding
    - mask_downscaling: Conv2d cascade (1→16→256 via k=2,s=2 convs)
  sam_mask_decoder: MaskDecoder
    - TwoWayTransformer (standard Attention, NO RoPE)
    - dynamic_multimask_via_stability=True (data-dependent torch.where — OK for ONNX)
    - use_high_res_features=True, pred_obj_scores=True
  obj_ptr_proj: MLP(256→256→256, 3 layers)
  no_obj_ptr: Parameter(1, 256) — constant blended when object absent

RoPE analysis: TwoWayTransformer uses standard nn.Attention (not RoPEAttention).
  No replace_rope_freqs needed. No complex ops expected.

Key design decisions:
  - Loads equiv-source via importlib + sys.path injection (same pattern as B-1/B-2).
  - Only tracker.sam_prompt_encoder.*, tracker.sam_mask_decoder.*,
    tracker.obj_ptr_proj.*, tracker.no_obj_ptr weights loaded.
  - multimask_output=False baked into wrapper forward (avoids Python if-branch).
  - has_mask_input treated as float32 scalar (1,) for blending mask embeds,
    matching ryouchinsa TrackerMaskDecoderWrapper approach.
  - high_res_features already conv_s0/s1-projected (not raw backbone_fpn).
  - image_pe = prompt_encoder.get_dense_pe() computed once inside forward.
  - is_obj_appearing = (object_score_logits > 0) — data-dependent torch.where, no If.
  - _patch_output_dims() sets concrete static shapes in graph output ValueInfo.
  - opset_version=18, dynamo=False.

Fixed-shape I/O contract:
  Inputs:
    image_embeddings  : float32 (1, 256, 72, 72)
    high_res_feat0    : float32 (1, 32, 288, 288)   conv_s0-projected FPN level 0
    high_res_feat1    : float32 (1, 64, 144, 144)   conv_s1-projected FPN level 1
    point_coords      : float32 (1, 1, 2)            absolute pixel coords (x,y)
    point_labels      : int32   (1, 1)               1=pos,0=neg,-1=pad
    mask_input        : float32 (1, 1, 288, 288)     prior mask logits
    has_mask_input    : float32 (1,)                 1.0 if mask_input valid

  Outputs:
    low_res_masks         : float32 (1, 1, 288, 288)
    iou_scores            : float32 (1, 1)
    object_score_logits   : float32 (1, 1)
    obj_ptr               : float32 (1, 256)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from sam3_onnx_equiv.export._equiv_loader import (
    equiv_sam3_on_path,
    evict_sam3_cache,
    load_checkpoint,
    patch_output_dims,
)

log = logging.getLogger(__name__)

# Fixed dimensions.
B = 1
EMB_DIM = 256  # image_embeddings channel dim
EMB_H = EMB_W = 72  # spatial resolution (1008 / 14)
HRF0_C = 32  # high_res_feat0 channels (conv_s0: 256 // 8)
HRF0_H = HRF0_W = 288  # high_res_feat0 spatial (72 * 4)
HRF1_C = 64  # high_res_feat1 channels (conv_s1: 256 // 4)
HRF1_H = HRF1_W = 144  # high_res_feat1 spatial (72 * 2)
N_POINTS = 1  # fixed number of point prompts
MASK_IN_H = MASK_IN_W = 288  # mask_input spatial (4 * 72)

OPSET_VERSION = 18

# I/O names (used by ORT inference session).
_INPUT_NAMES = [
    "image_embeddings",
    "high_res_feat0",
    "high_res_feat1",
    "point_coords",
    "point_labels",
    "mask_input",
    "has_mask_input",
]
_OUTPUT_NAMES = [
    "low_res_masks",
    "iou_scores",
    "object_score_logits",
    "obj_ptr",
]

# Known static output shapes for _patch_output_dims.
_KNOWN_SHAPES: dict[str, list[int]] = {
    "low_res_masks": [B, 1, MASK_IN_H, MASK_IN_W],
    "iou_scores": [B, 1],
    "object_score_logits": [B, 1],
    "obj_ptr": [B, EMB_DIM],
}


class DecodeHeadWrapper(nn.Module):
    """Thin wrapper for ONNX export of SAM3 decode head.

    Combines prompt_encoder, mask_decoder, and obj_ptr_proj into a single
    traceable forward pass with fixed input shapes and multimask_output=False.

    Args:
        prompt_encoder: PromptEncoder from equiv-source tracker.
        mask_decoder:   MaskDecoder from equiv-source tracker.
        obj_ptr_proj:   MLP(256→256→256, 3 layers) for object pointer.
        no_obj_ptr:     Parameter (1, 256) — blended when object absent.
    """

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        obj_ptr_proj: nn.Module,
        no_obj_ptr: torch.Tensor,
        multimask_output: bool = True,
    ) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.obj_ptr_proj = obj_ptr_proj
        # multimask_output mirrors _use_multimask (sam3_tracker_base.py:1108).
        # The oracle (build_tracker config: multimask_output_in_sam=True,
        # multimask_output_for_tracking=True, min/max_pt=0/1) returns True for EVERY
        # frame (frame0=1pt, frames>=1=0pt), so we bake True here and replicate the
        # best-IoU selection from _forward_sam_heads:344-368.
        self._multimask_output = multimask_output
        # Register as buffer so it travels with state_dict and stays on device.
        self.register_buffer("no_obj_ptr", no_obj_ptr)

    def forward(
        self,
        image_embeddings: Float[Tensor, "1 256 72 72"],  # (B, 256, 72, 72)
        high_res_feat0: Float[Tensor, "1 32 288 288"],  # conv_s0-projected
        high_res_feat1: Float[Tensor, "1 64 144 144"],  # conv_s1-projected
        point_coords: Float[Tensor, "1 n 2"],  # (B, N, 2)  float32
        point_labels: Int[Tensor, "1 n"],  # (B, N)     int32
        mask_input: Float[Tensor, "1 1 288 288"],  # (B, 1, 288, 288) prior mask logits
        # jaxtyping single-axis shape "b"; quotes are required (noqa: UP037 false positive).
        has_mask_input: Float[Tensor, "b"],  # noqa: UP037  # (1,) float32; 1.0 if mask valid
    ) -> tuple[
        Float[Tensor, "1 1 288 288"],
        Float[Tensor, "1 1"],
        Float[Tensor, "1 1"],
        Float[Tensor, "1 256"],
    ]:
        """Forward through decode head.

        multimask_output=False is baked in to avoid Python if-branch in ONNX graph.
        has_mask_input blends mask_embed and no_mask_embed analogous to
        ryouchinsa TrackerMaskDecoderWrapper.

        Returns:
            low_res_masks        : (B, 1, 288, 288)
            iou_scores           : (B, 1)
            object_score_logits  : (B, 1)
            obj_ptr              : (B, 256)
        """
        batch_size = image_embeddings.shape[0]

        # --- Sparse embeddings: point prompt ---
        # _embed_points pads one extra point internally when pad=True.
        sparse_embeddings = self.prompt_encoder._embed_points(
            point_coords, point_labels, pad=True
        )  # (B, N+1, 256)

        # --- Dense embeddings: blended mask / no-mask ---
        # mask_embed expects float32 (B, 1, H, W) downscaled mask.
        # has_mask_input is (1,) float32 → broadcast multiply.
        mask_embed = self.prompt_encoder.mask_downscaling(mask_input)  # (B, 256, 72, 72)
        # no_mask_embed: (1, 256, 1, 1) → expanded to (B, 256, 72, 72)
        no_mask_embed = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, EMB_H, EMB_W
        )
        # Blend: if has_mask=1.0 → use mask_embed; else → use no_mask_embed.
        has = has_mask_input.view(1, 1, 1, 1)  # broadcast shape
        dense_embeddings = has * mask_embed + (1.0 - has) * no_mask_embed  # (B, 256, 72, 72)

        # --- Image positional encoding ---
        # get_dense_pe() returns (1, 256, 72, 72) — no dynamic ops.
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, 256, 72, 72)

        # --- MaskDecoder forward ---
        # When multimask_output=True, mask_decoder returns the 3 multimask tokens
        # (masks[:,1:], iou[:,1:]) and sam_output_tokens=mask_tokens_out[:,1:] (B,3,256).
        (
            low_res_multimasks,
            iou_scores,
            sam_output_tokens,
            object_score_logits,
        ) = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self._multimask_output,  # Python literal — single branch traced
            repeat_image=False,
            high_res_features=[high_res_feat0, high_res_feat1],
        )

        if self._multimask_output:
            # Best-IoU selection (mirrors _forward_sam_heads:361-368).
            # low_res_multimasks: (B, 3, 288, 288); iou_scores: (B, 3);
            # sam_output_tokens: (B, 3, 256).
            best_iou_inds = torch.argmax(iou_scores, dim=-1)  # (B,)
            batch_inds = torch.arange(low_res_multimasks.size(0), device=low_res_multimasks.device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(
                1
            )  # (B,1,288,288)
            iou_out = iou_scores[batch_inds, best_iou_inds].unsqueeze(1)  # (B,1)
            sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]  # (B,256)
        else:
            low_res_masks = low_res_multimasks  # (B, 1, 288, 288)
            iou_out = iou_scores  # (B, 1)
            sam_output_token = sam_output_tokens[:, 0]  # (B, 256)

        # --- Object pointer ---
        is_obj_appearing = (object_score_logits > 0).float()  # (B, 1)
        obj_ptr = self.obj_ptr_proj(sam_output_token)  # (B, 256)
        # Blend: obj present → proj(token), absent → no_obj_ptr
        obj_ptr = (
            is_obj_appearing * obj_ptr + (1.0 - is_obj_appearing) * self.no_obj_ptr
        )  # (B, 256)

        return low_res_masks, iou_out, object_score_logits, obj_ptr


def _load_equiv_decode_head(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> DecodeHeadWrapper:
    """Load decode head components from the equiv-source tracker.

    Loads sam_prompt_encoder, sam_mask_decoder, obj_ptr_proj, no_obj_ptr
    from the checkpoint using tracker.* prefix keys.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path:   Path to models/sam3.pt.

    Returns:
        DecodeHeadWrapper in eval mode, float32, on CPU.

    Raises:
        FileNotFoundError: if either path is absent.
        RuntimeError:      if required tracker.* keys are not found.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with equiv_sam3_on_path(equiv_source_root):
        from sam3.model_builder import build_tracker  # type: ignore[import]

        log.info("Building SAM3 tracker (decode head) from equiv source ...")
        # Build a minimal tracker (no backbone) to get the decode head modules.
        tracker = build_tracker(
            apply_temporal_disambiguation=False,
            with_backbone=False,
            use_rope_real=True,
        )
    # Drop the freshly imported equiv sam3.* modules so later imports start clean.
    evict_sam3_cache()

    tracker = tracker.cpu().float().eval()

    # --- Load weights from checkpoint ---
    log.info("Loading tracker decode head weights from %s ...", checkpoint_path)
    ckpt = load_checkpoint(checkpoint_path)

    # Prefixes to load (map checkpoint prefix → module attribute on tracker).
    prefix_to_attr = {
        "tracker.sam_prompt_encoder.": "sam_prompt_encoder",
        "tracker.sam_mask_decoder.": "sam_mask_decoder",
        "tracker.obj_ptr_proj.": "obj_ptr_proj",
    }

    for ckpt_prefix, attr_name in prefix_to_attr.items():
        state = {k[len(ckpt_prefix) :]: v for k, v in ckpt.items() if k.startswith(ckpt_prefix)}
        if not state:
            raise RuntimeError(
                f"No '{ckpt_prefix}*' keys found in {checkpoint_path}. Verify checkpoint format."
            )
        module = getattr(tracker, attr_name)
        missing, unexpected = module.load_state_dict(state, strict=True)
        if missing:
            raise RuntimeError(f"Missing keys for {attr_name}: {missing[:10]}")
        if unexpected:
            log.warning("Unexpected keys for %s (ignored): %s", attr_name, unexpected[:5])
        log.info("Loaded %d keys for %s.", len(state), attr_name)

    # Load no_obj_ptr (it's a Parameter on the tracker itself).
    no_obj_ptr_key = "tracker.no_obj_ptr"
    if no_obj_ptr_key not in ckpt:
        raise RuntimeError(f"Key '{no_obj_ptr_key}' not found in {checkpoint_path}.")
    no_obj_ptr = ckpt[no_obj_ptr_key].cpu().float()  # (1, 256)
    log.info("Loaded no_obj_ptr: shape=%s", no_obj_ptr.shape)

    wrapper = (
        DecodeHeadWrapper(
            prompt_encoder=tracker.sam_prompt_encoder,
            mask_decoder=tracker.sam_mask_decoder,
            obj_ptr_proj=tracker.obj_ptr_proj,
            no_obj_ptr=no_obj_ptr,
        )
        .cpu()
        .float()
        .eval()
    )

    return wrapper


def build_decode_head_module(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> DecodeHeadWrapper:
    """Build and return a DecodeHeadWrapper (no export).

    Useful for computing PyTorch reference outputs for parity checks.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path:   Path to models/sam3.pt.

    Returns:
        DecodeHeadWrapper in eval mode, float32, on CPU.
    """
    return _load_equiv_decode_head(equiv_source_root, checkpoint_path)


def export_decode_head(
    equiv_source_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Export the SAM3 decode head to ONNX.

    Exports prompt_encoder + mask_decoder + obj_ptr_proj with fixed input
    shapes and multimask_output=False baked in.

    Idempotent: if output_path already exists and passes onnx.checker, export
    is skipped.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path:   Path to models/sam3.pt.
        output_path:       Destination for decode_head.onnx.
    """
    import onnx  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            existing = onnx.load(str(output_path))
            onnx.checker.check_model(existing)
            log.info(
                "ONNX file already exists and is valid — skipping export: %s",
                output_path,
            )
            return
        except Exception as exc:
            log.warning(
                "Existing ONNX at %s failed checker (%s); re-exporting.",
                output_path,
                exc,
            )
            output_path.unlink()

    wrapper = _load_equiv_decode_head(equiv_source_root, checkpoint_path)

    # Dummy fixed-shape inputs (float32).
    dummy_image_embeddings = torch.zeros(B, EMB_DIM, EMB_H, EMB_W, dtype=torch.float32)
    dummy_high_res_feat0 = torch.zeros(B, HRF0_C, HRF0_H, HRF0_W, dtype=torch.float32)
    dummy_high_res_feat1 = torch.zeros(B, HRF1_C, HRF1_H, HRF1_W, dtype=torch.float32)
    dummy_point_coords = torch.zeros(B, N_POINTS, 2, dtype=torch.float32)
    dummy_point_labels = torch.ones(B, N_POINTS, dtype=torch.int32)
    dummy_mask_input = torch.zeros(B, 1, MASK_IN_H, MASK_IN_W, dtype=torch.float32)
    dummy_has_mask = torch.zeros(1, dtype=torch.float32)  # no mask by default

    dummy_inputs = (
        dummy_image_embeddings,
        dummy_high_res_feat0,
        dummy_high_res_feat1,
        dummy_point_coords,
        dummy_point_labels,
        dummy_mask_input,
        dummy_has_mask,
    )

    log.info("Exporting decode_head to %s (opset %d) ...", output_path, OPSET_VERSION)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=dummy_inputs,
            f=str(output_path),
            input_names=_INPUT_NAMES,
            output_names=_OUTPUT_NAMES,
            opset_version=OPSET_VERSION,
            dynamo=False,
        )
    log.info("Export complete. Validating with onnx.checker ...")

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    log.info("ONNX graph is valid.")

    op_types = {node.op_type for node in model_proto.graph.node}
    log.info("Op types in decode_head graph: %s", sorted(op_types))

    forbidden = {"ComplexFloat", "Complex", "Polar", "ViewAsComplex", "ViewAsReal", "If"}
    found = {op for op in op_types if op.lower() in {f.lower() for f in forbidden}}
    if found:
        raise RuntimeError(
            f"Forbidden ops found in decode_head ONNX graph: {found}. "
            "Check multimask_output baking and dynamic_multimask_via_stability."
        )

    patch_output_dims(model_proto, output_path, _KNOWN_SHAPES)
