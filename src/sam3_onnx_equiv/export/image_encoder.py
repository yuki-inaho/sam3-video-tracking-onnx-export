"""SAM3 image encoder ONNX export (D5-2, with static-shape fix D5-3).

Exports the ViT backbone + Sam3DualViTDetNeck (image feature pyramid + positional
encodings) to ONNX with a fixed input shape of (1, 3, 1008, 1008).

Key design decisions:
  - The equiv-source is loaded as the primary sam3 package (sys.path insertion)
    so that the two-branch Attention._apply_rope is active.
  - replace_rope_freqs() converts all freqs_cis (complex) buffers to
    freqs_cos / freqs_sin (real) before export.  This eliminates
    torch.polar / view_as_complex / view_as_real from the ONNX graph.
  - freeze_abs_pos_for_export() (D5-3) precomputes the absolute positional
    embedding for the fixed 1008² resolution and replaces the learnable
    pos_embed parameter in-place.  This removes the dynamic `get_abs_pos`
    branching (the ViT `If` node) from the ONNX graph, eliminating the ORT
    MergeShapeInfo error that caused intermittent NaN in backbone_fpn outputs.
  - opset_version=18 (ONNX opset 18 supports all required ops).
  - dynamo=False (TorchScript-based export, stable for this graph).
  - Input name: pixel_values  (float32, (1,3,1008,1008), pre-normalised).
  - Output names: vision_pos_enc_0..2, backbone_fpn_0..2 (matching wkentaro recipe).

Output contract (matches wkentaro/sam3@onnx export_onnx.py backbone split):
  vision_pos_enc_i : (1, 256, H_i, W_i)  positional encodings per FPN level
  backbone_fpn_i   : (1, 256, H_i, W_i)  feature maps per FPN level
  Actual measured at H=W=1008: [288,288], [144,144], [72,72] per level.

Root cause of D5-3 NaN:
  get_abs_pos (vitdet.py) contains `if size != h or size != w` which became an
  ONNX `If` node with condition hard-coded to False during tracing (TorchScript
  evaluated the comparison at trace time as False because the test was run with a
  size that matched -- race condition with module caching).  ORT's constant folding
  then inlined the *else* branch, which produces a 4-D tensor, conflicting with the
  MergeShapeInfo that expected a 5-D result from the *then* branch, causing NaN.

  Fix: precompute pos_embed to the target (72×72) grid so `size == h == w` at
  trace time; the else branch is trivially a reshape and generates no `If` node.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from sam3_onnx_equiv.export._equiv_loader import (
    equiv_sam3_on_path,
    evict_sam3_cache,
    extract_prefixed_state,
    load_checkpoint,
    patch_output_dims,
)
from sam3_onnx_equiv.export.rope_freqs import replace_rope_freqs

# Re-exported for backward compatibility (tests import _evict_sam3_cache via the
# equiv loaders; keep the historical name available from this module too).
_evict_sam3_cache = evict_sam3_cache

log = logging.getLogger(__name__)

# Fixed input shape for the image encoder (H=W=1008 resolution)
INPUT_SHAPE: tuple[int, int, int, int] = (1, 3, 1008, 1008)

# Patch size used by the ViT backbone (1008 / 14 = 72, confirmed at runtime).
# This is the spatial resolution at which the ViT trunk operates.
_PATCH_STRIDE = 14  # pixels per patch edge
_VIT_H = INPUT_SHAPE[2] // _PATCH_STRIDE  # 72
_VIT_W = INPUT_SHAPE[3] // _PATCH_STRIDE  # 72

# Input/output names (following wkentaro recipe)
INPUT_NAME = "pixel_values"
OUTPUT_NAMES = [
    "vision_pos_enc_0",
    "vision_pos_enc_1",
    "vision_pos_enc_2",
    "backbone_fpn_0",
    "backbone_fpn_1",
    "backbone_fpn_2",
]

# Static output shapes at 1008×1008 input (FPN level 0 = finest), used by
# patch_output_dims to replace symbolic output dims with concrete values.
_KNOWN_OUTPUT_SHAPES: dict[str, list[int]] = {
    "vision_pos_enc_0": [1, 256, 288, 288],
    "vision_pos_enc_1": [1, 256, 144, 144],
    "vision_pos_enc_2": [1, 256, 72, 72],
    "backbone_fpn_0": [1, 256, 288, 288],
    "backbone_fpn_1": [1, 256, 144, 144],
    "backbone_fpn_2": [1, 256, 72, 72],
}

OPSET_VERSION = 18


class ImageEncoderWrapper(nn.Module):
    """Thin wrapper around SAM3VLBackbone._forward_image_no_act_ckpt.

    Accepts a pre-normalised float32 image tensor of shape (1, 3, H, W) and
    returns the six FPN feature maps and positional encodings as a flat tuple.

    Output order:
        vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2,
        backbone_fpn_0,   backbone_fpn_1,   backbone_fpn_2

    Pre-processing (resize + normalise) is intentionally kept outside this
    wrapper so that it can be done in Python without going through ONNX.
    """

    def __init__(self, backbone: nn.Module, use_sam2_neck: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        # The detector path uses the top-level sam3 FPN (vision_pos_enc/backbone_fpn).
        # The TRACKER path uses the SAM2 neck output (sam2_backbone_out), which is
        # produced by Sam3DualViTDetNeck.sam2_convs when add_sam2_neck=True.  These
        # are the features tracker.forward_image (sam3_tracker_base.py:447) consumes
        # via backbone.forward_image(...)["sam2_backbone_out"].
        self.use_sam2_neck = use_sam2_neck

    def forward(self, pixel_values: Float[Tensor, "1 3 1008 1008"]) -> tuple[Tensor, ...]:
        out = self.backbone._forward_image_no_act_ckpt(pixel_values)
        if self.use_sam2_neck:
            sam2 = out["sam2_backbone_out"]
            if sam2 is None:
                raise RuntimeError(
                    "use_sam2_neck=True but sam2_backbone_out is None — the backbone "
                    "was built without add_sam2_neck (enable_inst_interactivity=False)."
                )
            vision_pos_enc = sam2["vision_pos_enc"]
            backbone_fpn = sam2["backbone_fpn"]
        else:
            vision_pos_enc = out["vision_pos_enc"]
            backbone_fpn = out["backbone_fpn"]
        assert len(vision_pos_enc) == 3, f"Expected 3 pos enc levels, got {len(vision_pos_enc)}"
        assert len(backbone_fpn) == 3, f"Expected 3 FPN levels, got {len(backbone_fpn)}"
        return (*vision_pos_enc, *backbone_fpn)


def _load_equiv_sam3_model(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> nn.Module:
    """Load the SAM3 image model from the equiv-source with use_rope_real=True.

    The equiv-source is inserted at the front of sys.path so that sam3.model.vitdet
    resolves to the patched version with the two-branch _apply_rope.  After loading,
    the path insertion is removed to avoid side-effects on subsequent imports.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.

    Returns:
        Loaded SAM3 image model (eval mode, on CPU).

    Raises:
        FileNotFoundError: if equiv_source_root or checkpoint_path are absent.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with equiv_sam3_on_path(equiv_source_root):
        from sam3.model_builder import build_sam3_image_model  # type: ignore[import]

        log.info("Loading SAM3 image model from equiv source (use_rope_real=True) ...")
        model = build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            use_rope_real=True,
        )

    log.info("Model loaded.  Applying rope freqs replacement to backbone ...")
    replaced = replace_rope_freqs(model.backbone)
    log.info("replace_rope_freqs: replaced %d freqs_cis buffers in backbone", replaced)

    # Full-model guard: verify no freqs_cis remain anywhere (includes backbone).
    fc_count = sum(1 for n, _ in model.named_buffers() if "freqs_cis" in n)
    fcos_count = sum(1 for n, _ in model.named_buffers() if "freqs_cos" in n)
    log.info("freqs_cis buffers remaining (full model): %d  (should be 0)", fc_count)
    log.info("freqs_cos buffers registered (full model): %d", fcos_count)
    if fc_count != 0:
        raise RuntimeError(
            f"replace_rope_freqs: {fc_count} freqs_cis buffers remain in full model.  "
            "ONNX export would include complex ops."
        )

    # Precompute pos_embed to eliminate get_abs_pos dynamic If node (D5-3 fix).
    trunk = model.backbone.vision_backbone.trunk
    freeze_abs_pos_for_export(trunk)

    return model


def _resize_pos_embed_grid(
    abs_pos: torch.Tensor, size: int, h: int, w: int, tiling: bool
) -> torch.Tensor:
    """Expand a (1, size², C) positional grid to (1, h*w, C) like get_abs_pos.

    Args:
        abs_pos: Spatial pos-embed tokens, (1, size², C).
        size:    Source grid edge length.
        h, w:    Target grid height / width.
        tiling:  If True, tile-then-crop; otherwise bicubic interpolate.

    Returns:
        (1, h*w, C) token sequence.
    """
    grid = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)  # (1, C, size, size)
    if tiling:
        grid = grid.tile([1, 1] + [x // y + 1 for x, y in zip((h, w), grid.shape[2:])])[
            :, :, :h, :w
        ]
    else:
        grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, h * w, -1)  # (1, h*w, C)


def _verify_pos_embed_size(
    new_pos_embed: torch.Tensor, has_cls_token: bool, h: int, w: int
) -> None:
    """Raise if the precomputed pos-embed grid does not match the target (h, w).

    Ensures get_abs_pos will take its trivial else-branch (no ONNX ``If`` node).
    """
    abs_pos_new = new_pos_embed[:, 1:] if has_cls_token else new_pos_embed
    new_size = int(math.sqrt(abs_pos_new.shape[1]))
    if new_size != h or new_size != w:
        raise RuntimeError(
            f"freeze_abs_pos_for_export: after precompute, new_size={new_size} "
            f"still != target ({h}, {w}).  Check that h*w is a perfect square."
        )


def freeze_abs_pos_for_export(trunk: nn.Module, h: int = _VIT_H, w: int = _VIT_W) -> None:
    """Precompute ViT absolute positional embedding for a fixed spatial grid.

    Replaces ``trunk.pos_embed`` (shape ``(1, 1+size², C)``) in-place with a
    precomputed version at the target resolution ``(1, 1+h*w, C)``.  After this
    call ``get_abs_pos`` will see ``size == h == w`` and take the trivial
    ``else`` branch (a reshape only), which produces **no** ONNX ``If`` node.

    Without this fix the TorchScript exporter traces ``get_abs_pos`` with a
    boolean condition that is hard-coded to ``False`` (else-branch) even though
    the actual runtime value is ``True`` (then-branch / tiling).  ORT's constant
    folding then inlines the wrong branch and raises a ``MergeShapeInfo`` error
    that propagates NaN through the backbone FPN outputs.

    Scope:
        Applies only to the ViT trunk's ``pos_embed`` parameter.  The tracker's
        ``RoPEAttention`` is unaffected (no ``pos_embed`` buffer).

    Args:
        trunk: ViT module with a ``pos_embed`` parameter and the attributes
               ``tile_abs_pos``, ``pretrain_use_cls_token``, ``retain_cls_token``.
        h: Target spatial height (default: 1008 // 14 = 72).
        w: Target spatial width  (default: 1008 // 14 = 72).

    Raises:
        AttributeError: if trunk does not have a ``pos_embed`` parameter.
        RuntimeError: if the precomputed embedding shape does not match h and w.
    """
    if not hasattr(trunk, "pos_embed") or trunk.pos_embed is None:
        log.info("freeze_abs_pos_for_export: trunk has no pos_embed — skipping.")
        return

    has_cls_token: bool = getattr(trunk, "pretrain_use_cls_token", False)
    retain_cls: bool = getattr(trunk, "retain_cls_token", False)
    tiling: bool = getattr(trunk, "tile_abs_pos", False)

    abs_pos_full = trunk.pos_embed.data  # (1, N, C)
    if has_cls_token:
        cls_pos = abs_pos_full[:, :1]  # (1, 1, C)
        abs_pos = abs_pos_full[:, 1:]  # (1, size², C)
    else:
        cls_pos = None
        abs_pos = abs_pos_full

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    if size == h and size == w:
        log.info(
            "freeze_abs_pos_for_export: pos_embed already at target (%d×%d) — skipping.",
            h,
            w,
        )
        return

    log.info(
        "freeze_abs_pos_for_export: resizing pos_embed from %d×%d to %d×%d (tiling=%s).",
        size,
        size,
        h,
        w,
        tiling,
    )

    # Expand pos_embed to (h, w) grid using the same logic as get_abs_pos.
    spatial_tokens = _resize_pos_embed_grid(abs_pos, size, h, w, tiling)  # (1, h*w, C)

    # Reassemble with optional cls_token prefix.
    if has_cls_token and cls_pos is not None and not retain_cls:
        new_pos_embed = torch.cat([cls_pos, spatial_tokens], dim=1)  # (1, 1+h*w, C)
    else:
        new_pos_embed = spatial_tokens

    _verify_pos_embed_size(new_pos_embed, has_cls_token, h, w)

    # Replace the parameter data in-place (preserves requires_grad etc.).
    trunk.pos_embed = nn.Parameter(new_pos_embed.detach(), requires_grad=False)
    log.info(
        "freeze_abs_pos_for_export: pos_embed replaced → shape %s.",
        tuple(trunk.pos_embed.shape),
    )


def build_image_encoder_module(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> ImageEncoderWrapper:
    """Build and return an ImageEncoderWrapper (no export).

    Useful for computing PyTorch reference outputs for parity checks.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.

    Returns:
        ImageEncoderWrapper in eval mode on CPU.
    """
    model = _load_equiv_sam3_model(equiv_source_root, checkpoint_path)
    return ImageEncoderWrapper(model.backbone).eval()


def export_image_encoder(
    equiv_source_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Export the SAM3 image encoder to ONNX.

    Idempotent: if output_path already exists and has no `If` nodes (i.e. the
    D5-3 freeze_abs_pos fix has been applied), the export is skipped and the
    existing file is validated with onnx.checker instead.  If an `If` node is
    found (pre-D5-3 export), the old file is deleted and re-exported.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.
        output_path: Destination for image_encoder.onnx.
    """
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        model_proto = onnx.load(str(output_path))
        onnx.checker.check_model(model_proto)
        # D5-3 idempotency check: re-export only if the pre-D5-3 ONNX still
        # contains an `If` node from get_abs_pos (the root cause of NaN).
        # Symbolic dims from Shape/Range/Mod ops are harmless in terms of NaN
        # and are handled by onnxsim in _ensure_static_shape() in the tests.
        if_count = sum(1 for n in model_proto.graph.node if n.op_type == "If")
        if if_count == 0:
            log.info(
                "ONNX file already exists with no `If` nodes (D5-3 fix applied) — skipping export.",
            )
            return
        log.info(
            "Existing ONNX at %s has %d `If` node(s) (pre-D5-3).  Re-exporting ...",
            output_path,
            if_count,
        )
        output_path.unlink()

    wrapper = build_image_encoder_module(equiv_source_root, checkpoint_path)

    dummy_input = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)

    log.info("Exporting image encoder to %s (opset %d) ...", output_path, OPSET_VERSION)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(dummy_input,),
            f=str(output_path),
            input_names=[INPUT_NAME],
            output_names=OUTPUT_NAMES,
            opset_version=OPSET_VERSION,
            dynamo=False,
        )
    log.info("Export complete.  Validating with onnx.checker ...")

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    log.info("ONNX graph is valid.")

    op_types = {node.op_type for node in model_proto.graph.node}
    log.info("Op types in graph: %s", sorted(op_types))

    # Patch graph output type metadata: replace symbolic dim_params with
    # concrete dim_values.  The actual computation graph (nodes/tensors)
    # already uses fixed shapes at 1008² resolution; only the ValueInfo
    # output descriptors still contain symbolic names from the exporter.
    # Patching them here avoids the cost of onnxsim on a ~1.8GB model.
    # Concrete shapes (at 1008×1008 input) verified by ORT inference after export.
    patch_output_dims(model_proto, output_path, _KNOWN_OUTPUT_SHAPES)


# ---------------------------------------------------------------------------
# Tracker image encoder (SAM2 neck) — used by the video orchestrator (C-2)
# ---------------------------------------------------------------------------


def _load_equiv_tracker_backbone(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> nn.Module:
    """Load the SAM3 *tracker* backbone (add_sam2_neck=True) from the equiv source.

    build_tracker(with_backbone=True) builds Sam3DualViTDetNeck with
    enable_inst_interactivity=True (model_builder.py:446,499) so the neck has a
    sam2_convs branch.  tracker.forward_image consumes the SAM2 neck output
    (backbone.forward_image(...)["sam2_backbone_out"], sam3_tracker_base.py:447),
    NOT the top-level sam3 FPN.  The detector image_encoder.onnx exported from
    build_sam3_image_model (add_sam2_neck=False) returns the wrong features for
    the tracker — hence this dedicated loader.

    Checkpoint key mapping mirrors tools/run_pytorch_video.py:_load_tracker:
      tracker.*            -> strip "tracker."
      detector.backbone.*  -> "backbone.*"  (shared backbone weights)

    Returns:
        SAM3VLBackbone (tracker.backbone) in eval mode, float32, CPU, with
        complex RoPE removed and pos_embed frozen for ONNX export.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with equiv_sam3_on_path(equiv_source_root):
        from sam3.model_builder import build_tracker  # type: ignore[import]

        log.info("Building SAM3 tracker backbone from equiv source (use_rope_real=True) ...")
        tracker = build_tracker(
            apply_temporal_disambiguation=False,
            with_backbone=True,
            compile_mode=None,
            use_rope_real=True,
        )

    ckpt = load_checkpoint(checkpoint_path)
    # tracker.* weights plus the shared backbone stored under detector.backbone.*.
    state: dict[str, torch.Tensor] = extract_prefixed_state(ckpt, "tracker.")
    state.update(
        {k[len("detector.") :]: v for k, v in ckpt.items() if k.startswith("detector.backbone.")}
    )
    missing, unexpected = tracker.load_state_dict(state, strict=False)
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        raise RuntimeError(
            f"Missing backbone keys when loading tracker backbone: {backbone_missing[:10]}"
        )
    log.info("Tracker loaded (missing=%d, unexpected=%d).", len(missing), len(unexpected))

    backbone = tracker.backbone.cpu().float().eval()

    # Remove complex RoPE from the ViT trunk (same as detector path).
    replaced = replace_rope_freqs(backbone)
    log.info("replace_rope_freqs: replaced %d freqs_cis buffers in tracker backbone", replaced)
    fc_count = sum(1 for n, _ in backbone.named_buffers() if "freqs_cis" in n)
    if fc_count != 0:
        raise RuntimeError(
            f"replace_rope_freqs: {fc_count} freqs_cis buffers remain in tracker backbone."
        )

    # Freeze abs pos to remove the get_abs_pos If node (D5-3 fix).
    trunk = backbone.vision_backbone.trunk
    freeze_abs_pos_for_export(trunk)
    return backbone


def build_tracker_image_encoder_module(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> ImageEncoderWrapper:
    """Build an ImageEncoderWrapper that returns the SAM2 neck features (tracker path)."""
    backbone = _load_equiv_tracker_backbone(equiv_source_root, checkpoint_path)
    return ImageEncoderWrapper(backbone, use_sam2_neck=True).eval()


def export_tracker_image_encoder(
    equiv_source_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Export the SAM3 tracker image encoder (SAM2 neck) to ONNX.

    Output contract matches image_encoder.onnx (6 outputs, same names/shapes) but
    the FPN values come from the SAM2 neck (sam2_backbone_out), which is what the
    tracker decode/memory path expects.

    Idempotent: skipped if a valid ONNX with no `If` nodes already exists.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path:   Path to models/sam3.pt.
        output_path:       Destination (outputs/onnx/image_encoder_tracker.onnx).
    """
    import onnx  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        model_proto = onnx.load(str(output_path))
        onnx.checker.check_model(model_proto)
        if_count = sum(1 for n in model_proto.graph.node if n.op_type == "If")
        if if_count == 0:
            log.info("Tracker image encoder already exists with no `If` nodes — skipping.")
            return
        log.info("Existing tracker image encoder has %d `If` node(s); re-exporting.", if_count)
        output_path.unlink()

    wrapper = build_tracker_image_encoder_module(equiv_source_root, checkpoint_path)
    dummy_input = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)

    log.info("Exporting tracker image encoder to %s (opset %d) ...", output_path, OPSET_VERSION)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(dummy_input,),
            f=str(output_path),
            input_names=[INPUT_NAME],
            output_names=OUTPUT_NAMES,
            opset_version=OPSET_VERSION,
            dynamo=False,
        )
    log.info("Export complete.  Validating with onnx.checker ...")
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    op_types = {node.op_type for node in model_proto.graph.node}
    forbidden = {"ComplexFloat", "Complex", "Polar"}
    found = sorted(op_types & forbidden)
    if found:
        raise RuntimeError(f"Complex ops found in tracker image encoder: {found}")
    log.info("Op types (%d): no complex ops. %s", len(op_types), sorted(op_types))
    patch_output_dims(model_proto, output_path, _KNOWN_OUTPUT_SHAPES)
