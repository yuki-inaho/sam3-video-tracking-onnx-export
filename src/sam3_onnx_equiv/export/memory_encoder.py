"""SAM3 memory_encoder ONNX export (B-2).

Exports SimpleMaskEncoder (tracker's maskmem_backbone) to ONNX with fixed
input shapes using the established equiv-source importlib + sys.path pattern.

Architecture (sam3/model/memory.py:158-202, model_builder.py:330-363):
  mask_downsampler: SimpleMaskDownSampler
    - F.interpolate bilinear 1008→1152 (antialias=False in equiv-source patch)
    - Conv2d(1,4,k=3,s=2,p=1) + LayerNorm2d + GELU  × 2 layers (stride 4)
    - Conv2d(16,256,k=1)  (embed projection)
    → output: (B, 256, 144, 144) → stride-2 again → (B, 256, 72, 72)
  pix_feat_proj: Conv2d(256,256,k=1)
  fuser: SimpleFuser (CXBlock × 2, depthwise 7×7 + GELU + residual)
  out_proj: Conv2d(256,64,k=1)
  position_encoding: PositionEmbeddingSine(num_pos_feats=64, temperature=10000)
    → output: (B, 64, 72, 72)

Key design decisions:
  - Loads equiv-source (not official sam3) via importlib + sys.path injection
    and cleans up afterwards (same pattern as memory_attention.py).
  - Only tracker weights (maskmem_backbone.*) are loaded from models/sam3.pt;
    this avoids building the heavy ViT backbone.
  - SimpleMaskDownSampler.forward in equiv-source uses antialias=False so that
    aten::_upsample_bilinear2d_aa (unsupported in opset 18) is not emitted.
    Numerical difference vs antialias=True for the upsample 1008→1152: <5e-7.
  - PositionEmbeddingSine cache is pre-populated for (72,72) before tracing;
    the cache-hit path returns a constant tensor (avoids dynamic torch.arange).
  - skip_mask_sigmoid=True is baked into MemoryEncoderWrapper.forward because
    the production call site (sam3_tracker_base.py:835) always passes
    skip_mask_sigmoid=True (sigmoid is applied externally).
  - Two outputs are returned as a flat tuple (ONNX does not support dict/list
    outputs): maskmem_features (B,64,72,72) and maskmem_pos_enc (B,64,72,72).
  - opset_version=18, dynamo=False.
  - _patch_output_dims() sets concrete static dim values in graph output
    ValueInfo metadata (same approach as image_encoder.py / memory_attention.py).

Fixed-shape I/O contract:
  Input:
    pix_feat     : float32 (1, 256, 72, 72)   vision features from image encoder
    mask_for_mem : float32 (1,   1, 1008, 1008) high-res mask (pre-sigmoid)

  Output:
    maskmem_features : float32 (1, 64, 72, 72) memory features
    maskmem_pos_enc  : float32 (1, 64, 72, 72) spatial positional encoding
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Fixed spatial dimensions.
B = 1
IN_DIM = 256      # pix_feat channel count
OUT_DIM = 64      # maskmem_features / pos_enc channel count
FEAT_H = FEAT_W = 72
MASK_H = MASK_W = 1008

# ONNX opset.
OPSET_VERSION = 18

# I/O names (used by ORT inference session).
_PIX_FEAT_NAME = "pix_feat"
_MASK_NAME = "mask_for_mem"
_FEATURES_NAME = "maskmem_features"
_POS_ENC_NAME = "maskmem_pos_enc"


class MemoryEncoderWrapper(nn.Module):
    """Thin wrapper around SimpleMaskEncoder for ONNX export.

    Accepts two float32 tensors (pix_feat, mask_for_mem) with fixed shapes and
    returns a tuple of (maskmem_features, maskmem_pos_enc) — both float32.

    skip_mask_sigmoid=True is baked in: the production caller
    (Sam3TrackerBase._encode_new_memory, sam3_tracker_base.py:835) applies
    sigmoid externally and passes skip_mask_sigmoid=True.  Baking it avoids a
    TracerWarning about Python boolean control flow.
    """

    def __init__(self, memory_encoder: nn.Module) -> None:
        super().__init__()
        self.memory_encoder = memory_encoder

    def forward(
        self,
        pix_feat: torch.Tensor,      # (B, IN_DIM, FEAT_H, FEAT_W)
        mask_for_mem: torch.Tensor,  # (B, 1, MASK_H, MASK_W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through SimpleMaskEncoder with skip_mask_sigmoid=True.

        Args:
            pix_feat:     Vision features, float32 (1, 256, 72, 72).
            mask_for_mem: High-res mask logits (pre-sigmoid), float32
                          (1, 1, 1008, 1008).

        Returns:
            Tuple of:
              maskmem_features : float32 (1, 64, 72, 72)
              maskmem_pos_enc  : float32 (1, 64, 72, 72)
        """
        out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        features = out["vision_features"]          # (B, 64, 72, 72)
        pos_enc = out["vision_pos_enc"][0]          # (B, 64, 72, 72)
        return features, pos_enc


def _evict_sam3_cache() -> None:
    """Remove all sam3.* entries from sys.modules to force a clean re-import."""
    to_del = [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]
    for k in to_del:
        del sys.modules[k]


def _load_equiv_memory_encoder(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> nn.Module:
    """Load SimpleMaskEncoder from the equiv-source tracker.

    Uses importlib + sys.path injection (same pattern as memory_attention.py).
    Only maskmem_backbone.* weights are loaded from the checkpoint, which is
    much faster than building the full model.

    The equiv-source's SimpleMaskDownSampler.forward uses antialias=False so
    that aten::_upsample_bilinear2d_aa is not emitted during ONNX export.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.

    Returns:
        SimpleMaskEncoder in eval mode, float32, on CPU.

    Raises:
        FileNotFoundError: if either path is absent.
        RuntimeError: if maskmem_backbone.* keys are not found in checkpoint.
    """
    if not equiv_source_root.exists():
        raise FileNotFoundError(f"Equiv source not found: {equiv_source_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    equiv_root_str = str(equiv_source_root.resolve())
    inserted = False
    if equiv_root_str not in sys.path:
        sys.path.insert(0, equiv_root_str)
        inserted = True

    _evict_sam3_cache()

    try:
        from sam3.model_builder import _create_tracker_maskmem_backbone  # type: ignore[import]

        log.info("Building SAM3 memory encoder (SimpleMaskEncoder) from equiv source ...")
        memory_encoder = _create_tracker_maskmem_backbone()
    finally:
        if inserted and equiv_root_str in sys.path:
            sys.path.remove(equiv_root_str)

    # Load maskmem_backbone weights from checkpoint (strip prefix).
    log.info("Loading maskmem_backbone weights from %s ...", checkpoint_path)
    with open(str(checkpoint_path), "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    prefix = "tracker.maskmem_backbone."
    maskmem_state = {
        k[len(prefix):]: v
        for k, v in ckpt.items()
        if k.startswith(prefix)
    }
    if not maskmem_state:
        raise RuntimeError(
            f"No '{prefix}*' keys found in {checkpoint_path}. "
            "Verify the checkpoint format (expected tracker.maskmem_backbone.*)."
        )
    missing, unexpected = memory_encoder.load_state_dict(maskmem_state, strict=True)
    if missing:
        raise RuntimeError(
            f"Missing keys when loading maskmem_backbone weights: {missing[:10]}"
        )
    if unexpected:
        log.warning("Unexpected keys (ignored): %s", unexpected[:5])
    log.info(
        "maskmem_backbone weights loaded: %d parameters.", len(maskmem_state)
    )

    memory_encoder = memory_encoder.cpu().float().eval()

    # Pre-populate the PositionEmbeddingSine cache for (FEAT_H, FEAT_W) = (72, 72).
    # The cache-hit path in PositionEmbeddingSine.forward returns a constant tensor
    # indexed by (x.shape[-2], x.shape[-1]).  With a fixed-shape trace input, the
    # cache key evaluates to (72, 72) as a concrete Python tuple → the if-branch
    # that uses torch.arange (dynamic) is NOT traced.
    dummy_feat = torch.zeros(B, OUT_DIM, FEAT_H, FEAT_W)
    with torch.no_grad():
        _ = memory_encoder.position_encoding(dummy_feat)
    log.info(
        "PositionEmbeddingSine cache pre-populated for (%d, %d): %s",
        FEAT_H, FEAT_W,
        (FEAT_H, FEAT_W) in memory_encoder.position_encoding.cache,
    )

    return memory_encoder


def _patch_output_dims(model_proto: "onnx.ModelProto", output_path: Path) -> None:  # noqa: F821
    """Replace symbolic dim_params in graph output ValueInfo with concrete values.

    Args:
        model_proto: Loaded ONNX ModelProto (modified in-place).
        output_path: Path where the patched model is saved.
    """
    import onnx  # noqa: PLC0415

    known_shapes: dict[str, list[int]] = {
        _FEATURES_NAME: [B, OUT_DIM, FEAT_H, FEAT_W],
        _POS_ENC_NAME:  [B, OUT_DIM, FEAT_H, FEAT_W],
    }

    patched = 0
    for out in model_proto.graph.output:
        if out.name not in known_shapes:
            continue
        shape = known_shapes[out.name]
        for i, d in enumerate(out.type.tensor_type.shape.dim):
            if d.HasField("dim_param"):
                d.ClearField("dim_param")
                d.dim_value = shape[i]
                patched += 1

    if patched:
        log.info(
            "_patch_output_dims: patched %d symbolic dims → concrete values.", patched
        )
        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, str(output_path))
        log.info("Patched model saved to %s.", output_path)
    else:
        log.info(
            "_patch_output_dims: all output dims already concrete — no patch needed."
        )


def build_memory_encoder_module(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> MemoryEncoderWrapper:
    """Build and return a MemoryEncoderWrapper (no export).

    Useful for computing PyTorch reference outputs for parity checks.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.

    Returns:
        MemoryEncoderWrapper in eval mode, float32, on CPU.
    """
    memory_encoder = _load_equiv_memory_encoder(equiv_source_root, checkpoint_path)
    return MemoryEncoderWrapper(memory_encoder).eval()


def export_memory_encoder(
    equiv_source_root: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    """Export the SAM3 memory_encoder (SimpleMaskEncoder) to ONNX.

    Idempotent: if output_path already exists and passes onnx.checker, export
    is skipped.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.
        output_path: Destination for memory_encoder.onnx.
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
                output_path, exc,
            )
            output_path.unlink()

    wrapper = build_memory_encoder_module(equiv_source_root, checkpoint_path)

    # Dummy inputs (fixed shape, float32).  mask_for_mem uses sigmoid=True (baked
    # into wrapper), so zeros are valid dummy values for the trace.
    dummy_pix_feat = torch.zeros(B, IN_DIM, FEAT_H, FEAT_W, dtype=torch.float32)
    dummy_mask = torch.zeros(B, 1, MASK_H, MASK_W, dtype=torch.float32)

    log.info(
        "Exporting memory_encoder to %s (opset %d) ...", output_path, OPSET_VERSION
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(dummy_pix_feat, dummy_mask),
            f=str(output_path),
            input_names=[_PIX_FEAT_NAME, _MASK_NAME],
            output_names=[_FEATURES_NAME, _POS_ENC_NAME],
            opset_version=OPSET_VERSION,
            dynamo=False,
        )
    log.info("Export complete. Validating with onnx.checker ...")

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    log.info("ONNX graph is valid.")

    op_types = {node.op_type for node in model_proto.graph.node}
    log.info("Op types in memory_encoder graph: %s", sorted(op_types))

    forbidden = {"ComplexFloat", "Complex", "Polar", "ViewAsComplex", "ViewAsReal", "If"}
    found = {op for op in op_types if op.lower() in {f.lower() for f in forbidden}}
    if found:
        raise RuntimeError(
            f"Forbidden ops found in memory_encoder ONNX graph: {found}. "
            "Check antialias patch and PositionEmbeddingSine cache pre-population."
        )

    _patch_output_dims(model_proto, output_path)
