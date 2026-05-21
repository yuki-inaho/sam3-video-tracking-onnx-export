"""SAM3 memory_attention ONNX export (B-1).

Exports TransformerEncoderCrossAttention (4 layers, self-attn + cross-attn with
cos/sin RoPE) to ONNX with fixed input shapes.

Key design decisions (mirrors image_encoder.py approach):
  - The equiv-source is loaded via importlib + sys.path injection so that
    build_tracker() picks up the patched model_builder where
    use_rope_real=True is wired through _create_tracker_transformer().
  - RoPEAttention.use_rope_real=True is set at construction time; it uses
    freqs_cis_real / freqs_cis_imag (float32) rather than freqs_cis (complex).
    No further buffer replacement is needed (unlike the ViT backbone).
  - freeze_rope_for_export() ensures that at trace time
    freqs_cis.shape[0] == q.shape[-2], so the dynamic re-compute branch in
    RoPEAttention.forward() is NOT traced (the if-body is dead code).
  - We export only TransformerEncoderCrossAttention (tracker.transformer.encoder),
    not the full Sam3TrackerPredictor, to keep the ONNX graph small.
  - The memory tokens (prompt) are padded to a fixed length (mem_len) so that
    ONNX gets static shapes throughout.
  - num_k_exclude_rope (obj_ptr count) is fixed as a Python int at export time
    and baked into the wrapper's forward; it does NOT appear as an ONNX input.
  - opset_version=18 (same as image encoder).
  - dynamo=False (TorchScript-based, stable).

Fixed-shape I/O contract:
  Input:
    src       : float32 (HW, B, 256)          current-frame vision features
    src_pos   : float32 (HW, B, 256)          positional encoding for src
    prompt    : float32 (mem_len, B, 64)       padded memory tokens
    prompt_pos: float32 (mem_len, B, 64)       pos encoding for prompt

  Output:
    memory    : float32 (HW, B, 256)          updated features after cross-attn

  Constants baked at export time:
    B=1, HW=5184, D_MODEL=256, MEM_DIM=64, num_k_exclude_rope (int)

Production memory length: 7 * 5184 + 64 = 36352 (7 frames + 16 obj_ptrs × 4
tokens each).  A smaller mem_len (e.g. 2 * 5184 = 10368) may be used for fast
parity testing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Fixed spatial token count: 1008 / 14 = 72 → 72 × 72 = 5184.
HW = 72 * 72   # 5184  spatial tokens per frame
B = 1
D_MODEL = 256
MEM_DIM = 64

# Canonical memory lengths for parity testing and full production:
#   2-frame test (fast, ~10K tokens)
TWO_FRAME_MEM_LEN: int = 2 * HW               # 10368
#   Full production: 7 maskmem frames + 16 obj_ptrs × (D_MODEL // MEM_DIM) tokens
FULL_MEM_LEN: int = 7 * HW + 16 * (D_MODEL // MEM_DIM)  # 36352

# Input / output names (used by ORT inference session)
_SRC_NAME = "src"
_SRC_POS_NAME = "src_pos"
_PROMPT_NAME = "prompt"
_PROMPT_POS_NAME = "prompt_pos"
_MEMORY_NAME = "memory"

OPSET_VERSION = 18


class MemoryAttentionWrapper(nn.Module):
    """Thin wrapper around TransformerEncoderCrossAttention for ONNX export.

    Accepts four float32 tensors (src, src_pos, prompt, prompt_pos) with fixed
    shapes and returns the updated current-frame features (memory).

    The wrapper mirrors the actual call site in sam3_tracker_base.py:
      encoder_out = self.transformer.encoder(
          src=[src_seq_first],
          src_key_padding_mask=[None],
          src_pos=[src_pos_seq_first],
          prompt=prompt,
          prompt_pos=prompt_pos_embed,
          prompt_key_padding_mask=None,
          feat_sizes=feat_sizes,
          num_obj_ptr_tokens=num_k_exclude_rope,
      )

    num_k_exclude_rope is baked in at construction time so it does not become
    an ONNX dynamic input (it is a fixed integer in the production pipeline).

    The TransformerEncoderCrossAttention.forward expects:
      - src / src_pos / src_key_padding_mask as *lists* (it unpacks [0]).
      - batch_first=True (internal transpose is handled by the module).
      - feat_sizes is passed as a kwarg (unused except to pass down, no-op here
        since the actual feat_sizes only matters for window-attention modes which
        are not present in TransformerEncoderCrossAttention).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_k_exclude_rope: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self._num_k_exclude_rope = num_k_exclude_rope

    def forward(
        self,
        src: torch.Tensor,         # (HW, B, D_MODEL) seq-first
        src_pos: torch.Tensor,     # (HW, B, D_MODEL) seq-first
        prompt: torch.Tensor,      # (mem_len, B, MEM_DIM) seq-first
        prompt_pos: torch.Tensor,  # (mem_len, B, MEM_DIM) seq-first
    ) -> torch.Tensor:
        """Forward pass through the 4-layer cross-attention encoder.

        Args:
            src:        Current-frame vision features, seq-first (HW, B, 256).
            src_pos:    Positional encoding for src, seq-first (HW, B, 256).
            prompt:     Memory tokens (padded), seq-first (M, B, 64).
            prompt_pos: Positional encoding for prompt, seq-first (M, B, 64).

        Returns:
            Updated current-frame features, seq-first (HW, B, 256).
        """
        out = self.encoder(
            src=[src],
            src_key_padding_mask=[None],
            src_pos=[src_pos],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=None,
            feat_sizes=[[HW, 1]],   # dummy; not used by TransformerEncoderCrossAttention
            num_obj_ptr_tokens=self._num_k_exclude_rope,
        )
        # TransformerEncoderCrossAttention.forward with batch_first=True:
        # Internally transposes seq-first→batch-first, processes layers, then
        # transposes back.  The returned "memory" is seq-first (HW, B, D_MODEL).
        return out["memory"]  # (HW, B, D_MODEL)


def _evict_sam3_cache() -> None:
    """Remove all sam3.* entries from sys.modules to force a clean re-import."""
    to_del = [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]
    for k in to_del:
        del sys.modules[k]


def _load_equiv_tracker_encoder(
    equiv_source_root: Path,
    checkpoint_path: Path,
) -> nn.Module:
    """Load TransformerEncoderCrossAttention from the equiv-source tracker.

    Uses the same importlib + sys.path injection pattern as image_encoder.py.
    The equiv-source model_builder has use_rope_real=True wired through
    build_tracker() → _create_tracker_transformer(), so both self-attn and
    cross-attn RoPEAttention modules are constructed with use_rope_real=True.

    Loading strategy: call build_tracker(use_rope_real=True, with_backbone=False)
    directly, then load only the tracker.* weights from the checkpoint.  This
    avoids building the large detector (ViT + text encoder) and is much faster
    than build_sam3_video_model().

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.

    Returns:
        TransformerEncoderCrossAttention in eval mode, float32, on CPU.

    Raises:
        FileNotFoundError: if either path is absent.
        RuntimeError: if any RoPEAttention has use_rope_real=False after loading.
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
        from sam3.model_builder import build_tracker  # type: ignore[import]

        log.info(
            "Building SAM3 tracker from equiv source (use_rope_real=True) ..."
        )
        tracker = build_tracker(
            apply_temporal_disambiguation=True,
            with_backbone=False,
            compile_mode=None,
            use_rope_real=True,
        )
    finally:
        if inserted and equiv_root_str in sys.path:
            sys.path.remove(equiv_root_str)

    # Load tracker weights from checkpoint (tracker.* prefix → strip prefix).
    log.info("Loading tracker weights from %s ...", checkpoint_path)
    with open(str(checkpoint_path), "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    tracker_state = {
        k[len("tracker."):]: v
        for k, v in ckpt.items()
        if k.startswith("tracker.")
    }
    if not tracker_state:
        raise RuntimeError(
            f"No 'tracker.*' keys found in {checkpoint_path}. "
            "Verify the checkpoint format."
        )
    missing, unexpected = tracker.load_state_dict(tracker_state, strict=True)
    if missing:
        raise RuntimeError(
            f"Missing keys when loading tracker weights: {missing[:10]}"
        )
    if unexpected:
        log.warning("Unexpected keys (ignored): %s", unexpected[:5])
    log.info("Tracker weights loaded: %d parameters.", len(tracker_state))

    # Extract the transformer encoder (TransformerEncoderCrossAttention).
    encoder = tracker.transformer.encoder

    # Guard: verify use_rope_real=True on all RoPEAttention modules.
    _verify_use_rope_real(encoder)

    # Move to CPU and convert to float32 for ORT CPU parity.
    # build_tracker() initialises freqs_cis on CUDA (if available).  The module's
    # .cpu() call moves registered parameters/buffers, but freqs_cis / freqs_cis_real /
    # freqs_cis_imag are plain Python attributes → must be moved explicitly.
    encoder = encoder.cpu().float().eval()
    _move_rope_attrs_to_cpu(encoder)

    log.info(
        "Loaded TransformerEncoderCrossAttention: %d layers, batch_first=%s.",
        encoder.num_layers,
        encoder.batch_first,
    )
    return encoder


def _verify_use_rope_real(module: nn.Module) -> None:
    """Raise RuntimeError if any RoPEAttention in the module has use_rope_real=False.

    This guards against the patcher not having wired use_rope_real=True for the
    tracker path, which would leave complex RoPE in the ONNX graph.

    Args:
        module: Module subtree to inspect.

    Raises:
        RuntimeError: if any RoPEAttention has use_rope_real=False.
    """
    complex_modules = []
    for name, mod in module.named_modules():
        cls_name = type(mod).__name__
        if cls_name == "RoPEAttention":
            if not getattr(mod, "use_rope_real", False):
                complex_modules.append(name)
    if complex_modules:
        raise RuntimeError(
            f"Found RoPEAttention modules with use_rope_real=False: {complex_modules}. "
            "The equiv-source patcher did not wire use_rope_real=True for the tracker. "
            "Check sam3_source_patcher.py MODEL_BUILDER_REPLACEMENTS."
        )
    log.info(
        "_verify_use_rope_real: all RoPEAttention modules have use_rope_real=True ✓"
    )


def _move_rope_attrs_to_cpu(encoder: nn.Module) -> None:
    """Move RoPEAttention plain tensor attributes to CPU.

    RoPEAttention stores freqs_cis / freqs_cis_real / freqs_cis_imag as plain
    Python attributes (not nn.Parameter or register_buffer), so .cpu() on the
    parent module does not move them.  This function explicitly moves them.

    Args:
        encoder: Module subtree containing RoPEAttention modules.
    """
    for name, mod in encoder.named_modules():
        cls_name = type(mod).__name__
        if cls_name == "RoPEAttention":
            for attr in ("freqs_cis", "freqs_cis_real", "freqs_cis_imag"):
                val = getattr(mod, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    if val.device.type != "cpu":
                        setattr(mod, attr, val.cpu())
                        log.info(
                            "_move_rope_attrs_to_cpu: %s.%s moved to CPU.", name, attr
                        )


def _freeze_rope_for_export(encoder: nn.Module) -> None:
    """Pre-validate that RoPE freqs are sized for HW=5184 tokens.

    RoPEAttention.forward() has a dynamic re-compute branch:
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(...)
            self.freqs_cis_real = self.freqs_cis.real
            self.freqs_cis_imag = self.freqs_cis.imag

    When use_rope_real=True and feat_sizes=[72, 72] (at construction), freqs_cis
    already has shape (5184, ...) which matches q.shape[-2]=5184 for self-attn.
    For cross-attn, q is the *query* (current features, 5184 tokens) and k is
    the memory (mem_len tokens); q.shape[-2]=5184 also matches freqs_cis.shape[0].

    So at trace time with our fixed shapes, the if-condition evaluates to False and
    the complex re-compute branch is NOT traced → no complex ops in the ONNX graph.

    This function logs the freqs_cis shapes so we can confirm correctness.

    Args:
        encoder: TransformerEncoderCrossAttention to inspect.
    """
    for name, mod in encoder.named_modules():
        cls_name = type(mod).__name__
        if cls_name == "RoPEAttention":
            if hasattr(mod, "freqs_cis"):
                log.info(
                    "%s: freqs_cis.shape=%s (expected (%d, ...))",
                    name, mod.freqs_cis.shape, HW,
                )
                if mod.freqs_cis.shape[0] != HW:
                    log.warning(
                        "%s: freqs_cis.shape[0]=%d != HW=%d. "
                        "The dynamic re-compute branch WILL be traced → complex ops risk.",
                        name, mod.freqs_cis.shape[0], HW,
                    )
            if hasattr(mod, "freqs_cis_real"):
                log.info(
                    "%s: freqs_cis_real.shape=%s, freqs_cis_imag.shape=%s",
                    name, mod.freqs_cis_real.shape,
                    mod.freqs_cis_imag.shape if hasattr(mod, "freqs_cis_imag") else "?",
                )


def _patch_output_dims(model_proto: "onnx.ModelProto", output_path: Path, hw: int, b: int, d_model: int) -> None:  # noqa: F821
    """Replace symbolic dim_params in graph output ValueInfo with concrete values.

    Args:
        model_proto: Loaded ONNX ModelProto (modified in-place).
        output_path: Path where the patched model is saved.
        hw: Number of spatial tokens (5184).
        b: Batch size (1).
        d_model: Feature dimension (256).
    """
    import onnx  # noqa: PLC0415

    known_shapes: dict[str, list[int]] = {
        _MEMORY_NAME: [hw, b, d_model],
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


def build_memory_attention_module(
    equiv_source_root: Path,
    checkpoint_path: Path,
    mem_len: int,
    num_k_exclude_rope: int = 0,
) -> MemoryAttentionWrapper:
    """Build and return a MemoryAttentionWrapper (no export).

    Useful for computing PyTorch reference outputs for parity checks.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.
        mem_len: Fixed memory token length (prompt dim 0).
        num_k_exclude_rope: Number of obj_ptr tokens excluded from RoPE.

    Returns:
        MemoryAttentionWrapper in eval mode, float32, on CPU.
    """
    encoder = _load_equiv_tracker_encoder(equiv_source_root, checkpoint_path)
    _freeze_rope_for_export(encoder)
    return MemoryAttentionWrapper(encoder, num_k_exclude_rope=num_k_exclude_rope).eval()


def export_memory_attention(
    equiv_source_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    mem_len: int,
    num_k_exclude_rope: int = 0,
) -> None:
    """Export the SAM3 memory_attention (TransformerEncoderCrossAttention) to ONNX.

    Idempotent: if output_path already exists and passes onnx.checker, the export
    is skipped.

    Args:
        equiv_source_root: Path to outputs/sam3_equiv_source.
        checkpoint_path: Path to models/sam3.pt.
        output_path: Destination for memory_attention.onnx.
        mem_len: Fixed memory token length (M in prompt shape (M, B, MEM_DIM)).
        num_k_exclude_rope: Number of obj_ptr tokens excluded from cross-attn RoPE.
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
                "Existing ONNX at %s failed checker (%s); re-exporting.", output_path, exc
            )
            output_path.unlink()

    wrapper = build_memory_attention_module(
        equiv_source_root, checkpoint_path, mem_len, num_k_exclude_rope
    )

    # Dummy inputs (fixed shape, float32, all zeros — model is in eval mode).
    dummy_src = torch.zeros(HW, B, D_MODEL, dtype=torch.float32)
    dummy_src_pos = torch.zeros(HW, B, D_MODEL, dtype=torch.float32)
    dummy_prompt = torch.zeros(mem_len, B, MEM_DIM, dtype=torch.float32)
    dummy_prompt_pos = torch.zeros(mem_len, B, MEM_DIM, dtype=torch.float32)

    log.info(
        "Exporting memory_attention to %s (opset %d, mem_len=%d, "
        "num_k_exclude_rope=%d) ...",
        output_path, OPSET_VERSION, mem_len, num_k_exclude_rope,
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(dummy_src, dummy_src_pos, dummy_prompt, dummy_prompt_pos),
            f=str(output_path),
            input_names=[_SRC_NAME, _SRC_POS_NAME, _PROMPT_NAME, _PROMPT_POS_NAME],
            output_names=[_MEMORY_NAME],
            opset_version=OPSET_VERSION,
            dynamo=False,
        )
    log.info("Export complete. Validating with onnx.checker ...")

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    log.info("ONNX graph is valid.")

    op_types = {node.op_type for node in model_proto.graph.node}
    log.info("Op types in memory_attention graph: %s", sorted(op_types))

    # Patch symbolic output dims to concrete values.
    _patch_output_dims(model_proto, output_path, hw=HW, b=B, d_model=D_MODEL)
