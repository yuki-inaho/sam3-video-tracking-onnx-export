"""Generate an ONNX-oriented equivalent source copy for the official SAM3 tree.

This is intentionally conservative. It does not alter weights and it does not hide
unsupported branches. The patcher only rewires builder defaults so that official
RoPEAttention modules use the real-valued RoPE path that SAM3 already contains,
and it leaves checkpoint loading semantics unchanged.

Authored and validated against official SAM3 commit
84cc43bca4347b772f17d1078a1ddb4c054655c2. Exact-once replacements are expected
to fail loudly if the upstream source changes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | Path


@dataclass(frozen=True)
class PatchResult:
    """Result of creating an equivalent SAM3 source copy."""

    source_root: Path
    output_root: Path
    modified_files: tuple[Path, ...]


@dataclass(frozen=True)
class TextReplacement:
    """A single exact text rewrite with a human-readable audit label."""

    old: str
    new: str
    label: str


MODEL_BUILDER_REPLACEMENTS: tuple[TextReplacement, ...] = (
    TextReplacement(
        old=(
            "def _create_position_encoding(precompute_resolution=None):\n"
            '    """Create position encoding for visual backbone."""\n'
        ),
        new=(
            "def _create_position_encoding(precompute_resolution=None):\n"
            '    """Create position encoding for visual backbone."""\n'
            "    # The official precompute allocates its cache on CUDA; on CPU,\n"
            "    # compute the same encoding lazily on the input device.\n"
            "    if not torch.cuda.is_available():\n"
            "        precompute_resolution = None\n"
        ),
        label="position encoding: skip CUDA precompute on CPU",
    ),
    TextReplacement(
        old="        resolution=1008,\n        stride=14,\n",
        new=(
            "        # The decoder precomputes coordinates on CUDA when resolution is set.\n"
            "        resolution=1008 if torch.cuda.is_available() else None,\n"
            "        stride=14,\n"
        ),
        label="decoder: compute coordinates lazily on CPU",
    ),
    TextReplacement(
        old=(
            "        num_pos_feats=64,\n"
            "        normalize=True,\n"
            "        scale=None,\n"
            "        temperature=10000,\n"
            "        precompute_resolution=1008,\n"
        ),
        new=(
            "        num_pos_feats=64,\n"
            "        normalize=True,\n"
            "        scale=None,\n"
            "        temperature=10000,\n"
            "        precompute_resolution=1008 if torch.cuda.is_available() else None,\n"
        ),
        label="tracker memory position encoding: skip CUDA precompute on CPU",
    ),
    TextReplacement(
        old="def _create_tracker_transformer():\n",
        new="def _create_tracker_transformer(use_rope_real: bool = True):\n",
        label="_create_tracker_transformer signature",
    ),
    # NOTE: _create_vit_backbone gains use_rope_real parameter for API consistency.
    # The ViT class in vitdet.py uses its own Attention (not RoPEAttention) and does not
    # yet support use_rope_real. The parameter is accepted but not forwarded to ViT until
    # D5 (vitdet.py patch). Numerically, ViT complex RoPE == real RoPE (proven in D3).
    TextReplacement(
        old="def _create_vit_backbone(compile_mode=None):\n",
        new=(
            "def _create_vit_backbone(compile_mode=None, use_rope_real: bool = True):\n"
            "    # use_rope_real is accepted for API consistency; vitdet.py ViT Attention\n"
            "    # uses its own complex RoPE path which is numerically equivalent (D3).\n"
            "    # Forwarding to ViT is deferred to D5 (vitdet.py patch).\n"
        ),
        label="_create_vit_backbone signature",
    ),
    TextReplacement(
        old=(
            "        use_fa3=False,\n        use_rope_real=False,\n    )\n\n    # Cross attention\n"
        ),
        new=(
            "        use_fa3=False,\n"
            "        use_rope_real=use_rope_real,\n"
            "    )\n\n"
            "    # Cross attention\n"
        ),
        label="tracker self attention use_rope_real",
    ),
    TextReplacement(
        old=(
            "        rope_k_repeat=True,\n"
            "        use_fa3=False,\n"
            "        use_rope_real=False,\n"
            "    )\n\n"
            "    # Encoder layer\n"
        ),
        new=(
            "        rope_k_repeat=True,\n"
            "        use_fa3=False,\n"
            "        use_rope_real=use_rope_real,\n"
            "    )\n\n"
            "    # Encoder layer\n"
        ),
        label="tracker cross attention use_rope_real",
    ),
    TextReplacement(
        old=(
            "def build_tracker(\n"
            "    apply_temporal_disambiguation: bool, with_backbone: bool = False, "
            "compile_mode=None\n"
            ") -> Sam3TrackerPredictor:\n"
        ),
        new=(
            "def build_tracker(\n"
            "    apply_temporal_disambiguation: bool,\n"
            "    with_backbone: bool = False,\n"
            "    compile_mode=None,\n"
            "    use_rope_real: bool = True,\n"
            ") -> Sam3TrackerPredictor:\n"
        ),
        label="build_tracker signature",
    ),
    TextReplacement(
        old="    transformer = _create_tracker_transformer()\n",
        new="    transformer = _create_tracker_transformer(use_rope_real=use_rope_real)\n",
        label="build_tracker transformer construction",
    ),
    TextReplacement(
        old=(
            "def _create_vision_backbone(\n"
            "    compile_mode=None, enable_inst_interactivity=True\n"
            ") -> Sam3DualViTDetNeck:\n"
        ),
        new=(
            "def _create_vision_backbone(\n"
            "    compile_mode=None,\n"
            "    enable_inst_interactivity=True,\n"
            "    use_rope_real: bool = True,\n"
            ") -> Sam3DualViTDetNeck:\n"
        ),
        label="_create_vision_backbone signature",
    ),
    TextReplacement(
        old="    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)\n",
        new=(
            "    vit_backbone: ViT = _create_vit_backbone(\n"
            "        compile_mode=compile_mode, use_rope_real=use_rope_real\n"
            "    )\n"
        ),
        label="_create_vision_backbone vit construction",
    ),
    TextReplacement(
        old=('    compile=False,\n):\n    """\n    Build SAM3 image model\n'),
        new=(
            "    compile=False,\n"
            "    use_rope_real: bool = True,\n"
            "):\n"
            '    """\n'
            "    Build SAM3 image model\n"
        ),
        label="build_sam3_image_model signature",
    ),
    TextReplacement(
        old=(
            "        compile_mode=compile_mode, "
            "enable_inst_interactivity=enable_inst_interactivity\n"
            "    )\n"
        ),
        new=(
            "        compile_mode=compile_mode,\n"
            "        enable_inst_interactivity=enable_inst_interactivity,\n"
            "        use_rope_real=use_rope_real,\n"
            "    )\n"
        ),
        label="build_sam3_image_model vision call",
    ),
    TextReplacement(
        old=("    compile=False,\n) -> Sam3VideoInferenceWithInstanceInteractivity:\n"),
        new=(
            "    compile=False,\n"
            "    use_rope_real: bool = True,\n"
            ") -> Sam3VideoInferenceWithInstanceInteractivity:\n"
        ),
        label="build_sam3_video_model signature",
    ),
    TextReplacement(
        old=(
            "    tracker = build_tracker("
            "apply_temporal_disambiguation=apply_temporal_disambiguation)\n"
        ),
        new=(
            "    tracker = build_tracker(\n"
            "        apply_temporal_disambiguation=apply_temporal_disambiguation,\n"
            "        use_rope_real=use_rope_real,\n"
            "    )\n"
        ),
        label="build_sam3_video_model tracker call",
    ),
    TextReplacement(
        old="    visual_neck = _create_vision_backbone()\n",
        new="    visual_neck = _create_vision_backbone(use_rope_real=use_rope_real)\n",
        label="build_sam3_video_model vision call",
    ),
)


VITDET_REPLACEMENTS: tuple[TextReplacement, ...] = (
    # (1) Insert apply_rotary_enc2 (cos/sin, ONNX-safe) after apply_rotary_enc.
    #     Placed immediately before window_partition to minimise diff.
    TextReplacement(
        old=(
            "    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)\n"
            "\n"
            "\n"
            "def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:\n"
        ),
        new=(
            "    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)\n"
            "\n"
            "\n"
            "def apply_rotary_enc2(\n"
            "    xq: torch.Tensor,\n"
            "    xk: torch.Tensor,\n"
            "    freqs_cos: torch.Tensor,\n"
            "    freqs_sin: torch.Tensor,\n"
            "    repeat_freqs_k: bool = False,\n"
            ") -> Tuple[torch.Tensor, torch.Tensor]:\n"
            '    """ONNX-safe cos/sin RoPE equivalent to apply_rotary_enc (complex).\n'
            "\n"
            "    Replaces torch.polar / view_as_complex / view_as_real with real-valued\n"
            "    arithmetic.  freqs_cos = freqs_cis.real, freqs_sin = freqs_cis.imag.\n"
            '    """\n'
            "    xq_reshaped = xq.view(*xq.shape[:-1], -1, 2)\n"
            "    xq_cos = xq_reshaped[..., 0]\n"
            "    xq_sin = xq_reshaped[..., 1]\n"
            "\n"
            "    xk_reshaped = xk.view(*xk.shape[:-1], -1, 2)\n"
            "    xk_cos = xk_reshaped[..., 0]\n"
            "    xk_sin = xk_reshaped[..., 1]\n"
            "\n"
            "    freqs_cos = freqs_cos[None, None]\n"
            "    freqs_sin = freqs_sin[None, None]\n"
            "\n"
            "    if repeat_freqs_k:\n"
            "        r = xk_reshaped.shape[-3] // xq_reshaped.shape[-3]\n"
            "        freqs_cos = freqs_cos.repeat(*([1] * (freqs_cos.ndim - 2)), r, 1)\n"
            "        freqs_sin = freqs_sin.repeat(*([1] * (freqs_sin.ndim - 2)), r, 1)\n"
            "\n"
            "    xq_out = torch.stack(\n"
            "        [xq_cos * freqs_cos - xq_sin * freqs_sin,\n"
            "         xq_cos * freqs_sin + xq_sin * freqs_cos],\n"
            "        dim=-1,\n"
            "    ).flatten(3)\n"
            "    xk_out = torch.stack(\n"
            "        [xk_cos * freqs_cos - xk_sin * freqs_sin,\n"
            "         xk_cos * freqs_sin + xk_sin * freqs_cos],\n"
            "        dim=-1,\n"
            "    ).flatten(3)\n"
            "\n"
            "    return xq_out.type_as(xq), xk_out.type_as(xk)\n"
            "\n"
            "\n"
            "def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:\n"
        ),
        label="vitdet: insert apply_rotary_enc2",
    ),
    # (2) Replace _apply_rope to branch on freqs_cis vs freqs_cos/freqs_sin.
    #     When freqs_cis is present (training / normal PyTorch): complex path.
    #     When absent (ONNX export): cos/sin path via apply_rotary_enc2.
    TextReplacement(
        old=(
            "    def _apply_rope(self, q, k) -> Tuple[Tensor, Tensor]:\n"
            "        if not self.use_rope:\n"
            "            return q, k\n"
            "\n"
            "        assert self.freqs_cis is not None\n"
            "        return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)\n"
        ),
        new=(
            "    def _apply_rope(self, q, k) -> Tuple[Tensor, Tensor]:\n"
            "        if not self.use_rope:\n"
            "            return q, k\n"
            "\n"
            '        if hasattr(self, "freqs_cis"):\n'
            "            # Standard PyTorch path: complex RoPE "
            "(numerically equivalent to cos/sin).\n"
            "            assert self.freqs_cis is not None\n"
            "            return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)\n"
            "        # ONNX-export path: cos/sin RoPE (no complex ops).\n"
            "        # Activated by replacing freqs_cis buffer with freqs_cos / freqs_sin.\n"
            '        assert hasattr(self, "freqs_cos") and hasattr(self, "freqs_sin"), (\n'
            '            "_apply_rope: freqs_cis is absent but freqs_cos / freqs_sin are not "\n'
            '            "registered.  Call model.replace_rope_freqs() before export."\n'
            "        )\n"
            "        return apply_rotary_enc2(q, k, self.freqs_cos, self.freqs_sin)\n"
        ),
        label="vitdet: _apply_rope branch on freqs_cis / freqs_cos-sin",
    ),
)


MEMORY_REPLACEMENTS: tuple[TextReplacement, ...] = (
    # SimpleMaskDownSampler.forward uses F.interpolate(..., antialias=True) for the
    # 1008->1152 bilinear upsample.  antialias=True emits aten::_upsample_bilinear2d_aa,
    # which torch.onnx (opset 18, jit exporter) does not support.  The export tooling
    # (export/memory_encoder.py docstring) documents that the equiv-source must use
    # antialias=False so this op is not emitted; the numerical difference for this
    # upsample is <5e-7 (verified by the memory_encoder ORT parity test).
    TextReplacement(
        old=(
            "                align_corners=False,\n"
            '                mode="bilinear",\n'
            "                antialias=True,\n"
        ),
        new=(
            "                align_corners=False,\n"
            '                mode="bilinear",\n'
            "                antialias=False,\n"
        ),
        label="memory: SimpleMaskDownSampler antialias=False for ONNX export",
    ),
)


# The official interactive tracker assumes CUDA in a few device transfers.  Keep
# its source read-only and make the generated copy follow the model's device so
# the PyTorch oracle can run on CPU alongside ORT CPUExecutionProvider.
CPU_PREDICTOR_REPLACEMENTS: tuple[TextReplacement, ...] = (
    TextReplacement(
        old=(
            '        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)\n'
        ),
        new=(
            "        self.bf16_context = torch.autocast(\n"
            "            device_type=self.device.type, dtype=torch.bfloat16,\n"
            '            enabled=self.device.type == "cuda",\n'
            "        )\n"
        ),
        label="predictor: autocast only on CUDA",
    ),
    TextReplacement(
        old='            inference_state["storage_device"] = torch.device("cuda")\n',
        new='            inference_state["storage_device"] = self.device\n',
        label="predictor: storage device follows model",
    ),
    TextReplacement(
        old=(
            '                prev_sam_mask_logits = prev_out["pred_masks"].cuda('
            "non_blocking=True)\n"
        ),
        new=(
            '                prev_sam_mask_logits = prev_out["pred_masks"].to(\n'
            "                    self.device, non_blocking=True\n"
            "                )\n"
        ),
        label="predictor: previous mask follows model device",
    ),
    TextReplacement(
        old=(
            '                image = inference_state["images"][frame_idx].cuda().float()'
            ".unsqueeze(0)\n"
        ),
        new=(
            '                image = inference_state["images"][frame_idx].to(self.device).float()'
            ".unsqueeze(0)\n"
        ),
        label="predictor: image follows model device",
    ),
)

CPU_TRACKER_BASE_REPLACEMENTS: tuple[TextReplacement, ...] = (
    TextReplacement(
        old=(
            "            torch.tensor(rel_pos_list).pin_memory().to("
            "device=device, non_blocking=True)\n"
        ),
        new="            torch.tensor(rel_pos_list).to(device=device, non_blocking=True)\n",
        label="tracker: temporal position encoding works without CUDA pinning",
    ),
    TextReplacement(
        old='                feats = prev["maskmem_features"].cuda(non_blocking=True)\n',
        new='                feats = prev["maskmem_features"].to(device, non_blocking=True)\n',
        label="tracker: memory features follow current feature device",
    ),
    TextReplacement(
        old='                maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()\n',
        new='                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)\n',
        label="tracker: memory position follows current feature device",
    ),
)

CPU_GEOMETRY_REPLACEMENTS: tuple[TextReplacement, ...] = (
    TextReplacement(
        old=(
            "            scale = scale.pin_memory().to("
            "device=boxes_xyxy.device, non_blocking=True)\n"
        ),
        new=(
            '            if boxes_xyxy.device.type == "cuda":\n'
            "                scale = scale.pin_memory()\n"
            "            scale = scale.to(device=boxes_xyxy.device, non_blocking=True)\n"
        ),
        label="geometry: pin host scale only for CUDA transfer",
    ),
)


def _replace_once(text: str, replacement: TextReplacement) -> str:
    count = text.count(replacement.old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence for {replacement.label}, found {count}"
        )
    return text.replace(replacement.old, replacement.new, 1)


def patch_model_builder_text(text: str) -> str:
    """Patch ``sam3/model_builder.py`` text for explicit real-valued RoPE wiring."""
    for replacement in MODEL_BUILDER_REPLACEMENTS:
        text = _replace_once(text, replacement)
    return text


def patch_vitdet_text(text: str) -> str:
    """Patch ``sam3/model/vitdet.py`` to add cos/sin RoPE path (ONNX-safe)."""
    for replacement in VITDET_REPLACEMENTS:
        text = _replace_once(text, replacement)
    return text


def patch_memory_text(text: str) -> str:
    """Patch ``sam3/model/memory.py`` to use ONNX-exportable bilinear upsample.

    Switches ``SimpleMaskDownSampler.forward`` from ``antialias=True`` to
    ``antialias=False`` so that ``aten::_upsample_bilinear2d_aa`` (unsupported in
    ONNX opset 18) is not emitted during ``export-memory-encoder``.
    """
    for replacement in MEMORY_REPLACEMENTS:
        text = _replace_once(text, replacement)
    return text


def patch_cpu_text(text: str, replacements: tuple[TextReplacement, ...]) -> str:
    """Replace official CUDA assumptions with transfers to the tracker device."""
    for replacement in replacements:
        text = _replace_once(text, replacement)
    return text


def create_equivalent_source_copy(source_root: PathLike, output_root: PathLike) -> PatchResult:
    """Copy a SAM3 tree and patch ONNX and CPU oracle paths in the copy."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    builder = source_root / "sam3" / "model_builder.py"
    if not builder.exists():
        raise FileNotFoundError(f"Not a SAM3 source root: {source_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(
        source_root,
        output_root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    copied_builder = output_root / "sam3" / "model_builder.py"
    copied_builder.write_text(
        patch_model_builder_text(copied_builder.read_text()), encoding="utf-8"
    )
    copied_vitdet = output_root / "sam3" / "model" / "vitdet.py"
    copied_vitdet.write_text(patch_vitdet_text(copied_vitdet.read_text()), encoding="utf-8")
    copied_memory = output_root / "sam3" / "model" / "memory.py"
    copied_memory.write_text(patch_memory_text(copied_memory.read_text()), encoding="utf-8")
    copied_predictor = output_root / "sam3" / "model" / "sam3_tracking_predictor.py"
    copied_predictor.write_text(
        patch_cpu_text(copied_predictor.read_text(), CPU_PREDICTOR_REPLACEMENTS),
        encoding="utf-8",
    )
    copied_tracker_base = output_root / "sam3" / "model" / "sam3_tracker_base.py"
    copied_tracker_base.write_text(
        patch_cpu_text(copied_tracker_base.read_text(), CPU_TRACKER_BASE_REPLACEMENTS),
        encoding="utf-8",
    )
    copied_geometry = output_root / "sam3" / "model" / "geometry_encoders.py"
    copied_geometry.write_text(
        patch_cpu_text(copied_geometry.read_text(), CPU_GEOMETRY_REPLACEMENTS),
        encoding="utf-8",
    )
    return PatchResult(
        source_root=source_root,
        output_root=output_root,
        modified_files=(
            copied_builder,
            copied_vitdet,
            copied_memory,
            copied_predictor,
            copied_tracker_base,
            copied_geometry,
        ),
    )
