"""Make an exact, auditable CPU/ONNX copy of the pinned SAM 3.1 source.

The official ``sam31`` Git submodule is never edited. Replacements fail if the
upstream lines change, so an updated release must be reviewed explicitly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sam3_onnx_equiv.sam3_source_patcher import PatchResult


def _replace(text: str, old: str, new: str, *, count: int = 1) -> str:
    found = text.count(old)
    if found != count:
        raise ValueError(
            f"SAM 3.1 source rewrite expected {count} occurrence(s), found {found}: {old!r}"
        )
    return text.replace(old, new)


def create_sam31_cpu_source_copy(source_root: Path, output_root: Path) -> PatchResult:
    """Copy the official SAM 3.1 tree and remove CPU-only CUDA assumptions."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if (
        source_root == output_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise ValueError("SAM 3.1 source and generated copy must be separate directories")
    if not (source_root / "sam3/model/video_tracking_multiplex.py").is_file():
        raise FileNotFoundError(f"Not an Object Multiplex SAM 3.1 source: {source_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(
        source_root,
        output_root,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )

    replacements: dict[str, list[tuple[str, str, int]]] = {
        "sam3/model/position_encoding.py": [
            (
                'tensors = torch.zeros((1, 1) + size, device="cuda")',
                'tensors = torch.zeros((1, 1) + size, device="cpu")',
                1,
            ),
        ],
        "sam3/model/decoder.py": [
            (
                'feat_size, feat_size, device="cuda"',
                'feat_size, feat_size, device="cpu"',
                1,
            ),
        ],
        "sam3/model_builder.py": [
            ("import pkg_resources", "from importlib.resources import files as resource_files", 1),
            (
                "pkg_resources.resource_filename(\n"
                '            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"\n'
                "        )",
                'str(resource_files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))',
                3,
            ),
            (
                "    demo_model.cuda().eval()",
                "    demo_model.cpu().eval()",
                1,
            ),
            (
                "tri_neck = _create_multiplex_tri_backbone(\n"
                '        compile_mode="max-autotune" if compile else None\n'
                "    )",
                "tri_neck = _create_multiplex_tri_backbone(\n"
                '        compile_mode="max-autotune" if compile else None,\n'
                "        use_rope_real=use_rope_real,\n"
                "    )",
                1,
            ),
        ],
        "sam3/model/memory.py": [
            ("antialias=True,", "antialias=False,", 1),
        ],
        "sam3/perflib/fused.py": [
            (
                "    if torch.is_grad_enabled():\n"
                '        raise ValueError("Expected grad to be disabled.")\n'
                "    self = linear.bias.detach()",
                "    if torch.is_grad_enabled():\n"
                '        raise ValueError("Expected grad to be disabled.")\n'
                '    if mat1.device.type != "cuda":\n'
                "        value = linear(mat1)\n"
                "        return activation()(value) if isinstance(activation, type) "
                "else activation(value)\n"
                "    self = linear.bias.detach()",
                1,
            ),
        ],
        "sam3/model/video_tracking_multiplex.py": [
            (
                "torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)",
                "torch.tensor(rel_pos_list).to(device=device, non_blocking=True)",
                1,
            ),
            (
                "feats = feats.cuda(non_blocking=True)",
                "feats = feats.to(device, non_blocking=True)",
                1,
            ),
            (
                "maskmem_enc = maskmem_enc.cuda(non_blocking=True)",
                "maskmem_enc = maskmem_enc.to(device, non_blocking=True)",
                1,
            ),
            (
                'image_feat = prev["image_features"].cuda()',
                'image_feat = prev["image_features"].to(device)',
                1,
            ),
            (
                'image_pos_embed = prev["image_pos_enc"].cuda() + tpos_enc',
                'image_pos_embed = prev["image_pos_enc"].to(device) + tpos_enc',
                1,
            ),
        ],
        "sam3/model/video_tracking_multiplex_demo.py": [
            (
                "            use_torchcodec=use_torchcodec,\n            use_cv2=use_cv2,\n",
                "",
                1,
            ),
            (
                'inference_state["device"] = torch.device("cuda")',
                'inference_state["device"] = next(self.parameters()).device',
                2,
            ),
            (
                'inference_state["storage_device"] = torch.device("cuda")',
                'inference_state["storage_device"] = inference_state["device"]',
                2,
            ),
            (
                'prev_sam_mask_logits_singleton = prev_out["pred_masks"].cuda(',
                'prev_sam_mask_logits_singleton = prev_out["pred_masks"].to('
                'inference_state["device"],',
                1,
            ),
            (
                'image = inference_state["images"][frame_idx].cuda().float().unsqueeze(0)',
                'image = inference_state["images"][frame_idx].to('
                'inference_state["device"]).float().unsqueeze(0)',
                1,
            ),
            (
                "                need_sam3_out=True,\n"
                "                need_interactive_out=True,\n"
                "                need_propagation_out=True,",
                "                need_sam3_out=False,\n"
                "                need_interactive_out=True,\n"
                "                need_propagation_out=True,",
                1,
            ),
        ],
        "sam3/model/geometry_encoders.py": [
            (
                "scale = scale.pin_memory().to(device=boxes_xyxy.device, non_blocking=True)",
                'scale = scale.pin_memory() if boxes_xyxy.device.type == "cuda" else scale\n'
                "            scale = scale.to(device=boxes_xyxy.device, non_blocking=True)",
                1,
            ),
        ],
    }

    modified: list[Path] = []
    for relative, edits in replacements.items():
        destination = output_root / relative
        text = destination.read_text(encoding="utf-8")
        for old, new, count in edits:
            text = _replace(text, old, new, count=count)
        destination.write_text(text, encoding="utf-8")
        modified.append(destination)
    return PatchResult(source_root, output_root, tuple(modified))
