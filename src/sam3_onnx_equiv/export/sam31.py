"""Fixed-bucket SAM 3.1 Object Multiplex ONNX tensor boundaries."""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from sam3_onnx_equiv.export.image_encoder import freeze_abs_pos_for_export
from sam3_onnx_equiv.path_config import model_paths
from sam3_onnx_equiv.sam31_model import build_sam31_tracker

OPSET = 18
BUCKET_CAPACITY = 16
IMAGE_SIZE = 1008
FEATURE_SIZE = 72


class TriHeadEncoder(nn.Module):
    """Shared ViT with distinct interactive and propagation feature heads."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixels: Tensor) -> tuple[Tensor, ...]:
        output = self.model.forward_image(
            pixels,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=True,
        )
        interactive = output["interactive"]
        propagation = output["sam2_backbone_out"]
        return (
            *interactive["vision_pos_enc"],
            *(feature.tensors for feature in interactive["backbone_fpn"]),
            *propagation["vision_pos_enc"],
            *(feature.tensors for feature in propagation["backbone_fpn"]),
        )


class MultiplexMemoryAttention(nn.Module):
    """One bucket with a fixed number of previous memory frames."""

    def __init__(self, encoder: nn.Module, *, pointer_tokens: int = BUCKET_CAPACITY) -> None:
        super().__init__()
        self.encoder = encoder
        self.pointer_tokens = pointer_tokens

    def forward(
        self,
        image: Tensor,
        src: Tensor,
        memory_image: Tensor,
        memory: Tensor,
        image_pos: Tensor,
        src_pos: Tensor,
        memory_image_pos: Tensor,
        memory_pos: Tensor,
    ) -> Tensor:
        output = self.encoder(
            image=image,
            src=src,
            memory_image=memory_image,
            memory=memory,
            image_pos=image_pos,
            src_pos=src_pos,
            memory_image_pos=memory_image_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=self.pointer_tokens,
        )
        return output["memory"]


class MultiplexMaskHead(nn.Module):
    """Predict masks, quality, scores, and pointers for all 16 slots."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        high_res_0: Tensor,
        high_res_1: Tensor,
        slot_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            multimask_output=True,
            high_res_features=[high_res_0, high_res_1],
            extra_per_object_embeddings=slot_embeddings,
        )
        return (
            output["masks"],
            output["iou_pred"],
            output["object_score_logits"],
            output["sam_tokens_out"],
        )


class MultiplexMemoryEncoder(nn.Module):
    """Encode 16 mask channels plus 16 conditioning channels once per bucket."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pix_feat: Tensor, bucket_masks: Tensor) -> tuple[Tensor, Tensor]:
        output = self.encoder(pix_feat, bucket_masks, skip_mask_sigmoid=True)
        return output["vision_features"], output["vision_pos_enc"][0]


def _export(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(path))


def _export_memory(model: nn.Module, output_dir: Path) -> Path:
    path = output_dir / "multiplex_memory_encoder.onnx"
    _export(
        MultiplexMemoryEncoder(model.maskmem_backbone),
        (
            torch.zeros(1, 256, FEATURE_SIZE, FEATURE_SIZE),
            torch.zeros(1, 32, IMAGE_SIZE, IMAGE_SIZE),
        ),
        ["pix_feat", "bucket_masks"],
        ["maskmem_features", "maskmem_pos_enc"],
        path,
    )
    return path


def _export_decoder(model: nn.Module, output_dir: Path) -> Path:
    path = output_dir / "multiplex_mask_decoder.onnx"
    _export(
        MultiplexMaskHead(model.sam_mask_decoder),
        (
            torch.zeros(1, 256, FEATURE_SIZE, FEATURE_SIZE),
            torch.zeros(1, 256, FEATURE_SIZE, FEATURE_SIZE),
            torch.zeros(1, 32, 288, 288),
            torch.zeros(1, 64, 144, 144),
            torch.zeros(1, BUCKET_CAPACITY, 256),
        ),
        ["image_embeddings", "image_pe", "high_res_0", "high_res_1", "slot_embeddings"],
        ["masks", "iou_pred", "object_score_logits", "sam_tokens_out"],
        path,
    )
    return path


def _export_attention_variant(
    model: nn.Module, output_dir: Path, *, memory_frames: int, pointer_frames: int
) -> Path:
    suffix = (
        "" if (memory_frames, pointer_frames) == (1, 1) else f"_m{memory_frames}_p{pointer_frames}"
    )
    path = output_dir / f"multiplex_memory_attention{suffix}.onnx"
    seq = FEATURE_SIZE * FEATURE_SIZE
    memory_seq = memory_frames * seq
    pointer_tokens = pointer_frames * BUCKET_CAPACITY
    inputs = (
        torch.zeros(seq, 1, 256),
        torch.zeros(seq, 1, 256),
        torch.zeros(memory_seq, 1, 256),
        torch.zeros(memory_seq + pointer_tokens, 1, 256),
        torch.zeros(seq, 1, 256),
        torch.zeros(seq, 1, 256),
        torch.zeros(memory_seq, 1, 256),
        torch.zeros(memory_seq + pointer_tokens, 1, 256),
    )
    _export(
        MultiplexMemoryAttention(model.transformer.encoder, pointer_tokens=pointer_tokens),
        inputs,
        [
            "image",
            "src",
            "memory_image",
            "memory",
            "image_pos",
            "src_pos",
            "memory_image_pos",
            "memory_pos",
        ],
        ["conditioned_features"],
        path,
    )
    return path


def _export_image(model: nn.Module, output_dir: Path) -> Path:
    path = output_dir / "trihead_image_encoder.onnx"
    freeze_abs_pos_for_export(model.backbone.vision_backbone.trunk)
    _export(
        TriHeadEncoder(model),
        (torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),),
        ["pixels"],
        [
            *(f"interactive_pos_{index}" for index in range(3)),
            *(f"interactive_fpn_{index}" for index in range(3)),
            *(f"propagation_pos_{index}" for index in range(3)),
            *(f"propagation_fpn_{index}" for index in range(3)),
        ],
        path,
    )
    return path


def export_sam31(*, only: str | None = None, output_dir: Path | None = None) -> list[Path]:
    """Export the four tensor boundaries and bounded attention length variants."""
    exporters = {
        "memory_encoder": _export_memory,
        "mask_decoder": _export_decoder,
        "image_encoder": _export_image,
    }
    attention_variants = {
        "memory_attention": (1, 1),
        "memory_attention_m1_p2": (1, 2),
        "memory_attention_m2_p1": (2, 1),
        "memory_attention_m2_p2": (2, 2),
    }
    if only is not None and only not in exporters and only not in attention_variants:
        raise ValueError(f"Unknown SAM 3.1 export module: {only}")
    output_dir = output_dir or model_paths("sam31").onnx_dir
    model = build_sam31_tracker(use_rope_real=True)
    names = [*exporters, *attention_variants] if only is None else [only]
    paths = []
    for name in names:
        if name in exporters:
            paths.append(exporters[name](model, output_dir))
        else:
            memory_frames, pointer_frames = attention_variants[name]
            paths.append(
                _export_attention_variant(
                    model, output_dir, memory_frames=memory_frames, pointer_frames=pointer_frames
                )
            )
    return paths
