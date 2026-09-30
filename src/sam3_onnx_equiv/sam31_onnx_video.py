"""Run official SAM 3.1 bucket and video state with ONNX tensor modules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from sam3_onnx_equiv.path_config import model_paths


def configure_sam31_onnx_window(model: torch.nn.Module) -> None:
    """Limit temporal context to one previous mask memory and two pointer frames.

    This keeps the official tracker and its weights intact while making the
    decoupled attention input shapes stable across an arbitrary-length clip.
    The same configuration must be used for a PyTorch reference comparison.
    """
    model.num_maskmem = 2
    model.max_obj_ptrs_in_encoder = 2


class Sam31OnnxSessions:
    """CPU ONNX sessions for the four fixed, 16-slot Object Multiplex modules.

    The official Python tracker keeps prompt handling, bucket assignment, mask
    selection, and the temporal memory bank. Only its tensor-heavy calls are
    replaced. Call ``configure_sam31_onnx_window`` before installing sessions.
    """

    FILES = {
        "image": "trihead_image_encoder.onnx",
        "attention": "multiplex_memory_attention.onnx",
        "attention_m1_p2": "multiplex_memory_attention_m1_p2.onnx",
        "attention_m2_p1": "multiplex_memory_attention_m2_p1.onnx",
        "attention_m2_p2": "multiplex_memory_attention_m2_p2.onnx",
        "decoder": "multiplex_mask_decoder.onnx",
        "memory": "multiplex_memory_encoder.onnx",
    }

    def __init__(
        self, onnx_dir: Path | None = None, *, threads: int = 8, capture_attention: bool = False
    ) -> None:
        onnx_dir = onnx_dir or model_paths("sam31").onnx_dir
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.sessions = {
            name: ort.InferenceSession(
                str(onnx_dir / filename),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for name, filename in self.FILES.items()
        }
        self.calls: Counter[str] = Counter()
        self.capture_attention = capture_attention
        self.first_attention_inputs: dict[str, torch.Tensor] | None = None
        self.first_attention_output: torch.Tensor | None = None
        self.first_attention_name: str | None = None

    def run(self, name: str, tensors: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Validate the static boundary and return CPU float32 tensors."""
        session = self.sessions[name]
        feed = {
            info.name: tensors[info.name].detach().cpu().contiguous().numpy()
            for info in session.get_inputs()
        }
        for info in session.get_inputs():
            actual = feed[info.name].shape
            expected = tuple(info.shape)
            if actual != expected:
                raise ValueError(
                    f"SAM 3.1 {name} input {info.name}: expected {expected}, got {actual}; "
                    f"all inputs={ {key: value.shape for key, value in feed.items()} }"
                )
        result = session.run(None, feed)
        self.calls[name] += 1
        if name.startswith("attention_"):
            self.calls["attention"] += 1
        return [torch.from_numpy(np.asarray(item)) for item in result]

    def run_buckets(
        self,
        name: str,
        tensors: dict[str, torch.Tensor],
        *,
        bucket_axis: int,
        bucket_count: int,
    ) -> list[torch.Tensor]:
        """Use the one-bucket graph for each independent official bucket."""
        bucket_outputs = []
        for bucket in range(bucket_count):
            inputs = {}
            for key, value in tensors.items():
                size = value.shape[bucket_axis]
                if size == bucket_count:
                    inputs[key] = value.narrow(bucket_axis, bucket, 1)
                elif size == 1:
                    inputs[key] = value
                else:
                    raise ValueError(
                        f"SAM 3.1 {name} {key}: bucket axis {bucket_axis} has size {size}, "
                        f"expected 1 or {bucket_count}"
                    )
            bucket_outputs.append(self.run(name, inputs))
        return [torch.cat(items, dim=bucket_axis) for items in zip(*bucket_outputs, strict=True)]

    def install(self, model: torch.nn.Module) -> None:
        """Route official tracker calls through ORT while retaining official state code."""
        if model.num_maskmem != 2 or model.max_obj_ptrs_in_encoder != 2:
            raise ValueError("Configure the SAM 3.1 bounded ONNX memory window first")

        def forward_image(
            _: torch.nn.Module,
            img_batch: Any,
            *,
            need_sam3_out: bool = False,
            need_interactive_out: bool = False,
            need_propagation_out: bool = False,
        ) -> dict[str, Any]:
            if need_sam3_out:
                raise ValueError("SAM 3.1 tracker-only ONNX image graph has no detector head")
            pixels = img_batch.tensors if hasattr(img_batch, "tensors") else img_batch
            values = self.run("image", {"pixels": pixels})
            nested_type = type(img_batch) if hasattr(img_batch, "tensors") else None

            def neck(start: int) -> dict[str, Any]:
                features = values[start + 3 : start + 6]
                return {
                    "vision_pos_enc": values[start : start + 3],
                    "backbone_fpn": [
                        nested_type(feature, None) if nested_type else feature
                        for feature in features
                    ],
                }

            result: dict[str, Any] = {}
            if need_interactive_out:
                result["interactive"] = neck(0)
            if need_propagation_out:
                result["sam2_backbone_out"] = neck(6)
            return result

        def memory_attention(
            _: torch.nn.Module,
            *,
            image: torch.Tensor,
            src: torch.Tensor,
            memory_image: torch.Tensor,
            memory: torch.Tensor,
            image_pos: torch.Tensor,
            src_pos: torch.Tensor,
            memory_image_pos: torch.Tensor,
            memory_pos: torch.Tensor,
            num_obj_ptr_tokens: int,
        ) -> dict[str, torch.Tensor]:
            memory_frames, remainder = divmod(memory_image.shape[0], 72 * 72)
            pointer_frames, pointer_remainder = divmod(num_obj_ptr_tokens, 16)
            if (
                remainder
                or pointer_remainder
                or memory_frames not in (1, 2)
                or pointer_frames not in (1, 2)
            ):
                raise ValueError(
                    "SAM 3.1 bounded attention requires 1 or 2 memory/pointer frames; "
                    f"got memory={memory_image.shape[0]}, pointers={num_obj_ptr_tokens}"
                )
            attention_name = (
                "attention"
                if (memory_frames, pointer_frames) == (1, 1)
                else f"attention_m{memory_frames}_p{pointer_frames}"
            )
            tensors = {
                "image": image,
                "src": src,
                "memory_image": memory_image,
                "memory": memory,
                "image_pos": image_pos,
                "src_pos": src_pos,
                "memory_image_pos": memory_image_pos,
                "memory_pos": memory_pos,
            }
            output = self.run_buckets(
                attention_name,
                tensors,
                bucket_axis=1,
                bucket_count=src.shape[1],
            )
            if self.capture_attention and self.first_attention_inputs is None:
                self.first_attention_inputs = {key: value.clone() for key, value in tensors.items()}
                self.first_attention_output = output[0].clone()
                self.first_attention_name = attention_name
            return {"memory": output[0]}

        def mask_decoder(
            _: torch.nn.Module,
            *,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            high_res_features: list[torch.Tensor],
            multimask_output: bool,
            extra_per_object_embeddings: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            if not multimask_output:
                raise ValueError("SAM 3.1 mask graph requires multimask_output=True")
            output = self.run_buckets(
                "decoder",
                {
                    "image_embeddings": image_embeddings,
                    "image_pe": image_pe,
                    "high_res_0": high_res_features[0],
                    "high_res_1": high_res_features[1],
                    "slot_embeddings": extra_per_object_embeddings,
                },
                bucket_axis=0,
                bucket_count=image_embeddings.shape[0],
            )
            return dict(
                zip(
                    ("masks", "iou_pred", "object_score_logits", "sam_tokens_out"),
                    output,
                    strict=True,
                )
            )

        def memory_encoder(
            _: torch.nn.Module,
            pix_feat: torch.Tensor,
            masks: torch.Tensor,
            skip_mask_sigmoid: bool = False,
        ) -> dict[str, Any]:
            if not skip_mask_sigmoid:
                raise ValueError("SAM 3.1 memory graph requires preprocessed masks")
            features, position = self.run_buckets(
                "memory",
                {"pix_feat": pix_feat, "bucket_masks": masks},
                bucket_axis=0,
                bucket_count=masks.shape[0],
            )
            return {"vision_features": features, "vision_pos_enc": [position]}

        setattr(model, "forward_image", MethodType(forward_image, model))
        attention_module = model.transformer.encoder
        decoder_module = model.sam_mask_decoder
        memory_module = model.maskmem_backbone
        setattr(attention_module, "forward", MethodType(memory_attention, attention_module))
        setattr(decoder_module, "forward", MethodType(mask_decoder, decoder_module))
        setattr(memory_module, "forward", MethodType(memory_encoder, memory_module))
