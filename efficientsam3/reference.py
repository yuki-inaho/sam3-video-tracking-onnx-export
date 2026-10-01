"""Pinned CPU reference and tensor-only ONNX boundaries."""

import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from contract import CONTEXT, SOURCE_REVISION, check_state, verify_file


def build_reference(source, checkpoint):
    source = Path(source).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_REVISION:
        raise ValueError("Reference source revision mismatch")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise ValueError("Reference source checkout has local modifications")
    verify_file(checkpoint)
    sys.path.insert(0, str(source / "sam3"))
    from sam3.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        device="cpu",
        load_from_HF=False,
        checkpoint_path=None,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=CONTEXT,
        enable_inst_interactivity=False,
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint_data["model"]
    check_state(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    return model.eval()


class Vision(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, image):
        out = self.backbone.forward_image(image)
        return tuple(out["backbone_fpn"] + out["vision_pos_enc"])


class Text(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.student = model.backbone.language_backbone

    def forward(self, tokens):
        encoder = self.student.encoder
        embeds = encoder.forward_embedding(tokens)
        memory = encoder(embeds, return_all_tokens=True, input_is_embeddings=True)
        return self.student.projector(memory).transpose(0, 1), tokens == 0


class Grounding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        from sam3.model.data_misc import FindStage

        self.find = FindStage(
            img_ids=torch.tensor([0]),
            text_ids=torch.tensor([0]),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def forward(self, f0, f1, f2, pe0, pe1, pe2, text, mask):
        out = self.model.forward_grounding(
            backbone_out={
                "backbone_fpn": [f0, f1, f2],
                "vision_pos_enc": [pe0, pe1, pe2],
                "vision_features": f2,
                "language_features": text,
                "language_mask": mask,
            },
            find_input=self.find,
            find_target=None,
            geometric_prompt=self.model._get_dummy_prompt(),
        )
        return out["pred_boxes"], out["pred_logits"], out["presence_logit_dec"], out["pred_masks"]
