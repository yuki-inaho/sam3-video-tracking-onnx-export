"""ViT RoPE freqs replacement helper: freqs_cis -> freqs_cos / freqs_sin.

SCOPE
-----
This module targets the **ViT image encoder backbone** (`model.backbone`).
In SAM3 there are two independent RoPE implementations:

  1. ViT backbone (``sam3/model/vitdet.py`` ``Attention``):
     Uses ``freqs_cis`` (complex) buffers by default.  This helper converts
     them to real-valued ``freqs_cos``/``freqs_sin`` pairs so the ONNX-safe
     branch of the equiv-source ``Attention._apply_rope`` is taken.
     Call: ``replace_rope_freqs(model.backbone)``

  2. Tracker / memory_attention (``sam3/sam/transformer.py`` ``RoPEAttention``):
     Uses the ``use_rope_real`` constructor flag.  When ``use_rope_real=True``
     (set by the equiv-source patcher) the module calls
     ``apply_rotary_enc_real`` (cos/sin) instead of the complex path.
     **This helper does NOT touch ``RoPEAttention`` modules** -- their
     real-path is already activated at construction time.

Usage for image encoder export (apply to backbone only, not the full model):
    replace_rope_freqs(model.backbone)

Residual guard (image_encoder.py): after the call, a scan over
``model.named_buffers()`` (full model) verifies that zero ``freqs_cis``
buffers remain anywhere.  The guard exists to catch accidental use in a
context where the backbone is not the only source of ``freqs_cis``
(e.g. future multi-component models).  For the image encoder this guard
is always satisfied because backbone IS the only scope that carries
``freqs_cis`` buffers.

This mirrors the wkentaro/sam3@onnx recipe in infer_torch.py::get_replace_freqs_cis,
adapted for the equiv-source Attention class that has the two-branch _apply_rope.
"""

from __future__ import annotations

import torch.nn as nn


def replace_rope_freqs(root: nn.Module) -> int:
    """Replace freqs_cis (complex) buffers with freqs_cos / freqs_sin (real) in-place.

    Recurses over all sub-modules of *root*.  Any module that has a
    ``freqs_cis`` buffer receives two new buffers derived from its real and
    imaginary parts, and the original complex buffer is deleted.

    After this call:
      - module.freqs_cis  -> absent
      - module.freqs_cos  -> freqs_cis.real  (float32)
      - module.freqs_sin  -> freqs_cis.imag  (float32)

    This triggers the ONNX-safe branch in the equiv-source
    ``Attention._apply_rope``.

    Scope note:
        Pass ``model.backbone`` (ViT image encoder) rather than the full
        model to limit the replacement to ViT Attention modules.
        The tracker's ``RoPEAttention`` (``sam3/sam/transformer.py``) does
        NOT carry ``freqs_cis`` buffers; it uses ``use_rope_real=True``
        instead.  Passing the full model here is harmless (no extra modules
        match) but misleading -- prefer the explicit backbone argument.

    Args:
        root: nn.Module sub-tree to walk.  Use ``model.backbone`` for the
              image encoder export.

    Returns:
        Number of ``freqs_cis`` buffers that were replaced.
    """
    replaced = 0
    for module in root.modules():
        if hasattr(module, "freqs_cis") and module.freqs_cis is not None:
            freqs_cos = module.freqs_cis.real.float()
            freqs_sin = module.freqs_cis.imag.float()
            # Register real-valued buffers before deleting complex buffer.
            module.register_buffer("freqs_cos", freqs_cos)
            module.register_buffer("freqs_sin", freqs_sin)
            del module.freqs_cis
            replaced += 1
    return replaced
