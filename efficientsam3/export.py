"""Export fixed-shape EV-M vision, text and grounding graphs with strict weights."""

import argparse
import json
from pathlib import Path

import onnx
import torch

from contract import CONTEXT, HF_REVISION, MODEL_SHA256, RESOLUTION, SOURCE_REVISION, VARIANT
from reference import Grounding, Text, Vision, build_reference

VISION_NAMES = ["fpn0", "fpn1", "fpn2", "pe0", "pe1", "pe2"]


def export_graph(module, inputs, path, input_names, output_names):
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(path))
    print(f"Validated {path.name}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/efficientsam3/source")
    p.add_argument("--checkpoint", default="models/efficientsam3_ev_m.pt")
    p.add_argument("--output", default="outputs/efficientsam3/onnx")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    m = build_reference(a.source, a.checkpoint)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    image = torch.zeros(1, 3, RESOLUTION, RESOLUTION)
    tokens = m.backbone.language_backbone.tokenizer(["circle"], context_length=CONTEXT)
    with torch.inference_mode():
        v, t, g = Vision(m), Text(m), Grounding(m)
        features = v(image)
        text, mask = t(tokens)
        export_graph(v, (image,), output / "vision.onnx", ["image"], VISION_NAMES)
        export_graph(t, (tokens,), output / "text.onnx", ["tokens"], ["text", "mask"])
        export_graph(
            g,
            (*features, text, mask),
            output / "grounding.onnx",
            [*VISION_NAMES, "text", "mask"],
            ["boxes", "logits", "presence", "masks"],
        )
    tokenizer = m.backbone.language_backbone.tokenizer
    (output / "tokenizer.json").write_text(
        json.dumps(
            {
                "encoder": tokenizer.encoder,
                "merges": [
                    list(pair)
                    for pair, _ in sorted(tokenizer.bpe_ranks.items(), key=lambda v: v[1])
                ],
            }
        )
        + "\n"
    )
    manifest = {
        "format_version": 1,
        "variant": VARIANT,
        "source_revision": SOURCE_REVISION,
        "hf_revision": HF_REVISION,
        "checkpoint_sha256": MODEL_SHA256,
        "resolution": RESOLUTION,
        "context": CONTEXT,
        "sequence_mode": "independent_detection",
        "parameter_count": sum(v.numel() for v in m.parameters()),
        "graphs": ["vision.onnx", "text.onnx", "grounding.onnx"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("EV-M export complete", flush=True)


if __name__ == "__main__":
    main()
