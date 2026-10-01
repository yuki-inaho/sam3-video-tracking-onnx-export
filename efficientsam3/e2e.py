"""Compare actual EV-M PyTorch and ORT for an image and six moving frames."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops

from contract import CONTEXT, MODEL_SHA256, SOURCE_REVISION
from reference import Grounding, Text, Vision, build_reference
from runtime import VISION_NAMES, Runtime, preprocess, run_session, save_annotation, sigmoid


def compare_mask_outputs(reference, actual, threshold=0.5):
    rs = (sigmoid(reference[1]) * sigmoid(reference[2]).reshape(1, 1, 1)).reshape(-1)
    os = (sigmoid(actual[1]) * sigmoid(actual[2]).reshape(1, 1, 1)).reshape(-1)
    selected = np.flatnonzero(rs > threshold)
    if not len(selected):
        raise ValueError("Reference has no confident detection: use a suitable public image")
    if not np.array_equal(selected, np.flatnonzero(os > threshold)):
        raise ValueError("Selected query IDs differ")
    ious = []
    for query in selected:
        r, o = reference[3][0, query] > 0, actual[3][0, query] > 0
        union = np.logical_or(r, o).sum()
        if not union or not r.any():
            raise ValueError("Reference mask is empty")
        iou = float(np.logical_and(r, o).sum() / union)
        if iou < 0.90:
            raise ValueError(f"Mask IoU below 0.90: {iou}")
        ious.append({"query_id": int(query), "iou": iou, "foreground_pixels": int(r.sum())})
    return ious


def run_e2e(source, checkpoint, models, output, threads=16):
    torch.set_num_threads(threads)
    model = build_reference(source, checkpoint)
    ort = Runtime(models, threads)
    vision, text, grounding = Vision(model), Text(model), Grounding(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(Path(source) / "sam3/assets/dog_person.jpeg").convert("RGB")
    tokens = ort.tokenizer("dog")
    official = model.backbone.language_backbone.tokenizer(["dog"], context_length=CONTEXT).numpy()
    np.testing.assert_array_equal(tokens, official)
    for caption in ["A yellow school bus!", "犬と猫", "école &amp; dog", "word " * 40]:
        np.testing.assert_array_equal(
            ort.tokenizer(caption),
            model.backbone.language_backbone.tokenizer([caption], context_length=CONTEXT).numpy(),
        )
    with torch.inference_mode():
        rt = text(torch.from_numpy(tokens))
        ot = run_session(ort.sessions["text"], {"tokens": tokens})
        np.testing.assert_allclose(rt[0].numpy(), ot[0], rtol=2e-4, atol=2e-4)
        np.testing.assert_array_equal(rt[1].numpy(), ot[1])
        reports = []
        for frame in range(6):
            shifted = ImageChops.offset(image, frame * 4, frame * 2)
            shifted.save(output / f"frame_{frame:06d}.png")
            inp = preprocess(shifted)
            np.save(output / f"input_{frame:06d}.npy", inp)
            start = time.perf_counter()
            rv = vision(torch.from_numpy(inp))
            rr = grounding(*rv, *rt)
            rr = [r.numpy() for r in rr]
            reference_ms = (time.perf_counter() - start) * 1000
            ov = run_session(ort.sessions["vision"], {"image": inp})
            vision_errors = [float(np.max(np.abs(r.numpy() - o))) for r, o in zip(rv, ov)]
            for r, o in zip(rv, ov):
                np.testing.assert_allclose(r.numpy(), o, rtol=3e-3, atol=3e-3)
            oo = run_session(
                ort.sessions["grounding"],
                {
                    **dict(zip(VISION_NAMES, ov)),
                    "text": ot[0],
                    "mask": ot[1],
                },
            )
            if not all(np.isfinite(r).all() for r in oo):
                raise ValueError("Non-finite ORT output")
            ious = compare_mask_outputs(rr, oo)
            np.savez(
                output / f"oracle_{frame:06d}.npz",
                boxes=rr[0],
                logits=rr[1],
                presence=rr[2],
                masks=rr[3],
                text=rt[0].numpy(),
                tokens=tokens,
            )
            if frame == 0:
                np.savez(
                    output / "stages.npz",
                    **{n: r.numpy() for n, r in zip(VISION_NAMES, rv)},
                    text=rt[0].numpy(),
                    mask=rt[1].numpy(),
                )
            _, timing = ort.raw(shifted, "dog")
            detections = save_annotation(oo, shifted, output / f"annotation_{frame:06d}", 0.5)
            report = {
                "frame_index": frame,
                "vision_max_errors": vision_errors,
                "raw_max_errors": [float(np.max(np.abs(r - o))) for r, o in zip(rr, oo)],
                "masks": ious,
                "detections": len(detections),
                "reference_ms": reference_ms,
                "ort_timing": timing,
            }
            reports.append(report)
            print(f"frame {frame}: min IoU {min(v['iou'] for v in ious):.6f}", flush=True)
    result = {
        "passed": True,
        "checkpoint_sha256": MODEL_SHA256,
        "source_revision": SOURCE_REVISION,
        "sequence_mode": "independent_detection",
        "text": "dog",
        "frames": reports,
        "threads": threads,
        "text_max_error": float(np.max(np.abs(rt[0].numpy() - ot[0]))),
    }
    (output / "e2e.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/efficientsam3/source")
    p.add_argument("--checkpoint", default="models/efficientsam3_ev_m.pt")
    p.add_argument("--models", default="outputs/efficientsam3/onnx")
    p.add_argument("--output", default="outputs/efficientsam3/e2e")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    run_e2e(a.source, a.checkpoint, a.models, a.output, a.threads)


if __name__ == "__main__":
    main()
