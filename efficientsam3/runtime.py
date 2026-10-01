"""EV-M image and frame-sequence annotation using ONNX Runtime CPU only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from contract import CONTEXT, MODEL_SHA256, RESOLUTION, SOURCE_REVISION, VARIANT
from tokenizer import Tokenizer

VISION_NAMES = ["fpn0", "fpn1", "fpn2", "pe0", "pe1", "pe2"]


def validate_manifest(manifest):
    expected = {
        "format_version": 1,
        "variant": VARIANT,
        "resolution": RESOLUTION,
        "context": CONTEXT,
        "checkpoint_sha256": MODEL_SHA256,
        "source_revision": SOURCE_REVISION,
        "sequence_mode": "independent_detection",
        "graphs": ["vision.onnx", "text.onnx", "grounding.onnx"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Invalid manifest {key}: expected {value!r}")


def preprocess(image):
    if image.width < 1 or image.height < 1:
        raise ValueError("Empty image")
    resized = image.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.Resampling.BILINEAR)
    return (np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1).copy()


def run_session(session, values):
    return session.run(None, {v.name: values[v.name] for v in session.get_inputs()})


class Runtime:
    def __init__(self, directory, threads=16):
        if threads < 1:
            raise ValueError("threads must be positive")
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        validate_manifest(self.manifest)
        self.tokenizer = Tokenizer(directory / "tokenizer.json")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.sessions = {
            key: ort.InferenceSession(
                str(directory / f"{key}.onnx"), options, providers=["CPUExecutionProvider"]
            )
            for key in ("vision", "text", "grounding")
        }
        self.text_cache = {}

    def raw(self, image, prompt):
        start = time.perf_counter()
        vision = run_session(self.sessions["vision"], {"image": preprocess(image)})
        vision_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if prompt not in self.text_cache:
            self.text_cache[prompt] = run_session(
                self.sessions["text"], {"tokens": self.tokenizer(prompt)}
            )
        text, mask = self.text_cache[prompt]
        text_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        raw = run_session(
            self.sessions["grounding"],
            {
                **dict(zip(VISION_NAMES, vision)),
                "text": text,
                "mask": mask,
            },
        )
        for value in raw:
            if not np.isfinite(value).all():
                raise ValueError("Non-finite model output")
        timing = {
            "vision_ms": vision_ms,
            "text_ms": text_ms,
            "grounding_ms": (time.perf_counter() - start) * 1000,
        }
        return raw, timing


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -80, 80)))


def save_annotation(raw, image, output, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    boxes, logits, presence, masks = raw
    scores = (sigmoid(logits) * sigmoid(presence).reshape(1, 1, 1)).reshape(-1)
    detections = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for query in np.flatnonzero(scores > threshold):
        logit = Image.fromarray(masks[0, query].astype(np.float32))
        mask = np.asarray(logit.resize(image.size, Image.Resampling.BILINEAR)) > 0
        filename = f"mask_{query:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / filename)
        cx, cy, w, h = boxes[0, query]
        detections.append(
            {
                "query_id": int(query),
                "score": float(scores[query]),
                "mask": filename,
                "box_xyxy": [
                    float((cx - w / 2) * image.width),
                    float((cy - h / 2) * image.height),
                    float((cx + w / 2) * image.width),
                    float((cy + h / 2) * image.height),
                ],
                "foreground_pixels": int(mask.sum()),
            }
        )
    return detections


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="outputs/efficientsam3/onnx")
    p.add_argument("--images", nargs="+", required=True, help="Explicit ordered image/frame paths")
    p.add_argument("--text", required=True)
    p.add_argument("--output", default="outputs/efficientsam3/annotations")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    runtime = Runtime(a.models, a.threads)
    frames = []
    for frame_index, filename in enumerate(a.images):
        with Image.open(filename) as source:
            image = source.convert("RGB")
        raw, timing = runtime.raw(image, a.text)
        detections = save_annotation(
            raw, image, Path(a.output) / f"frame_{frame_index:06d}", a.threshold
        )
        frames.append(
            {
                "frame_index": frame_index,
                "width": image.width,
                "height": image.height,
                "timing": timing,
                "detections": detections,
            }
        )
        print(
            f"frame {frame_index}: {len(detections)} detections, {sum(timing.values()):.0f} ms",
            flush=True,
        )
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output) / "annotations.json").write_text(
        json.dumps(
            {
                "backend": "onnxruntime-cpu",
                "variant": VARIANT,
                "sequence_mode": "independent_detection",
                "text": a.text,
                "frames": frames,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
