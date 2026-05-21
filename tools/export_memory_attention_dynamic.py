"""Export SAM3 memory_attention with dynamic prompt dimension.

This tool exports ONNX files `outputs/onnx/memory_attention_dynamic_k{N}.onnx`
where the prompt dimension (mem_len) is dynamic (variable maskmem frames) and the
obj_ptr token count (= num_k_exclude_rope) is baked per file.

WHY per-N files: the cross-attention applies RoPE to the FIRST
(mem_len - num_k_exclude_rope) keys with repeat_freqs_k.  num_k_rope must be an
exact multiple of HW=5184, so num_k_exclude_rope MUST equal the real obj_ptr token
count.  The oracle uses a variable obj_ptr count per frame (min(frame_idx,6) sets ×
4 tokens = {0,4,8,...,24}).  Zero-padding obj_ptr to a single baked value distorts
the softmax (zero tokens become k_proj.bias / v_proj.bias).  So we export one graph
per valid count and the orchestrator selects the matching one (no padding).

This is distinct from the fixed memory_attention.onnx / memory_attention_full_36352.onnx.

Usage:
    uv run python tools/export_memory_attention_dynamic.py
    uv run python tools/export_memory_attention_dynamic.py --num-k 4 8 12 16 20

Output:
    outputs/onnx/memory_attention_dynamic_k{N}.onnx  for each N
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

EQUIV_SOURCE_ROOT = SANDBOX_ROOT / "outputs" / "sam3_equiv_source"
CHECKPOINT_PATH = SANDBOX_ROOT / "models" / "sam3.pt"
ONNX_DIR = SANDBOX_ROOT / "outputs" / "onnx"

HW = 72 * 72    # 5184
B = 1
D_MODEL = 256
MEM_DIM = 64
OPSET_VERSION = 18

# Valid obj_ptr token counts for a 6-frame clip: min(frame_idx, 6) frames × 4 tokens.
# frame 1->4, 2->8, 3->12, 4->16, 5->20.  (0 = no obj_ptr; not used since frame>=1 always
# has at least the conditioning frame's pointer.)
DEFAULT_NUM_K = [4, 8, 12, 16, 20]


def _output_path(num_k: int) -> Path:
    return ONNX_DIR / f"memory_attention_dynamic_k{num_k}.onnx"


def export_one(num_k: int) -> None:
    """Export a dynamic-maskmem memory_attention graph with obj_ptr count = num_k."""
    from sam3_onnx_equiv.export.memory_attention import build_memory_attention_module  # noqa: E402

    out_path = _output_path(num_k)
    if out_path.exists():
        try:
            onnx.checker.check_model(onnx.load(str(out_path)))
            log.info("Dynamic ONNX (k=%d) already valid — skipping: %s", num_k, out_path)
            return
        except Exception as exc:
            log.warning("Existing dynamic ONNX k=%d invalid (%s); re-exporting.", num_k, exc)
            out_path.unlink()

    # Trace with 1 maskmem frame + num_k obj_ptr tokens; dim 0 is dynamic.
    trace_mem_len = HW + num_k
    log.info("Building memory_attention module (num_k_exclude_rope=%d) ...", num_k)
    wrapper = build_memory_attention_module(
        EQUIV_SOURCE_ROOT, CHECKPOINT_PATH, mem_len=trace_mem_len, num_k_exclude_rope=num_k,
    )

    dummy_src = torch.zeros(HW, B, D_MODEL, dtype=torch.float32)
    dummy_src_pos = torch.zeros(HW, B, D_MODEL, dtype=torch.float32)
    dummy_prompt = torch.zeros(trace_mem_len, B, MEM_DIM, dtype=torch.float32)
    dummy_prompt_pos = torch.zeros(trace_mem_len, B, MEM_DIM, dtype=torch.float32)

    # dim 0 of prompt/prompt_pos is symbolic ("mem_len"); at runtime it must equal
    # N * HW + num_k so that num_k_rope = N * HW is a multiple of HW.
    dynamic_axes = {"prompt": {0: "mem_len"}, "prompt_pos": {0: "mem_len"}}

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Exporting %s (opset=%d, trace_mem_len=%d, num_k=%d) ...",
             out_path.name, OPSET_VERSION, trace_mem_len, num_k)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args=(dummy_src, dummy_src_pos, dummy_prompt, dummy_prompt_pos),
            f=str(out_path),
            input_names=["src", "src_pos", "prompt", "prompt_pos"],
            output_names=["memory"],
            opset_version=OPSET_VERSION,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    op_types = sorted({node.op_type for node in model_proto.graph.node})
    forbidden = {"ComplexFloat", "Complex", "Polar"}
    found = [op for op in op_types if op in forbidden]
    if found:
        raise RuntimeError(f"Complex ops found in k={num_k} graph: {found}")
    log.info("OK k=%d: no complex ops (%d op types).", num_k, len(op_types))

    # Verify dynamic maskmem dim with a couple of N values.
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for n_mask in [1, 2]:
        test_len = n_mask * HW + num_k
        out = sess.run(None, {
            "src": np.zeros((HW, 1, D_MODEL), np.float32),
            "src_pos": np.zeros((HW, 1, D_MODEL), np.float32),
            "prompt": np.zeros((test_len, 1, MEM_DIM), np.float32),
            "prompt_pos": np.zeros((test_len, 1, MEM_DIM), np.float32),
        })
        assert out[0].shape == (HW, 1, D_MODEL), f"bad shape {out[0].shape}"
    log.info("Dynamic ONNX export COMPLETE (k=%d): %s", num_k, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-k", type=int, nargs="+", default=DEFAULT_NUM_K,
                        help="obj_ptr token counts (= num_k_exclude_rope) to export.")
    args = parser.parse_args()
    for num_k in args.num_k:
        export_one(num_k)


if __name__ == "__main__":
    main()
