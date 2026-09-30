"""Official two-object SAM 3.1 video tracking against CPU ONNX propagation."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np
import torch

from sam3_onnx_equiv.path_config import model_paths
from sam3_onnx_equiv.sam31_model import build_sam31_tracker
from sam3_onnx_equiv.sam31_onnx_video import Sam31OnnxSessions, configure_sam31_onnx_window


def _make_video(path: Path) -> None:
    source = model_paths("sam31").source_root / "assets/videos/0001"
    for frame in range(6):
        copyfile(source / f"{frame}.jpg", path / f"{frame:06d}.jpg")


def _init_state(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo

    return VideoTrackingMultiplexDemo.init_state(
        model,
        video_path=str(path),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )


def _initial_masks_from_points(model: torch.nn.Module, path: Path) -> torch.Tensor:
    """Create realistic masks, then prompt both objects into one shared bucket."""
    with torch.inference_mode():
        state = _init_state(model, path)
        for object_id, xy in [(1, (0.59, 0.69)), (2, (0.62, 0.35))]:
            model.add_new_points(
                state,
                0,
                object_id,
                torch.tensor([xy], dtype=torch.float32),
                torch.tensor([1], dtype=torch.int32),
                clear_old_points=True,
            )
        model.propagate_in_video_preflight(state, run_mem_encoder=True)
        _, ids, _, masks, _ = next(model.propagate_in_video(state, 0, 0, False, tqdm_disable=True))
        assert list(ids) == [1, 2]
        return (masks[:, 0] > 0).float().cpu()


def _track(
    model: torch.nn.Module, path: Path, prompt_masks: torch.Tensor
) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    with torch.inference_mode():
        state = _init_state(model, path)
        model.add_new_masks(state, 0, [1, 2], prompt_masks)
        model.propagate_in_video_preflight(state, run_mem_encoder=True)
        multiplex_state = state["multiplex_state"]
        assert multiplex_state.num_buckets == 1
        assert multiplex_state.assignments == [[0, 1] + [-1] * 14]
        result = []
        for frame, ids, _, masks, _scores in model.propagate_in_video(
            state, 0, 6, False, tqdm_disable=True
        ):
            assert frame == len(result)
            result.append((list(ids), masks.detach().cpu().numpy(), _scores.detach().cpu().numpy()))
        return result


def test_sam31_two_object_onnx_video_matches_official(tmp_path: Path) -> None:
    _make_video(tmp_path)
    model = build_sam31_tracker(use_rope_real=True)
    configure_sam31_onnx_window(model)
    prompt_masks = _initial_masks_from_points(model, tmp_path)
    oracle_attention: dict[str, Any] = {}
    attention_module = model.transformer.encoder
    original_attention_forward = attention_module.forward

    def capture_attention(**kwargs: Any) -> dict[str, Any]:
        output = original_attention_forward(**kwargs)
        if not oracle_attention:
            oracle_attention["inputs"] = {
                key: value.clone()
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            }
            oracle_attention["output"] = output["memory"].clone()
        return output

    setattr(attention_module, "forward", capture_attention)
    oracle = _track(model, tmp_path, prompt_masks)
    setattr(attention_module, "forward", original_attention_forward)
    sessions = Sam31OnnxSessions(capture_attention=True)
    sessions.install(model)
    actual = _track(model, tmp_path, prompt_masks)

    assert len(oracle) == len(actual) == 6
    max_logit_errors = []
    mask_ious = []
    for frame, (
        (expected_ids, expected, expected_scores),
        (actual_ids, got, got_scores),
    ) in enumerate(zip(oracle, actual, strict=True)):
        assert expected_ids == actual_ids == [1, 2]
        assert expected.shape == got.shape == (2, 1, 720, 1280)
        assert np.isfinite(expected).all() and np.isfinite(got).all()
        max_abs = float(np.max(np.abs(expected - got)))
        finite_scores = np.isfinite(expected_scores)
        assert finite_scores.all(), f"Official SAM 3.1 returned nonfinite scores in frame {frame}"
        assert np.array_equal(finite_scores, np.isfinite(got_scores))
        np.testing.assert_allclose(
            expected_scores, got_scores, atol=0.05, rtol=0.05, equal_nan=True
        )
        score_max_abs = (
            float(np.max(np.abs(expected_scores[finite_scores] - got_scores[finite_scores])))
            if np.any(finite_scores)
            else 0.0
        )
        print(
            f"frame={frame} max_logit_abs={max_abs:.6f} "
            f"max_finite_score_abs={score_max_abs:.6f} "
            f"nonfinite_scores={int((~finite_scores).sum())}"
        )
        max_logit_errors.append(max_abs)
        for object_index, object_id in enumerate(expected_ids):
            oracle_mask = expected[object_index, 0] > 0
            onnx_mask = got[object_index, 0] > 0
            union = np.logical_or(oracle_mask, onnx_mask).sum()
            intersection = np.logical_and(oracle_mask, onnx_mask).sum()
            iou = float(intersection / union) if union else 1.0
            print(
                f"frame={frame} id={object_id} IoU={iou:.6f} "
                f"oracle_pixels={oracle_mask.sum()} onnx_pixels={onnx_mask.sum()}"
            )
            mask_ious.append(iou)

    assert all((oracle[0][1][object_index] > 0).sum() > 1000 for object_index in range(2))
    for frame in range(2, 6):
        assert all((oracle[frame][1][object_index] > 0).sum() > 100 for object_index in range(2))

    print(f"ONNX calls={dict(sessions.calls)}")
    assert sessions.calls["image"] >= 2
    assert sessions.calls["attention"] >= 3
    assert sessions.calls["decoder"] >= 1
    assert sessions.calls["memory"] >= 1
    assert sessions.first_attention_inputs is not None
    assert sessions.first_attention_output is not None
    assert sessions.first_attention_name is not None
    assert torch.isfinite(sessions.first_attention_output).all()
    oracle_inputs = oracle_attention["inputs"]
    assert isinstance(oracle_inputs, dict)
    oracle_output = oracle_attention["output"]
    assert isinstance(oracle_output, torch.Tensor)
    direct_onnx = sessions.run_buckets(
        sessions.first_attention_name,
        oracle_inputs,
        bucket_axis=1,
        bucket_count=oracle_inputs["src"].shape[1],
    )[0]
    direct_attention_error = float((direct_onnx - oracle_output).abs().max())
    print(f"same-input attention max_abs={direct_attention_error:.6f}")
    ablated_inputs = dict(sessions.first_attention_inputs)
    ablated_inputs["memory"] = torch.zeros_like(ablated_inputs["memory"])
    ablated_inputs["memory_image"] = torch.zeros_like(ablated_inputs["memory_image"])
    ablated = sessions.run_buckets(
        sessions.first_attention_name,
        ablated_inputs,
        bucket_axis=1,
        bucket_count=ablated_inputs["src"].shape[1],
    )[0]
    memory_delta = float((sessions.first_attention_output - ablated).abs().mean())
    print(f"memory ablation mean_abs={memory_delta:.6f}")
    print(f"worst mask IoU={min(mask_ious):.6f} max_logit_abs={max(max_logit_errors):.6f}")
    assert direct_attention_error <= 0.01
    assert memory_delta > 0.01
    assert min(mask_ious) >= 0.90
