"""Unit contracts for the public SAM 3.1 annotation layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from sam3_onnx_equiv.annotation import (
    MAX_OBJECTS,
    MAX_PROMPT_POINTS,
    PromptPoint,
    Sam31AnnotationEngine,
    decode_uncompressed_rle,
    encode_uncompressed_rle,
    mask_annotation,
    render_overlay,
    validate_prompt_points,
)


def test_prompt_points_validate_normalized_coordinates_and_labels() -> None:
    point = PromptPoint.from_mapping({"x": 0, "y": 1.0, "label": 1})
    assert point == PromptPoint(0.0, 1.0, 1)
    assert point.to_dict() == {"x": 0.0, "y": 1.0, "label": 1}
    assert validate_prompt_points([{"x": 0.25, "y": 0.75}]) == (PromptPoint(0.25, 0.75, 1),)

    for invalid in (-0.01, 1.01, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises(ValueError):
            PromptPoint(invalid, 0.5, 1)  # type: ignore[arg-type]
    for invalid_label in (-1, 2, True, 1.0):
        with pytest.raises(ValueError):
            PromptPoint(0.5, 0.5, invalid_label)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="foreground"):
        validate_prompt_points([PromptPoint(0.5, 0.5, 0)])
    with pytest.raises(ValueError, match="at least one point"):
        validate_prompt_points([])
    with pytest.raises(ValueError, match="unknown"):
        PromptPoint.from_mapping({"x": 0.5, "y": 0.5, "path": "/private"})


def test_uncompressed_rle_is_column_major_and_round_trips_exactly() -> None:
    mask = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    rle = encode_uncompressed_rle(mask)
    assert rle == {"size": [2, 3], "counts": [0, 2, 1, 2, 1]}
    decoded = decode_uncompressed_rle(rle)
    assert decoded.dtype == np.bool_
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(decoded, mask.astype(bool))

    assert encode_uncompressed_rle(np.zeros((2, 2), dtype=bool))["counts"] == [4]
    assert encode_uncompressed_rle(np.ones((2, 2), dtype=bool))["counts"] == [0, 4]


@pytest.mark.parametrize(
    "rle",
    [
        None,
        {"size": [2, 2]},
        {"size": [2, 2], "counts": [4], "area": 0},
        {"size": [2], "counts": [2]},
        {"size": [True, 2], "counts": [2]},
        {"size": [2, 2], "counts": "4"},
        {"size": [2, 2], "counts": []},
        {"size": [2, 2], "counts": [3]},
        {"size": [2, 2], "counts": [0, 1, 0, 3]},
        {"size": [2, 2], "counts": [False, 4]},
    ],
)
def test_uncompressed_rle_rejects_malformed_or_noncanonical_data(rle: object) -> None:
    with pytest.raises(ValueError):
        decode_uncompressed_rle(rle)


def test_mask_annotation_has_exact_bbox_area_and_json_safe_score() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 2:5] = True
    mask[2, 3] = False

    annotation = mask_annotation(mask, frame_index=7, object_id=42, score=np.float32(0.75))
    assert annotation.bbox == (2, 1, 3, 3)
    assert annotation.area == 8
    assert annotation.score == pytest.approx(0.75)
    assert annotation.mask == annotation.segmentation
    assert json.loads(json.dumps(annotation.to_dict())) == annotation.to_dict()

    nonfinite = mask_annotation(mask, frame_index=7, object_id=42, score=float("nan"))
    assert nonfinite.score is None
    assert nonfinite.to_dict()["score"] is None


def test_overlay_is_deterministic_and_does_not_mutate_input() -> None:
    image = np.full((6, 7, 3), 40, dtype=np.uint8)
    original = image.copy()
    first_mask = np.zeros((6, 7), dtype=bool)
    first_mask[1:5, 1:4] = True
    second_mask = np.zeros((6, 7), dtype=bool)
    second_mask[2:5, 3:6] = True
    first = mask_annotation(first_mask, frame_index=0, object_id=11, score=0.5)
    second = mask_annotation(second_mask, frame_index=0, object_id=3, score=0.6)

    rendered = np.asarray(render_overlay(image, [first, second], alpha=0.4))
    reversed_order = np.asarray(render_overlay(image, [second, first], alpha=0.4))
    np.testing.assert_array_equal(rendered, reversed_order)
    np.testing.assert_array_equal(image, original)
    assert rendered.dtype == np.uint8
    assert not np.array_equal(rendered, original)


class _FakeModel:
    def __init__(self) -> None:
        self.point_calls: list[tuple[int, int, np.ndarray, np.ndarray, bool]] = []
        self.propagation_calls: list[str] = []

    def propagate_in_video_preflight(self, state: dict[str, Any], *, run_mem_encoder: bool) -> None:
        assert state["num_frames"] > 0
        assert run_mem_encoder is True
        self.propagation_calls.append("preflight")

    def add_new_points(
        self,
        state: dict[str, Any],
        frame_index: int,
        object_id: int,
        points: Any,
        labels: Any,
        *,
        clear_old_points: bool,
    ) -> tuple[int, list[int], None, np.ndarray, np.ndarray]:
        point_array = points.detach().cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        self.point_calls.append(
            (frame_index, object_id, point_array, label_array, clear_old_points)
        )
        if object_id not in state["obj_ids"]:
            state["obj_id_to_idx"][object_id] = len(state["obj_ids"])
            state["obj_ids"].append(object_id)
        mask = np.full((1, 1, state["video_height"], state["video_width"]), -1.0)
        mask[:, :, 1:3, 2:5] = 1.0
        return frame_index, [object_id], None, mask, np.array([[0.8]], dtype=np.float32)

    def propagate_in_video(
        self,
        state: dict[str, Any],
        start_frame: int,
        last_frame: int,
        reverse: bool,
        *,
        tqdm_disable: bool,
    ) -> Any:
        self.propagation_calls.append("propagate")
        assert (start_frame, last_frame, reverse, tqdm_disable) == (
            0,
            state["num_frames"] - 1,
            False,
            True,
        )
        object_ids = list(state["obj_ids"])
        for frame_index in range(state["num_frames"]):
            masks = np.full(
                (len(object_ids), 1, state["video_height"], state["video_width"]),
                -1.0,
            )
            for object_index in range(len(object_ids)):
                masks[
                    object_index,
                    0,
                    frame_index : frame_index + 2,
                    object_index : object_index + 2,
                ] = 1.0
            scores = np.array([[0.25], [np.inf]][: len(object_ids)], dtype=np.float32)
            yield frame_index, object_ids, None, masks, scores


class _FakeEngine(Sam31AnnotationEngine):
    def _ensure_backend(self) -> None:
        if self._model is None:
            self._model = _FakeModel()
            self._sessions = object()

    def _initialize_state(self, frame_dir: Path) -> dict[str, Any]:
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        with Image.open(frame_paths[0]) as image:
            width, height = image.size
        return {
            "num_frames": len(frame_paths),
            "video_width": width,
            "video_height": height,
            "obj_ids": [],
            "obj_id_to_idx": {},
        }


class _FourTuplePropagationModel(_FakeModel):
    def propagate_in_video(
        self,
        state: dict[str, Any],
        start_frame: int,
        last_frame: int,
        reverse: bool,
        *,
        tqdm_disable: bool,
    ) -> Any:
        self.propagation_calls.append("propagate")
        assert (start_frame, last_frame, reverse, tqdm_disable) == (
            0,
            state["num_frames"] - 1,
            False,
            True,
        )
        object_ids = list(state["obj_ids"])
        for frame_index in range(state["num_frames"]):
            masks = np.full(
                (len(object_ids), 1, state["video_height"], state["video_width"]),
                -1.0,
            )
            for object_index, object_id in enumerate(object_ids):
                masks[object_index, 0, 1:3, object_index : object_index + 2] = 1.0
                state.setdefault("output_dict_per_obj", {}).setdefault(
                    state["obj_id_to_idx"][object_id],
                    {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                )["non_cond_frame_outputs"][frame_index] = {
                    "object_score_logits": np.array(
                        [[0.0 if object_index == 0 else -2.0]], dtype=np.float32
                    )
                }
            # SAM 3.1's official generator exposes masks in a four-tuple and
            # retains object presence logits in output_dict_per_obj.
            yield frame_index, object_ids, None, masks


class _FourTuplePropagationEngine(_FakeEngine):
    def _ensure_backend(self) -> None:
        if self._model is None:
            self._model = _FourTuplePropagationModel()
            self._sessions = object()


class _ReorderingCorrectionModel(_FakeModel):
    """Mirror SAM 3.1 moving a refined tracked object to the registry tail."""

    def propagate_in_video_preflight(self, state: dict[str, Any], *, run_mem_encoder: bool) -> None:
        super().propagate_in_video_preflight(state, run_mem_encoder=run_mem_encoder)
        state["tracking_has_started"] = True

    def add_new_points(
        self,
        state: dict[str, Any],
        frame_index: int,
        object_id: int,
        points: Any,
        labels: Any,
        *,
        clear_old_points: bool,
    ) -> tuple[int, list[int], None, np.ndarray, np.ndarray]:
        if state.get("tracking_has_started") and object_id in state["obj_ids"]:
            state["obj_ids"].remove(object_id)
            state["obj_ids"].append(object_id)
            state["obj_id_to_idx"] = {
                current_id: index for index, current_id in enumerate(state["obj_ids"])
            }
        return super().add_new_points(
            state,
            frame_index,
            object_id,
            points,
            labels,
            clear_old_points=clear_old_points,
        )


class _ReorderingCorrectionEngine(_FakeEngine):
    def _ensure_backend(self) -> None:
        if self._model is None:
            self._model = _ReorderingCorrectionModel()
            self._sessions = object()


class _SingletonCacheRefinementModel(_FakeModel):
    """Model the official tracked-object extraction and its one-frame cache."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_calls: list[tuple[int, bool, bool]] = []

    def _get_image_feature(
        self, state: dict[str, Any], frame_index: int, batch_size: int
    ) -> tuple[object, object]:
        assert batch_size >= 1
        cached = state["cached_features"].get(frame_index)
        cache_hit = cached is not None
        has_images = "images" in state
        if cached is None:
            # This is the precise failure in the official singleton state: its
            # init_state accepts cached features but deliberately omits images.
            image = state["images"][frame_index]
            cached = (image, object())
            state["cached_features"] = {frame_index: cached}
        self.feature_calls.append((frame_index, has_images, cache_hit))
        return cached

    def propagate_in_video_preflight(self, state: dict[str, Any], *, run_mem_encoder: bool) -> None:
        super().propagate_in_video_preflight(state, run_mem_encoder=run_mem_encoder)
        state["tracking_has_started"] = True

    def add_new_points(
        self,
        state: dict[str, Any],
        frame_index: int,
        object_id: int,
        points: Any,
        labels: Any,
        *,
        clear_old_points: bool,
    ) -> tuple[int, list[int], None, np.ndarray, np.ndarray]:
        if state.get("tracking_has_started") and object_id in state["obj_ids"]:
            singleton_state = {"cached_features": state["cached_features"]}
            self._get_image_feature(singleton_state, frame_index, 1)
        return super().add_new_points(
            state,
            frame_index,
            object_id,
            points,
            labels,
            clear_old_points=clear_old_points,
        )

    def propagate_in_video(
        self,
        state: dict[str, Any],
        start_frame: int,
        last_frame: int,
        reverse: bool,
        *,
        tqdm_disable: bool,
    ) -> Any:
        for output in super().propagate_in_video(
            state,
            start_frame,
            last_frame,
            reverse,
            tqdm_disable=tqdm_disable,
        ):
            self._get_image_feature(state, output[0], len(state["obj_ids"]))
            yield output


class _SingletonCacheRefinementEngine(_FakeEngine):
    def _ensure_backend(self) -> None:
        if self._model is None:
            self._model = _SingletonCacheRefinementModel()
            self._sessions = object()

    def _initialize_state(self, frame_dir: Path) -> dict[str, Any]:
        state = super()._initialize_state(frame_dir)
        state.update(
            images=tuple(sorted(frame_dir.glob("*.jpg"))),
            cached_features={},
            tracking_has_started=False,
        )
        return state


class _DtypeSensitiveMultiplexState:
    def __init__(self) -> None:
        self.mux_matrix = torch.eye(1, dtype=torch.float32)
        self.demux_matrix = torch.eye(1, dtype=torch.float32)

    def mux(self, value: torch.Tensor) -> torch.Tensor:
        return self.mux_matrix @ value.reshape(1, -1)


class _DtypeSensitiveRefinementModel(_SingletonCacheRefinementModel):
    def __init__(self) -> None:
        super().__init__()
        self.merged_dtypes: list[torch.dtype] = []

    def add_new_points(
        self,
        state: dict[str, Any],
        frame_index: int,
        object_id: int,
        points: Any,
        labels: Any,
        *,
        clear_old_points: bool,
    ) -> tuple[int, list[int], None, np.ndarray, np.ndarray]:
        if state.get("tracking_has_started") and object_id in state["obj_ids"]:
            output = state["output_dict"]["cond_frame_outputs"][0]
            tensors = [output["maskmem_features"], *output["maskmem_pos_enc"], output["obj_ptr"]]
            self.merged_dtypes = [state["multiplex_state"].mux(tensor).dtype for tensor in tensors]
        return super().add_new_points(
            state,
            frame_index,
            object_id,
            points,
            labels,
            clear_old_points=clear_old_points,
        )


class _DtypeSensitiveRefinementEngine(_SingletonCacheRefinementEngine):
    def _ensure_backend(self) -> None:
        if self._model is None:
            self._model = _DtypeSensitiveRefinementModel()
            self._sessions = object()

    def _initialize_state(self, frame_dir: Path) -> dict[str, Any]:
        state = super()._initialize_state(frame_dir)
        memory = torch.ones((1, 2), dtype=torch.bfloat16)
        position = torch.ones((1, 2), dtype=torch.bfloat16)
        state.update(
            multiplex_state=_DtypeSensitiveMultiplexState(),
            output_dict={
                "cond_frame_outputs": {
                    0: {
                        "maskmem_features": memory,
                        "maskmem_pos_enc": [position],
                        "obj_ptr": torch.ones((1, 2), dtype=torch.bfloat16),
                    }
                },
                "non_cond_frame_outputs": {},
            },
            constants={"maskmem_pos_enc": [position]},
        )
        return state


def _write_frames(directory: Path, count: int = 3) -> None:
    for frame_index in range(count):
        Image.new("RGB", (8, 6), (frame_index, 0, 0)).save(directory / f"{frame_index:06d}.jpg")


def test_engine_uses_all_prompts_and_propagates_stable_ids(tmp_path: Path) -> None:
    _write_frames(tmp_path)
    engine = _FakeEngine(threads=2)
    assert engine.open(tmp_path) == {"frame_count": 3, "width": 8, "height": 6}
    assert dict(engine.media_info) == {"frame_count": 3, "width": 8, "height": 6}

    first = engine.segment(
        0,
        101,
        [PromptPoint(0.25, 0.5, 1), PromptPoint(0.75, 0.5, 0)],
    )
    assert [annotation.object_id for annotation in first] == [101]
    call = engine._model.point_calls[-1]
    np.testing.assert_array_equal(call[2], [[0.25, 0.5], [0.75, 0.5]])
    np.testing.assert_array_equal(call[3], [1, 0])
    assert call[4] is True

    current = engine.segment(0, 7, [{"x": 0.5, "y": 0.25, "label": 1}])
    assert [annotation.object_id for annotation in current] == [101, 7]
    propagated = engine.propagate()
    assert engine._model.propagation_calls == ["preflight", "propagate"]
    assert list(propagated) == [0, 1, 2]
    assert all(
        [annotation.object_id for annotation in frame] == [101, 7] for frame in propagated.values()
    )
    assert propagated[0][0].score == pytest.approx(0.5621765)
    assert propagated[0][1].score is None
    assert list(engine.results) == [0, 1, 2]
    read_only_results: Any = engine.results
    with pytest.raises(TypeError):
        read_only_results[3] = ()


def test_four_tuple_propagation_reads_official_state_scores(tmp_path: Path) -> None:
    _write_frames(tmp_path, count=2)
    engine = _FourTuplePropagationEngine()
    engine.open(tmp_path)
    engine.segment(0, 101, [PromptPoint(0.25, 0.5)])
    engine.segment(0, 7, [PromptPoint(0.75, 0.5)])

    propagated = engine.propagate()

    assert list(propagated) == [0, 1]
    assert all(record.score is not None for records in propagated.values() for record in records)
    for records in propagated.values():
        assert records[0].score == pytest.approx(0.5)
        assert records[1].score == pytest.approx(0.11920292)


def test_correction_resynchronises_reordered_object_ids_before_propagation(
    tmp_path: Path,
) -> None:
    _write_frames(tmp_path, count=2)
    engine = _ReorderingCorrectionEngine()
    engine.open(tmp_path)
    engine.segment(0, 1, [PromptPoint(0.25, 0.5)])
    engine.segment(0, 2, [PromptPoint(0.75, 0.5)])
    engine.propagate()

    engine.segment(0, 1, [PromptPoint(0.3, 0.5)])

    assert engine.object_ids == (2, 1)
    propagated = engine.propagate()
    assert all(
        [annotation.object_id for annotation in records] == [2, 1]
        for records in propagated.values()
    )
    assert [(record.object_id, record.bbox[0]) for record in propagated[0]] == [
        (2, 0),
        (1, 1),
    ]
    assert propagated[0][0].score == pytest.approx(0.5621765)
    assert propagated[0][1].score is None


def test_correction_primes_frame_before_official_singleton_extraction(tmp_path: Path) -> None:
    _write_frames(tmp_path, count=3)
    engine = _SingletonCacheRefinementEngine()
    engine.open(tmp_path)
    engine.segment(0, 1, [PromptPoint(0.25, 0.5)])
    engine.propagate()

    state = engine._state
    assert state is not None
    assert set(state["cached_features"]) == {2}

    corrected = engine.segment(0, 1, [PromptPoint(0.3, 0.5)])

    assert [record.object_id for record in corrected] == [1]
    assert engine._model.feature_calls[-2:] == [
        (0, True, False),  # wrapper primes from the source state's decoded frames
        (0, False, True),  # official singleton state then gets a cache hit
    ]
    assert list(engine.propagate()) == [0, 1, 2]


def test_correction_aligns_compact_memory_with_cpu_multiplex_dtype(tmp_path: Path) -> None:
    _write_frames(tmp_path, count=2)
    engine = _DtypeSensitiveRefinementEngine()
    engine.open(tmp_path)
    engine.segment(0, 1, [PromptPoint(0.25, 0.5)])
    engine.propagate()

    state = engine._state
    assert state is not None
    output = state["output_dict"]["cond_frame_outputs"][0]
    assert output["maskmem_features"].dtype == torch.bfloat16

    engine.segment(0, 1, [PromptPoint(0.3, 0.5)])

    assert engine._model.merged_dtypes == [torch.float32, torch.float32, torch.float32]
    assert output["maskmem_features"].dtype == torch.float32
    assert output["maskmem_pos_enc"][0].dtype == torch.float32
    assert output["obj_ptr"].dtype == torch.float32
    assert state["constants"]["maskmem_pos_enc"][0].dtype == torch.float32
    assert list(engine.propagate()) == [0, 1]


def test_engine_enforces_object_limit_and_validates_frame_directory(tmp_path: Path) -> None:
    _write_frames(tmp_path, count=1)
    engine = _FakeEngine()
    engine.open(tmp_path)
    for object_id in range(MAX_OBJECTS):
        engine.segment(0, object_id, [PromptPoint(0.5, 0.5)])
    with pytest.raises(ValueError, match=str(MAX_OBJECTS)):
        engine.segment(0, MAX_OBJECTS, [PromptPoint(0.5, 0.5)])

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    Image.new("RGB", (8, 6)).save(malformed / "000001.jpg")
    with pytest.raises(ValueError, match="contiguous"):
        engine.open(malformed)


def test_engine_caps_prompt_points() -> None:
    points = [PromptPoint(0.5, 0.5)] * (MAX_PROMPT_POINTS + 1)
    with pytest.raises(ValueError, match=str(MAX_PROMPT_POINTS)):
        validate_prompt_points(points)
