"""Opt-in trained-model acceptance test; skips are not validation evidence."""

import os

import pytest


@pytest.mark.skipif(os.environ.get("EFFICIENTSAM3_REAL") != "1", reason="Set EFFICIENTSAM3_REAL=1")
def test_ev_m_image_and_six_frame_sequence(tmp_path):
    from e2e import run_e2e

    report = run_e2e(
        os.environ.get("EFFICIENTSAM3_SOURCE", "outputs/efficientsam3/source"),
        os.environ.get("EFFICIENTSAM3_CHECKPOINT", "models/efficientsam3_ev_m.pt"),
        os.environ.get("EFFICIENTSAM3_MODELS", "outputs/efficientsam3/onnx"),
        tmp_path,
    )
    assert report["passed"] and len(report["frames"]) == 6
