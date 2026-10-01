import hashlib

import pytest
import torch

from contract import check_state, verify_file


def test_missing_weight_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        check_state({}, {"weight": torch.zeros(2, 3)})


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        check_state({"weight": torch.zeros(3, 2)}, {"weight": torch.zeros(2, 3)})


def test_extra_weight_is_rejected():
    with pytest.raises(ValueError, match="unexpected"):
        check_state({"weight": torch.zeros(2), "other": torch.zeros(1)}, {"weight": torch.zeros(2)})


def test_non_tensor_is_rejected():
    with pytest.raises(ValueError, match="tensor"):
        check_state({"weight": "invalid"}, {"weight": torch.zeros(2)})


def test_nonfinite_weight_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        check_state({"weight": torch.tensor([float("nan")])}, {"weight": torch.zeros(1)})


def test_valid_state():
    check_state({"weight": torch.ones(2, 3)}, {"weight": torch.zeros(2, 3)})


def test_checksum_is_required(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"test")
    with pytest.raises(ValueError, match="SHA256"):
        verify_file(p, "0" * 64)
    assert (
        verify_file(p, hashlib.sha256(b"test").hexdigest()) == hashlib.sha256(b"test").hexdigest()
    )
