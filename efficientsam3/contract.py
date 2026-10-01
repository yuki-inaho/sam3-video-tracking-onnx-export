"""Identity and strict loading rules for the released EV-M Stage3 checkpoint."""

import hashlib
from pathlib import Path

SOURCE_URL = "https://github.com/SimonZeng7108/efficientsam3.git"
SOURCE_REVISION = "bd0936c788fed8d51fa799437f05abd97b401b06"
HF_REPO = "Simon7108528/EfficientSAM3"
HF_REVISION = "85b05896928f974e308f889d7ccb2eefc069de98"
HF_FILE = "efficientsam3_ft/efficientsam3_efficientvit.pt"
MODEL_SHA256 = "086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17"
CONTEXT = 16
RESOLUTION = 1008
VARIANT = "ev-m-stage3"


def verify_file(path, expected=MODEL_SHA256):
    with Path(path).open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def check_state(state, expected):
    """Refuse partial, incompatible or invalid state dictionaries before loading."""
    import torch

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"missing keys: {missing}; unexpected keys: {unexpected}")
    for name, reference in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"non-tensor weight: {name}")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"shape/dtype mismatch for {name}: {value.shape}, {value.dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"non-finite weight: {name}")
