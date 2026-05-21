"""ONNX-safe rotary position encoding helpers for SAM-family memory attention.

The official SAM3 source already contains a real-valued RoPE path in some modules,
but the base SAM3 video builder still creates tracker attention with
``use_rope_real=False``. This module provides an isolated equivalence target:
complex-number RoPE and real-valued cos/sin RoPE produce the same tensors, while the
real-valued path contains only standard floating-point tensor operations that can be
exported to ONNX.

The functions are intentionally rank-generic for tensors shaped as
``[..., seq_len, head_dim]``. Tests cover the common SAM attention layout
``[batch, heads, seq_len, head_dim]`` and the ``repeat_freqs_k`` mode used for
cross-attention to memory tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from jaxtyping import Float
from torch import Tensor, nn

PathLike: TypeAlias = str | Path
DeviceLike: TypeAlias = torch.device | str | None
RotaryTensor: TypeAlias = Float[Tensor, "*batch seq head_dim"]
FrequencyTensor: TypeAlias = Float[Tensor, "seq pair_dim"]
RotaryPair: TypeAlias = tuple[RotaryTensor, RotaryTensor]


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_even_head_dim(x: Tensor, name: str) -> None:
    if x.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions, got shape={tuple(x.shape)}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"{name} last dimension must be even for RoPE, got {x.shape[-1]}")


def init_t_xy(
    end_x: int,
    end_y: int,
    scale: float = 1.0,
    offset: int = 0,
    device: DeviceLike = None,
) -> tuple[Tensor, Tensor]:
    """Return flattened x/y coordinates matching SAM3's axial RoPE order."""
    _validate_positive_int(end_x, "end_x")
    _validate_positive_int(end_y, "end_y")
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def _axis_frequencies(dim: int, theta: float, device: DeviceLike) -> Tensor:
    if dim % 4 != 0:
        raise ValueError(f"SAM axial RoPE expects dim divisible by 4, got dim={dim}")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got theta={theta}")
    dim_range = torch.arange(0, dim, 4, device=device)[: dim // 4].float()
    return 1.0 / (theta ** (dim_range / dim))


def _axial_angles(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device: DeviceLike = None,
) -> FrequencyTensor:
    """Compute SAM-style axial RoPE angles with shape ``[end_x*end_y, dim//2]``."""
    freqs = _axis_frequencies(dim, theta, device)
    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    angles_x = torch.outer(t_x, freqs)
    angles_y = torch.outer(t_y, freqs)
    return torch.cat([angles_x, angles_y], dim=-1)


def compute_axial_cis_real(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device: DeviceLike = None,
) -> tuple[FrequencyTensor, FrequencyTensor]:
    """Compute real/imaginary RoPE factors without complex tensors.

    Returns:
        ``(cos, sin)``, each shaped ``[end_x * end_y, dim // 2]``. These are
        equivalent to ``compute_axial_cis(...).real`` and ``.imag`` in the official
        complex implementation.
    """
    angles = _axial_angles(dim, end_x, end_y, theta, scale_pos, offset, device=device)
    return angles.cos(), angles.sin()


def compute_axial_cis_complex_reference(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device: DeviceLike = None,
) -> Tensor:
    """Reference complex RoPE factors matching the official SAM3 formulation."""
    angles = _axial_angles(dim, end_x, end_y, theta, scale_pos, offset, device=device)
    return torch.polar(torch.ones_like(angles), angles)


def _reshape_freq_for_broadcast(freq: Tensor, x_pairs: Tensor) -> Tensor:
    """Broadcast a ``[seq_len, dim/2]`` frequency tensor over ``x_pairs``."""
    expected_shape = (x_pairs.shape[-2], x_pairs.shape[-1])
    if freq.shape != expected_shape:
        raise ValueError(
            "frequency shape must match the target sequence and pair dimensions: "
            f"freq={tuple(freq.shape)}, target={expected_shape}"
        )
    view_shape = [dim if idx >= x_pairs.ndim - 2 else 1 for idx, dim in enumerate(x_pairs.shape)]
    return freq.view(*view_shape)


def _rotate_real_pairs(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    _validate_even_head_dim(x, "x")
    x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
    x_real = x_pair[..., 0]
    x_imag = x_pair[..., 1]
    cos_broadcast = _reshape_freq_for_broadcast(cos, x_real)
    sin_broadcast = _reshape_freq_for_broadcast(sin, x_imag)
    out_real = x_real * cos_broadcast - x_imag * sin_broadcast
    out_imag = x_real * sin_broadcast + x_imag * cos_broadcast
    return torch.stack([out_real, out_imag], dim=-1).flatten(-2).type_as(x).to(x.device)


def _repeat_key_frequencies(
    xq_seq_len: int,
    xk_seq_len: int,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    if xk_seq_len % xq_seq_len != 0:
        raise ValueError(
            "repeat_freqs_k=True requires key sequence length to be an integer "
            f"multiple of query sequence length, got k={xk_seq_len}, q={xq_seq_len}"
        )
    repeat = xk_seq_len // xq_seq_len
    return cos.repeat(repeat, 1), sin.repeat(repeat, 1)


def apply_rotary_enc_real_safe(
    xq: RotaryTensor,
    xk: RotaryTensor,
    cos: FrequencyTensor,
    sin: FrequencyTensor,
    repeat_freqs_k: bool = False,
) -> RotaryPair:
    """Apply ONNX-safe RoPE to query/key tensors.

    Args:
        xq: Query tensor shaped ``[..., q_seq_len, head_dim]``.
        xk: Key tensor shaped ``[..., k_seq_len, head_dim]``.
        cos: Cosine factors shaped ``[q_seq_len, head_dim // 2]``.
        sin: Sine factors shaped ``[q_seq_len, head_dim // 2]``.
        repeat_freqs_k: If true, repeat query frequencies along the key sequence.
            This matches SAM/SAM2/SAM3 cross-attention where memory keys repeat per
            query grid position.
    """
    _validate_even_head_dim(xq, "xq")
    _validate_even_head_dim(xk, "xk")
    if xq.shape[-1] != xk.shape[-1]:
        raise ValueError(f"xq/xk head dims must match: {xq.shape[-1]} vs {xk.shape[-1]}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shapes must match: {tuple(cos.shape)} vs {tuple(sin.shape)}")

    q_out = _rotate_real_pairs(xq, cos.to(device=xq.device), sin.to(device=xq.device))
    if xk.shape[-2] == 0:
        return q_out, xk

    k_cos, k_sin = cos, sin
    if repeat_freqs_k:
        k_cos, k_sin = _repeat_key_frequencies(xq.shape[-2], xk.shape[-2], cos, sin)
    k_out = _rotate_real_pairs(xk, k_cos.to(device=xk.device), k_sin.to(device=xk.device))
    return q_out, k_out


def apply_rotary_enc_complex_reference(
    xq: RotaryTensor,
    xk: RotaryTensor,
    freqs_cis: Tensor,
    repeat_freqs_k: bool = False,
) -> RotaryPair:
    """Reference complex RoPE matching the official non-ONNX-safe formulation."""
    _validate_even_head_dim(xq, "xq")
    _validate_even_head_dim(xk, "xk")
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs = _reshape_freq_for_broadcast(freqs_cis, xq_complex)
    q_out = torch.view_as_real(xq_complex * freqs).flatten(-2).type_as(xq).to(xq.device)

    if xk.shape[-2] == 0:
        return q_out, xk

    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    k_freqs = freqs
    if repeat_freqs_k:
        if xk_complex.shape[-2] % xq_complex.shape[-2] != 0:
            raise ValueError(
                "repeat_freqs_k=True requires key sequence length to be an integer "
                f"multiple of query sequence length, got "
                f"k={xk_complex.shape[-2]}, q={xq_complex.shape[-2]}"
            )
        repeat = xk_complex.shape[-2] // xq_complex.shape[-2]
        k_freqs = freqs.repeat(*([1] * (freqs.ndim - 2)), repeat, 1)
    k_out = torch.view_as_real(xk_complex * k_freqs).flatten(-2).type_as(xk).to(xk.device)
    return q_out, k_out


@dataclass(frozen=True)
class RotaryExportSpec:
    """Small deterministic RoPE export shape specification."""

    batch: int = 1
    heads: int = 2
    end_x: int = 4
    end_y: int = 4
    head_dim: int = 8
    repeat_freqs_k: bool = False

    def __post_init__(self) -> None:
        for name in ("batch", "heads", "end_x", "end_y", "head_dim"):
            _validate_positive_int(getattr(self, name), name)
        if self.head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {self.head_dim}")

    @property
    def seq_len(self) -> int:
        return self.end_x * self.end_y

    @property
    def key_seq_len(self) -> int:
        return self.seq_len * (2 if self.repeat_freqs_k else 1)


class TinyRotaryModule(nn.Module):
    """Minimal export target that exercises the ONNX-safe RoPE path."""

    cos: Tensor
    sin: Tensor

    def __init__(self, spec: RotaryExportSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or RotaryExportSpec()
        cos, sin = compute_axial_cis_real(
            dim=self.spec.head_dim,
            end_x=self.spec.end_x,
            end_y=self.spec.end_y,
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.repeat_freqs_k = self.spec.repeat_freqs_k

    def forward(self, xq: Tensor, xk: Tensor) -> tuple[Tensor, Tensor]:
        return apply_rotary_enc_real_safe(xq, xk, self.cos, self.sin, self.repeat_freqs_k)


def make_rotary_sample_inputs(spec: RotaryExportSpec) -> tuple[Tensor, Tensor]:
    """Create sample query/key tensors for export and ONNX Runtime verification."""
    q = torch.randn(spec.batch, spec.heads, spec.seq_len, spec.head_dim, dtype=torch.float32)
    k = torch.randn(
        spec.batch,
        spec.heads,
        spec.key_seq_len,
        spec.head_dim,
        dtype=torch.float32,
    )
    return q, k


def export_tiny_rotary_onnx(output_path: PathLike, spec: RotaryExportSpec | None = None) -> Path:
    """Export ``TinyRotaryModule`` to ONNX and return the output path."""
    resolved_spec = spec or RotaryExportSpec()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module = TinyRotaryModule(resolved_spec).eval()
    q, k = make_rotary_sample_inputs(resolved_spec)
    torch.onnx.export(
        module,
        (q, k),
        str(output_path),
        input_names=["xq", "xk"],
        output_names=["xq_rot", "xk_rot"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "xq": {0: "batch", 2: "q_seq"},
            "xk": {0: "batch", 2: "k_seq"},
            "xq_rot": {0: "batch", 2: "q_seq"},
            "xk_rot": {0: "batch", 2: "k_seq"},
        },
    )
    return output_path
