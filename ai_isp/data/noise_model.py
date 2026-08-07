"""用于数据增强与链路冒烟的可复现物理噪声模型。"""

from __future__ import annotations

import torch


def synthesize_correlated_ryyb_noise(
    clean: torch.Tensor,
    shot_coefficients: torch.Tensor,
    correlation_matrix: torch.Tensor,
    read_covariance: torch.Tensor,
    bit_depth: int = 12,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """采样 ``A diag(a*max(x,0)) A^T + Sigma_read`` 的四通道相关噪声。"""

    if clean.ndim not in (3, 4) or clean.shape[-3] != 4:
        raise ValueError("RYYB Clean RAW 必须为 4×H×W 或 N×4×H×W")
    if bit_depth not in (10, 12, 14, 16):
        raise ValueError("bit_depth 仅支持 10/12/14/16")
    a = torch.as_tensor(shot_coefficients, dtype=clean.dtype, device=clean.device)
    matrix = torch.as_tensor(correlation_matrix, dtype=clean.dtype, device=clean.device)
    read = torch.as_tensor(read_covariance, dtype=clean.dtype, device=clean.device)
    if a.shape != (4,) or matrix.shape != (4, 4) or read.shape != (4, 4):
        raise ValueError("a/A/Sigma_read 的 Shape 必须分别为 4、4×4、4×4")
    if bool((a < 0).any()):
        raise ValueError("Shot Noise 系数不得为负数")
    if not torch.allclose(read, read.transpose(0, 1), atol=1e-8, rtol=0.0):
        raise ValueError("Sigma_read 必须对称")
    try:
        read_factor = torch.linalg.cholesky(read + torch.eye(4, device=clean.device, dtype=clean.dtype) * 1e-12)
    except RuntimeError as error:
        raise ValueError("Sigma_read 必须为正半定") from error
    batched = clean.unsqueeze(0) if clean.ndim == 3 else clean
    shot_std = torch.sqrt(a[None, :, None, None] * batched.clamp_min(0.0))
    shot_white = torch.randn(batched.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    shot_correlated = torch.einsum("ij,bjhw->bihw", matrix, shot_white * shot_std)
    read_white = torch.randn(batched.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    read_correlated = torch.einsum("ij,bjhw->bihw", read_factor, read_white)
    noisy = batched + shot_correlated + read_correlated
    levels = float((1 << bit_depth) - 1)
    quantized = torch.round(noisy.clamp(0.0, 1.0) * levels) / levels
    return quantized.squeeze(0) if clean.ndim == 3 else quantized


def synthesize_sensor_noise(
    clean: torch.Tensor,
    shot_a: float,
    read_b: float,
    bit_depth: int = 12,
    row_sigma: float = 0.0,
    column_sigma: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """叠加 Shot/Read/行列/量化噪声，输出仍限制在归一化 RAW 域。"""

    if shot_a <= 0 or read_b < 0:
        raise ValueError("shot_a 必须为正数，read_b 不得为负数")
    if bit_depth not in (10, 12, 14, 16):
        raise ValueError("bit_depth 仅支持 10/12/14/16")
    variance = torch.clamp(clean, min=0.0) * shot_a + read_b
    gaussian = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    noisy = clean + gaussian * torch.sqrt(torch.clamp(variance, min=1e-12))
    if row_sigma > 0:
        row = torch.randn((*clean.shape[:-1], 1), dtype=clean.dtype, device=clean.device, generator=generator)
        noisy = noisy + row * row_sigma
    if column_sigma > 0:
        column = torch.randn((*clean.shape[:-2], 1, clean.shape[-1]), dtype=clean.dtype, device=clean.device, generator=generator)
        noisy = noisy + column * column_sigma
    levels = float((1 << bit_depth) - 1)
    return (torch.round(noisy.clamp(0.0, 1.0) * levels) / levels).clamp(0.0, 1.0)
