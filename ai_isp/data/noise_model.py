"""用于数据增强与链路冒烟的可复现物理噪声模型。"""

from __future__ import annotations

import torch


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

