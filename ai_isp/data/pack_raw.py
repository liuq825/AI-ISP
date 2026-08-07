"""Bayer RAW 与固定 R/Gr/Gb/B Packed RAW 之间的转换。"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch


# 偏移顺序固定为 R、Gr、Gb、B。Gr 表示与 R 同行的绿色像素。
CFA_OFFSETS: Final[dict[str, tuple[tuple[int, int], ...]]] = {
    "rggb": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "bggr": ((1, 1), (1, 0), (0, 1), (0, 0)),
    "grbg": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "gbrg": ((1, 0), (1, 1), (0, 0), (0, 1)),
}


def _规范化_cfa(cfa: str) -> str:
    value = cfa.lower()
    if value not in CFA_OFFSETS:
        raise ValueError(f"不支持的 CFA: {cfa!r}")
    return value


def pack_bayer(raw: np.ndarray | torch.Tensor, cfa: str) -> np.ndarray | torch.Tensor:
    """将二维 Bayer RAW 打包为 ``4×H/2×W/2``，通道为 R/Gr/Gb/B。"""

    cfa = _规范化_cfa(cfa)
    if raw.ndim < 2:
        raise ValueError("RAW 至少需要两个空间维度")
    height, width = raw.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(f"Bayer 尺寸必须为偶数，收到 {height}×{width}")
    planes = [raw[..., row::2, col::2] for row, col in CFA_OFFSETS[cfa]]
    if isinstance(raw, torch.Tensor):
        return torch.stack(planes, dim=-3)
    return np.stack(planes, axis=-3)


def unpack_bayer(packed: np.ndarray | torch.Tensor, cfa: str) -> np.ndarray | torch.Tensor:
    """将固定通道顺序的 Packed RAW 位精确还原为 Bayer 格点。"""

    cfa = _规范化_cfa(cfa)
    if packed.ndim < 3 or packed.shape[-3] != 4:
        raise ValueError(f"Packed RAW 通道必须为 4，收到 {packed.shape}")
    height, width = packed.shape[-2:]
    output_shape = (*packed.shape[:-3], height * 2, width * 2)
    if isinstance(packed, torch.Tensor):
        output = torch.empty(output_shape, dtype=packed.dtype, device=packed.device)
    else:
        output = np.empty(output_shape, dtype=packed.dtype)
    for channel, (row, col) in enumerate(CFA_OFFSETS[cfa]):
        output[..., row::2, col::2] = packed[..., channel, :, :]
    return output


def normalize_packed_raw(
    packed: np.ndarray | torch.Tensor,
    black_level: float | tuple[float, float, float, float],
    white_level: float | tuple[float, float, float, float],
) -> np.ndarray | torch.Tensor:
    """按四通道 Black/White Level 归一化并裁剪到 [0, 1]。"""

    if packed.shape[-3] != 4:
        raise ValueError("归一化输入必须是四通道 Packed RAW")
    black = _通道常量(packed, black_level)
    white = _通道常量(packed, white_level)
    denominator = white - black
    if bool((denominator <= 0).any()):
        raise ValueError("White Level 必须逐通道大于 Black Level")
    normalized = (packed - black) / denominator
    return normalized.clamp(0.0, 1.0) if isinstance(normalized, torch.Tensor) else np.clip(normalized, 0.0, 1.0)


def normalize_post_blc_lsc_packed_raw(
    packed_post_blc_lsc: np.ndarray | torch.Tensor,
    black_level: float | tuple[float, float, float, float],
    white_level: float | tuple[float, float, float, float],
) -> np.ndarray | torch.Tensor:
    """归一化已完成 BLC/LSC 的四通道 RAW，禁止二次减 Black Level。

    ``black_level`` 只参与可用动态范围 ``white-black`` 的计算。调用方必须先通过
    RAW Domain 准入，证明输入为 ``LINEAR_POST_BLC_LSC_PRE_DGAIN``。
    """

    if packed_post_blc_lsc.shape[-3] != 4:
        raise ValueError("Post-BLC/LSC 归一化输入必须是四通道 Packed RAW")
    black = _通道常量(packed_post_blc_lsc, black_level)
    white = _通道常量(packed_post_blc_lsc, white_level)
    denominator = white - black
    if bool((denominator <= 0).any()):
        raise ValueError("White Level 必须逐通道大于 Black Level")
    normalized = packed_post_blc_lsc / denominator
    return normalized.clamp(0.0, 1.0) if isinstance(normalized, torch.Tensor) else np.clip(normalized, 0.0, 1.0)


def denormalize_packed_raw(
    normalized: np.ndarray | torch.Tensor,
    black_level: float | tuple[float, float, float, float],
    white_level: float | tuple[float, float, float, float],
) -> np.ndarray | torch.Tensor:
    """将归一化 Packed RAW 转回原始数值域。"""

    black = _通道常量(normalized, black_level)
    white = _通道常量(normalized, white_level)
    clipped = normalized.clamp(0.0, 1.0) if isinstance(normalized, torch.Tensor) else np.clip(normalized, 0.0, 1.0)
    return clipped * (white - black) + black


def _通道常量(
    reference: np.ndarray | torch.Tensor,
    value: float | tuple[float, float, float, float],
) -> np.ndarray | torch.Tensor:
    values = (value,) * 4 if isinstance(value, (float, int)) else tuple(value)
    if len(values) != 4:
        raise ValueError("Black/White Level 必须为标量或四通道值")
    shape = (1,) * (reference.ndim - 3) + (4, 1, 1)
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device).reshape(shape)
    return np.asarray(values, dtype=reference.dtype).reshape(shape)
