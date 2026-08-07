"""V3 规定的 RAW、Tone 与 Gradient 组合损失。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


def raw_charbonnier_loss(output: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """饱和区 Mask 与暗部加权 Charbonnier Loss。"""

    saturation_mask = (target < 0.98).to(target.dtype)
    shadow_weight = 1.0 + 2.0 * (1.0 - target.mean(dim=1, keepdim=True))
    difference = torch.sqrt((output - target).square() + epsilon * epsilon)
    weight = saturation_mask * shadow_weight
    return (difference * weight).sum() / weight.sum().clamp_min(1.0)


def fixed_reference_isp(packed: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
    """可微 RYYB Reference ISP 工程骨架；量产矩阵必须由两颗 Sensor 标定替换。"""

    # 3×4 光谱解混矩阵分别对应主摄和长焦。当前仅用于无目标数据的工程验链，
    # release_ready=false 时禁止把这里的数值解释为真实 RYYB 色彩标定。
    main_unmix = packed.new_tensor(((1.65, -0.30, -0.25, -0.10), (-0.12, 0.58, 0.58, -0.04), (-0.08, -0.22, -0.25, 1.55)))
    tele_unmix = packed.new_tensor(((1.58, -0.26, -0.22, -0.10), (-0.10, 0.56, 0.58, -0.04), (-0.06, -0.20, -0.24, 1.50)))
    main_rgb = torch.einsum("ij,bjhw->bihw", main_unmix, packed)
    tele_rgb = torch.einsum("ij,bjhw->bihw", tele_unmix, packed)
    if condition is None:
        rgb = main_rgb
    else:
        main_mask = condition[:, 10:11, None, None]
        tele_mask = condition[:, 12:13, None, None]
        fallback = (1.0 - main_mask - tele_mask).clamp(0.0, 1.0)
        rgb = main_rgb * (main_mask + fallback) + tele_rgb * tele_mask
    ccm = packed.new_tensor(((1.08, -0.05, -0.03), (-0.04, 1.08, -0.04), (-0.02, -0.08, 1.10)))
    rgb = torch.einsum("ij,bjhw->bihw", ccm, rgb)
    return torch.log1p(8.0 * rgb.clamp(0.0, 1.0)) / math.log(9.0)


def tone_loss(output: torch.Tensor, target: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
    return functional.l1_loss(fixed_reference_isp(output, condition), fixed_reference_isp(target, condition))


def _sobel(luma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = luma.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    return functional.conv2d(luma, kernel_x, padding=1), functional.conv2d(luma, kernel_y, padding=1)


def gradient_loss(output: torch.Tensor, target: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
    output_luma = (fixed_reference_isp(output, condition) * output.new_tensor((0.2126, 0.7152, 0.0722))[None, :, None, None]).sum(1, keepdim=True)
    target_luma = (fixed_reference_isp(target, condition) * target.new_tensor((0.2126, 0.7152, 0.0722))[None, :, None, None]).sum(1, keepdim=True)
    output_x, output_y = _sobel(output_luma)
    target_x, target_y = _sobel(target_luma)
    return functional.l1_loss(output_x, target_x) + functional.l1_loss(output_y, target_y)


class DarkPreviewLoss(nn.Module):
    """返回总损失和可单独记录的未加权分量。"""

    def __init__(self, raw_weight: float = 0.55, tone_weight: float = 0.30, gradient_weight: float = 0.15) -> None:
        super().__init__()
        self.raw_weight = raw_weight
        self.tone_weight = tone_weight
        self.gradient_weight = gradient_weight

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, condition: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        raw = raw_charbonnier_loss(output, target)
        tone = tone_loss(output, target, condition)
        gradient = gradient_loss(output, target, condition)
        total = self.raw_weight * raw + self.tone_weight * tone + self.gradient_weight * gradient
        return {"total": total, "raw": raw, "tone": tone, "gradient": gradient}


def _attention_map(feature: torch.Tensor) -> torch.Tensor:
    spatial = feature.abs().mean(dim=1, keepdim=True)
    return spatial / spatial.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def attention_distillation_loss(
    student_features: tuple[torch.Tensor, torch.Tensor],
    teacher_features: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Stage3/Middle 通道绝对值均值空间图归一化后等权 L1。"""

    if len(student_features) != 2 or len(teacher_features) != 2:
        raise ValueError("Attention KD 必须同时提供 Stage3 和 Middle 特征")
    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape[-2:] != teacher.shape[-2:]:
            raise ValueError("Attention KD 特征空间尺寸不一致")
        losses.append(functional.l1_loss(_attention_map(student), _attention_map(teacher)))
    return torch.stack(losses).mean()


def temporal_consistency_loss(
    clean: torch.Tensor,
    noisy_1: torch.Tensor,
    noise_pred_1: torch.Tensor,
    noisy_2: torch.Tensor,
    noise_pred_2: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    motion_mask: torch.Tensor | None = None,
    saturation_threshold: float = 0.98,
) -> torch.Tensor:
    """按 ``y_i=clip(z_i-N_i)`` 计算带高光/运动/有效区屏蔽的时序损失。"""

    tensors = (clean, noisy_1, noise_pred_1, noisy_2, noise_pred_2)
    if any(item.shape != clean.shape for item in tensors[1:]):
        raise ValueError("Temporal 输入、预测和 Clean 必须同 Shape")
    if not 0.0 < saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold 必须位于 (0,1]")
    y_1 = torch.clamp(noisy_1 - noise_pred_1, 0.0, 1.0)
    y_2 = torch.clamp(noisy_2 - noise_pred_2, 0.0, 1.0)
    mask = (
        (clean < saturation_threshold)
        & (noisy_1 < saturation_threshold)
        & (noisy_2 < saturation_threshold)
    ).to(clean.dtype)
    for external in (valid_mask, motion_mask):
        if external is not None:
            try:
                mask = mask * torch.broadcast_to(external.to(device=clean.device, dtype=clean.dtype), clean.shape)
            except RuntimeError as error:
                raise ValueError("Temporal Mask 无法广播到 RAW Shape") from error
    return (mask * (y_1 - y_2).abs()).sum() / mask.sum().clamp_min(1.0)


def gate_scale_alignment_loss(scale_pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """SimpleGate 两分支 Per-tensor Scale 的对数距离约束。"""

    if not scale_pairs:
        return torch.tensor(0.0)
    values = [
        (torch.log(left.clamp_min(1e-8)) - torch.log(right.clamp_min(1e-8))).abs().mean()
        for left, right in scale_pairs
    ]
    return torch.stack(values).mean()


@dataclass(frozen=True)
class StudentLossWeights:
    raw: float = 1.0
    tone: float = 0.5
    gradient: float = 0.1
    feature_kd: float = 0.1
    attention_kd: float = 0.05
    temporal: float = 0.05
    gate: float = 0.01


class StudentCompositeLoss(nn.Module):
    """V6.1 冻结的 Student/KD/QAT 完整损失组合。"""

    def __init__(self, weights: StudentLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or StudentLossWeights()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        feature_kd: torch.Tensor | None = None,
        attention_kd: torch.Tensor | None = None,
        temporal: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raw = raw_charbonnier_loss(output, target)
        tone = tone_loss(output, target, condition)
        gradient = gradient_loss(output, target, condition)
        zero = output.new_tensor(0.0)
        feature_kd = zero if feature_kd is None else feature_kd
        attention_kd = zero if attention_kd is None else attention_kd
        temporal = zero if temporal is None else temporal
        gate = zero if gate is None else gate
        weights = self.weights
        total = (
            weights.raw * raw
            + weights.tone * tone
            + weights.gradient * gradient
            + weights.feature_kd * feature_kd
            + weights.attention_kd * attention_kd
            + weights.temporal * temporal
            + weights.gate * gate
        )
        return {
            "total": total,
            "raw": raw,
            "tone": tone,
            "gradient": gradient,
            "feature_kd": feature_kd,
            "attention_kd": attention_kd,
            "temporal": temporal,
            "gate": gate,
        }


def raw_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    """计算归一化 RAW 的 Batch 平均 PSNR。"""

    mse = functional.mse_loss(output, target).detach().double().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def global_ssim(output: torch.Tensor, target: torch.Tensor) -> float:
    """无窗口全局 SSIM，仅用于快速 CPU 回归，不替代发布画质评测。"""

    x = output.detach().double().flatten(1)
    y = target.detach().double().flatten(1)
    mean_x, mean_y = x.mean(1), y.mean(1)
    variance_x, variance_y = x.var(1, unbiased=False), y.var(1, unbiased=False)
    covariance = ((x - mean_x[:, None]) * (y - mean_y[:, None])).mean(1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2))
    return float(score.mean())
