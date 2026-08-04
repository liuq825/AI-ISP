"""ConditionSchemaV2 的唯一编码实现。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, log2

import torch


CONDITION_NAMES = (
    "exposure_time_s", "iso", "analog_gain", "digital_gain",
    "noise_level", "noise_shot_a", "noise_read_b", "sensor_temperature_c",
    "scene_brightness", "scene_ev", "camera_main", "camera_ultrawide",
    "camera_tele", "camera_other", "sensor_profile_0", "sensor_profile_1",
    "sensor_profile_2", "sensor_profile_other", "lens_wide", "lens_ultrawide",
    "lens_tele", "lens_other", "metadata_valid", "enhancement_strength",
)


@dataclass(frozen=True)
class ConditionMetadata:
    """构造 24 维 Condition 所需的物理元数据。"""

    exposure_time_s: float
    iso: float
    analog_gain: float = 1.0
    digital_gain: float = 1.0
    noise_level: float = 0.0
    noise_shot_a: float = 1e-6
    noise_read_b: float = 1e-10
    sensor_temperature_c: float = 25.0
    scene_brightness: float = 0.5
    scene_ev: float = 0.0
    camera_type: str = "other"
    sensor_profile: str = "other"
    lens_profile: str = "other"
    metadata_valid: bool = True
    enhancement_strength: float = 1.0


def _one_hot(value: str, labels: tuple[str, ...]) -> list[float]:
    normalized = value.lower()
    if normalized not in labels:
        normalized = "other"
    return [float(normalized == label) for label in labels]


def encode_condition_v2(metadata: ConditionMetadata) -> torch.Tensor:
    """将物理元数据编码为经过 Clamp 的 ``float32[24]``。"""

    # log 输入先用规格下限保护，再统一 Clamp，避免非法元数据产生 NaN/Inf。
    continuous = [
        (log2(max(metadata.exposure_time_s, 1.0 / 16000.0)) - log2(1.0 / 16000.0)) / log2(16000.0),
        log2(max(metadata.iso, 50.0) / 50.0) / 9.0,
        log2(max(metadata.analog_gain, 1.0)) / 6.0,
        log2(max(metadata.digital_gain, 1.0)) / 3.0,
        metadata.noise_level / 0.25,
        (log10(max(metadata.noise_shot_a, 1e-6)) + 6.0) / 5.0,
        (log10(max(metadata.noise_read_b, 1e-10)) + 10.0) / 7.0,
        (metadata.sensor_temperature_c + 20.0) / 120.0,
        metadata.scene_brightness,
        (metadata.scene_ev + 8.0) / 16.0,
    ]
    values = (
        continuous
        + _one_hot(metadata.camera_type, ("main", "ultrawide", "tele", "other"))
        + _one_hot(metadata.sensor_profile, ("0", "1", "2", "other"))
        + _one_hot(metadata.lens_profile, ("wide", "ultrawide", "tele", "other"))
        + [float(metadata.metadata_valid), metadata.enhancement_strength]
    )
    condition = torch.tensor(values, dtype=torch.float32).clamp_(0.0, 1.0)
    if condition.numel() != 24:
        raise AssertionError("ConditionSchemaV2 必须恰好为 24 维")
    return condition


def validate_condition_v2(condition: torch.Tensor) -> None:
    """验证维度、有限值、范围和三个 one-hot 组。"""

    if condition.shape[-1] != 24:
        raise ValueError(f"Condition 最后一维必须为 24，收到 {condition.shape}")
    if not torch.isfinite(condition).all():
        raise ValueError("Condition 含 NaN 或 Inf")
    if bool(((condition < 0.0) | (condition > 1.0)).any()):
        raise ValueError("Condition 必须位于 [0,1]")
    for start in (10, 14, 18):
        group_sum = condition[..., start:start + 4].sum(dim=-1)
        if not torch.allclose(group_sum, torch.ones_like(group_sum), atol=1e-6):
            raise ValueError(f"Condition one-hot 组 {start}:{start + 4} 非法")

