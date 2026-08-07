"""RYYB 主摄/长焦的固定输入、相位打包与失败闭锁契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Final

import numpy as np
import torch


RYYB_CHANNELS: Final[tuple[str, ...]] = ("R", "Yr", "Yb", "B")
RAW_DOMAIN_STATE: Final = "LINEAR_POST_BLC_LSC_PRE_DGAIN"
BUFFER_CONTRACT_VERSION: Final = "v1"
RYYB_CFA_OFFSETS: Final[dict[str, tuple[tuple[int, int], ...]]] = {
    # Pattern 名称按物理 2×2 宏像素逐行书写，输出统一为 R/Yr/Yb/B。
    "ryyb": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "byyr": ((1, 1), (1, 0), (0, 1), (0, 0)),
    "yryb": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "ybyr": ((1, 0), (1, 1), (0, 0), (0, 1)),
}
ALLOWED_CAMERAS: Final[tuple[str, ...]] = ("main", "tele")
FIXED_RAW_WIDTH: Final = 2048
FIXED_RAW_HEIGHT: Final = 1536
FIXED_PACKED_WIDTH: Final = 1024
FIXED_PACKED_HEIGHT: Final = 768


@dataclass(frozen=True)
class RyybFrameDescriptor:
    """HAL 交给 AI 节点前必须完整填写的帧描述。"""

    camera_id: str
    sensor_profile: str
    cfa_pattern: str
    raw_width: int = FIXED_RAW_WIDTH
    raw_height: int = FIXED_RAW_HEIGHT
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = FIXED_RAW_WIDTH
    crop_height: int = FIXED_RAW_HEIGHT
    row_stride_bytes: int = FIXED_RAW_WIDTH * 2
    bit_depth: int = 12
    black_level: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    white_level: tuple[float, float, float, float] = (4095.0, 4095.0, 4095.0, 4095.0)
    raw_domain_state: str = RAW_DOMAIN_STATE
    blc_applied: bool = True
    lsc_applied: bool = True
    raw_domain_profile_hash: str = ""
    lsc_profile_hash: str = ""
    unpack_profile_hash: str = ""
    buffer_contract_version: str = BUFFER_CONTRACT_VERSION
    buffer_fd: int = 0
    buffer_index: int = 0
    plane_offset_bytes: int = 0
    input_fence_fd: int = -1
    extra_cpu_memcpy_bytes: int = 0
    model_hash: str = ""
    quant_policy_hash: str = ""


@dataclass(frozen=True)
class AdmissionPolicy:
    """由发布 Manifest 注入的不可猜测准入信息。"""

    sensor_profiles: tuple[str, ...]
    sensor_cfa_phases: tuple[tuple[str, str], ...]
    model_hash: str = ""
    quant_policy_hash: str = ""
    raw_domain_profile_hash: str = ""
    lsc_profile_hashes: tuple[tuple[str, str], ...] = ()
    unpack_profile_hashes: tuple[tuple[str, str], ...] = ()
    buffer_contract_version: str = BUFFER_CONTRACT_VERSION


def validate_ai_admission(descriptor: RyybFrameDescriptor, policy: AdmissionPolicy) -> None:
    """验证 Camera、相位、固定 Shape、Buffer 和版本；失败时直接抛错供上层 Bypass。"""

    camera = descriptor.camera_id.lower()
    cfa = descriptor.cfa_pattern.lower()
    if camera not in ALLOWED_CAMERAS:
        raise ValueError(f"Camera {descriptor.camera_id!r} 不准入 AI RAW Denoise")
    if descriptor.sensor_profile not in policy.sensor_profiles:
        raise ValueError(f"Sensor Profile {descriptor.sensor_profile!r} 未注册")
    if cfa not in RYYB_CFA_OFFSETS:
        raise ValueError(f"CFA {descriptor.cfa_pattern!r} 不是已注册 RYYB 相位")
    expected_cfa = dict(policy.sensor_cfa_phases).get(descriptor.sensor_profile)
    if expected_cfa is None or cfa != expected_cfa:
        raise ValueError(f"Sensor {descriptor.sensor_profile!r} 的注册CFA相位不是 {cfa!r}")
    if descriptor.raw_domain_state != RAW_DOMAIN_STATE:
        raise ValueError(f"RAW Domain 必须为 {RAW_DOMAIN_STATE}")
    if not descriptor.blc_applied or not descriptor.lsc_applied:
        raise ValueError("AI 输入必须已经完成 BLC 和 LSC")
    if policy.raw_domain_profile_hash and descriptor.raw_domain_profile_hash != policy.raw_domain_profile_hash:
        raise ValueError("RAW Domain Profile Hash 不匹配")
    expected_lsc_hash = dict(policy.lsc_profile_hashes).get(descriptor.sensor_profile)
    if expected_lsc_hash is not None and descriptor.lsc_profile_hash != expected_lsc_hash:
        raise ValueError("LSC Profile Hash 不匹配")
    expected_unpack_hash = dict(policy.unpack_profile_hashes).get(descriptor.sensor_profile)
    if expected_unpack_hash is not None and descriptor.unpack_profile_hash != expected_unpack_hash:
        raise ValueError("Unpack Profile Hash 不匹配")
    if descriptor.buffer_contract_version != policy.buffer_contract_version:
        raise ValueError("Buffer Contract Version 不匹配")
    crop_values = (descriptor.crop_x, descriptor.crop_y, descriptor.crop_width, descriptor.crop_height)
    if any(value < 0 or value % 2 for value in crop_values):
        raise ValueError("RYYB Crop 起点和宽高必须是非负偶数，只能按 2×2 宏像素裁剪")
    if (descriptor.raw_width, descriptor.raw_height) != (FIXED_RAW_WIDTH, FIXED_RAW_HEIGHT):
        raise ValueError("上游 RAW 必须精确输出 2048×1536，禁止 Resize 或动态 Shape")
    if (descriptor.crop_width, descriptor.crop_height) != (FIXED_RAW_WIDTH, FIXED_RAW_HEIGHT):
        raise ValueError("有效 Crop 必须精确为 2048×1536")
    if descriptor.bit_depth not in (10, 12, 14, 16):
        raise ValueError("RYYB 位深仅允许 RAW10/12/14/16")
    if descriptor.row_stride_bytes < FIXED_RAW_WIDTH * 2:
        raise ValueError("RAW Row Stride 小于 uint16 容器的最小行字节数")
    if descriptor.buffer_fd < 0 or descriptor.buffer_index < 0:
        raise ValueError("DMA-BUF FD 和 Buffer Index 不得为负数")
    if descriptor.plane_offset_bytes < 0 or descriptor.plane_offset_bytes % 2:
        raise ValueError("Plane Offset 必须是非负且按 uint16 对齐")
    if descriptor.input_fence_fd < -1:
        raise ValueError("Input Fence FD 非法")
    if descriptor.extra_cpu_memcpy_bytes != 0:
        raise ValueError("V6 Buffer 契约禁止每帧额外 CPU memcpy")
    if len(descriptor.black_level) != 4 or len(descriptor.white_level) != 4:
        raise ValueError("Black/White Level 必须各包含四个语义通道")
    if any(white <= black for black, white in zip(descriptor.black_level, descriptor.white_level)):
        raise ValueError("每个通道的 White Level 必须大于 Black Level")
    if policy.model_hash and descriptor.model_hash != policy.model_hash:
        raise ValueError("模型 Hash 不匹配")
    if policy.quant_policy_hash and descriptor.quant_policy_hash != policy.quant_policy_hash:
        raise ValueError("量化策略 Hash 不匹配")


def _normalize_pattern(cfa_pattern: str) -> str:
    pattern = cfa_pattern.lower()
    if pattern not in RYYB_CFA_OFFSETS:
        raise ValueError(f"不支持的 RYYB CFA: {cfa_pattern!r}")
    return pattern


def pack_ryyb(raw: np.ndarray | torch.Tensor, cfa_pattern: str) -> np.ndarray | torch.Tensor:
    """把二维 RYYB Mosaic 打包为语义固定的 ``R/Yr/Yb/B`` 四平面。"""

    pattern = _normalize_pattern(cfa_pattern)
    if raw.ndim < 2:
        raise ValueError("RAW 至少需要两个空间维度")
    height, width = raw.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("RYYB RAW 的宽高必须是偶数")
    planes = [raw[..., row::2, column::2] for row, column in RYYB_CFA_OFFSETS[pattern]]
    return torch.stack(planes, dim=-3) if isinstance(raw, torch.Tensor) else np.stack(planes, axis=-3)


def unpack_ryyb(packed: np.ndarray | torch.Tensor, cfa_pattern: str) -> np.ndarray | torch.Tensor:
    """把语义四平面位精确还原到指定 RYYB 相位。"""

    pattern = _normalize_pattern(cfa_pattern)
    if packed.ndim < 3 or packed.shape[-3] != 4:
        raise ValueError("Packed RYYB 必须包含四个语义通道")
    height, width = packed.shape[-2:]
    shape = (*packed.shape[:-3], height * 2, width * 2)
    output = (
        torch.empty(shape, dtype=packed.dtype, device=packed.device)
        if isinstance(packed, torch.Tensor)
        else np.empty(shape, dtype=packed.dtype)
    )
    for channel, (row, column) in enumerate(RYYB_CFA_OFFSETS[pattern]):
        output[..., row::2, column::2] = packed[..., channel, :, :]
    return output


def reconstruct_and_unpack_ryyb(
    packed_raw: np.ndarray | torch.Tensor,
    noise_pred: np.ndarray | torch.Tensor,
    cfa_pattern: str,
) -> np.ndarray | torch.Tensor:
    """在语义平面执行 Subtract/Clamp，再按 Sensor 物理相位恢复二维 Mosaic。"""

    if packed_raw.shape != noise_pred.shape:
        raise ValueError("packed_raw 与 noise_pred 必须同 Shape")
    if isinstance(packed_raw, torch.Tensor) != isinstance(noise_pred, torch.Tensor):
        raise TypeError("packed_raw 与 noise_pred 必须使用相同 Tensor 类型")
    reconstructed = packed_raw - noise_pred
    reconstructed = (
        reconstructed.clamp(0.0, 1.0)
        if isinstance(reconstructed, torch.Tensor)
        else np.clip(reconstructed, 0.0, 1.0)
    )
    return unpack_ryyb(reconstructed, cfa_pattern)


@dataclass(frozen=True)
class RyybManifestRecord:
    """量产 RYYB JSONL Manifest 的最小稳定字段。"""

    sample_id: str
    scene_id: str
    split: str
    camera_id: str
    sensor_profile: str
    cfa_pattern: str
    noisy_path: str
    clean_path: str
    iso: float
    exposure_time_s: float
    bit_depth: int
    black_level: tuple[float, float, float, float]
    white_level: tuple[float, float, float, float]
    analog_gain: float = 1.0
    digital_gain: float = 1.0
    sensor_temperature_c: float = 25.0
    scene_brightness: float = 0.5
    scene_ev: float = 0.0
    noise_level: float = 0.0
    noise_shot_a: float = 1e-6
    noise_read_b: float = 1e-10
    burst_id: str = ""
    smoke_only: bool = False
    raw_domain_state: str = RAW_DOMAIN_STATE
    blc_applied: bool = True
    lsc_applied: bool = True
    lsc_profile_hash: str = ""


@dataclass(frozen=True)
class RyybReleaseDataRequirements:
    """每颗 Camera 的独立场景发布下限。"""

    train_scenes: int = 3000
    validation_scenes: int = 300
    blind_scenes: int = 500


def validate_release_dataset_requirements(
    records: list[RyybManifestRecord],
    requirements: RyybReleaseDataRequirements | None = None,
) -> dict[str, dict[str, int]]:
    """按 Camera/Split 统计唯一场景，并拒绝任何 Smoke 数据混入量产集。"""

    requirements = requirements or RyybReleaseDataRequirements()
    expected = {
        "train": requirements.train_scenes,
        "validation": requirements.validation_scenes,
        "blind": requirements.blind_scenes,
    }
    if any(record.smoke_only for record in records):
        raise ValueError("量产 RYYB Manifest 不得包含 smoke_only=true 的样本")
    report: dict[str, dict[str, int]] = {}
    for camera in ALLOWED_CAMERAS:
        report[camera] = {}
        for split, minimum in expected.items():
            scenes = {record.scene_id for record in records if record.camera_id == camera and record.split == split}
            report[camera][split] = len(scenes)
            if len(scenes) < minimum:
                raise ValueError(f"{camera}/{split} 独立场景不足: {len(scenes)} < {minimum}")
    return report


def load_ryyb_manifest(path: str | Path) -> list[RyybManifestRecord]:
    """读取 JSONL，并在训练前拒绝字段缺失、重复 Sample 和非目标 Camera。"""

    path = Path(path)
    records: list[RyybManifestRecord] = []
    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                item["black_level"] = tuple(item["black_level"])
                item["white_level"] = tuple(item["white_level"])
                record = RyybManifestRecord(**item)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Manifest 第 {line_number} 行非法: {error}") from error
            if record.sample_id in sample_ids:
                raise ValueError(f"Manifest Sample ID 重复: {record.sample_id}")
            if record.camera_id not in ALLOWED_CAMERAS:
                raise ValueError(f"Manifest 含非目标 Camera: {record.camera_id}")
            if record.cfa_pattern not in RYYB_CFA_OFFSETS:
                raise ValueError(f"Manifest 含非法 RYYB 相位: {record.cfa_pattern}")
            if record.split not in ("train", "validation", "blind"):
                raise ValueError(f"Manifest Split 只允许 train/validation/blind: {record.split}")
            if not record.scene_id or not record.sensor_profile:
                raise ValueError("Manifest 的 scene_id/sensor_profile 不能为空")
            if record.bit_depth not in (10, 12, 14, 16):
                raise ValueError(f"Manifest 位深非法: {record.bit_depth}")
            if record.raw_domain_state != RAW_DOMAIN_STATE or not record.blc_applied or not record.lsc_applied:
                raise ValueError("Manifest RAW 必须处于 Post-BLC/LSC Pre-DGain 域")
            if not record.smoke_only and not record.lsc_profile_hash:
                raise ValueError("量产 RYYB Manifest 必须记录 lsc_profile_hash")
            if len(record.black_level) != 4 or len(record.white_level) != 4:
                raise ValueError("Manifest Black/White Level 必须各包含四通道")
            if any(white <= black for black, white in zip(record.black_level, record.white_level)):
                raise ValueError("Manifest 每通道 White Level 必须大于 Black Level")
            if record.iso <= 0 or record.exposure_time_s <= 0:
                raise ValueError("Manifest ISO 和曝光时间必须大于0")
            for raw_path in (record.noisy_path, record.clean_path):
                resolved = Path(raw_path) if Path(raw_path).is_absolute() else path.parent / raw_path
                if not resolved.is_file():
                    raise ValueError(f"Manifest RAW 文件不存在: {resolved}")
            sample_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError("RYYB Manifest 为空")
    sensor_contracts: dict[str, tuple[str, str]] = {}
    for record in records:
        contract = (record.camera_id, record.cfa_pattern)
        previous = sensor_contracts.setdefault(record.sensor_profile, contract)
        if previous != contract:
            raise ValueError(
                f"Sensor {record.sensor_profile!r} 的Camera/CFA不唯一: {previous} 与 {contract}"
            )
    return records
