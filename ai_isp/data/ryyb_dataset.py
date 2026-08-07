"""Manifest 驱动的真实 RYYB 配对 Patch 数据集。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .condition_v2 import ConditionMetadata, encode_condition_v2
from .pack_raw import normalize_post_blc_lsc_packed_raw
from .ryyb_contract import RyybManifestRecord, load_ryyb_manifest, pack_ryyb


def validate_scene_splits(records: list[RyybManifestRecord]) -> None:
    """禁止同一 Scene/Burst 跨 train/validation/blind 泄漏。"""

    owners: dict[str, str] = {}
    for record in records:
        keys = (f"scene:{record.scene_id}", f"burst:{record.burst_id}") if record.burst_id else (f"scene:{record.scene_id}",)
        for key in keys:
            previous = owners.setdefault(key, record.split)
            if previous != record.split:
                raise ValueError(f"{key} 同时出现在 {previous} 和 {record.split}")


class RyybRawPatchDataset(Dataset[dict[str, torch.Tensor | str | bool]]):
    """从规范化前 uint16/float NPY 按需读取主摄或长焦 RYYB Patch。"""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        patch_size: int | tuple[int, int] = 256,
        samples_per_epoch: int | None = None,
        seed: int = 20260804,
        deterministic: bool = False,
    ) -> None:
        patch_shape = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        if len(patch_shape) != 2 or any(value <= 0 or value % 16 for value in patch_shape):
            raise ValueError("Packed RYYB Patch 高宽必须为正数且能被 16 整除")
        records = load_ryyb_manifest(manifest_path)
        validate_scene_splits(records)
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"Manifest 没有 split={split!r} 的样本")
        self.manifest_root = Path(manifest_path).resolve().parent
        self.patch_height, self.patch_width = patch_shape
        self.mosaic_height, self.mosaic_width = self.patch_height * 2, self.patch_width * 2
        self.samples_per_epoch = samples_per_epoch or len(self.records)
        self.seed = seed
        self.deterministic = deterministic

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest_root / path

    @staticmethod
    def _load_mosaic(path: Path) -> np.ndarray:
        if path.suffix.lower() != ".npy":
            raise ValueError(f"规范 RYYB 数据只接受 NPY；请先转换 {path}")
        value = np.load(path, mmap_mode="r")
        if value.ndim != 2:
            raise ValueError(f"RYYB NPY 必须为二维 Mosaic: {path}")
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        rng = random.Random(self.seed + index) if self.deterministic else random
        record = self.records[index % len(self.records)] if self.deterministic else rng.choice(self.records)
        noisy_raw = self._load_mosaic(self._resolve(record.noisy_path))
        clean_raw = self._load_mosaic(self._resolve(record.clean_path))
        if noisy_raw.shape != clean_raw.shape:
            raise ValueError(f"{record.sample_id} 的 Noisy/Clean Shape 不一致")
        height, width = noisy_raw.shape
        if height < self.mosaic_height or width < self.mosaic_width:
            raise ValueError(f"{record.sample_id} 小于请求的 RYYB Patch")
        # 起点乘 2，保证任何训练 Crop 都不改变 2×2 CFA 相位。
        top = 2 * rng.randrange(0, (height - self.mosaic_height) // 2 + 1)
        left = 2 * rng.randrange(0, (width - self.mosaic_width) // 2 + 1)
        noisy_patch = np.asarray(
            noisy_raw[top : top + self.mosaic_height, left : left + self.mosaic_width], dtype=np.float32
        )
        clean_patch = np.asarray(
            clean_raw[top : top + self.mosaic_height, left : left + self.mosaic_width], dtype=np.float32
        )
        noisy = torch.from_numpy(np.ascontiguousarray(pack_ryyb(noisy_patch, record.cfa_pattern)))
        clean = torch.from_numpy(np.ascontiguousarray(pack_ryyb(clean_patch, record.cfa_pattern)))
        noisy = normalize_post_blc_lsc_packed_raw(noisy, record.black_level, record.white_level)
        clean = normalize_post_blc_lsc_packed_raw(clean, record.black_level, record.white_level)
        if rng.randrange(2):
            noisy, clean = torch.flip(noisy, (-1,)), torch.flip(clean, (-1,))
        if rng.randrange(2):
            noisy, clean = torch.flip(noisy, (-2,)), torch.flip(clean, (-2,))
        camera = record.camera_id
        condition = encode_condition_v2(ConditionMetadata(
            exposure_time_s=record.exposure_time_s,
            iso=record.iso,
            analog_gain=record.analog_gain,
            digital_gain=record.digital_gain,
            noise_level=record.noise_level,
            noise_shot_a=record.noise_shot_a,
            noise_read_b=record.noise_read_b,
            sensor_temperature_c=record.sensor_temperature_c,
            scene_brightness=record.scene_brightness,
            scene_ev=record.scene_ev,
            camera_type=camera,
            sensor_profile="0" if camera == "main" else "1",
            lens_profile="wide" if camera == "main" else "tele",
            enhancement_strength=1.0,
        ))
        return {
            "noisy": noisy.contiguous(),
            "clean": clean.contiguous(),
            "noise_gt": (noisy - clean).contiguous(),
            "condition": condition,
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "camera_id": camera,
            "smoke_only": record.smoke_only,
        }


class BalancedCameraSampler(Sampler[int]):
    """按 main/tele 交替取样，使一个偶数长度窗口包含等量 Camera。"""

    def __init__(self, dataset: RyybRawPatchDataset, seed: int = 20260804) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.indices = {
            camera: [index for index, record in enumerate(dataset.records) if record.camera_id == camera]
            for camera in ("main", "tele")
        }
        if not all(self.indices.values()):
            raise ValueError("平衡采样要求 Manifest 同时包含 main 和 tele")

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {name: list(values) for name, values in self.indices.items()}
        for values in pools.values():
            rng.shuffle(values)
        positions = {"main": 0, "tele": 0}
        for index in range(len(self)):
            camera = "main" if index % 2 == 0 else "tele"
            pool = pools[camera]
            yield pool[positions[camera] % len(pool)]
            positions[camera] += 1
