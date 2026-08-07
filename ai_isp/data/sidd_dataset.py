"""SIDD Medium RAW 配对数据的低内存 Patch 数据集。"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .condition_v2 import ConditionMetadata, encode_condition_v2
from .pack_raw import CFA_OFFSETS, pack_bayer


@dataclass(frozen=True)
class SiddPair:
    scene_name: str
    noisy_path: Path
    clean_path: Path
    camera_id: str
    iso: float
    exposure_denominator: float
    cfa: str
    shot_a: float
    read_b: float


def _读取映射(csv_path: Path, key: str, value: str) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row[value] for row in csv.DictReader(handle)}


def _读取噪声参数(csv_path: Path) -> dict[str, tuple[float, float]]:
    if not csv_path.exists():
        return {}
    output: dict[str, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            shot = float(row["beta1_g"])
            read = float(row["beta2_g"])
            output[row["scene_instance_id"]] = (max(shot, 1e-6), max(read, 1e-10))
    return output


def discover_sidd_pairs(dataset_root: str | Path, metadata_root: str | Path | None = None) -> list[SiddPair]:
    """发现成对的 ``NOISY_RAW``/``GT_RAW`` MAT 文件并解析场景元数据。"""

    dataset_root = Path(dataset_root)
    metadata_root = Path(metadata_root) if metadata_root else dataset_root.parent / "SIDD_Blocks"
    cfa_by_camera = _读取映射(metadata_root / "bayer_patterns.csv", "camera_id", "bayer_pattern")
    noise_by_scene = _读取噪声参数(metadata_root / "noise_level_functions.csv")
    pairs: list[SiddPair] = []
    for noisy_path in sorted(dataset_root.rglob("*NOISY_RAW*.MAT")):
        clean_path = noisy_path.with_name(noisy_path.name.replace("NOISY_RAW", "GT_RAW"))
        if not clean_path.exists():
            continue
        scene_name = noisy_path.parent.name
        parts = scene_name.split("_")
        if len(parts) < 6:
            continue
        camera_id = parts[2]
        iso = float(parts[3])
        exposure_denominator = float(parts[4])
        cfa = cfa_by_camera.get(camera_id, "rggb").lower()
        if cfa not in CFA_OFFSETS:
            raise ValueError(f"场景 {scene_name} 的 CFA 非法: {cfa}")
        shot_a, read_b = noise_by_scene.get(scene_name, (1e-3, 1e-6))
        pairs.append(SiddPair(scene_name, noisy_path, clean_path, camera_id, iso, exposure_denominator, cfa, shot_a, read_b))
    if not pairs:
        raise FileNotFoundError(f"在 {dataset_root} 中没有发现 SIDD RAW 配对")
    return pairs


def split_pairs_by_scene(
    pairs: list[SiddPair], validation_ratio: float = 0.2, seed: int = 20260804
) -> tuple[list[SiddPair], list[SiddPair]]:
    """以场景目录为最小单元划分，阻止同一 Scene/Burst 跨集合泄漏。"""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio 必须位于 (0,1)")
    scenes = sorted({pair.scene_name for pair in pairs})
    random.Random(seed).shuffle(scenes)
    validation_count = max(1, round(len(scenes) * validation_ratio))
    validation_scenes = set(scenes[:validation_count])
    train = [pair for pair in pairs if pair.scene_name not in validation_scenes]
    validation = [pair for pair in pairs if pair.scene_name in validation_scenes]
    if not train or not validation:
        raise ValueError("场景数不足以同时形成训练集和验证集")
    return train, validation


def augment_packed_pair(
    noisy: torch.Tensor, clean: torch.Tensor, rotation_quarters: int, horizontal_flip: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """在语义 Packed 域同步旋转/镜像输入和 GT。

    R/Gr/Gb/B 已是显式语义平面，因此只变换空间格点，不交换通道；这等价于
    变换 2×2 Bayer 单元后重新按语义打包，避免把 Gr/Gb 误当普通 RGB 通道。
    """

    k = int(rotation_quarters) % 4
    if k % 2:
        raise ValueError("V6.1 绝对禁止 90°/270° Rotation")
    noisy = torch.rot90(noisy, k, dims=(-2, -1))
    clean = torch.rot90(clean, k, dims=(-2, -1))
    if horizontal_flip:
        noisy = torch.flip(noisy, dims=(-1,))
        clean = torch.flip(clean, dims=(-1,))
    return noisy.contiguous(), clean.contiguous()


class SiddRawPatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    """从 HDF5 MAT 中按需读取 Patch，避免把整幅 12MP RAW 常驻内存。"""

    def __init__(
        self,
        dataset_root: str | Path,
        patch_size: int = 128,
        samples_per_epoch: int = 128,
        seed: int = 20260804,
        max_pairs: int | None = None,
        deterministic: bool = False,
        augment: bool = True,
    ) -> None:
        if patch_size <= 0 or patch_size % 8:
            raise ValueError("Packed Patch 必须为正数且能被 8 整除")
        pairs = discover_sidd_pairs(dataset_root)
        self.pairs = pairs[:max_pairs] if max_pairs else pairs
        self.patch_size = patch_size
        self.mosaic_size = patch_size * 2
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.deterministic = deterministic
        self.augment = augment

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _shape(path: Path) -> tuple[int, int]:
        with h5py.File(path, "r") as handle:
            # MATLAB v7.3 的二维数组由 h5py 读取时轴顺序相反。
            width, height = handle["x"].shape
        return int(height), int(width)

    @staticmethod
    def _read_patch(path: Path, top: int, left: int, size: int) -> np.ndarray:
        with h5py.File(path, "r") as handle:
            # 在文件布局中先按 [x, y] 切片，再转为标准 [H, W]。
            patch = np.asarray(handle["x"][left:left + size, top:top + size], dtype=np.float32).T
        return np.ascontiguousarray(patch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = random.Random(self.seed + index) if self.deterministic else random
        pair = self.pairs[index % len(self.pairs)] if self.deterministic else rng.choice(self.pairs)
        height, width = self._shape(pair.noisy_path)
        if min(height, width) < self.mosaic_size:
            raise ValueError(f"场景 {pair.scene_name} 小于请求 Patch")
        # Bayer Crop 起点保持偶数，确保 CFA 相位不变。
        top = 2 * rng.randrange(0, (height - self.mosaic_size) // 2 + 1)
        left = 2 * rng.randrange(0, (width - self.mosaic_size) // 2 + 1)
        noisy = self._read_patch(pair.noisy_path, top, left, self.mosaic_size)
        clean = self._read_patch(pair.clean_path, top, left, self.mosaic_size)
        packed_noisy = torch.from_numpy(pack_bayer(noisy, pair.cfa))
        packed_clean = torch.from_numpy(pack_bayer(clean, pair.cfa))
        if self.augment:
            packed_noisy, packed_clean = augment_packed_pair(
                packed_noisy, packed_clean, rng.choice((0, 2)), bool(rng.randrange(2))
            )
        strength = rng.choices((0.0, 0.25, 0.5, 0.75, 1.0), weights=(10, 10, 10, 10, 50), k=1)[0]
        noise_level = float(torch.std(packed_noisy - packed_clean).clamp(0.0, 0.25))
        condition = encode_condition_v2(ConditionMetadata(
            exposure_time_s=1.0 / max(pair.exposure_denominator, 1.0),
            iso=pair.iso,
            analog_gain=max(pair.iso / 100.0, 1.0),
            noise_level=noise_level,
            noise_shot_a=pair.shot_a,
            noise_read_b=pair.read_b,
            scene_brightness=float(packed_noisy.mean()),
            camera_type="main",
            sensor_profile="other",
            lens_profile="wide",
            enhancement_strength=strength,
        ))
        noise_gt = strength * (packed_noisy - packed_clean)
        return {
            "noisy": packed_noisy,
            "clean": packed_clean,
            "noise_gt": noise_gt,
            "condition": condition,
            "scene_name": pair.scene_name,
            "cfa": pair.cfa,
            "smoke_only": True,
        }
