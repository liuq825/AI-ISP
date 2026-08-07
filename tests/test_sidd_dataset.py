from pathlib import Path

import pytest

import torch

from ai_isp.data.sidd_dataset import (
    SiddRawPatchDataset,
    augment_packed_pair,
    discover_sidd_pairs,
    split_pairs_by_scene,
)


DATASET_ROOT = Path("datasets/SIDD_Training_Subset")


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="本地未提供 SIDD Training Subset")
def test_real_sidd_pair_and_patch_can_be_read() -> None:
    pairs = discover_sidd_pairs(DATASET_ROOT)
    assert len(pairs) >= 6
    dataset = SiddRawPatchDataset(DATASET_ROOT, patch_size=32, samples_per_epoch=1, max_pairs=1, deterministic=True)
    sample = dataset[0]
    assert sample["noisy"].shape == (4, 32, 32)
    assert sample["clean"].shape == (4, 32, 32)
    assert sample["condition"].shape == (24,)
    assert sample["cfa"] in {"rggb", "bggr", "grbg", "gbrg"}


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="本地未提供 SIDD Training Subset")
def test_scene_split_has_no_leakage() -> None:
    train, validation = split_pairs_by_scene(discover_sidd_pairs(DATASET_ROOT), validation_ratio=0.2)
    assert {pair.scene_name for pair in train}.isdisjoint({pair.scene_name for pair in validation})


def test_packed_augmentation_keeps_noisy_clean_alignment() -> None:
    noisy = torch.arange(4 * 8 * 12).reshape(4, 8, 12)
    clean = noisy + 7
    with pytest.raises(ValueError, match="90"):
        augment_packed_pair(noisy, clean, rotation_quarters=1, horizontal_flip=True)
    aug_noisy, aug_clean = augment_packed_pair(noisy, clean, rotation_quarters=2, horizontal_flip=True)
    assert aug_noisy.shape == (4, 8, 12)
    assert torch.equal(aug_clean - aug_noisy, torch.full_like(aug_noisy, 7))
