import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_isp.data.condition_v2 import ConditionMetadata, encode_condition_v2, validate_ryyb_release_condition_v2
from ai_isp.data.ryyb_contract import (
    AdmissionPolicy,
    RyybFrameDescriptor,
    RyybManifestRecord,
    RyybReleaseDataRequirements,
    load_ryyb_manifest,
    pack_ryyb,
    unpack_ryyb,
    validate_ai_admission,
    validate_release_dataset_requirements,
)
from ai_isp.data.ryyb_dataset import RyybRawPatchDataset


@pytest.mark.parametrize("pattern", ("ryyb", "byyr", "yryb", "ybyr"))
def test_ryyb_pack_unpack_is_bit_exact(pattern: str) -> None:
    raw = torch.arange(8 * 10, dtype=torch.int32).reshape(8, 10)
    packed = pack_ryyb(raw, pattern)
    assert packed.shape == (4, 4, 5)
    assert torch.equal(unpack_ryyb(packed, pattern), raw)


def test_admission_rejects_ultrawide_and_odd_crop() -> None:
    policy = AdmissionPolicy(
        ("main_ryyb_0", "tele_ryyb_0"),
        (("main_ryyb_0", "ryyb"), ("tele_ryyb_0", "byyr")),
        "model", "quant",
    )
    valid = RyybFrameDescriptor("main", "main_ryyb_0", "ryyb", model_hash="model", quant_policy_hash="quant")
    validate_ai_admission(valid, policy)
    with pytest.raises(ValueError):
        validate_ai_admission(RyybFrameDescriptor("ultrawide", "main_ryyb_0", "ryyb"), policy)
    with pytest.raises(ValueError):
        validate_ai_admission(RyybFrameDescriptor("main", "main_ryyb_0", "ryyb", crop_x=1), policy)
    with pytest.raises(ValueError):
        validate_ai_admission(RyybFrameDescriptor("tele", "tele_ryyb_0", "ryyb"), policy)


def test_release_condition_only_allows_main_or_tele_mapping() -> None:
    main = encode_condition_v2(ConditionMetadata(1 / 30, 1600, camera_type="main", sensor_profile="0", lens_profile="wide"))
    tele = encode_condition_v2(ConditionMetadata(1 / 30, 1600, camera_type="tele", sensor_profile="1", lens_profile="tele"))
    validate_ryyb_release_condition_v2(torch.stack((main, tele)))
    ultrawide = encode_condition_v2(ConditionMetadata(1 / 30, 1600, camera_type="ultrawide", lens_profile="ultrawide"))
    with pytest.raises(ValueError):
        validate_ryyb_release_condition_v2(ultrawide)


def test_manifest_rejects_non_target_camera(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "sample_id": "s0", "scene_id": "scene0", "split": "train", "camera_id": "ultrawide",
        "sensor_profile": "uw", "cfa_pattern": "ryyb", "noisy_path": "n.npy", "clean_path": "c.npy",
        "iso": 1600, "exposure_time_s": 0.01, "bit_depth": 12,
        "black_level": [64, 64, 64, 64], "white_level": [4095, 4095, 4095, 4095],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ryyb_manifest(manifest)


def test_manifest_dataset_returns_phase_safe_semantic_patch(tmp_path: Path) -> None:
    raw = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 4000 + 64
    np.save(tmp_path / "noisy.npy", raw)
    np.save(tmp_path / "clean.npy", raw)
    record = {
        "sample_id": "s0", "scene_id": "scene0", "split": "train", "camera_id": "main",
        "sensor_profile": "main_ryyb_0", "cfa_pattern": "ryyb",
        "noisy_path": "noisy.npy", "clean_path": "clean.npy", "iso": 1600,
        "exposure_time_s": 0.01, "bit_depth": 12,
        "black_level": [64, 64, 64, 64], "white_level": [4095, 4095, 4095, 4095],
        "lsc_profile_hash": "test-lsc-profile",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    sample = RyybRawPatchDataset(manifest, "train", patch_size=16, deterministic=True)[0]
    assert sample["noisy"].shape == (4, 16, 16)
    assert torch.equal(sample["noisy"], sample["clean"])
    validate_ryyb_release_condition_v2(sample["condition"])


def test_release_dataset_gate_counts_each_camera_and_split() -> None:
    records = []
    for camera in ("main", "tele"):
        for split in ("train", "validation", "blind"):
            records.append(RyybManifestRecord(
                sample_id=f"{camera}_{split}", scene_id=f"{camera}_{split}", split=split,
                camera_id=camera, sensor_profile=f"{camera}_ryyb_0", cfa_pattern="ryyb",
                noisy_path="n.npy", clean_path="c.npy", iso=1600, exposure_time_s=0.01,
                bit_depth=12, black_level=(64, 64, 64, 64), white_level=(4095, 4095, 4095, 4095),
            ))
    report = validate_release_dataset_requirements(records, RyybReleaseDataRequirements(1, 1, 1))
    assert report["main"]["train"] == 1
    assert report["tele"]["blind"] == 1
