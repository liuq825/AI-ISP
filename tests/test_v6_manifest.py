from pathlib import Path

import pytest

from ai_isp.export.om_release import (
    V6_1_REQUIRED_ARTIFACTS,
    build_v6_1_engineering_manifest,
    promote_v6_1_manifest,
)


def test_v6_manifest_contains_all_hashes_and_cannot_promote_without_target_evidence(tmp_path: Path) -> None:
    artifacts = {}
    for name in V6_1_REQUIRED_ARTIFACTS:
        filename = "model_mixed_qat.safetensors" if name == "weights" else f"{name}.json"
        path = tmp_path / filename
        path.write_text(name, encoding="utf-8")
        artifacts[name] = path
    manifest = build_v6_1_engineering_manifest(
        tmp_path / "model_manifest_v6_1.json", artifacts, "P36-16"
    )
    assert manifest["development_selected"] is True
    assert manifest["dynamic_affine_target_pending"] is True
    assert manifest["target_validated"] is False
    assert manifest["release_ready"] is False
    assert manifest["smoke_only"] is False
    assert manifest["unpack_profile_hash"]
    assert manifest["lsc_profile_hash"]
    evidence = {
        "real_ryyb_quality": True,
        "dynamic_affine_no_fp16_fallback": False,
        "npu_coverage_100_percent": True,
        "latency_6_8_9_10ms": True,
        "photo_preview_30fps": True,
        "dmabuf_contract": True,
        "power_thermal_stability": True,
        "ten_thousand_frames": True,
        "rollback_verified": True,
    }
    with pytest.raises(RuntimeError):
        promote_v6_1_manifest(manifest, evidence)
