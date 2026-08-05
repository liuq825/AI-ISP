import hashlib
import json
from pathlib import Path

import yaml

from ai_isp.runtime.profiles import FIXED_RYYB_PROFILE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_project_json_and_yaml_are_parseable() -> None:
    roots = (Path("configs"), Path("artifacts"))
    for root in roots:
        for path in root.rglob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))
    for path in Path("configs").rglob("*.yaml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))


def test_condition_and_profile_config_match_code() -> None:
    condition = json.loads(Path("configs/release/condition_schema_v2.json").read_text(encoding="utf-8"))
    assert condition["shape"] == [1, 24]
    assert [field["index"] for field in condition["fields"]] == list(range(24))
    runtime = json.loads(Path("configs/runtime/dark_preview_profiles.json").read_text(encoding="utf-8"))
    assert len(runtime["profiles"]) == 1
    item = runtime["profiles"][0]
    profile = FIXED_RYYB_PROFILE
    assert item["id"] == profile.profile_id
    assert (item["valid_width"], item["valid_height"]) == (profile.valid_width, profile.valid_height)
    assert (item["raw_width"], item["raw_height"]) == (profile.raw_width, profile.raw_height)


def test_engineering_manifest_is_safe_and_hashes_local_artifacts() -> None:
    manifest = json.loads(Path("artifacts/release/model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_ready"] is False
    assert manifest["om"]["available"] is False
    assert manifest["artifact_layout"] == "single_static_ryyb_4x3"
    for key in ("condition_schema", "sensor_profiles"):
        item = manifest[key]
        assert _sha256(Path(item["path"])) == item["sha256"]
    quant = manifest["quantization"]
    assert _sha256(Path(quant["policy"])) == quant["policy_sha256"]
    weights = Path(manifest["weights"]["path"])
    if weights.exists():
        assert _sha256(weights) == manifest["weights"]["sha256"]
    item = manifest["onnx"]
    path = Path(item["path"])
    if path.exists():
        assert _sha256(path) == item["sha256"]
