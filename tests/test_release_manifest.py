import hashlib
import json
from pathlib import Path

import yaml

from ai_isp.runtime.profiles import PROFILES


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
    for item in runtime["profiles"]:
        profile = PROFILES[item["id"]]
        assert (item["valid_width"], item["valid_height"]) == (profile.valid_width, profile.valid_height)
        assert (item["compile_width"], item["compile_height"]) == (profile.compile_width, profile.compile_height)


def test_engineering_manifest_is_safe_and_hashes_local_artifacts() -> None:
    manifest = json.loads(Path("artifacts/release/model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_ready"] is False
    assert manifest["om"]["available"] is False
    weights = Path(manifest["weights"]["path"])
    if weights.exists():
        assert _sha256(weights) == manifest["weights"]["sha256"]
    for item in manifest["onnx"]:
        path = Path(item["path"])
        if path.exists():
            assert _sha256(path) == item["sha256"]

