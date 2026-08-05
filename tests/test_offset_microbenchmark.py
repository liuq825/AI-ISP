import json
from pathlib import Path

from ai_isp.export.quant_microbenchmark import export_offset_microbenchmark_pair
from ai_isp.models import build_mobile_nafnet_w16


def test_offset_microbenchmark_exports_both_qdq_paths_and_fails_closed(tmp_path: Path) -> None:
    report = export_offset_microbenchmark_pair(build_mobile_nafnet_w16(), tmp_path)
    assert report["executed_before_long_training"] is True
    assert report["allow_learnable_offset"] is False
    assert report["candidates"]["symmetric"]["contains_qdq"] is True
    assert report["candidates"]["asymmetric"]["contains_qdq"] is True


def test_offset_microbenchmark_accepts_only_passing_target_result(tmp_path: Path) -> None:
    result = tmp_path / "target.json"
    result.write_text(json.dumps({
        "compile_succeeded": True,
        "npu_only": True,
        "fusion_preserved": True,
        "symmetric_p95_ms": 1.0,
        "asymmetric_p95_ms": 1.02,
    }), encoding="utf-8")
    report = export_offset_microbenchmark_pair(build_mobile_nafnet_w16(), tmp_path / "onnx", result)
    assert report["allow_learnable_offset"] is True
