from pathlib import Path

from ai_isp.export.static_profiles import export_static_profiles
from ai_isp.models.mobile_nafnet import build_mobile_nafnet_w16


def test_three_static_onnx_smoke_exports_are_aligned(tmp_path: Path) -> None:
    report = export_static_profiles(build_mobile_nafnet_w16(), tmp_path, profile_mode="smoke")
    assert set(report) == {"P0", "P1", "P2"}
    for item in report.values():
        assert item["alignment"]["max_abs_error"] <= 1e-4
        assert item["dynamic_slice_inputs"] == []
        assert item["unsupported_operators"] == []

