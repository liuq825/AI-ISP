from pathlib import Path

from ai_isp.export.static_profiles import export_fixed_model
from ai_isp.models.mobile_nafnet import build_mobile_nafnet_w16


def test_single_static_onnx_smoke_export_is_aligned(tmp_path: Path) -> None:
    report = export_fixed_model(build_mobile_nafnet_w16(), tmp_path, profile_mode="smoke")
    assert set(report) == {"RYYB_4X3"}
    for item in report.values():
        assert item["alignment"]["max_abs_error"] <= 1e-4
        assert item["dynamic_slice_inputs"] == []
        assert item["forbidden_gate_operators"] == []
        assert item["unsupported_operators"] == []
