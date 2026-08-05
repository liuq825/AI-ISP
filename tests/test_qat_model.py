from pathlib import Path

import torch

from ai_isp.export.static_profiles import export_fixed_model
from ai_isp.models import build_mobile_nafnet_w16
from ai_isp.quantization import (
    FilmPrecisionGateResult,
    QatPolicy,
    calibrate_qat_model,
    configure_qat_phase,
    prepare_qat_model,
    prepare_qdq_export,
)
from ai_isp.qat_training import QatTrainingConfig, train_qat


def _inputs():
    image = torch.rand(1, 4, 32, 48)
    condition = torch.zeros(1, 24)
    condition[0, (10, 14, 18, 22, 23)] = 1.0
    return image, condition


def test_qat_phases_and_q3_freeze_do_not_change_forward() -> None:
    image, condition = _inputs()
    model = prepare_qat_model(build_mobile_nafnet_w16(), QatPolicy(activation_offset=True))
    calibrate_qat_model(model, [(image, condition)])
    configure_qat_phase(model, "q2")
    before = model(image, condition)
    configure_qat_phase(model, "q3")
    after = model(image, condition)
    assert torch.equal(before, after)


def test_qdq_onnx_is_explicit_and_aligned(tmp_path: Path) -> None:
    image, condition = _inputs()
    model = prepare_qat_model(build_mobile_nafnet_w16(), QatPolicy())
    calibrate_qat_model(model, [(image, condition)])
    prepare_qdq_export(model)
    item = export_fixed_model(model, tmp_path, "smoke")["RYYB_4X3"]
    assert "QuantizeLinear" in item["operators"]
    assert "DequantizeLinear" in item["operators"]
    assert item["alignment"]["max_abs_error"] <= 1e-4


def test_qat_can_resume_from_phase_checkpoint_in_new_output_dir(tmp_path: Path) -> None:
    image, condition = _inputs()
    loader = [{"noisy": image, "clean": image, "condition": condition}]
    first_config = QatTrainingConfig(
        output_dir=str(tmp_path / "first"), device="cpu",
        q1_steps=1, q2_steps=1, q3_steps=1, checkpoint_interval=1,
    )
    train_qat(build_mobile_nafnet_w16(), loader, [(image, condition)], QatPolicy(), first_config)
    resume = tmp_path / "first" / "checkpoints" / "q2_step_1.pt"
    second_config = QatTrainingConfig(
        output_dir=str(tmp_path / "second"), device="cpu",
        q1_steps=1, q2_steps=1, q3_steps=1, checkpoint_interval=1,
        resume_from=str(resume),
    )
    _, report = train_qat(build_mobile_nafnet_w16(), loader, [(image, condition)], QatPolicy(), second_config)
    assert {item["phase"] for item in report["history"]} == {"q3"}
    assert report["resumed_from"] == str(resume)


def test_film_fp16_requires_both_quality_trigger_and_target_gates() -> None:
    passing = FilmPrecisionGateResult(0.04, 0.005, False, True, True, 7.9)
    assert passing.select_fp16_npu_island() is True
    assert FilmPrecisionGateResult(0.04, 0.005, False, True, False, 7.0).select_fp16_npu_island() is False
    assert FilmPrecisionGateResult(0.01, 0.005, False, True, True, 7.0).select_fp16_npu_island() is False
