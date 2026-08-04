import torch

from ai_isp.quantization.lsqplus_qat import LearnableFakeQuant
from ai_isp.quantization.ptq_validate import validate_ptq_tensor


def test_lsqplus_initialization_and_audit() -> None:
    value = torch.linspace(-0.2, 1.0, 1000)
    quantizer = LearnableFakeQuant(bits=8, symmetric=False, learnable_offset=True)
    quantizer.initialize_from(value)
    output = quantizer(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    audit = quantizer.audit()
    assert audit["scale"] > 0
    report = validate_ptq_tensor(value)
    assert report["max_abs_error"] < 0.01


def test_symmetric_weight_quantizer_forces_zero_offset() -> None:
    quantizer = LearnableFakeQuant(bits=8, symmetric=True, learnable_offset=True)
    quantizer.initialize_from(torch.randn(100))
    assert quantizer.audit()["offset"] == 0.0
    assert not quantizer.offset.requires_grad

