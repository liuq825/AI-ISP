"""Stage3/Middle INT8 Dynamic Affine 的整数数学参考实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class QuantTensorSpec:
    scale: float
    zero_point: int = 0
    qmin: int = -128
    qmax: int = 127

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("Quant Scale 必须大于 0")
        if not self.qmin <= self.zero_point <= self.qmax:
            raise ValueError("Zero-point 超出整数范围")


def _quantize(value: torch.Tensor, spec: QuantTensorSpec) -> torch.Tensor:
    return torch.round(value / spec.scale + spec.zero_point).clamp(spec.qmin, spec.qmax).to(torch.int32)


def _dequantize(value: torch.Tensor, spec: QuantTensorSpec) -> torch.Tensor:
    return (value.to(torch.float32) - spec.zero_point) * spec.scale


def integer_dynamic_affine_reference(
    feature: torch.Tensor,
    gamma_affine: torch.Tensor,
    beta_affine: torch.Tensor,
    feature_spec: QuantTensorSpec,
    gamma_spec: QuantTensorSpec,
    beta_spec: QuantTensorSpec,
    output_spec: QuantTensorSpec,
) -> tuple[torch.Tensor, dict[str, object]]:
    """INT8×INT8→INT32，Bias 对齐到乘积 Scale 后统一 Requant。"""

    q_feature = _quantize(feature, feature_spec)
    q_gamma = _quantize(gamma_affine, gamma_spec)
    q_beta = _quantize(beta_affine, beta_spec)
    product_scale = feature_spec.scale * gamma_spec.scale
    product = (q_feature - feature_spec.zero_point) * (q_gamma - gamma_spec.zero_point)
    beta_real = (q_beta - beta_spec.zero_point).to(torch.float64) * beta_spec.scale
    beta_accumulator = torch.round(beta_real / product_scale).to(torch.int32)
    accumulator = product + beta_accumulator
    requantized = torch.round(
        accumulator.to(torch.float64) * product_scale / output_spec.scale
    ).to(torch.int64) + output_spec.zero_point
    requantized = requantized.clamp(output_spec.qmin, output_spec.qmax).to(torch.int32)
    output = _dequantize(requantized, output_spec)
    audit = {
        "feature": asdict(feature_spec),
        "gamma": asdict(gamma_spec),
        "beta": asdict(beta_spec),
        "output": asdict(output_spec),
        "product_scale": product_scale,
        "int32_abs_max": int(accumulator.abs().max()),
        "int32_overflow": bool(accumulator.abs().max() > torch.iinfo(torch.int32).max),
    }
    return output, audit


def audit_dynamic_affine_equivalence(
    feature: torch.Tensor,
    gamma_affine: torch.Tensor,
    beta_affine: torch.Tensor,
) -> dict[str, object]:
    """比较 FakeQuant 浮点表达与整数参考表达，供无 ATC 环境阻断审计。"""

    def symmetric_spec(value: torch.Tensor) -> QuantTensorSpec:
        scale = float(value.detach().abs().max().clamp_min(1e-8) / 127.0)
        return QuantTensorSpec(scale=scale)

    feature_spec = symmetric_spec(feature)
    gamma_spec = symmetric_spec(gamma_affine)
    beta_spec = symmetric_spec(beta_affine)
    fake = _dequantize(_quantize(feature, feature_spec), feature_spec) * _dequantize(
        _quantize(gamma_affine, gamma_spec), gamma_spec
    ) + _dequantize(_quantize(beta_affine, beta_spec), beta_spec)
    output_spec = symmetric_spec(fake)
    fake_requant = _dequantize(_quantize(fake, output_spec), output_spec)
    integer, integer_audit = integer_dynamic_affine_reference(
        feature,
        gamma_affine,
        beta_affine,
        feature_spec,
        gamma_spec,
        beta_spec,
        output_spec,
    )
    error = (fake_requant - integer).abs()
    return {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "integer": integer_audit,
        "dynamic_affine_target_pending": True,
    }
