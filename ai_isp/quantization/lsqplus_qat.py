"""LSQ/LSQ+ W8A8 量化感知训练、校准、Q/DQ 导出与审计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Iterable

import torch
import torch.nn.functional as functional
from torch import nn

from ai_isp.models.mobile_nafnet import MobileNAFBlockDW
from ai_isp.models.static_simple_gate import StaticSimpleGate


def round_ste(value: torch.Tensor) -> torch.Tensor:
    """前向取整、反向恒等的 Straight-Through Estimator。"""

    return value + (torch.round(value) - value).detach()


def grad_scale(value: torch.Tensor, factor: float) -> torch.Tensor:
    """前向保持原值，把反向梯度缩放为 LSQ 规定的量级。"""

    return (value - value * factor).detach() + value * factor


class _QdqExportFunction(torch.autograd.Function):
    """Eager 执行 Fake Quant，ONNX 导出显式 QuantizeLinear/DequantizeLinear。"""

    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int):
        scale = scale.clamp_min(1e-8)
        if axis < 0:
            quantized = torch.round(value / scale) + zero_point.to(value.dtype)
            quantized = quantized.clamp(-128, 127)
            return (quantized - zero_point.to(value.dtype)) * scale
        shape = [1] * value.ndim
        shape[axis] = scale.numel()
        view_scale = scale.reshape(shape)
        view_zero = zero_point.to(value.dtype).reshape(shape)
        quantized = (torch.round(value / view_scale) + view_zero).clamp(-128, 127)
        return (quantized - view_zero) * view_scale

    @staticmethod
    def symbolic(graph, value, scale, zero_point, axis):
        if axis < 0:
            quantized = graph.op("QuantizeLinear", value, scale, zero_point)
            return graph.op("DequantizeLinear", quantized, scale, zero_point)
        quantized = graph.op("QuantizeLinear", value, scale, zero_point, axis_i=axis)
        return graph.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)


class LearnableFakeQuant(nn.Module):
    """支持 LSQ 对称与 LSQ+ 可学习 Offset 的训练/导出量化器。"""

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        learnable_offset: bool = True,
        channel_count: int | None = None,
        channel_axis: int = 0,
        observer_max_samples: int = 65536,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("量化位宽必须位于 [2,16]")
        self.bits = bits
        self.symmetric = symmetric
        self.channel_axis = channel_axis if channel_count is not None else -1
        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1
        shape = (channel_count,) if channel_count is not None else ()
        self.log_scale = nn.Parameter(torch.full(shape, -4.0))
        self.offset = nn.Parameter(
            torch.zeros(shape), requires_grad=(learnable_offset and not symmetric)
        )
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("observer_enabled", torch.tensor(False))
        self.register_buffer("export_mode", torch.tensor(False))
        self.register_buffer("frozen_scale", torch.ones(shape))
        self.register_buffer("frozen_zero_point", torch.zeros(shape, dtype=torch.int8))
        self.observer_max_samples = int(observer_max_samples)
        self._samples: list[torch.Tensor] = []

    @property
    def scale(self) -> torch.Tensor:
        return functional.softplus(self.log_scale).clamp_min(1e-8)

    def _reshape(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale
        offset = torch.zeros_like(self.offset) if self.symmetric else self.offset
        if self.channel_axis < 0:
            return scale, offset
        shape = [1] * value.ndim
        shape[self.channel_axis] = scale.numel()
        return scale.reshape(shape), offset.reshape(shape)

    def initialize_from(self, value: torch.Tensor) -> None:
        """以张量范围初始化；per-channel 权重沿非通道维独立计算。"""

        detached = value.detach().float()
        with torch.no_grad():
            if self.channel_axis < 0:
                minimum, maximum = detached.amin(), detached.amax()
            else:
                dimensions = tuple(index for index in range(detached.ndim) if index != self.channel_axis)
                minimum, maximum = detached.amin(dim=dimensions), detached.amax(dim=dimensions)
            if self.symmetric:
                target_scale = torch.maximum(minimum.abs(), maximum.abs()) / max(self.qmax, 1)
                target_offset = torch.zeros_like(target_scale)
            else:
                target_scale = (maximum - minimum).clamp_min(1e-8) / max(self.qmax - self.qmin, 1)
                # beta 是反量化域 Offset；让 qmin 对齐观测最小值。
                target_offset = minimum - self.qmin * target_scale
            target_scale = target_scale.clamp_min(1e-8)
            self.log_scale.copy_(torch.log(torch.expm1(target_scale).clamp_min(1e-12)))
            self.offset.copy_(target_offset)
            self.initialized.fill_(True)

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled.fill_(enabled)
        if enabled:
            self._samples.clear()

    def _observe(self, value: torch.Tensor) -> None:
        if self.channel_axis >= 0:
            return
        remaining = self.observer_max_samples - sum(sample.numel() for sample in self._samples)
        if remaining <= 0:
            return
        flat = value.detach().float().flatten()
        if flat.numel() > remaining:
            step = max(flat.numel() // remaining, 1)
            flat = flat[::step][:remaining]
        self._samples.append(flat.cpu())

    @staticmethod
    def _balanced_bucket_mse(original: torch.Tensor, quantized: torch.Tensor) -> float:
        minimum, maximum = original.min(), original.max()
        if float(maximum - minimum) < 1e-12:
            return float(functional.mse_loss(quantized, original))
        normalized = (original - minimum) / (maximum - minimum)
        boundaries = (0.0, 0.02, 0.25, 0.75, 1.000001)
        scores = []
        for lower, upper in zip(boundaries, boundaries[1:]):
            mask = (normalized >= lower) & (normalized < upper)
            if bool(mask.any()):
                denominator = original[mask].square().mean().clamp_min(1e-8)
                scores.append(functional.mse_loss(quantized[mask], original[mask]) / denominator)
        return float(torch.stack(scores).mean()) if scores else 0.0

    def finalize_calibration(
        self,
        percentile_candidates: tuple[float, ...] = (99.9, 99.95, 99.99, 100.0),
    ) -> dict[str, float]:
        """用亮度桶等权 MSE 选择范围，避免暗部数量掩盖高光。"""

        if self.channel_axis >= 0:
            self.observer_enabled.fill_(False)
            return {"percentile": 100.0, "balanced_mse": 0.0, "saturation_rate": 0.0}
        if not self._samples:
            raise ValueError("量化器没有收集到校准样本")
        values = torch.cat(self._samples)
        best: tuple[float, float, float, float, float] | None = None
        for percentile in percentile_candidates:
            tail = max((100.0 - percentile) / 200.0, 0.0)
            minimum = torch.quantile(values, tail) if tail else values.min()
            maximum = torch.quantile(values, 1.0 - tail) if tail else values.max()
            scale = (maximum - minimum).clamp_min(1e-8) / max(self.qmax - self.qmin, 1)
            offset = torch.tensor(0.0) if self.symmetric else minimum - self.qmin * scale
            if self.symmetric:
                scale = torch.maximum(minimum.abs(), maximum.abs()) / max(self.qmax, 1)
            quantized = (torch.round((values - offset) / scale).clamp(self.qmin, self.qmax) * scale + offset)
            score = self._balanced_bucket_mse(values, quantized)
            q = (values - offset) / scale
            saturation = float(((q <= self.qmin) | (q >= self.qmax)).float().mean())
            candidate = (score, saturation, percentile, float(scale), float(offset))
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        assert best is not None
        with torch.no_grad():
            scale_tensor = torch.tensor(best[3], device=self.log_scale.device).clamp_min(1e-8)
            self.log_scale.copy_(torch.log(torch.expm1(scale_tensor).clamp_min(1e-12)))
            self.offset.copy_(torch.zeros_like(self.offset) if self.symmetric else torch.tensor(best[4], device=self.offset.device))
            self.initialized.fill_(True)
            self.observer_enabled.fill_(False)
        self._samples.clear()
        return {"percentile": best[2], "balanced_mse": best[0], "saturation_rate": best[1]}

    def freeze_for_export(self) -> None:
        """把连续 Offset 吸收到可由 ONNX int8 Zero-point 表示的部署网格。"""

        with torch.no_grad():
            zero_point = torch.round(-self.offset / self.scale).clamp(self.qmin, self.qmax)
            if not self.symmetric:
                self.offset.copy_(-zero_point * self.scale)
            self.frozen_scale.copy_(self.scale)
            self.frozen_zero_point.copy_(zero_point.to(torch.int8))
            self.log_scale.requires_grad_(False)
            self.offset.requires_grad_(False)
            self.export_mode.fill_(True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            # Q/DQ 导出前必须调用 prepare_qdq_export；这里避免把状态判断追踪进静态图。
            return _QdqExportFunction.apply(value, self.frozen_scale, self.frozen_zero_point, self.channel_axis)
        if bool(self.observer_enabled):
            self._observe(value)
        if not bool(self.initialized):
            self.initialize_from(value)
        if bool(self.export_mode):
            return _QdqExportFunction.apply(value, self.frozen_scale, self.frozen_zero_point, self.channel_axis)
        scale, offset = self._reshape(value)
        factor = 1.0 / math.sqrt(max(value.numel() * self.qmax, 1))
        scale = grad_scale(scale, factor)
        offset = grad_scale(offset, factor)
        quantized = round_ste((value - offset) / scale).clamp(self.qmin, self.qmax)
        return quantized * scale + offset

    def audit(self) -> dict[str, object]:
        scale = self.scale.detach().cpu().flatten()
        offset = (torch.zeros_like(self.offset) if self.symmetric else self.offset).detach().cpu().flatten()
        return {
            "bits": self.bits,
            "symmetric": self.symmetric,
            "per_channel": self.channel_axis >= 0,
            "channel_axis": self.channel_axis,
            "scale": float(scale[0]) if scale.numel() == 1 else scale.tolist(),
            "offset": float(offset[0]) if offset.numel() == 1 else offset.tolist(),
            "qmin": self.qmin,
            "qmax": self.qmax,
            "initialized": bool(self.initialized),
            "export_mode": bool(self.export_mode),
        }


class QatConv2d(nn.Module):
    """输入、权重和输出均显式 Fake Quant 的卷积。"""

    def __init__(self, conv: nn.Conv2d, activation_offset: bool) -> None:
        super().__init__()
        self.conv = copy.deepcopy(conv)
        self.input_quant = LearnableFakeQuant(8, symmetric=not activation_offset, learnable_offset=activation_offset)
        self.weight_quant = LearnableFakeQuant(8, symmetric=True, learnable_offset=False, channel_count=conv.out_channels)
        self.output_quant = LearnableFakeQuant(8, symmetric=not activation_offset, learnable_offset=activation_offset)
        self.weight_quant.initialize_from(self.conv.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_quant(value)
        weight = self.weight_quant(self.conv.weight)
        output = functional.conv2d(value, weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return self.output_quant(output)


class QatLinear(nn.Module):
    """Condition Encoder 使用的可审计 Linear QAT 包装。"""

    def __init__(self, linear: nn.Linear, activation_offset: bool) -> None:
        super().__init__()
        self.linear = copy.deepcopy(linear)
        self.input_quant = LearnableFakeQuant(8, symmetric=not activation_offset, learnable_offset=activation_offset)
        self.weight_quant = LearnableFakeQuant(8, symmetric=True, learnable_offset=False, channel_count=linear.out_features)
        self.output_quant = LearnableFakeQuant(8, symmetric=not activation_offset, learnable_offset=activation_offset)
        self.weight_quant.initialize_from(self.linear.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output_quant(functional.linear(self.input_quant(value), self.weight_quant(self.linear.weight), self.linear.bias))


@dataclass(frozen=True)
class QatPolicy:
    activation_offset: bool = False
    film_precision: str = "int8"
    weight_bits: int = 8
    activation_bits: int = 8

    def __post_init__(self) -> None:
        if self.film_precision not in ("int8", "fp16_npu"):
            raise ValueError("FiLM 精度只允许 int8 或 fp16_npu")
        if self.weight_bits != 8 or self.activation_bits != 8:
            raise ValueError("V4 发布策略固定为 W8A8")


def _replace_layers(parent: nn.Module, policy: QatPolicy, prefix: str = "") -> None:
    for name, child in list(parent.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        protect_film = policy.film_precision == "fp16_npu" and path.startswith("condition_encoder")
        if isinstance(child, nn.Conv2d):
            setattr(parent, name, QatConv2d(child, policy.activation_offset))
        elif isinstance(child, nn.Linear) and not protect_film:
            setattr(parent, name, QatLinear(child, policy.activation_offset))
        else:
            _replace_layers(child, policy, path)


def prepare_qat_model(model: nn.Module, policy: QatPolicy) -> nn.Module:
    """在模型副本上插入量化器，Dense/Pruned FP32 模型保持不变。"""

    qat_model = copy.deepcopy(model)
    _replace_layers(qat_model, policy)
    for module in qat_model.modules():
        if isinstance(module, StaticSimpleGate):
            module.output_quant = LearnableFakeQuant(8, symmetric=not policy.activation_offset, learnable_offset=policy.activation_offset)
        elif isinstance(module, MobileNAFBlockDW):
            module.spatial_residual_quant = LearnableFakeQuant(8, symmetric=not policy.activation_offset, learnable_offset=policy.activation_offset)
            module.channel_residual_quant = LearnableFakeQuant(8, symmetric=not policy.activation_offset, learnable_offset=policy.activation_offset)
    return qat_model


def iter_quantizers(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LearnableFakeQuant):
            yield name, module


def calibrate_qat_model(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Q0：收集激活并以亮度分桶均衡 MSE 初始化全部量化器。"""

    quantizers = list(iter_quantizers(model))
    for _, quantizer in quantizers:
        if quantizer.channel_axis < 0:
            quantizer.enable_observer(True)
    training = model.training
    model.eval()
    with torch.no_grad():
        for image, condition in batches:
            model(image, condition)
    model.train(training)
    report = {}
    for name, quantizer in quantizers:
        if quantizer.channel_axis < 0:
            report[name] = quantizer.finalize_calibration()
    return report


def configure_qat_phase(model: nn.Module, phase: str) -> None:
    """Q1仅量化参数、Q2联合训练、Q3固定量化网格。"""

    if phase not in ("q1", "q2", "q3"):
        raise ValueError("QAT Phase 只允许 q1/q2/q3")
    quantizer_parameters = {id(parameter) for _, quantizer in iter_quantizers(model) for parameter in quantizer.parameters()}
    for parameter in model.parameters():
        is_quantizer = id(parameter) in quantizer_parameters
        parameter.requires_grad_(is_quantizer if phase == "q1" else not (phase == "q3" and is_quantizer))


def prepare_qdq_export(model: nn.Module) -> dict[str, dict[str, object]]:
    """冻结全部量化器并返回逐层 Quant Policy 审计。"""

    report = {}
    for name, quantizer in iter_quantizers(model):
        quantizer.freeze_for_export()
        report[name] = quantizer.audit()
    return report


def audit_qat_model(model: nn.Module) -> dict[str, dict[str, object]]:
    return {name: quantizer.audit() for name, quantizer in iter_quantizers(model)}


def audit_film_quantization(fp32_model: nn.Module, qat_model: nn.Module) -> dict[str, object]:
    """用固定 main/tele 条件分别记录 FiLM gamma/beta 量化误差。"""

    if not hasattr(fp32_model, "condition_encoder") or not hasattr(qat_model, "condition_encoder"):
        raise ValueError("FiLM 审计要求模型包含 condition_encoder")
    fp32_encoder = copy.deepcopy(fp32_model.condition_encoder).eval().cpu()
    qat_encoder = copy.deepcopy(qat_model.condition_encoder).eval().cpu()
    conditions = torch.full((2, 24), 0.5, dtype=torch.float32)
    conditions[:, 10:22] = 0.0
    conditions[0, (10, 14, 18)] = 1.0
    conditions[1, (12, 15, 20)] = 1.0
    conditions[:, 22:24] = 1.0
    with torch.no_grad():
        references = fp32_encoder(conditions)
        candidates = qat_encoder(conditions)
    per_camera: dict[str, dict[str, float]] = {}
    for batch_index, camera in enumerate(("main", "tele")):
        gamma_errors = []
        beta_errors = []
        for reference, candidate in zip(references, candidates):
            channels = reference.shape[1] // 2
            gamma_errors.append((reference[batch_index, :channels] - candidate[batch_index, :channels]).abs())
            beta_errors.append((reference[batch_index, channels:] - candidate[batch_index, channels:]).abs())
        gamma = torch.cat(gamma_errors)
        beta = torch.cat(beta_errors)
        per_camera[camera] = {
            "gamma_mean_abs_error": float(gamma.mean()),
            "gamma_max_abs_error": float(gamma.max()),
            "beta_mean_abs_error": float(beta.mean()),
            "beta_max_abs_error": float(beta.max()),
        }
    quantizers = {
        name: quantizer.audit()
        for name, quantizer in iter_quantizers(qat_model)
        if name.startswith("condition_encoder")
    }
    return {"per_camera": per_camera, "condition_encoder_quantizers": quantizers}


@dataclass(frozen=True)
class OffsetMicrobenchmarkResult:
    compile_succeeded: bool
    npu_only: bool
    fusion_preserved: bool
    symmetric_p95_ms: float
    asymmetric_p95_ms: float

    def __post_init__(self) -> None:
        if self.compile_succeeded and (self.symmetric_p95_ms <= 0.0 or self.asymmetric_p95_ms <= 0.0):
            raise ValueError("Offset微基准P95必须为正数")

    def allow_learnable_offset(self) -> bool:
        if not (self.compile_succeeded and self.npu_only and self.fusion_preserved):
            return False
        regression = self.asymmetric_p95_ms - self.symmetric_p95_ms
        return regression <= min(0.1, self.symmetric_p95_ms * 0.03)


@dataclass(frozen=True)
class FilmPrecisionGateResult:
    """FiLM FP16 候选必须先由画质触发，再同时通过部署门禁。"""

    psnr_drop_db: float
    critical_saturation_rate: float
    camera_bucket_abnormal: bool
    fp16_compiled: bool
    fp16_npu_only: bool
    fp16_p95_ms: float

    def __post_init__(self) -> None:
        if self.psnr_drop_db < 0.0 or not 0.0 <= self.critical_saturation_rate <= 1.0:
            raise ValueError("FiLM画质门禁指标范围非法")
        if self.fp16_compiled and self.fp16_p95_ms <= 0.0:
            raise ValueError("FiLM FP16 P95必须为正数")

    def quality_triggered(self) -> bool:
        return should_enable_film_fp16(
            self.psnr_drop_db,
            self.critical_saturation_rate,
            self.camera_bucket_abnormal,
        )

    def select_fp16_npu_island(self) -> bool:
        return (
            self.quality_triggered()
            and self.fp16_compiled
            and self.fp16_npu_only
            and self.fp16_p95_ms <= 8.0
        )


def should_enable_film_fp16(
    psnr_drop_db: float,
    critical_saturation_rate: float,
    camera_bucket_abnormal: bool,
) -> bool:
    """FiLM FP16 只由画质证据触发，最终仍需 NPU/时延门禁。"""

    return psnr_drop_db > 0.03 or critical_saturation_rate > 0.01 or camera_bucket_abnormal
