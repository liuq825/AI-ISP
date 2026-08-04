"""Torch-Pruning 之外的 MobileNAFNet 静态拓扑业务校验器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ai_isp.models.static_simple_gate import StaticSimpleGate


@dataclass(frozen=True)
class ValidationIssue:
    module_name: str
    message: str


class NAFNetPruningValidator:
    """验证 Gate、DWConv、残差参数、FiLM 和 8 通道对齐。"""

    def __init__(self, round_to: int = 8) -> None:
        self.round_to = round_to

    def validate(self, model: nn.Module, example_image: torch.Tensor, example_condition: torch.Tensor) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for name, module in model.named_modules():
            if isinstance(module, StaticSimpleGate):
                if module.channels <= 0 or module.channels % self.round_to:
                    issues.append(ValidationIssue(name, f"Gate channels={module.channels} 未按 {self.round_to} 对齐"))
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                if not (module.groups == module.in_channels == module.out_channels):
                    issues.append(ValidationIssue(name, "DWConv 必须满足 groups=in_channels=out_channels"))
            if isinstance(module, nn.Conv2d) and module.out_channels >= self.round_to and module.out_channels % self.round_to:
                issues.append(ValidationIssue(name, f"输出通道 {module.out_channels} 未按 {self.round_to} 对齐"))
        for name, parameter in model.named_parameters():
            if name.endswith(("beta", "gamma", "scale")) and (parameter.ndim != 4 or parameter.shape[0] != 1):
                issues.append(ValidationIssue(name, "残差缩放参数必须为 1×C×1×1"))
        try:
            with torch.no_grad():
                output = model(example_image, example_condition)
            if output.shape != example_image.shape:
                issues.append(ValidationIssue("<output>", f"输出 {output.shape} 与输入 {example_image.shape} 不同格点"))
            if not torch.isfinite(output).all():
                issues.append(ValidationIssue("<output>", "输出包含 NaN/Inf"))
        except Exception as error:  # noqa: BLE001 - 校验器需要把异常转成结构化报告
            issues.append(ValidationIssue("<forward>", str(error)))
        return issues

    def assert_valid(self, model: nn.Module, example_image: torch.Tensor, example_condition: torch.Tensor) -> None:
        issues = self.validate(model, example_image, example_condition)
        if issues:
            detail = "; ".join(f"{item.module_name}: {item.message}" for item in issues)
            raise ValueError(f"剪枝拓扑验证失败: {detail}")

