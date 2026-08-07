"""MobileNAFNet 的 Torch-Pruning 依赖图、结构化裁剪与业务校验。"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Iterable

import torch
from torch import nn
import torch_pruning as tp

from ai_isp.models.mobile_nafnet import MobileNAFBlockDW, MobileNAFNetW16
from ai_isp.models.static_simple_gate import StaticSimpleGate


@dataclass(frozen=True)
class ValidationIssue:
    module_name: str
    message: str


@dataclass(frozen=True)
class PruningReport:
    target_ratio: float
    parameter_count_before: int
    parameter_count_after: int
    parameter_reduction: float
    macs_before: int
    macs_after: int
    mac_reduction: float
    feature_channels: tuple[int, int, int, int]
    rounds: tuple[dict[str, object], ...]


def _unwrapped_parameters(model: nn.Module) -> list[tuple[nn.Parameter, int]]:
    output: list[tuple[nn.Parameter, int]] = []
    for module in model.modules():
        for attribute in ("beta", "gamma", "scale"):
            parameter = getattr(module, attribute, None)
            if isinstance(parameter, nn.Parameter) and parameter.ndim == 4:
                output.append((parameter, 1))
    return output


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_macs(model: nn.Module, image: torch.Tensor, condition: torch.Tensor) -> int:
    """通过 Hook 统计 Conv/Linear 的 MAC，固定 Shape 报告使用同一口径。"""

    total = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal total
        batch, out_channels, out_height, out_width = output.shape
        kernel = module.kernel_size[0] * module.kernel_size[1]
        total += batch * out_channels * out_height * out_width * (module.in_channels // module.groups) * kernel

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal total
        total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    training = model.training
    model.eval()
    with torch.no_grad():
        model(image, condition)
    model.train(training)
    for hook in hooks:
        hook.remove()
    return int(total)


def estimate_macs_at_shape(
    model: nn.Module,
    example_image: torch.Tensor,
    condition: torch.Tensor,
    target_height: int = 768,
    target_width: int = 1024,
) -> int:
    """以小图 Hook 捕获层级比例，外推发布 Shape 的 Conv/Linear MAC。"""

    source_height, source_width = example_image.shape[-2:]
    if target_height % source_height or target_width % source_width:
        raise ValueError("目标 Shape 必须是示例 Shape 的整数倍，避免 MAC 口径漂移")
    height_ratio, width_ratio = target_height // source_height, target_width // source_width
    total = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output) -> None:
        nonlocal total
        batch, output_channels, output_height, output_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        mac_per_output = (module.in_channels // module.groups) * kernel_height * kernel_width
        total += (
            batch
            * output_channels
            * output_height
            * height_ratio
            * output_width
            * width_ratio
            * mac_per_output
        )

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal total
        total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            model(example_image, condition)
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)


class NAFNetPruningValidator:
    """验证并执行满足 Gate、DWConv、Skip、FiLM 与 16 通道约束的剪枝。"""

    minimum_widths = (16, 32, 48, 96)

    def __init__(self, round_to: int = 16) -> None:
        if round_to != 16:
            raise ValueError("V6.1 结构化剪枝只允许 round_to=16")
        self.round_to = round_to

    def build_dependency_graph(
        self,
        model: nn.Module,
        example_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tp.DependencyGraph:
        """显式登记残差参数的通道维，消除 Torch-Pruning 猜测。"""

        return tp.DependencyGraph().build_dependency(
            model,
            example_inputs=example_inputs,
            unwrapped_parameters=_unwrapped_parameters(model),
            verbose=False,
        )

    def validate_group(self, graph: tp.DependencyGraph, group) -> None:
        if not graph.check_pruning_group(group):
            raise ValueError("Torch-Pruning DependencyGroup 非法或会裁空通道")
        if not list(group):
            raise ValueError("Torch-Pruning DependencyGroup 为空")

    def update_static_attributes(self, model: nn.Module) -> None:
        """裁剪后同步 Torch-Pruning 无法理解的 StaticSimpleGate 属性。"""

        for module in model.modules():
            if not isinstance(module, MobileNAFBlockDW):
                continue
            if module.spatial_expand.out_channels % 2 or module.channel_expand.out_channels % 2:
                raise ValueError("Gate 展开通道必须为偶数")
            module.spatial_gate.channels = module.spatial_expand.out_channels // 2
            module.channel_gate.channels = module.channel_expand.out_channels // 2
        if isinstance(model, MobileNAFNetW16):
            model.film_stage2.channels = model.downs[0].out_channels
            model.film_stage3.channels = model.downs[1].out_channels
            model.film_middle.channels = model.downs[2].out_channels

    @staticmethod
    def _blocks_for_stage(model: MobileNAFNetW16, stage_index: int) -> list[MobileNAFBlockDW]:
        if stage_index == 1:
            containers = (model.encoders[1], model.decoders[1])
        elif stage_index == 2:
            containers = (model.encoders[2], model.decoders[0])
        elif stage_index == 3:
            containers = (model.middle,)
        else:
            raise ValueError("Stage 1 不剪，剪枝 Stage 仅允许 1/2/3")
        return [module for container in containers for module in container if isinstance(module, MobileNAFBlockDW)]

    @staticmethod
    def _paired_hidden_indices(expand: nn.Conv2d, project: nn.Conv2d, pair_count: int) -> list[int]:
        half = expand.out_channels // 2
        if pair_count <= 0 or pair_count >= half:
            raise ValueError("Gate 隐藏通道裁剪数量非法")
        weight_score = expand.weight.detach().abs().flatten(1).mean(1)
        project_score = project.weight.detach().abs().mean(dim=(0, 2, 3))
        pair_score = weight_score[:half] + weight_score[half:] + 0.5 * project_score
        selected = torch.argsort(pair_score)[:pair_count].tolist()
        return sorted(selected + [index + half for index in selected])

    def _prune_hidden_pair(
        self,
        model: MobileNAFNetW16,
        block: MobileNAFBlockDW,
        branch: str,
        pair_count: int,
        example_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        expand = getattr(block, f"{branch}_expand")
        project = getattr(block, f"{branch}_project")
        indices = self._paired_hidden_indices(expand, project, pair_count)
        graph = self.build_dependency_graph(model, example_inputs)
        group = graph.get_pruning_group(expand, tp.prune_conv_out_channels, idxs=indices)
        self.validate_group(graph, group)
        group.prune()
        self.update_static_attributes(model)

    def prune_stage_channels(
        self,
        model: MobileNAFNetW16,
        stage_index: int,
        indices: list[int],
        example_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """裁一个 Feature Stage，并把所有 NAFBlock 的 2C 隐藏宽度同步收紧。"""

        if len(indices) != self.round_to:
            raise ValueError(f"每轮必须恰好裁剪 {self.round_to} 个 Feature 通道")
        root = model.downs[stage_index - 1]
        old_width = root.out_channels
        if old_width - len(indices) < self.minimum_widths[stage_index]:
            raise ValueError("剪枝会突破 Stage 最小宽度")
        graph = self.build_dependency_graph(model, example_inputs)
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=sorted(indices))
        self.validate_group(graph, group)
        group.prune()
        # 根 Stage 已变化，必须先同步 FiLM/Gate 静态属性，后续依赖图才能再次前向。
        self.update_static_attributes(model)
        # Feature 宽度减少 16 后，每个 Gate 的两半各减少 16，恢复固定 2C 结构。
        for block in self._blocks_for_stage(model, stage_index):
            self._prune_hidden_pair(model, block, "spatial", len(indices), example_inputs)
            self._prune_hidden_pair(model, block, "channel", len(indices), example_inputs)
        self.update_static_attributes(model)

    def validate(
        self,
        model: nn.Module,
        example_image: torch.Tensor,
        example_condition: torch.Tensor,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for name, module in model.named_modules():
            if isinstance(module, StaticSimpleGate):
                if module.channels <= 0 or module.channels % self.round_to:
                    issues.append(ValidationIssue(name, f"Gate channels={module.channels} 未按 {self.round_to} 对齐"))
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                if not (module.groups == module.in_channels == module.out_channels):
                    issues.append(ValidationIssue(name, "DWConv 必须满足 groups=in_channels=out_channels"))
            if isinstance(module, MobileNAFBlockDW):
                channels = module.spatial_expand.in_channels
                if module.spatial_expand.out_channels != channels * 2:
                    issues.append(ValidationIssue(name, "Spatial Expand 必须保持 2C"))
                if module.channel_expand.out_channels != channels * 2:
                    issues.append(ValidationIssue(name, "Channel Expand 必须保持 2C"))
                if module.spatial_project.out_channels != channels or module.channel_project.out_channels != channels:
                    issues.append(ValidationIssue(name, "残差投影输出必须与主干同宽"))
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
        except Exception as error:  # noqa: BLE001 - 校验器把异常转为结构化报告
            issues.append(ValidationIssue("<forward>", str(error)))
        return issues

    def assert_valid(self, model: nn.Module, example_image: torch.Tensor, example_condition: torch.Tensor) -> None:
        issues = self.validate(model, example_image, example_condition)
        if issues:
            detail = "; ".join(f"{item.module_name}: {item.message}" for item in issues)
            raise ValueError(f"剪枝拓扑验证失败: {detail}")


class StructuredMobileNAFPruner:
    """按 Stage 重要度渐进生成 V6.1 固定 16 对齐候选。"""

    def __init__(self, validator: NAFNetPruningValidator | None = None) -> None:
        self.validator = validator or NAFNetPruningValidator()

    @staticmethod
    def _stage_scores(
        model: MobileNAFNetW16,
        stage_index: int,
        calibration_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        root = model.downs[stage_index - 1]
        activation_sum = torch.zeros(root.out_channels, device=root.weight.device)
        activation_count = 0

        def hook(module, inputs, output) -> None:
            nonlocal activation_count
            activation_sum.add_(output.detach().abs().mean(dim=(0, 2, 3)))
            activation_count += 1

        handle = root.register_forward_hook(hook)
        model.zero_grad(set_to_none=True)
        for image, condition in calibration_batches:
            output = model(image.to(root.weight.device), condition.to(root.weight.device))
            output.square().mean().backward()
        handle.remove()
        magnitude = root.weight.detach().abs().flatten(1).mean(1)
        taylor = (
            (root.weight.detach() * root.weight.grad.detach()).abs().flatten(1).mean(1)
            if root.weight.grad is not None
            else torch.zeros_like(magnitude)
        )
        activation = activation_sum / max(activation_count, 1)
        blocks = NAFNetPruningValidator._blocks_for_stage(model, stage_index)
        residual = torch.stack([
            (block.beta.detach().abs() + block.gamma.detach().abs()).flatten()
            for block in blocks
        ]).mean(0)

        def normalize(value: torch.Tensor) -> torch.Tensor:
            minimum, maximum = value.min(), value.max()
            return (value - minimum) / (maximum - minimum).clamp_min(1e-12)

        return 0.5 * normalize(taylor) + 0.3 * normalize(magnitude + residual) + 0.2 * normalize(activation)

    def prune_to_ratio(
        self,
        model: MobileNAFNetW16,
        example_inputs: tuple[torch.Tensor, torch.Tensor],
        calibration_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        target_ratio: float,
    ) -> PruningReport:
        """裁到目标下限；若一个16通道组导致略微越界，报告真实比例而不伪造名称。"""

        if not 0.0 < target_ratio < 0.5:
            raise ValueError("目标剪枝比例必须位于 (0,0.5)")
        batches = list(calibration_batches)
        if not batches:
            raise ValueError("重要度计算至少需要一个校准 Batch")
        image, condition = example_inputs
        before_params = _count_parameters(model)
        before_macs = estimate_macs(model, image, condition)
        rounds: list[dict[str, object]] = []
        # P10/P15 允许 ±2 个百分点；达到区间下界即停止，避免名义 P10 被裁成 P14。
        lower_bound = max(target_ratio - 0.02, self.validator.round_to / before_params)
        while 1.0 - _count_parameters(model) / before_params < lower_bound:
            choices: list[tuple[float, float, int, list[int]]] = []
            for stage_index in (1, 2, 3):
                width = model.downs[stage_index - 1].out_channels
                if width - self.validator.round_to < self.validator.minimum_widths[stage_index]:
                    continue
                scores = self._stage_scores(model, stage_index, batches)
                indices = torch.argsort(scores)[: self.validator.round_to].tolist()
                # 在副本上预演，避免一个高影响 Stage 把 P10 直接裁成 P14。
                candidate = copy.deepcopy(model)
                self.validator.prune_stage_channels(candidate, stage_index, indices, example_inputs)
                projected = 1.0 - _count_parameters(candidate) / before_params
                choices.append((float(scores[indices].mean()), projected, stage_index, indices))
            if not choices:
                raise RuntimeError("已达到各 Stage 最小宽度，仍无法满足目标剪枝比例")
            upper_bound = target_ratio + 0.02
            inside = [item for item in choices if lower_bound <= item[1] <= upper_bound]
            below = [item for item in choices if item[1] < lower_bound]
            if inside:
                # 画质优先：在合法区间选重要度最低者，再选更接近名义目标者。
                selected = min(inside, key=lambda item: (item[0], abs(item[1] - target_ratio)))
            elif below:
                selected = min(below, key=lambda item: item[0])
            else:
                selected = min(choices, key=lambda item: item[1])
            _, _, stage_index, indices = selected
            old_width = model.downs[stage_index - 1].out_channels
            self.validator.prune_stage_channels(model, stage_index, indices, example_inputs)
            self.validator.assert_valid(model, image, condition)
            rounds.append({
                "round": len(rounds) + 1,
                "stage": stage_index,
                "removed_indices": sorted(indices),
                "width_before": old_width,
                "width_after": model.downs[stage_index - 1].out_channels,
                "parameter_reduction": 1.0 - _count_parameters(model) / before_params,
            })
        after_params = _count_parameters(model)
        after_macs = estimate_macs(model, image, condition)
        feature_channels = (model.intro.out_channels, *(layer.out_channels for layer in model.downs))
        return PruningReport(
            target_ratio=target_ratio,
            parameter_count_before=before_params,
            parameter_count_after=after_params,
            parameter_reduction=1.0 - after_params / before_params,
            macs_before=before_macs,
            macs_after=after_macs,
            mac_reduction=1.0 - after_macs / before_macs,
            feature_channels=tuple(int(value) for value in feature_channels),
            rounds=tuple(rounds),
        )

    def prune_to_feature_channels(
        self,
        model: MobileNAFNetW16,
        example_inputs: tuple[torch.Tensor, torch.Tensor],
        calibration_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        target_channels: tuple[int, int, int, int],
    ) -> PruningReport:
        """真实裁剪到 P10-16/P18-16/P36-16 的精确发布拓扑。"""

        if len(target_channels) != 4 or any(value <= 0 or value % 16 for value in target_channels):
            raise ValueError("目标 Feature Channels 必须包含四级正 16 对齐宽度")
        current = (model.intro.out_channels, *(layer.out_channels for layer in model.downs))
        if target_channels[0] != current[0]:
            raise ValueError("V6.1 Intro/Stage1 不允许主动剪枝")
        for index, (target, width, minimum) in enumerate(zip(target_channels, current, self.validator.minimum_widths)):
            if target > width or target < minimum:
                raise ValueError(f"Stage{index} 目标宽度 {target} 超出 [{minimum},{width}]")
        batches = list(calibration_batches)
        if not batches:
            raise ValueError("重要度计算至少需要一个校准 Batch")
        image, condition = example_inputs
        before_params = _count_parameters(model)
        before_macs = estimate_macs(model, image, condition)
        rounds: list[dict[str, object]] = []
        for stage_index in (1, 2, 3):
            while model.downs[stage_index - 1].out_channels > target_channels[stage_index]:
                width_before = model.downs[stage_index - 1].out_channels
                scores = self._stage_scores(model, stage_index, batches)
                indices = torch.argsort(scores)[: self.validator.round_to].tolist()
                self.validator.prune_stage_channels(model, stage_index, indices, example_inputs)
                self.validator.assert_valid(model, image, condition)
                rounds.append({
                    "round": len(rounds) + 1,
                    "stage": stage_index,
                    "removed_indices": sorted(indices),
                    "width_before": width_before,
                    "width_after": model.downs[stage_index - 1].out_channels,
                    "parameter_reduction": 1.0 - _count_parameters(model) / before_params,
                })
        feature_channels = (model.intro.out_channels, *(layer.out_channels for layer in model.downs))
        if tuple(feature_channels) != target_channels:
            raise RuntimeError(f"剪枝结果 {feature_channels} 未达到目标 {target_channels}")
        after_params = _count_parameters(model)
        after_macs = estimate_macs(model, image, condition)
        return PruningReport(
            target_ratio=1.0 - after_params / before_params,
            parameter_count_before=before_params,
            parameter_count_after=after_params,
            parameter_reduction=1.0 - after_params / before_params,
            macs_before=before_macs,
            macs_after=after_macs,
            mac_reduction=1.0 - after_macs / before_macs,
            feature_channels=tuple(int(value) for value in feature_channels),
            rounds=tuple(rounds),
        )
