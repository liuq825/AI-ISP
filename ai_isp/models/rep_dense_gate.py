"""RepDenseGateBlock 训练图和折叠后的部署图。"""

from __future__ import annotations

import torch
from torch import nn

from .static_simple_gate import StaticSimpleGate


class RepDenseGateDeploy(nn.Module):
    """折叠后仅保留单个 Dense 3×3 分支。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fused = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.gate = StaticSimpleGate(channels)
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(self.gate(self.fused(x))) * self.scale


class RepDenseGateBlock(nn.Module):
    """并行 3×3/1×1 线性分支的训练态候选。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.branch_3x3 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.branch_1x1 = nn.Conv2d(channels, channels * 2, 1)
        self.gate = StaticSimpleGate(channels)
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self.branch_3x3(x) + self.branch_1x1(x)
        return x + self.project(self.gate(merged)) * self.scale

    def to_deploy(self) -> RepDenseGateDeploy:
        """把 1×1 Kernel 填入 3×3 中心后与 Dense 3×3 求和。"""

        deploy = RepDenseGateDeploy(self.channels).to(self.branch_3x3.weight.device)
        with torch.no_grad():
            kernel = self.branch_3x3.weight.detach().clone()
            kernel[:, :, 1:2, 1:2] += self.branch_1x1.weight.detach()
            bias = self.branch_3x3.bias.detach().clone() + self.branch_1x1.bias.detach()
            deploy.fused.weight.copy_(kernel)
            deploy.fused.bias.copy_(bias)
            deploy.project.load_state_dict(self.project.state_dict())
            deploy.scale.copy_(self.scale)
        return deploy

