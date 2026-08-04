"""切点为构造期常量的 SimpleGate。"""

from __future__ import annotations

import torch
from torch import nn


class StaticSimpleGate(nn.Module):
    """把固定 ``2C`` 通道切为 ``C/C`` 后逐元素相乘。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须为正整数")
        self.channels = int(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and x.shape[1] != self.channels * 2:
            raise ValueError(f"StaticSimpleGate 期望 {self.channels * 2} 通道，收到 {x.shape[1]}")
        left = torch.narrow(x, 1, 0, self.channels)
        right = torch.narrow(x, 1, self.channels, self.channels)
        return left * right

