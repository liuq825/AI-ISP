"""简洁、可审计的 LSQ+ Fake Quant 基元。"""

from __future__ import annotations

import torch
from torch import nn


def round_ste(value: torch.Tensor) -> torch.Tensor:
    """前向取整、反向恒等的 Straight-Through Estimator。"""

    return value + (torch.round(value) - value).detach()


class LearnableFakeQuant(nn.Module):
    """激活支持可学习 Offset，权重使用 Offset=0 对称策略。"""

    def __init__(self, bits: int = 8, symmetric: bool = False, learnable_offset: bool = True) -> None:
        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("量化位宽必须位于 [2,16]")
        self.bits = bits
        self.symmetric = symmetric
        self.qmin = -(1 << (bits - 1)) if symmetric else 0
        self.qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
        self.log_scale = nn.Parameter(torch.tensor(-4.0))
        self.offset = nn.Parameter(torch.tensor(0.0), requires_grad=(learnable_offset and not symmetric))

    @property
    def scale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_scale).clamp_min(1e-8)

    def initialize_from(self, value: torch.Tensor) -> None:
        """以张量范围初始化 Scale/Offset，供 PTQ→QAT 接续。"""

        with torch.no_grad():
            minimum, maximum = value.detach().amin(), value.detach().amax()
            if self.symmetric:
                target_scale = torch.maximum(minimum.abs(), maximum.abs()) / max(self.qmax, 1)
                self.offset.zero_()
            else:
                target_scale = (maximum - minimum).clamp_min(1e-8) / max(self.qmax - self.qmin, 1)
                self.offset.copy_(minimum)
            self.log_scale.copy_(torch.log(torch.expm1(target_scale.clamp_min(1e-8))))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        offset = torch.zeros_like(self.offset) if self.symmetric else self.offset
        quantized = round_ste((value - offset) / scale).clamp(self.qmin, self.qmax)
        return quantized * scale + offset

    def audit(self) -> dict[str, float | int | bool]:
        return {
            "bits": self.bits,
            "symmetric": self.symmetric,
            "scale": float(self.scale.detach()),
            "offset": float((torch.zeros_like(self.offset) if self.symmetric else self.offset).detach()),
            "qmin": self.qmin,
            "qmax": self.qmax,
        }

