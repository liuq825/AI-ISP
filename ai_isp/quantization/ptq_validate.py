"""PTQ 数值链路的 CPU 快速校验。"""

from __future__ import annotations

import torch

from .lsqplus_qat import LearnableFakeQuant


def validate_ptq_tensor(value: torch.Tensor) -> dict[str, float]:
    """初始化 W8A8 激活量化并报告量化误差和饱和率。"""

    quantizer = LearnableFakeQuant(bits=8, symmetric=False, learnable_offset=False)
    quantizer.initialize_from(value)
    quantized = quantizer(value).detach()
    scale = quantizer.scale.detach()
    q = (value - quantizer.offset.detach()) / scale
    saturation = ((q <= quantizer.qmin) | (q >= quantizer.qmax)).float().mean()
    return {
        "max_abs_error": float((quantized - value).abs().max()),
        "mean_abs_error": float((quantized - value).abs().mean()),
        "saturation_rate": float(saturation),
        "scale": float(scale),
        "offset": float(quantizer.offset.detach()),
    }
