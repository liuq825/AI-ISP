"""V4 固定 RYYB 4:3 输入的 Conditional MobileNAFNet。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn

from .static_simple_gate import StaticSimpleGate


class MobileNAFBlockDW(nn.Module):
    """仅由移动 NPU 白名单算子组成的双残差 NAF Block。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.spatial_expand = nn.Conv2d(channels, expanded, 1)
        self.spatial_dw = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.spatial_gate = StaticSimpleGate(channels)
        self.spatial_project = nn.Conv2d(channels, channels, 1)
        self.channel_expand = nn.Conv2d(channels, expanded, 1)
        self.channel_gate = StaticSimpleGate(channels)
        self.channel_project = nn.Conv2d(channels, channels, 1)
        # 零初始化保证网络初始行为接近恒等去噪。
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.spatial_residual_quant: nn.Module = nn.Identity()
        self.channel_residual_quant: nn.Module = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_project(self.spatial_gate(self.spatial_dw(self.spatial_expand(x))))
        x = self.spatial_residual_quant(x + spatial * self.beta)
        channel = self.channel_project(self.channel_gate(self.channel_expand(x)))
        return self.channel_residual_quant(x + channel * self.gamma)


class ConditionEncoder(nn.Module):
    """把 24 维 Condition 编码为三个 FiLM 的 gamma/beta。"""

    def __init__(self, target_channels: Sequence[int]) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.heads = nn.ModuleList(nn.Linear(128, channels * 2) for channels in target_channels)
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded = self.trunk(condition)
        return tuple(head(encoded) for head in self.heads)


def apply_film(feature: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """应用 ``F*(1+0.1*tanh(gamma))+0.1*tanh(beta)``。"""

    channels = feature.shape[1]
    gamma = parameters[:, :channels, None, None]
    beta = parameters[:, channels:, None, None]
    return feature * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)


class UpsampleConv(nn.Module):
    """固定 2 倍 Nearest Resize 加 3×3 Conv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(functional.interpolate(x, scale_factor=2.0, mode="nearest"))


@dataclass(frozen=True)
class MobileNAFNetConfig:
    image_channels: int = 4
    base_width: int = 16
    encoder_blocks: tuple[int, int, int] = (2, 2, 4)
    middle_blocks: int = 2
    decoder_blocks: tuple[int, int, int] = (2, 2, 2)
    condition_dim: int = 24
    feature_channels: tuple[int, int, int, int] | None = None


class MobileNAFNetW16(nn.Module):
    """三层 U-Net Student，输出与输入同格点的 ``noise_pred``。"""

    def __init__(self, config: MobileNAFNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or MobileNAFNetConfig()
        cfg = self.config
        if cfg.condition_dim != 24:
            raise ValueError("V4 Condition 维度必须为 24")
        widths = cfg.feature_channels or (
            cfg.base_width,
            cfg.base_width * 2,
            cfg.base_width * 4,
            cfg.base_width * 8,
        )
        if len(widths) != 4 or any(channel <= 0 or channel % 8 for channel in widths):
            raise ValueError("四级 Feature Channels 必须为正数且按 8 通道对齐")
        self.intro = nn.Conv2d(cfg.image_channels, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*(MobileNAFBlockDW(widths[index]) for _ in range(blocks)))
            for index, blocks in enumerate(cfg.encoder_blocks)
        )
        self.downs = nn.ModuleList(
            nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1)
            for index in range(3)
        )
        self.middle = nn.Sequential(*(MobileNAFBlockDW(widths[3]) for _ in range(cfg.middle_blocks)))
        self.ups = nn.ModuleList(
            (
                UpsampleConv(widths[3], widths[2]),
                UpsampleConv(widths[2], widths[1]),
                UpsampleConv(widths[1], widths[0]),
            )
        )
        decoder_widths = (widths[2], widths[1], widths[0])
        self.decoders = nn.ModuleList(
            nn.Sequential(*(MobileNAFBlockDW(decoder_widths[index]) for _ in range(blocks)))
            for index, blocks in enumerate(reversed(cfg.decoder_blocks))
        )
        self.condition_encoder = ConditionEncoder((widths[1], widths[2], widths[3]))
        self.ending = nn.Conv2d(widths[0], cfg.image_channels, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)
        self.activation_checkpointing = False

    def enable_activation_checkpointing(self, enabled: bool = True) -> None:
        """显存超软门槛时按 Block 重计算 Student 激活；不改变发布拓扑。"""

        self.activation_checkpointing = enabled

    def _run_blocks(self, blocks: nn.Sequential, feature: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled() and len(blocks) > 0:
            return checkpoint_sequential(blocks, len(blocks), feature, use_reentrant=False)
        return blocks(feature)

    def _forward_impl(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if not torch.jit.is_tracing():
            if image.ndim != 4 or image.shape[1] != 4:
                raise ValueError(f"image 必须为 N×4×H×W，收到 {image.shape}")
            if image.shape[-2] % 8 or image.shape[-1] % 8:
                raise ValueError("Packed RAW 的 H/W 必须能被 8 整除")
            if condition.ndim != 2 or condition.shape[0] != image.shape[0] or condition.shape[1] != 24:
                raise ValueError("condition 必须为与 image 同 Batch 的 N×24")
        film_e1, film_e2, film_middle = self.condition_encoder(condition)
        feature = self.intro(image)
        encoder_outputs: list[torch.Tensor] = []
        feature = self._run_blocks(self.encoders[0], feature)
        encoder_outputs.append(feature)
        feature = self.downs[0](feature)
        feature = self._run_blocks(self.encoders[1], apply_film(feature, film_e1))
        encoder_outputs.append(feature)
        feature = self.downs[1](feature)
        feature = self._run_blocks(self.encoders[2], apply_film(feature, film_e2))
        encoder_outputs.append(feature)
        feature = self.downs[2](feature)
        feature = self._run_blocks(self.middle, apply_film(feature, film_middle))
        middle_feature = feature
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(encoder_outputs)):
            feature = self._run_blocks(decoder, up(feature) + skip)
        noise_pred = self.ending(feature)
        # 强度在图内形成严格的 s=0 恒等路径，Runtime 不再二次缩放。
        strength = condition[:, 23:24, None, None]
        return noise_pred * strength, (encoder_outputs[2], middle_feature)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        noise_pred, _ = self._forward_impl(image, condition)
        return noise_pred

    def forward_with_features(
        self, image: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """训练期返回 Encoder Stage 3 与 Middle；部署导出仍只走 ``forward``。"""

        return self._forward_impl(image, condition)

    @staticmethod
    def denoise(image: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """按冻结符号约定执行 ``raw_out=clamp(raw_in-noise_pred)``。"""

        return torch.clamp(image - noise_pred, 0.0, 1.0)

    def topology_manifest(self) -> dict[str, object]:
        """返回不依赖 Python Pickle 的冻结拓扑描述。"""

        return {
            "model": "Conditional MobileNAFNet Dark Preview V4",
            "image_channels": self.config.image_channels,
            "base_width": self.config.base_width,
            "encoder_blocks": list(self.config.encoder_blocks),
            "middle_blocks": self.config.middle_blocks,
            "decoder_blocks": list(self.config.decoder_blocks),
            "feature_channels": [self.downs[0].in_channels, *(layer.out_channels for layer in self.downs)],
            "condition_dim": 24,
            "film_injection": ["encoder_stage_2", "encoder_stage_3", "middle"],
            "output_semantics": "noise_pred",
            "strength_policy": "inside_graph_output_scale",
        }


def build_mobile_nafnet_w16() -> MobileNAFNetW16:
    """创建 V4 Dense W16 基线模型。"""

    return MobileNAFNetW16(MobileNAFNetConfig())


def build_mobile_nafnet_from_topology(feature_channels: Sequence[int]) -> MobileNAFNetW16:
    """按剪枝后四级宽度重建可加载 safetensors 的模型。"""

    channels = tuple(int(value) for value in feature_channels)
    if len(channels) != 4:
        raise ValueError("feature_channels 必须恰好包含四级宽度")
    return MobileNAFNetW16(MobileNAFNetConfig(base_width=channels[0], feature_channels=channels))
