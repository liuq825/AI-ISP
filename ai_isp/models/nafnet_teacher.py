"""仅用于训练与蒸馏的 Conditional NAFNet-W32 Teacher。"""

from __future__ import annotations

import torch
from torch import nn

from .mobile_nafnet import ConditionEncoder, MobileNAFBlockDW, UpsampleConv, apply_film


class ConditionalNAFNetW32Teacher(nn.Module):
    """四层 W32 Teacher；不进入移动端发布制品。"""

    def __init__(self) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 512)
        encoder_blocks = (2, 2, 4, 8)
        decoder_blocks = (2, 2, 2, 2)
        self.intro = nn.Conv2d(4, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            nn.Sequential(*(MobileNAFBlockDW(widths[index]) for _ in range(blocks)))
            for index, blocks in enumerate(encoder_blocks)
        )
        self.downs = nn.ModuleList(nn.Conv2d(widths[i], widths[i + 1], 3, 2, 1) for i in range(4))
        self.middle = nn.Sequential(*(MobileNAFBlockDW(widths[4]) for _ in range(4)))
        self.ups = nn.ModuleList(UpsampleConv(widths[i + 1], widths[i]) for i in reversed(range(4)))
        self.decoders = nn.ModuleList(
            nn.Sequential(*(MobileNAFBlockDW(widths[index]) for _ in range(decoder_blocks[index])))
            for index in reversed(range(4))
        )
        self.condition_encoder = ConditionEncoder((widths[1], widths[2], widths[3], widths[4]))
        self.ending = nn.Conv2d(widths[0], 4, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        films = self.condition_encoder(condition)
        feature = self.intro(image)
        skips: list[torch.Tensor] = []
        for index, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            if index > 0:
                feature = apply_film(feature, films[index - 1])
            feature = encoder(feature)
            skips.append(feature)
            feature = down(feature)
        feature = self.middle(apply_film(feature, films[-1]))
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            feature = decoder(up(feature) + skip)
        return self.ending(feature) * condition[:, 23:24, None, None]

