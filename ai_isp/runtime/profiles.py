"""三种静态 Preview Profile 的唯一来源。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StaticProfile:
    profile_id: str
    purpose: str
    valid_height: int
    valid_width: int
    compile_height: int
    compile_width: int

    @property
    def needs_padding(self) -> bool:
        return self.valid_height != self.compile_height or self.valid_width != self.compile_width


PROFILES = {
    "P0": StaticProfile("P0", "4:3 Photo Preview", 768, 1024, 768, 1024),
    "P1": StaticProfile("P1", "16:9 Full-screen Preview", 540, 960, 544, 960),
    "P2": StaticProfile("P2", "3:2 Preview/兼容流", 640, 960, 640, 960),
}


def select_profile(valid_width: int, valid_height: int) -> StaticProfile:
    """只按协商后的有效宽高选择 Profile，禁止按 Camera 名称猜测。"""

    for profile in PROFILES.values():
        if profile.valid_width == valid_width and profile.valid_height == valid_height:
            return profile
    raise ValueError(f"不支持的有效 Packed RAW 尺寸: width={valid_width}, height={valid_height}")

