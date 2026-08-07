"""V6.1 唯一 RYYB 4:3 静态输入的唯一来源。"""

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

    raw_height: int
    raw_width: int


FIXED_RYYB_PROFILE = StaticProfile(
    "RYYB_4X3",
    "RYYB Main/Tele 4:3 Photo Preview",
    768,
    1024,
    768,
    1024,
    1536,
    2048,
)


def select_profile(valid_width: int, valid_height: int) -> StaticProfile:
    """兼容旧调用点；V4 只接受一个固定 Packed RAW 尺寸。"""

    profile = FIXED_RYYB_PROFILE
    if profile.valid_width == valid_width and profile.valid_height == valid_height:
        return profile
    raise ValueError(f"不支持的有效 Packed RAW 尺寸: width={valid_width}, height={valid_height}")
