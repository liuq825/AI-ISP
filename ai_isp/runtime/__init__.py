"""可在 CPU 上验证的 Profile、Trigger 与失败安全逻辑。"""

from .dark_trigger import DarkTriggerStateMachine
from .profiles import PROFILES, select_profile

__all__ = ["DarkTriggerStateMachine", "PROFILES", "select_profile"]

