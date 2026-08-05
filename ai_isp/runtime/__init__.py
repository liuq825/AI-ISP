"""可在 CPU 上验证的固定 RYYB Profile、Trigger 与失败安全逻辑。"""

from .dark_trigger import DarkTriggerStateMachine
from .profiles import FIXED_RYYB_PROFILE, select_profile

__all__ = ["DarkTriggerStateMachine", "FIXED_RYYB_PROFILE", "select_profile"]
