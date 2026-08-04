"""V3 暗光触发滞回、淡入淡出和异常立即 Bypass 状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TriggerState(str, Enum):
    BYPASS_BRIGHT = "BYPASS_BRIGHT"
    ARMING = "ARMING"
    ACTIVE_RAMP = "ACTIVE_RAMP"
    ACTIVE = "ACTIVE"
    EXIT_PENDING = "EXIT_PENDING"
    BYPASS_ERROR = "BYPASS_ERROR"


@dataclass(frozen=True)
class TriggerConfig:
    enter_dark_score: float = 0.70
    exit_dark_score: float = 0.45
    enter_iso: float = 1600.0
    exit_iso: float = 1200.0
    enter_ev: float = -1.5
    exit_ev: float = -1.0
    enter_noise: float = 0.08
    exit_noise: float = 0.06
    enter_frames: int = 3
    exit_frames: int = 10
    ramp: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00)


@dataclass(frozen=True)
class TriggerResult:
    state: TriggerState
    enhancement_strength: float
    bypass: bool


class DarkTriggerStateMachine:
    def __init__(self, config: TriggerConfig | None = None) -> None:
        self.config = config or TriggerConfig()
        self.reset()

    def reset(self) -> None:
        self.state = TriggerState.BYPASS_BRIGHT
        self.enter_counter = 0
        self.exit_counter = 0
        self.ramp_index = 0

    def fail_immediately(self) -> TriggerResult:
        """异常、热策略或 Camera Transition 不等待淡出。"""

        self.state = TriggerState.BYPASS_ERROR
        self.enter_counter = self.exit_counter = self.ramp_index = 0
        return TriggerResult(self.state, 0.0, True)

    def recover(self) -> TriggerResult:
        self.reset()
        return TriggerResult(self.state, 0.0, True)

    def update(self, iso: float, scene_ev: float, noise_level: float, dark_score: float | None = None) -> TriggerResult:
        if self.state == TriggerState.BYPASS_ERROR:
            return TriggerResult(self.state, 0.0, True)
        enter = (
            (dark_score is not None and dark_score >= self.config.enter_dark_score)
            or (iso >= self.config.enter_iso and scene_ev <= self.config.enter_ev)
            or noise_level >= self.config.enter_noise
        )
        exit_ready = (
            (dark_score is None or dark_score <= self.config.exit_dark_score)
            and iso <= self.config.exit_iso
            and scene_ev >= self.config.exit_ev
            and noise_level <= self.config.exit_noise
        )
        if self.state in (TriggerState.BYPASS_BRIGHT, TriggerState.ARMING):
            if enter:
                self.enter_counter += 1
                self.state = TriggerState.ARMING
                if self.enter_counter >= self.config.enter_frames:
                    self.state = TriggerState.ACTIVE_RAMP
                    self.ramp_index = 0
            else:
                self.state = TriggerState.BYPASS_BRIGHT
                self.enter_counter = 0
        elif self.state == TriggerState.ACTIVE_RAMP:
            self.ramp_index += 1
            if self.ramp_index >= len(self.config.ramp):
                self.state = TriggerState.ACTIVE
        elif self.state in (TriggerState.ACTIVE, TriggerState.EXIT_PENDING):
            if exit_ready:
                self.exit_counter += 1
                self.state = TriggerState.EXIT_PENDING
                if self.exit_counter >= self.config.exit_frames:
                    self.state = TriggerState.BYPASS_BRIGHT
                    self.enter_counter = self.exit_counter = self.ramp_index = 0
            else:
                self.state = TriggerState.ACTIVE
                self.exit_counter = 0
        return self._result()

    def _result(self) -> TriggerResult:
        if self.state == TriggerState.ACTIVE_RAMP:
            strength = self.config.ramp[min(self.ramp_index, len(self.config.ramp) - 1)]
            return TriggerResult(self.state, strength, False)
        if self.state == TriggerState.ACTIVE:
            return TriggerResult(self.state, 1.0, False)
        if self.state == TriggerState.EXIT_PENDING:
            # 10 帧退出窗口中按 5 档反向淡出；其余帧保持最低强度。
            index = min(self.exit_counter - 1, len(self.config.ramp) - 1)
            return TriggerResult(self.state, tuple(reversed(self.config.ramp))[index], False)
        return TriggerResult(self.state, 0.0, True)

