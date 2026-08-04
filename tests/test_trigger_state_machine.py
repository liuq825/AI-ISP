from ai_isp.runtime.dark_trigger import DarkTriggerStateMachine, TriggerState


def test_enter_ramp_exit_and_error_bypass() -> None:
    trigger = DarkTriggerStateMachine()
    results = [trigger.update(1600, -2.0, 0.09) for _ in range(3)]
    assert results[-1].state == TriggerState.ACTIVE_RAMP
    assert results[-1].enhancement_strength == 0.2
    ramp = [trigger.update(1600, -2.0, 0.09).enhancement_strength for _ in range(4)]
    assert ramp == [0.4, 0.6, 0.8, 1.0]
    assert trigger.update(1600, -2.0, 0.09).state == TriggerState.ACTIVE
    exit_results = [trigger.update(100, 0.0, 0.01, 0.1) for _ in range(10)]
    assert exit_results[-1].state == TriggerState.BYPASS_BRIGHT
    assert exit_results[-1].bypass
    trigger.update(1600, -2.0, 0.09)
    assert trigger.fail_immediately().state == TriggerState.BYPASS_ERROR
    assert trigger.recover().state == TriggerState.BYPASS_BRIGHT

