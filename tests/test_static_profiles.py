import pytest
from torch import nn

from ai_isp.export.static_profiles import prepare_export_model
from ai_isp.runtime.profiles import FIXED_RYYB_PROFILE, select_profile


def test_only_fixed_ryyb_profile_is_accepted() -> None:
    assert select_profile(valid_width=1024, valid_height=768) == FIXED_RYYB_PROFILE
    assert FIXED_RYYB_PROFILE.raw_width == 2048
    assert FIXED_RYYB_PROFILE.raw_height == 1536
    with pytest.raises(ValueError):
        select_profile(valid_width=960, valid_height=540)
    with pytest.raises(ValueError):
        select_profile(valid_width=960, valid_height=640)


def test_legacy_simple_gate_is_replaced_only_on_copy() -> None:
    class SimpleGate(nn.Module):
        channels = 8

        def forward(self, value):
            first, second = value.chunk(2, dim=1)
            return first * second

    source = nn.Sequential(SimpleGate())
    deploy, report = prepare_export_model(source, allow_legacy_replacement=True, legacy_gate_channels={"0": 8})
    assert source[0].__class__.__name__ == "SimpleGate"
    assert deploy[0].__class__.__name__ == "StaticSimpleGate"
    assert report["global_monkey_patch"] is False
    assert report["legacy_replacements"] == [{"name": "0", "channels": 8}]
