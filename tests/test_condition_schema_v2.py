import pytest
import torch

from ai_isp.data.condition_v2 import ConditionMetadata, encode_condition_v2, validate_condition_v2


def test_condition_has_exact_shape_and_one_hot() -> None:
    condition = encode_condition_v2(ConditionMetadata(
        exposure_time_s=1 / 100,
        iso=1600,
        camera_type="main",
        sensor_profile="1",
        lens_profile="wide",
    ))
    assert condition.shape == (24,)
    assert condition.dtype == torch.float32
    validate_condition_v2(condition)
    assert condition[10:14].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert condition[14:18].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_condition_clamps_out_of_range_metadata() -> None:
    condition = encode_condition_v2(ConditionMetadata(exposure_time_s=100, iso=1e9, scene_ev=-99))
    assert bool(((condition >= 0) & (condition <= 1)).all())


def test_invalid_one_hot_is_rejected() -> None:
    condition = torch.zeros(24)
    with pytest.raises(ValueError, match="one-hot"):
        validate_condition_v2(condition)

