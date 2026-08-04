import pytest
import torch

from ai_isp.data.pack_raw import crop_p1, pad_p1
from ai_isp.runtime.profiles import select_profile


def test_profiles_select_by_explicit_width_height() -> None:
    assert select_profile(valid_width=1024, valid_height=768).profile_id == "P0"
    assert select_profile(valid_width=960, valid_height=540).profile_id == "P1"
    assert select_profile(valid_width=960, valid_height=640).profile_id == "P2"
    with pytest.raises(ValueError):
        select_profile(valid_width=540, valid_height=960)


def test_p1_fixed_padding_and_crop() -> None:
    value = torch.arange(4 * 540 * 16).reshape(1, 4, 540, 16)
    padded = pad_p1(value)
    assert padded.shape == (1, 4, 544, 16)
    assert torch.equal(padded[..., 540:, :], value[..., -1:, :].expand(1, 4, 4, 16))
    assert torch.equal(crop_p1(padded), value)

