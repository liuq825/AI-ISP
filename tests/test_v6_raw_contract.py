import numpy as np
import pytest
import torch

from ai_isp.data import (
    AdmissionPolicy,
    RyybFrameDescriptor,
    normalize_post_blc_lsc_packed_raw,
    pack_ryyb,
    reconstruct_and_unpack_ryyb,
    unpack_ryyb,
    validate_ai_admission,
)
from ai_isp.data.pack_raw import normalize_packed_raw


def test_post_blc_lsc_normalization_does_not_subtract_black_twice() -> None:
    packed = np.full((4, 2, 2), 64.0, dtype=np.float32)
    legacy = normalize_packed_raw(packed, 64.0, 1023.0)
    post_blc = normalize_post_blc_lsc_packed_raw(packed, 64.0, 1023.0)
    assert np.count_nonzero(legacy) == 0
    assert np.allclose(post_blc, 64.0 / (1023.0 - 64.0))


@pytest.mark.parametrize("phase", ("ryyb", "byyr", "yryb", "ybyr"))
def test_all_physical_phases_pack_unpack_are_bit_exact(phase: str) -> None:
    raw = np.arange(8 * 10, dtype=np.uint16).reshape(8, 10)
    assert np.array_equal(unpack_ryyb(pack_ryyb(raw, phase), phase), raw)


def test_reconstruction_is_subtract_clamp_then_physical_unpack() -> None:
    packed = torch.full((4, 2, 3), 0.5)
    noise = torch.full_like(packed, 0.1)
    physical = reconstruct_and_unpack_ryyb(packed, noise, "byyr")
    assert physical.shape == (4, 6)
    assert torch.allclose(physical, torch.full_like(physical, 0.4))


def test_admission_checks_raw_lsc_unpack_and_buffer_hashes() -> None:
    policy = AdmissionPolicy(
        sensor_profiles=("main_ryyb_0",),
        sensor_cfa_phases=(("main_ryyb_0", "ryyb"),),
        model_hash="model",
        quant_policy_hash="quant",
        raw_domain_profile_hash="raw-domain",
        lsc_profile_hashes=(("main_ryyb_0", "lsc"),),
        unpack_profile_hashes=(("main_ryyb_0", "unpack"),),
    )
    descriptor = RyybFrameDescriptor(
        "main",
        "main_ryyb_0",
        "ryyb",
        raw_domain_profile_hash="raw-domain",
        lsc_profile_hash="lsc",
        unpack_profile_hash="unpack",
        model_hash="model",
        quant_policy_hash="quant",
    )
    validate_ai_admission(descriptor, policy)
    with pytest.raises(ValueError, match="LSC"):
        validate_ai_admission(RyybFrameDescriptor(**{**descriptor.__dict__, "lsc_profile_hash": "wrong"}), policy)
    with pytest.raises(ValueError, match="memcpy"):
        validate_ai_admission(RyybFrameDescriptor(**{**descriptor.__dict__, "extra_cpu_memcpy_bytes": 2}), policy)
