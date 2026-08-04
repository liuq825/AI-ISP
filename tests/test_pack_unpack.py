import numpy as np
import pytest
import torch

from ai_isp.data.pack_raw import CFA_OFFSETS, pack_bayer, unpack_bayer


@pytest.mark.parametrize("cfa", sorted(CFA_OFFSETS))
def test_numpy_pack_unpack_bit_exact(cfa: str) -> None:
    raw = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
    packed = pack_bayer(raw, cfa)
    restored = unpack_bayer(packed, cfa)
    assert packed.shape == (4, 6, 8)
    np.testing.assert_array_equal(restored, raw)


@pytest.mark.parametrize("cfa", sorted(CFA_OFFSETS))
def test_torch_pack_unpack_bit_exact(cfa: str) -> None:
    raw = torch.arange(2 * 12 * 16, dtype=torch.int32).reshape(2, 12, 16)
    restored = unpack_bayer(pack_bayer(raw, cfa), cfa)
    assert torch.equal(restored, raw)


def test_odd_bayer_is_rejected() -> None:
    with pytest.raises(ValueError, match="偶数"):
        pack_bayer(np.zeros((9, 8), dtype=np.uint16), "rggb")

