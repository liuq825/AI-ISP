import torch

from ai_isp.models.static_simple_gate import StaticSimpleGate


def test_static_gate_matches_manual_split() -> None:
    value = torch.randn(2, 16, 8, 8)
    output = StaticSimpleGate(8)(value)
    assert torch.equal(output, value[:, :8] * value[:, 8:])

