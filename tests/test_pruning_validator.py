import torch

from ai_isp.models.mobile_nafnet import build_mobile_nafnet_w16
from ai_isp.pruning.nafnet_pruning_validator import NAFNetPruningValidator


def test_unpruned_baseline_satisfies_pruning_invariants() -> None:
    model = build_mobile_nafnet_w16().eval()
    image = torch.rand(1, 4, 32, 32)
    condition = torch.rand(1, 24)
    NAFNetPruningValidator().assert_valid(model, image, condition)

