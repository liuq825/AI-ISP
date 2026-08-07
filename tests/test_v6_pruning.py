import torch

from ai_isp.models import build_mobile_nafnet_w16
from ai_isp.pruning import NAFNetPruningValidator, StructuredMobileNAFPruner


def test_p36_16_exact_topology_is_real_and_valid() -> None:
    torch.manual_seed(9)
    model = build_mobile_nafnet_w16()
    image = torch.rand(1, 4, 32, 32)
    condition = torch.rand(1, 24)
    report = StructuredMobileNAFPruner().prune_to_feature_channels(
        model, (image, condition), [(image, condition)], (16, 32, 48, 96)
    )
    assert report.feature_channels == (16, 32, 48, 96)
    assert report.parameter_reduction > 0.30
    assert all(width % 16 == 0 for width in report.feature_channels)
    NAFNetPruningValidator().assert_valid(model, image, condition)
