from pathlib import Path

import torch

from ai_isp.export.freeze_topology import freeze_topology, load_frozen_topology
from ai_isp.models.mobile_nafnet import build_mobile_nafnet_from_topology, build_mobile_nafnet_w16
from ai_isp.pruning import NAFNetPruningValidator, StructuredMobileNAFPruner


def test_structured_pruner_changes_real_topology_and_can_rebuild(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = build_mobile_nafnet_w16()
    with torch.no_grad():
        model.ending.weight.normal_(0.0, 1e-3)
    image = torch.rand(1, 4, 32, 32)
    condition = torch.rand(1, 24)
    report = StructuredMobileNAFPruner().prune_to_ratio(
        model, (image, condition), [(image, condition)], target_ratio=0.10
    )
    assert 0.08 <= report.parameter_reduction <= 0.12
    assert report.mac_reduction > 0.0
    NAFNetPruningValidator().assert_valid(model, image, condition)
    rebuilt = build_mobile_nafnet_from_topology(report.feature_channels)
    rebuilt.load_state_dict(model.state_dict())
    assert rebuilt(image, condition).shape == image.shape
    frozen = freeze_topology(model, tmp_path)
    reloaded = load_frozen_topology(frozen["topology"])
    with torch.no_grad():
        assert torch.equal(reloaded(image, condition), model.eval()(image, condition))
