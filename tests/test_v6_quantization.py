import torch

from ai_isp.models import build_mobile_nafnet_w16
from ai_isp.quantization import QatPolicy, audit_dynamic_affine_equivalence, prepare_qat_model


def test_fixed_mixed_precision_preserves_fp16_islands_and_quantizes_dynamic_affine() -> None:
    model = prepare_qat_model(build_mobile_nafnet_w16(), QatPolicy())
    assert model.intro.__class__.__name__ == "Conv2d"
    assert model.ending.__class__.__name__ == "Conv2d"
    assert model.condition_encoder.trunk[0].__class__.__name__ == "Linear"
    assert model.encoders[0][0].spatial_expand.__class__.__name__ == "QatConv2d"
    assert model.film_stage2.feature_quant.__class__.__name__ == "Identity"
    assert model.film_stage3.feature_quant.__class__.__name__ == "LearnableFakeQuant"
    assert model.film_middle.beta_quant.__class__.__name__ == "LearnableFakeQuant"


def test_dynamic_affine_integer_reference_matches_fakequant_requant() -> None:
    torch.manual_seed(8)
    feature = torch.randn(1, 16, 4, 4) * 0.2
    gamma = 1.0 + torch.randn(1, 16, 1, 1) * 0.02
    beta = torch.randn(1, 16, 1, 1) * 0.01
    report = audit_dynamic_affine_equivalence(feature, gamma, beta)
    assert report["integer"]["int32_overflow"] is False
    assert report["max_abs_error"] < 0.01
    assert report["dynamic_affine_target_pending"] is True
