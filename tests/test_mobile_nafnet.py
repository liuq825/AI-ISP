import torch

from ai_isp.models.mobile_nafnet import build_mobile_nafnet_w16


def test_model_shape_parameter_budget_and_strength_zero() -> None:
    model = build_mobile_nafnet_w16().eval()
    image = torch.rand(1, 4, 32, 48)
    condition = torch.rand(1, 24)
    condition[:, 23] = 1.0
    with torch.no_grad():
        output = model(image, condition)
    assert output.shape == image.shape
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 550_000 <= parameter_count <= 800_000
    condition[:, 23] = 0.0
    with torch.no_grad():
        zero_output = model(image, condition)
    assert torch.count_nonzero(zero_output) == 0
    assert torch.equal(model.denoise(image, zero_output), image)


def test_invalid_spatial_multiple_is_rejected() -> None:
    model = build_mobile_nafnet_w16()
    try:
        model(torch.rand(1, 4, 31, 32), torch.rand(1, 24))
    except ValueError as error:
        assert "8" in str(error)
    else:
        raise AssertionError("非法 Shape 未被拒绝")

