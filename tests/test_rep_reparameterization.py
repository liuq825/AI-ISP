import torch

from ai_isp.models.rep_dense_gate import RepDenseGateBlock


def test_rep_training_graph_equals_deploy_graph() -> None:
    torch.manual_seed(7)
    block = RepDenseGateBlock(8).eval()
    with torch.no_grad():
        block.scale.fill_(0.7)
    deploy = block.to_deploy().eval()
    value = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        expected = block(value)
        actual = deploy(value)
    assert torch.max(torch.abs(expected - actual)).item() <= 1e-5
    cosine = torch.nn.functional.cosine_similarity(expected.flatten(), actual.flatten(), dim=0)
    assert float(cosine) >= 0.999999

