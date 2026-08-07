import torch

from ai_isp.losses import attention_distillation_loss, temporal_consistency_loss


def test_attention_kd_uses_normalized_stage3_and_middle_maps() -> None:
    student = (torch.rand(1, 8, 4, 4), torch.rand(1, 16, 2, 2))
    teacher = (student[0].clone(), student[1].clone())
    assert attention_distillation_loss(student, teacher) == 0
    changed = (student[0] * torch.linspace(0.5, 1.5, 4)[None, None, None, :], student[1])
    assert attention_distillation_loss(changed, teacher) > 0


def test_temporal_loss_uses_noisy_input_reconstruction_and_masks_highlights() -> None:
    clean = torch.full((1, 4, 2, 2), 0.5)
    noisy_1 = torch.full_like(clean, 0.6)
    noisy_2 = torch.full_like(clean, 0.4)
    pred_1 = torch.full_like(clean, 0.1)
    pred_2 = torch.full_like(clean, -0.1)
    assert temporal_consistency_loss(clean, noisy_1, pred_1, noisy_2, pred_2) == 0
    assert temporal_consistency_loss(clean, noisy_1, torch.zeros_like(clean), noisy_2, pred_2) > 0

    highlight = torch.ones_like(clean)
    high_pred_1 = torch.zeros_like(clean, requires_grad=True)
    high_pred_2 = torch.zeros_like(clean, requires_grad=True)
    loss = temporal_consistency_loss(highlight, highlight, high_pred_1, highlight, high_pred_2)
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(high_pred_1.grad) == 0
    assert torch.count_nonzero(high_pred_2.grad) == 0
