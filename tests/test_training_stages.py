from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ai_isp.models import ConditionalNAFNetW32Teacher, build_mobile_nafnet_w16
from ai_isp.training_stages import TrainingStageConfig, train_distillation_stage, train_supervised_stage


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index)
        clean = torch.rand(4, 32, 32, generator=generator)
        noisy = (clean + 0.01 * torch.randn(4, 32, 32, generator=generator)).clamp(0.0, 1.0)
        condition = torch.zeros(24)
        condition[[10, 14, 18, 22, 23]] = 1.0
        return {"noisy": noisy, "clean": clean, "condition": condition}


def _config(tmp_path: Path, stage: str) -> TrainingStageConfig:
    return TrainingStageConfig(
        stage_name=stage,
        output_dir=str(tmp_path / stage),
        steps=1,
        learning_rate=1e-4,
        device="cpu",
        checkpoint_interval=0,
    )


def test_supervised_and_kd_stages_run_real_backward(tmp_path: Path) -> None:
    loader = DataLoader(_TinyDataset(), batch_size=1)
    teacher = ConditionalNAFNetW32Teacher()
    student = build_mobile_nafnet_w16()
    teacher_report = train_supervised_stage(teacher, loader, _config(tmp_path, "teacher"))
    student_report = train_supervised_stage(student, loader, _config(tmp_path, "student"))
    kd_report = train_distillation_stage(teacher, student, loader, _config(tmp_path, "kd"))
    assert teacher_report["history"][0]["gradient_norm"] > 0
    assert student_report["history"][0]["gradient_norm"] > 0
    assert kd_report["teacher_mode"] == "eval_fp32_no_grad"
    assert kd_report["history"][0]["loss_feature_kd"] > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert Path(teacher_report["best_model"]).is_file()
    assert Path(student_report["final_model"]).is_file()
    assert Path(kd_report["best_model"]).is_file()
