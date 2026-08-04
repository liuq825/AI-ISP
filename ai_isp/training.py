"""CPU 可运行的 Student FP32 小样本训练与验证。"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai_isp.data.sidd_dataset import SiddRawPatchDataset
from ai_isp.losses.dark_preview_losses import DarkPreviewLoss, global_ssim, raw_psnr
from ai_isp.models.mobile_nafnet import MobileNAFNetW16, build_mobile_nafnet_w16


@dataclass(frozen=True)
class CpuTrainingConfig:
    dataset_root: str
    output_dir: str
    patch_size: int = 32
    batch_size: int = 1
    steps: int = 4
    samples_per_epoch: int = 8
    max_pairs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 20260804
    torch_threads: int = 4


def set_reproducible(seed: int, torch_threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch_threads))
    torch.use_deterministic_algorithms(True)


def train_student_cpu(config: CpuTrainingConfig) -> tuple[MobileNAFNetW16, dict[str, object]]:
    """以少量真实 SIDD Patch 跑通反向传播、优化、评估和权重保存前链路。"""

    set_reproducible(config.seed, config.torch_threads)
    dataset = SiddRawPatchDataset(
        config.dataset_root,
        patch_size=config.patch_size,
        samples_per_epoch=max(config.samples_per_epoch, config.steps),
        seed=config.seed,
        max_pairs=config.max_pairs,
        deterministic=True,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_mobile_nafnet_w16().cpu().train()
    criterion = DarkPreviewLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        if step > config.steps:
            break
        noisy = batch["noisy"].float()
        clean = batch["clean"].float()
        condition = batch["condition"].float()
        optimizer.zero_grad(set_to_none=True)
        noise_pred = model(noisy, condition)
        output = model.denoise(noisy, noise_pred)
        losses = criterion(output, clean)
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        history.append({
            "step": step,
            "loss_total": float(losses["total"].detach()),
            "loss_raw": float(losses["raw"].detach()),
            "loss_tone": float(losses["tone"].detach()),
            "loss_gradient": float(losses["gradient"].detach()),
            "gradient_norm": float(gradient_norm),
            "raw_psnr": raw_psnr(output, clean),
        })
    elapsed = time.perf_counter() - started
    model.eval()
    validation_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    with torch.no_grad():
        noisy = validation_batch["noisy"].float()
        clean = validation_batch["clean"].float()
        condition = validation_batch["condition"].float()
        output = model.denoise(noisy, model(noisy, condition))
    report: dict[str, object] = {
        "scope": "CPU 小样本工程闭环，不代表量产画质",
        "config": asdict(config),
        "device": "cpu",
        "torch_version": torch.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(len(history), 1),
        "history": history,
        "validation": {
            "raw_psnr": raw_psnr(output, clean),
            "global_ssim_smoke": global_ssim(output, clean),
            "finite": bool(torch.isfinite(output).all()),
            "min": float(output.min()),
            "max": float(output.max()),
        },
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cpu_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return model, report

