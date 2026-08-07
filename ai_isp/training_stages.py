"""Teacher、Student、KD 与剪枝恢复的可恢复训练阶段。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path
import time
from typing import Iterable, Mapping

import torch
import numpy as np
import torch.nn.functional as functional
from torch import nn
from safetensors.torch import save_file

from ai_isp.losses.dark_preview_losses import (
    DarkPreviewLoss,
    StudentCompositeLoss,
    attention_distillation_loss,
    raw_psnr,
    temporal_consistency_loss,
)


@dataclass(frozen=True)
class TrainingStageConfig:
    """所有训练阶段共享的确定性配置。"""

    stage_name: str
    output_dir: str
    steps: int
    learning_rate: float
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    warmup_steps: int = 0
    seed: int = 20260804
    device: str = "auto"
    student_amp: bool = True
    checkpoint_interval: int = 1000
    resume_from: str | None = None
    memory_soft_limit_gb: float = 22.0
    student_activation_checkpointing: bool = False


def _config_hash(config: TrainingStageConfig) -> str:
    # Checkpoint 的存放位置和恢复入口不影响训练数学配置，迁移目录后仍应允许续训。
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("resume_from", None)
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但当前环境没有可用 GPU")
    return device


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _build_grad_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=(enabled and device.type == "cuda"))


def _learning_rate_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(step + 1, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    config: TrainingStageConfig,
    extra_modules: Mapping[str, nn.Module] | None = None,
) -> None:
    """保存仅供可信研发环境恢复的训练态；发布权重仍使用 safetensors。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ai_isp_training_checkpoint_v4",
        "stage": config.stage_name,
        "config_hash": _config_hash(config),
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "extra_modules": {name: module.state_dict() for name, module in (extra_modules or {}).items()},
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    config: TrainingStageConfig,
    extra_modules: Mapping[str, nn.Module] | None = None,
) -> int:
    """恢复模型、优化器和AMP状态，并拒绝阶段或配置串用。"""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "ai_isp_training_checkpoint_v4":
        raise ValueError("不是 V4 训练 Checkpoint")
    if payload.get("stage") != config.stage_name:
        raise ValueError("Checkpoint 训练阶段与当前配置不一致")
    if payload.get("config_hash") != _config_hash(config):
        raise ValueError("Checkpoint 配置 Hash 与当前配置不一致")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    for name, module in (extra_modules or {}).items():
        if name not in payload.get("extra_modules", {}):
            raise ValueError(f"Checkpoint 缺少附加模块 {name}")
        module.load_state_dict(payload["extra_modules"][name])
    return int(payload["step"])


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _denoise(model: nn.Module, noisy: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
    method = getattr(model, "denoise", None)
    return method(noisy, noise_pred) if method else torch.clamp(noisy - noise_pred, 0.0, 1.0)


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))


def _snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """复制到 CPU，避免后续优化器更新污染最佳权重。"""

    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}


def train_supervised_stage(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    config: TrainingStageConfig,
) -> dict[str, object]:
    """运行 Teacher 或 Student 的监督训练，支持累积、AMP和断点恢复。"""

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config.device)
    model = model.to(device).train()
    if config.student_activation_checkpointing and hasattr(model, "enable_activation_checkpointing"):
        model.enable_activation_checkpointing(True)
    criterion = DarkPreviewLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = _build_grad_scaler(device, config.student_amp)
    start_step = 0
    if config.resume_from:
        start_step = load_training_checkpoint(config.resume_from, model, optimizer, scaler, config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    best_step = start_step
    best_state = _snapshot_state_dict(model)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step in range(start_step, config.steps):
        scale = _learning_rate_scale(step, config.steps, config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        batch, iterator = _next_batch(iterator, loader)
        noisy = batch["noisy"].to(device).float()
        clean = batch["clean"].to(device).float()
        condition = batch["condition"].to(device).float()
        with _autocast(device, config.student_amp):
            noise_pred = model(noisy, condition)
            output = _denoise(model, noisy, noise_pred)
            losses = criterion(output, clean, condition)
            scaled_loss = losses["total"] / config.accumulation_steps
        scaler.scale(scaled_loss).backward()
        update = (step + 1) % config.accumulation_steps == 0 or step + 1 == config.steps
        gradient_norm = 0.0
        if update:
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        score = raw_psnr(output, clean)
        history.append({
            "step": step + 1,
            "loss_total": float(losses["total"].detach()),
            "raw_psnr": score,
            "gradient_norm": gradient_norm,
            "updated": int(update),
        })
        best_candidate = update and (
            config.checkpoint_interval <= 0
            or (step + 1) % config.checkpoint_interval == 0
            or step + 1 == config.steps
        )
        if best_candidate and score > best_score:
            best_score = score
            best_step = step + 1
            best_state = _snapshot_state_dict(model)
        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            save_training_checkpoint(
                Path(config.output_dir) / "checkpoints" / f"step_{step + 1}.pt",
                model,
                optimizer,
                scaler,
                step + 1,
                config,
            )
    peak_memory = _peak_memory_gb(device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.safetensors"
    final_path = output_dir / "final_model.safetensors"
    final_state = _snapshot_state_dict(model)
    save_file(best_state, best_path)
    save_file(final_state, final_path)
    model.load_state_dict(best_state)
    report = {
        "stage": config.stage_name,
        "config": asdict(config),
        "config_hash": _config_hash(config),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory,
        "memory_soft_limit_exceeded": peak_memory > config.memory_soft_limit_gb,
        "best_step": best_step,
        "best_raw_psnr": best_score if best_score != float("-inf") else None,
        "best_model": str(best_path),
        "final_model": str(final_path),
        "selected_for_next_stage": str(best_path),
        "history": history,
    }
    (output_dir / f"{config.stage_name}_训练报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


class FeatureDistillationAdapters(nn.Module):
    """把 Student Stage3/Middle 映射到 Teacher 对应特征，仅训练期存在。"""

    def __init__(self, student_channels: tuple[int, int], teacher_channels: tuple[int, int] = (128, 512)) -> None:
        super().__init__()
        self.stage3 = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
        self.middle = nn.Conv2d(student_channels[1], teacher_channels[1], 1)

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        return value / value.square().mean(dim=(-2, -1), keepdim=True).add(1e-8).sqrt()

    def align(
        self,
        student_features: tuple[torch.Tensor, torch.Tensor],
        teacher_features: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        stage3 = self.stage3(student_features[0])
        middle = functional.avg_pool2d(self.middle(student_features[1]), kernel_size=2, stride=2)
        if stage3.shape[-2:] != teacher_features[0].shape[-2:]:
            raise ValueError("Stage3 KD 特征空间尺寸不一致")
        if middle.shape[-2:] != teacher_features[1].shape[-2:]:
            raise ValueError("Middle KD 只允许固定 2× Average Pool 对齐")
        return (stage3, middle), teacher_features

    def loss(
        self,
        student_features: tuple[torch.Tensor, torch.Tensor],
        teacher_features: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        student_aligned, teacher_aligned = self.align(student_features, teacher_features)
        stage3 = functional.l1_loss(
            self._rms_normalize(student_aligned[0]), self._rms_normalize(teacher_aligned[0])
        )
        middle = functional.l1_loss(
            self._rms_normalize(student_aligned[1]), self._rms_normalize(teacher_aligned[1])
        )
        return 0.5 * (stage3 + middle)

    def attention_loss(
        self,
        student_features: tuple[torch.Tensor, torch.Tensor],
        teacher_features: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        student_aligned, teacher_aligned = self.align(student_features, teacher_features)
        return attention_distillation_loss(student_aligned, teacher_aligned)


def train_distillation_stage(
    teacher: nn.Module,
    student: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    config: TrainingStageConfig,
) -> dict[str, object]:
    """Teacher FP32/no_grad、Student AMP 的可恢复 Feature KD。"""

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config.device)
    teacher = teacher.to(device).eval()
    teacher.requires_grad_(False)
    student = student.to(device).train()
    if config.student_activation_checkpointing and hasattr(student, "enable_activation_checkpointing"):
        student.enable_activation_checkpointing(True)
    student_widths = tuple(int(layer.out_channels) for layer in student.downs)
    adapters = FeatureDistillationAdapters((student_widths[1], student_widths[2])).to(device).train()
    parameters = list(student.parameters()) + list(adapters.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = _build_grad_scaler(device, config.student_amp)
    start_step = 0
    if config.resume_from:
        start_step = load_training_checkpoint(
            config.resume_from, student, optimizer, scaler, config, {"feature_adapters": adapters}
        )
    criterion = StudentCompositeLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    best_step = start_step
    best_state = _snapshot_state_dict(student)
    started = time.perf_counter()
    for step in range(start_step, config.steps):
        scale = _learning_rate_scale(step, config.steps, config.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        batch, iterator = _next_batch(iterator, loader)
        noisy = batch["noisy"].to(device).float()
        clean = batch["clean"].to(device).float()
        condition = batch["condition"].to(device).float()
        # no_grad 比 inference_mode 更适合：Teacher 张量随后会作为 Student Loss 的 Target。
        with torch.no_grad():
            _, teacher_features = teacher.forward_with_features(noisy, condition)
            teacher_features = tuple(value.detach() for value in teacher_features)
        with _autocast(device, config.student_amp):
            noise_pred, student_features = student.forward_with_features(noisy, condition)
            output = _denoise(student, noisy, noise_pred)
            feature_kd = adapters.loss(student_features, teacher_features)
            attention_kd = adapters.attention_loss(student_features, teacher_features)
            residual_sigma = (noisy - clean).flatten(1).std(dim=1, keepdim=True).clamp_min(1e-4)
            residual_sigma = residual_sigma[:, :, None, None]
            temporal_1 = (clean + torch.randn_like(clean) * residual_sigma).clamp(0.0, 1.0)
            temporal_2 = (clean + torch.randn_like(clean) * residual_sigma).clamp(0.0, 1.0)
            temporal_pred_1 = student(temporal_1, condition)
            temporal_pred_2 = student(temporal_2, condition)
            temporal = temporal_consistency_loss(
                clean, temporal_1, temporal_pred_1, temporal_2, temporal_pred_2
            )
            losses = criterion(
                output,
                clean,
                condition,
                feature_kd=feature_kd,
                attention_kd=attention_kd,
                temporal=temporal,
            )
            total = losses["total"]
            scaled_loss = total / config.accumulation_steps
        scaler.scale(scaled_loss).backward()
        update = (step + 1) % config.accumulation_steps == 0 or step + 1 == config.steps
        gradient_norm = 0.0
        if update:
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        score = raw_psnr(output, clean)
        history.append({
            "step": step + 1,
            "loss_total": float(total.detach()),
            "loss_feature_kd": float(feature_kd.detach()),
            "loss_attention_kd": float(attention_kd.detach()),
            "loss_temporal": float(temporal.detach()),
            "raw_psnr": score,
            "gradient_norm": gradient_norm,
            "updated": int(update),
        })
        best_candidate = update and (
            config.checkpoint_interval <= 0
            or (step + 1) % config.checkpoint_interval == 0
            or step + 1 == config.steps
        )
        if best_candidate and score > best_score:
            best_score = score
            best_step = step + 1
            best_state = _snapshot_state_dict(student)
        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            save_training_checkpoint(
                Path(config.output_dir) / "checkpoints" / f"step_{step + 1}.pt",
                student,
                optimizer,
                scaler,
                step + 1,
                config,
                {"feature_adapters": adapters},
            )
    peak_memory = _peak_memory_gb(device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.safetensors"
    final_path = output_dir / "final_model.safetensors"
    final_state = _snapshot_state_dict(student)
    save_file(best_state, best_path)
    save_file(final_state, final_path)
    student.load_state_dict(best_state)
    report = {
        "stage": config.stage_name,
        "teacher_mode": "eval_fp32_no_grad",
        "student_amp": bool(config.student_amp and device.type == "cuda"),
        "config": asdict(config),
        "config_hash": _config_hash(config),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": peak_memory,
        "memory_soft_limit_exceeded": peak_memory > config.memory_soft_limit_gb,
        "best_step": best_step,
        "best_raw_psnr": best_score if best_score != float("-inf") else None,
        "best_model": str(best_path),
        "final_model": str(final_path),
        "selected_for_next_stage": str(best_path),
        "history": history,
    }
    (output_dir / f"{config.stage_name}_训练报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
