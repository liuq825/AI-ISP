"""Q0/Q1/Q2/Q3 完整 LSQ/LSQ+ QAT 阶段执行器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import torch
import numpy as np
from torch import nn
from safetensors.torch import save_file

from ai_isp.losses.dark_preview_losses import DarkPreviewLoss, raw_psnr
from ai_isp.quantization.lsqplus_qat import (
    QatPolicy,
    audit_film_quantization,
    audit_qat_model,
    calibrate_qat_model,
    configure_qat_phase,
    iter_quantizers,
    prepare_qat_model,
)
from ai_isp.training_stages import FeatureDistillationAdapters


@dataclass(frozen=True)
class QatTrainingConfig:
    output_dir: str
    device: str = "auto"
    q1_steps: int = 2000
    q2_steps: int = 48000
    q3_steps: int = 10000
    quant_learning_rate: float = 1e-4
    weight_learning_rate: float = 1e-5
    q3_learning_rate: float = 5e-6
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    seed: int = 20260804
    checkpoint_interval: int = 1000
    resume_from: str | None = None


def _config_hash(config: QatTrainingConfig) -> str:
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("resume_from", None)
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("QAT 请求 CUDA，但当前没有可用 GPU")
    return result


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _parameter_groups(model: nn.Module, quant_lr: float, weight_lr: float):
    quant_ids = {id(parameter) for _, quantizer in iter_quantizers(model) for parameter in quantizer.parameters()}
    quant = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in quant_ids]
    weights = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in quant_ids]
    groups = []
    if weights:
        groups.append({"params": weights, "lr": weight_lr, "name": "network_weights"})
    if quant:
        groups.append({"params": quant, "lr": quant_lr, "name": "quant_parameters"})
    if not groups:
        raise RuntimeError("当前 QAT Phase 没有可训练参数")
    return groups


def _run_phase(
    model: nn.Module,
    loader,
    phase: str,
    steps: int,
    config: QatTrainingConfig,
    teacher: nn.Module | None,
    adapters: FeatureDistillationAdapters | None,
    start_step: int = 0,
    optimizer_state: dict[str, object] | None = None,
    checkpoint_callback=None,
) -> list[dict[str, float | int | str]]:
    if start_step < 0 or start_step > steps:
        raise ValueError(f"{phase} 恢复 Step 越界: {start_step}/{steps}")
    configure_qat_phase(model, phase)
    if adapters is not None:
        adapters.requires_grad_(phase != "q1")
    groups = _parameter_groups(
        model,
        config.quant_learning_rate,
        config.q3_learning_rate if phase == "q3" else config.weight_learning_rate,
    )
    if adapters is not None and phase != "q1":
        groups.append({"params": list(adapters.parameters()), "lr": groups[0]["lr"], "name": "kd_adapters"})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    criterion = DarkPreviewLoss(raw_weight=0.50, tone_weight=0.25, gradient_weight=0.15)
    history = []
    iterator = iter(loader)
    for step in range(start_step, steps):
        batch, iterator = _next_batch(iterator, loader)
        device = next(model.parameters()).device
        noisy = batch["noisy"].to(device).float()
        clean = batch["clean"].to(device).float()
        condition = batch["condition"].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        noise_pred, student_features = model.forward_with_features(noisy, condition)
        output = model.denoise(noisy, noise_pred)
        losses = criterion(output, clean, condition)
        feature_kd = output.new_tensor(0.0)
        if teacher is not None and adapters is not None:
            with torch.no_grad():
                _, teacher_features = teacher.forward_with_features(noisy, condition)
            feature_kd = adapters.loss(student_features, tuple(item.detach() for item in teacher_features))
        total = losses["total"] + 0.10 * feature_kd
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for group in groups for parameter in group["params"]], config.gradient_clip
        ))
        optimizer.step()
        history.append({
            "phase": phase,
            "step": step + 1,
            "loss_total": float(total.detach()),
            "loss_feature_kd": float(feature_kd.detach()),
            "raw_psnr": raw_psnr(output, clean),
            "gradient_norm": gradient_norm,
        })
        if checkpoint_callback is not None and (
            (config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0)
            or step + 1 == steps
        ):
            checkpoint_callback(optimizer, step + 1)
    return history


def _save_qat_checkpoint(
    path: Path,
    model: nn.Module,
    adapters: FeatureDistillationAdapters | None,
    optimizer: torch.optim.Optimizer,
    phase: str,
    phase_step: int,
    q0_report: dict[str, object],
    policy: QatPolicy,
    config: QatTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "ai_isp_qat_checkpoint_v4",
        "config_hash": _config_hash(config),
        "policy": asdict(policy),
        "phase": phase,
        "phase_step": phase_step,
        "q0": q0_report,
        "model": model.state_dict(),
        "adapters": adapters.state_dict() if adapters is not None else None,
        "optimizer": optimizer.state_dict(),
    }, path)


def train_qat(
    fp32_model: nn.Module,
    loader,
    calibration_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    policy: QatPolicy,
    config: QatTrainingConfig,
    teacher: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    """运行完整QAT；Smoke配置可把每段步数降为1，但不会跳过任何阶段。"""

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = _device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = prepare_qat_model(fp32_model, policy).to(device)
    calibration_iterator = iter(calibration_batches)
    try:
        first_image, first_condition = next(calibration_iterator)
    except StopIteration as error:
        raise ValueError("Q0 校准集不能为空") from error
    reference_image = first_image.to(device).float()
    reference_condition = first_condition.to(device).float()

    teacher = teacher.to(device).eval().requires_grad_(False) if teacher is not None else None
    adapters = None
    if teacher is not None:
        widths = tuple(int(layer.out_channels) for layer in fp32_model.downs)
        adapters = FeatureDistillationAdapters((widths[1], widths[2])).to(device)

    resume_payload = None
    if config.resume_from:
        resume_payload = torch.load(Path(config.resume_from), map_location="cpu", weights_only=False)
        if resume_payload.get("format") != "ai_isp_qat_checkpoint_v4":
            raise ValueError("不是 V4 QAT Checkpoint")
        if resume_payload.get("config_hash") != _config_hash(config):
            raise ValueError("QAT Checkpoint 配置 Hash 不一致")
        if resume_payload.get("policy") != asdict(policy):
            raise ValueError("QAT Checkpoint Quant Policy 不一致")
        model.load_state_dict(resume_payload["model"])
        if adapters is not None:
            if resume_payload.get("adapters") is None:
                raise ValueError("QAT Checkpoint 缺少 Feature Adapter")
            adapters.load_state_dict(resume_payload["adapters"])
        q0_report = resume_payload["q0"]
    else:
        def device_calibration_batches():
            # 仅保留首批做 Q3 等价性检查，其余校准数据流式送入，避免 4096 帧常驻内存。
            yield reference_image, reference_condition
            for image, condition in calibration_iterator:
                yield image.to(device).float(), condition.to(device).float()

        q0_report = calibrate_qat_model(model, device_calibration_batches())

    phase_specs = (("q1", config.q1_steps), ("q2", config.q2_steps), ("q3", config.q3_steps))
    phase_names = [name for name, _ in phase_specs]
    resume_phase = resume_payload.get("phase") if resume_payload else None
    if resume_phase is not None and resume_phase not in phase_names:
        raise ValueError(f"QAT Checkpoint Phase 非法: {resume_phase}")
    resume_index = phase_names.index(resume_phase) if resume_phase else 0
    history = []
    freeze_drift = 0.0
    for phase_index, (phase, steps) in enumerate(phase_specs):
        if resume_payload is not None and phase_index < resume_index:
            continue
        if phase == "q3":
            # Q3 只切换 requires_grad，冻结前后必须完全等价。
            model.eval()
            with torch.no_grad():
                before_freeze = model(reference_image, reference_condition)
            configure_qat_phase(model, "q3")
            with torch.no_grad():
                after_freeze = model(reference_image, reference_condition)
            freeze_drift = float((before_freeze - after_freeze).abs().max())
            if freeze_drift != 0.0:
                raise RuntimeError(f"Q3 冻结改变前向结果: {freeze_drift}")
            model.train()
        matching_resume = resume_payload is not None and phase == resume_phase
        start_step = int(resume_payload["phase_step"]) if matching_resume else 0
        optimizer_state = resume_payload.get("optimizer") if matching_resume else None

        def checkpoint_callback(optimizer, phase_step, phase_name=phase):
            _save_qat_checkpoint(
                output_dir / "checkpoints" / f"{phase_name}_step_{phase_step}.pt",
                model, adapters, optimizer, phase_name, phase_step, q0_report, policy, config,
            )

        history.extend(_run_phase(
            model, loader, phase, steps, config, teacher, adapters,
            start_step=start_step, optimizer_state=optimizer_state, checkpoint_callback=checkpoint_callback,
        ))
        if matching_resume:
            resume_payload = None
    qat_weights_path = output_dir / "qat_model.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()},
        qat_weights_path,
    )
    report = {
        "policy": asdict(policy),
        "config": asdict(config),
        "config_hash": _config_hash(config),
        "resumed_from": config.resume_from,
        "qat_weights": str(qat_weights_path),
        "q0": q0_report,
        "q3_freeze_max_abs_drift": freeze_drift,
        "history": history,
        "quantizers": audit_qat_model(model),
        "film_audit": audit_film_quantization(fp32_model, model),
        "limitations": [
            "本机 QAT 只验证算法链路；LSQ 与 LSQ+ 的发布选择必须使用真实 RYYB 数据",
            "非零 Offset、FiLM FP16 Island 和最终 OM 必须通过目标 DDK/麒麟9000门禁",
        ],
    }
    (output_dir / "QAT训练与量化审计.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save({
        "format": "ai_isp_qat_checkpoint_v4_final",
        "config_hash": _config_hash(config),
        "model": model.state_dict(),
        "adapters": adapters.state_dict() if adapters is not None else None,
        "policy": asdict(policy),
    }, output_dir / "qat_checkpoint.pt")
    return model, report
