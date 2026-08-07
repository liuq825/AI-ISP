"""V6.1 完整开发流水线：训练→三候选→Q1选优→唯一Q2/Q3→制品闭锁。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
import yaml

from ai_isp.data.ryyb_contract import RYYB_CFA_OFFSETS, validate_release_dataset_requirements
from ai_isp.data.ryyb_dataset import BalancedCameraSampler, RyybRawPatchDataset
from ai_isp.data.sidd_dataset import SiddRawPatchDataset
from ai_isp.export.freeze_topology import freeze_topology
from ai_isp.export.om_release import (
    OmCompileConfig,
    build_v6_1_engineering_manifest,
    compile_single_om,
    sha256_file,
)
from ai_isp.export.static_profiles import export_fixed_model
from ai_isp.losses.dark_preview_losses import fixed_reference_isp, global_ssim, raw_psnr
from ai_isp.models import ConditionalNAFNetW32Teacher, FiLMAffine, build_mobile_nafnet_w16
from ai_isp.pruning import StructuredMobileNAFPruner, estimate_macs_at_shape
from ai_isp.qat_training import QatTrainingConfig, train_qat
from ai_isp.quantization import (
    QatPolicy,
    audit_dynamic_affine_equivalence,
    iter_quantizers,
    prepare_qdq_export,
)
from ai_isp.runtime import DmaBufFrame, DmaBufPoolContract
from ai_isp.selection import CandidateMetrics, converge_candidate_artifacts, select_best_candidate
from ai_isp.training_stages import TrainingStageConfig, train_distillation_stage, train_supervised_stage


V6_CANDIDATE_TOPOLOGIES = {
    "P10-16": (16, 32, 64, 112),
    "P18-16": (16, 32, 48, 128),
    "P36-16": (16, 32, 48, 96),
}


@dataclass(frozen=True)
class V6PipelineConfig:
    output_dir: str
    dataset_root: str = "datasets/SIDD_Training_Subset"
    ryyb_manifest: str | None = None
    patch_size: int = 32
    max_pairs: int = 2
    samples_per_epoch: int = 8
    batch_size: int = 1
    teacher_steps: int = 1
    student_steps: int = 1
    kd_steps: int = 1
    candidate_recovery_steps: int | dict[str, int] = 1
    candidate_recovery_max_steps: dict[str, int] | None = None
    phase1_steps: int = 1
    calibration_frames: int = 1
    q1_steps: int = 1
    q2_steps: int = 1
    q3_steps: int = 1
    device: str = "cpu"
    seed: int = 20260807
    export_mode: str = "smoke"
    soc_version: str = "Kirin9000_TARGET_DDK_REQUIRED"
    student_activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.export_mode not in ("smoke", "release"):
            raise ValueError("export_mode 只允许 smoke/release")
        if self.export_mode == "release":
            expected = {"P10-16": 80000, "P18-16": 120000, "P36-16": 180000}
            maximum = {"P10-16": 120000, "P18-16": 180000, "P36-16": 240000}
            if self.candidate_recovery_steps != expected or self.candidate_recovery_max_steps != maximum:
                raise ValueError("量产剪枝恢复周期必须与 V6.1 冻结值一致")
            if self.phase1_steps != 10000 or self.q1_steps != 2000:
                raise ValueError("量产三候选必须执行 Phase1=10k、Q1=2k")
            if not 50000 <= self.q2_steps <= 80000 or self.q3_steps != 10000:
                raise ValueError("量产获胜候选必须执行 Q2=50k~80k、Q3=10k")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V6PipelineConfig":
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def recovery_steps_for(self, candidate_id: str) -> int:
        if isinstance(self.candidate_recovery_steps, int):
            return self.candidate_recovery_steps
        return int(self.candidate_recovery_steps[candidate_id])

    def recovery_max_steps_for(self, candidate_id: str) -> int:
        if self.candidate_recovery_max_steps is None:
            return self.recovery_steps_for(candidate_id)
        return int(self.candidate_recovery_max_steps[candidate_id])


def _loader(config: V6PipelineConfig, full_size: bool = False):
    if config.ryyb_manifest:
        patch_size: int | tuple[int, int] = (768, 1024) if full_size else config.patch_size
        dataset = RyybRawPatchDataset(
            config.ryyb_manifest,
            split="train",
            patch_size=patch_size,
            samples_per_epoch=config.samples_per_epoch,
            seed=config.seed,
            deterministic=(config.export_mode == "smoke"),
        )
        if config.export_mode == "release":
            validate_release_dataset_requirements(dataset.records)
            if config.batch_size != 1 or config.calibration_frames < 4096 or config.calibration_frames % 2:
                raise ValueError("量产 Q0 要求 Batch=1、校准帧不少于4096且 Main/Tele 等量")
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=BalancedCameraSampler(dataset, config.seed),
            num_workers=0,
        ), False
    if full_size:
        # SIDD 只用于 CPU 工程冒烟，不能伪造发布全尺寸训练证据。
        return _loader(config, full_size=False)
    dataset = SiddRawPatchDataset(
        config.dataset_root,
        patch_size=config.patch_size,
        samples_per_epoch=config.samples_per_epoch,
        seed=config.seed,
        max_pairs=config.max_pairs,
        deterministic=True,
        augment=True,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0), True


def _stage_config(
    config: V6PipelineConfig,
    name: str,
    steps: int,
    learning_rate: float,
    base_dir: Path | None = None,
) -> TrainingStageConfig:
    return TrainingStageConfig(
        stage_name=name,
        output_dir=str((base_dir or (Path(config.output_dir) / "training")) / name),
        steps=steps,
        learning_rate=learning_rate,
        accumulation_steps=1 if config.device == "cpu" else 16,
        warmup_steps=0 if steps < 10 else min(5000, steps // 10),
        checkpoint_interval=0 if steps < 10 else 1000,
        device=config.device,
        seed=config.seed,
        student_activation_checkpointing=(
            config.student_activation_checkpointing and name != "teacher_fp32"
        ),
    )


def _first_inputs(loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = next(iter(loader))
    return batch["noisy"].float(), batch["clean"].float(), batch["condition"].float()


def _calibration_batches(loader, frame_count: int, force_balanced_camera: bool = False):
    iterator = iter(loader)
    for _ in range(frame_count):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        condition = batch["condition"].float()
        if force_balanced_camera:
            condition = _camera_condition(condition, "main" if _ % 2 == 0 else "tele")
        yield batch["noisy"].float(), condition


def _activation_saturation_probe(
    model: nn.Module,
    loader,
    frame_count: int,
    force_balanced_camera: bool,
) -> dict[str, object]:
    """Measure the hard gate after Q1, excluding provisional training forwards and weights."""

    quantizers = list(iter_quantizers(model))
    for _, quantizer in quantizers:
        quantizer.last_saturation_rate.zero_()
        quantizer.max_saturation_rate.zero_()
    model.eval()
    with torch.no_grad():
        for image, condition in _calibration_batches(loader, frame_count, force_balanced_camera):
            model(image, condition)
    layers = {
        name: float(quantizer.max_saturation_rate)
        for name, quantizer in quantizers
        if quantizer.channel_axis < 0
    }
    return {
        "definition": "quantization_pre_range_overflow / activation_elements",
        "frames": frame_count,
        "camera_balanced": True,
        "excludes_per_channel_weights": True,
        "layers": layers,
        "maximum": max(layers.values(), default=0.0),
    }


def _tune_activation_ranges(
    model: nn.Module,
    loader,
    frame_count: int,
    force_balanced_camera: bool,
) -> dict[str, object]:
    """Expand only failing activation ranges; never relax the 0.1% gate."""

    history = []
    quantizers = dict(iter_quantizers(model))
    for iteration in range(9):
        probe = _activation_saturation_probe(
            model, loader, frame_count, force_balanced_camera
        )
        history.append({"iteration": iteration, "maximum": probe["maximum"]})
        failing = {
            name for name, rate in probe["layers"].items() if float(rate) >= 0.001
        }
        if not failing:
            return {**probe, "range_tuning_history": history, "passed": True}
        with torch.no_grad():
            for name in failing:
                quantizer = quantizers[name]
                old_scale = quantizer.scale.detach()
                new_scale = old_scale * 1.05
                if not quantizer.symmetric:
                    midpoint_q = (quantizer.qmin + quantizer.qmax) / 2.0
                    center = quantizer.offset.detach() + midpoint_q * old_scale
                    quantizer.offset.copy_(center - midpoint_q * new_scale)
                quantizer.log_scale.copy_(
                    torch.log(torch.expm1(new_scale).clamp_min(1e-12))
                )
    return {**probe, "range_tuning_history": history, "passed": False}


def _activation_peak_at_release_shape(
    model: nn.Module, image: torch.Tensor, condition: torch.Tensor
) -> int:
    source_height, source_width = image.shape[-2:]
    ratio = (768 // source_height) * (1024 // source_width)
    peaks = []
    hooks = []

    def record(_module, _inputs, output) -> None:
        if isinstance(output, torch.Tensor) and output.ndim == 4:
            peaks.append(output.numel() * ratio * 2)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, FiLMAffine)):
            hooks.append(module.register_forward_hook(record))
    try:
        with torch.no_grad():
            model.eval()(image, condition)
    finally:
        for hook in hooks:
            hook.remove()
    return max(peaks, default=0)


def _camera_condition(condition: torch.Tensor, camera: str) -> torch.Tensor:
    result = condition.clone()
    result[:, 10:22] = 0.0
    if camera == "main":
        result[:, (10, 14, 18)] = 1.0
    else:
        result[:, (12, 15, 20)] = 1.0
    result[:, 22:24] = 1.0
    return result


def _quality_metrics(
    teacher: nn.Module,
    candidate: nn.Module,
    image: torch.Tensor,
    clean: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[float, float, float, float]:
    teacher.eval().cpu()
    candidate.eval().cpu()
    camera_drops = []
    ssim_drops = []
    delta_e_proxies = []
    with torch.no_grad():
        for camera in ("main", "tele"):
            camera_condition = _camera_condition(condition, camera)
            teacher_output = torch.clamp(image - teacher(image, camera_condition), 0.0, 1.0)
            candidate_output = torch.clamp(image - candidate(image, camera_condition), 0.0, 1.0)
            camera_drops.append(max(0.0, raw_psnr(teacher_output, clean) - raw_psnr(candidate_output, clean)))
            ssim_drops.append(max(0.0, global_ssim(teacher_output, clean) - global_ssim(candidate_output, clean)))
            teacher_rgb = fixed_reference_isp(teacher_output, camera_condition)
            candidate_rgb = fixed_reference_isp(candidate_output, camera_condition)
            delta_e_proxies.append(float((teacher_rgb - candidate_rgb).abs().mean() * 10.0))
    return max(camera_drops), max(ssim_drops), max(delta_e_proxies), max(camera_drops)


def _benchmark_reference_onnx(path: Path, image: torch.Tensor, condition: torch.Tensor) -> float:
    session = onnxruntime.InferenceSession(str(path), providers=("CPUExecutionProvider",))
    input_shape = session.get_inputs()[0].shape
    expected_height, expected_width = int(input_shape[-2]), int(input_shape[-1])
    benchmark_image = image
    if image.shape[-2:] != (expected_height, expected_width):
        benchmark_image = functional.interpolate(
            image, size=(expected_height, expected_width), mode="bilinear", align_corners=False
        )
    feeds = {"packed_raw": benchmark_image.numpy(), "condition": condition.numpy()}
    session.run(None, feeds)
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        session.run(None, feeds)
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(samples, 95))


def _dynamic_affine_audit(model: nn.Module, image: torch.Tensor, condition: torch.Tensor) -> dict[str, object]:
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = []

    def capture(name: str):
        def hook(_module, inputs) -> None:
            captured[name] = (inputs[0].detach(), inputs[1].detach())
        return hook

    hooks.append(model.film_stage3.register_forward_pre_hook(capture("stage3")))
    hooks.append(model.film_middle.register_forward_pre_hook(capture("middle")))
    try:
        with torch.no_grad():
            model.eval()(image, condition)
    finally:
        for hook in hooks:
            hook.remove()
    report = {}
    for name, (feature, parameters) in captured.items():
        channels = feature.shape[1]
        gamma = 1.0 + 0.1 * functional.hardtanh(parameters[:, :channels, None, None], -1.0, 1.0)
        beta = 0.1 * functional.hardtanh(parameters[:, channels:, None, None], -1.0, 1.0)
        report[name] = audit_dynamic_affine_equivalence(feature, gamma, beta)
        if report[name]["integer"]["int32_overflow"] or report[name]["max_abs_error"] >= 0.01:
            raise RuntimeError(f"{name} Dynamic Affine 整数参考审计失败")
    return report


def _write_selection_markdown(path: Path, report: dict[str, object], smoke_only: bool) -> None:
    rows = []
    for item in report["candidates"]:
        rows.append(
            f"|{item['candidate_id']}|{item['raw_psnr_drop_db']:.6f}|{item['ssim_drop']:.6f}|"
            f"{item['macs']}|{item['max_saturation_rate']:.6f}|"
            f"{'通过' if not item['hard_gate_failures'] else '淘汰'}|"
        )
    path.write_text(
        "# V6.1 三候选模拟选优报告\n\n"
        f"- 选中候选：`{report['selected_candidate_id']}`\n"
        f"- 数据性质：`{'SIDD smoke_only' if smoke_only else '真实 RYYB'}`\n"
        "- 当前结论仅用于开发候选相对排序，不代表麒麟 9000 目标端通过。\n\n"
        "|候选|RAW PSNR回归(dB)|SSIM回归|发布Shape MAC|最大饱和率|结果|\n"
        "|---|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _copy_release_profiles(release_dir: Path) -> dict[str, Path]:
    mapping = {
        "condition_schema": Path("configs/release/condition_schema_v2.json"),
        "sensor_profiles": Path("configs/release/sensor_profiles_ryyb.json"),
        "unpack_profiles": Path("configs/release/unpack_profiles_ryyb.json"),
        "raw_domain_profile": Path("configs/release/raw_domain_profile.json"),
        "lsc_profiles": Path("configs/release/lsc_profiles_ryyb.json"),
        "reference_isp_profile": Path("configs/release/reference_isp_profile.json"),
        "noise_profiles": Path("configs/release/noise_profiles_ryyb.json"),
        "buffer_contract": Path("configs/release/buffer_contract_v1.json"),
        "quant_policy": Path("configs/release/quant_policy_v6_1_mixed.json"),
    }
    copied = {}
    for name, source in mapping.items():
        target = release_dir / source.name
        shutil.copy2(source, target)
        copied[name] = target
    return copied


def run_v6_pipeline(config: V6PipelineConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, smoke_only = _loader(config)
    phase1_loader, phase1_smoke_only = _loader(config, full_size=True)
    image, clean, condition = _first_inputs(loader)

    teacher = ConditionalNAFNetW32Teacher()
    student = build_mobile_nafnet_w16()
    teacher_report = train_supervised_stage(
        teacher, loader, _stage_config(config, "teacher_fp32", config.teacher_steps, 2e-4)
    )
    student_report = train_supervised_stage(
        student, loader, _stage_config(config, "student_fp32", config.student_steps, 2e-4)
    )
    kd_report = train_distillation_stage(
        teacher, student, loader, _stage_config(config, "student_kd_temporal", config.kd_steps, 1e-4)
    )

    candidate_root = output_dir / "candidates"
    candidate_models: dict[str, nn.Module] = {}
    candidate_q1_checkpoints: dict[str, Path] = {}
    candidate_reports: dict[str, object] = {}
    candidate_metrics: list[CandidateMetrics] = []
    pruner = StructuredMobileNAFPruner()

    for candidate_id, topology in V6_CANDIDATE_TOPOLOGIES.items():
        candidate_dir = candidate_root / candidate_id
        candidate = copy.deepcopy(student).cpu()
        pruning_report = pruner.prune_to_feature_channels(
            candidate, (image, condition), [(image, condition)], topology
        )
        frozen = freeze_topology(candidate, candidate_dir / "fp16_topology")
        recovery_initial = train_distillation_stage(
            teacher,
            candidate,
            loader,
            _stage_config(
                config,
                f"{candidate_id}_recovery",
                config.recovery_steps_for(candidate_id),
                5e-5,
                candidate_dir / "training",
            ),
        )
        recovery_raw_drop, recovery_ssim_drop, _, _ = _quality_metrics(
            teacher, candidate, image, clean, condition
        )
        recovery_extension = None
        initial_recovery_steps = config.recovery_steps_for(candidate_id)
        maximum_recovery_steps = config.recovery_max_steps_for(candidate_id)
        if maximum_recovery_steps < initial_recovery_steps:
            raise ValueError(f"{candidate_id} 恢复最大步数小于默认步数")
        if (
            recovery_raw_drop > 0.10 or recovery_ssim_drop > 0.002
        ) and maximum_recovery_steps > initial_recovery_steps:
            recovery_extension = train_distillation_stage(
                teacher,
                candidate,
                loader,
                _stage_config(
                    config,
                    f"{candidate_id}_recovery_extension",
                    maximum_recovery_steps - initial_recovery_steps,
                    2e-5,
                    candidate_dir / "training",
                ),
            )
        recovery = {
            "default_steps": initial_recovery_steps,
            "maximum_steps": maximum_recovery_steps,
            "quality_after_default": {
                "raw_psnr_drop_db": recovery_raw_drop,
                "ssim_drop": recovery_ssim_drop,
            },
            "extended_to_maximum": recovery_extension is not None,
            "initial": recovery_initial,
            "extension": recovery_extension,
        }
        phase1 = train_supervised_stage(
            candidate,
            phase1_loader,
            _stage_config(
                config,
                f"{candidate_id}_phase1_fullsize",
                config.phase1_steps,
                1e-5,
                candidate_dir / "training",
            ),
        )
        q1_dir = candidate_dir / "q1_probe"
        qat_model, q1_report = train_qat(
            candidate,
            loader,
            lambda: _calibration_batches(loader, config.calibration_frames, smoke_only),
            QatPolicy(),
            QatTrainingConfig(
                output_dir=str(q1_dir),
                device=config.device,
                q1_steps=config.q1_steps,
                q2_steps=config.q2_steps,
                q3_steps=config.q3_steps,
                checkpoint_interval=max(config.q1_steps, 1),
                run_through_phase="q1",
            ),
            teacher=teacher,
        )
        q1_checkpoint = q1_dir / "checkpoints" / f"q1_step_{config.q1_steps}.pt"
        if not q1_checkpoint.is_file():
            raise RuntimeError(f"{candidate_id} 缺少 Q1 续训 Checkpoint")
        candidate_q1_checkpoints[candidate_id] = q1_checkpoint
        saturation_probe = _tune_activation_ranges(
            qat_model, loader, config.calibration_frames, smoke_only
        )
        if not saturation_probe["passed"]:
            raise RuntimeError(f"{candidate_id} 激活范围自适应后仍未通过 0.1% 饱和率门禁")
        checkpoint_payload = torch.load(q1_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_payload["model"] = qat_model.state_dict()
        checkpoint_payload["post_q1_range_tuning"] = saturation_probe["range_tuning_history"]
        torch.save(checkpoint_payload, q1_checkpoint)
        (candidate_dir / "activation_saturation_probe.json").write_text(
            json.dumps(saturation_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        prepare_qdq_export(qat_model)
        reference_report = export_fixed_model(
            qat_model,
            candidate_dir / "onnx_reference",
            config.export_mode,
            filename="candidate_reference_qdq.onnx",
        )["RYYB_4X3"]
        mixed_report = export_fixed_model(
            qat_model,
            candidate_dir / "onnx_mixed",
            config.export_mode,
            mixed_precision_boundary=True,
            filename="dark_preview_ryyb_4x3_mixed.onnx",
        )["RYYB_4X3"]
        reference_path = candidate_dir / "onnx_reference" / "candidate_reference_qdq.onnx"
        mixed_path = candidate_dir / "onnx_mixed" / "dark_preview_ryyb_4x3_mixed.onnx"
        raw_drop, ssim_drop, delta_e, bucket_drop = _quality_metrics(
            teacher, qat_model, image, clean, condition
        )
        max_saturation = float(saturation_probe["maximum"])
        operator_counts = reference_report["operator_counts"]
        metrics = CandidateMetrics(
            candidate_id=candidate_id,
            raw_psnr_drop_db=raw_drop,
            ssim_drop=ssim_drop,
            delta_e00_increase=delta_e,
            worst_camera_iso_drop_db=bucket_drop,
            macs=estimate_macs_at_shape(candidate, image, condition),
            activation_peak_bytes=_activation_peak_at_release_shape(candidate, image, condition),
            cast_count=int(mixed_report["operator_counts"].get("Cast", 0)),
            qdq_count=int(operator_counts.get("QuantizeLinear", 0) + operator_counts.get("DequantizeLinear", 0)),
            ort_p95_ms=_benchmark_reference_onnx(reference_path, image, condition),
            parameter_count=sum(parameter.numel() for parameter in candidate.parameters()),
            onnx_size_bytes=mixed_path.stat().st_size,
            max_saturation_rate=max_saturation,
            numerical_valid=bool(reference_report["alignment"]["available"]),
            graph_valid=(
                not reference_report["unsupported_operators"]
                and not mixed_report["unsupported_operators"]
                and mixed_report["condition_cast_count"] == 1
            ),
            mixed_precision_valid=(
                "QuantizeLinear" in reference_report["operators"]
                and mixed_report["input_element_types"]["packed_raw"] == 10
            ),
        )
        candidate_metrics.append(metrics)
        candidate_models[candidate_id] = candidate
        candidate_reports[candidate_id] = {
            "pruning": asdict(pruning_report),
            "frozen": frozen,
            "recovery": recovery,
            "phase1": phase1,
            "q1": {
                "policy": q1_report["policy"],
                "config_hash": q1_report["config_hash"],
                "run_through_phase": q1_report["run_through_phase"],
                "q0_quantizer_count": len(q1_report["q0"]),
                "history": q1_report["history"],
                "film_audit": q1_report["film_audit"],
            },
            "q1_activation_saturation_probe": saturation_probe,
            "reference_onnx": reference_report,
            "mixed_onnx": mixed_report,
            "metrics": asdict(metrics),
        }

    preselection_path = candidate_root / "preselection_metrics_v6_1.json"
    preselection_path.write_text(
        json.dumps([asdict(item) for item in candidate_metrics], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected, selection_report = select_best_candidate(candidate_metrics)
    selection_report["smoke_only"] = smoke_only
    selection_report["phase1_full_release_shape_executed"] = not phase1_smoke_only
    selection_summary = converge_candidate_artifacts(
        candidate_root, selected.candidate_id, selection_report
    )
    selected_model = candidate_models[selected.candidate_id]
    selected_checkpoint = candidate_q1_checkpoints[selected.candidate_id]

    final_qat_model, final_qat_report = train_qat(
        selected_model,
        loader,
        lambda: _calibration_batches(loader, config.calibration_frames, smoke_only),
        QatPolicy(),
        QatTrainingConfig(
            output_dir=str(output_dir / "winner_q2_q3"),
            device=config.device,
            q1_steps=config.q1_steps,
            q2_steps=config.q2_steps,
            q3_steps=config.q3_steps,
            checkpoint_interval=max(config.q1_steps, config.q2_steps, config.q3_steps, 1),
            resume_from=str(selected_checkpoint),
            run_through_phase="q3",
        ),
        teacher=teacher,
    )
    dynamic_affine_report = _dynamic_affine_audit(final_qat_model.cpu(), image, condition)
    prepare_qdq_export(final_qat_model)

    release_dir = output_dir / "release_v6_1"
    release_dir.mkdir(parents=True, exist_ok=True)
    final_onnx_report = export_fixed_model(
        final_qat_model,
        release_dir,
        config.export_mode,
        mixed_precision_boundary=True,
        filename="dark_preview_ryyb_4x3_mixed.onnx",
    )["RYYB_4X3"]
    weights_path = release_dir / "model_mixed_qat.safetensors"
    shutil.copy2(final_qat_report["qat_weights"], weights_path)
    topology_path = release_dir / "topology_v6_1.json"
    topology = selected_model.topology_manifest()
    topology.update({
        "selected_candidate_id": selected.candidate_id,
        "weights_file": weights_path.name,
        "weights_sha256": sha256_file(weights_path),
        "quantized_wrapper": "fixed_v6_mixed_lsqplus",
    })
    topology_path.write_text(json.dumps(topology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    copied_profiles = _copy_release_profiles(release_dir)
    development_selection_path = release_dir / "development_selection_v6_1.json"
    shutil.copy2(candidate_root / "development_selection_v6_1.json", development_selection_path)
    selection_markdown_path = release_dir / "selection_report_v6_1.md"
    _write_selection_markdown(selection_markdown_path, selection_summary, smoke_only)
    (release_dir / "dynamic_affine_integer_audit.json").write_text(
        json.dumps(dynamic_affine_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dma_pool = DmaBufPoolContract()
    dma_pool.import_once(0, 10, 2048 * 1536 * 2)
    dma_pool.submit(DmaBufFrame(0, 10, 0, 4096, 2048, 1536, 1))
    dma_pool.signal_consumer_ready(0, 2)
    dma_pool.release(0, 2)
    (release_dir / "buffer_contract_simulation.json").write_text(
        json.dumps(dma_pool.audit(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    onnx_path = release_dir / "dark_preview_ryyb_4x3_mixed.onnx"
    om_report = compile_single_om(OmCompileConfig(
        onnx_path=str(onnx_path),
        output_dir=str(release_dir / "om"),
        soc_version=config.soc_version,
        profile_mode=config.export_mode,
    ))
    artifacts = {
        "weights": weights_path,
        "topology": topology_path,
        **copied_profiles,
        "onnx": onnx_path,
        "development_selection": development_selection_path,
        "selection_report": selection_markdown_path,
    }
    manifest = build_v6_1_engineering_manifest(
        release_dir / "model_manifest_v6_1.json",
        artifacts,
        selected.candidate_id,
        om_report.get("om_path"),
        profile_mode=config.export_mode,
        compile_height=int(final_onnx_report["compile_height"]),
        compile_width=int(final_onnx_report["compile_width"]),
    )
    final_qat_summary = {
        "policy": final_qat_report["policy"],
        "config": final_qat_report["config"],
        "config_hash": final_qat_report["config_hash"],
        "resumed_from": final_qat_report["resumed_from"],
        "q0_quantizer_count": len(final_qat_report["q0"]),
        "q3_freeze_max_abs_drift": final_qat_report["q3_freeze_max_abs_drift"],
        "run_through_phase": final_qat_report["run_through_phase"],
        "history": final_qat_report["history"],
        "quantizer_count": len(final_qat_report["quantizers"]),
        "film_audit": final_qat_report["film_audit"],
        "full_report_local_path": str(
            output_dir / "winner_q2_q3" / "QAT训练与量化审计.json"
        ),
        "limitations": final_qat_report["limitations"],
    }

    # 最终制品已复制并 Hash；临时候选区不再保留任何可执行模型。
    selected_candidate_dir = (candidate_root / selected.candidate_id).resolve()
    if selected_candidate_dir.parent != candidate_root.resolve():
        raise RuntimeError("最终候选清理目标越界")
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    # V6.1 early development builds placed candidate training outputs in the
    # shared training directory. Remove only those validated legacy targets so
    # loser weights cannot survive an upgraded rerun.
    legacy_candidate_outputs = []
    training_root = (output_dir / "training").resolve()
    for candidate_id in V6_CANDIDATE_TOPOLOGIES:
        for suffix in ("recovery", "recovery_extension", "phase1_fullsize"):
            target = (training_root / f"{candidate_id}_{suffix}").resolve()
            if target.parent != training_root:
                raise RuntimeError("遗留候选清理目标越界")
            if target.exists():
                shutil.rmtree(target)
                legacy_candidate_outputs.append(str(target))

    summary = {
        "status": "v6_1_engineering_completed",
        "config": asdict(config),
        "smoke_only": smoke_only,
        "selected_candidate_id": selected.candidate_id,
        "development_selected": True,
        "dynamic_affine_target_pending": True,
        "target_validated": False,
        "release_ready": False,
        "stages": {
            "teacher": teacher_report,
            "student": student_report,
            "kd_temporal_attention": kd_report,
            "candidates": candidate_reports,
            "selection": selection_summary,
            "winner_q2_q3": final_qat_summary,
            "dynamic_affine_integer": dynamic_affine_report,
            "final_onnx": final_onnx_report,
            "om": om_report,
        },
        "manifest": manifest,
        "limitations": [
            "SIDD smoke_only 只能证明工程链路与相对选优流程",
            "没有商用 ATC/麒麟9000，Dynamic Affine Fusion 与 100% NPU 未验证",
            "0.2ms Fence同步、8ms P95、功耗与热稳态未验证",
        ],
        "removed_legacy_candidate_outputs": legacy_candidate_outputs,
    }
    summary_path = output_dir / "V6.1全流程摘要.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI RAW Denoise V6.1 完整开发流水线")
    parser.add_argument("--config", default="configs/train/v6_1_cpu_全流程.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = V6PipelineConfig.from_yaml(args.config)
    result = run_v6_pipeline(config)
    print(json.dumps({
        "status": result["status"],
        "selected_candidate_id": result["selected_candidate_id"],
        "output_dir": config.output_dir,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
