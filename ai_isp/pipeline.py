"""V4 数据→训练→KD→剪枝→恢复→QAT→单ONNX/OM 的阶段运行器。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from ai_isp.data.ryyb_contract import validate_release_dataset_requirements
from ai_isp.data.ryyb_dataset import BalancedCameraSampler, RyybRawPatchDataset
from ai_isp.data.sidd_dataset import SiddRawPatchDataset
from ai_isp.export.freeze_topology import freeze_topology
from ai_isp.export.om_release import OmCompileConfig, build_v4_engineering_manifest, compile_single_om
from ai_isp.export.quant_microbenchmark import export_offset_microbenchmark_pair
from ai_isp.export.static_profiles import export_fixed_model
from ai_isp.models import ConditionalNAFNetW32Teacher, build_mobile_nafnet_w16
from ai_isp.pruning import StructuredMobileNAFPruner
from ai_isp.qat_training import QatTrainingConfig, train_qat
from ai_isp.quantization import QatPolicy, prepare_qdq_export
from ai_isp.training_stages import TrainingStageConfig, train_distillation_stage, train_supervised_stage


@dataclass(frozen=True)
class PipelineConfig:
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
    p10_recovery_steps: int = 1
    p15_recovery_steps: int = 1
    calibration_frames: int = 1
    q1_steps: int = 1
    q2_steps: int = 1
    q3_steps: int = 1
    device: str = "cpu"
    seed: int = 20260804
    export_mode: str = "smoke"
    soc_version: str = "Kirin9000_TARGET_DDK_REQUIRED"
    offset_microbenchmark_result: str | None = None
    run_lsqplus_smoke_candidate: bool = True
    student_activation_checkpointing: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**values)


def _loader(config: PipelineConfig):
    if config.ryyb_manifest:
        dataset = RyybRawPatchDataset(
            config.ryyb_manifest,
            split="train",
            patch_size=config.patch_size,
            samples_per_epoch=config.samples_per_epoch,
            seed=config.seed,
            deterministic=(config.export_mode == "smoke"),
        )
        if config.export_mode == "release":
            validate_release_dataset_requirements(dataset.records)
            if config.batch_size != 1 or config.calibration_frames < 4096 or config.calibration_frames % 2:
                raise ValueError("量产Q0要求Batch=1、校准帧不少于4096且main/tele等量")
        sampler = BalancedCameraSampler(dataset, config.seed)
        return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=0), False
    dataset = SiddRawPatchDataset(
        config.dataset_root,
        patch_size=config.patch_size,
        samples_per_epoch=config.samples_per_epoch,
        seed=config.seed,
        max_pairs=config.max_pairs,
        deterministic=True,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0), True


def _stage_config(config: PipelineConfig, name: str, steps: int, learning_rate: float) -> TrainingStageConfig:
    return TrainingStageConfig(
        stage_name=name,
        output_dir=str(Path(config.output_dir) / "training" / name),
        steps=steps,
        learning_rate=learning_rate,
        accumulation_steps=1 if config.device == "cpu" else 8,
        warmup_steps=0 if steps < 10 else min(5000, steps // 10),
        checkpoint_interval=0 if steps < 10 else 1000,
        device=config.device,
        seed=config.seed,
        student_activation_checkpointing=(config.student_activation_checkpointing and name != "teacher_fp32"),
    )


def _first_inputs(loader) -> tuple[torch.Tensor, torch.Tensor]:
    batch = next(iter(loader))
    return batch["noisy"].float(), batch["condition"].float()


def _calibration_batches(loader, frame_count: int):
    """循环数据加载器并流式生成 Q0 校准帧，禁止把量产校准集整体装入内存。"""

    if frame_count <= 0:
        raise ValueError("calibration_frames 必须大于 0")
    iterator = iter(loader)
    for _ in range(frame_count):
        batch, iterator = _next_calibration_batch(iterator, loader)
        yield batch["noisy"].float(), batch["condition"].float()


def _next_calibration_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """执行全部算法阶段；没有目标数据/设备时仍保持失败闭锁。"""

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader, smoke_only = _loader(config)
    image, condition = _first_inputs(loader)

    # 必须位于长周期 Teacher/KD/恢复/QAT 前，目标结果缺失时正式策略固定 Offset=0。
    student = build_mobile_nafnet_w16()
    offset_microbenchmark = export_offset_microbenchmark_pair(
        student,
        output_dir / "microbenchmark" / "offset",
        config.offset_microbenchmark_result,
    )

    teacher = ConditionalNAFNetW32Teacher()
    teacher_report = train_supervised_stage(
        teacher, loader, _stage_config(config, "teacher_fp32", config.teacher_steps, 2e-4)
    )
    student_report = train_supervised_stage(
        student, loader, _stage_config(config, "student_fp32", config.student_steps, 2e-4)
    )
    kd_report = train_distillation_stage(
        teacher, student, loader, _stage_config(config, "student_kd", config.kd_steps, 1e-4)
    )

    pruner = StructuredMobileNAFPruner()
    p10 = copy.deepcopy(student).cpu()
    p10_report = pruner.prune_to_ratio(p10, (image, condition), [(image, condition)], 0.10)
    p10_recovery = train_distillation_stage(
        teacher, p10, loader, _stage_config(config, "p10_recovery", config.p10_recovery_steps, 5e-5)
    )
    p10_topology = freeze_topology(p10, output_dir / "pruning" / "p10")

    p15 = copy.deepcopy(student).cpu()
    p15_report = pruner.prune_to_ratio(p15, (image, condition), [(image, condition)], 0.15)
    p15_recovery = train_distillation_stage(
        teacher, p15, loader, _stage_config(config, "p15_recovery", config.p15_recovery_steps, 5e-5)
    )
    freeze_topology(p15, output_dir / "pruning" / "p15")

    qat_models: dict[str, nn.Module] = {}
    qat_reports: dict[str, dict[str, object]] = {}
    policies = [("symmetric_lsq", False)]
    if offset_microbenchmark["allow_learnable_offset"] or config.run_lsqplus_smoke_candidate:
        policies.append(("asymmetric_lsqplus", True))
    for pruning_name, candidate in (("p10", p10), ("p15", p15)):
        for policy_name, activation_offset in policies:
            key = f"{pruning_name}_{policy_name}"
            qat_model, qat_report = train_qat(
                candidate,
                loader,
                _calibration_batches(loader, config.calibration_frames),
                QatPolicy(activation_offset=activation_offset),
                QatTrainingConfig(
                    output_dir=str(output_dir / "qat" / pruning_name / policy_name),
                    device=config.device,
                    q1_steps=config.q1_steps,
                    q2_steps=config.q2_steps,
                    q3_steps=config.q3_steps,
                ),
                teacher=teacher,
            )
            qat_models[key] = qat_model
            qat_reports[key] = qat_report

    # 工程冒烟固定导出低风险 P10+对称 LSQ；量产选择仍由 RYYB 画质和真机时延门禁决定。
    symmetric_model = qat_models["p10_symmetric_lsq"]
    # 缺少目标微基准时按保守规则选择 Offset=0；LSQ+ 仍完整训练和报告。
    quant_policy = {
        "selected": "symmetric_lsq",
        "reason": "目标 DDK/麒麟9000 Offset 微基准不可用，按失败闭锁选择 Offset=0",
        "candidates": {name: report["policy"] for name, report in qat_reports.items()},
        "offset_microbenchmark": offset_microbenchmark,
        "release_ready": False,
    }
    quant_policy_path = output_dir / "release" / "quant_policy.json"
    quant_policy_path.parent.mkdir(parents=True, exist_ok=True)
    quant_policy_path.write_text(json.dumps(quant_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepare_qdq_export(symmetric_model)
    onnx_report = export_fixed_model(symmetric_model, output_dir / "release" / "onnx", config.export_mode)
    onnx_path = output_dir / "release" / "onnx" / "dark_preview_ryyb_4x3.onnx"
    om_report = compile_single_om(OmCompileConfig(
        onnx_path=str(onnx_path), output_dir=str(output_dir / "release" / "om"), soc_version=config.soc_version
    ))
    manifest = build_v4_engineering_manifest(
        output_dir / "release" / "model_manifest_v4.json",
        qat_reports["p10_symmetric_lsq"]["qat_weights"],
        p10_topology["topology"],
        onnx_path,
        quant_policy_path,
        om_report.get("om_path") if om_report.get("available") else None,
    )
    summary = {
        "status": "engineering_passed",
        "config": asdict(config),
        "smoke_only": smoke_only,
        "release_ready": False,
        "stages": {
            "teacher": teacher_report,
            "student": student_report,
            "kd": kd_report,
            "p10": asdict(p10_report),
            "p10_recovery": p10_recovery,
            "p15": asdict(p15_report),
            "p15_recovery": p15_recovery,
            "offset_microbenchmark": offset_microbenchmark,
            "qat_candidates": qat_reports,
            "onnx": onnx_report,
            "om": om_report,
        },
        "manifest": manifest,
        "limitations": [
            "SIDD 只证明代码链路可运行，不代表 RYYB 画质",
            "P10/P15 最终选择必须使用真实 RYYB 盲测和麒麟9000时延/内存证据",
            "4K30、6/8/9/10ms、NPU覆盖、功耗和热稳态尚未实测",
        ],
    }
    (output_dir / "V4全流程摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI RAW Denoise V4 全阶段训练压缩流水线")
    parser.add_argument("--config", default="configs/train/v4_cpu_全流程.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_yaml(args.config)
    result = run_pipeline(config)
    print(json.dumps({"status": result["status"], "output_dir": config.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
