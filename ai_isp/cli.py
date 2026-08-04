"""AI ISP CPU 全流程命令入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_isp.export.freeze_topology import freeze_topology
from ai_isp.export.static_profiles import export_static_profiles
from ai_isp.pruning.nafnet_pruning_validator import NAFNetPruningValidator
from ai_isp.quantization.ptq_validate import validate_ptq_tensor
from ai_isp.training import CpuTrainingConfig, train_student_cpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用少量 SIDD RAW 数据跑通 CPU 训练、冻结、PTQ 与 ONNX 全流程")
    parser.add_argument("--dataset-root", default="datasets/SIDD_Training_Subset")
    parser.add_argument("--output-dir", default="artifacts/cpu_smoke")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--profile-mode", choices=("smoke", "release"), default="smoke")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    model, training_report = train_student_cpu(CpuTrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=str(output_dir / "reports"),
        patch_size=args.patch_size,
        steps=args.steps,
        samples_per_epoch=max(args.steps, 4),
        max_pairs=args.max_pairs,
    ))
    import torch
    example_image = torch.rand(1, 4, args.patch_size, args.patch_size)
    example_condition = torch.rand(1, 24)
    NAFNetPruningValidator().assert_valid(model, example_image, example_condition)
    topology = freeze_topology(model, output_dir / "pruning" / "p0")
    ptq = validate_ptq_tensor(example_image)
    onnx_report = export_static_profiles(model, output_dir / "onnx", args.profile_mode)
    summary = {
        "status": "passed",
        "training": training_report,
        "topology": topology,
        "ptq_smoke": ptq,
        "onnx": onnx_report,
        "limitations": [
            "CPU 小样本结果仅证明工程链路可运行，不代表完整训练收敛或量产画质",
            "OM 转换、NPU 100% 落点、30fps、功耗、热稳态必须在麒麟 9000 与目标 DDK 实测",
            "smoke 模式导出等拓扑小 Shape；发布 Shape 使用 --profile-mode release 另行导出",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "全流程摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output_dir": args.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()

