"""P0/P1/P2 静态 ONNX 导出、算子审计和 ONNX Runtime 对齐。"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from ai_isp.runtime.profiles import PROFILES


ALLOWED_ONNX_OPS = {
    "Add", "Cast", "Clip", "Concat", "Constant", "Conv", "Expand", "Gather",
    "Gemm", "Identity", "MatMul", "Mul", "Relu", "Reshape", "Resize", "Shape",
    "Slice", "Tanh", "Transpose", "Unsqueeze",
}

SMOKE_SHAPES = {
    "P0": (32, 48),
    "P1": (32, 48),
    "P2": (40, 48),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def inspect_onnx(path: str | Path) -> dict[str, object]:
    """审计静态 Shape、Slice 来源和算子白名单。"""

    path = Path(path)
    graph = onnx.load(path)
    operators = sorted({node.op_type for node in graph.graph.node})
    unsupported = sorted(set(operators) - ALLOWED_ONNX_OPS)
    producer_by_output = {output: node.op_type for node in graph.graph.node for output in node.output}
    initializers = {item.name for item in graph.graph.initializer}
    dynamic_slice_inputs: list[str] = []
    for node in graph.graph.node:
        if node.op_type != "Slice":
            continue
        for name in node.input[1:]:
            if name and name not in initializers and producer_by_output.get(name) != "Constant":
                dynamic_slice_inputs.append(name)
    input_shapes = {
        item.name: [dimension.dim_value for dimension in item.type.tensor_type.shape.dim]
        for item in graph.graph.input
    }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "operators": operators,
        "unsupported_operators": unsupported,
        "dynamic_slice_inputs": sorted(set(dynamic_slice_inputs)),
        "input_shapes": input_shapes,
    }


def _export_one(model: torch.nn.Module, path: Path, height: int, width: int) -> dict[str, object]:
    torch.manual_seed(20260804)
    image = torch.rand(1, 4, height, width, dtype=torch.float32)
    condition = torch.rand(1, 24, dtype=torch.float32)
    condition[:, 10:14] = torch.tensor((1.0, 0.0, 0.0, 0.0))
    condition[:, 14:18] = torch.tensor((1.0, 0.0, 0.0, 0.0))
    condition[:, 18:22] = torch.tensor((1.0, 0.0, 0.0, 0.0))
    condition[:, 23] = 1.0
    with torch.no_grad():
        torch_output = model(image, condition).cpu().numpy()
    # 当前冻结 legacy exporter 是为了确保 Narrow→Slice 的常量切点稳定；
    # PyTorch 的弃用提示已在设计文档登记，迁移必须单独做图回归。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch(\.|$)")
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="You are using the legacy.*")
        torch.onnx.export(
            model,
            (image, condition),
            str(path),
            input_names=("packed_raw", "condition"),
            output_names=("noise_pred",),
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    session = onnxruntime.InferenceSession(str(path), providers=("CPUExecutionProvider",))
    ort_output = session.run(None, {"packed_raw": image.numpy(), "condition": condition.numpy()})[0]
    alignment = {
        "max_abs_error": float(np.max(np.abs(torch_output - ort_output))),
        "mean_abs_error": float(np.mean(np.abs(torch_output - ort_output))),
    }
    audit = inspect_onnx(path)
    if audit["unsupported_operators"]:
        raise ValueError(f"ONNX 出现非白名单算子: {audit['unsupported_operators']}")
    if audit["dynamic_slice_inputs"]:
        raise ValueError(f"ONNX Slice 切点不是常量: {audit['dynamic_slice_inputs']}")
    if alignment["max_abs_error"] > 1e-4:
        raise ValueError(f"PyTorch→ONNX 最大误差超限: {alignment['max_abs_error']}")
    return {**audit, "alignment": alignment}


def export_static_profiles(
    model: torch.nn.Module,
    output_dir: str | Path,
    profile_mode: str = "smoke",
) -> dict[str, dict[str, object]]:
    """导出三份静态 ONNX；CPU 默认用等拓扑小 Shape 验链。"""

    if profile_mode not in ("smoke", "release"):
        raise ValueError("profile_mode 必须为 smoke 或 release")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    report: dict[str, dict[str, object]] = {}
    for profile_id, profile in PROFILES.items():
        height, width = SMOKE_SHAPES[profile_id] if profile_mode == "smoke" else (profile.compile_height, profile.compile_width)
        path = output_dir / f"dark_preview_{profile_id.lower()}.onnx"
        report[profile_id] = {
            "profile_mode": profile_mode,
            "compile_height": height,
            "compile_width": width,
            "release_compile_height": profile.compile_height,
            "release_compile_width": profile.compile_width,
            **_export_one(model, path, height, width),
        }
    (output_dir / "onnx_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
