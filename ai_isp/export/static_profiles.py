"""V4 单一 RYYB 4:3 静态 ONNX 导出、图审计与数值对齐。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn

from ai_isp.models.static_simple_gate import StaticSimpleGate
from ai_isp.runtime.profiles import FIXED_RYYB_PROFILE


ALLOWED_ONNX_OPS = {
    "Add", "Cast", "Clip", "Concat", "Constant", "Conv", "DequantizeLinear",
    "Expand", "Gather", "Gemm", "Identity", "MatMul", "Mul", "QuantizeLinear",
    "Relu", "Reshape", "Resize", "Shape", "Slice", "Tanh", "Transpose", "Unsqueeze",
}
FORBIDDEN_DYNAMIC_GATE_OPS = {"Split", "SplitToSequence"}
SMOKE_SHAPE = (32, 48)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_legacy_simple_gates(
    model: nn.Module,
    channel_map: dict[str, int] | None = None,
) -> tuple[nn.Module, list[dict[str, object]]]:
    """只在模型副本上显式替换遗留 SimpleGate，绝不修改类或全局 forward。"""

    deploy = copy.deepcopy(model)
    replacements: list[dict[str, object]] = []
    channel_map = channel_map or {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if child.__class__.__name__ == "SimpleGate":
                channels = channel_map.get(path)
                if channels is None:
                    for attribute in ("channels", "channel", "dim"):
                        value = getattr(child, attribute, None)
                        if isinstance(value, int):
                            channels = value
                            break
                if channels is None or channels <= 0:
                    raise ValueError(f"遗留 SimpleGate {path} 无法推断静态半通道；请提供 channel_map")
                setattr(parent, name, StaticSimpleGate(channels))
                replacements.append({"name": path, "channels": channels})
            else:
                visit(child, path)

    visit(deploy)
    return deploy, replacements


def prepare_export_model(
    model: nn.Module,
    allow_legacy_replacement: bool = False,
    legacy_gate_channels: dict[str, int] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    """在副本上检查静态 Gate；当前发布模型禁止全局 Monkey Patch。"""

    if allow_legacy_replacement:
        deploy, replacements = replace_legacy_simple_gates(model, legacy_gate_channels)
    else:
        deploy, replacements = copy.deepcopy(model), []
    deploy = deploy.eval().cpu()
    gates = []
    for name, module in deploy.named_modules():
        if isinstance(module, StaticSimpleGate):
            if module.channels <= 0 or module.channels % 8:
                raise ValueError(f"StaticSimpleGate {name} 通道未按 8 对齐")
            gates.append({"name": name, "channels": module.channels})
        elif module.__class__.__name__ == "SimpleGate":
            raise ValueError(f"检测到遗留动态 SimpleGate {name}；必须在模型副本上受控转换")
    if not gates:
        raise ValueError("部署模型没有 StaticSimpleGate，疑似导出了错误拓扑")
    return deploy, {"global_monkey_patch": False, "static_gates": gates, "legacy_replacements": replacements}


def inspect_onnx(path: str | Path) -> dict[str, object]:
    """审计固定 Shape、动态 Slice、Gate 风险算子和部署白名单。"""

    path = Path(path)
    graph = onnx.load(path)
    operators = sorted({node.op_type for node in graph.graph.node})
    unsupported = sorted(set(operators) - ALLOWED_ONNX_OPS)
    forbidden = sorted(set(operators) & FORBIDDEN_DYNAMIC_GATE_OPS)
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
        "forbidden_gate_operators": forbidden,
        "dynamic_slice_inputs": sorted(set(dynamic_slice_inputs)),
        "input_shapes": input_shapes,
    }


def _valid_main_condition() -> torch.Tensor:
    condition = torch.zeros(1, 24, dtype=torch.float32)
    condition[:, :10] = 0.5
    condition[:, 10] = 1.0
    condition[:, 14] = 1.0
    condition[:, 18] = 1.0
    condition[:, 22] = 1.0
    condition[:, 23] = 1.0
    return condition


def _export_one(model: nn.Module, path: Path, height: int, width: int) -> dict[str, object]:
    torch.manual_seed(20260804)
    image = torch.rand(1, 4, height, width, dtype=torch.float32)
    condition = _valid_main_condition()
    deploy, gate_report = prepare_export_model(model)
    with torch.no_grad():
        torch_output = deploy(image, condition).cpu().numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch(\.|$)")
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="You are using the legacy.*")
        torch.onnx.export(
            deploy,
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
    if audit["forbidden_gate_operators"] or audit["dynamic_slice_inputs"]:
        raise ValueError("ONNX 含动态 Gate/Slice，禁止进入 ATC")
    if alignment["max_abs_error"] > 1e-4:
        raise ValueError(f"PyTorch→ONNX 最大误差超限: {alignment['max_abs_error']}")
    return {**audit, "alignment": alignment, "gate_export": gate_report}


def export_fixed_model(
    model: nn.Module,
    output_dir: str | Path,
    profile_mode: str = "smoke",
) -> dict[str, dict[str, object]]:
    """只导出一个 RYYB_4X3 ONNX；release 模式使用固定发布 Shape。"""

    if profile_mode not in ("smoke", "release"):
        raise ValueError("profile_mode 必须为 smoke 或 release")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = FIXED_RYYB_PROFILE
    height, width = SMOKE_SHAPE if profile_mode == "smoke" else (profile.compile_height, profile.compile_width)
    path = output_dir / "dark_preview_ryyb_4x3.onnx"
    report = {
        profile.profile_id: {
            "profile_mode": profile_mode,
            "compile_height": height,
            "compile_width": width,
            "release_compile_height": profile.compile_height,
            "release_compile_width": profile.compile_width,
            "raw_height": profile.raw_height,
            "raw_width": profile.raw_width,
            **_export_one(model, path, height, width),
        }
    }
    (output_dir / "onnx_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def export_static_profiles(model: nn.Module, output_dir: str | Path, profile_mode: str = "smoke"):
    """V3 名称兼容层；行为已经冻结为 V4 单 Shape。"""

    return export_fixed_model(model, output_dir, profile_mode)
