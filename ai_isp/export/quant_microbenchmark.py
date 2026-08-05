"""在长周期训练前生成两层 MobileNAFBlock 的 Offset 微基准图。"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import warnings

import numpy as np
import onnxruntime
import torch
from torch import nn

from ai_isp.models.mobile_nafnet import MobileNAFNetW16
from ai_isp.quantization.lsqplus_qat import (
    OffsetMicrobenchmarkResult,
    QatPolicy,
    iter_quantizers,
    prepare_qat_model,
    prepare_qdq_export,
)
from .static_profiles import inspect_onnx


class TwoBlockMicrobenchmark(nn.Module):
    """只保留两个连续 MobileNAFBlock，用于比较对称/非对称激活量化。"""

    def __init__(self, model: MobileNAFNetW16) -> None:
        super().__init__()
        if len(model.encoders[1]) < 2:
            raise ValueError("Dense W16 Encoder Stage2 至少需要两个连续 Block")
        self.blocks = nn.Sequential(copy.deepcopy(model.encoders[1][0]), copy.deepcopy(model.encoders[1][1]))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.blocks(feature)


def _calibrate_feature_model(model: nn.Module, feature: torch.Tensor) -> None:
    quantizers = list(iter_quantizers(model))
    for _, quantizer in quantizers:
        if quantizer.channel_axis < 0:
            quantizer.enable_observer(True)
    model.eval()
    with torch.no_grad():
        model(feature)
    for _, quantizer in quantizers:
        if quantizer.channel_axis < 0:
            quantizer.finalize_calibration()
    prepare_qdq_export(model)


def _export_candidate(
    source: TwoBlockMicrobenchmark,
    activation_offset: bool,
    feature: torch.Tensor,
    path: Path,
) -> dict[str, object]:
    model = prepare_qat_model(source, QatPolicy(activation_offset=activation_offset)).eval().cpu()
    _calibrate_feature_model(model, feature)
    with torch.no_grad():
        expected = model(feature).numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch(\.|$)")
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="You are using the legacy.*")
        torch.onnx.export(
            model,
            (feature,),
            str(path),
            input_names=("feature",),
            output_names=("output",),
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    session = onnxruntime.InferenceSession(str(path), providers=("CPUExecutionProvider",))
    actual = session.run(None, {"feature": feature.numpy()})[0]
    audit = inspect_onnx(path)
    if audit["unsupported_operators"] or audit["forbidden_gate_operators"] or audit["dynamic_slice_inputs"]:
        raise ValueError(f"Offset 微基准 ONNX 审计失败: {audit}")
    max_abs_error = float(np.max(np.abs(expected - actual)))
    if max_abs_error > 1e-4:
        raise ValueError(f"Offset 微基准 PyTorch/ONNX 误差超限: {max_abs_error}")
    return {
        **audit,
        "activation_offset": activation_offset,
        "max_abs_error": max_abs_error,
        "contains_qdq": "QuantizeLinear" in audit["operators"] and "DequantizeLinear" in audit["operators"],
    }


def load_target_microbenchmark_result(path: str | Path) -> OffsetMicrobenchmarkResult:
    """读取目标机结果；字段缺失或伪造的纯主机结果不得影响量产策略。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "compile_succeeded",
        "npu_only",
        "fusion_preserved",
        "symmetric_p95_ms",
        "asymmetric_p95_ms",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Offset 微基准结果缺少字段: {missing}")
    return OffsetMicrobenchmarkResult(**{name: payload[name] for name in required})


def export_offset_microbenchmark_pair(
    dense_model: MobileNAFNetW16,
    output_dir: str | Path,
    target_result_path: str | Path | None = None,
) -> dict[str, object]:
    """生成对称/非对称 Q/DQ 图，并在目标结果缺失时失败闭锁为 Offset=0。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260804)
    feature = torch.rand(1, 32, 16, 16)
    source = TwoBlockMicrobenchmark(dense_model).eval()
    candidates = {
        "symmetric": _export_candidate(source, False, feature, output_dir / "two_blocks_symmetric_qdq.onnx"),
        "asymmetric": _export_candidate(source, True, feature, output_dir / "two_blocks_asymmetric_qdq.onnx"),
    }
    target_result = load_target_microbenchmark_result(target_result_path) if target_result_path else None
    allow_offset = bool(target_result and target_result.allow_learnable_offset())
    report = {
        "executed_before_long_training": True,
        "feature_shape": list(feature.shape),
        "candidates": candidates,
        "target_result_available": target_result is not None,
        "target_result": asdict(target_result) if target_result else None,
        "allow_learnable_offset": allow_offset,
        "formal_qat_activation_offset": allow_offset,
        "decision_reason": (
            "目标商用DDK编译、100% NPU、融合和P95门禁通过"
            if allow_offset
            else "目标结果缺失或门禁失败，正式QAT固定Offset=0"
        ),
    }
    (output_dir / "Offset前置微基准报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
