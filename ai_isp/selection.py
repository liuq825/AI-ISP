"""V6.1 三候选模拟门禁、确定性选优与落选可执行制品清理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    raw_psnr_drop_db: float
    ssim_drop: float
    delta_e00_increase: float
    worst_camera_iso_drop_db: float
    macs: int
    activation_peak_bytes: int
    cast_count: int
    qdq_count: int
    ort_p95_ms: float
    parameter_count: int
    onnx_size_bytes: int
    max_saturation_rate: float
    numerical_valid: bool = True
    graph_valid: bool = True
    mixed_precision_valid: bool = True

    def hard_gate_failures(self) -> list[str]:
        failures = []
        if self.raw_psnr_drop_db > 0.10:
            failures.append("RAW PSNR 阶段回归超过 0.10dB")
        if self.ssim_drop > 0.002:
            failures.append("SSIM 阶段回归超过 0.002")
        if self.worst_camera_iso_drop_db > 0.50:
            failures.append("Camera/ISO 最差分桶回归超过 0.50dB")
        if self.max_saturation_rate >= 0.001:
            failures.append("关键层或分桶饱和率达到 0.1%")
        if not self.numerical_valid:
            failures.append("数值一致性失败")
        if not self.graph_valid:
            failures.append("静态图审计失败")
        if not self.mixed_precision_valid:
            failures.append("固定混合精度策略失败")
        return failures


def _quality_equivalent(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    return (
        abs(left.raw_psnr_drop_db - right.raw_psnr_drop_db) <= 0.03
        and abs(left.ssim_drop - right.ssim_drop) <= 0.0005
        and abs(left.delta_e00_increase - right.delta_e00_increase) <= 0.1
        and abs(left.worst_camera_iso_drop_db - right.worst_camera_iso_drop_db) <= 0.03
    )


def select_best_candidate(candidates: list[CandidateMetrics]) -> tuple[CandidateMetrics, dict[str, object]]:
    """先过硬门槛，再按画质等价→MAC→Activation→Cast/QDQ→ORT→体积选唯一方案。"""

    if {item.candidate_id for item in candidates} != {"P10-16", "P18-16", "P36-16"}:
        raise ValueError("V6.1 必须且只能比较 P10-16/P18-16/P36-16")
    failures = {item.candidate_id: item.hard_gate_failures() for item in candidates}
    passing = [item for item in candidates if not failures[item.candidate_id]]
    if not passing:
        raise RuntimeError("三个 16 通道候选均未通过模拟硬门槛")
    quality_best = min(
        passing,
        key=lambda item: (
            item.worst_camera_iso_drop_db,
            item.raw_psnr_drop_db,
            item.ssim_drop,
            item.delta_e00_increase,
        ),
    )
    equivalent = [item for item in passing if _quality_equivalent(item, quality_best)]
    selected = min(
        equivalent,
        key=lambda item: (
            item.macs,
            item.activation_peak_bytes,
            item.cast_count + item.qdq_count,
            item.ort_p95_ms,
            item.parameter_count,
            item.onnx_size_bytes,
            item.candidate_id,
        ),
    )
    report = {
        "selected_candidate_id": selected.candidate_id,
        "development_selected": True,
        "precision_policy": "fixed_v6_mixed_lsqplus",
        "dynamic_affine_target_pending": True,
        "target_validated": False,
        "release_ready": False,
        "selection_order": [
            "hard_gates",
            "worst_bucket_quality",
            "quality_equivalence",
            "macs",
            "activation_peak",
            "cast_qdq_count",
            "ort_p95_proxy",
            "parameters_onnx_size",
        ],
        "candidates": [
            {**asdict(item), "hard_gate_failures": failures[item.candidate_id]}
            for item in sorted(candidates, key=lambda value: value.candidate_id)
        ],
    }
    return selected, report


def converge_candidate_artifacts(
    candidate_root: str | Path,
    selected_candidate_id: str,
    report: dict[str, object],
) -> dict[str, object]:
    """删除落选候选可执行目录，仅在根目录保留不可执行指标摘要。"""

    root = Path(candidate_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_ids = {"P10-16", "P18-16", "P36-16"}
    if selected_candidate_id not in candidate_ids:
        raise ValueError("selected_candidate_id 非法")
    removed = []
    for candidate_id in sorted(candidate_ids - {selected_candidate_id}):
        target = (root / candidate_id).resolve()
        if target.parent != root:
            raise RuntimeError("候选清理目标越出临时评测根目录")
        if target.exists():
            shutil.rmtree(target)
            removed.append(candidate_id)
    summary = {
        **report,
        "removed_executable_candidates": removed,
        "loser_retention_policy": "metrics_hash_reason_only_non_executable",
    }
    (root / "development_selection_v6_1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
