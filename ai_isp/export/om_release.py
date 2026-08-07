"""单一 RYYB OM 编译适配、失败闭锁与 V4/V6.1 发布清单。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class OmCompileConfig:
    onnx_path: str
    output_dir: str
    soc_version: str
    atc_executable: str = "atc"
    framework: int = 5
    precision_mode: str = "must_keep_origin_dtype"
    profile_mode: str = "release"


def compile_single_om(config: OmCompileConfig) -> dict[str, object]:
    """只允许一个固定 Shape ONNX；缺少商用工具链时返回明确的未就绪报告。"""

    onnx_path = Path(config.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    executable = shutil.which(config.atc_executable)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "OM编译报告.json"
    prefix = output_dir / "dark_preview_ryyb_4x3_mixed_int8_fp16"
    om_path = prefix.with_suffix(".om")
    if config.profile_mode != "release":
        report = {
            "available": False,
            "release_ready": False,
            "reason": "smoke Shape 不允许进入 ATC/OM 发布编译",
            "om_path": str(om_path),
            "om_sha256": "",
            "config": asdict(config),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    if executable is None:
        report = {
            "available": False,
            "release_ready": False,
            "reason": "本机没有目标商用 DDK/ATC；禁止用 CPU 或伪 OM 代替",
            "om_path": str(om_path),
            "om_sha256": "",
            "config": asdict(config),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    command = [
        executable,
        f"--model={onnx_path}",
        f"--framework={config.framework}",
        f"--output={prefix}",
        f"--soc_version={config.soc_version}",
        "--input_shape=packed_raw:1,4,768,1024;condition:1,24",
        f"--precision_mode={config.precision_mode}",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
    report = {
        "available": completed.returncode == 0 and om_path.exists(),
        "release_ready": False,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "om_path": str(om_path),
        "om_sha256": sha256_file(om_path) if om_path.exists() else "",
        "config": asdict(config),
        "mandatory_next_gate": "麒麟9000 100% NPU、6/8/9/10ms、4K30整链和热稳态",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_v4_engineering_manifest(
    path: str | Path,
    weights_path: str | Path,
    topology_path: str | Path,
    onnx_path: str | Path,
    quant_policy_path: str | Path,
    om_path: str | Path | None = None,
    condition_schema_path: str | Path = "configs/release/condition_schema_v2.json",
    sensor_profiles_path: str | Path = "configs/release/sensor_profiles_ryyb.json",
) -> dict[str, object]:
    """生成单 ONNX/单 OM 清单；工程环境永远不会自行宣称 release_ready。"""

    weights_path, topology_path, onnx_path, quant_policy_path, condition_schema_path, sensor_profiles_path = map(
        Path,
        (weights_path, topology_path, onnx_path, quant_policy_path, condition_schema_path, sensor_profiles_path),
    )
    for required in (
        weights_path, topology_path, onnx_path, quant_policy_path, condition_schema_path, sensor_profiles_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    om = Path(om_path) if om_path else None
    manifest = {
        "manifest_version": "4.0.0-engineering",
        "release_ready": False,
        "scope": "RYYB main+tele single static model",
        "model_name": "Conditional MobileNAFNet Dark Preview V4",
        "input": {"packed_raw": [1, 4, 768, 1024], "condition": [1, 24], "channels": ["R", "Yr", "Yb", "B"]},
        "allowed_cameras": ["main", "tele"],
        "bypass_cameras": ["ultrawide"],
        "artifact_layout": "single_static_ryyb_4x3",
        "weights": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
        "topology": {"path": str(topology_path), "sha256": sha256_file(topology_path)},
        "onnx": {"path": str(onnx_path), "sha256": sha256_file(onnx_path)},
        "quant_policy": {"path": str(quant_policy_path), "sha256": sha256_file(quant_policy_path)},
        "condition_schema": {"path": str(condition_schema_path), "sha256": sha256_file(condition_schema_path)},
        "sensor_profiles": {"path": str(sensor_profiles_path), "sha256": sha256_file(sensor_profiles_path)},
        "om": {
            "available": bool(om and om.exists()),
            "path": str(om) if om else "artifacts/release/dark_preview_ryyb_4x3_int8.om",
            "sha256": sha256_file(om) if om and om.exists() else "",
        },
        "latency_budget_ms": {"p50": 6.0, "p95": 8.0, "p99": 9.0, "hard_timeout": 10.0},
        "mandatory_target_validation": [
            "真实主摄/长焦RYYB完整训练、盲测及分桶画质",
            "目标DDK Q/DQ适配、ATC和逐层Quant Policy审计",
            "麒麟9000 100% NPU且CPU/GPU回退为0",
            "AI节点6/8/9/10ms和整条4K30 Pipeline",
            "10000帧、10/30分钟热稳态、功耗、内存及Camera切换",
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


V6_1_REQUIRED_ARTIFACTS = {
    "weights",
    "topology",
    "condition_schema",
    "sensor_profiles",
    "unpack_profiles",
    "raw_domain_profile",
    "lsc_profiles",
    "reference_isp_profile",
    "noise_profiles",
    "buffer_contract",
    "quant_policy",
    "onnx",
    "development_selection",
    "selection_report",
}


def build_v6_1_engineering_manifest(
    path: str | Path,
    artifacts: dict[str, str | Path],
    selected_candidate_id: str,
    om_path: str | Path | None = None,
    profile_mode: str = "release",
    compile_height: int = 768,
    compile_width: int = 1024,
) -> dict[str, object]:
    """生成 V6.1 唯一混合精度工程清单，缺真机证据时强制失败闭锁。"""

    if selected_candidate_id not in ("P10-16", "P18-16", "P36-16"):
        raise ValueError("V6.1 selected_candidate_id 非法")
    if profile_mode not in ("smoke", "release"):
        raise ValueError("profile_mode 只允许 smoke/release")
    if compile_height <= 0 or compile_width <= 0:
        raise ValueError("compile Shape 必须为正数")
    missing = sorted(V6_1_REQUIRED_ARTIFACTS - set(artifacts))
    extra = sorted(set(artifacts) - V6_1_REQUIRED_ARTIFACTS)
    if missing or extra:
        raise ValueError(f"V6.1 制品键不完整: missing={missing}, extra={extra}")
    resolved = {name: Path(value) for name, value in artifacts.items()}
    for name, artifact in resolved.items():
        if not artifact.is_file():
            raise FileNotFoundError(f"V6.1 制品不存在 {name}: {artifact}")
    if resolved["weights"].name != "model_mixed_qat.safetensors":
        raise ValueError("V6.1 最终权重必须命名为 model_mixed_qat.safetensors")
    manifest_path = Path(path)
    om = Path(om_path) if om_path else None

    def portable_path(artifact: Path) -> str:
        try:
            return artifact.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
        except ValueError:
            return str(artifact)
    manifest = {
        "manifest_version": "6.1.0-engineering",
        "selected_candidate_id": selected_candidate_id,
        "development_selected": True,
        "dynamic_affine_target_pending": True,
        "target_validated": False,
        "release_ready": False,
        "smoke_only": profile_mode == "smoke",
        "profile_mode": profile_mode,
        "scope": "RYYB main+tele single static mixed-precision model",
        "model_name": "Conditional MobileNAFNet Dark Preview V6.1",
        "input": {
            "packed_raw": {"shape": [1, 4, compile_height, compile_width], "dtype": "float16"},
            "condition": {"shape": [1, 24], "dtype": "float32", "cast_count": 1},
            "channels": ["R", "Yr", "Yb", "B"],
        },
        "output": {"noise_pred": {"shape": [1, 4, compile_height, compile_width], "dtype": "float16"}},
        "release_shape_target": [1, 4, 768, 1024],
        "precision_policy": "fixed_v6_mixed_lsqplus",
        "artifact_layout": (
            "single_static_ryyb_4x3_v6_1_smoke"
            if profile_mode == "smoke"
            else "single_static_ryyb_4x3_v6_1"
        ),
        "artifacts": {
            name: {"path": portable_path(artifact), "sha256": sha256_file(artifact)}
            for name, artifact in sorted(resolved.items())
        },
        "unpack_profile_hash": sha256_file(resolved["unpack_profiles"]),
        "lsc_profile_hash": sha256_file(resolved["lsc_profiles"]),
        "buffer_contract_version": "v1",
        "om": {
            "available": bool(om and om.is_file()),
            "path": portable_path(om) if om else "om/dark_preview_ryyb_4x3_mixed_int8_fp16.om",
            "sha256": sha256_file(om) if om and om.is_file() else "",
        },
        "latency_budget_ms": {"p50": 6.0, "p95": 8.0, "p99": 9.0, "hard_timeout": 10.0},
        "buffer_contract": {
            "extra_cpu_memcpy_bytes_per_frame": 0,
            "fd_import": "stream_init_once",
            "per_frame_map_unmap": False,
            "fence_sync_optimization_target_ms": 0.2,
        },
        "mandatory_target_validation": [
            "真实主摄/长焦RYYB完整训练、盲测及分桶画质",
            "目标DDK Dynamic Affine无FP16回退且dynamic_affine_target_pending=false",
            "麒麟9000 100% NPU且CPU/GPU回退为0",
            "AI节点6/8/9/10ms和整条Photo Preview 30fps",
            "DMA-BUF/Fence/Cache/0-byte CPU memcpy",
            "10000帧、10/30分钟热稳态、功耗、内存及Camera切换",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def promote_v6_1_manifest(manifest: dict[str, object], target_evidence: dict[str, bool]) -> dict[str, object]:
    """只有全部目标端证据为真时才允许生成 release_ready 清单副本。"""

    required = {
        "real_ryyb_quality",
        "dynamic_affine_no_fp16_fallback",
        "npu_coverage_100_percent",
        "latency_6_8_9_10ms",
        "photo_preview_30fps",
        "dmabuf_contract",
        "power_thermal_stability",
        "ten_thousand_frames",
        "rollback_verified",
    }
    if set(target_evidence) != required:
        raise ValueError("目标端证据键不完整")
    failed = sorted(name for name, passed in target_evidence.items() if not passed)
    if failed:
        raise RuntimeError(f"目标端发布门禁未通过: {failed}")
    promoted = json.loads(json.dumps(manifest))
    promoted["dynamic_affine_target_pending"] = False
    promoted["target_validated"] = True
    promoted["release_ready"] = True
    promoted["target_evidence"] = target_evidence
    return promoted
