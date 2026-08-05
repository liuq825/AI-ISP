"""单一 RYYB OM 编译适配、失败闭锁与 V4 发布清单。"""

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


def compile_single_om(config: OmCompileConfig) -> dict[str, object]:
    """只允许一个固定 Shape ONNX；缺少商用工具链时返回明确的未就绪报告。"""

    onnx_path = Path(config.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    executable = shutil.which(config.atc_executable)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "OM编译报告.json"
    if executable is None:
        report = {
            "available": False,
            "release_ready": False,
            "reason": "本机没有目标商用 DDK/ATC；禁止用 CPU 或伪 OM 代替",
            "config": asdict(config),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    prefix = output_dir / "dark_preview_ryyb_4x3_int8"
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
    om_path = prefix.with_suffix(".om")
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
