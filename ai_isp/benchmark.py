"""发布分辨率的 PyTorch CPU 能力边界基准。"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from ai_isp.models.mobile_nafnet import build_mobile_nafnet_w16
from ai_isp.runtime.profiles import PROFILES


def benchmark_release_profiles(
    weights_path: str | Path,
    output_path: str | Path,
    warmup: int = 1,
    iterations: int = 3,
    torch_threads: int = 4,
) -> dict[str, object]:
    """测量 CPU 推理耗时；结果只用于证明 CPU 不是 30fps 回退路径。"""

    torch.set_num_threads(torch_threads)
    model = build_mobile_nafnet_w16().eval()
    model.load_state_dict(load_file(str(weights_path)))
    condition = torch.rand(1, 24)
    condition[:, 23] = 1.0
    results: dict[str, object] = {}
    for profile_id, profile in PROFILES.items():
        image = torch.rand(1, 4, profile.compile_height, profile.compile_width)
        with torch.inference_mode():
            for _ in range(warmup):
                model(image, condition)
            samples_ms: list[float] = []
            output = None
            for _ in range(iterations):
                started = time.perf_counter()
                output = model(image, condition)
                samples_ms.append((time.perf_counter() - started) * 1000.0)
        ordered = sorted(samples_ms)
        results[profile_id] = {
            "shape": [1, 4, profile.compile_height, profile.compile_width],
            "samples_ms": samples_ms,
            "mean_ms": statistics.mean(samples_ms),
            "p50_ms": statistics.median(samples_ms),
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "finite": bool(output is not None and torch.isfinite(output).all()),
        }
    report = {
        "scope": "PyTorch CPU 能力边界，不代表麒麟 9000 NPU 性能",
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_threads": torch_threads,
        "warmup": warmup,
        "iterations": iterations,
        "profiles": results,
        "cpu_fallback_allowed": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

