"""运行 P0/P1/P2 发布分辨率的 PyTorch CPU 基准。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_isp.benchmark import benchmark_release_profiles  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/cpu_smoke/pruning/p0/model_fp32.safetensors")
    parser.add_argument("--output", default="artifacts/reports/CPU发布分辨率性能.json")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    report = benchmark_release_profiles(args.weights, args.output, iterations=args.iterations)
    print(json.dumps(report["profiles"], ensure_ascii=False))


if __name__ == "__main__":
    main()

