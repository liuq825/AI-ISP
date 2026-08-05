"""V3 命令兼容入口；新工程统一转交 V4 全阶段 Pipeline。"""

from __future__ import annotations

from ai_isp.pipeline import build_parser, run_pipeline, PipelineConfig


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_yaml(args.config)
    result = run_pipeline(config)
    print(f"{result['status']}: {config.output_dir}")


if __name__ == "__main__":
    main()
