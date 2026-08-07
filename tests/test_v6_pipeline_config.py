from dataclasses import asdict
from pathlib import Path

import pytest

from ai_isp.pipeline_v6 import V6PipelineConfig, V6_CANDIDATE_TOPOLOGIES


def test_v6_smoke_config_loads_complete_flow() -> None:
    config = V6PipelineConfig.from_yaml(Path("configs/train/v6_1_cpu_全流程.yaml"))
    assert config.export_mode == "smoke"
    assert config.calibration_frames == 8
    assert all(asdict(config)[name] > 0 for name in (
        "teacher_steps", "student_steps", "kd_steps", "phase1_steps",
        "q1_steps", "q2_steps", "q3_steps",
    ))
    assert set(V6_CANDIDATE_TOPOLOGIES) == {"P10-16", "P18-16", "P36-16"}
    assert all(all(channel % 16 == 0 for channel in topology)
               for topology in V6_CANDIDATE_TOPOLOGIES.values())


def test_v6_release_schedule_is_fail_closed() -> None:
    release = V6PipelineConfig.from_yaml(Path("configs/train/v6_1_量产训练.yaml"))
    assert release.export_mode == "release"
    invalid = asdict(release)
    invalid["q1_steps"] = 1999
    with pytest.raises(ValueError, match="Phase1=10k、Q1=2k"):
        V6PipelineConfig(**invalid)
