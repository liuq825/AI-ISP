from pathlib import Path

from ai_isp.selection import CandidateMetrics, converge_candidate_artifacts, select_best_candidate


def _metrics(candidate_id: str, macs: int) -> CandidateMetrics:
    return CandidateMetrics(
        candidate_id, 0.02, 0.0002, 0.05, 0.03, macs, 1000, 2, 20, 1.0, macs // 100, 100, 0.0001
    )


def test_quality_equivalent_candidates_select_lower_mac_and_delete_loser_artifacts(tmp_path: Path) -> None:
    candidates = [_metrics("P10-16", 34), _metrics("P18-16", 30), _metrics("P36-16", 28)]
    selected, report = select_best_candidate(candidates)
    assert selected.candidate_id == "P36-16"
    for item in candidates:
        folder = tmp_path / item.candidate_id
        folder.mkdir()
        (folder / "executable.onnx").write_bytes(b"candidate")
    summary = converge_candidate_artifacts(tmp_path, selected.candidate_id, report)
    assert (tmp_path / "P36-16" / "executable.onnx").is_file()
    assert not (tmp_path / "P10-16").exists()
    assert not (tmp_path / "P18-16").exists()
    assert summary["development_selected"] is True
