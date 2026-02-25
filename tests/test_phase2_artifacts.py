"""Phase2 artifact schema tests."""

import json
from types import SimpleNamespace

from train import _write_phase2_reports, _write_run_manifest


def test_write_run_manifest_outputs_standard_and_legacy(tmp_path):
    args = SimpleNamespace(
        run_id="test-run",
        phase=2,
        algorithm="mappo",
        seed=7,
        config=None,
        episodes=3,
        checkpoint_dir=str(tmp_path),
    )
    out = _write_run_manifest(args)

    assert out.endswith("run_manifest.json")
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "run_manifest_test-run.json").exists()


def test_write_phase2_reports_outputs_metrics_and_eval(tmp_path):
    args = SimpleNamespace(
        run_id="test-run",
        phase=2,
        seed=7,
        checkpoint_dir=str(tmp_path),
    )
    metrics = {
        "win_rate": 0.6,
        "mission_time": 12.0,
        "blue_casualties": 3.0,
        "force_reduction": 0.2,
        "stability": 0.1,
        "episodes": 10,
    }
    _write_phase2_reports(args, "mappo", metrics)

    metrics_json = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    eval_json = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))

    assert metrics_json["schema_version"] == "falcon-phase2-metrics-v1"
    assert metrics_json["algorithm"] == "mappo"
    assert metrics_json["metrics"]["win_rate"] == 0.6

    assert eval_json["schema_version"] == "falcon-evaluation-summary-v1"
    assert eval_json["summary"]["mission_time"] == 12.0
