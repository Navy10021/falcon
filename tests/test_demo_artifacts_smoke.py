import json
import subprocess
import sys
from pathlib import Path


def test_demo_cli_generates_fixed_artifacts(tmp_path: Path):
    out_dir = tmp_path / "demo_urban"
    cmd = [
        sys.executable,
        "-m",
        "falcon.demo",
        "--scenario",
        "urban_defense",
        "--seed",
        "42",
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    expected = [
        out_dir / "summary.json",
        out_dir / "metrics.csv",
        out_dir / "fig_episode.png",
        out_dir / "aar.html",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    required_keys = {
        "scenario",
        "seed",
        "success_rate",
        "friendly_loss",
        "roe_violation_rate",
        "runtime_sec",
    }
    assert required_keys.issubset(summary)
