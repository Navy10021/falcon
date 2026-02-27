from __future__ import annotations

import csv
import json
import subprocess
import sys


def test_eval_small_suite_generates_leaderboard(tmp_path):
    out_dir = tmp_path / "eval_small"
    cmd = [
        sys.executable,
        "-m",
        "demo.evaluate",
        "--suite",
        "small",
        "--seed",
        "0",
        "--mc",
        "10",
        "--out",
        str(out_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    leaderboard_path = out_dir / "leaderboard.csv"
    aggregate_path = out_dir / "metrics_aggregate.json"

    assert leaderboard_path.exists()
    assert aggregate_path.exists()

    with leaderboard_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    assert rows, "leaderboard must contain at least one policy row"
    required_columns = {"policy", "success_rate", "friendly_loss", "roe_violation_rate"}
    assert required_columns.issubset(rows[0].keys())

    with aggregate_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    assert payload["suite"] == "small"
    assert payload["seed"] == 0
    assert payload["mc"] == 10


def test_eval_accepts_output_dir_alias(tmp_path):
    out_dir = tmp_path / "eval_alias"
    cmd = [
        sys.executable,
        "-m",
        "demo.evaluate",
        "--suite",
        "small",
        "--seed",
        "42",
        "--mc",
        "5",
        "--output-dir",
        str(out_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "metrics_aggregate.json").exists()
