import csv
import json
import subprocess
import sys
from pathlib import Path


def _run_demo(out_dir: Path, roe: str, preference: str) -> None:
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
        "--roe",
        roe,
        "--preference",
        preference,
    ]
    subprocess.run(cmd, check=True)


def test_roe_blocked_event_and_preference_tradeoff(tmp_path: Path):
    out_roe_on = tmp_path / "roe_on_loss"
    out_roe_off = tmp_path / "roe_off_time"

    _run_demo(out_roe_on, roe="on", preference="loss_min")
    _run_demo(out_roe_off, roe="off", preference="time_min")

    with (out_roe_on / "metrics.csv").open(encoding="utf-8") as fp:
        on_rows = list(csv.DictReader(fp))
    assert any(row["event_type"] == "ROE_BLOCKED" for row in on_rows)

    on_summary = json.loads((out_roe_on / "summary.json").read_text(encoding="utf-8"))
    off_summary = json.loads((out_roe_off / "summary.json").read_text(encoding="utf-8"))

    assert on_summary["loss_tradeoff"] != off_summary["loss_tradeoff"]
    assert on_summary["time_tradeoff"] != off_summary["time_tradeoff"]
