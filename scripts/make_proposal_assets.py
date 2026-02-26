#!/usr/bin/env python3
"""Generate proposal-ready assets from evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List, Tuple

REQUIRED_COLUMNS = ["policy", "success_rate", "friendly_loss", "roe_violation_rate"]
GUIDANCE = (
    "leaderboard.csv not found. Run: python -m falcon.evaluate --suite small --out <EVAL_DIR>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate proposal assets from evaluation outputs."
    )
    parser.add_argument("--eval", required=True, dest="eval_dir", help="Evaluation output directory")
    parser.add_argument("--out", required=True, dest="out_dir", help="Output asset directory")
    return parser.parse_args()


def maybe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


def sort_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            -maybe_float(row.get("success_rate", "")),
            maybe_float(row.get("friendly_loss", "")),
            row.get("policy", ""),
        ),
    )


def validate_columns(fieldnames: Iterable[str] | None) -> Tuple[bool, List[str]]:
    headers = list(fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in headers]
    return (len(missing) == 0, missing)


def read_leaderboard(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        valid, missing = validate_columns(reader.fieldnames)
        if not valid:
            missing_text = ", ".join(missing)
            print(
                f"Missing required columns in leaderboard.csv: {missing_text}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return list(reader)


def write_results_table(rows: List[Dict[str, str]], out_path: Path) -> None:
    lines = [
        "# FALCON Evaluation Summary",
        "",
        "| policy | success_rate | friendly_loss | roe_violation_rate |",
        "|---|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {policy} | {success_rate} | {friendly_loss} | {roe_violation_rate} |".format(
                policy=row.get("policy", ""),
                success_rate=row.get("success_rate", ""),
                friendly_loss=row.get("friendly_loss", ""),
                roe_violation_rate=row.get("roe_violation_rate", ""),
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(out_dir: Path, eval_dir: Path) -> None:
    content = f"""# Proposal Assets

This directory contains proposal-ready artifacts generated from evaluation outputs.

## Regenerate assets

Run the command below from the repository root:

```bash
python scripts/make_proposal_assets.py --eval {eval_dir.as_posix()} --out {out_dir.as_posix()}
```

## Expected input files

- `{eval_dir.as_posix()}/leaderboard.csv` (required)
- `{eval_dir.as_posix()}/calibration.png` (optional; copied if present)

## Generated output files

- `{out_dir.as_posix()}/results_table.md`
- `{out_dir.as_posix()}/README.md`
- `{out_dir.as_posix()}/calibration.png` (optional)
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def maybe_copy_calibration(eval_dir: Path, out_dir: Path) -> None:
    calibration_src = eval_dir / "calibration.png"
    calibration_dst = out_dir / "calibration.png"
    if calibration_src.exists():
        shutil.copyfile(calibration_src, calibration_dst)
    else:
        print("Warning: calibration.png not found in eval outputs; skipping.", file=sys.stderr)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    leaderboard_path = eval_dir / "leaderboard.csv"

    if not leaderboard_path.exists():
        print(GUIDANCE, file=sys.stderr)
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_leaderboard(leaderboard_path)
    sorted_rows = sort_rows(rows)

    write_results_table(sorted_rows, out_dir / "results_table.md")
    write_readme(out_dir, eval_dir)
    maybe_copy_calibration(eval_dir, out_dir)


if __name__ == "__main__":
    main()
