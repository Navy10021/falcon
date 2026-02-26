from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

REQUIRED_ARTIFACTS = ("summary.json", "metrics.csv", "fig_episode.png", "aar.html")


def resolve_output_dir(out: str) -> Path:
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    if out_path.parts and out_path.parts[0] == "runs":
        return out_path
    return Path("runs") / out_path


def ensure_output_dir(out: str) -> Path:
    output_dir = resolve_output_dir(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def artifact_paths(output_dir: Path) -> Dict[str, Path]:
    return {name: output_dir / name for name in REQUIRED_ARTIFACTS}


def write_summary(path: Path, summary: Dict[str, float]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows_list: List[Dict[str, float]] = list(rows)
    base_headers = ["episode", "outcome", "friendly_loss", "enemy_loss", "roe_violations", "duration_sec"]
    extra_headers = []
    if rows_list:
        extra_headers = [k for k in rows_list[0].keys() if k not in base_headers]
    headers = base_headers + extra_headers
    lines = [",".join(headers)]
    for row in rows_list:
        values = []
        for header in headers:
            if header in {"friendly_loss", "enemy_loss", "duration_sec", "loss_tradeoff", "time_tradeoff", "decision_score"}:
                values.append(f"{float(row[header]):.6f}")
            elif header in {"episode", "roe_violations"}:
                values.append(str(int(row[header])))
            else:
                values.append(str(row[header]))
        lines.append(
            ",".join(values)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
