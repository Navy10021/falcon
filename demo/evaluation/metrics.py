from __future__ import annotations

from typing import Dict, Iterable


def aggregate_policy_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {
            "success_rate": 0.0,
            "friendly_loss": 0.0,
            "roe_violation_rate": 0.0,
        }

    success_rate = sum(1.0 for row in rows if row["outcome"] == "success") / len(rows)
    friendly_loss = sum(float(row["friendly_loss"]) for row in rows) / len(rows)
    roe_violation_rate = sum(float(row["roe_violations"]) for row in rows) / len(rows)
    return {
        "success_rate": success_rate,
        "friendly_loss": friendly_loss,
        "roe_violation_rate": roe_violation_rate,
    }
