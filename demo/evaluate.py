from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

from demo.baselines.rule import rule_policy_action
from demo.evaluation.metrics import aggregate_policy_metrics
from demo.evaluation.suites import EvalScenario, get_suite


def _resolve_output_dir(out: str) -> Path:
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    if out_path.parts and out_path.parts[0] in {"outputs", "runs"}:
        return out_path
    return Path("outputs") / out_path


def _falcon_policy_action(blue_strength: float, red_strength: float) -> str:
    ratio = blue_strength / max(1e-6, red_strength)
    if ratio >= 1.08:
        return "assault"
    if ratio <= 0.95:
        return "fallback"
    return "hold"


def _simulate_episode(
    scenario: EvalScenario,
    policy_action_fn: Callable[[float, float], str],
    rng: random.Random,
) -> Dict[str, float]:
    blue_strength = 100.0 + rng.uniform(-15.0, 15.0)
    red_strength = 100.0 + rng.uniform(-15.0, 15.0)
    action = policy_action_fn(blue_strength, red_strength)
    action_bonus = {"assault": 0.05, "hold": 0.02, "fallback": -0.03}[action]

    win_prob = min(
        0.95,
        max(
            0.05,
            scenario.blue_advantage + action_bonus + rng.uniform(-scenario.volatility, scenario.volatility),
        ),
    )
    is_success = rng.random() < win_prob

    friendly_loss = max(0.04, min(0.90, 0.33 - (win_prob - 0.5) * 0.30 + rng.uniform(-0.05, 0.05)))
    roe_violations = int(action == "assault" and rng.random() < 0.16)

    return {
        "outcome": "success" if is_success else "failure",
        "friendly_loss": friendly_loss,
        "roe_violations": roe_violations,
    }


def _write_leaderboard(path: Path, rows: List[Dict[str, float]]) -> None:
    headers = ["policy", "success_rate", "friendly_loss", "roe_violation_rate"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "policy": row["policy"],
                    "success_rate": f"{float(row['success_rate']):.6f}",
                    "friendly_loss": f"{float(row['friendly_loss']):.6f}",
                    "roe_violation_rate": f"{float(row['roe_violation_rate']):.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FALCON evaluation suite runner")
    parser.add_argument("--suite", default="small", choices=["small", "standard", "stress"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mc", type=int, default=200, help="Monte Carlo runs per scenario")
    parser.add_argument("--out", "--output-dir", dest="out", default="outputs/eval_small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = get_suite(args.suite)
    output_dir = _resolve_output_dir(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows: Dict[str, List[Dict[str, float]]] = {"falcon": [], "rule": []}

    for scenario_idx, scenario in enumerate(suite.scenarios):
        for trial_idx in range(args.mc):
            seed = args.seed + scenario_idx * 100_003 + trial_idx
            rng = random.Random(seed)
            policy_rows["falcon"].append(_simulate_episode(scenario=scenario, policy_action_fn=_falcon_policy_action, rng=rng))
            policy_rows["rule"].append(_simulate_episode(scenario=scenario, policy_action_fn=rule_policy_action, rng=random.Random(seed + 17)))

    leaderboard: List[Dict[str, float]] = []
    for policy_name, rows in policy_rows.items():
        aggregate = aggregate_policy_metrics(rows)
        leaderboard.append({"policy": policy_name, **aggregate})

    leaderboard.sort(key=lambda x: (-x["success_rate"], x["friendly_loss"], x["roe_violation_rate"]))
    aggregate_payload = {
        "suite": suite.name,
        "seed": args.seed,
        "mc": args.mc,
        "leaderboard": leaderboard,
    }

    _write_leaderboard(output_dir / "leaderboard.csv", leaderboard)
    (output_dir / "metrics_aggregate.json").write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")

    print(f"[demo.evaluate] suite={suite.name} mc={args.mc} seed={args.seed}")
    print(f"[demo.evaluate] leaderboard={output_dir / 'leaderboard.csv'}")
    print(f"[demo.evaluate] aggregate={output_dir / 'metrics_aggregate.json'}")


if __name__ == "__main__":
    main()
