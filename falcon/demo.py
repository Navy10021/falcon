from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Dict, List

from ontology.combat_schema import ForceAlignment, ScenarioFactory

from falcon.io.artifacts import artifact_paths, ensure_output_dir, write_metrics_csv, write_summary
from falcon.io.logging import log_info
from falcon.report import render_aar_html
from falcon.visualization.plots import save_episode_plot


@dataclass
class EpisodeResult:
    episode: int
    outcome: str
    friendly_loss: float
    enemy_loss: float
    roe_violations: int
    duration_sec: float


def _scenario_blue_advantage(scenario: str) -> float:
    presets = {
        "urban_defense": 0.58,
        "coastal_raids": 0.51,
        "air_intercept": 0.55,
    }
    return presets.get(scenario, 0.5)


def _rule_policy_action(blue_strength: float, red_strength: float) -> str:
    if blue_strength > red_strength * 1.1:
        return "assault"
    if blue_strength < red_strength * 0.9:
        return "fallback"
    return "hold"


def run_demo_simulation(scenario: str, seed: int, policy: str = "rule") -> List[EpisodeResult]:
    random.seed(seed)

    kg = ScenarioFactory.create_standard_scenario(seed=seed)
    blue_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
    red_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.RED]

    blue_strength = float(sum(max(1, unit.headcount) * unit.combat_power for unit in blue_units))
    red_strength = float(sum(max(1, unit.headcount) * unit.combat_power for unit in red_units))

    results: List[EpisodeResult] = []
    scenario_advantage = _scenario_blue_advantage(scenario)
    for episode in range(1, 6):
        t0 = time.perf_counter()
        action = _rule_policy_action(blue_strength, red_strength) if policy == "rule" else "fallback"
        action_bonus = {"assault": 0.05, "hold": 0.02, "fallback": -0.02}[action]
        win_prob = min(0.95, max(0.05, scenario_advantage + action_bonus + random.uniform(-0.08, 0.08)))
        won = random.random() < win_prob

        friendly_loss = max(0.05, min(0.85, 0.32 - (win_prob - 0.5) * 0.28 + random.uniform(-0.05, 0.05)))
        enemy_loss = max(0.05, min(0.95, 0.35 + (win_prob - 0.5) * 0.30 + random.uniform(-0.05, 0.05)))
        roe_violations = 0 if action != "assault" else int(random.random() < 0.2)

        results.append(
            EpisodeResult(
                episode=episode,
                outcome="success" if won else "failure",
                friendly_loss=friendly_loss,
                enemy_loss=enemy_loss,
                roe_violations=roe_violations,
                duration_sec=max(0.01, time.perf_counter() - t0),
            )
        )

        blue_strength *= max(0.5, 1.0 - friendly_loss * 0.25)
        red_strength *= max(0.5, 1.0 - enemy_loss * 0.25)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command FALCON demo")
    parser.add_argument("--scenario", default="urban_defense")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/demo_urban")
    parser.add_argument("--policy", default="rule", choices=["rule", "fallback"])
    return parser.parse_args()


def _build_summary(scenario: str, seed: int, runtime_sec: float, metrics_rows: List[Dict[str, float]]) -> Dict[str, float]:
    success_count = sum(1 for row in metrics_rows if row["outcome"] == "success")
    friendly_loss = sum(float(row["friendly_loss"]) for row in metrics_rows) / len(metrics_rows)
    roe_violation_rate = sum(int(row["roe_violations"]) for row in metrics_rows) / len(metrics_rows)
    return {
        "scenario": scenario,
        "seed": seed,
        "success_rate": success_count / len(metrics_rows),
        "friendly_loss": friendly_loss,
        "roe_violation_rate": roe_violation_rate,
        "runtime_sec": runtime_sec,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.out)
    artifacts = artifact_paths(output_dir)

    log_info(f"scenario={args.scenario}, seed={args.seed}, policy={args.policy}")
    t0 = time.perf_counter()
    results = run_demo_simulation(scenario=args.scenario, seed=args.seed, policy=args.policy)
    runtime_sec = time.perf_counter() - t0

    metrics_rows = [
        {
            "episode": r.episode,
            "outcome": r.outcome,
            "friendly_loss": r.friendly_loss,
            "enemy_loss": r.enemy_loss,
            "roe_violations": r.roe_violations,
            "duration_sec": r.duration_sec,
        }
        for r in results
    ]
    summary = _build_summary(args.scenario, args.seed, runtime_sec, metrics_rows)

    write_summary(artifacts["summary.json"], summary)
    write_metrics_csv(artifacts["metrics.csv"], metrics_rows)
    save_episode_plot(artifacts["fig_episode.png"], metrics_rows)
    artifacts["aar.html"].write_text(render_aar_html(summary, metrics_rows), encoding="utf-8")

    log_info(f"Artifacts written to {output_dir}")
    for name, path in artifacts.items():
        log_info(f" - {name}: {path}")


if __name__ == "__main__":
    main()
