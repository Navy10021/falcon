from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EvalScenario:
    name: str
    blue_advantage: float
    volatility: float


@dataclass(frozen=True)
class EvalSuite:
    name: str
    scenarios: List[EvalScenario]


SUITES: Dict[str, EvalSuite] = {
    "small": EvalSuite(
        name="small",
        scenarios=[
            EvalScenario(name="urban_defense", blue_advantage=0.57, volatility=0.08),
            EvalScenario(name="coastal_raids", blue_advantage=0.52, volatility=0.10),
            EvalScenario(name="air_intercept", blue_advantage=0.54, volatility=0.09),
        ],
    ),
    "standard": EvalSuite(
        name="standard",
        scenarios=[
            EvalScenario(name="urban_defense", blue_advantage=0.56, volatility=0.09),
            EvalScenario(name="coastal_raids", blue_advantage=0.51, volatility=0.11),
            EvalScenario(name="air_intercept", blue_advantage=0.55, volatility=0.10),
            EvalScenario(name="joint_logistics", blue_advantage=0.50, volatility=0.12),
            EvalScenario(name="cyber_denial", blue_advantage=0.49, volatility=0.13),
        ],
    ),
    "stress": EvalSuite(
        name="stress",
        scenarios=[
            EvalScenario(name="urban_defense", blue_advantage=0.53, volatility=0.13),
            EvalScenario(name="coastal_raids", blue_advantage=0.49, volatility=0.15),
            EvalScenario(name="air_intercept", blue_advantage=0.51, volatility=0.14),
            EvalScenario(name="joint_logistics", blue_advantage=0.48, volatility=0.16),
            EvalScenario(name="cyber_denial", blue_advantage=0.47, volatility=0.17),
            EvalScenario(name="satcom_jamming", blue_advantage=0.46, volatility=0.18),
        ],
    ),
}


def get_suite(name: str) -> EvalSuite:
    try:
        return SUITES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown suite: {name}") from exc
