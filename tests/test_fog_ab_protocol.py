"""Fog Curriculum A/B 프로토콜 단위 테스트."""
import json
import os

import numpy as np
import pytest


def test_fog_ab_protocol_instantiation():
    """FogABProtocol should instantiate with default config."""
    from evaluation.fog_ab_protocol import FogABProtocol, FogABConfig

    protocol = FogABProtocol()
    assert protocol.config.n_episodes == 100
    assert len(protocol.config.conditions) == 2


def test_fog_ab_config_custom_conditions():
    """Custom conditions should be accepted."""
    from evaluation.fog_ab_protocol import FogABProtocol, FogABConfig, ABCondition

    config = FogABConfig(
        n_episodes=10,
        conditions=[
            ABCondition(name="A", use_curriculum=True),
            ABCondition(name="B", use_curriculum=False, fixed_fog_level="HEAVY"),
        ],
    )
    protocol = FogABProtocol(config)
    assert len(protocol.config.conditions) == 2
    assert protocol.config.conditions[1].fixed_fog_level == "HEAVY"


def test_fog_ab_run_minimal():
    """A/B protocol run with minimal episodes should produce valid report."""
    from evaluation.fog_ab_protocol import FogABProtocol, FogABConfig, ABCondition
    from rl_agent.blue_agent import BlueAgent
    from simulator.lanchester_engine import LanchesterEngine

    config = FogABConfig(
        n_episodes=3,
        max_steps_per_episode=5,
        seed=42,
        conditions=[
            ABCondition(name="curriculum", use_curriculum=True),
            ABCondition(name="fixed", use_curriculum=False, fixed_fog_level="CLEAR"),
        ],
    )
    protocol = FogABProtocol(config)
    agent = BlueAgent()
    engine = LanchesterEngine(seed=42)

    report = protocol.run(agent, engine)
    assert "curriculum" in report.conditions
    assert "fixed" in report.conditions
    assert 0.0 <= report.conditions["curriculum"].win_rate <= 1.0
    assert isinstance(report.delta_win_rate, float)
    assert isinstance(report.effect_size_cohens_d, float)
    assert len(report.summary) > 0


def test_fog_ab_save_report(tmp_path):
    """Report should be serializable to JSON."""
    from evaluation.fog_ab_protocol import (
        FogABProtocol, ABReport, ConditionMetrics,
    )

    report = ABReport(
        config={"n_episodes": 10, "seed": 42},
        conditions={
            "A": ConditionMetrics(
                condition_name="A", win_rate=0.7, avg_reward=5.0,
                reward_std=1.2, high_fog_win_rate=0.5,
                episode_rewards=[5.0] * 10, episode_wins=[1] * 7 + [0] * 3,
            ),
            "B": ConditionMetrics(
                condition_name="B", win_rate=0.5, avg_reward=3.0,
                reward_std=1.5, high_fog_win_rate=0.3,
                episode_rewards=[3.0] * 10, episode_wins=[1] * 5 + [0] * 5,
            ),
        },
        delta_win_rate=0.2,
        delta_reward=2.0,
        effect_size_cohens_d=1.48,
        curriculum_benefit=True,
        summary="test summary",
    )

    path = str(tmp_path / "fog_ab.json")
    FogABProtocol.save_report(report, path)
    assert os.path.isfile(path)
    with open(path) as f:
        data = json.load(f)
    assert data["curriculum_benefit"] is True
    assert data["delta_win_rate"] == pytest.approx(0.2)
    assert "A" in data["conditions"]
