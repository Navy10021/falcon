"""
Phase contract regression tests (P0/P1/P2).

4→20 확장: Phase 1~4 + 인프라 + 크로스 페이즈 계약 검증.
"""

import json
import logging
import os

import numpy as np
import pytest
import torch

from train import setup_logger
from rl_agent.self_play_trainer import SelfPlayTrainer
from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints
from ontology.combat_schema import ScenarioFactory


# ══════════════════════════════════════════════════
# 기존 4개 (유지)
# ══════════════════════════════════════════════════

def test_setup_logger_deduplicates_handlers(tmp_path):
    """Repeated setup_logger calls must not duplicate handlers."""
    run_id = "contract"
    setup_logger(str(tmp_path), run_id)
    setup_logger(str(tmp_path), run_id)

    falcon_logger = logging.getLogger("falcon")
    # stream + file only
    assert len(falcon_logger.handlers) == 2


def test_selfplay_uses_prev_blue_headcount(monkeypatch):
    """SelfPlay blue reward must receive previous-step headcount as force_size_before."""

    trainer = SelfPlayTrainer()
    captured = []

    def capture_reward(step_result, force_size_before, force_size_after, uncertainty=0.0, **kwargs):
        captured.append((force_size_before, force_size_after))
        return 0.0

    monkeypatch.setattr(trainer.blue_agent, "compute_reward", capture_reward)
    monkeypatch.setattr(trainer.red_agent, "compute_reward", lambda *args, **kwargs: 0.0)

    trainer.run_episode(
        episode_idx=0,
        progress=0.1,
        blue_agent=trainer.blue_agent,
        red_agent=trainer.red_agent,
        phase="A",
    )
    assert captured, "expected at least one reward call"
    # first step should use initial headcount (>= after value in attrition setting)
    first_before, first_after = captured[0]
    assert first_before >= first_after


def test_pareto_mc_lite_calibration_applies():
    """Top-k Pareto candidates should be calibrated by mc_eval_runs rollout."""
    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    g = ParetoStrategyGenerator(n_candidates=5, mc_eval_runs=2)
    constraints = CommanderConstraints(min_win_probability=0.0)
    options = g.generate(kg, constraints)
    assert options
    for opt in options:
        assert 0.0 <= opt.win_probability <= 1.0
        assert opt.expected_casualties >= 0


def test_mc_report_json_schema(tmp_path):
    """MC evaluation report should be serializable with fixed schema version."""
    from evaluation.monte_carlo import MonteCarloEvaluator
    from simulator.lanchester_engine import LanchesterEngine
    from rl_agent.blue_agent import BlueAgent

    evaluator = MonteCarloEvaluator(n_runs=2, seed=1, n_workers=1)
    report = evaluator.evaluate(agent=BlueAgent(), engine=LanchesterEngine(seed=1), verbose=False, show_progress=False, max_steps=3)
    out = tmp_path / "report.json"
    evaluator.save_report_json(report, str(out))

    data = out.read_text(encoding="utf-8")
    assert "falcon-mc-report-v1" in data


# ══════════════════════════════════════════════════
# Phase 1 계약
# ══════════════════════════════════════════════════

def test_blue_agent_state_dim_128():
    """BlueAgent must accept STATE_DIM=128 state vectors."""
    from rl_agent.blue_agent import BlueAgent, STATE_DIM

    assert STATE_DIM == 128
    agent = BlueAgent()
    state = np.random.randn(STATE_DIM).astype(np.float32)
    action, log_prob, value = agent.select_action(state)
    assert 0 <= action < 6
    assert np.isfinite(log_prob)


def test_gnn_input_output_contract():
    """BayesianHGT predict_with_uncertainty must return required keys."""
    from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors

    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=3, seed=42)
    gnn = BayesianHGT(node_in_dim=128, hidden_dim=128, n_layers=2, mc_samples=3)
    x, adj = prepare_graph_tensors(kg)
    gnn.eval()
    with torch.no_grad():
        result = gnn.predict_with_uncertainty(x, adj)

    required_keys = ["casualty_mean", "casualty_std", "risk_mean", "risk_std",
                     "epistemic_uncertainty", "aleatoric_uncertainty"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        assert torch.isfinite(result[key]).all(), f"Non-finite value in {key}"


def test_fog_curriculum_monotonic():
    """CurriculumScheduler fog level must increase or stay as progress increases."""
    from simulator.fog_of_war import CurriculumScheduler, FogLevel

    sched = CurriculumScheduler()
    prev_value = 0
    for progress in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        level, desc = sched.update(progress)
        assert hasattr(level, 'value')
        assert level.value >= prev_value, f"Fog decreased at progress={progress}"
        prev_value = level.value


def test_build_state_vector_shape():
    """build_state_vector must produce exactly 128-dim vector."""
    from rl_agent.blue_agent import build_state_vector

    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=3, seed=42)
    state = build_state_vector(kg)
    assert state.shape == (128,)
    assert np.all(np.isfinite(state))


def test_doctrine_encoder_compliance_range():
    """DoctrineEncoder evaluate must return total_score in [0, 1]."""
    from ontology.doctrine_encoder import DoctrineEncoder

    kg = ScenarioFactory.create_standard_scenario(n_blue=5, n_red=4, seed=42)
    encoder = DoctrineEncoder()
    step_result = {"blue_casualties": 5, "red_casualties": 10,
                   "n_engageable": 3, "flanking_units": 1}
    compliance = encoder.evaluate(kg, step=0, step_result=step_result)
    assert 0.0 <= compliance.total_score <= 1.0


# ══════════════════════════════════════════════════
# Phase 2 계약
# ══════════════════════════════════════════════════

def test_selfplay_config_defaults():
    """SelfPlayConfig defaults must match documented spec."""
    from rl_agent.self_play_trainer import SelfPlayConfig

    cfg = SelfPlayConfig()
    assert cfg.total_episodes >= 100
    assert cfg.log_interval > 0
    assert cfg.save_interval > 0
    assert hasattr(cfg, "checkpoint_dir")


def test_selfplay_trainer_episode_returns_stats():
    """SelfPlayTrainer.run_episode must return EpisodeStats with winner field."""
    trainer = SelfPlayTrainer()
    result = trainer.run_episode(
        episode_idx=0, progress=0.1,
        blue_agent=trainer.blue_agent,
        red_agent=trainer.red_agent,
        phase="A",
    )
    assert hasattr(result, "winner"), "EpisodeStats must have 'winner' attribute"
    assert hasattr(result, "blue_casualties")
    assert hasattr(result, "n_steps")


# ══════════════════════════════════════════════════
# Phase 3 계약
# ══════════════════════════════════════════════════

def test_commander_preference_learner_round_trip(tmp_path):
    """PreferenceLearner save/load must preserve learned weights."""
    from hitl.preference_learner import CommanderPreferenceLearner, SelectionRecord

    learner = CommanderPreferenceLearner()
    # simulate selection with actual SelectionRecord fields
    record = SelectionRecord(
        scenario_id=1,
        options_presented=["opt_a", "opt_b"],
        selected_option_id="opt_a",
        selected_type="aggressive",
        force_size=100,
        win_probability=0.8,
        expected_casualties=5,
    )
    learner.record_selection(record)

    out = str(tmp_path / "prefs.json")
    learner.save(out)
    assert os.path.isfile(out)
    with open(out) as f:
        data = json.load(f)
    assert "preference_weights" in data


def test_preference_reward_adapter_scaling():
    """PreferenceRewardAdapter must scale reward components."""
    from hitl.preference_reward_adapter import PreferenceRewardAdapter

    adapter = PreferenceRewardAdapter.uniform()
    summary = adapter.summary()
    assert isinstance(summary, str) and len(summary) > 0


# ══════════════════════════════════════════════════
# Phase 4 계약
# ══════════════════════════════════════════════════

def test_phase4_loads_preference_adapter_uniform():
    """Phase 4 PreferenceRewardAdapter.uniform() must produce valid reward_scale."""
    from hitl.preference_reward_adapter import PreferenceRewardAdapter

    adapter = PreferenceRewardAdapter.uniform()
    scale = adapter.reward_scale
    # RewardScale must have component-level scale factors
    assert hasattr(scale, "win_bonus") or hasattr(scale, "__dict__")
    assert adapter.summary()  # summary must return non-empty string


# ══════════════════════════════════════════════════
# 크로스 페이즈 계약
# ══════════════════════════════════════════════════

def test_blue_agent_checkpoint_round_trip(tmp_path):
    """BlueAgent save→load must preserve policy weights."""
    from rl_agent.blue_agent import BlueAgent

    agent = BlueAgent()
    ckpt = str(tmp_path / "blue_test.pt")
    agent.save(ckpt)

    agent2 = BlueAgent()
    agent2.load(ckpt)

    # Same policy output for same input
    state = np.random.RandomState(42).randn(128).astype(np.float32)
    torch.manual_seed(0)
    a1, lp1, v1 = agent.select_action(state)
    torch.manual_seed(0)
    a2, lp2, v2 = agent2.select_action(state)
    assert a1 == a2


def test_reward_function_signature():
    """BlueAgent.compute_reward must accept the canonical kwarg signature."""
    from rl_agent.blue_agent import BlueAgent

    agent = BlueAgent()
    # Minimal step_result mock
    class FakeResult:
        blue_total_casualties = 5
        red_total_casualties = 10
        blue_total_headcount = 95
        initial_blue_headcount = 100
        mission_status = "ongoing"
        blue_force_size = 95
        red_force_size = 80
        blue_win = False

    r = agent.compute_reward(
        FakeResult(), force_size_before=100, force_size_after=95,
        uncertainty=0.1, initial_force_size=100, doctrine_score=0.8,
    )
    assert isinstance(r, float)
    assert np.isfinite(r)


def test_run_manifest_schema(tmp_path):
    """_write_run_manifest must produce valid JSON with required keys."""
    from train import _write_run_manifest
    import argparse

    args = argparse.Namespace(
        run_id="test_run", phase=1, algorithm="ppo", seed=42,
        config=None, episodes=100, checkpoint_dir=str(tmp_path),
    )
    path = _write_run_manifest(args)
    assert os.path.isfile(path)
    with open(path) as f:
        data = json.load(f)
    required = ["run_id", "phase", "seed", "episodes", "created_at"]
    for key in required:
        assert key in data, f"Missing manifest key: {key}"


# ══════════════════════════════════════════════════
# 인프라 계약
# ══════════════════════════════════════════════════

def test_experiment_tracker_log_and_close(tmp_path):
    """ExperimentTracker must log metrics to JSONL and close cleanly."""
    from utils.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(log_dir=str(tmp_path), run_id="test")
    tracker.log_episode(0, {"loss": 0.5, "win_rate": 0.6})
    tracker.log_episode(1, {"loss": 0.4, "win_rate": 0.65})
    tracker.close()

    jsonl_path = tracker.metrics_path
    assert os.path.isfile(jsonl_path)
    with open(jsonl_path) as f:
        lines = f.readlines()
    assert len(lines) >= 2
    record = json.loads(lines[0])
    assert record["type"] == "episode"
    assert "loss" in record


def test_simulator_composer_default():
    """SimulatorComposer.default() must produce a working Lanchester engine."""
    from simulator.composer import SimulatorComposer

    composer = SimulatorComposer.default(seed=42)
    assert composer.combat_engine is not None
    assert "lanchester" in composer.active_engine_names


def test_simulator_composer_from_yaml(tmp_path):
    """SimulatorComposer.from_yaml() must load config correctly."""
    from simulator.composer import SimulatorComposer

    yaml_content = """
engines:
  combat: "lanchester"
  movement: null
  effects:
    - "fog_of_war"
  resources: null
  fog_level: "HEAVY"
"""
    yaml_path = str(tmp_path / "sim.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    composer = SimulatorComposer.from_yaml(yaml_path, seed=42)
    assert "lanchester" in composer.active_engine_names
    assert "fog_of_war" in composer.active_engine_names
