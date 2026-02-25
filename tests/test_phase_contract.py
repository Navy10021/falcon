"""Phase contract regression tests (P0/P1/P2)."""

import logging

from train import setup_logger
from rl_agent.self_play_trainer import SelfPlayTrainer
from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints
from ontology.combat_schema import ScenarioFactory


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

