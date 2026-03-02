"""
Phase III R11/R12 테스트.

R11: 다도메인 시나리오 4×4 배치 실험 프레임워크
R12: Tier C → Tier B 승격 (RARL + NFSP 학습 루프 연결)
"""
import json
import os

import numpy as np
import pytest


# ══════════════════════════════════════════════════
# R11: 다도메인 배치 실험
# ══════════════════════════════════════════════════

def test_multidomain_runner_matrix_shape():
    """4×4 매트릭스의 모든 셀이 채워져야 한다."""
    from utils.multidomain_runner import (
        MultiDomainRunner, DomainExperimentConfig, DOMAINS,
    )

    config = DomainExperimentConfig(seeds=[0], eval_runs_per_seed=2, max_steps=5)
    runner = MultiDomainRunner(config)
    matrix = runner.run_matrix(verbose=False)

    for td in DOMAINS:
        for ed in DOMAINS:
            cell = matrix.get_cell(td, ed)
            assert cell["n"] > 0, f"Cell ({td},{ed}) has no results"
            assert 0.0 <= cell["win_rate_mean"] <= 1.0


def test_multidomain_ood_degradation():
    """OOD 성능 저하율은 실수값으로 반환되어야 한다."""
    from utils.multidomain_runner import (
        MultiDomainRunner, DomainExperimentConfig,
    )

    config = DomainExperimentConfig(seeds=[0], eval_runs_per_seed=2, max_steps=5)
    runner = MultiDomainRunner(config)
    runner.run_matrix(verbose=False)

    degradation = runner.matrix.ood_degradation("ground")
    assert isinstance(degradation, dict)
    # ground→ground 은 포함되지 않아야 함
    assert "ground" not in degradation


def test_multidomain_to_table():
    """매트릭스 텍스트 테이블이 생성되어야 한다."""
    from utils.multidomain_runner import (
        MultiDomainRunner, DomainExperimentConfig,
    )

    config = DomainExperimentConfig(seeds=[0], eval_runs_per_seed=2, max_steps=5)
    runner = MultiDomainRunner(config)
    runner.run_matrix(verbose=False)

    table = runner.matrix.to_table()
    assert isinstance(table, str)
    assert "ground" in table
    assert "naval" in table


def test_multidomain_batch_experiment_saves_json(tmp_path):
    """run_batch_experiment는 JSON 리포트를 저장해야 한다."""
    from utils.multidomain_runner import (
        MultiDomainRunner, DomainExperimentConfig,
    )

    config = DomainExperimentConfig(
        seeds=[0], eval_runs_per_seed=2, max_steps=5,
        train_domains=["ground", "air"],
        eval_domains=["ground", "air"],
    )
    runner = MultiDomainRunner(config)
    out_dir = str(tmp_path / "md_output")
    report = runner.run_batch_experiment(output_dir=out_dir)

    # JSON 파일 존재
    assert os.path.isfile(os.path.join(out_dir, "multidomain_report.json"))
    assert os.path.isfile(os.path.join(out_dir, "matrix_table.txt"))

    # 보고서 구조
    assert "version" in report
    assert report["version"] == "falcon-multidomain-v1"
    assert "matrix" in report
    assert "ood_analysis" in report
    assert "summary" in report
    assert "in_domain_win_rate_avg" in report["summary"]


# ══════════════════════════════════════════════════
# R12: Tier C → Tier B 승격
# ══════════════════════════════════════════════════

def test_tier_c_rarl_instantiate():
    """TierCTrainer(algorithm='rarl') 인스턴스화가 성공해야 한다."""
    from rl_agent.tier_c_trainer import TierCTrainer, TierCConfig

    config = TierCConfig(algorithm="rarl")
    trainer = TierCTrainer(config)
    assert trainer.agent is not None


def test_tier_c_nfsp_instantiate():
    """TierCTrainer(algorithm='nfsp') 인스턴스화가 성공해야 한다."""
    from rl_agent.tier_c_trainer import TierCTrainer, TierCConfig

    config = TierCConfig(algorithm="nfsp")
    trainer = TierCTrainer(config)
    assert trainer.agent is not None


def test_tier_c_rarl_train_minimal(tmp_path):
    """RARL 3에피소드 학습이 크래시 없이 완료되어야 한다."""
    from rl_agent.tier_c_trainer import TierCTrainer, TierCConfig

    config = TierCConfig(
        algorithm="rarl",
        episodes=3,
        max_steps=5,
        seed=42,
        log_interval=10,
        save_interval=100,
        checkpoint_dir=str(tmp_path),
    )
    trainer = TierCTrainer(config)
    metrics = trainer.train()

    assert metrics["algorithm"] == "rarl"
    assert metrics["episodes"] == 3
    assert 0.0 <= metrics["final_win_rate"] <= 1.0
    assert isinstance(metrics["final_avg_reward"], float)


def test_tier_c_nfsp_train_minimal(tmp_path):
    """NFSP 3에피소드 학습이 크래시 없이 완료되어야 한다."""
    from rl_agent.tier_c_trainer import TierCTrainer, TierCConfig

    config = TierCConfig(
        algorithm="nfsp",
        episodes=3,
        max_steps=5,
        seed=42,
        log_interval=10,
        save_interval=100,
        checkpoint_dir=str(tmp_path),
    )
    trainer = TierCTrainer(config)
    metrics = trainer.train()

    assert metrics["algorithm"] == "nfsp"
    assert metrics["episodes"] == 3
    assert 0.0 <= metrics["final_win_rate"] <= 1.0


def test_tier_c_evaluate_vs_baseline(tmp_path):
    """Tier C vs PPO 비교 평가가 유효한 보고서를 생성해야 한다."""
    from rl_agent.tier_c_trainer import TierCTrainer, TierCConfig

    config = TierCConfig(
        algorithm="rarl",
        episodes=2,
        max_steps=3,
        seed=42,
        checkpoint_dir=str(tmp_path),
    )
    trainer = TierCTrainer(config)
    trainer.train(episodes=2)

    comparison = trainer.evaluate_vs_baseline(n_runs=3)
    assert "tier_c" in comparison
    assert "ppo" in comparison
    assert "delta_win_rate" in comparison
    assert isinstance(comparison["delta_win_rate"], float)


def test_tier_c_save_comparison_report(tmp_path):
    """비교 보고서 JSON 저장이 정상 동작해야 한다."""
    from rl_agent.tier_c_trainer import TierCTrainer

    report = {
        "tier_c": {"win_rate": 0.6, "avg_survival": 0.5},
        "ppo": {"win_rate": 0.55, "avg_survival": 0.45},
        "delta_win_rate": 0.05,
        "tier_c_algorithm": "rarl",
        "n_runs": 30,
    }
    path = str(tmp_path / "comparison.json")
    TierCTrainer.save_comparison_report(report, path)
    assert os.path.isfile(path)

    with open(path) as f:
        data = json.load(f)
    assert data["delta_win_rate"] == pytest.approx(0.05)
