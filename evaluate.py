"""
evaluate.py
===========
AI Combat Optimization System — 평가 스크립트
저장된 체크포인트를 불러와 Monte Carlo 평가 실행
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.reproducibility import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="AI Combat Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Blue Agent 체크포인트 경로")
    parser.add_argument("--monte-carlo", type=int, default=500, help="Monte Carlo 실행 횟수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fog-level", type=str, default="moderate",
                        choices=["clear", "light", "moderate", "heavy", "maximum"])
    parser.add_argument("--benchmark", type=str, default=None, choices=["historical"],
                        help="벤치마크 모드 (historical)")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                        help="벤치마크 시나리오별 반복 실행 횟수")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Monte Carlo 단일 시나리오 최대 스텝")
    parser.add_argument("--no-progress", action="store_true",
                        help="진행률 바 비활성화")
    parser.add_argument("--fast", action="store_true",
                        help="빠른 스모크 평가 프리셋 (monte-carlo=50, max-steps=30)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.fast and args.benchmark is None:
        args.monte_carlo = min(args.monte_carlo, 50)
        args.max_steps = min(args.max_steps, 30)
        print(f"⚡ Fast preset enabled: monte_carlo={args.monte_carlo}, max_steps={args.max_steps}")

    from simulator.lanchester_engine import LanchesterEngine
    from simulator.fog_of_war import FogOfWarFilter, FogLevel
    from rl_agent.blue_agent import BlueAgent
    from evaluation.monte_carlo import MonteCarloEvaluator

    if args.benchmark == "historical":
        from evaluation.historical_benchmark import HistoricalBenchmark

        print(f"\n📚 Historical 벤치마크 시작 (runs={args.benchmark_runs})")
        bench = HistoricalBenchmark(seed=args.seed)
        report = bench.run_all(n_runs=args.benchmark_runs)
        print(report.summary())
        return

    engine = LanchesterEngine(seed=args.seed)
    blue_agent = BlueAgent()

    if args.checkpoint and os.path.exists(args.checkpoint):
        blue_agent.load(args.checkpoint)
        print(f"✅ 체크포인트 로드: {args.checkpoint}")
    else:
        print("ℹ️ 체크포인트 없음 → 랜덤 초기화 에이전트로 평가")

    fog_level_map = {
        "clear": FogLevel.CLEAR, "light": FogLevel.LIGHT,
        "moderate": FogLevel.MODERATE, "heavy": FogLevel.HEAVY,
        "maximum": FogLevel.MAXIMUM
    }
    fog_filter = FogOfWarFilter(fog_level_map[args.fog_level], seed=args.seed)

    print(f"\n📊 Monte Carlo 평가 시작 ({args.monte_carlo} runs, Fog={args.fog_level.upper()}, MaxSteps={args.max_steps})")
    evaluator = MonteCarloEvaluator(n_runs=args.monte_carlo, seed=args.seed)
    report = evaluator.evaluate(
        blue_agent,
        engine,
        fog_filter=fog_filter,
        verbose=True,
        show_progress=not args.no_progress,
        max_steps=args.max_steps,
    )
    evaluator.print_report(report)

    # 목표 지표 달성 여부
    print("\n🎯 목표 지표 달성 여부:")
    checks = [
        ("Blue 승률 ≥ 50%",         report.blue_win_rate >= 0.50),
        ("전략 강건성 ≥ 0.3",        report.strategy_robustness >= 0.30),
        ("평균 병력 절감 ≥ 15%",     report.avg_force_reduction >= 0.15),
    ]
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")


if __name__ == "__main__":
    main()
