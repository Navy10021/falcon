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


def main():
    parser = argparse.ArgumentParser(description="AI Combat Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Blue Agent 체크포인트 경로")
    parser.add_argument("--monte-carlo", type=int, default=500, help="Monte Carlo 실행 횟수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fog-level", type=str, default="moderate",
                        choices=["clear", "light", "moderate", "heavy", "maximum"])
    args = parser.parse_args()

    from simulator.lanchester_engine import LanchesterEngine
    from simulator.fog_of_war import FogOfWarFilter, FogLevel
    from rl_agent.blue_agent import BlueAgent
    from evaluation.monte_carlo import MonteCarloEvaluator

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

    print(f"\n📊 Monte Carlo 평가 시작 ({args.monte_carlo} runs, Fog={args.fog_level.upper()})")
    evaluator = MonteCarloEvaluator(n_runs=args.monte_carlo, seed=args.seed)
    report = evaluator.evaluate(blue_agent, engine, fog_filter=fog_filter, verbose=True)
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
