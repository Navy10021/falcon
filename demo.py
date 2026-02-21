"""
demo.py
=======
AI Combat Optimization System — 전체 파이프라인 데모
Phase 1, 2, 3 통합 시연
"""

import argparse
import os
import sys
import numpy as np
import torch
import time

sys.path.insert(0, os.path.dirname(__file__))


def separator(title: str = "", char: str = "═", width: int = 65):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{char * pad} {title} {char * pad}")
    else:
        print(char * width)


def run_demo(args):
    print("\n")
    separator("🧠 AI Combat Optimization System v2.0 — DEMO")
    print("\n  Ontology-Driven GNN + RL for Force Minimization")
    print("  학술 연구용 가상 전투 데이터 기반\n")
    separator()
    time.sleep(0.5)

    # ──────────────────────────────────────────────
    # Step 1: 온톨로지 및 시나리오 생성
    # ──────────────────────────────────────────────
    separator("STEP 1: 전장 온톨로지 & 시나리오 생성")

    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
    from ontology.temporal_extension import TemporalStateTracker

    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=args.seed)
    stats = kg.get_stats()

    print(f"\n✅ Knowledge Graph 생성 완료:")
    print(f"   전체 노드: {stats['total_nodes']}개, 엣지: {stats['total_edges']}개")
    print(f"   Blue 유닛: {stats['blue_units']}개 ({stats['blue_headcount']}명)")
    print(f"   Red 유닛:  {stats['red_units']}개 ({stats['red_headcount']}명)")
    print(f"   Blue 전투력: {stats['blue_combat_power']:.3f}")
    print(f"   Red 전투력:  {stats['red_combat_power']:.3f}")

    tracker = TemporalStateTracker()
    tracker.record_snapshot(kg.units, events=["시나리오 초기화"])
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 2: Fog of War
    # ──────────────────────────────────────────────
    separator("STEP 2: Fog of War (부분 관측 환경)")

    from simulator.fog_of_war import FogOfWarFilter, FogLevel, CurriculumScheduler

    print("\n📊 Fog Level별 관측 비교:")
    for level in [FogLevel.CLEAR, FogLevel.MODERATE, FogLevel.MAXIMUM]:
        fog = FogOfWarFilter(level, seed=args.seed)
        obs_kg, unc_map = fog.observe(kg, ForceAlignment.BLUE)
        obs_stats = obs_kg.get_stats()
        avg_unc = np.mean(list(unc_map.values())) if unc_map else 0.0
        print(f"   [{level.name:8s}] 관측 Red={obs_stats['red_units']}개  "
              f"평균 불확실성={avg_unc:.3f}")
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 3: Bayesian GNN
    # ──────────────────────────────────────────────
    separator("STEP 3: Phase 1 — Bayesian GNN (불확실성 예측)")

    from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors
    from gnn_model.uncertainty_utils import risk_action_mapping

    fog_filter = FogOfWarFilter(FogLevel.MODERATE, seed=args.seed)
    obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)
    x, adj = prepare_graph_tensors(obs_kg)

    gnn = BayesianHGT(node_in_dim=32, hidden_dim=64, n_layers=2, mc_samples=15)

    print(f"\n🔮 MC Dropout GNN 예측 (N=15):")
    gnn_out = gnn.predict_with_uncertainty(x, adj)

    cas_mean = gnn_out["casualty_mean"].item()
    cas_std  = gnn_out["casualty_std"].item()
    ci_l = gnn_out["casualty_ci_low"].item()
    ci_h = gnn_out["casualty_ci_high"].item()
    ep_unc = gnn_out["epistemic_uncertainty"].item()
    al_unc = gnn_out["aleatoric_uncertainty"].item()

    print(f"   예상 사상자: {cas_mean:.1f}명 ± {cas_std:.1f}명")
    print(f"   95% CI: [{max(0, ci_l):.1f}, {ci_h:.1f}]명")
    print(f"   위험도: {gnn_out['risk_mean'].item():.3f} ± {gnn_out['risk_std'].item():.3f}")
    print(f"   Epistemic 불확실성: {ep_unc:.4f}  (모델 불확실성)")
    print(f"   Aleatoric 불확실성: {al_unc:.4f}  (데이터 노이즈)")

    action_rec = risk_action_mapping(ep_unc + al_unc)
    print(f"\n   🎯 행동 권고: {action_rec['strategy'].upper()}")
    print(f"   {action_rec['description']}")
    print(f"   병력 조정 배수: {action_rec['force_multiplier']}")
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 4: PPO 에이전트 행동
    # ──────────────────────────────────────────────
    separator("STEP 4: Phase 2 — Blue vs Red PPO 에이전트")

    from rl_agent.blue_agent import BlueAgent, build_state_vector, BlueActionSpace
    from rl_agent.red_agent import RedAgent, RedActionSpace

    blue_agent = BlueAgent()
    red_agent  = RedAgent()
    print("\nℹ️  Demo uses randomly initialized agents unless you load trained checkpoints.")
    print("   Low win rate in this demo run is expected and does not indicate a runtime error.")

    policy_source = "random_init"
    if args.blue_checkpoint and os.path.exists(args.blue_checkpoint):
        blue_agent.load(args.blue_checkpoint)
        policy_source = "checkpoint"
        print(f"\n✅ Blue checkpoint loaded: {args.blue_checkpoint}")
    if args.red_checkpoint and os.path.exists(args.red_checkpoint):
        red_agent.load(args.red_checkpoint)
        policy_source = "checkpoint"
        print(f"✅ Red checkpoint loaded: {args.red_checkpoint}")

    if policy_source == "random_init":
        print("\nℹ️  Demo mode: baseline (random_init policy)")
        print("   Low win rate in this demo run is expected and does not indicate a runtime error.")
    else:
        print("\nℹ️  Demo mode: checkpoint policy")

    gnn_ext = gnn.compute_ppo_state_extension(x, adj)
    state = build_state_vector(obs_kg, gnn_extension=gnn_ext, uncertainty_map=uncertainty_map)

    blue_action, blue_lp, blue_val = blue_agent.select_action(state)
    red_obs_kg, _ = fog_filter.observe(kg, ForceAlignment.RED)
    red_state = build_state_vector(red_obs_kg)
    red_action, red_lp, red_val = red_agent.select_action(red_state)

    print(f"\n🔵 Blue Agent 행동: {BlueActionSpace.ACTION_NAMES[blue_action]}")
    print(f"   log_prob={blue_lp:.4f}, value estimate={blue_val:.4f}")
    print(f"\n🔴 Red Agent 행동: {RedActionSpace.ACTION_NAMES[red_action]}")
    print(f"   log_prob={red_lp:.4f}, value estimate={red_val:.4f}")

    if red_action == RedActionSpace.DECEPTION:
        print("   ⚠️ Red가 기만 전술 실행! GNN 불확실성 증가 예상")
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 5: Lanchester 전투 시뮬레이션
    # ──────────────────────────────────────────────
    separator("STEP 5: 전투 시뮬레이션 (Lanchester 엔진)")

    from simulator.lanchester_engine import LanchesterEngine

    engine = LanchesterEngine(seed=args.seed)
    kg_sim = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=args.seed)
    steps, result = engine.run_episode(kg_sim, max_steps=20)

    print(f"\n⚔️ 에피소드 결과: {result.upper()}")
    print(f"   총 {len(steps)} 스텝")
    total_blue_cas = sum(s.blue_total_casualties for s in steps)
    total_red_cas  = sum(s.red_total_casualties  for s in steps)
    print(f"   Blue 누적 사상: {total_blue_cas}명")
    print(f"   Red 누적 사상:  {total_red_cas}명")
    print(f"   최종 Blue 병력: {steps[-1].blue_total_headcount}명")
    print(f"   최종 Red 병력:  {steps[-1].red_total_headcount}명")
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 6: HITL — Pareto 전략 후보
    # ──────────────────────────────────────────────
    separator("STEP 6: Phase 3 — Human-in-the-Loop (HITL)")

    from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints
    from hitl.preference_learner import CommanderPreferenceLearner, SelectionRecord
    from hitl.constraint_parser import ConstraintParser

    print("\n📋 지휘관 제약 조건 파싱:")
    constraint_text = "가용 병력 800명 이하로, 승률 65% 이상 유지. 포병 지원 선호."
    parser = ConstraintParser()
    constraints = parser.parse_text(constraint_text)
    constraints.prefer_artillery = True
    print(f"   입력: '{constraint_text}'")
    print(f"   파싱 결과: 최대병력={constraints.max_force_size}, "
          f"최소승률={constraints.min_win_probability:.0%}, "
          f"포병선호={constraints.prefer_artillery}")

    gnn_output = {
        "total_uncertainty": (ep_unc + al_unc),
        "casualty_mean": cas_mean
    }
    generator = ParetoStrategyGenerator(n_candidates=4)
    options = generator.generate(kg, constraints, gnn_output=gnn_output)

    print(f"\n{generator.display_options(options)}")

    # 지휘관 선호도 학습
    learner = CommanderPreferenceLearner("commander_alpha")
    if options:
        selected = options[1]  # 균형 옵션 선택 시뮬레이션
        record = SelectionRecord(
            scenario_id=0,
            options_presented=[o.option_id for o in options],
            selected_option_id=selected.option_id,
            selected_type=selected.strategy_type.value,
            force_size=selected.force_size,
            win_probability=selected.win_probability,
            expected_casualties=selected.expected_casualties
        )
        learner.record_selection(record)
        print(learner.get_insight_report())
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 7: Explainability
    # ──────────────────────────────────────────────
    separator("STEP 7: Explainability — 결정 근거 보고서")

    from explainability.attention_viz import generate_decision_report

    attention_scores = {
        "Urban terrain":         0.71,
        "Enemy artillery":       0.68,
        "Low ammo state":        0.54,
        "Adjacent friendly":     0.42,
        "River obstacle":        0.31,
        "EW interference":       0.28,
    }
    report = generate_decision_report(
        unit_id="blue_3",
        unit_type="armor",
        terrain="urban",
        attention_scores=attention_scores,
        uncertainty={"epistemic": ep_unc, "aleatoric": al_unc},
        recommended_action=BlueActionSpace.ACTION_NAMES[blue_action],
        casualty_prediction={"mean": cas_mean, "std": cas_std}
    )
    print(f"\n{report}")
    time.sleep(0.3)

    # ──────────────────────────────────────────────
    # Step 8: Monte Carlo 평가 (단축)
    # ──────────────────────────────────────────────
    separator("STEP 8: Monte Carlo 강건성 평가 (50 runs)")

    from evaluation.monte_carlo import MonteCarloEvaluator

    evaluator = MonteCarloEvaluator(n_runs=50, seed=args.seed)
    mc_report = evaluator.evaluate(blue_agent, engine, verbose=True)
    evaluator.print_report(mc_report)

    # ──────────────────────────────────────────────
    # 최종 요약
    # ──────────────────────────────────────────────
    separator("🎯 전체 파이프라인 완료")
    print(f"""
  온톨로지 → GNN → PPO → HITL → 평가 파이프라인 실행 완료!

  📌 핵심 결과:
     Blue 승률:     {mc_report.blue_win_rate:.1%}
     평균 사상자:    {mc_report.avg_blue_casualties:.1f}명
     전략 강건성:    {mc_report.strategy_robustness:.4f}
     불확실성(총):   {ep_unc + al_unc:.4f}
     Pareto 후보:   {len(options)}개 생성 완료
     정책 출처:      {policy_source}

  🔗 GitHub 배포:
     README.md → requirements.txt → 각 모듈 → train.py → evaluate.py → demo.py

  ⚠️  본 시스템은 학술 연구용 합성 데이터 기반입니다.
    """)
    separator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FALCON end-to-end demo")
    parser.add_argument("--blue-checkpoint", type=str, default=None,
                        help="Optional Blue agent checkpoint path")
    parser.add_argument("--red-checkpoint", type=str, default=None,
                        help="Optional Red agent checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    from utils.reproducibility import set_global_seed
    set_global_seed(args.seed)

    run_demo(args)
