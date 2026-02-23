"""tests/test_tier2.py — TIER 2 통합 검증 (⑤~⑧)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

results = []
def check(name, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

print("\n" + "="*55)
print("TIER 2 통합 검증")
print("="*55)

# ⑤ Temporal GNN
print("\n[⑤ Temporal GNN]")
try:
    from gnn_model.temporal_gnn import NumpyTemporalGNN, TemporalDataCollector
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=0)
    node_feats = kg.get_node_features()
    edge_idx, _ = kg.get_adjacency_info()
    n = len(node_feats)
    adj = np.zeros((n,n), dtype=np.float32)
    if edge_idx.shape[1] > 0:
        adj[edge_idx[0], edge_idx[1]] = 1.0
    np.fill_diagonal(adj, 1.0)

    # 실제 노드 피처 차원을 읽어 초기화 (하드코딩 32 → 동적 차원)
    actual_node_dim = node_feats.shape[1] if node_feats.ndim == 2 else node_feats.shape[-1]
    tgnn = NumpyTemporalGNN(node_in_dim=actual_node_dim, gnn_embed_dim=64,
                             lstm_hidden_dim=128, window_size=8, seed=42)

    emb = tgnn.encode_graph(node_feats, adj)
    check("그래프 임베딩 생성", emb.shape == (64,), f"shape={emb.shape}")

    engine = MixedLanchesterEngine(seed=42)
    collector = TemporalDataCollector(window_size=8)
    preds = []
    for step in range(12):
        pred = tgnn.step(node_feats, adj)
        emb_s = tgnn.gnn_encoder.encode(node_feats, adj)
        collector.add_step(emb_s, {"casualties": pred["casualty_mean"],
                                    "risk": pred["risk_mean"], "done": step > 9})
        preds.append(pred)
        engine.run_step_mixed(kg)

    check("예측 딕셔너리 키", all(k in preds[-1] for k in
          ["casualty_mean","risk_mean","mission_done_prob","temporal_uncertainty"]))
    check("버퍼 충전", preds[-1]["buffer_fill"] == 8,
          f"fill={preds[-1]['buffer_fill']}")
    check("불확실성 감소 (버퍼 채워질수록)",
          preds[-1]["temporal_uncertainty"] < preds[0]["temporal_uncertainty"],
          f"{preds[0]['temporal_uncertainty']:.3f}→{preds[-1]['temporal_uncertainty']:.3f}")
    state_vec = tgnn.get_temporal_state_vector()
    check("시계열 상태 벡터 8D", state_vec.shape == (8,), f"shape={state_vec.shape}")
    seqs = collector.get_sequences()
    check("학습 시퀀스 수집", len(seqs) >= 4, f"n={len(seqs)}")
    tgnn.reset_state()
    check("LSTM 상태 리셋", np.all(tgnn.h == 0))
except Exception as e:
    check("⑤ Temporal GNN 로드 실패", False, str(e))

# ⑥ MAPPO
print("\n[⑥ MAPPO]")
try:
    from rl_agent.mappo import (MAPPOManager, UNIT_ACTION_MASKS, N_ACTIONS,
                                 ACTION_NAMES, GLOBAL_STATE_DIM)
    from ontology.combat_schema import ScenarioFactory, ForceAlignment

    kg2 = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=0)
    manager = MAPPOManager(seed=42)
    check("MAPPO 초기화", True)
    check("행동 마스크 정의",
          all(len(v) > 0 for v in UNIT_ACTION_MASKS.values()),
          f"{len(UNIT_ACTION_MASKS)}종 유닛")

    actions = manager.select_actions(kg2)
    blue_count = sum(1 for u in kg2.units.values()
                     if u.alignment == ForceAlignment.BLUE
                     and u.status.value != "destroyed")
    check("전체 Blue 행동 선택",
          len(actions) == blue_count, f"{len(actions)}/{blue_count}")
    check("행동 범위 유효",
          all(0 <= a < N_ACTIONS for a,_,_ in actions.values()))
    check("행동 이름 유효",
          all(name in ACTION_NAMES for _,_,name in actions.values()))

    sample_unit = next(u for u in kg2.units.values()
                        if u.alignment == ForceAlignment.BLUE)
    obs = manager.build_unit_observation(sample_unit, kg2)
    check("관측 벡터 24D", obs.obs_vector.shape[0] == 24,
          f"dim={obs.obs_vector.shape[0]}")
    check("행동 마스크 Bool",
          obs.action_mask.dtype == bool and len(obs.action_mask) == N_ACTIONS)

    global_state = np.random.randn(GLOBAL_STATE_DIM).astype(np.float32)
    value = manager.critic.value(global_state)
    check("Critic 가치 함수 실수 반환", isinstance(value, float))
    dist_stats = manager.get_action_distribution_stats(kg2)
    check("행동 분포 통계", len(dist_stats) == len(actions),
          f"n={len(dist_stats)}")
except Exception as e:
    check("⑥ MAPPO 로드 실패", False, str(e))

# ⑦ Bandit 선호도 학습
print("\n[⑦ Bandit 선호도 학습]")
try:
    from hitl.bandit_preference import (BanditPreferenceLearner, SelectionEvent,
                                         StrategyContext, simulate_commander_choices)

    learner = BanditPreferenceLearner("test_cmd", seed=42)
    check("Bandit 초기화", len(learner.beta_arms) == 5)

    ctx = StrategyContext("opt1","minimum_casualty",0.8,0.75,0.05,0.3,0.4,0.3)
    event = SelectionEvent(0,"minimum_casualty",
                           ["minimum_casualty","balanced","maximum_win_rate"],
                           ctx, 0.9)
    learner.record_selection(event)
    check("선택 이벤트 기록", learner.n_selections == 1)
    check("Beta α 증가", learner.beta_arms["minimum_casualty"].alpha > 1.0,
          f"α={learner.beta_arms['minimum_casualty'].alpha:.2f}")

    accs = simulate_commander_choices(learner, n_episodes=30,
                                       true_preference="minimum_casualty", seed=42)
    final_acc = float(np.mean(accs[-10:]))
    check("30ep 후 정확도 >= 0.3", final_acc >= 0.30, f"acc={final_acc:.2f}")

    contexts = [StrategyContext(f"o{i}",st,0.7,0.75,0.05,0.3,0.4,0.3)
                for i,st in enumerate(learner.STRATEGY_TYPES)]
    ranked = learner.rank_options(contexts)
    check("선호도 순위 정렬", len(ranked) == len(contexts),
          f"top={ranked[0][0].strategy_type}")
    rec = learner.recommend(contexts)
    check("추천 전략 유효", rec.strategy_type in learner.STRATEGY_TYPES, rec.strategy_type)
    check("탐색률 감소", learner.exploration_rate < 0.20,
          f"ε={learner.exploration_rate:.3f}")
    check("인사이트 보고서", len(learner.get_insight_report()) > 10)
except Exception as e:
    check("⑦ Bandit 로드 실패", False, str(e))

# ⑧ 반사실 설명
print("\n[⑧ 반사실 설명 생성]")
try:
    from explainability.counterfactual import (CounterfactualAnalyzer,
                                                CounterfactualVariable)
    from ontology.combat_schema import ScenarioFactory

    kg3 = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    analyzer = CounterfactualAnalyzer(n_runs=10, max_steps=15, seed=42)

    var = CounterfactualVariable("force_size","병력 규모",0.0,0.15,True,"both")
    base = analyzer._run_simulation(kg3)
    check("기준 시뮬레이션",
          "win_rate" in base and "casualties" in base,
          f"승률={base['win_rate']:.2f}")

    cf_results = analyzer.analyze_variable(kg3, var, base)
    check("단일 변수 반사실 (up+down)", len(cf_results) == 2,
          f"n={len(cf_results)}")
    nl = cf_results[0].natural_language()
    check("자연어 설명 생성", len(nl) > 10 and "%" in nl, nl[:55])

    variables = [
        CounterfactualVariable("ammo_level","탄약",0.0,0.20,False,"up"),
        CounterfactualVariable("morale","사기",0.0,0.15,False,"down"),
    ]
    all_results = analyzer.full_analysis(kg3, variables, verbose=False)
    check("다중 변수 분석", len(all_results) >= 2, f"n={len(all_results)}")
    importance = analyzer.variable_importance(all_results)
    check("변수 중요도 계산", len(importance) > 0,
          f"top={importance[0][0] if importance else 'N/A'}")
    if len(all_results) >= 2:
        i0 = abs(all_results[0].win_rate_delta)+abs(all_results[0].casualty_delta)/50
        i1 = abs(all_results[1].win_rate_delta)+abs(all_results[1].casualty_delta)/50
        check("영향도 내림차순 정렬", i0 >= i1 - 0.001)
except Exception as e:
    check("⑧ 반사실 로드 실패", False, str(e))

# 결과 출력
print("\n" + "="*55)
total = len(results)
passed = sum(1 for _,ok,_ in results if ok)
print(f"TIER 2 검증 결과: {passed}/{total} 통과")
if passed == total:
    print("🎉 TIER 2 전체 통과!")
else:
    print("⚠️  실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   ❌ {name}: {detail}")
print("="*55)


# ── pytest entry point ────────────────────────────────────────
def test_tier2_all():
    """pytest: TIER 2 통합 검증 (TemporalGNN/MAPPO/Bandit/반사실)."""
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, (
        "TIER 2 failures:\n"
        + "\n".join(f"  \u274c {n}: {d}" for n, d in failed)
    )
