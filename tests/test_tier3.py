"""tests/test_tier3.py — TIER 3 통합 검증 (⑨~⑫)"""
import numpy as np

results = []
def check(name, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

print("\n" + "="*55)
print("TIER 3 통합 검증")
print("="*55)

# ⑨ 계층적 RL
print("\n[⑨ 계층적 RL]")
try:
    from rl_agent.hierarchical_rl import (
        HierarchicalRLCoordinator, OperationalObjective,
        NumpyOperationalAgent, NumpyTacticalAgent,
        OBJECTIVE_TACTICAL_WEIGHTS, N_OP_ACTIONS
    )
    from ontology.combat_schema import ScenarioFactory

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    coord = HierarchicalRLCoordinator(decision_interval=3, seed=42)
    check("계층적 RL 코디네이터 초기화", True)
    check("작전 목표 종류", N_OP_ACTIONS == 6, f"N={N_OP_ACTIONS}")
    check("전술 가중치 테이블", len(OBJECTIVE_TACTICAL_WEIGHTS) == 6)

    # 작전 상태 빌드
    op_state = coord.build_operational_state(kg, step=3, max_steps=20)
    sv = op_state.to_vector()
    check("작전 상태 벡터 20D", sv.shape == (20,), f"shape={sv.shape}")

    # 목표 선택
    manager = coord.manager
    obj = manager.select_objective(op_state, force_update=True)
    check("작전 목표 선택", obj in list(OperationalObjective), f"obj={obj.value}")

    # 전술 가중치
    weights = manager.get_tactical_weights()
    check("전술 가중치 정규화", abs(weights.sum() - 1.0) < 0.01,
          f"sum={weights.sum():.4f}")

    # 전술 에이전트
    tac = NumpyTacticalAgent(seed=42)
    local_obs = np.random.randn(24).astype(np.float32)
    mask = np.ones(8, dtype=bool)
    action, lp = tac.select_action(local_obs, weights, mask)
    check("전술 행동 선택 유효", 0 <= action < 8, f"action={action}")

    # 내재적 보상
    intr = tac.compute_intrinsic_reward(action, weights)
    check("내재적 보상 >= 0", intr >= 0, f"intr={intr:.3f}")

    # 코디네이터 스텝
    result = coord.step(kg, step=0, max_steps=20)
    check("코디네이터 스텝 실행", "operational_objective" in result)
    check("전술 행동 선택 (전 유닛)",
          result["n_blue_acting"] > 0, f"n={result['n_blue_acting']}")

    # 15 스텝 실행
    prev_obj = None
    obj_changes = 0
    for s in range(15):
        r = coord.step(kg, step=s, max_steps=20)
        if r["operational_objective"] != prev_obj:
            obj_changes += 1
            prev_obj = r["operational_objective"]
    stats = coord.get_stats()
    check("목표 변경 발생", stats["n_objective_changes"] >= 1,
          f"changes={stats['n_objective_changes']}")

    # 보상 계산
    from rl_agent.hierarchical_rl import OperationalState
    prev_s = coord.build_operational_state(kg, 5, 20)
    curr_s = coord.build_operational_state(kg, 10, 20)
    op_reward = manager.compute_operational_reward({}, prev_s, curr_s)
    check("작전 보상 계산 (실수)", isinstance(op_reward, float), f"r={op_reward:.3f}")

except Exception as e:
    check("⑨ 계층적 RL 로드 실패", False, str(e))

# ⑩ League Self-Play
print("\n[⑩ League Self-Play]")
try:
    from rl_agent.league_selfplay import (
        LeagueManager, LeagueAgent, AgentRole,
        PFSPScheduler
    )
    league = LeagueManager(n_main=2, n_main_exp=2, n_league_exp=2,
                            snapshot_interval=5, seed=42)
    check("League 초기화",
          len(league.agents) == 6, f"agents={len(league.agents)}")
    check("초기 스냅샷 생성", len(league.snapshots) >= 2,
          f"snaps={len(league.snapshots)}")

    # 대전 스케줄
    blue, red = league.schedule_match()
    check("대전 쌍 스케줄", blue.agent_id != red.agent_id,
          f"{blue.agent_id} vs {red.agent_id}")

    # ELO 업데이트
    elo_before = blue.elo_rating
    league.record_match_result(blue, red, result=0.8)
    check("ELO 업데이트 (승리→증가)", blue.elo_rating > elo_before,
          f"{elo_before:.0f}→{blue.elo_rating:.0f}")

    # PFSP 스케줄러
    pfsp = PFSPScheduler(mode="hard", seed=42)
    focal = list(league.agents.values())[0]
    candidates = list(league.agents.values())[1:]
    opponent = pfsp.select_opponent(focal, candidates)
    check("PFSP 상대 선택", opponent.agent_id != focal.agent_id,
          f"opp={opponent.agent_id}")

    # 30 경기 시뮬레이션
    rng = np.random.RandomState(42)
    for _ in range(30):
        b, r = league.schedule_match()
        league.record_match_result(b, r, float(rng.beta(2, 2)))

    # 다양성 지표
    diversity = league.compute_diversity_index()
    check("다양성 지표 >= 0", diversity >= 0, f"diversity={diversity:.4f}")

    nash_gap = league.nash_gap_estimate()
    check("Nash Gap 계산", 0 <= nash_gap <= 1.0, f"gap={nash_gap:.4f}")

    # 스냅샷 추가 확인
    check("스냅샷 누적", len(league.snapshots) >= 2,
          f"snaps={len(league.snapshots)}")

    # 역할별 에이전트
    roles = {a.role for a in league.agents.values()}
    check("역할 다양성", AgentRole.MAIN in roles and AgentRole.MAIN_EXP in roles)

except Exception as e:
    check("⑩ League Self-Play 로드 실패", False, str(e))

# ⑪ 실시간 재계획 HITL
print("\n[⑪ 실시간 재계획 HITL]")
try:
    from hitl.realtime_replanner import (
        RealtimeReplanner, AlertLevel, ReplanTrigger,
        BattleSnapshot
    )
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine
    from simulator.resource_manager import ResourceManager

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=6, seed=42)
    replanner = RealtimeReplanner(cooldown_steps=2, seed=42)
    engine = MixedLanchesterEngine(seed=42)
    rm = ResourceManager(seed=42)
    check("재계획 시스템 초기화", True)

    # 트리거 조건 테스트 (승률 급락)
    trigger = ReplanTrigger(
        win_rate_drop_threshold=0.10,
        cooldown_steps=0
    )
    from ontology.combat_schema import ForceAlignment, UnitStatus
    blues = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
    total_hc = sum(u.headcount for u in blues)
    snap1 = BattleSnapshot(1, 0.70, total_hc, total_hc, 3.0, 3.0, 0.8, 0.8, 400, "ongoing")
    snap2 = BattleSnapshot(2, 0.50, total_hc, total_hc, 2.5, 3.0, 0.8, 0.8, 380, "ongoing")
    snap3 = BattleSnapshot(3, 0.40, total_hc, total_hc, 2.0, 3.0, 0.8, 0.8, 360, "ongoing")
    for s in [snap1, snap2, snap3]:
        alert, reasons = trigger.evaluate(s)
    check("승률 급락 트리거", alert != AlertLevel.NORMAL,
          f"alert={alert.value}")

    # 실시간 재계획 스텝
    replan_triggered = False
    for step in range(20):
        result = engine.run_step_mixed(kg, resource_manager=rm)
        win_rate = 0.70 - step * 0.04 + np.random.normal(0, 0.02)
        info = replanner.process_step(kg, result, max(0, win_rate))
        if info is not None:
            replan_triggered = True
            check("재계획 정보 구조",
                  "alert_level" in info and "emergency_options" in info)
            check("긴급 전략 후보 생성",
                  len(info["emergency_options"]) > 0,
                  f"n={len(info['emergency_options'])}")
            break

    check("재계획 트리거 발동", replan_triggered)

    summary = replanner.get_summary()
    check("재계획 요약", "total_replans" in summary and "adoption_rate" in summary,
          f"total={summary['total_replans']}")

    # 경보 수준 열거형
    check("경보 수준 정의",
          all(lvl in AlertLevel for lvl in
              [AlertLevel.NORMAL, AlertLevel.WARNING, AlertLevel.CRITICAL]))

except Exception as e:
    check("⑪ 실시간 재계획 로드 실패", False, str(e))

# ⑫ 멀티 도메인
print("\n[⑫ 멀티 도메인 확장]")
try:
    from ontology.multidomain import (
        MultiDomainAnalyzer, MultiDomainState,
        DomainType, DOMAIN_UNITS, DOMAIN_EFFECT_TABLE
    )
    from ontology.combat_schema import ScenarioFactory

    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    analyzer = MultiDomainAnalyzer(seed=42)
    check("멀티 도메인 분석기 초기화", True)
    check("도메인 분류 테이블",
          len(DOMAIN_UNITS) >= 3 and "ground" in DOMAIN_UNITS)
    check("도메인 효과 테이블", len(DOMAIN_EFFECT_TABLE) >= 3)

    # 도메인 균형
    balance = analyzer.compute_domain_balance(kg)
    check("도메인 균형 계산",
          all(d in balance for d in ["ground", "air", "ew"]),
          f"domains={list(balance.keys())}")
    check("균형 수치 범위",
          all(-1 <= v["superiority"] <= 1 for v in balance.values()))

    # 멀티 도메인 상태
    md_state = analyzer.compute_multidomain_state(kg)
    check("멀티 도메인 상태 생성",
          isinstance(md_state, MultiDomainState))
    ext_vec = md_state.to_extension_vector()
    check("확장 벡터 4D", ext_vec.shape == (4,), f"shape={ext_vec.shape}")

    # 도메인 효과 적용
    multipliers = analyzer.apply_domain_effects(kg, md_state)
    check("도메인 배수 생성",
          len(multipliers) >= 6 and all(v > 0 for v in multipliers.values()),
          f"n={len(multipliers)}")

    # 불확실성 증폭
    unc_mult = analyzer.get_uncertainty_multiplier(md_state)
    check("불확실성 증폭 [0.8, 1.8]",
          0.8 <= unc_mult <= 1.8, f"×{unc_mult:.3f}")

    # 5 스텝 후 트렌드
    from simulator.mixed_lanchester import MixedLanchesterEngine
    engine = MixedLanchesterEngine(seed=42)
    for _ in range(5):
        analyzer.compute_multidomain_state(kg)
        engine.run_step_mixed(kg)
    trends = analyzer.domain_trend(5)
    check("트렌드 분석", len(trends) >= 2, f"trends={list(trends.keys())}")

    # 도메인 보고서
    report = analyzer.generate_domain_report(md_state)
    check("도메인 보고서 생성",
          "멀티 도메인" in report and len(report) > 50)

except Exception as e:
    check("⑫ 멀티 도메인 로드 실패", False, str(e))

# 결과 출력
print("\n" + "="*55)
total = len(results)
passed = sum(1 for _,ok,_ in results if ok)
print(f"TIER 3 검증 결과: {passed}/{total} 통과")
if passed == total:
    print("🎉 TIER 3 전체 통과!")
else:
    print("⚠️  실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   ❌ {name}: {detail}")
print("="*55)


# ── pytest entry point ────────────────────────────────────────
def test_tier3_all():
    """pytest: TIER 3 통합 검증 (계층적RL/LeagueSelfPlay/실시간재계획/멀티도메인)."""
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, (
        "TIER 3 failures:\n"
        + "\n".join(f"  \u274c {n}: {d}" for n, d in failed)
    )
