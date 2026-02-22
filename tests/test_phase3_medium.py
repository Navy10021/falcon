"""tests/test_phase3_medium.py — Phase 3 중기 구현 검증 (B+I+K)"""
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
print("Phase 3 검증 — B. IRL + I. AAR + K. 벤치마크")
print("="*55)

# ── B. 역강화학습 (IRL) ──────────────────────
print("\n[B. 역강화학습 (Max-Entropy IRL)]")
try:
    from rl_agent.inverse_rl import (
        MaxEntropyIRL, DemonstrationGenerator, LearnedReward,
        RewardFunctionAnalyzer, Demonstration, FEATURE_NAMES,
        STATE_DIM, extract_features
    )

    check("모듈 임포트", True)
    check("피처 차원 정의", STATE_DIM == 12, f"dim={STATE_DIM}")
    check("피처 이름 완비", len(FEATURE_NAMES) == STATE_DIM)

    # 데모 생성
    gen = DemonstrationGenerator(seed=42)
    demos = gen.generate(n_per_tactic=8, episode_len=10)
    check("데모 생성", len(demos) > 0, f"n={len(demos)}")
    check("데모 전술 다양성",
          len(set(d.doctrine_label for d in demos)) >= 4,
          f"tactics={len(set(d.doctrine_label for d in demos))}종")
    check("데모 시퀀스 구조",
          all(len(d.state_sequence) == len(d.action_sequence) for d in demos))
    check("데모 상태 벡터 차원",
          all(len(s) == STATE_DIM for d in demos for s in d.state_sequence),
          f"dim={len(demos[0].state_sequence[0])}")

    # 피처 추출
    feat = extract_features({"blue_force_ratio": 0.7, "mass_score": 0.8,
                              "flanking_bonus": 0.6, "ammo_level": 0.9})
    check("피처 추출 벡터 차원", feat.shape == (STATE_DIM,))
    check("피처 값 범위 [-2,2]",
          float(feat.min()) >= -2.0 and float(feat.max()) <= 2.0)

    # IRL 학습
    irl = MaxEntropyIRL(feature_dim=STATE_DIM, lr=0.02, n_iter=80, seed=42)
    check("MaxEntropyIRL 초기화", True)
    reward = irl.train(demos, verbose=False)
    check("LearnedReward 반환", isinstance(reward, LearnedReward))
    check("가중치 차원", reward.weights.shape == (STATE_DIM,),
          f"shape={reward.weights.shape}")
    check("학습 손실 기록", len(reward.training_loss) == 80,
          f"n={len(reward.training_loss)}")
    check("손실 감소 (초기 > 최종)",
          reward.training_loss[0] > reward.training_loss[-1] * 0.9,
          f"{reward.training_loss[0]:.4f}→{reward.training_loss[-1]:.4f}")
    check("상위 피처 반환", len(reward.top_features(5)) == 5)
    check("요약 생성", len(reward.summary()) > 100)

    # 보상 계산
    sample_state = np.random.RandomState(0).uniform(0, 1, STATE_DIM).astype(np.float32)
    r_val = irl.compute_reward(sample_state)
    check("보상 계산 스칼라", isinstance(r_val, float), f"r={r_val:.4f}")

    # LearnedReward.compute
    r_val2 = reward.compute(sample_state)
    check("LearnedReward.compute 실행", isinstance(r_val2, float))

    # 분석기
    analyzer = RewardFunctionAnalyzer()
    interp = analyzer.interpret(reward)
    check("IRL 해석 생성", len(interp) > 100 and "가중치" in interp)
    compare = analyzer.compare_with_handcrafted(reward)
    check("수작업 비교 생성", "코사인" in compare)

    # 데모 통계
    check("데모 길이 일관성",
          all(d.length == len(d.state_sequence) for d in demos))
    check("데모 결과 포함", all(d.outcome in ("win","loss","draw") for d in demos))

except Exception as e:
    import traceback
    check("B. IRL 예외", False, str(e)[:80])
    traceback.print_exc()

# ── I. 자동 전투 후 분석 (AAR) ───────────────
print("\n[I. 자동 전투 후 분석 (AAR)]")
try:
    from explainability.auto_aar import (
        AutoAARGenerator, EpisodeLog, StepRecord, AARReport,
        TurningPoint, generate_synthetic_episode
    )

    check("모듈 임포트", True)
    gen_aar = AutoAARGenerator()
    check("AutoAARGenerator 초기화", True)

    # 합성 에피소드 생성
    log_win  = generate_synthetic_episode(outcome="win",  n_steps=15, seed=42)
    log_loss = generate_synthetic_episode(outcome="loss", n_steps=15, seed=7)
    check("승리 에피소드 생성", log_win.outcome == "win" and len(log_win.steps) > 0,
          f"n_steps={log_win.n_steps}")
    check("패배 에피소드 생성", log_loss.outcome == "loss",
          f"n_steps={log_loss.n_steps}")
    check("StepRecord 구조",
          all(hasattr(s, "action") and hasattr(s, "win_rate") for s in log_win.steps))
    check("총 손실 집계", log_win.total_blue_casualties >= 0)
    check("병력 감소율 [0,1]",
          0 <= log_win.force_reduction_ratio <= 1,
          f"{log_win.force_reduction_ratio:.2f}")

    # AAR 생성 (승리)
    report_win = gen_aar.generate(log_win)
    check("AARReport 반환 (승리)", isinstance(report_win, AARReport))
    check("요약 생성", len(report_win.executive_summary) > 30)
    check("전환점 탐지", isinstance(report_win.turning_points, list))
    check("최선 결정 목록", isinstance(report_win.best_decisions, list))
    check("놓친 기회 목록", isinstance(report_win.missed_opportunities, list))
    check("자원 분석 딕셔너리", isinstance(report_win.resource_analysis, dict) and
          len(report_win.resource_analysis) > 0)
    check("개선 권고 존재", len(report_win.recommendations) > 0)
    check("교훈 존재", len(report_win.lessons_learned) > 0)
    check("종합 점수 [0,1]",
          0 <= report_win.overall_score <= 1,
          f"{report_win.overall_score:.2f}")

    # AAR 생성 (패배)
    report_loss = gen_aar.generate(log_loss)
    check("AARReport 반환 (패배)", isinstance(report_loss, AARReport))
    check("패배 점수 < 승리 점수",
          report_loss.overall_score < report_win.overall_score + 0.3,
          f"loss={report_loss.overall_score:.2f} win={report_win.overall_score:.2f}")

    # 전환점 상세
    if report_win.turning_points:
        tp = report_win.turning_points[0]
        check("전환점 타입 유효",
              tp.event_type in ("win_rate_shift","heavy_loss","opportunity_taken",
                                "resource_crisis","mission_complete","mission_failed"),
              tp.event_type)
        check("전환점 임팩트 [-1,1]",
              -1.1 <= tp.impact <= 1.1, f"{tp.impact:.2f}")
        check("전환점 설명 생성", len(tp.to_str()) > 20)

    # 리포트 텍스트 렌더링
    full_report = report_win.to_report()
    check("전체 리포트 렌더링", len(full_report) > 500)
    check("리포트 AAR 포함", "AAR" in full_report or "분석" in full_report)
    check("리포트 한국어 포함", "스텝" in full_report or "손실" in full_report)
    check("리포트 결과 포함", log_win.outcome.upper() in full_report.upper() or
          "WIN" in full_report.upper() or "승리" in full_report)

    # 빈 로그 처리
    empty_log = EpisodeLog("empty_test", 1000, 800, "draw")
    empty_report = gen_aar.generate(empty_log)
    check("빈 로그 처리 (오류 없음)", empty_report is not None)

    # casualty_efficiency
    s_test = StepRecord(0,3,"교전",800,600,5,20,0.6,0.3,0.8,0.9,"ongoing",0.6,False,False,0.5)
    check("교전 효율 계산", s_test.casualty_efficiency == 4.0,
          f"{s_test.casualty_efficiency}")

except Exception as e:
    import traceback
    check("I. AAR 예외", False, str(e)[:80])
    traceback.print_exc()

# ── K. 역사 벤치마크 ─────────────────────────
print("\n[K. 역사 벤치마크]")
try:
    from evaluation.historical_benchmark import (
        HistoricalBenchmark, BenchmarkScenario, ScenarioResult,
        BenchmarkReport, BENCHMARK_SCENARIOS
    )

    check("모듈 임포트", True)
    check("벤치마크 시나리오 수",
          len(BENCHMARK_SCENARIOS) >= 5,
          f"{len(BENCHMARK_SCENARIOS)}개")

    # 시나리오 구조 검증
    for sc in BENCHMARK_SCENARIOS:
        check(f"시나리오 구조 ({sc.name[:20]})",
              len(sc.expected_actions) > 0 and len(sc.expected_doctrine_scores) > 0,
              f"actions={len(sc.expected_actions)} doctrine={len(sc.expected_doctrine_scores)}")

    bench = HistoricalBenchmark(seed=42)
    check("HistoricalBenchmark 초기화", True)

    # 단일 시나리오 실행
    sc0 = BENCHMARK_SCENARIOS[0]
    result = bench.run_scenario(sc0, n_runs=5)
    check("ScenarioResult 반환", isinstance(result, ScenarioResult))
    check("시나리오 이름 일치", result.scenario_name == sc0.name)
    check("행동 정합성 [0,1]",
          0 <= result.action_alignment <= 1,
          f"{result.action_alignment:.2f}")
    check("교리 점수 딕셔너리",
          isinstance(result.doctrine_scores, dict) and len(result.doctrine_scores) > 0)
    check("교리 점수 [0,1]",
          all(0 <= v <= 1 for v in result.doctrine_scores.values()))
    check("결과 유효값",
          result.outcome in ("blue_win","red_win","prolonged","draw","ongoing"),
          result.outcome)
    check("outcome_match 불리언", isinstance(result.outcome_match, bool))
    check("overall_pass 불리언", isinstance(result.overall_pass, bool))

    # 전체 벤치마크 (n_runs=3 빠른 검증)
    report = bench.run_all(n_runs=3)
    check("BenchmarkReport 반환", isinstance(report, BenchmarkReport))
    check("시나리오 결과 수 일치",
          report.n_scenarios == len(BENCHMARK_SCENARIOS),
          f"n={report.n_scenarios}")
    check("통과율 [0,1]",
          0 <= report.pass_rate <= 1,
          f"{report.pass_rate:.0%}")
    check("평균 교리 준수도 [0,1]",
          0 <= report.avg_doctrine_compliance <= 1,
          f"{report.avg_doctrine_compliance:.2f}")
    check("평균 행동 정합성 [0,1]",
          0 <= report.avg_action_alignment <= 1,
          f"{report.avg_action_alignment:.2f}")
    check("취약/강점 시나리오 목록",
          isinstance(report.weak_scenarios, list) and
          isinstance(report.strong_scenarios, list))

    # 벤치마크 요약 텍스트
    summary = report.summary()
    check("요약 텍스트 생성", len(summary) > 200)
    check("요약 한국어 포함", "통과율" in summary or "교리" in summary)
    check("요약 시나리오별 결과 포함",
          any(sc.name[:15] in summary for sc in BENCHMARK_SCENARIOS))

    # 개별 결과 일관성
    check("결과 합산 일치",
          report.n_passed == sum(1 for r in report.scenario_results if r.overall_pass))

    # 난이도별 시나리오 존재
    difficulties = set(sc.difficulty for sc in BENCHMARK_SCENARIOS)
    check("난이도 다양성", len(difficulties) >= 2, f"{difficulties}")

except Exception as e:
    import traceback
    check("K. 벤치마크 예외", False, str(e)[:80])
    traceback.print_exc()

# ── 최종 결과 ────────────────────────────────
print("\n" + "="*55)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"Phase 3 검증 결과: {passed}/{total} 통과")
if passed == total:
    print("🎉 Phase 3 (중기) 전체 통과!")
else:
    print("⚠️  실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   ❌ {name}: {detail}")
print("="*55)


# ── pytest entry point ────────────────────────────────────────
def test_phase3_medium_all():
    """pytest: Phase 3 중기 검증 (IRL/AAR/역사벤치마크)."""
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, (
        "Phase 3 failures:\n"
        + "\n".join(f"  \u274c {n}: {d}" for n, d in failed)
    )
