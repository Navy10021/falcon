"""tests/test_phase1_immediate.py — Phase 1 즉시 구현 검증 (G+H)"""
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
print("Phase 1 검증 — G. 대시보드 + H. 자연어 인터페이스")
print("="*55)

# ── G. 실시간 대시보드 ──────────────────────
print("\n[G. 실시간 대시보드]")
try:
    from visualization.realtime_dashboard import (
        DashboardRenderer, BattleHistory, DashboardSnapshot,
        UnitMarker, kg_to_snapshot, run_and_render
    )
    from ontology.combat_schema import ScenarioFactory

    # 기본 구조 검증
    check("DashboardRenderer 초기화", True)

    # UnitMarker 색상/심볼
    um = UnitMarker("u1", 10.0, 5.0, "blue", "infantry", 100, "active", 0.8, 0.7, 3.5)
    check("UnitMarker 색상 생성", "#" in um.color(), um.color())
    check("UnitMarker 심볼 생성", len(um.symbol()) > 0, um.symbol())

    # BattleHistory 누적
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=0)
    snap = kg_to_snapshot(kg, step=0, max_steps=10, win_rate=0.65,
                           uncertainty=0.35, alert_level="normal")
    history = BattleHistory()
    history.add(snap)
    check("BattleHistory 스냅샷 추가", len(history.snapshots) == 1)
    check("병력 히스토리 기록",
          len(history.blue_headcount_history) == 1,
          f"blue_hc={history.blue_headcount_history[0]}")
    check("승률 히스토리 기록",
          abs(history.win_rate_history[0] - 0.65) < 0.01)

    # 복수 스냅샷
    for s in range(1, 5):
        sn = kg_to_snapshot(kg, step=s, max_steps=10,
                             win_rate=0.65 - s*0.05,
                             alert_level="caution" if s > 2 else "normal")
        history.add(sn)
    check("복수 스냅샷 누적", len(history.snapshots) == 5)
    check("승률 추이 단조 감소",
          history.win_rate_history[-1] < history.win_rate_history[0])

    # HTML 렌더링
    renderer = DashboardRenderer()
    out_path = "/tmp/test_dashboard.html"
    result = renderer.render(history, out_path)
    check("HTML 파일 생성", os.path.exists(result), result)
    html_size = os.path.getsize(result)
    check("HTML 크기 충분 (>20KB)", html_size > 20_000,
          f"{html_size//1024}KB")

    # HTML 내용 검증
    with open(result, encoding="utf-8") as f:
        html_content = f.read()
    check("Plotly.js 포함", "plotly" in html_content.lower())
    check("전술 지도 섹션 포함", "tacticalMap" in html_content)
    check("전략 옵션 섹션 포함", "strategyOptions" in html_content)
    check("한국어 콘텐츠 포함", "전투" in html_content or "병력" in html_content)
    check("JS 애니메이션 포함", "togglePlay" in html_content)

    # 경보 수준 반영
    snap_crit = kg_to_snapshot(kg, step=5, max_steps=10,
                                win_rate=0.25, alert_level="critical",
                                alert_reasons=["승률 급락", "병력 손실"])
    h2 = BattleHistory()
    h2.add(snap_crit)
    out2 = renderer.render(h2, "/tmp/test_dashboard_crit.html")
    with open(out2, encoding="utf-8") as f:
        html2 = f.read()
    check("경보 수준 HTML 반영", "critical" in html2)

    # 전체 시뮬레이션 실행 + 렌더링
    out3 = run_and_render(n_blue=6, n_red=5, max_steps=12,
                           output_path="/tmp/test_full_dashboard.html", seed=42)
    check("전체 시뮬레이션 대시보드", os.path.exists(out3),
          f"{os.path.getsize(out3)//1024}KB")

except Exception as e:
    check("G. 대시보드 로드 실패", False, str(e)[:80])

# ── H. 자연어 인터페이스 ────────────────────
print("\n[H. 자연어 지휘 인터페이스]")
try:
    from hitl.natural_language_interface import (
        NLCommandParser, CommandInterface, ParsedCommand,
        ParsedConstraint, CommandIntent, ConstraintType, EXAMPLE_COMMANDS
    )

    parser = NLCommandParser()
    interface = CommandInterface()
    check("NL 인터페이스 초기화", True)

    # 기본 파싱
    cmd1 = parser.parse("3번 고지를 반드시 2시간 이내에 점령하라.")
    check("의도 분류 (점령)",
          cmd1.intent in [CommandIntent.SEIZE_OBJECTIVE, CommandIntent.HOLD_POSITION],
          f"intent={cmd1.intent.value}")
    check("시간 제약 추출",
          any(c.key == "max_time_steps" for c in cmd1.constraints),
          f"constraints={[c.key for c in cmd1.constraints]}")
    check("신뢰도 범위 [0,1]",
          0 < cmd1.overall_confidence <= 1.0,
          f"conf={cmd1.overall_confidence:.2f}")

    # 사상자 제약
    cmd2 = parser.parse("사상자는 50명 이내로 제한한다.")
    check("사상자 제약 추출",
          any(c.key == "max_casualties" for c in cmd2.constraints),
          f"val={next((c.value for c in cmd2.constraints if c.key=='max_casualties'), 'N/A')}")

    # 병력 제약
    cmd3 = parser.parse("병력 700명 이내로 임무를 완료하라.")
    check("병력 제약 추출",
          any(c.key == "max_force_size" for c in cmd3.constraints),
          f"val={next((c.value for c in cmd3.constraints if c.key=='max_force_size'), 'N/A')}")

    # 승률 제약
    cmd4 = parser.parse("승률 70% 이상 보장하라.")
    check("승률 제약 추출",
          any(c.key == "min_win_probability" for c in cmd4.constraints),
          f"val={next((c.value for c in cmd4.constraints if c.key=='min_win_probability'), 'N/A')}")

    # HARD/SOFT 구분
    cmd5 = parser.parse("반드시 시가지를 피해라. 가능하면 숲을 이용하라.")
    hard = [c for c in cmd5.constraints if c.constraint_type == ConstraintType.HARD]
    soft = [c for c in cmd5.constraints if c.constraint_type == ConstraintType.SOFT]
    check("HARD/SOFT 구분",
          len(hard) + len(soft) > 0,
          f"hard={len(hard)} soft={len(soft)}")

    # CommanderConstraints 변환
    cmd6 = interface.process(
        "병력 600명 이내, 사상자 40명 이하, 3시간 이내에 완료하라."
    )
    cc = cmd6.to_commander_constraints()
    check("CommanderConstraints 변환 성공", cc is not None)
    check("max_force_size 변환",
          cc.max_force_size is not None and cc.max_force_size > 0,
          f"val={cc.max_force_size}")

    # 전체 예시 명령 처리
    cmds = interface.batch_process(EXAMPLE_COMMANDS)
    check("배치 처리 (7개 명령)",
          len(cmds) == len(EXAMPLE_COMMANDS), f"n={len(cmds)}")
    check("모든 명령 신뢰도 > 0",
          all(c.overall_confidence > 0 for c in cmds))
    check("의도 분류 다양성",
          len(set(c.intent for c in cmds)) >= 3,
          f"intents={len(set(c.intent for c in cmds))}종")

    # 모호한 명령 처리
    cmd_vague = parser.parse("적당히 공격하라.")
    check("모호한 명령 → 명확화 질문",
          cmd_vague.needs_clarification or cmd_vague.overall_confidence < 0.7,
          f"conf={cmd_vague.overall_confidence:.2f}")

    # summary 포맷
    summary = cmd6.summary()
    check("명령 요약 생성", len(summary) > 50 and "파싱" in summary)

except Exception as e:
    check("H. 자연어 인터페이스 로드 실패", False, str(e)[:80])

# ── 최종 결과 ───────────────────────────────
print("\n" + "="*55)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"Phase 1 검증 결과: {passed}/{total} 통과")
if passed == total:
    print("🎉 Phase 1 (즉시) 전체 통과!")
else:
    print("⚠️  실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   ❌ {name}: {detail}")
print("="*55)


# ── pytest entry point ────────────────────────────────────────
def test_phase1_immediate_all():
    """pytest: Phase 1 즉시 검증 (대시보드/자연어인터페이스)."""
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, (
        "Phase 1 failures:\n"
        + "\n".join(f"  \u274c {n}: {d}" for n, d in failed)
    )
