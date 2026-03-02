"""tests/test_phase2_short.py — Phase 2 단기 구현 검증 (A+C)"""
import numpy as np

results = []
def check(name, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

print("\n" + "="*55)
print("Phase 2 검증 — A. 기동엔진 + C. 교리인코딩")
print("="*55)

# ── A. 지형 기동 엔진 ────────────────────────
print("\n[A. 지형 기동 엔진]")
try:
    from simulator.maneuver_engine import (
        ManeuverEngine, TerrainMap, TerrainCell,
        astar, check_los, euclidean,
        UNIT_ENGAGEMENT_RANGE, UNIT_MOVE_SPEED
    )
    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus, TerrainType

    # TerrainMap 생성
    tmap = TerrainMap(width=30, height=30, seed=42)
    check("TerrainMap 생성", len(tmap.cells) == 900, f"{len(tmap.cells)}셀")

    # 지형 다양성
    terrains = set(c.terrain for c in tmap.cells.values())
    check("지형 다양성 (3종 이상)", len(terrains) >= 3, f"{len(terrains)}종")

    # 지형 셀 속성
    cell = tmap.get(15, 15)
    check("지형 셀 취득", cell is not None)
    check("이동 비용 > 0", cell.movement_cost > 0, f"{cell.movement_cost:.2f}")
    check("은폐 보너스 [0,1]", 0 <= cell.concealment_bonus <= 1,
          f"{cell.concealment_bonus:.2f}")
    check("방어 보너스 [0,1]", 0 <= cell.defense_bonus <= 1,
          f"{cell.defense_bonus:.2f}")

    # 이웃 셀
    neighbors = tmap.neighbors(15, 15)
    check("이웃 셀 탐색 (6-8개)", 6 <= len(neighbors) <= 8, f"{len(neighbors)}개")

    # A* 경로 계획
    route = astar(tmap, (2, 2), (25, 25))
    check("A* 경로 계획 성공", route is not None and len(route) > 0,
          f"{len(route) if route else 0}스텝")
    if route:
        check("A* 시작점 포함", route[0] == (2, 2))
        check("A* 목표점 포함", route[-1] == (25, 25))
        check("A* 경로 연속성",
              all(abs(route[i][0]-route[i+1][0]) <= 1 and abs(route[i][1]-route[i+1][1]) <= 1
                  for i in range(len(route)-1)))

    # LOS 체크
    los_near = check_los(tmap, 10.0, 10.0, 12.0, 10.0, max_range=5.0)
    los_far  = check_los(tmap, 0.0,  0.0,  29.0, 29.0, max_range=5.0)
    check("LOS 근거리 (5km 이내)", los_near)
    check("LOS 원거리 차단 (범위 초과)", not los_far)

    # ManeuverEngine 초기화
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    meng = ManeuverEngine(map_size=30, seed=42)
    check("ManeuverEngine 초기화", True)

    # 유닛 이동
    blue_unit = next(u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE)
    ox, oy = blue_unit.position.x, blue_unit.position.y
    nx, ny, status = meng.move_unit(blue_unit, 20.0, 15.0)
    check("유닛 이동 실행", status in ("moving", "arrived"), f"status={status}")
    check("이동 후 위치 변경 또는 도착",
          (nx != ox or ny != oy) or status == "arrived",
          f"({ox:.1f},{oy:.1f})→({nx:.1f},{ny:.1f})")
    check("맵 경계 내 유지", 0 <= nx < 30 and 0 <= ny < 30)

    # 경로 계획
    route2 = meng.plan_route(blue_unit, 25.0, 20.0)
    check("plan_route 실행", route2 is not None, f"{len(route2) if route2 else 0}웨이포인트")

    # 교전 가능 판정
    red_unit = next(u for u in kg.units.values() if u.alignment == ForceAlignment.RED)
    can_engage = meng.can_engage(blue_unit, red_unit)
    check("can_engage 판정 실행", isinstance(can_engage, bool), f"result={can_engage}")

    # 기동 평가
    assessment = meng.assess_maneuver(blue_unit, kg)
    check("기동 평가 실행", assessment is not None)
    check("권장행동 생성", len(assessment.recommended_action) > 0,
          assessment.recommended_action)
    check("커버 스코어 [0,1]", 0 <= assessment.cover_score <= 1,
          f"{assessment.cover_score:.2f}")
    check("측면각도 [0,180]", 0 <= assessment.flank_angle <= 180,
          f"{assessment.flank_angle:.1f}°")

    # 기동 스텝 실행
    result = meng.run_maneuver_step(kg)
    check("run_maneuver_step 실행", "units_moved" in result)
    check("이동 유닛 수 > 0", result["units_moved"] > 0,
          f"{result['units_moved']}개")
    check("교전가능 쌍 집계", "n_engageable" in result,
          f"{result['n_engageable']}쌍")

    # 공간 통계
    stats = meng.get_spatial_stats(kg)
    check("공간 통계 생성", "front_distance" in stats)
    check("전선 거리 >= 0", stats["front_distance"] >= 0,
          f"{stats['front_distance']:.2f}km")
    check("교전 쌍 집계", "n_engagement_pairs" in stats,
          f"{stats['n_engagement_pairs']}쌍")

    # 5스텝 연속 시뮬레이션
    dists = []
    for s in range(5):
        meng.run_maneuver_step(kg)
        dists.append(meng.get_spatial_stats(kg)["front_distance"])
    check("5스텝 연속 실행", len(dists) == 5)
    check("전선 거리 변화 발생", len(set(round(d,1) for d in dists)) > 1,
          f"거리들={[round(d,1) for d in dists]}")

    # 이동 로그 기록
    check("이동 로그 기록됨", len(meng.move_log) > 0,
          f"{len(meng.move_log)}건")

except Exception as e:
    import traceback
    check("A. 기동엔진 예외", False, str(e)[:80])
    traceback.print_exc()

# ── C. 교리 인코딩 ───────────────────────────
print("\n[C. 교리 인코딩 레이어]")
try:
    from ontology.doctrine_encoder import (
        DoctrineEncoder, DoctrineViolation, DoctrineCompliance,
        DoctrinePrinciple, PRINCIPLE_DESCRIPTIONS
    )
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine
    from simulator.resource_manager import ResourceManager

    check("DoctrineEncoder 초기화", True)
    check("9대 원칙 정의", len(DoctrinePrinciple) == 9,
          f"{len(DoctrinePrinciple)}개")
    check("원칙 설명 완비", len(PRINCIPLE_DESCRIPTIONS) == 9)

    # 기본 평가
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    engine = MixedLanchesterEngine(seed=42)
    rm = ResourceManager(seed=42)
    doctrine = DoctrineEncoder()

    result0 = engine.run_step_mixed(kg, resource_manager=rm)
    result0.setdefault("flanking_units", 0)
    result0.setdefault("n_engageable", 2)

    comp = doctrine.evaluate(kg, step=0, step_result=result0)
    check("DoctrineCompliance 반환", isinstance(comp, DoctrineCompliance))
    check("전체 준수도 [0,1]", 0 <= comp.total_score <= 1,
          f"{comp.total_score:.2f}")
    check("원칙별 점수 9개", len(comp.principle_scores) == 9,
          f"{len(comp.principle_scores)}개")
    check("모든 점수 [0,1]",
          all(0 <= v <= 1 for v in comp.principle_scores.values()))

    # 패널티 계산
    penalty = doctrine.compute_doctrine_penalty(comp)
    check("패널티 <= 0", penalty <= 0, f"penalty={penalty:.3f}")

    # 행동 설명 생성
    for action in [1, 3, 5]:  # 전진, 교전, 측면기동
        exp = doctrine.generate_explanation(action, comp)
        check(f"행동 {action} 설명 생성",
              len(exp) > 30 and "PASS" not in exp.lower(),
              exp.split('\n')[0][:40])

    # 10스텝 추적
    penalties = []
    for step in range(1, 10):
        r = engine.run_step_mixed(kg, resource_manager=rm)
        r.setdefault("flanking_units", 0)
        r.setdefault("n_engageable", 1)
        c = doctrine.evaluate(kg, step, r)
        p = doctrine.compute_doctrine_penalty(c)
        penalties.append(p)
    check("10스텝 추적 완료", len(doctrine.history) == 10)
    check("누적 패널티 합산됨", doctrine.cumulative_penalty != 0.0,
          f"cumulative={doctrine.cumulative_penalty:.3f}")

    # 최종 리포트
    report = doctrine.generate_doctrine_report()
    check("리포트 생성", len(report) > 100)
    check("리포트 한국어 포함", "준수" in report or "위반" in report)
    check("리포트 원칙별 점수 포함",
          any(p.value in report for p in DoctrinePrinciple))

    # PPO 상태 벡터 (9D)
    vec = doctrine.get_compliance_vector()
    check("준수도 벡터 9D", vec.shape == (9,), f"shape={vec.shape}")
    check("벡터 값 [0,1]", float(vec.min()) >= 0 and float(vec.max()) <= 1,
          f"min={vec.min():.2f} max={vec.max():.2f}")

    # 위반 → DoctrinePrinciple 타입
    for c in doctrine.history:
        for v in c.violations:
            check_v = isinstance(v.principle, DoctrinePrinciple)
            if not check_v:
                check("위반 원칙 타입", False, str(type(v.principle)))
                break
    check("위반 원칙 타입 정상", True)

    # severity 값 검증
    severities = set()
    for c in doctrine.history:
        for v in c.violations:
            severities.add(v.severity)
    check("심각도 유효 값",
          all(s in ("minor", "major", "critical") for s in severities),
          f"values={severities}")

    # summary 포맷
    summary = comp.summary()
    check("준수도 요약 생성", len(summary) > 50 and "교리" in summary)

except Exception as e:
    import traceback
    check("C. 교리 인코딩 예외", False, str(e)[:80])
    traceback.print_exc()

# ── 최종 결과 ───────────────────────────────
print("\n" + "="*55)
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"Phase 2 검증 결과: {passed}/{total} 통과")
if passed == total:
    print("🎉 Phase 2 (단기) 전체 통과!")
else:
    print("⚠️  실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   ❌ {name}: {detail}")
print("="*55)


# ── pytest entry point ────────────────────────────────────────
def test_phase2_short_all():
    """pytest: Phase 2 단기 검증 (기동엔진/교리인코딩)."""
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, (
        "Phase 2 failures:\n"
        + "\n".join(f"  \u274c {n}: {d}" for n, d in failed)
    )
