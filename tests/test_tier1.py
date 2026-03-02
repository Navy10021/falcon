"""
tests/test_tier1.py
===================
TIER 1 통합 검증 테스트
① 동적 자원 소모
② 혼합 란체스터
③ Temperature Scaling
④ MC Pareto 검증
"""
import numpy as np

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

# ─────────────────────────────────────
def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

# ═════════════════════════════════════
print("\n" + "="*55)
print("TIER 1 통합 검증")
print("="*55)

# ① 동적 자원 소모
print("\n[① 동적 자원 소모]")
try:
    from ontology.combat_schema import ScenarioFactory, ForceAlignment
    from simulator.resource_manager import ResourceManager, AMMO_CONSUMPTION_RATE

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=0)
    rm = ResourceManager(seed=0)
    blue_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]

    # 초기 탄약 저장
    init_ammo = {u.unit_id: u.capability.ammo_level for u in blue_units}

    # 5 스텝 소모
    for _ in range(5):
        for u in blue_units:
            from ontology.combat_schema import UnitStatus
            if u.status != UnitStatus.DESTROYED:
                rm.consume_ammo(u, n_engagements=1)
                rm.consume_fuel(u, distance_moved_km=1.0)

    # 검증
    ammo_decreased = all(
        u.capability.ammo_level < init_ammo[u.unit_id]
        for u in blue_units
    )
    check("탄약 소모 감소", ammo_decreased)

    fuel_decreased = all(u.capability.fuel_level < 1.0 for u in blue_units)
    check("연료 소모 감소", fuel_decreased)

    # 화력 패널티 적용
    fp_mults = [rm.get_firepower_multiplier(u) for u in blue_units]
    check("화력 패널티 [0,1] 범위", all(0 <= m <= 1.0 for m in fp_mults),
          f"min={min(fp_mults):.2f}")

    # 자동 보급
    supply_events = rm.auto_supply_step(kg)
    check("자동 보급 실행", isinstance(supply_events, list))

    # 요약
    summary = rm.get_summary(kg)
    check("자원 요약 생성", "blue" in summary and "total_supply_events" in summary,
          f"events={summary['total_supply_events']}")

except Exception as e:
    check("① 동적 자원 소모 모듈 로드", False, str(e))

# ② 혼합 란체스터
print("\n[② 혼합 란체스터 법칙]")
try:
    from simulator.mixed_lanchester import MixedLanchesterEngine, CombatMode

    engine = MixedLanchesterEngine(seed=42)
    kg2 = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    blue_u = list(kg2.units.values())[0]
    red_u  = list(kg2.units.values())[-1]

    # 지형별 혼합 비율
    urban_lr, urban_mode = engine.determine_mix_ratio(blue_u, red_u, "urban", 0.5)
    open_lr,  open_mode  = engine.determine_mix_ratio(blue_u, red_u, "open",  0.5)

    check("도시 지형 게릴라 편향", urban_lr < open_lr,
          f"urban={urban_lr:.2f} < open={open_lr:.2f}")
    check("혼합 모드 분류", urban_mode in list(CombatMode),
          f"mode={urban_mode.value}")

    # 포위 판정
    from ontology.combat_schema import Position
    enemies = list(kg2.units.values())[1:5]
    for i, e in enumerate(enemies):
        angle = i * 90
        e.position = Position(
            blue_u.position.x + 2 * np.cos(np.radians(angle)),
            blue_u.position.y + 2 * np.sin(np.radians(angle))
        )
    is_enc, penalty = engine.detect_encirclement(blue_u, enemies)
    check("포위 감지 동작", isinstance(is_enc, bool) and penalty >= 1.0,
          f"encircled={is_enc}, penalty={penalty:.2f}")

    # 시뮬레이션 스텝
    step_result = engine.run_step_mixed(kg2)
    check("혼합 스텝 실행", "mission_status" in step_result
          and "combat_modes" in step_result,
          f"status={step_result['mission_status']}")

    check("전투 양식 기록", len(step_result["combat_modes"]) > 0,
          f"n={len(step_result['combat_modes'])}")

except Exception as e:
    check("② 혼합 란체스터 모듈 로드", False, str(e))

# ③ Temperature Scaling
print("\n[③ GNN Temperature Scaling 보정]")
try:
    from gnn_model.temperature_scaling import (
        TemperatureScaler, IsotonicCalibrator,
        GNNCalibrationPipeline, generate_calibration_dataset
    )
    from gnn_model.uncertainty_utils import compute_ece

    means, stds, actuals = generate_calibration_dataset(300, overconfident=True, seed=42)
    ece_raw = compute_ece(means, stds, actuals)

    pipeline = GNNCalibrationPipeline()
    reports = pipeline.fit_all(means, stds, actuals)

    best_ece = min(r.ece_after for r in reports)
    check("보정 후 ECE 개선", best_ece < ece_raw,
          f"before={ece_raw:.4f} → after={best_ece:.4f}")
    check("ECE < 0.15", best_ece < 0.15, f"ECE={best_ece:.4f}")
    check("Temperature 값 유효", 0.05 < pipeline.temp_scaler.temperature < 10.0,
          f"T={pipeline.temp_scaler.temperature:.3f}")

    cal_mean, cal_std = pipeline.calibrate_single(25.0, 3.0)
    check("단일값 보정 동작", cal_std > 0, f"std: 3.00→{cal_std:.2f}")

    check("최적 방법 선택", pipeline.best_method in
          ["Temperature Scaling", "Isotonic Regression"],
          pipeline.best_method)

except Exception as e:
    check("③ Temperature Scaling 모듈 로드", False, str(e))

# ④ MC Pareto 검증
print("\n[④ MC 기반 Pareto 실검증]")
try:
    from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints
    from hitl.mc_pareto_validator import MCParetoValidator, ValidatedStrategyOption

    kg3 = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    constraints = CommanderConstraints(max_force_size=1500, min_win_probability=0.3)

    gen = ParetoStrategyGenerator(n_candidates=3)
    options = gen.generate(kg3, constraints)
    check("후보 생성", len(options) > 0, f"n={len(options)}")

    # 빠른 검증 (15 sims)
    validator = MCParetoValidator(n_simulations=15, max_steps_per_sim=15, seed=42)
    validated = validator.validate_all(options, kg3, constraints, verbose=False)

    check("검증 후보 수 일치", len(validated) == len(options),
          f"{len(validated)}/{len(options)}")
    check("MC 승률 [0,1] 범위",
          all(0 <= v.mc_win_rate <= 1.0 for v in validated),
          f"rates={[round(v.mc_win_rate,2) for v in validated]}")
    check("신뢰구간 유효",
          all(v.mc_win_rate_ci[0] <= v.mc_win_rate <= v.mc_win_rate_ci[1]
              for v in validated),
          "CI 포함 확인")
    check("Pareto 재계산",
          any(v.is_pareto_dominant for v in validated))

    # 유의차 검정 (2개 이상일 때)
    if len(validated) >= 2:
        sig = validator.significance_test(validated[0], validated[1])
        check("유의차 검정 실행", "p_value" in sig,
              f"p={sig['p_value']:.3f}")

except Exception as e:
    check("④ MC Pareto 검증 모듈 로드", False, str(e))

# ═════════════════════════════════════
print("\n" + "="*55)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"TIER 1 검증 결과: {passed}/{total} 통과")
if passed == total:
    print("🎉 TIER 1 전체 통과!")
else:
    print("⚠️  실패 항목:")
    for name, ok, detail in results:
        if not ok:
            print(f"   ❌ {name}: {detail}")
print("="*55)


# ── pytest entry point ────────────────────────────────────────
def test_tier1_all():
    """pytest: TIER 1 통합 검증 (동적자원/혼합란체스터/Temperature/MCPareto)."""
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert not failed, (
        "TIER 1 failures:\n"
        + "\n".join(f"  \u274c {n}: {d}" for n, d in failed)
    )
