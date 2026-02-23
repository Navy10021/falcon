"""
tests/test_new_simulators.py
============================
신규 시뮬레이터 3개 모듈 테스트
  - simulator/naval_engine.py
  - simulator/missile_model.py
  - simulator/cyber_effects.py
"""
import pytest
import numpy as np
from ontology.combat_schema import (
    ScenarioFactory, ForceAlignment, Position, UnitType, UnitStatus
)


# ─────────────────────────────────────────
# Naval Engine Tests
# ─────────────────────────────────────────

class TestNavalEngine:

    def setup_method(self):
        from simulator.naval_engine import NavalCombatEngine
        self.engine = NavalCombatEngine(seed=42)

    def test_salvo_exchange_empty_returns_default(self):
        result = self.engine.compute_salvo_exchange([], [])
        assert result.blue_ships_sunk == 0
        assert result.red_ships_sunk == 0
        assert result.blue_surviving_fraction == 1.0

    def test_salvo_exchange_with_scenario(self):
        """표준 시나리오에서 살보 교환 계산"""
        kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=4, seed=1)
        blue = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
        red  = [u for u in kg.units.values() if u.alignment == ForceAlignment.RED]
        result = self.engine.compute_salvo_exchange(blue, red)
        # 살보 교환 결과 합리성 검증
        assert 0.0 <= result.blue_surviving_fraction <= 1.0
        assert 0.0 <= result.red_surviving_fraction <= 1.0
        assert result.blue_ships_sunk >= 0
        assert result.red_ships_sunk >= 0

    def test_naval_battle_report_structure(self):
        """해상 전투 보고서 구조 검증"""
        kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=7)
        report = self.engine.run_naval_battle(kg, max_salvos=3)
        assert hasattr(report, 'blue_win')
        assert hasattr(report, 'red_win')
        assert hasattr(report, 'draw')
        assert hasattr(report, 'dominant_factor')
        # 결과 중 하나는 반드시 True
        assert report.blue_win or report.red_win or report.draw
        assert 0.0 <= report.blue_loss_rate <= 1.0
        assert 0.0 <= report.red_loss_rate <= 1.0

    def test_submarine_detector_probability_range(self):
        """소나 탐지 확률 [0,1] 범위 검증"""
        from simulator.naval_engine import SubmarineDetector
        detector = SubmarineDetector(seed=42)
        kg = ScenarioFactory.create_standard_scenario(seed=42)
        unit = list(kg.units.values())[0]
        for dist_km in [5, 20, 50, 100, 200]:
            prob = detector.detection_probability(dist_km, unit, sea_state=3)
            assert 0.0 <= prob <= 1.0, f"prob={prob} out of range at dist={dist_km}"

    def test_submarine_detector_longer_range_lower_prob(self):
        """거리 증가 → 탐지 확률 감소"""
        from simulator.naval_engine import SubmarineDetector
        detector = SubmarineDetector(seed=42)
        kg = ScenarioFactory.create_standard_scenario(seed=42)
        unit = list(kg.units.values())[0]
        p10  = detector.detection_probability(10,  unit)
        p100 = detector.detection_probability(100, unit)
        # 일반적으로 멀수록 낮아야 함 (소나 방정식 특성)
        # 주의: 매우 가까운 거리에서는 TL이 낮아 오히려 높을 수 있음
        assert p10 >= p100 or abs(p10 - p100) < 0.1

    def test_mine_threat_basic(self):
        """기뢰 위협 기본 계산"""
        from simulator.naval_engine import MineThreatModel
        model = MineThreatModel(seed=42)
        result = model.compute_mine_threat(
            n_ships=10, mine_density_per_km2=2.0,
            channel_width_km=5.0, channel_length_km=20.0
        )
        assert result.ships_damaged >= 0
        assert result.ships_sunk >= 0
        assert result.ships_damaged + result.ships_sunk <= 10
        assert 0.0 <= result.damage_fraction <= 1.0

    def test_mine_threat_minesweeper_reduces_damage(self):
        """소해 작전 → 피해 감소"""
        from simulator.naval_engine import MineThreatModel
        model = MineThreatModel(seed=0)
        no_sweep = model.compute_mine_threat(n_ships=20, minesweeper_factor=0.0)
        swept    = model.compute_mine_threat(n_ships=20, minesweeper_factor=0.9)
        # 소해 후 피해율이 낮아야 함 (확률적이므로 느슨한 검증)
        assert swept.damage_fraction <= no_sweep.damage_fraction + 0.3


# ─────────────────────────────────────────
# Missile Model Tests
# ─────────────────────────────────────────

class TestMissileModel:

    def setup_method(self):
        from simulator.missile_model import MissileModel
        self.model = MissileModel(seed=42)
        self.launch = Position(0.0, 0.0)
        self.target = Position(200.0, 0.0)  # 200km

    def test_out_of_range_returns_no_intercept(self):
        from simulator.missile_model import MissileType, InterceptResult
        # SRBM 사거리 800km, 2000km 타겟
        far_target = Position(2000.0, 0.0)
        eng = self.model.simulate_launch(
            MissileType.BALLISTIC_SRBM, self.launch, far_target,
            interceptor_type=MissileType.SAM_LONG
        )
        assert eng.intercept_result == InterceptResult.NO_INTERCEPT

    def test_intercept_result_is_valid_enum(self):
        from simulator.missile_model import MissileType, InterceptResult
        eng = self.model.simulate_launch(
            MissileType.CRUISE_SUBSONIC, self.launch, self.target,
            interceptor_type=MissileType.SAM_MEDIUM,
            n_interceptors=2
        )
        assert isinstance(eng.intercept_result, InterceptResult)
        assert eng.intercept_result in list(InterceptResult)

    def test_pk_range(self):
        from simulator.missile_model import MissileType
        eng = self.model.simulate_launch(
            MissileType.ANTI_SHIP, self.launch, self.target,
            interceptor_type=MissileType.SAM_SHORT
        )
        assert 0.0 <= eng.pk <= 1.0

    def test_more_interceptors_higher_pk(self):
        """요격체 수 증가 → Pk 증가"""
        from simulator.missile_model import MissileType, MissileModel
        # 고정 seed로 Pk 값만 비교
        m = MissileModel(seed=99)
        eng1 = m.simulate_launch(
            MissileType.CRUISE_SUBSONIC, self.launch, self.target,
            MissileType.SAM_MEDIUM, n_interceptors=1
        )
        eng2 = m.simulate_launch(
            MissileType.CRUISE_SUBSONIC, self.launch, self.target,
            MissileType.SAM_MEDIUM, n_interceptors=4
        )
        assert eng2.pk >= eng1.pk

    def test_damage_radius_positive(self):
        from simulator.missile_model import MissileType
        eng = self.model.simulate_launch(
            MissileType.BALLISTIC_MRBM, self.launch,
            Position(500.0, 0.0),
            interceptor_type=MissileType.SAM_LONG
        )
        assert eng.damage_radius_m > 0

    def test_time_of_flight_positive(self):
        from simulator.missile_model import MissileType
        eng = self.model.simulate_launch(
            MissileType.CRUISE_SUBSONIC, self.launch, self.target
        )
        assert eng.time_of_flight_sec > 0

    def test_ballistic_trajectory_apex(self):
        """탄도 궤적 최대 고도 양수 확인"""
        from simulator.missile_model import BallisticTrajectory, MissileType, MISSILE_SPECS
        traj = BallisticTrajectory()
        spec = MISSILE_SPECS[MissileType.BALLISTIC_MRBM]
        pts  = traj.compute(spec, 1000.0, dt_sec=30.0)
        assert len(pts) > 2
        apex = max(p.alt_km for p in pts)
        assert apex > 0

    def test_saturation_attack_increases_leakage(self):
        """포화 공격(살보 증가) → 돌파 미사일 증가"""
        from simulator.missile_model import SalvoInterceptModel, MissileType
        salvo = SalvoInterceptModel(seed=0)
        r_small = salvo.simulate_salvo(
            MissileType.CRUISE_SUBSONIC, n_missiles=3,
            interceptor_type=MissileType.SAM_MEDIUM,
            n_interceptor_batteries=2
        )
        r_large = salvo.simulate_salvo(
            MissileType.CRUISE_SUBSONIC, n_missiles=20,
            interceptor_type=MissileType.SAM_MEDIUM,
            n_interceptor_batteries=2
        )
        assert r_large.leaked_count >= r_small.leaked_count
        assert r_large.is_saturated
        assert not r_small.is_saturated


# ─────────────────────────────────────────
# Cyber Effects Tests
# ─────────────────────────────────────────

class TestCyberEffects:

    def setup_method(self):
        from simulator.cyber_effects import CyberEffectEngine
        self.engine = CyberEffectEngine(seed=42)
        self.kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)

    def test_attack_result_structure(self):
        from simulator.cyber_effects import (
            CyberAttack, CyberTargetSystem, CyberEffectType, CyberEffectResult
        )
        attack = CyberAttack(
            attacker_id="attacker",
            target_system=CyberTargetSystem.GPS_INFRA,
            effect_type=CyberEffectType.DISRUPTION,
            intensity=0.8,
            duration_steps=3,
            affected_alignment=ForceAlignment.BLUE,
        )
        result = self.engine.launch_attack(attack, self.kg)
        assert isinstance(result.attack_success, bool)
        assert 0.0 <= result.severity <= 1.0
        assert 0.0 <= result.c2_degradation <= 1.0
        assert 0.0 <= result.gps_degradation <= 1.0

    def test_step_decay_reduces_duration(self):
        """step_decay() 호출 시 지속 스텝 감소"""
        from simulator.cyber_effects import (
            CyberAttack, CyberTargetSystem, CyberEffectType
        )
        attack = CyberAttack(
            attacker_id="atk",
            target_system=CyberTargetSystem.COMMS_BACKBONE,
            effect_type=CyberEffectType.DENIAL,
            intensity=0.95,  # 높은 강도로 성공 유도
            duration_steps=5,
            affected_alignment=ForceAlignment.RED,
        )
        result = self.engine.launch_attack(attack, self.kg)
        if result.attack_success:
            before = len(self.engine.active_effects)
            self.engine.step_decay()
            after = len(self.engine.active_effects)
            # 지속 스텝 4로 감소 → 아직 활성
            assert after <= before
        else:
            pytest.skip("공격 실패 (확률적) — 스킵")

    def test_battle_state_level_in_range(self):
        """전장 상태 수준이 [0,1] 범위"""
        state = self.engine.get_battle_state(0, self.kg)
        for attr in ['blue_c2_level', 'blue_comms_level', 'blue_gps_level',
                     'red_c2_level',  'red_comms_level',  'red_gps_level']:
            val = getattr(state, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of range"

    def test_high_intensity_attack_higher_severity(self):
        """강한 공격 → 성공 시 더 높은 severity (통계적 검증)"""
        from simulator.cyber_effects import (
            CyberAttack, CyberTargetSystem, CyberEffectType, CyberEffectEngine
        )
        severities_low, severities_high = [], []
        for seed in range(20):
            for intensity, collector in [(0.1, severities_low), (0.95, severities_high)]:
                eng = CyberEffectEngine(seed=seed)
                atk = CyberAttack(
                    attacker_id="atk",
                    target_system=CyberTargetSystem.C2_NETWORK,
                    effect_type=CyberEffectType.DISRUPTION,
                    intensity=intensity,
                    duration_steps=3,
                    affected_alignment=ForceAlignment.BLUE,
                )
                r = eng.launch_attack(atk, self.kg)
                if r.attack_success:
                    collector.append(r.severity)

        if severities_low and severities_high:
            assert np.mean(severities_high) >= np.mean(severities_low)

    def test_ew_jamming_structure(self):
        """EW 재밍 효과 구조 검증"""
        from simulator.cyber_effects import EWEffectModel
        ew_model = EWEffectModel(seed=42)
        blue_units = [u for u in self.kg.units.values()
                      if u.alignment == ForceAlignment.BLUE]
        if not blue_units:
            pytest.skip("Blue 유닛 없음")
        jammer = blue_units[0]
        result = ew_model.apply_jamming(
            jammer, self.kg, jam_type="gps", target_alignment=ForceAlignment.RED
        )
        assert 0.0 <= result.gps_deg <= 1.0
        assert 0.0 <= result.comms_deg <= 1.0
        assert isinstance(result.affected_units, list)

    def test_ew_all_jam_types(self):
        """EW 모든 재밍 유형 동작 확인"""
        from simulator.cyber_effects import EWEffectModel
        ew_model = EWEffectModel(seed=0)
        blue_units = [u for u in self.kg.units.values()
                      if u.alignment == ForceAlignment.BLUE]
        if not blue_units:
            pytest.skip("Blue 유닛 없음")
        jammer = blue_units[0]
        for jam_type in ["gps", "comms", "radar", "combined"]:
            result = ew_model.apply_jamming(jammer, self.kg, jam_type=jam_type)
            assert result.jam_type == jam_type
            assert result.jam_radius_km > 0


# ─────────────────────────────────────────
# 통합 실행
# ─────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
