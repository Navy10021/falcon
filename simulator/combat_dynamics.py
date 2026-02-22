"""
combat_dynamics.py
==================
ONTOLOGY ROADMAP STEP 8: 현실 전투 역학 모델

포함 내용:
  - AmmoConsumptionModel  : 교전당 탄약 소모 (UnitType별)
  - BattleDamageAssessment: 전투피해평가 (BDA) — 안개전쟁 불확실성 반영
  - SupplyStatusManager   : 보급/군수 상태 추적 및 전투력 감쇄
  - ElectromagneticEnv    : 전자기 환경 (EMS) — EW·GPS 교란 효과
  - CombatDynamicsManager : 위 4개를 통합 관리 (파이프라인 연동)

파이프라인 연동:
  LanchesterEngine.run_step() 결과 → CombatDynamicsManager.update()
  → 탄약 소모/보급 감소/EMS 적용 → RL 보상 함수에 반영

학술 연구용 합성 데이터
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, UnitStatus, ForceAlignment, CombatKnowledgeGraph, Capability,
)


# ──────────────────────────────────────────────────────────────
# 1. 탄약 소모 모델
# ──────────────────────────────────────────────────────────────

# UnitType → {탄종: 1교전당 소모량}
AMMO_CONSUMPTION: Dict[str, Dict[str, float]] = {
    "infantry":            {"small_arms": 200, "grenade": 3,   "atgm": 0.3},
    "armor":               {"main_gun": 5,      "coax": 100,   "smoke": 4},
    "artillery":           {"155mm": 40,        "illu": 5},
    "aviation":            {"rocket": 14,       "atgm": 2},
    "engineer":            {"explosive": 5},
    "signal":              {},
    "logistics":           {},
    "electronic_warfare":  {},
    # 육군 신규
    "mechanized_infantry": {"small_arms": 180, "30mm": 60,  "atgm": 0.5},
    "airborne":            {"small_arms": 160, "grenade": 4},
    "special_forces":      {"small_arms": 100, "explosive": 3, "atgm": 0.2},
    "anti_armor":          {"atgm": 2,         "small_arms": 50},
    "air_defense_artillery":{"sam": 2,         "gun": 100},
    "reconnaissance":      {"small_arms": 80},
    "nbc_defense":         {},
    "military_police":     {"small_arms": 60},
    # 해군
    "destroyer":           {"ssm": 2,    "sam": 4,   "gun_5in": 50},
    "frigate":             {"ssm": 2,    "sam": 2,   "gun_76mm": 80},
    "submarine":           {"torpedo": 1,"cruise_missile": 2},
    "mine_warfare":        {"mine": 10},
    "amphibious_ship":     {},
    "naval_aviation":      {"aim120": 2, "aim9x": 2, "jdam": 4},
    "coastal_defense":     {"ssm": 3,   "gun": 40},
    # 공군
    "fighter":             {"aim120": 2, "aim9x": 2, "jdam": 2},
    "strike_aircraft":     {"jdam": 6,  "sdb": 8,   "agm": 2},
    "bomber":              {"jdam": 16, "cruise_missile": 4},
    "transport_aircraft":  {},
    "isr_aircraft":        {},
    "early_warning":       {},
    "air_refueling":       {"fuel_tons": 30},
    "sam_battery":         {"sam": 4},
    # 해병대
    "marine_infantry":     {"small_arms": 190, "grenade": 3, "atgm": 0.3},
    "marine_armor":        {"main_gun": 5,     "coax": 100},
    "marine_aviation":     {"aim120": 2,       "agm": 3, "jdam": 4},
    # 미사일
    "ballistic_missile":   {"missile": 1},
    "cruise_missile":      {"missile": 2},
    "mlrs":                {"rocket": 12},
    # 드론
    "uav_recon":           {"sensor_flight_hr": 4},
    "uav_strike":          {"missile": 2,    "bomb": 2},
    "loitering_munition":  {"munition": 1},
    # 사이버/우주
    "cyber_unit":          {},
    "space_asset":         {},
}

# 탄약 소모량을 ammo_level 비율 감소로 변환하는 계수
AMMO_LEVEL_DECAY_PER_ENGAGEMENT: Dict[str, float] = {
    "infantry":            0.02,  "armor":              0.03,
    "artillery":           0.05,  "aviation":           0.06,
    "engineer":            0.01,  "signal":             0.00,
    "logistics":           0.00,  "electronic_warfare": 0.00,
    "mechanized_infantry": 0.025, "airborne":           0.02,
    "special_forces":      0.015, "anti_armor":         0.04,
    "air_defense_artillery":0.05, "reconnaissance":     0.01,
    "nbc_defense":         0.00,  "military_police":    0.01,
    "destroyer":           0.04,  "frigate":            0.03,
    "submarine":           0.05,  "mine_warfare":       0.03,
    "amphibious_ship":     0.00,  "naval_aviation":     0.07,
    "coastal_defense":     0.04,
    "fighter":             0.07,  "strike_aircraft":    0.08,
    "bomber":              0.10,  "transport_aircraft": 0.00,
    "isr_aircraft":        0.00,  "early_warning":      0.00,
    "air_refueling":       0.00,  "sam_battery":        0.05,
    "marine_infantry":     0.02,  "marine_armor":       0.03,
    "marine_aviation":     0.07,
    "ballistic_missile":   0.50,  "cruise_missile":     0.30,
    "mlrs":                0.08,
    "uav_recon":           0.01,  "uav_strike":         0.15,
    "loitering_munition":  1.00,  # 1발 = 유닛 소진
    "cyber_unit":          0.00,  "space_asset":        0.00,
}


@dataclass
class AmmoConsumptionModel:
    """탄약 소모 모델"""

    @staticmethod
    def compute_ammo_decay(unit: Unit, n_engagements: int = 1) -> float:
        """교전 횟수에 따른 ammo_level 감소량 반환"""
        decay_rate = AMMO_LEVEL_DECAY_PER_ENGAGEMENT.get(unit.unit_type.value, 0.02)
        return float(decay_rate * n_engagements)

    @staticmethod
    def get_consumption_report(unit: Unit, n_engagements: int = 1) -> Dict[str, float]:
        """소모 탄약 상세 (학술 시각화용)"""
        base = AMMO_CONSUMPTION.get(unit.unit_type.value, {})
        return {k: v * n_engagements for k, v in base.items()}

    @staticmethod
    def apply_ammo_consumption(unit: Unit, n_engagements: int = 1) -> None:
        """unit.capability.ammo_level 직접 감소"""
        decay = AmmoConsumptionModel.compute_ammo_decay(unit, n_engagements)
        unit.capability.ammo_level = float(np.clip(
            unit.capability.ammo_level - decay, 0.0, 1.0
        ))

    @staticmethod
    def firepower_penalty_from_ammo(unit: Unit) -> float:
        """탄약 부족에 의한 화력 감쇄 계수 [0.3, 1.0]"""
        ammo = unit.capability.ammo_level
        if ammo >= 0.5:
            return 1.0
        elif ammo >= 0.2:
            return 0.7
        else:
            return 0.3


# ──────────────────────────────────────────────────────────────
# 2. 전투피해평가 (BDA)
# ──────────────────────────────────────────────────────────────

@dataclass
class BattleDamageAssessment:
    """
    전투 피해 평가

    confidence_level: 안개전쟁에 의한 BDA 불확실성 [0, 1]
      - 1.0: 완전 정확 (ISR 자산 풍부)
      - 0.5: 중간 (FOW 보통)
      - 0.2: 낮음 (고강도 EW 환경)
    """
    target_unit_id:      str
    strike_type:         str        # "direct_fire" | "indirect" | "air_strike" | "missile" | "cyber"
    pre_strike_strength: float      # 타격 전 전투력
    post_strike_strength:float      # 타격 후 전투력
    killed_in_action:    int
    wounded_in_action:   int
    equipment_destroyed: int
    equipment_damaged:   int
    confidence_level:    float = 0.7  # BDA 신뢰도

    @property
    def damage_ratio(self) -> float:
        return 1.0 - (self.post_strike_strength / max(self.pre_strike_strength, 1e-6))

    @property
    def bda_uncertainty(self) -> Tuple[float, float]:
        """(최소 손실 추정, 최대 손실 추정) — 신뢰도 기반"""
        spread = (1.0 - self.confidence_level) * 0.4
        return (
            float(np.clip(self.damage_ratio - spread, 0.0, 1.0)),
            float(np.clip(self.damage_ratio + spread, 0.0, 1.0)),
        )

    def to_dict(self) -> Dict:
        lo, hi = self.bda_uncertainty
        return {
            "target": self.target_unit_id,
            "strike_type": self.strike_type,
            "damage_ratio": round(self.damage_ratio, 3),
            "kia": self.killed_in_action,
            "wia": self.wounded_in_action,
            "eq_destroyed": self.equipment_destroyed,
            "eq_damaged": self.equipment_damaged,
            "confidence": round(self.confidence_level, 2),
            "est_damage_range": f"[{lo:.2f}, {hi:.2f}]",
        }


class BDAEngine:
    """BDA 생성 엔진 (시뮬레이터 교전 결과 → BDA)"""

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        target: Unit,
        losses: int,
        strike_type: str = "direct_fire",
        isr_quality: float = 0.7,          # ISR 자산 품질 → BDA 신뢰도
    ) -> BattleDamageAssessment:
        pre = target.combat_power + (losses / max(target.headcount + losses, 1)) * 0.3
        post = target.combat_power

        # KIA/WIA 비율 (strike_type별 차이)
        kia_ratio = {"direct_fire": 0.3, "indirect": 0.25, "air_strike": 0.35,
                     "missile": 0.40, "cyber": 0.0}.get(strike_type, 0.25)
        kia = int(losses * kia_ratio)
        wia = losses - kia

        # 장비 피해 (병력 손실에 비례)
        hc = max(target.headcount + losses, 1)
        eq_scale = losses / hc
        eq_destroyed = int(eq_scale * 5 * (1 + self.rng.uniform(-0.2, 0.2)))
        eq_damaged    = int(eq_scale * 8 * (1 + self.rng.uniform(-0.2, 0.2)))

        # 신뢰도 = ISR 품질 + 노이즈
        confidence = float(np.clip(isr_quality + self.rng.normal(0, 0.1), 0.1, 1.0))

        return BattleDamageAssessment(
            target_unit_id=target.unit_id,
            strike_type=strike_type,
            pre_strike_strength=pre,
            post_strike_strength=post,
            killed_in_action=kia,
            wounded_in_action=wia,
            equipment_destroyed=eq_destroyed,
            equipment_damaged=eq_damaged,
            confidence_level=confidence,
        )


# ──────────────────────────────────────────────────────────────
# 3. 보급/군수 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class SupplyStatus:
    """부대 보급 상태"""
    unit_id:              str
    ammo_days:            float = 5.0   # 잔여 탄약 (일수 기준)
    fuel_days:            float = 5.0   # 잔여 연료
    food_days:            float = 7.0   # 잔여 식량
    medical_capacity:     float = 1.0   # 의무 지원 능력 [0, 1]
    resupply_route_open:  bool  = True  # 보급선 개방 여부

    def combat_sustainability(self) -> float:
        """전투 지속 가능 일수 (최소값 기준)"""
        return min(self.ammo_days, self.fuel_days, self.food_days)

    def combat_power_multiplier(self) -> float:
        """보급 상태에 의한 전투력 배수 [0.3, 1.0]"""
        sustainability = self.combat_sustainability()
        if not self.resupply_route_open:
            sustainability *= 0.5
        if sustainability >= 3.0:   return 1.0
        elif sustainability >= 1.5: return 0.75
        elif sustainability >= 0.5: return 0.5
        else:                       return 0.3


class SupplyStatusManager:
    """군수 상태 관리자"""

    def __init__(self):
        self.statuses: Dict[str, SupplyStatus] = {}

    def initialize_unit(self, unit: Unit) -> SupplyStatus:
        ss = SupplyStatus(
            unit_id=unit.unit_id,
            ammo_days=unit.capability.ammo_level * 5.0,   # ammo_level → 일수
            fuel_days=unit.capability.fuel_level * 5.0,
            food_days=7.0,
            medical_capacity=1.0,
            resupply_route_open=True,
        )
        self.statuses[unit.unit_id] = ss
        return ss

    def initialize_from_kg(self, kg: CombatKnowledgeGraph) -> None:
        for unit in kg.units.values():
            self.initialize_unit(unit)

    def consume_per_step(
        self,
        unit: Unit,
        n_engagements: int = 1,
        step_days: float = 0.1,   # 1스텝 = 0.1일 (약 2.4시간)
    ) -> SupplyStatus:
        """1 시뮬레이션 스텝당 보급 소모"""
        ss = self.statuses.get(unit.unit_id)
        if ss is None:
            ss = self.initialize_unit(unit)

        ammo_consume = AMMO_LEVEL_DECAY_PER_ENGAGEMENT.get(
            unit.unit_type.value, 0.02) * n_engagements * 5.0 * step_days
        fuel_consume = 0.05 * step_days  # 기본 연료 소모
        food_consume = step_days

        ss.ammo_days = max(0.0, ss.ammo_days - ammo_consume)
        ss.fuel_days = max(0.0, ss.fuel_days - fuel_consume)
        ss.food_days = max(0.0, ss.food_days - food_consume)

        # ammo_level 동기화
        unit.capability.ammo_level = float(np.clip(ss.ammo_days / 5.0, 0.0, 1.0))
        unit.capability.fuel_level = float(np.clip(ss.fuel_days / 5.0, 0.0, 1.0))

        return ss

    def apply_resupply(self, unit_id: str, ammo: float = 2.0,
                       fuel: float = 2.0, food: float = 3.0) -> None:
        """재보급 적용"""
        if unit_id in self.statuses:
            ss = self.statuses[unit_id]
            ss.ammo_days = min(ss.ammo_days + ammo, 5.0)
            ss.fuel_days = min(ss.fuel_days + fuel, 5.0)
            ss.food_days = min(ss.food_days + food, 7.0)

    def cut_supply_line(self, unit_id: str) -> None:
        """보급선 차단"""
        if unit_id in self.statuses:
            self.statuses[unit_id].resupply_route_open = False

    def restore_supply_line(self, unit_id: str) -> None:
        if unit_id in self.statuses:
            self.statuses[unit_id].resupply_route_open = True

    def get_multiplier(self, unit_id: str) -> float:
        ss = self.statuses.get(unit_id)
        return ss.combat_power_multiplier() if ss else 1.0


# ──────────────────────────────────────────────────────────────
# 4. 전자기 환경 (EMS)
# ──────────────────────────────────────────────────────────────

@dataclass
class ElectromagneticEnv:
    """
    전자기 환경 상태

    ew_intensity      : 전자전 강도 [0, 1] (양측 통합)
    gps_degradation   : GPS 저하 수준 [0, 1]
    comms_jamming     : 통신 재밍 [0, 1]
    radar_range_factor: 레이더 탐지 거리 배수 (EW 영향, 0.2~1.0)
    """
    ew_intensity:       float = 0.0
    gps_degradation:    float = 0.0
    comms_jamming:      float = 0.0
    radar_range_factor: float = 1.0   # 1.0 = 정상, 0.2 = 심각 저하

    def apply_to_unit(self, unit: Unit) -> None:
        """EMS 효과를 유닛 capability에 즉시 적용 (in-place)"""
        # 통신 품질 저하
        jam_effect = self.comms_jamming * (1.0 - unit.capability.ew_resistance)
        unit.capability.comms_quality = float(np.clip(
            unit.capability.comms_quality * (1.0 - jam_effect), 0.0, 1.0
        ))
        # ISR 능력 저하 (GPS 기반 유닛)
        if hasattr(unit.capability, "isr_capability"):
            unit.capability.isr_capability = float(np.clip(
                unit.capability.isr_capability * (1.0 - self.gps_degradation * 0.5),
                0.0, 1.0
            ))

    def effective_range_km(self, unit: Unit) -> float:
        """EW 환경을 고려한 실효 사거리"""
        gps_precision_loss = 0.0
        # GPS 유도 무기 의존 유닛: 정밀도 저하 → 유효 사거리 감소
        gps_dependent = {
            "ballistic_missile", "cruise_missile", "sam_battery",
            "fighter", "strike_aircraft", "bomber",
        }
        if unit.unit_type.value in gps_dependent:
            gps_precision_loss = self.gps_degradation * 0.3  # 최대 30% 사거리 감소
        return unit.capability.range_km * (1.0 - gps_precision_loss) * self.radar_range_factor

    def gnn_uncertainty_multiplier(self) -> float:
        """GNN 불확실성 증폭 배수 (EW 강도 기반)"""
        return float(np.clip(1.0 + self.ew_intensity * 0.8, 1.0, 2.0))

    def to_state_vector(self) -> np.ndarray:
        """RL 상태 벡터 확장용 (4D)"""
        return np.array([
            self.ew_intensity, self.gps_degradation,
            self.comms_jamming, self.radar_range_factor,
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# 5. 통합 관리자
# ──────────────────────────────────────────────────────────────

class CombatDynamicsManager:
    """
    전투 역학 통합 관리자

    파이프라인 연동:
      1. 시뮬레이터 스텝 완료 후 step_update() 호출
      2. BDA 생성 → 로그 기록
      3. 탄약/연료 소모 → RL 보상 함수의 보급 패널티 계산
      4. EMS 상태 → GNN 불확실성 증폭에 반영
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.ammo_model  = AmmoConsumptionModel()
        self.bda_engine  = BDAEngine(seed=seed)
        self.supply_mgr  = SupplyStatusManager()
        self.ems_env     = ElectromagneticEnv()
        self.bda_history: List[BattleDamageAssessment] = []
        self._step: int = 0

    def initialize_from_kg(self, kg: CombatKnowledgeGraph) -> None:
        self.supply_mgr.initialize_from_kg(kg)

    def step_update(
        self,
        kg: CombatKnowledgeGraph,
        step_result: Dict,
        isr_quality: float = 0.7,
        n_engagements_per_unit: int = 1,
    ) -> Dict:
        """
        1 시뮬레이션 스텝 완료 후 호출.

        Parameters
        ----------
        step_result : MixedLanchesterEngine.run_step_mixed() 반환값
        isr_quality : 아군 ISR 자산 품질 (BDA 신뢰도에 반영)

        Returns
        -------
        dict with keys: bda_list, supply_summary, ems_state
        """
        self._step += 1
        new_bdas: List[BattleDamageAssessment] = []

        # ── BDA 생성 ──────────────────────────────────────────
        for result in step_result.get("results", []):
            for uid, losses in [
                (result.defender_id, result.defender_losses),
                (result.attacker_id, result.attacker_losses),
            ]:
                if uid in kg.units and losses > 0:
                    bda = self.bda_engine.generate(
                        kg.units[uid], losses,
                        strike_type="direct_fire",
                        isr_quality=isr_quality,
                    )
                    new_bdas.append(bda)
                    self.bda_history.append(bda)

        # ── 탄약/연료 소모 ────────────────────────────────────
        supply_summary: Dict[str, float] = {}
        for unit in kg.units.values():
            if unit.status == UnitStatus.DESTROYED:
                continue
            ss = self.supply_mgr.consume_per_step(unit, n_engagements_per_unit)
            self.ems_env.apply_to_unit(unit)
            supply_summary[unit.unit_id] = ss.combat_sustainability()

        # ── EMS 상태 업데이트 (EW 유닛 활동에 따라) ──────────
        ew_blue = sum(
            u.combat_power for u in kg.units.values()
            if u.unit_type in {UnitType.ELECTRONIC_WARFARE, UnitType.CYBER_UNIT}
            and u.alignment == ForceAlignment.BLUE
            and u.status != UnitStatus.DESTROYED
        )
        ew_red = sum(
            u.combat_power for u in kg.units.values()
            if u.unit_type in {UnitType.ELECTRONIC_WARFARE, UnitType.CYBER_UNIT}
            and u.alignment == ForceAlignment.RED
            and u.status != UnitStatus.DESTROYED
        )
        total_ew = ew_blue + ew_red + 1e-6
        self.ems_env.ew_intensity     = float(np.clip(total_ew / 10.0, 0.0, 1.0))
        self.ems_env.comms_jamming    = float(np.clip(ew_red / total_ew * 0.5, 0.0, 0.8))
        self.ems_env.gps_degradation  = float(np.clip(ew_red / total_ew * 0.4, 0.0, 0.7))
        self.ems_env.radar_range_factor = float(np.clip(
            1.0 - ew_red / total_ew * 0.6, 0.2, 1.0
        ))

        return {
            "bda_list":       [b.to_dict() for b in new_bdas],
            "supply_summary": supply_summary,
            "ems_state":      self.ems_env.to_state_vector().tolist(),
            "gnn_unc_multiplier": self.ems_env.gnn_uncertainty_multiplier(),
        }

    def get_supply_penalty(self, alignment: ForceAlignment,
                            kg: CombatKnowledgeGraph) -> float:
        """
        보급 패널티 [-1, 0] → RL 보상 함수에 가산
        탄약/연료가 부족한 유닛 비율에 비례
        """
        units = [u for u in kg.units.values()
                 if u.alignment == alignment and u.status != UnitStatus.DESTROYED]
        if not units:
            return 0.0
        penalty = 0.0
        for u in units:
            mult = self.supply_mgr.get_multiplier(u.unit_id)
            if mult < 1.0:
                penalty += (1.0 - mult)
        return -float(penalty / len(units))

    def get_ems_summary(self) -> Dict:
        return {
            "ew_intensity":      round(self.ems_env.ew_intensity, 3),
            "gps_degradation":   round(self.ems_env.gps_degradation, 3),
            "comms_jamming":     round(self.ems_env.comms_jamming, 3),
            "radar_range_factor":round(self.ems_env.radar_range_factor, 3),
            "gnn_unc_multiplier":round(self.ems_env.gnn_uncertainty_multiplier(), 3),
        }

    def generate_bda_report(self, last_n: int = 10) -> str:
        lines = [f"=== BDA 보고 (최근 {last_n}건) ==="]
        for bda in self.bda_history[-last_n:]:
            lo, hi = bda.bda_uncertainty
            lines.append(
                f"  [{bda.target_unit_id}] {bda.strike_type}: "
                f"피해율={bda.damage_ratio:.2f}  "
                f"KIA={bda.killed_in_action}  WIA={bda.wounded_in_action}  "
                f"신뢰도={bda.confidence_level:.2f} "
                f"추정범위=[{lo:.2f},{hi:.2f}]"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine

    print("=== STEP 8: 전투 역학 모델 테스트 ===")

    kg = ScenarioFactory.create_joint_scenario(seed=42)
    engine = MixedLanchesterEngine(seed=42)
    dynamics = CombatDynamicsManager(seed=42)
    dynamics.initialize_from_kg(kg)

    print(f"초기 EMS: {dynamics.get_ems_summary()}")

    total_bda = 0
    for step in range(5):
        result = engine.run_step_mixed(kg)
        info = dynamics.step_update(kg, result, isr_quality=0.7)
        total_bda += len(info["bda_list"])
        ems = info["ems_state"]
        print(f"  스텝{step+1}: BDA={len(info['bda_list'])}건  "
              f"EW={ems[0]:.2f}  GPS저하={ems[1]:.2f}  "
              f"Blue보급={dynamics.get_supply_penalty(ForceAlignment.BLUE, kg):.3f}")
        if result["mission_status"] != "ongoing":
            break

    print(dynamics.generate_bda_report(last_n=5))
    print(f"총 BDA 기록: {total_bda}건")
    print("✅ STEP 8 전투 역학 모델 테스트 완료!")
