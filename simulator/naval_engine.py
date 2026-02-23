"""
naval_engine.py
===============
해상 전투 역학 엔진 (Naval Combat Engine)

Hughes' Salvo 방정식 기반:
  남은 전력 = 공격력 - α × 적 살보 수
  α: 방어력 계수 (요격 + 방호력)

참고:
  - Wayne P. Hughes Jr., "Fleet Tactics and Coastal Combat" (2000)
  - Salvo Model: B' = B - α·R, R' = R - β·B
    (B, R = Blue/Red 타격 유닛 수; α, β = 방어력 계수)

주요 클래스:
  NavalCombatEngine  : 살보 교환 시뮬레이션 (주)
  SubmarineDetector  : 음파탐지 방정식 (Sonar Equation)
  MineThreatModel    : 기뢰 위협 확률 계산
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ontology.combat_schema import (
    ForceAlignment, Unit, UnitType, UnitStatus, CombatKnowledgeGraph,
)


# ──────────────────────────────────────────────────────────────
# 해상 유닛 분류
# ──────────────────────────────────────────────────────────────

NAVAL_UNIT_TYPES = {
    UnitType.DESTROYER,
    UnitType.FRIGATE,
    UnitType.SUBMARINE,
    UnitType.MINE_WARFARE,
    UnitType.AMPHIBIOUS_SHIP,
    UnitType.NAVAL_AVIATION,
    UnitType.COASTAL_DEFENSE,
}

SUBMARINE_TYPES = {UnitType.SUBMARINE}
SURFACE_TYPES   = {UnitType.DESTROYER, UnitType.FRIGATE,
                   UnitType.AMPHIBIOUS_SHIP, UnitType.MINE_WARFARE}


# ──────────────────────────────────────────────────────────────
# 데이터클래스
# ──────────────────────────────────────────────────────────────

@dataclass
class SalvoResult:
    """단일 살보 교환 결과"""
    blue_salvos_fired: int          # Blue 발사 살보 수
    red_salvos_fired: int           # Red 발사 살보 수
    blue_ships_damaged: float       # Blue 피해 함정 (fractional)
    red_ships_damaged: float        # Red 피해 함정
    blue_ships_sunk: int            # Blue 침몰 함정
    red_ships_sunk: int             # Red 침몰 함정
    blue_surviving_fraction: float  # Blue 생존율 [0, 1]
    red_surviving_fraction: float   # Red 생존율 [0, 1]
    engagement_range_km: float
    salvos_intercepted_blue: int    # Blue가 요격한 적 살보
    salvos_intercepted_red: int     # Red가 요격한 적 살보


@dataclass
class SonarContact:
    """소나 탐지 결과"""
    target_unit_id: str
    detection_probability: float   # [0, 1]
    bearing_error_deg: float       # 방위각 오차 (도)
    range_estimate_km: float       # 추정 거리
    contact_quality: str           # "firm" | "probable" | "possible" | "none"


@dataclass
class MineEngagementResult:
    """기뢰 교전 결과"""
    mines_encountered: int
    ships_damaged: int
    ships_sunk: int
    damage_fraction: float
    channel_cleared: bool


@dataclass
class NavalBattleReport:
    """해상 전투 종합 보고서"""
    total_salvos: int
    blue_initial_ships: int
    red_initial_ships: int
    blue_surviving_ships: int
    red_surviving_ships: int
    blue_win: bool
    red_win: bool
    draw: bool
    blue_loss_rate: float           # Blue 손실율 [0, 1]
    red_loss_rate: float            # Red 손실율 [0, 1]
    dominant_factor: str            # "firepower" | "defense" | "range" | "surprise"
    salvo_log: List[SalvoResult]    = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Hughes Salvo Model 파라미터
# ──────────────────────────────────────────────────────────────

# 함종별 살보 공격력 (발사체 / 교전)
SALVO_FIREPOWER: Dict[str, int] = {
    "destroyer":       8,    # SSM 8발 (하푼·해성)
    "frigate":         4,    # SSM 4발
    "submarine":       4,    # 어뢰·TLAM 4기
    "naval_aviation":  6,    # 함재기 공대함 6발
    "coastal_defense": 6,    # 해안포·SSM
    "mine_warfare":    0,    # 비공격 전용
    "amphibious_ship": 0,    # 비공격
    "fighter":         4,    # 공대함 4발 (함 지원)
    "strike_aircraft": 6,
}

# 함종별 방어 계수 α (적 살보 1발 대비 요격·방호 성능 [0, 1])
DEFENSE_COEFFICIENT: Dict[str, float] = {
    "destroyer":       0.70,  # CIWS + ESSM
    "frigate":         0.55,  # CIWS
    "submarine":       0.30,  # 잠항 기동
    "naval_aviation":  0.40,  # ECM + 기동
    "coastal_defense": 0.45,
    "mine_warfare":    0.20,
    "amphibious_ship": 0.25,
    "fighter":         0.60,  # 공중 요격
    "strike_aircraft": 0.50,
}

# 함종별 최대 교전 거리 (km)
ENGAGEMENT_RANGE_KM: Dict[str, float] = {
    "destroyer":       200.0,  # 하푼·해성 사거리
    "frigate":         150.0,
    "submarine":        80.0,  # 어뢰 + TLAM
    "naval_aviation":  200.0,
    "coastal_defense": 180.0,
    "fighter":         300.0,  # 공대함
    "strike_aircraft": 400.0,
}


# ──────────────────────────────────────────────────────────────
# NavalCombatEngine
# ──────────────────────────────────────────────────────────────

class NavalCombatEngine:
    """
    Hughes Salvo 방정식 기반 해상 전투 엔진.

    핵심 수식 (살보 교환 1회):
      B_damage = max(0, R_salvo - α_B × B_ships)
      R_damage = max(0, B_salvo - α_R × R_ships)

    여기서
      B_salvo  = Blue 함정 수 × 함종별 살보 화력
      α_B      = Blue 방어 계수 × Blue 함정 수 (요격 능력 합산)
      B_damage : Red 살보 중 방어 불가 부분 → Blue 피해 함정 수
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    # ── 내부 헬퍼 ────────────────────────────────────────────

    def _unit_firepower(self, unit: Unit) -> int:
        """유닛의 살보 화력 (발사체 수)"""
        base = SALVO_FIREPOWER.get(unit.unit_type.value, 0)
        # 전투력·탄약 수준 보정
        modifier = float(unit.capability.firepower) * float(unit.capability.ammo_level)
        return max(0, int(base * max(0.2, modifier)))

    def _unit_defense(self, unit: Unit) -> float:
        """유닛의 방어 계수"""
        base = DEFENSE_COEFFICIENT.get(unit.unit_type.value, 0.2)
        return base * float(unit.capability.protection)

    def _max_range(self, unit: Unit) -> float:
        return ENGAGEMENT_RANGE_KM.get(unit.unit_type.value, 100.0)

    def _engagement_range(
        self,
        blue_units: List[Unit],
        red_units: List[Unit],
    ) -> float:
        """양측 교전 거리 결정 (단거리 무장이 결정)"""
        if not blue_units or not red_units:
            return 0.0
        # 거리 샘플: 양측 중심 거리
        bcx = float(np.mean([u.position.x for u in blue_units]))
        bcy = float(np.mean([u.position.y for u in blue_units]))
        rcx = float(np.mean([u.position.x for u in red_units]))
        rcy = float(np.mean([u.position.y for u in red_units]))
        return float(math.sqrt((bcx - rcx) ** 2 + (bcy - rcy) ** 2))

    # ── 주요 API ─────────────────────────────────────────────

    def compute_salvo_exchange(
        self,
        blue_units: List[Unit],
        red_units: List[Unit],
    ) -> SalvoResult:
        """
        단일 살보 교환 계산.
        양측 함정이 동시에 발사 → 요격 처리 → 피해 계산.
        """
        if not blue_units or not red_units:
            return SalvoResult(0, 0, 0.0, 0.0, 0, 0, 1.0, 1.0, 0.0, 0, 0)

        # ── 살보 화력 계산
        b_total_fire  = sum(self._unit_firepower(u) for u in blue_units)
        r_total_fire  = sum(self._unit_firepower(u) for u in red_units)

        # ── 방어 능력 계산 (함종별 방어 계수 합산)
        b_total_def   = sum(self._unit_defense(u) * u.headcount for u in blue_units)
        r_total_def   = sum(self._unit_defense(u) * u.headcount for u in red_units)

        b_ship_count  = len(blue_units)
        r_ship_count  = len(red_units)

        # ── 요격 (intercept): 방어 계수 × 적 살보
        b_intercepted = min(b_total_def / max(b_ship_count, 1), r_total_fire)
        r_intercepted = min(r_total_def / max(r_ship_count, 1), b_total_fire)

        # 확률 노이즈 ±15%
        noise = self.rng.uniform(-0.15, 0.15)
        b_intercepted = max(0.0, b_intercepted * (1 + noise))
        r_intercepted = max(0.0, r_intercepted * (1 + noise))

        # ── 피해 살보 (방어 후 남은 적 살보)
        red_penetrated  = max(0.0, r_total_fire - b_intercepted)
        blue_penetrated = max(0.0, b_total_fire - r_intercepted)

        # 함정당 타격력 → 피해 함정 수 (각 살보 1발 = 0.5~1.0척 피해)
        hit_ratio_blue = self.rng.uniform(0.5, 1.0)
        hit_ratio_red  = self.rng.uniform(0.5, 1.0)

        b_damaged_ships = float(red_penetrated  * hit_ratio_blue) / max(b_ship_count, 1)
        r_damaged_ships = float(blue_penetrated * hit_ratio_red)  / max(r_ship_count, 1)

        b_sunk = min(int(b_damaged_ships * b_ship_count), b_ship_count)
        r_sunk = min(int(r_damaged_ships * r_ship_count), r_ship_count)

        b_surviving = float(max(b_ship_count - b_sunk, 0)) / max(b_ship_count, 1)
        r_surviving = float(max(r_ship_count - r_sunk, 0)) / max(r_ship_count, 1)

        rng_km = self._engagement_range(blue_units, red_units)

        return SalvoResult(
            blue_salvos_fired     = b_total_fire,
            red_salvos_fired      = r_total_fire,
            blue_ships_damaged    = b_damaged_ships,
            red_ships_damaged     = r_damaged_ships,
            blue_ships_sunk       = b_sunk,
            red_ships_sunk        = r_sunk,
            blue_surviving_fraction = b_surviving,
            red_surviving_fraction  = r_surviving,
            engagement_range_km   = rng_km,
            salvos_intercepted_blue = int(b_intercepted),
            salvos_intercepted_red  = int(r_intercepted),
        )

    def run_naval_battle(
        self,
        kg: CombatKnowledgeGraph,
        max_salvos: int = 5,
    ) -> NavalBattleReport:
        """
        KG에서 해상 유닛을 추출해 살보 교환을 반복하여
        해상 전투 결과를 반환한다.
        """
        from ontology.combat_schema import UnitStatus

        blue_ships = [
            u for u in kg.units.values()
            if u.alignment == ForceAlignment.BLUE
            and u.unit_type in NAVAL_UNIT_TYPES
            and u.status != UnitStatus.DESTROYED
        ]
        red_ships = [
            u for u in kg.units.values()
            if u.alignment == ForceAlignment.RED
            and u.unit_type in NAVAL_UNIT_TYPES
            and u.status != UnitStatus.DESTROYED
        ]

        b_init = len(blue_ships)
        r_init = len(red_ships)
        salvo_log: List[SalvoResult] = []

        active_blue = list(blue_ships)
        active_red  = list(red_ships)

        for _ in range(max_salvos):
            if not active_blue or not active_red:
                break
            result = self.compute_salvo_exchange(active_blue, active_red)
            salvo_log.append(result)

            # 함정 제거 (침몰)
            for _ in range(result.blue_ships_sunk):
                if active_blue:
                    victim = self.rng.choice(len(active_blue))  # type: ignore[arg-type]
                    lost = active_blue.pop(victim)
                    if lost.unit_id in kg.units:
                        kg.units[lost.unit_id].status = UnitStatus.DESTROYED
                        kg.units[lost.unit_id].headcount = 0

            for _ in range(result.red_ships_sunk):
                if active_red:
                    victim = self.rng.choice(len(active_red))  # type: ignore[arg-type]
                    lost = active_red.pop(victim)
                    if lost.unit_id in kg.units:
                        kg.units[lost.unit_id].status = UnitStatus.DESTROYED
                        kg.units[lost.unit_id].headcount = 0

        b_surv = len(active_blue)
        r_surv = len(active_red)
        b_loss_rate = float(b_init - b_surv) / max(b_init, 1)
        r_loss_rate = float(r_init - r_surv) / max(r_init, 1)

        blue_win = b_surv > 0 and r_surv == 0
        red_win  = r_surv > 0 and b_surv == 0
        draw     = not blue_win and not red_win

        dominant = "firepower"
        if salvo_log:
            avg_int_b = np.mean([s.salvos_intercepted_blue for s in salvo_log])
            avg_int_r = np.mean([s.salvos_intercepted_red  for s in salvo_log])
            if max(avg_int_b, avg_int_r) > 0.6 * np.mean(
                [s.blue_salvos_fired + s.red_salvos_fired for s in salvo_log]
            ):
                dominant = "defense"

        return NavalBattleReport(
            total_salvos          = len(salvo_log),
            blue_initial_ships    = b_init,
            red_initial_ships     = r_init,
            blue_surviving_ships  = b_surv,
            red_surviving_ships   = r_surv,
            blue_win              = blue_win,
            red_win               = red_win,
            draw                  = draw,
            blue_loss_rate        = b_loss_rate,
            red_loss_rate         = r_loss_rate,
            dominant_factor       = dominant,
            salvo_log             = salvo_log,
        )


# ──────────────────────────────────────────────────────────────
# 소나 방정식 (Submarine Detection)
# ──────────────────────────────────────────────────────────────

class SubmarineDetector:
    """
    음파탐지 방정식 (Sonar Equation) 기반 잠수함 탐지 모델.

    탐지 마진 DM = SL - TL - NL + DI - DT
      SL  : 소나 음원 수준 (Source Level, dB)
      TL  : 전파 손실 (Transmission Loss, 20 log R)
      NL  : 주변 소음 수준 (Noise Level, dB)
      DI  : 방향성 지수 (Directivity Index, dB)
      DT  : 탐지 임계값 (Detection Threshold, dB)
    DM > 0 → 탐지 가능.
    탐지 확률 = sigmoid(DM / σ).
    """

    DEFAULT_PARAMS = {
        "source_level_db":         220.0,   # 활성 소나 음원 (dB re 1 μPa @ 1m)
        "noise_level_db":           90.0,   # 해양 배경 소음 (dB)
        "directivity_index_db":     15.0,   # 소나 배열 지향성
        "detection_threshold_db":   15.0,   # 탐지 문턱값
        "sigma":                     5.0,   # 불확실성 표준편차 (dB)
    }

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.rng = np.random.RandomState(seed)

    def transmission_loss(self, range_km: float) -> float:
        """전파 손실 TL ≈ 20 log10(R_m) dB (구형파 확산)"""
        r_m = max(range_km * 1000.0, 1.0)
        return 20.0 * math.log10(r_m)

    def detection_probability(
        self,
        range_km: float,
        target_unit: Unit,
        sea_state: int = 3,
    ) -> float:
        """
        거리 및 대상 유닛의 스텔스 계수를 반영한 탐지 확률 계산.
        sea_state: 해상 상태 (0~6, Beaufort 척도)
        """
        p = self.params
        tl  = self.transmission_loss(range_km)
        # 해상 상태별 소음 보정 (+3 dB per sea state 증가)
        nl  = p["noise_level_db"] + sea_state * 3.0
        # 잠수함의 스텔스 계수 반영 (스텔스 높을수록 SL 낮아짐)
        stealth = getattr(target_unit.capability, "stealth_factor", 0.0)
        sl  = p["source_level_db"] * (1.0 - stealth * 0.3)
        dm  = sl - tl - nl + p["directivity_index_db"] - p["detection_threshold_db"]
        # sigmoid
        prob = 1.0 / (1.0 + math.exp(-dm / p["sigma"]))
        return float(np.clip(prob, 0.0, 1.0))

    def detect(
        self,
        kg: CombatKnowledgeGraph,
        sonar_unit: Unit,
        sea_state: int = 3,
    ) -> List[SonarContact]:
        """
        소나 장착 유닛 기준으로 kg 내 모든 잠수함 탐지 시도.
        """
        contacts: List[SonarContact] = []
        for uid, target in kg.units.items():
            if target.unit_type not in SUBMARINE_TYPES:
                continue
            if target.alignment == sonar_unit.alignment:
                continue
            if target.status == UnitStatus.DESTROYED:
                continue

            range_km = sonar_unit.position.distance_to(target.position)
            prob = self.detection_probability(range_km, target, sea_state)

            if self.rng.random() < prob:
                bearing_err = self.rng.normal(0, 10.0 * (1 - prob))
                range_est   = range_km * self.rng.uniform(0.85, 1.15)
                if prob > 0.7:
                    quality = "firm"
                elif prob > 0.4:
                    quality = "probable"
                else:
                    quality = "possible"
            else:
                bearing_err = 0.0
                range_est   = 0.0
                quality     = "none"
                prob        = 0.0

            contacts.append(SonarContact(
                target_unit_id       = uid,
                detection_probability = prob,
                bearing_error_deg    = float(abs(bearing_err)),
                range_estimate_km    = float(range_est),
                contact_quality      = quality,
            ))

        return contacts


# ──────────────────────────────────────────────────────────────
# 기뢰 위협 모델
# ──────────────────────────────────────────────────────────────

class MineThreatModel:
    """
    기뢰 위협 확률 및 소해 모델.
    기뢰 밀도 × 통과 함정 수 → 피해 확률 계산.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def compute_mine_threat(
        self,
        n_ships: int,
        mine_density_per_km2: float = 2.0,
        channel_width_km: float    = 5.0,
        channel_length_km: float   = 20.0,
        minesweeper_factor: float  = 0.0,   # [0, 1] 소해 비율
    ) -> MineEngagementResult:
        """
        기뢰지대 통과 시 피해 계산.
        기뢰 위협 확률 = 1 - exp(-λ × 함정 수)
        λ = mine_density × channel_area × (1 - sweep)
        """
        channel_area = channel_width_km * channel_length_km
        effective_density = mine_density_per_km2 * (1.0 - minesweeper_factor)
        n_mines = int(effective_density * channel_area)

        # 각 함정이 기뢰를 만날 확률 (푸아송 근사)
        hit_prob_per_ship = 1.0 - math.exp(-effective_density * channel_width_km * 0.5)
        hit_prob_per_ship = float(np.clip(hit_prob_per_ship, 0, 1))

        ships_hit = 0
        for _ in range(n_ships):
            if self.rng.random() < hit_prob_per_ship:
                ships_hit += 1

        # 피격 함정 중 침몰 비율 (기뢰 유형에 따라 0.4~0.8)
        sink_prob  = self.rng.uniform(0.4, 0.8)
        ships_sunk = sum(
            1 for _ in range(ships_hit) if self.rng.random() < sink_prob
        )
        ships_damaged = ships_hit - ships_sunk

        channel_cleared = (minesweeper_factor > 0.6)

        return MineEngagementResult(
            mines_encountered = n_mines,
            ships_damaged     = ships_damaged,
            ships_sunk        = ships_sunk,
            damage_fraction   = float(ships_hit) / max(n_ships, 1),
            channel_cleared   = channel_cleared,
        )


# ──────────────────────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory, UnitType

    print("=== Naval Engine Test ===")
    engine = NavalCombatEngine(seed=42)
    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=4, seed=42)

    # 해상 유닛 강제 추가 테스트
    report = engine.run_naval_battle(kg, max_salvos=3)
    print(f"살보 횟수: {report.total_salvos}")
    print(f"Blue {report.blue_initial_ships}→{report.blue_surviving_ships}척 | "
          f"Red {report.red_initial_ships}→{report.red_surviving_ships}척")
    print(f"결과: {'Blue Win' if report.blue_win else 'Red Win' if report.red_win else 'Draw'}")

    print("\n=== Sonar Test ===")
    detector = SubmarineDetector(seed=42)
    for dist in [10, 30, 60, 100]:
        prob = detector.detection_probability(dist, list(kg.units.values())[0])
        print(f"  거리 {dist:3d}km → 탐지확률 {prob:.3f}")

    print("\n=== Mine Threat Test ===")
    mine_model = MineThreatModel(seed=42)
    result = mine_model.compute_mine_threat(n_ships=6, mine_density_per_km2=1.5)
    print(f"  피해함정: {result.ships_damaged}척 손상 / {result.ships_sunk}척 침몰 "
          f"(피해율 {result.damage_fraction:.1%})")
