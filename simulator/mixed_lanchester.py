"""
mixed_lanchester.py
===================
② 혼합 란체스터 법칙
- 1차 법칙 (게릴라/비선형전): dB/dt = -α,  dR/dt = -β    (상수 소모)
- 2차 법칙 (정규전):          dB/dt = -α·R, dR/dt = -β·B  (상호 소모)
- 혼합 비율 자동 판별: 지형·유닛 구성·전투 밀도로 결정
- 포위(encirclement) 판정 → 포위된 쪽 패널티
- 기동전(maneuver warfare) 보너스

학술 연구용 합성 데이터
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, UnitStatus, ForceAlignment, TerrainType, CombatKnowledgeGraph,
    DomainType,
)


class CombatMode(Enum):
    """전투 양식"""
    LINEAR      = "linear"       # 순수 2차 법칙 (정면전)
    GUERRILLA   = "guerrilla"    # 순수 1차 법칙 (게릴라전)
    MIXED       = "mixed"        # 혼합 (가중 평균)
    ENCIRCLEMENT = "encirclement" # 포위전


@dataclass
class MixedEngagementResult:
    """혼합 란체스터 교전 결과 (확장판)"""
    attacker_id: str
    defender_id: str
    combat_mode: CombatMode
    mix_ratio: float             # 선형전 비율 [0=게릴라, 1=정규전]
    attacker_losses: int
    defender_losses: int
    attacker_new_status: UnitStatus
    defender_new_status: UnitStatus
    terrain_bonus: float
    flanking_bonus: float        # 측면 공격 보너스
    encirclement_penalty: float  # 포위 패널티
    resource_penalty: float      # 자원 고갈 패널티
    description: str


class MixedLanchesterEngine:
    """
    혼합 란체스터 전투 엔진

    판별 기준:
    - 도시/산악 지형 → 게릴라전 비중 ↑
    - 정규 보병/기갑 → 정규전 비중 ↑
    - 병력 밀도 높음 → 정규전 비중 ↑
    - 항공 지원 → 게릴라 억제

    확장 요소:
    - 포위 판정: 적이 3면 이상 포위 시 추가 패널티
    - 기동전: 측면 공격 시 방어 보너스 무효화
    - 자원 패널티 연동
    """

    TERRAIN_GUERRILLA_BIAS: Dict[str, float] = {
        # 값이 높을수록 게릴라전 비중 증가
        "open":     0.05,
        "urban":    0.60,
        "forest":   0.50,
        "mountain": 0.55,
        "river":    0.20,
        "coastal":  0.25,
    }

    # 항공 지원으로 인정되는 UnitType 집합
    _AVIATION_TYPES: set = {
        UnitType.AVIATION, UnitType.FIGHTER, UnitType.STRIKE_AIRCRAFT, UnitType.BOMBER,
        UnitType.NAVAL_AVIATION, UnitType.MARINE_AVIATION, UnitType.UAV_STRIKE,
        UnitType.UAV_RECON, UnitType.LOITERING_MUNITION, UnitType.ISR_AIRCRAFT,
    }

    UNIT_LINEAR_BIAS: Dict[str, float] = {
        # 값이 높을수록 정규전(2차 법칙) 비중 증가
        # ── 기존 8종 ─────────────────────────────────
        "infantry":           0.50,
        "armor":              0.85,
        "artillery":          0.70,
        "aviation":           0.65,
        "engineer":           0.40,
        "signal":             0.20,
        "logistics":          0.15,
        "electronic_warfare": 0.30,
        # ── 육군 신규 ────────────────────────────────
        "mechanized_infantry":    0.80,
        "airborne":               0.45,
        "special_forces":         0.30,
        "anti_armor":             0.60,
        "air_defense_artillery":  0.75,
        "reconnaissance":         0.35,
        "nbc_defense":            0.30,
        "military_police":        0.40,
        # ── 해군 ─────────────────────────────────────
        "destroyer":          0.80,
        "frigate":            0.78,
        "submarine":          0.70,
        "mine_warfare":       0.60,
        "amphibious_ship":    0.65,
        "naval_aviation":     0.82,
        "coastal_defense":    0.65,
        # ── 공군 ─────────────────────────────────────
        "fighter":            0.88,
        "strike_aircraft":    0.85,
        "bomber":             0.85,
        "transport_aircraft": 0.20,
        "isr_aircraft":       0.20,
        "early_warning":      0.15,
        "air_refueling":      0.15,
        "sam_battery":        0.75,
        # ── 해병대 ───────────────────────────────────
        "marine_infantry":    0.55,
        "marine_armor":       0.82,
        "marine_aviation":    0.80,
        # ── 미사일 ───────────────────────────────────
        "ballistic_missile":  0.95,
        "cruise_missile":     0.92,
        "mlrs":               0.80,
        # ── 드론 ─────────────────────────────────────
        "uav_recon":          0.50,
        "uav_strike":         0.75,
        "loitering_munition": 0.80,
        # ── 사이버/우주 ───────────────────────────────
        "cyber_unit":         0.05,
        "space_asset":        0.10,
    }

    # 신규 유닛 타입 → 기존 카테고리 매핑 (전투 효율 행렬 폴백)
    _UNIT_CATEGORY_MAP: Dict[str, str] = {
        "mechanized_infantry": "armor",    "airborne": "infantry",
        "special_forces": "infantry",      "anti_armor": "artillery",
        "air_defense_artillery": "artillery", "reconnaissance": "infantry",
        "nbc_defense": "engineer",         "military_police": "infantry",
        "destroyer": "aviation",           "frigate": "aviation",
        "submarine": "aviation",           "mine_warfare": "artillery",
        "amphibious_ship": "logistics",    "naval_aviation": "aviation",
        "coastal_defense": "artillery",
        "fighter": "aviation",             "strike_aircraft": "aviation",
        "bomber": "aviation",              "transport_aircraft": "logistics",
        "isr_aircraft": "signal",          "early_warning": "signal",
        "air_refueling": "logistics",      "sam_battery": "artillery",
        "marine_infantry": "infantry",     "marine_armor": "armor",
        "marine_aviation": "aviation",
        "ballistic_missile": "artillery",  "cruise_missile": "artillery",
        "mlrs": "artillery",
        "uav_recon": "signal",             "uav_strike": "aviation",
        "loitering_munition": "artillery",
        "cyber_unit": "electronic_warfare","space_asset": "signal",
    }

    COMBAT_EFFICIENCY: Dict[str, Dict[str, float]] = {
        "infantry":          {"infantry": 1.0, "armor": 0.3, "artillery": 1.2, "aviation": 0.5,
                              "engineer": 1.1, "signal": 1.5, "logistics": 1.8, "electronic_warfare": 1.3},
        "armor":             {"infantry": 1.8, "armor": 1.0, "artillery": 1.5, "aviation": 0.4,
                              "engineer": 1.4, "signal": 2.0, "logistics": 2.2, "electronic_warfare": 1.6},
        "artillery":         {"infantry": 2.5, "armor": 1.2, "artillery": 1.0, "aviation": 0.2,
                              "engineer": 2.0, "signal": 2.5, "logistics": 2.8, "electronic_warfare": 2.0},
        "aviation":          {"infantry": 2.0, "armor": 2.5, "artillery": 1.8, "aviation": 1.0,
                              "engineer": 1.5, "signal": 1.8, "logistics": 2.5, "electronic_warfare": 1.2},
        "engineer":          {"infantry": 0.8, "armor": 0.5, "artillery": 0.8, "aviation": 0.3,
                              "engineer": 0.9, "signal": 1.0, "logistics": 1.2, "electronic_warfare": 0.8},
        "signal":            {"infantry": 0.2, "armor": 0.1, "artillery": 0.2, "aviation": 0.1,
                              "engineer": 0.3, "signal": 0.4, "logistics": 0.5, "electronic_warfare": 0.3},
        "logistics":         {"infantry": 0.3, "armor": 0.1, "artillery": 0.3, "aviation": 0.1,
                              "engineer": 0.4, "signal": 0.5, "logistics": 0.6, "electronic_warfare": 0.4},
        "electronic_warfare":{"infantry": 0.1, "armor": 0.2, "artillery": 0.3, "aviation": 0.4,
                              "engineer": 0.2, "signal": 2.0, "logistics": 0.5, "electronic_warfare": 1.5},
    }

    TERRAIN_DEFENSE_BONUS: Dict[str, float] = {
        "open": 1.0, "urban": 1.8, "forest": 1.4,
        "mountain": 1.6, "river": 1.2, "coastal": 1.3,
    }

    def _get_combat_efficiency(self, atk_type: str, def_type: str) -> float:
        """
        전투 효율 조회.
        1. 직접 조회 → 2. 카테고리 폴백 → 3. 기본값 1.0
        """
        direct = self.COMBAT_EFFICIENCY.get(atk_type, {}).get(def_type)
        if direct is not None:
            return direct
        # 카테고리 폴백
        atk_cat = self._UNIT_CATEGORY_MAP.get(atk_type, atk_type)
        def_cat = self._UNIT_CATEGORY_MAP.get(def_type, def_type)
        return self.COMBAT_EFFICIENCY.get(atk_cat, {}).get(def_cat, 1.0)

    def _has_aviation_support(self, kg: Optional[CombatKnowledgeGraph], alignment: ForceAlignment) -> bool:
        """아군 항공 지원 여부 판별 (확장된 항공 유닛 타입 포함)"""
        if kg is None:
            return False
        return any(
            u.unit_type in self._AVIATION_TYPES
            and u.alignment == alignment
            and u.status != UnitStatus.DESTROYED
            for u in kg.units.values()
        )

    def __init__(self, noise_std: float = 0.10, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    # ── 핵심 판별 로직 ─────────────────────────

    def determine_mix_ratio(
        self,
        attacker: Unit,
        defender: Unit,
        terrain: Optional[str] = None,
        density_factor: float = 0.5,    # 전투 밀도 [0, 1]
        aviation_support: bool = False
    ) -> Tuple[float, CombatMode]:
        """
        혼합 비율 자동 판별
        Returns: (linear_ratio, CombatMode)
        linear_ratio: 1.0 = 순수 정규전, 0.0 = 순수 게릴라전
        """
        # 지형 기반 게릴라 편향
        terrain_guerrilla = self.TERRAIN_GUERRILLA_BIAS.get(terrain or "open", 0.1)

        # 유닛 기반 정규전 편향 (공격/수비 평균)
        atk_linear = self.UNIT_LINEAR_BIAS.get(attacker.unit_type.value, 0.5)
        def_linear = self.UNIT_LINEAR_BIAS.get(defender.unit_type.value, 0.5)
        unit_linear = (atk_linear + def_linear) / 2

        # 병력 밀도: 높을수록 정규전
        density_linear = density_factor

        # 항공 지원: 게릴라 억제
        aviation_bonus = 0.20 if aviation_support else 0.0

        # 가중 합산 → 정규전 비율
        linear_ratio = (
            unit_linear * 0.40
            + density_linear * 0.25
            + (1 - terrain_guerrilla) * 0.25
            + aviation_bonus * 0.10
        )
        linear_ratio = float(np.clip(linear_ratio, 0.0, 1.0))

        # 모드 분류
        if linear_ratio >= 0.75:
            mode = CombatMode.LINEAR
        elif linear_ratio <= 0.25:
            mode = CombatMode.GUERRILLA
        else:
            mode = CombatMode.MIXED

        return linear_ratio, mode

    def detect_encirclement(
        self,
        target: Unit,
        enemy_units: List[Unit],
        angle_threshold: float = 210.0   # 3면 이상 (210도 이상)
    ) -> Tuple[bool, float]:
        """
        포위 감지 - 적 유닛들이 대상 주위를 몇 도 포위하는지 계산
        Returns: (is_encircled, penalty_multiplier)
        """
        if not enemy_units:
            return False, 1.0

        # 각 적까지의 방향 각도 계산
        angles = []
        tx, ty = target.position.x, target.position.y
        for enemy in enemy_units:
            if enemy.status == UnitStatus.DESTROYED:
                continue
            dx = enemy.position.x - tx
            dy = enemy.position.y - ty
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 0.1:
                continue
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            angles.append(angle)

        if len(angles) < 2:
            return False, 1.0

        angles_sorted = sorted(angles)
        # 최대 빈 구간 계산
        gaps = []
        for i in range(len(angles_sorted)):
            next_angle = angles_sorted[(i + 1) % len(angles_sorted)]
            curr_angle = angles_sorted[i]
            gap = (next_angle - curr_angle) % 360
            gaps.append(gap)
        max_gap = max(gaps)

        # 포위 각도 = 360 - 최대 빈 구간
        encircle_angle = 360.0 - max_gap
        is_encircled = bool(encircle_angle >= angle_threshold)

        # 패널티: 포위 각도에 비례 (최대 2배 피해)
        penalty = 1.0 + (encircle_angle / 360.0) * 1.0 if is_encircled else 1.0

        return bool(is_encircled), float(penalty)

    # ── 혼합 란체스터 적분 ─────────────────────

    def _integrate_mixed(
        self,
        B: float, R: float,
        alpha: float, beta: float,
        linear_ratio: float,
        duration: float = 1.0,
        dt: float = 0.05
    ) -> Tuple[float, float]:
        """
        혼합 란체스터 방정식 수치 적분

        dB/dt = -[lr·α·R + (1-lr)·α]      (선형전 + 게릴라전 혼합)
        dR/dt = -[lr·β·B + (1-lr)·β]
        """
        steps = max(1, int(duration / dt))
        for _ in range(steps):
            # 2차(선형전) 항
            dB_linear = -linear_ratio * alpha * R * dt
            dR_linear = -linear_ratio * beta  * B * dt
            # 1차(게릴라전) 항
            dB_guerr  = -(1 - linear_ratio) * alpha * dt
            dR_guerr  = -(1 - linear_ratio) * beta  * dt

            B = max(0.0, B + dB_linear + dB_guerr)
            R = max(0.0, R + dR_linear + dR_guerr)

            if B <= 0 or R <= 0:
                break
        return B, R

    # ── 메인 교전 계산 ──────────────────────────

    def calculate_engagement(
        self,
        attacker: Unit,
        defender: Unit,
        kg: Optional[CombatKnowledgeGraph] = None,
        defender_terrain: Optional[str] = None,
        is_flanking: bool = False,
        resource_manager=None
    ) -> MixedEngagementResult:
        """
        혼합 란체스터 교전 결과 계산
        """
        atk_type = attacker.unit_type.value
        def_type = defender.unit_type.value

        # 기본 전투 효율 (신규 UnitType 폴백 포함)
        alpha = self._get_combat_efficiency(atk_type, def_type)
        beta  = self._get_combat_efficiency(def_type, atk_type)

        # 지형 방어 보너스
        terrain_key = defender_terrain or "open"
        terrain_bonus = self.TERRAIN_DEFENSE_BONUS.get(terrain_key, 1.0)
        if not is_flanking:
            alpha /= terrain_bonus   # 정면 공격: 지형 보너스 적용
        flanking_bonus = terrain_bonus if is_flanking else 1.0  # 측면: 지형 무력화

        # 자원 패널티 적용
        resource_penalty = 1.0
        if resource_manager:
            fp_mult = resource_manager.get_firepower_multiplier(attacker)
            resource_penalty = fp_mult
            alpha *= fp_mult

        # 전투력·사기·경험 보정
        alpha *= attacker.combat_power * (1 + 0.2 * attacker.experience)
        beta  *= defender.combat_power  * (1 + 0.2 * defender.experience)

        # 전투 밀도 계산 (유닛 수 기반)
        if kg:
            n_all = len(kg.units)
            density = min(n_all / 20.0, 1.0)
        else:
            density = 0.5

        # 포위 판정
        encirclement_penalty = 1.0
        encircled = False
        if kg:
            enemies_of_defender = [
                u for u in kg.units.values()
                if u.alignment == attacker.alignment
                and u.status != UnitStatus.DESTROYED
                and u.unit_id != attacker.unit_id
            ]
            encircled, encirclement_penalty = self.detect_encirclement(
                defender, enemies_of_defender
            )
            alpha *= encirclement_penalty

        # 항공 지원 여부 (모든 항공 UnitType 포함)
        has_aviation = self._has_aviation_support(kg, attacker.alignment)

        # 혼합 비율 결정
        linear_ratio, mode = self.determine_mix_ratio(
            attacker, defender, terrain_key, density, has_aviation
        )

        # 노이즈
        alpha *= (1 + self.rng.normal(0, self.noise_std))
        beta  *= (1 + self.rng.normal(0, self.noise_std))
        alpha, beta = max(0, alpha), max(0, beta)

        # 혼합 란체스터 적분
        B_init = float(attacker.headcount)
        R_init = float(defender.headcount)
        B_final, R_final = self._integrate_mixed(
            B_init, R_init, alpha, beta, linear_ratio, duration=1.0
        )

        atk_losses = int(B_init - B_final)
        def_losses = int(R_init - R_final)

        def status_from_ratio(orig, remain):
            r = remain / max(orig, 1)
            if r <= 0.05: return UnitStatus.DESTROYED
            if r <= 0.30: return UnitStatus.CRITICAL
            if r <= 0.70: return UnitStatus.DEGRADED
            return UnitStatus.ACTIVE

        mode_display = mode.value
        if encircled:
            mode_display = f"{mode.value} + ENCIRCLED"

        desc = (
            f"[{mode_display.upper()}] "
            f"{atk_type}→{def_type} | "
            f"lr={linear_ratio:.2f} | "
            f"지형보너스={terrain_bonus:.2f} | "
            f"포위={encircled} | "
            f"손실: 공격={atk_losses}, 수비={def_losses}"
        )

        return MixedEngagementResult(
            attacker_id=attacker.unit_id,
            defender_id=defender.unit_id,
            combat_mode=mode,
            mix_ratio=linear_ratio,
            attacker_losses=atk_losses,
            defender_losses=def_losses,
            attacker_new_status=status_from_ratio(B_init, B_final),
            defender_new_status=status_from_ratio(R_init, R_final),
            terrain_bonus=terrain_bonus,
            flanking_bonus=flanking_bonus,
            encirclement_penalty=encirclement_penalty,
            resource_penalty=resource_penalty,
            description=desc,
        )

    def run_step_mixed(
        self,
        kg: CombatKnowledgeGraph,
        action_pairs: Optional[List[Tuple[str, str]]] = None,
        flanking_pairs: Optional[List[str]] = None,
        resource_manager=None
    ) -> Dict:
        """
        혼합 란체스터 1스텝 실행
        """
        if action_pairs is None:
            action_pairs = self._auto_pair(kg)
        flanking_set = set(flanking_pairs or [])

        results = []
        casualties: Dict[str, int] = {}
        new_statuses: Dict[str, UnitStatus] = {}

        for atk_id, def_id in action_pairs:
            if atk_id not in kg.units or def_id not in kg.units:
                continue
            atk = kg.units[atk_id]
            def_ = kg.units[def_id]
            if atk.status == UnitStatus.DESTROYED:
                continue

            terrain = self._get_terrain(kg, def_id)
            is_flanking = atk_id in flanking_set

            result = self.calculate_engagement(
                atk, def_, kg, terrain, is_flanking, resource_manager
            )
            results.append(result)

            casualties[atk_id] = casualties.get(atk_id, 0) + result.attacker_losses
            casualties[def_id] = casualties.get(def_id, 0) + result.defender_losses
            self._update_worse_status(new_statuses, atk_id, result.attacker_new_status)
            self._update_worse_status(new_statuses, def_id, result.defender_new_status)

        for uid, losses in casualties.items():
            if uid in kg.units:
                u = kg.units[uid]
                u.headcount = max(0, u.headcount - losses)
                if uid in new_statuses:
                    u.status = new_statuses[uid]
                u.morale = max(0.1, u.morale - 0.05 * (losses / max(u.headcount + losses, 1)))

                # 자원 소모 (교전에 의한)
                if resource_manager and u.status != UnitStatus.DESTROYED:
                    resource_manager.consume_ammo(u, n_engagements=1)
                    resource_manager.consume_fuel(u, distance_moved_km=0.3)

        blue_cas = sum(v for k, v in casualties.items()
                       if k in kg.units and kg.units[k].alignment == ForceAlignment.BLUE)
        red_cas  = sum(v for k, v in casualties.items()
                       if k in kg.units and kg.units[k].alignment == ForceAlignment.RED)
        blue_hc = sum(u.headcount for u in kg.units.values() if u.alignment == ForceAlignment.BLUE)
        red_hc  = sum(u.headcount for u in kg.units.values() if u.alignment == ForceAlignment.RED)

        status = "ongoing"
        if blue_hc < 10: status = "red_win"
        elif red_hc < 10: status = "blue_win"

        return {
            "results": results,
            "blue_casualties": blue_cas,
            "red_casualties": red_cas,
            "blue_headcount": blue_hc,
            "red_headcount": red_hc,
            "mission_status": status,
            "combat_modes": {r.defender_id: r.combat_mode.value for r in results},
        }

    def _auto_pair(self, kg: CombatKnowledgeGraph) -> List[Tuple[str, str]]:
        pairs = []
        blue = [u for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED]
        red  = [u for u in kg.units.values()
                if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED]
        for b in blue:
            in_range = [r for r in red
                        if b.position.distance_to(r.position) <= b.capability.range_km]
            if in_range:
                pairs.append((b.unit_id, max(in_range, key=lambda r: r.combat_power).unit_id))
        for r in red:
            in_range = [b for b in blue
                        if r.position.distance_to(b.position) <= r.capability.range_km]
            if in_range:
                pairs.append((r.unit_id, max(in_range, key=lambda b: b.combat_power).unit_id))
        return pairs

    def _get_terrain(self, kg: CombatKnowledgeGraph, uid: str) -> Optional[str]:
        if uid not in kg.units: return None
        unit = kg.units[uid]
        closest = min(kg.terrain_cells.values(),
                      key=lambda c: unit.position.distance_to(c.position), default=None)
        return closest.terrain_type.value if closest else "open"

    @staticmethod
    def _update_worse_status(d: Dict, uid: str, new: UnitStatus):
        order = [UnitStatus.ACTIVE, UnitStatus.DEGRADED, UnitStatus.CRITICAL, UnitStatus.DESTROYED]
        if uid not in d: d[uid] = new
        elif order.index(new) > order.index(d[uid]): d[uid] = new


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory, ForceAlignment
    print("=" * 55)
    print("② 혼합 란체스터 법칙 테스트")
    print("=" * 55)

    engine = MixedLanchesterEngine(seed=42)
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)

    print("\n[지형별 전투 양식 판별]")
    blue_u = list(kg.units.values())[0]
    red_u  = list(kg.units.values())[-1]
    for terrain in ["open", "urban", "forest", "mountain"]:
        lr, mode = engine.determine_mix_ratio(blue_u, red_u, terrain, 0.5)
        print(f"  {terrain:10s}: linear_ratio={lr:.2f}  mode={mode.value}")

    print("\n[포위 판정 테스트]")
    from ontology.combat_schema import Position
    target = blue_u
    enemies = list(kg.units.values())[3:7]
    # 적들을 원형 배치
    for i, e in enumerate(enemies):
        angle = i * (360 / len(enemies))
        e.position = Position(
            target.position.x + 3 * np.cos(np.radians(angle)),
            target.position.y + 3 * np.sin(np.radians(angle))
        )
    is_enc, penalty = engine.detect_encirclement(target, enemies)
    print(f"  포위 감지: {is_enc}  패널티 배수: {penalty:.2f}")

    print("\n[10 스텝 혼합 란체스터 시뮬레이션]")
    for step in range(10):
        result = engine.run_step_mixed(kg)
        modes = list(set(result["combat_modes"].values()))
        print(f"  스텝 {step+1:2d}: Blue={result['blue_headcount']}명  "
              f"Red={result['red_headcount']}명  "
              f"전투양식={modes}  상태={result['mission_status']}")
        if result["mission_status"] != "ongoing":
            break

    print("\n✅ 혼합 란체스터 법칙 정상 동작!")
