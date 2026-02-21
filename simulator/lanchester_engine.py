"""
lanchester_engine.py
====================
란체스터 방정식 기반 전투 시뮬레이션 엔진
물리 기반 교전 계산 + 지형 효과 반영

란체스터 제곱 법칙:
  dB/dt = -α × R
  dR/dt = -β × B
  여기서 α, β는 전투 효율 계수
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ontology.combat_schema import (
    Unit, UnitStatus, ForceAlignment, TerrainType, CombatKnowledgeGraph
)


@dataclass
class EngagementResult:
    """교전 결과"""
    attacker_id: str
    defender_id: str
    attacker_losses: int
    defender_losses: int
    attacker_new_status: UnitStatus
    defender_new_status: UnitStatus
    engagement_duration: float   # 스텝 수
    terrain_modifier: float
    description: str


@dataclass
class BattleStep:
    """시뮬레이션 1스텝 결과"""
    timestep: int
    engagements: List[EngagementResult]
    blue_total_casualties: int
    red_total_casualties: int
    blue_active_units: int
    red_active_units: int
    blue_total_headcount: int
    red_total_headcount: int
    mission_status: str    # "ongoing", "blue_win", "red_win", "draw"
    events: List[str] = field(default_factory=list)


class LanchesterEngine:
    """
    란체스터 방정식 기반 전투 시뮬레이션 엔진
    - 유닛 유형별 전투 효율 행렬
    - 지형 효과 (방어 보너스, 기동 패널티)
    - 사기/경험치 승수
    - 확률적 노이즈
    """

    # 유닛 유형별 전투 효율 행렬 (공격자 × 수비자)
    # 행: 공격 유닛 유형, 열: 방어 유닛 유형
    COMBAT_EFFICIENCY = {
        "infantry":          {"infantry": 1.0,  "armor": 0.3,  "artillery": 1.2, "aviation": 0.5, "engineer": 1.1, "signal": 1.5, "logistics": 1.8, "electronic_warfare": 1.3},
        "armor":             {"infantry": 1.8,  "armor": 1.0,  "artillery": 1.5, "aviation": 0.4, "engineer": 1.4, "signal": 2.0, "logistics": 2.2, "electronic_warfare": 1.6},
        "artillery":         {"infantry": 2.5,  "armor": 1.2,  "artillery": 1.0, "aviation": 0.2, "engineer": 2.0, "signal": 2.5, "logistics": 2.8, "electronic_warfare": 2.0},
        "aviation":          {"infantry": 2.0,  "armor": 2.5,  "artillery": 1.8, "aviation": 1.0, "engineer": 1.5, "signal": 1.8, "logistics": 2.5, "electronic_warfare": 1.2},
        "engineer":          {"infantry": 0.8,  "armor": 0.5,  "artillery": 0.8, "aviation": 0.3, "engineer": 0.9, "signal": 1.0, "logistics": 1.2, "electronic_warfare": 0.8},
        "signal":            {"infantry": 0.2,  "armor": 0.1,  "artillery": 0.2, "aviation": 0.1, "engineer": 0.3, "signal": 0.4, "logistics": 0.5, "electronic_warfare": 0.3},
        "logistics":         {"infantry": 0.3,  "armor": 0.1,  "artillery": 0.3, "aviation": 0.1, "engineer": 0.4, "signal": 0.5, "logistics": 0.6, "electronic_warfare": 0.4},
        "electronic_warfare":{"infantry": 0.1,  "armor": 0.2,  "artillery": 0.3, "aviation": 0.4, "engineer": 0.2, "signal": 2.0, "logistics": 0.5, "electronic_warfare": 1.5},
    }

    # 지형별 방어 보너스 (수비자 유리)
    TERRAIN_DEFENSE_BONUS = {
        TerrainType.OPEN:     1.0,
        TerrainType.URBAN:    1.8,
        TerrainType.FOREST:   1.4,
        TerrainType.MOUNTAIN: 1.6,
        TerrainType.RIVER:    1.2,
        TerrainType.COASTAL:  1.3,
    }

    def __init__(self, noise_std: float = 0.1, seed: Optional[int] = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def calculate_engagement(
        self,
        attacker: Unit,
        defender: Unit,
        defender_terrain: Optional[TerrainType] = None,
        duration: float = 1.0
    ) -> EngagementResult:
        """
        두 유닛 간 교전 결과 계산
        란체스터 제곱 법칙 기반
        """
        # 기본 전투 효율
        atk_type = attacker.unit_type.value
        def_type = defender.unit_type.value
        alpha = self.COMBAT_EFFICIENCY.get(atk_type, {}).get(def_type, 1.0)

        # 역방향 (수비자의 반격 효율)
        beta = self.COMBAT_EFFICIENCY.get(def_type, {}).get(atk_type, 1.0)

        # 지형 방어 보너스 적용
        terrain_modifier = 1.0
        if defender_terrain:
            terrain_modifier = self.TERRAIN_DEFENSE_BONUS[defender_terrain]
        alpha /= terrain_modifier   # 공격자 불리

        # 전투력 지수 반영
        alpha *= attacker.combat_power * (1 + 0.2 * attacker.experience)
        beta  *= defender.combat_power  * (1 + 0.2 * defender.experience)

        # 노이즈 (전장 불확실성)
        alpha *= (1 + self.rng.normal(0, self.noise_std))
        beta  *= (1 + self.rng.normal(0, self.noise_std))
        alpha = max(0, alpha)
        beta  = max(0, beta)

        # 란체스터 적분 (단순 오일러)
        B = float(attacker.headcount)
        R = float(defender.headcount)
        dt = 0.1
        steps = int(duration / dt)

        for _ in range(steps):
            dB = -beta * R * dt
            dR = -alpha * B * dt
            B = max(0, B + dB)
            R = max(0, R + dR)

        attacker_losses = int(attacker.headcount - B)
        defender_losses = int(defender.headcount - R)

        # 상태 판정
        def determine_status(original: int, remaining: float) -> UnitStatus:
            ratio = remaining / max(original, 1)
            if ratio <= 0.05:   return UnitStatus.DESTROYED
            if ratio <= 0.3:    return UnitStatus.CRITICAL
            if ratio <= 0.7:    return UnitStatus.DEGRADED
            return UnitStatus.ACTIVE

        atk_status = determine_status(attacker.headcount, B)
        def_status = determine_status(defender.headcount, R)

        desc = (f"{attacker.unit_id}({attacker.unit_type.value}) → "
                f"{defender.unit_id}({defender.unit_type.value}): "
                f"공격자 손실={attacker_losses}, 수비자 손실={defender_losses}, "
                f"지형보너스={terrain_modifier:.2f}")

        return EngagementResult(
            attacker_id=attacker.unit_id,
            defender_id=defender.unit_id,
            attacker_losses=attacker_losses,
            defender_losses=defender_losses,
            attacker_new_status=atk_status,
            defender_new_status=def_status,
            engagement_duration=duration,
            terrain_modifier=terrain_modifier,
            description=desc
        )

    def run_step(
        self,
        kg: CombatKnowledgeGraph,
        action_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> BattleStep:
        """
        1 시뮬레이션 스텝 실행
        action_pairs: [(attacker_id, defender_id), ...]
        None이면 자동 교전 (범위 내 적 공격)
        """
        engagements = []
        events = []

        # 자동 교전 쌍 결정
        if action_pairs is None:
            action_pairs = self._auto_pair_units(kg)

        # 교전 실행
        casualties: Dict[str, int] = {}
        new_statuses: Dict[str, UnitStatus] = {}

        for atk_id, def_id in action_pairs:
            if atk_id not in kg.units or def_id not in kg.units:
                continue
            attacker = kg.units[atk_id]
            defender = kg.units[def_id]

            if attacker.status == UnitStatus.DESTROYED:
                continue

            # 수비자 지형 파악
            def_terrain = self._get_unit_terrain(kg, def_id)

            result = self.calculate_engagement(attacker, defender, def_terrain)
            engagements.append(result)
            events.append(result.description)

            # 누적 손실
            casualties[atk_id] = casualties.get(atk_id, 0) + result.attacker_losses
            casualties[def_id] = casualties.get(def_id, 0) + result.defender_losses

            # 상태 업데이트 (더 나쁜 상태로)
            self._update_worse_status(new_statuses, atk_id, result.attacker_new_status)
            self._update_worse_status(new_statuses, def_id, result.defender_new_status)

        # 유닛 상태 반영
        for uid, losses in casualties.items():
            unit = kg.units[uid]
            unit.headcount = max(0, unit.headcount - losses)
            if uid in new_statuses:
                unit.status = new_statuses[uid]
            # 사기 감소
            loss_ratio = losses / max(unit.headcount + losses, 1)
            unit.morale = max(0.1, unit.morale - 0.1 * loss_ratio)

        # 통계 집계
        blue_casualties = sum(v for k, v in casualties.items() if kg.units.get(k) and kg.units[k].alignment == ForceAlignment.BLUE)
        red_casualties  = sum(v for k, v in casualties.items() if kg.units.get(k) and kg.units[k].alignment == ForceAlignment.RED)

        blue_active = sum(1 for u in kg.units.values() if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED)
        red_active  = sum(1 for u in kg.units.values() if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED)
        blue_headcount = sum(u.headcount for u in kg.units.values() if u.alignment == ForceAlignment.BLUE)
        red_headcount  = sum(u.headcount for u in kg.units.values() if u.alignment == ForceAlignment.RED)

        # 임무 상태 판정
        mission = "ongoing"
        if blue_active == 0 or blue_headcount < 10:
            mission = "red_win"
            events.append("⚠️ Blue 전력 소진 → Red 승리")
        elif red_active == 0 or red_headcount < 10:
            mission = "blue_win"
            events.append("✅ Red 전력 소진 → Blue 승리")

        return BattleStep(
            timestep=0,
            engagements=engagements,
            blue_total_casualties=blue_casualties,
            red_total_casualties=red_casualties,
            blue_active_units=blue_active,
            red_active_units=red_active,
            blue_total_headcount=blue_headcount,
            red_total_headcount=red_headcount,
            mission_status=mission,
            events=events
        )

    def run_episode(
        self,
        kg: CombatKnowledgeGraph,
        max_steps: int = 50,
        blue_actions=None   # Optional callable: (kg) → action_pairs
    ) -> Tuple[List[BattleStep], str]:
        """
        에피소드 전체 실행
        returns: (steps_list, final_status)
        """
        steps = []
        for t in range(max_steps):
            if blue_actions:
                pairs = blue_actions(kg)
            else:
                pairs = None

            step = self.run_step(kg, pairs)
            step.timestep = t
            steps.append(step)

            if step.mission_status != "ongoing":
                break

        final_status = steps[-1].mission_status if steps else "draw"
        return steps, final_status

    def generate_dataset(
        self,
        n_scenarios: int = 1000,
        max_steps: int = 30,
        seed: int = 0
    ) -> List[Dict]:
        """
        훈련용 합성 데이터셋 생성
        각 시나리오의 결과 (승리/패배, 사상자 수) 레이블 포함
        """
        from ontology.combat_schema import ScenarioFactory
        dataset = []
        self.rng = np.random.RandomState(seed)

        for i in range(n_scenarios):
            n_blue = self.rng.randint(5, 12)
            n_red  = self.rng.randint(4, 10)
            kg = ScenarioFactory.create_standard_scenario(
                n_blue=n_blue, n_red=n_red, seed=seed + i
            )
            initial_stats = kg.get_stats()
            steps, final_status = self.run_episode(kg, max_steps=max_steps)

            total_blue_casualties = sum(s.blue_total_casualties for s in steps)
            total_red_casualties  = sum(s.red_total_casualties  for s in steps)

            dataset.append({
                "scenario_id": i,
                "initial_blue_units": n_blue,
                "initial_red_units": n_red,
                "initial_blue_power": initial_stats["blue_combat_power"],
                "initial_red_power": initial_stats["red_combat_power"],
                "blue_casualties": total_blue_casualties,
                "red_casualties": total_red_casualties,
                "final_blue_headcount": steps[-1].blue_total_headcount if steps else 0,
                "final_red_headcount":  steps[-1].red_total_headcount  if steps else 0,
                "winner": final_status,
                "n_steps": len(steps),
                "blue_win": 1 if final_status == "blue_win" else 0,
            })

        return dataset

    def _auto_pair_units(self, kg: CombatKnowledgeGraph) -> List[Tuple[str, str]]:
        """범위 내 가장 위협적인 적과 자동 교전 페어링"""
        pairs = []
        blue_units = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED]
        red_units  = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED]

        for blue in blue_units:
            # 사거리 내 적 중 전투력이 가장 높은 대상
            in_range = [r for r in red_units
                        if blue.position.distance_to(r.position) <= blue.capability.range_km]
            if in_range:
                target = max(in_range, key=lambda r: r.combat_power)
                pairs.append((blue.unit_id, target.unit_id))

        for red in red_units:
            in_range = [b for b in blue_units
                        if red.position.distance_to(b.position) <= red.capability.range_km]
            if in_range:
                target = max(in_range, key=lambda b: b.combat_power)
                pairs.append((red.unit_id, target.unit_id))

        return pairs

    def _get_unit_terrain(self, kg: CombatKnowledgeGraph, unit_id: str) -> Optional[TerrainType]:
        """유닛이 위치한 지형 유형 반환"""
        if unit_id not in kg.units:
            return None
        unit = kg.units[unit_id]
        closest = min(
            kg.terrain_cells.values(),
            key=lambda c: unit.position.distance_to(c.position),
            default=None
        )
        return closest.terrain_type if closest else None

    @staticmethod
    def _update_worse_status(status_dict: Dict, uid: str, new_status: UnitStatus):
        """더 나쁜 상태로만 업데이트"""
        order = [UnitStatus.ACTIVE, UnitStatus.DEGRADED, UnitStatus.CRITICAL, UnitStatus.DESTROYED]
        if uid not in status_dict:
            status_dict[uid] = new_status
        else:
            current_idx = order.index(status_dict[uid])
            new_idx = order.index(new_status)
            if new_idx > current_idx:
                status_dict[uid] = new_status


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    print("=== Lanchester Engine Test ===")
    engine = LanchesterEngine(seed=42)
    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)

    steps, result = engine.run_episode(kg, max_steps=20)
    print(f"✅ 에피소드 완료: {len(steps)} 스텝, 결과={result}")
    print(f"   최종 Blue 병력: {steps[-1].blue_total_headcount}")
    print(f"   최종 Red 병력: {steps[-1].red_total_headcount}")
    print(f"   Blue 총 사상: {sum(s.blue_total_casualties for s in steps)}")
    print(f"   Red 총 사상: {sum(s.red_total_casualties for s in steps)}")

    print("\n📊 데이터셋 생성 중 (100 시나리오)...")
    dataset = engine.generate_dataset(n_scenarios=100)
    blue_wins = sum(d["blue_win"] for d in dataset)
    avg_cas = np.mean([d["blue_casualties"] for d in dataset])
    print(f"✅ 데이터셋 생성 완료: Blue 승률={blue_wins/len(dataset)*100:.1f}%, 평균 사상={avg_cas:.1f}")
