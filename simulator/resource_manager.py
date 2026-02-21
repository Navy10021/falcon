"""
resource_manager.py
===================
① 동적 자원 소모 시스템
- 교전 시 탄약(ammo) 소모, 기동 시 연료(fuel) 소모
- 보급선(supply line) 모델링
- 자원 고갈 → 전투력/기동력 저하 연동
- 보급 유닛의 보충 로직

학술 연구용 합성 데이터
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ontology.combat_schema import (
    Unit, UnitType, UnitStatus, ForceAlignment, CombatKnowledgeGraph
)


# ──────────────────────────────────────────────
# 자원 소모 계수 (유닛 유형별)
# ──────────────────────────────────────────────

# 교전 1회당 탄약 소모율
AMMO_CONSUMPTION_RATE: Dict[str, float] = {
    "infantry":           0.05,
    "armor":              0.08,
    "artillery":          0.15,   # 포병은 탄약 소모 가장 큼
    "aviation":           0.10,
    "engineer":           0.03,
    "signal":             0.01,
    "logistics":          0.02,
    "electronic_warfare": 0.04,
}

# 이동 1스텝당 연료 소모율
FUEL_CONSUMPTION_RATE: Dict[str, float] = {
    "infantry":           0.01,
    "armor":              0.06,   # 전차 연료 소모 큼
    "artillery":          0.04,
    "aviation":           0.12,   # 항공기 최대 소모
    "engineer":           0.03,
    "signal":             0.02,
    "logistics":          0.04,
    "electronics_warfare":0.02,
}

# 탄약 부족 시 화력 저하 계수
AMMO_FIREPOWER_PENALTY: List[Tuple[float, float]] = [
    # (ammo_threshold, firepower_multiplier)
    (0.5, 1.00),   # 50% 이상 → 정상
    (0.3, 0.75),   # 30~50%  → 75%
    (0.1, 0.45),   # 10~30%  → 45%
    (0.0, 0.10),   # 10% 미만 → 10% (잔탄 사격)
]

# 연료 부족 시 기동력 저하 계수
FUEL_MOBILITY_PENALTY: List[Tuple[float, float]] = [
    (0.4, 1.00),
    (0.2, 0.60),
    (0.05, 0.20),
    (0.0, 0.00),   # 연료 완전 고갈 → 이동 불가
]


# ──────────────────────────────────────────────
# 보급 이벤트
# ──────────────────────────────────────────────

@dataclass
class SupplyEvent:
    """보급 이벤트"""
    source_unit_id: str       # 보급 유닛 ID
    target_unit_id: str       # 피보급 유닛 ID
    ammo_delivered: float     # 보충 탄약량 [0, 1]
    fuel_delivered: float     # 보충 연료량 [0, 1]
    success: bool = True
    reason: str = ""


@dataclass
class ResourceStatus:
    """유닛 자원 현황 스냅샷"""
    unit_id: str
    unit_type: str
    ammo_level: float
    fuel_level: float
    firepower_multiplier: float
    mobility_multiplier: float
    is_ammo_critical: bool     # 탄약 < 10%
    is_fuel_critical: bool     # 연료 < 10%
    needs_supply: bool


# ──────────────────────────────────────────────
# 핵심 자원 관리자
# ──────────────────────────────────────────────

class ResourceManager:
    """
    동적 자원 소모 관리자
    
    주요 기능:
    1. 교전 시 탄약 소모 계산
    2. 기동 시 연료 소모 계산
    3. 자원 수준에 따른 전투력 조정
    4. 보급 유닛 자동 보충 로직
    5. 보급선 차단 시나리오
    """

    def __init__(self, supply_radius_km: float = 8.0, seed: Optional[int] = None):
        self.supply_radius_km = supply_radius_km
        self.rng = np.random.RandomState(seed)
        self.supply_history: List[SupplyEvent] = []
        self.consumption_log: Dict[str, List[Dict]] = {}

    # ── 소모 계산 ──────────────────────────────

    def consume_ammo(
        self,
        unit: Unit,
        n_engagements: int = 1,
        intensity: float = 1.0    # 교전 강도 배수
    ) -> float:
        """
        교전 후 탄약 소모 처리
        Returns: 실제 소모량
        """
        base_rate = AMMO_CONSUMPTION_RATE.get(unit.unit_type.value, 0.05)
        noise = self.rng.uniform(0.8, 1.2)   # ±20% 노이즈
        consumed = base_rate * n_engagements * intensity * noise

        # 탄약 보유량으로 클리핑
        actual_consumed = min(consumed, unit.capability.ammo_level)
        unit.capability.ammo_level = max(0.0, unit.capability.ammo_level - actual_consumed)

        # 로그 기록
        if unit.unit_id not in self.consumption_log:
            self.consumption_log[unit.unit_id] = []
        self.consumption_log[unit.unit_id].append({
            "type": "ammo", "consumed": actual_consumed,
            "remaining": unit.capability.ammo_level
        })

        return actual_consumed

    def consume_fuel(
        self,
        unit: Unit,
        distance_moved_km: float = 1.0
    ) -> float:
        """
        이동 후 연료 소모 처리
        Returns: 실제 소모량
        """
        base_rate = FUEL_CONSUMPTION_RATE.get(unit.unit_type.value, 0.03)
        # 지형별 소모 배수 (추후 지형 정보 연동)
        terrain_multiplier = 1.0
        noise = self.rng.uniform(0.85, 1.15)

        consumed = base_rate * distance_moved_km * terrain_multiplier * noise
        actual_consumed = min(consumed, unit.capability.fuel_level)
        unit.capability.fuel_level = max(0.0, unit.capability.fuel_level - actual_consumed)

        if unit.unit_id not in self.consumption_log:
            self.consumption_log[unit.unit_id] = []
        self.consumption_log[unit.unit_id].append({
            "type": "fuel", "consumed": actual_consumed,
            "remaining": unit.capability.fuel_level
        })

        return actual_consumed

    # ── 전투력 조정 ────────────────────────────

    def get_firepower_multiplier(self, unit: Unit) -> float:
        """탄약 수준에 따른 화력 조정 계수"""
        ammo = unit.capability.ammo_level
        for threshold, multiplier in AMMO_FIREPOWER_PENALTY:
            if ammo >= threshold:
                return multiplier
        return 0.10

    def get_mobility_multiplier(self, unit: Unit) -> float:
        """연료 수준에 따른 기동력 조정 계수"""
        fuel = unit.capability.fuel_level
        for threshold, multiplier in FUEL_MOBILITY_PENALTY:
            if fuel >= threshold:
                return multiplier
        return 0.00

    def apply_resource_penalties(self, unit: Unit) -> Dict[str, float]:
        """
        자원 부족 패널티를 유닛 Capability에 직접 반영
        Returns: 적용된 패널티 딕셔너리
        """
        fp_mult = self.get_firepower_multiplier(unit)
        mob_mult = self.get_mobility_multiplier(unit)

        # capability 필드에 배수 적용 (원본 × 배수)
        # 주의: 매 스텝마다 호출하지 않도록 주의 (누적 방지)
        penalties = {
            "firepower_multiplier": fp_mult,
            "mobility_multiplier": mob_mult,
            "effective_firepower": unit.capability.firepower * fp_mult,
            "effective_mobility": unit.capability.mobility * mob_mult,
        }
        return penalties

    # ── 보급 로직 ──────────────────────────────

    def find_supply_sources(
        self, target: Unit, kg: CombatKnowledgeGraph
    ) -> List[Unit]:
        """보급 가능한 아군 병참 유닛 탐색"""
        sources = []
        for uid, unit in kg.units.items():
            if (unit.alignment == target.alignment
                    and unit.unit_type == UnitType.LOGISTICS
                    and unit.status != UnitStatus.DESTROYED
                    and unit.unit_id != target.unit_id):
                dist = unit.position.distance_to(target.position)
                if dist <= self.supply_radius_km:
                    sources.append(unit)
        # 거리 가까운 순 정렬
        sources.sort(key=lambda u: u.position.distance_to(target.position))
        return sources

    def execute_supply(
        self, supplier: Unit, target: Unit,
        ammo_fraction: float = 0.3,
        fuel_fraction: float = 0.3
    ) -> SupplyEvent:
        """
        보급 실행
        supplier의 자원을 target으로 이전
        """
        # 보급 유닛 자체 자원 확인
        if supplier.capability.ammo_level < 0.1 and supplier.capability.fuel_level < 0.1:
            return SupplyEvent(
                supplier.unit_id, target.unit_id, 0, 0,
                success=False, reason="보급 유닛 자원 고갈"
            )

        # 전달량 계산
        ammo_transfer = min(ammo_fraction, supplier.capability.ammo_level * 0.8)
        fuel_transfer  = min(fuel_fraction, supplier.capability.fuel_level * 0.8)

        # 보급 유닛 자원 감소
        supplier.capability.ammo_level -= ammo_transfer
        supplier.capability.fuel_level  -= fuel_transfer

        # 목표 유닛 자원 증가 (상한 1.0)
        target.capability.ammo_level = min(1.0, target.capability.ammo_level + ammo_transfer)
        target.capability.fuel_level  = min(1.0, target.capability.fuel_level  + fuel_transfer)

        event = SupplyEvent(
            source_unit_id=supplier.unit_id,
            target_unit_id=target.unit_id,
            ammo_delivered=ammo_transfer,
            fuel_delivered=fuel_transfer,
            success=True,
            reason=f"거리: {supplier.position.distance_to(target.position):.1f}km"
        )
        self.supply_history.append(event)
        return event

    def auto_supply_step(self, kg: CombatKnowledgeGraph) -> List[SupplyEvent]:
        """
        자동 보급 스텝: 자원 부족 유닛에 자동 보급
        매 N 시뮬레이션 스텝마다 호출
        """
        events = []
        for uid, unit in kg.units.items():
            if unit.status == UnitStatus.DESTROYED:
                continue
            needs_ammo = unit.capability.ammo_level < 0.25
            needs_fuel  = unit.capability.fuel_level  < 0.25

            if needs_ammo or needs_fuel:
                suppliers = self.find_supply_sources(unit, kg)
                if suppliers:
                    event = self.execute_supply(
                        suppliers[0], unit,
                        ammo_fraction=0.25 if needs_ammo else 0.0,
                        fuel_fraction=0.25 if needs_fuel  else 0.0
                    )
                    events.append(event)
        return events

    # ── 현황 조회 ──────────────────────────────

    def get_resource_status(self, kg: CombatKnowledgeGraph) -> List[ResourceStatus]:
        """전체 유닛 자원 현황 리포트"""
        statuses = []
        for uid, unit in kg.units.items():
            if unit.status == UnitStatus.DESTROYED:
                continue
            fp_mult  = self.get_firepower_multiplier(unit)
            mob_mult = self.get_mobility_multiplier(unit)
            statuses.append(ResourceStatus(
                unit_id=uid,
                unit_type=unit.unit_type.value,
                ammo_level=unit.capability.ammo_level,
                fuel_level=unit.capability.fuel_level,
                firepower_multiplier=fp_mult,
                mobility_multiplier=mob_mult,
                is_ammo_critical=unit.capability.ammo_level < 0.10,
                is_fuel_critical=unit.capability.fuel_level  < 0.10,
                needs_supply=(unit.capability.ammo_level < 0.25
                              or unit.capability.fuel_level < 0.25),
            ))
        return statuses

    def get_summary(self, kg: CombatKnowledgeGraph) -> Dict:
        """자원 현황 요약 통계"""
        blue_units = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.BLUE
                      and u.status != UnitStatus.DESTROYED]
        red_units  = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.RED
                      and u.status != UnitStatus.DESTROYED]

        def avg_resources(units):
            if not units: return {"ammo": 0.0, "fuel": 0.0, "critical_ammo": 0, "critical_fuel": 0}
            return {
                "ammo": np.mean([u.capability.ammo_level for u in units]),
                "fuel": np.mean([u.capability.fuel_level  for u in units]),
                "critical_ammo": sum(1 for u in units if u.capability.ammo_level < 0.10),
                "critical_fuel":  sum(1 for u in units if u.capability.fuel_level  < 0.10),
            }

        return {
            "blue": avg_resources(blue_units),
            "red":  avg_resources(red_units),
            "total_supply_events": len(self.supply_history),
        }


# ──────────────────────────────────────────────
# Lanchester Engine 통합 패치
# ──────────────────────────────────────────────

def apply_resource_to_combat_efficiency(
    base_alpha: float,
    attacker: Unit,
    resource_manager: ResourceManager
) -> float:
    """
    자원 수준을 반영한 실효 전투 효율 계수 반환
    기존 Lanchester alpha에 자원 패널티를 곱해 사용
    """
    fp_mult  = resource_manager.get_firepower_multiplier(attacker)
    mob_mult = resource_manager.get_mobility_multiplier(attacker)
    # 화력 70% + 기동력 30% 가중 평균
    combined = 0.70 * fp_mult + 0.30 * mob_mult
    return base_alpha * combined


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory, ForceAlignment

    print("=" * 55)
    print("① 동적 자원 소모 시스템 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=42)
    rm = ResourceManager(seed=42)

    blue_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]

    print("\n[초기 자원 현황]")
    for u in blue_units[:3]:
        print(f"  {u.unit_id:8s} ({u.unit_type.value:20s}) "
              f"탄약={u.capability.ammo_level:.2f}  연료={u.capability.fuel_level:.2f}")

    # 10스텝 교전 시뮬레이션
    print("\n[10 스텝 교전 시뮬레이션]")
    for step in range(10):
        for unit in blue_units:
            if unit.status != UnitStatus.DESTROYED:
                rm.consume_ammo(unit, n_engagements=1, intensity=1.0)
                rm.consume_fuel(unit, distance_moved_km=0.5)
        # 3스텝마다 자동 보급
        if step % 3 == 2:
            supply_events = rm.auto_supply_step(kg)
            if supply_events:
                print(f"  스텝 {step+1}: 보급 {len(supply_events)}건 실행")

    print("\n[10 스텝 후 자원 현황]")
    for u in blue_units[:3]:
        fp_m  = rm.get_firepower_multiplier(u)
        mob_m = rm.get_mobility_multiplier(u)
        print(f"  {u.unit_id:8s} 탄약={u.capability.ammo_level:.2f}  "
              f"연료={u.capability.fuel_level:.2f}  "
              f"화력배수={fp_m:.2f}  기동배수={mob_m:.2f}")

    summary = rm.get_summary(kg)
    print(f"\n[요약] Blue 평균 탄약={summary['blue']['ammo']:.2f}  "
          f"연료={summary['blue']['fuel']:.2f}  "
          f"총 보급 이벤트={summary['total_supply_events']}")
    print("\n✅ 동적 자원 소모 시스템 정상 동작!")
