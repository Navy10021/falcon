"""
fog_of_war.py
=============
전장 안개(Fog of War) 부분 관측 환경
- 관측 범위 기반 정보 필터링
- 불확실성 노이즈 주입
- Curriculum Learning용 단계별 안개 강도 조절
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
import copy

from ontology.combat_schema import (
    Unit, UnitType, UnitStatus, ForceAlignment, Position,
    Capability, CombatKnowledgeGraph
)


class FogLevel(Enum):
    """Fog of War 강도 레벨"""
    CLEAR = 0           # Perfect Observation (훈련 초기)
    LIGHT = 1           # 약한 안개 (위치 오차 ±20%)
    MODERATE = 2        # 중간 안개 (전력 불확실)
    HEAVY = 3           # 강한 안개 (적 위치 부분 미지)
    MAXIMUM = 4         # 최대 안개 + 적 기만 전술


@dataclass
class ObservationNoise:
    """관측 노이즈 설정"""
    position_error_km: float = 0.0     # 위치 오차 (km)
    headcount_error_ratio: float = 0.0 # 병력 수 오차 비율 [0, 1]
    combat_power_error: float = 0.0    # 전투력 지수 오차
    detection_prob: float = 1.0        # 적 유닛 탐지 확률 [0, 1]
    deception_prob: float = 0.0        # 허위 유닛 생성 확률 [0, 1]

    @classmethod
    def from_fog_level(cls, level: FogLevel) -> "ObservationNoise":
        params = {
            FogLevel.CLEAR:    (0.0,  0.0,  0.0,  1.00, 0.00),
            FogLevel.LIGHT:    (1.0,  0.10, 0.05, 0.90, 0.05),
            FogLevel.MODERATE: (3.0,  0.25, 0.15, 0.75, 0.10),
            FogLevel.HEAVY:    (6.0,  0.40, 0.30, 0.55, 0.20),
            FogLevel.MAXIMUM:  (10.0, 0.60, 0.50, 0.35, 0.35),
        }
        p, h, c, d, dc = params[level]
        return cls(p, h, c, d, dc)


class FogOfWarFilter:
    """
    Fog of War 필터
    실제 전장 상태 → 부분 관측 상태로 변환
    Blue 에이전트 시점에서의 관측만 처리
    """

    def __init__(self, fog_level: FogLevel = FogLevel.CLEAR,
                 observation_radius_km: float = 15.0,
                 seed: Optional[int] = None):
        self.fog_level = fog_level
        self.observation_radius_km = observation_radius_km
        self.noise = ObservationNoise.from_fog_level(fog_level)
        self.rng = np.random.RandomState(seed)
        self._ghost_units: Dict[str, Unit] = {}  # 허위 유닛 (기만 전술)

    def set_fog_level(self, level: FogLevel):
        self.fog_level = level
        self.noise = ObservationNoise.from_fog_level(level)

    def observe(self, true_kg: CombatKnowledgeGraph,
                observer_alignment: ForceAlignment = ForceAlignment.BLUE
               ) -> Tuple[CombatKnowledgeGraph, Dict[str, float]]:
        """
        실제 Knowledge Graph → 부분 관측 KG 반환
        Returns:
            observed_kg: 관측된 KG (노이즈 포함)
            uncertainty_map: {unit_id: uncertainty_score}
        """
        observed_kg = CombatKnowledgeGraph()
        uncertainty_map: Dict[str, float] = {}

        # 관측자 유닛 위치 수집 → numpy 배열로 사전 빌드 (O(N²) Python 루프 → O(N) 벡터화)
        obs_pos_list = [
            u.position for u in true_kg.units.values()
            if u.alignment == observer_alignment and u.status != UnitStatus.DESTROYED
        ]
        if obs_pos_list:
            # shape: [M, 3]  (M = 관측자 수)
            obs_pos_arr = np.array(
                [[p.x, p.y, p.z] for p in obs_pos_list], dtype=np.float32
            )
        else:
            obs_pos_arr = np.empty((0, 3), dtype=np.float32)

        # 지형 정보는 완전 관측 (지도 정보)
        for cell in true_kg.terrain_cells.values():
            observed_kg.add_terrain(cell)

        for uid, true_unit in true_kg.units.items():
            # 아군: 완전 관측
            if true_unit.alignment == observer_alignment:
                observed_kg.add_unit(copy.deepcopy(true_unit))
                uncertainty_map[uid] = 0.0
                continue

            # 적군: Fog of War 적용 (벡터화된 거리 계산 사용)
            if not self._is_detectable(true_unit, obs_pos_arr):
                # 탐지 불가 → 관측 제외
                uncertainty_map[uid] = 1.0
                continue

            # 관측 노이즈 적용
            noisy_unit, uncertainty = self._apply_noise(true_unit)
            observed_kg.add_unit(noisy_unit)
            uncertainty_map[uid] = uncertainty

        # 허위 유닛 추가 (기만 전술)
        if self.noise.deception_prob > 0:
            ghosts = self._generate_ghost_units(true_kg, observer_alignment)
            for ghost in ghosts:
                observed_kg.add_unit(ghost)
                uncertainty_map[ghost.unit_id] = 0.8  # 높은 불확실성

        # 관측된 공간 관계 재생성
        observed_kg.build_spatial_relations()

        return observed_kg, uncertainty_map

    def _is_detectable(self, unit: Unit, obs_pos_arr: np.ndarray) -> bool:
        """
        유닛 탐지 가능 여부 판단.
        obs_pos_arr: shape [M, 3] numpy 배열 (관측자 위치 사전 계산).
        O(N²) Python 루프 대신 numpy 벡터화로 O(M) 연산.
        """
        if obs_pos_arr.shape[0] == 0:
            return False

        # 벡터화 거리 계산: [M, 3] - [3] → [M]
        unit_pos = np.array(
            [unit.position.x, unit.position.y, unit.position.z], dtype=np.float32
        )
        diffs    = obs_pos_arr - unit_pos        # [M, 3]
        min_dist = float(np.sqrt((diffs ** 2).sum(axis=1)).min())

        if min_dist > self.observation_radius_km:
            return False

        # 거리 기반 탐지 확률
        distance_factor = 1.0 - (min_dist / self.observation_radius_km) * 0.5
        base_prob = self.noise.detection_prob * distance_factor

        return bool(self.rng.random() < base_prob)

    def _apply_noise(self, true_unit: Unit) -> Tuple[Unit, float]:
        """관측 노이즈 적용 → (노이즈 유닛, 불확실성 점수)"""
        noisy = copy.deepcopy(true_unit)

        # 위치 노이즈
        if self.noise.position_error_km > 0:
            dx = self.rng.normal(0, self.noise.position_error_km)
            dy = self.rng.normal(0, self.noise.position_error_km)
            noisy.position = Position(
                true_unit.position.x + dx,
                true_unit.position.y + dy,
                true_unit.position.z
            )

        # 병력 수 노이즈
        if self.noise.headcount_error_ratio > 0:
            error = self.rng.normal(1.0, self.noise.headcount_error_ratio)
            noisy.headcount = max(1, int(true_unit.headcount * error))

        # 전투력 노이즈 (capability 수준)
        if self.noise.combat_power_error > 0:
            cap = noisy.capability
            noise = self.noise.combat_power_error
            noisy.capability = Capability(
                firepower=float(np.clip(cap.firepower + self.rng.normal(0, noise), 0, 1)),
                mobility=float(np.clip(cap.mobility + self.rng.normal(0, noise), 0, 1)),
                protection=float(np.clip(cap.protection + self.rng.normal(0, noise), 0, 1)),
                range_km=max(0.5, cap.range_km + self.rng.normal(0, 1.0)),
                ammo_level=float(np.clip(cap.ammo_level + self.rng.normal(0, noise), 0, 1)),
                fuel_level=float(np.clip(cap.fuel_level + self.rng.normal(0, noise), 0, 1)),
                comms_quality=cap.comms_quality,
                ew_resistance=cap.ew_resistance
            )

        # 불확실성 점수 계산
        pos_uncertainty = min(self.noise.position_error_km / 10.0, 1.0)
        hc_uncertainty  = self.noise.headcount_error_ratio
        cp_uncertainty  = self.noise.combat_power_error * 2
        uncertainty = float(np.mean([pos_uncertainty, hc_uncertainty, cp_uncertainty]))

        return noisy, uncertainty

    def _generate_ghost_units(self, true_kg: CombatKnowledgeGraph,
                               observer_alignment: ForceAlignment) -> list:
        """허위 유닛 (기만 전술) 생성"""
        ghosts = []
        if self.rng.random() > self.noise.deception_prob:
            return ghosts

        # 기존 적군 유닛 중 하나를 복사해 위치 변경
        enemy_units = [u for u in true_kg.units.values()
                       if u.alignment != observer_alignment and u.status != UnitStatus.DESTROYED]

        if not enemy_units:
            return ghosts

        n_ghosts = self.rng.randint(1, 3)
        for i in range(n_ghosts):
            template = self.rng.choice(enemy_units)
            ghost = copy.deepcopy(template)
            ghost.unit_id = f"ghost_{template.unit_id}_{i}"

            # 랜덤 위치에 배치 (원본 근처)
            ghost.position = Position(
                template.position.x + self.rng.uniform(-8, 8),
                template.position.y + self.rng.uniform(-8, 8)
            )
            ghosts.append(ghost)

        return ghosts


class CurriculumScheduler:
    """
    Curriculum Learning 스케줄러
    학습 진행도에 따라 Fog Level 자동 조절
    """

    CURRICULUM = [
        (0.0,  0.2,  FogLevel.CLEAR,    "기본 전투 전략 학습"),
        (0.2,  0.5,  FogLevel.LIGHT,    "부분 관측 적응"),
        (0.5,  0.8,  FogLevel.MODERATE, "불확실성 인식 전략"),
        (0.8,  0.95, FogLevel.HEAVY,    "보수적 전략 발현"),
        (0.95, 1.0,  FogLevel.MAXIMUM,  "최대 불확실성 + 기만 대응"),
    ]

    def __init__(self):
        self.current_progress = 0.0
        self.current_level = FogLevel.CLEAR

    def update(self, progress: float) -> Tuple[FogLevel, str]:
        """
        학습 진행도 업데이트 → 현재 Fog Level 반환
        progress: [0.0, 1.0]
        """
        self.current_progress = np.clip(progress, 0.0, 1.0)
        for start, end, level, desc in self.CURRICULUM:
            if start <= self.current_progress < end:
                self.current_level = level
                return level, desc
        self.current_level = FogLevel.MAXIMUM
        return FogLevel.MAXIMUM, "최대 난이도"

    def get_fog_filter(self, seed: Optional[int] = None) -> FogOfWarFilter:
        return FogOfWarFilter(self.current_level, seed=seed)


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    print("=== Fog of War Filter Test ===")

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    stats = kg.get_stats()
    print(f"실제 Red 유닛: {stats['red_units']}개, 병력: {stats['red_headcount']}")

    for level in FogLevel:
        fog_filter = FogOfWarFilter(level, seed=42)
        observed_kg, uncertainty = fog_filter.observe(kg)
        obs_stats = observed_kg.get_stats()
        avg_uncertainty = np.mean(list(uncertainty.values())) if uncertainty else 0
        print(f"  [{level.name:8s}] 관측 Red유닛={obs_stats['red_units']}, "
              f"평균 불확실성={avg_uncertainty:.3f}")

    print("\n📚 Curriculum Scheduler Test")
    scheduler = CurriculumScheduler()
    for progress in [0.1, 0.3, 0.6, 0.85, 0.97]:
        level, desc = scheduler.update(progress)
        print(f"  진행도={progress:.0%} → {level.name}: {desc}")
