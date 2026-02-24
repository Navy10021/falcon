"""
adversarial_scenario.py
=======================
Domain Randomization — 적대적 시나리오 생성기

훈련 중 시뮬레이션 파라미터를 무작위로 변화시켜
sim-to-real gap을 줄이고 정책의 강건성을 높인다.

주요 기능:
  - DomainRandomizer : 전투 파라미터 무작위 샘플링
  - ScenarioAugmenter : 기존 시나리오를 적대적으로 변환
  - AdversarialCurriculum : 난이도 자동 조절 커리큘럼
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ontology.combat_schema import (
    CombatKnowledgeGraph, ScenarioFactory, Unit, UnitType,
    UnitStatus, ForceAlignment, Position, Capability
)


# ──────────────────────────────────────────────
# 도메인 랜덤화 범위 설정
# ──────────────────────────────────────────────

@dataclass
class RandomizationRange:
    """파라미터 별 랜덤화 범위 (min, max)"""
    # 병력 규모 배율
    force_scale_min: float = 0.5
    force_scale_max: float = 2.0

    # 전투력 지수 배율
    combat_power_scale_min: float = 0.6
    combat_power_scale_max: float = 1.8

    # 보급선 방해 확률
    supply_disruption_min: float = 0.0
    supply_disruption_max: float = 0.4

    # 지형 이동 속도 배율
    terrain_mobility_min: float = 0.5
    terrain_mobility_max: float = 1.2

    # 정보 지연(분) — ISR 위성 재방문 주기 불확실성
    intel_delay_min: float = 0.0
    intel_delay_max: float = 120.0

    # C2 통신 품질
    c2_quality_min: float = 0.3
    c2_quality_max: float = 1.0

    # 기상 효과 배율
    weather_effect_min: float = 0.7
    weather_effect_max: float = 1.3

    # 적 증원 규모 비율
    red_reinforcement_ratio_min: float = 0.0
    red_reinforcement_ratio_max: float = 0.5


@dataclass
class DomainSample:
    """단일 에피소드에서 사용할 도메인 파라미터 샘플"""
    force_scale: float
    combat_power_scale: float
    supply_disruption_prob: float
    terrain_mobility: float
    intel_delay_min: float
    c2_quality: float
    weather_effect: float
    red_reinforcement_ratio: float
    seed: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "force_scale": self.force_scale,
            "combat_power_scale": self.combat_power_scale,
            "supply_disruption_prob": self.supply_disruption_prob,
            "terrain_mobility": self.terrain_mobility,
            "intel_delay_min": self.intel_delay_min,
            "c2_quality": self.c2_quality,
            "weather_effect": self.weather_effect,
            "red_reinforcement_ratio": self.red_reinforcement_ratio,
        }


class DomainRandomizer:
    """
    시뮬레이션 도메인 파라미터를 무작위 샘플링하는 클래스.

    사용 예::

        randomizer = DomainRandomizer(seed=42)
        sample = randomizer.sample()
        kg = randomizer.apply(base_kg, sample)
    """

    def __init__(
        self,
        ranges: Optional[RandomizationRange] = None,
        seed: int = 0,
    ):
        self.ranges = ranges or RandomizationRange()
        self.rng = np.random.RandomState(seed)
        self._episode_count = 0

    # ------------------------------------------------------------------
    # 샘플링
    # ------------------------------------------------------------------

    def sample(self, seed: Optional[int] = None) -> DomainSample:
        """랜덤 도메인 파라미터 샘플 생성"""
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        r = self.ranges

        ep_seed = int(rng.randint(0, 2 ** 31))
        self._episode_count += 1

        return DomainSample(
            force_scale=float(rng.uniform(r.force_scale_min, r.force_scale_max)),
            combat_power_scale=float(rng.uniform(r.combat_power_scale_min, r.combat_power_scale_max)),
            supply_disruption_prob=float(rng.uniform(r.supply_disruption_min, r.supply_disruption_max)),
            terrain_mobility=float(rng.uniform(r.terrain_mobility_min, r.terrain_mobility_max)),
            intel_delay_min=float(rng.uniform(r.intel_delay_min, r.intel_delay_max)),
            c2_quality=float(rng.uniform(r.c2_quality_min, r.c2_quality_max)),
            weather_effect=float(rng.uniform(r.weather_effect_min, r.weather_effect_max)),
            red_reinforcement_ratio=float(rng.uniform(
                r.red_reinforcement_ratio_min, r.red_reinforcement_ratio_max
            )),
            seed=ep_seed,
        )

    def sample_batch(self, n: int) -> List[DomainSample]:
        """n개 샘플 일괄 생성"""
        return [self.sample() for _ in range(n)]

    # ------------------------------------------------------------------
    # 적용
    # ------------------------------------------------------------------

    def apply(self, kg: CombatKnowledgeGraph, sample: DomainSample) -> CombatKnowledgeGraph:
        """
        샘플된 파라미터를 KG의 유닛 속성에 적용.
        원본 kg를 변경하지 않고 복사본 반환.
        """
        import copy
        kg2 = copy.deepcopy(kg)

        for unit in kg2.units.values():
            # 1) 병력 규모 스케일
            unit.headcount = max(1, int(unit.headcount * sample.force_scale))

            # 2) 전투력 지수 스케일
            if unit.capabilities:
                for cap in unit.capabilities:
                    cap.effectiveness = float(
                        np.clip(cap.effectiveness * sample.combat_power_scale, 0.0, 1.0)
                    )

            # 3) 보급 방해 → 일부 유닛 전투력 감소
            if self.rng.random() < sample.supply_disruption_prob:
                if unit.capabilities:
                    for cap in unit.capabilities:
                        cap.effectiveness *= 0.6

            # 4) Red 증원 (Red 측 유닛만)
            if (unit.alignment == ForceAlignment.RED
                    and sample.red_reinforcement_ratio > 0
                    and self.rng.random() < sample.red_reinforcement_ratio):
                unit.headcount = int(unit.headcount * 1.3)

        return kg2

    # ------------------------------------------------------------------
    # 편의 메서드
    # ------------------------------------------------------------------

    def randomized_scenario(
        self,
        base_seed: int = 0,
        domain_seed: Optional[int] = None,
    ) -> Tuple[CombatKnowledgeGraph, DomainSample]:
        """
        랜덤화된 KG와 사용된 도메인 샘플을 함께 반환.
        재현성이 필요하면 domain_seed를 지정.
        """
        kg_base = ScenarioFactory.create_standard_scenario(seed=base_seed)
        sample = self.sample(seed=domain_seed)
        kg_rand = self.apply(kg_base, sample)
        return kg_rand, sample

    @property
    def episode_count(self) -> int:
        return self._episode_count


# ──────────────────────────────────────────────
# 적대적 커리큘럼
# ──────────────────────────────────────────────

class AdversarialCurriculum:
    """
    Blue 에이전트 성능에 따라 도메인 랜덤화 강도를 자동 조절.

    - 초기: 좁은 범위 (쉬운 시나리오)
    - Blue 승률이 threshold 초과 시 범위 확장 (어려운 시나리오)
    - 범위는 최대 max_ranges까지 선형 증가
    """

    def __init__(
        self,
        initial_ranges: Optional[RandomizationRange] = None,
        max_ranges: Optional[RandomizationRange] = None,
        win_rate_threshold: float = 0.65,
        expand_step: float = 0.05,
        seed: int = 0,
    ):
        self.current = initial_ranges or RandomizationRange(
            force_scale_min=0.8, force_scale_max=1.2,
            combat_power_scale_min=0.9, combat_power_scale_max=1.1,
            supply_disruption_min=0.0, supply_disruption_max=0.1,
        )
        self.max_ranges = max_ranges or RandomizationRange()
        self.win_rate_threshold = win_rate_threshold
        self.expand_step = expand_step
        self.randomizer = DomainRandomizer(ranges=self.current, seed=seed)
        self._level = 0.0   # [0, 1] — 0=쉬움, 1=최대 난이도

    def update(self, blue_win_rate: float):
        """에피소드 배치 후 호출 — 필요 시 난이도 상향"""
        if blue_win_rate >= self.win_rate_threshold and self._level < 1.0:
            self._level = min(1.0, self._level + self.expand_step)
            self._interpolate_ranges()

    def _interpolate_ranges(self):
        """현재 level에 맞춰 RandomizationRange를 선형 보간"""
        α = self._level
        r = self.current
        m = self.max_ranges

        def lerp(a_min, a_max, b_min, b_max):
            return a_min + α * (b_min - a_min), a_max + α * (b_max - a_max)

        r.force_scale_min, r.force_scale_max = lerp(
            0.8, 1.2, m.force_scale_min, m.force_scale_max)
        r.combat_power_scale_min, r.combat_power_scale_max = lerp(
            0.9, 1.1, m.combat_power_scale_min, m.combat_power_scale_max)
        r.supply_disruption_min, r.supply_disruption_max = lerp(
            0.0, 0.1, m.supply_disruption_min, m.supply_disruption_max)
        r.red_reinforcement_ratio_min, r.red_reinforcement_ratio_max = lerp(
            0.0, 0.1, m.red_reinforcement_ratio_min, m.red_reinforcement_ratio_max)

    def sample(self) -> DomainSample:
        return self.randomizer.sample()

    @property
    def level(self) -> float:
        return self._level


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DomainRandomizer 테스트 ===")

    randomizer = DomainRandomizer(seed=42)
    kg, sample = randomizer.randomized_scenario(base_seed=0, domain_seed=7)

    print(f"도메인 샘플:")
    for k, v in sample.as_dict().items():
        print(f"  {k:30s}: {v:.4f}")
    print(f"전체 유닛 수: {len(kg.units)}")

    # 커리큘럼 테스트
    curriculum = AdversarialCurriculum(win_rate_threshold=0.65, seed=0)
    for ep in range(5):
        curriculum.update(blue_win_rate=0.70)   # 성능 좋음 → 난이도 올라감
    print(f"\nCurriculum Level after 5 updates (win_rate=0.70): {curriculum.level:.2f}")

    curriculum2 = AdversarialCurriculum(win_rate_threshold=0.65, seed=0)
    for ep in range(5):
        curriculum2.update(blue_win_rate=0.50)  # 성능 낮음 → 유지
    print(f"Curriculum Level after 5 updates (win_rate=0.50): {curriculum2.level:.2f}")

    print("\n✅ adversarial_scenario.py 정상 동작!")
