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
            cap = unit.capability  # Capability 단일 객체

            # 1) 병력 규모 스케일
            unit.headcount = max(1, int(unit.headcount * sample.force_scale))

            # 2) 전투력 지수 스케일 (firepower / mobility / protection 비례 조정)
            scale = float(np.clip(sample.combat_power_scale, 0.3, 2.5))
            cap.firepower   = float(np.clip(cap.firepower   * scale, 0.0, 1.0))
            cap.mobility    = float(np.clip(cap.mobility    * scale, 0.0, 1.0))
            cap.protection  = float(np.clip(cap.protection  * scale, 0.0, 1.0))

            # 3) 보급 방해 → 탄약·연료 수준 감소
            if self.rng.random() < sample.supply_disruption_prob:
                cap.ammo_level  = float(np.clip(cap.ammo_level  * 0.5, 0.0, 1.0))
                cap.fuel_level  = float(np.clip(cap.fuel_level  * 0.5, 0.0, 1.0))

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
# ACCEL (Adaptive Curriculum via Emergent Complexity)
# Parker-Holder et al., NeurIPS 2022
# ──────────────────────────────────────────────

@dataclass
class ScenarioRecord:
    """
    ACCEL 버퍼에 저장되는 시나리오 기록.
    성과 추적으로 학습 신호가 가장 큰 시나리오를 우선 샘플링.
    """
    domain_sample: DomainSample
    base_seed: int
    success_rate: float = 0.5   # 초기 추정값
    n_evaluations: int = 0
    regret: float = 0.0         # 어려움 지표 (적대 - 아군 성능)


@dataclass
class ACCELConfig:
    """ACCEL 스케줄러 설정"""
    buffer_capacity: int = 500          # 시나리오 버퍼 최대 크기
    # 학습 신호가 큰 구간 (중간 난이도)
    target_success_min: float = 0.30    # 너무 쉬우면(> max) 제외
    target_success_max: float = 0.70    # 너무 어려우면(< min) 제외
    regret_weight: float = 0.5          # 샘플링 가중치에서 regret 기여도
    mutation_std: float = 0.05          # 도메인 파라미터 변형 크기
    n_mutations: int = 3                # 한 시나리오에서 생성할 변형 수
    min_evaluations: int = 3            # 신뢰 있는 성과 추정에 필요한 최소 평가 횟수


class ACCELScheduler:
    """
    ACCEL (Adaptive Curriculum via Emergent Complexity) 스케줄러.

    동작 원리:
      1. 시나리오 버퍼에서 에이전트가 가장 어렵게 느끼는 시나리오 샘플링
         (성공률 30~70% 구간 — 학습 신호 최대)
      2. 샘플된 시나리오를 소폭 변형하여 새 후보 생성
      3. 에이전트 성능 변화 측정 → 학습 신호가 큰 시나리오 유지
      4. 성공률 편향 구간 벗어나면 버퍼에서 제거

    사용::
        scheduler = ACCELScheduler(randomizer, seed=42)
        # 초기 시나리오 채우기
        scheduler.initialize(n_initial=50)
        # 학습 루프
        scenario = scheduler.sample()
        kg, sample = scenario.domain_sample, ...
        # 에피소드 실행 후 결과 기록
        scheduler.update(scenario, blue_won=True)
        # 변형 시나리오 추가
        scheduler.mutate_and_add(scenario, n=3)
    """

    def __init__(
        self,
        randomizer: DomainRandomizer,
        cfg: Optional[ACCELConfig] = None,
        seed: int = 0,
    ):
        self.randomizer = randomizer
        self.cfg        = cfg or ACCELConfig()
        self.rng        = np.random.RandomState(seed)
        self.buffer: List[ScenarioRecord] = []
        self._total_sampled = 0

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def initialize(self, n_initial: int = 50):
        """무작위 시나리오로 버퍼 초기화."""
        for i in range(n_initial):
            sample = self.randomizer.sample()
            self.buffer.append(ScenarioRecord(domain_sample=sample, base_seed=i))

    # ------------------------------------------------------------------
    # 시나리오 샘플링
    # ------------------------------------------------------------------

    def _compute_weights(self) -> np.ndarray:
        """
        각 시나리오의 샘플링 가중치 계산.
        목표 성공률 구간 내 시나리오를 우선 샘플링.
        """
        cfg = self.cfg
        weights = np.zeros(len(self.buffer), dtype=np.float64)

        for i, rec in enumerate(self.buffer):
            sr = rec.success_rate
            # 목표 구간 내면 가중치 부여
            in_target = cfg.target_success_min <= sr <= cfg.target_success_max
            if not in_target:
                # 구간 밖이면 낮은 가중치 (완전히 제외하지 않음)
                dist_to_target = min(
                    abs(sr - cfg.target_success_min),
                    abs(sr - cfg.target_success_max),
                )
                weights[i] = max(1e-3, 0.1 * np.exp(-dist_to_target * 5))
            else:
                # 구간 내: regret가 높을수록 우선
                regret_bonus = 1.0 + cfg.regret_weight * rec.regret
                weights[i]   = regret_bonus

        total = weights.sum()
        if total < 1e-9:
            weights = np.ones(len(self.buffer)) / len(self.buffer)
        else:
            weights /= total
        return weights

    def sample(self) -> ScenarioRecord:
        """가중치 기반 시나리오 샘플링."""
        if not self.buffer:
            sample = self.randomizer.sample()
            return ScenarioRecord(domain_sample=sample, base_seed=0)
        weights = self._compute_weights()
        idx     = self.rng.choice(len(self.buffer), p=weights)
        self._total_sampled += 1
        return self.buffer[idx]

    def sample_kg(self) -> Tuple[CombatKnowledgeGraph, ScenarioRecord]:
        """시나리오 샘플 + KG 생성을 한 번에."""
        rec = self.sample()
        kg  = ScenarioFactory.create_standard_scenario(seed=rec.base_seed)
        kg  = self.randomizer.apply(kg, rec.domain_sample)
        return kg, rec

    # ------------------------------------------------------------------
    # 결과 기록
    # ------------------------------------------------------------------

    def update(self, record: ScenarioRecord, blue_won: bool, regret: float = 0.0):
        """
        에피소드 결과로 시나리오 성과 업데이트.

        Parameters
        ----------
        record : ScenarioRecord
            샘플된 시나리오 기록
        blue_won : bool
            Blue(아군) 승리 여부
        regret : float
            적대 에이전트 성능 - 아군 성능 (양수일수록 Blue에게 불리)
        """
        n = record.n_evaluations + 1
        record.success_rate  = (record.success_rate * record.n_evaluations + float(blue_won)) / n
        record.n_evaluations = n
        record.regret        = (record.regret * (n - 1) + regret) / n

    # ------------------------------------------------------------------
    # 시나리오 변형
    # ------------------------------------------------------------------

    def mutate_and_add(self, base_record: ScenarioRecord, n: Optional[int] = None):
        """
        기존 시나리오를 소폭 변형하여 n개의 새 시나리오를 버퍼에 추가.
        버퍼가 가득 차면 가중치가 가장 낮은 시나리오 제거.
        """
        n = n or self.cfg.n_mutations
        cfg = self.cfg

        for _ in range(n):
            ds = base_record.domain_sample
            std = cfg.mutation_std

            # 각 파라미터를 가우시안 노이즈로 변형 + 클리핑
            new_ds = DomainSample(
                force_scale=float(np.clip(ds.force_scale + self.rng.normal(0, std), 0.3, 3.0)),
                combat_power_scale=float(np.clip(ds.combat_power_scale + self.rng.normal(0, std), 0.3, 2.5)),
                supply_disruption_prob=float(np.clip(ds.supply_disruption_prob + self.rng.normal(0, std * 0.5), 0.0, 0.6)),
                terrain_mobility=float(np.clip(ds.terrain_mobility + self.rng.normal(0, std * 0.5), 0.3, 1.5)),
                intel_delay_min=float(np.clip(ds.intel_delay_min + self.rng.normal(0, 10.0), 0.0, 180.0)),
                c2_quality=float(np.clip(ds.c2_quality + self.rng.normal(0, std), 0.1, 1.0)),
                weather_effect=float(np.clip(ds.weather_effect + self.rng.normal(0, std * 0.3), 0.5, 1.5)),
                red_reinforcement_ratio=float(np.clip(ds.red_reinforcement_ratio + self.rng.normal(0, std * 0.3), 0.0, 0.7)),
                seed=int(self.rng.randint(0, 2 ** 31)),
            )
            new_rec = ScenarioRecord(
                domain_sample=new_ds,
                base_seed=base_record.base_seed,
                success_rate=base_record.success_rate,  # 부모 성과 상속
            )
            self._add_to_buffer(new_rec)

    def _add_to_buffer(self, record: ScenarioRecord):
        """버퍼 용량 초과 시 가장 낮은 가중치 제거 후 추가."""
        if len(self.buffer) >= self.cfg.buffer_capacity:
            weights = self._compute_weights()
            remove_idx = int(np.argmin(weights))
            self.buffer.pop(remove_idx)
        self.buffer.append(record)

    # ------------------------------------------------------------------
    # 버퍼 관리
    # ------------------------------------------------------------------

    def prune_easy_hard(self):
        """
        성과가 너무 높거나(쉬움) 너무 낮은(불가능) 시나리오 제거.
        충분한 평가 횟수를 갖춘 시나리오만 제거 대상.
        """
        cfg = self.cfg
        self.buffer = [
            rec for rec in self.buffer
            if (rec.n_evaluations < cfg.min_evaluations
                or cfg.target_success_min <= rec.success_rate <= cfg.target_success_max)
        ]

    @property
    def stats(self) -> Dict[str, float]:
        if not self.buffer:
            return {"n_scenarios": 0}
        success_rates = [r.success_rate for r in self.buffer]
        evaluated     = [r for r in self.buffer if r.n_evaluations >= self.cfg.min_evaluations]
        return {
            "n_scenarios":       float(len(self.buffer)),
            "n_evaluated":       float(len(evaluated)),
            "mean_success_rate": float(np.mean(success_rates)),
            "std_success_rate":  float(np.std(success_rates)),
            "n_in_target":       float(sum(
                self.cfg.target_success_min <= r.success_rate <= self.cfg.target_success_max
                for r in self.buffer
            )),
            "total_sampled":     float(self._total_sampled),
        }


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

    # ACCEL 스케줄러 테스트
    print("\n=== ACCEL 스케줄러 테스트 ===")
    rng = np.random.RandomState(0)
    accel = ACCELScheduler(randomizer=DomainRandomizer(seed=1), seed=0)
    accel.initialize(n_initial=30)
    print(f"초기 버퍼 크기: {len(accel.buffer)}")

    # 20 에피소드 시뮬레이션
    for ep in range(20):
        rec  = accel.sample()
        won  = bool(rng.random() > 0.5)
        regret = float(rng.uniform(0, 1))
        accel.update(rec, blue_won=won, regret=regret)
        if ep % 5 == 4:
            accel.mutate_and_add(rec)

    accel.prune_easy_hard()
    st = accel.stats
    print(f"에피소드 20회 후:")
    for k, v in st.items():
        print(f"  {k:25s}: {v:.1f}")

    print("\n✅ adversarial_scenario.py 정상 동작!")
