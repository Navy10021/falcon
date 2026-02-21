"""
multidomain.py
==============
⑫ 멀티 도메인 확장 (지상 + 공중 + 전자전)
- 도메인별 유닛 특성 확장
- 도메인 간 시너지/억제 관계
- 전자전(EW) → 적 GNN 불확실성 증폭
- 공중 우세 → 지상 전투력 배수

학술 연구용 합성 데이터
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, ForceAlignment, UnitStatus, CombatKnowledgeGraph
)


class DomainType(Enum):
    GROUND    = "ground"
    AIR       = "air"
    EW        = "electronic_warfare"
    MARITIME  = "maritime"


# 도메인별 유닛 분류
DOMAIN_UNITS: Dict[str, List[str]] = {
    "ground": ["infantry", "armor", "artillery", "engineer", "logistics"],
    "air":    ["aviation"],
    "ew":     ["electronic_warfare", "signal"],
    "maritime": [],
}


@dataclass
class DomainEffect:
    """도메인 간 시너지/억제 효과"""
    source_domain: str
    target_domain: str
    effect_type: str       # "synergy", "suppression", "denial"
    magnitude: float       # 효과 크기 [0, 1]
    condition: str         # 발동 조건 설명

    def apply(self, base_multiplier: float) -> float:
        if self.effect_type == "synergy":
            return base_multiplier * (1 + self.magnitude)
        elif self.effect_type == "suppression":
            return base_multiplier * (1 - self.magnitude * 0.5)
        return base_multiplier


# 도메인 효과 테이블
DOMAIN_EFFECT_TABLE: List[DomainEffect] = [
    DomainEffect("air",  "ground",  "synergy",    0.30, "공중 우세 → 지상 화력 30% 증가"),
    DomainEffect("ew",   "air",     "suppression", 0.40, "전자전 → 적 항공 탐지 40% 저하"),
    DomainEffect("ew",   "ground",  "suppression", 0.20, "전자전 → 적 지상 통신 20% 저하"),
    DomainEffect("air",  "ew",      "denial",      0.35, "공중 우세 → 적 EW 운용 35% 제한"),
    DomainEffect("ground","air",    "suppression", 0.15, "방공 → 적 항공 15% 억제"),
]


@dataclass
class MultiDomainState:
    """멀티 도메인 전장 상태"""
    air_superiority: float         # 공중 우세도 [-1, 1] (양수=아군 우세)
    ew_dominance: float            # EW 우세도 [-1, 1]
    ground_initiative: float       # 지상 주도권 [-1, 1]
    uncertainty_amplification: float  # EW → GNN 불확실성 증폭 배수

    def to_extension_vector(self) -> np.ndarray:
        """PPO 상태 벡터 확장용 (4D)"""
        return np.array([
            self.air_superiority,
            self.ew_dominance,
            self.ground_initiative,
            self.uncertainty_amplification,
        ], dtype=np.float32)


class MultiDomainAnalyzer:
    """
    멀티 도메인 전장 분석기
    - 각 도메인의 우세도 계산
    - 도메인 간 시너지 효과 계산
    - 전자전의 GNN 불확실성 증폭
    - 도메인별 전투력 조정 계수
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.history: List[MultiDomainState] = []

    def compute_domain_balance(self, kg: CombatKnowledgeGraph) -> Dict[str, Dict]:
        """각 도메인의 Blue/Red 전투력 균형 계산"""
        balance = {}
        for domain, unit_types in DOMAIN_UNITS.items():
            blue_cp = sum(
                u.combat_power for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE
                and u.unit_type.value in unit_types
                and u.status != UnitStatus.DESTROYED
            )
            red_cp = sum(
                u.combat_power for u in kg.units.values()
                if u.alignment == ForceAlignment.RED
                and u.unit_type.value in unit_types
                and u.status != UnitStatus.DESTROYED
            )
            total = blue_cp + red_cp + 1e-8
            balance[domain] = {
                "blue_cp": blue_cp,
                "red_cp": red_cp,
                "superiority": float((blue_cp - red_cp) / total),
            }
        return balance

    def compute_multidomain_state(self, kg: CombatKnowledgeGraph) -> MultiDomainState:
        """현재 KG → 멀티 도메인 상태"""
        balance = self.compute_domain_balance(kg)

        air_sup  = balance.get("air",  {"superiority": 0.0})["superiority"]
        ew_dom   = balance.get("ew",   {"superiority": 0.0})["superiority"]
        gnd_init = balance.get("ground", {"superiority": 0.0})["superiority"]

        # 전자전 우세도 → GNN 불확실성 증폭
        # Red EW 우세 시 Blue의 GNN 불확실성 증가
        ew_amplification = float(np.clip(1.0 - ew_dom * 0.5, 0.8, 1.8))

        state = MultiDomainState(
            air_superiority=float(air_sup),
            ew_dominance=float(ew_dom),
            ground_initiative=float(gnd_init),
            uncertainty_amplification=ew_amplification,
        )
        self.history.append(state)
        return state

    def apply_domain_effects(
        self, kg: CombatKnowledgeGraph, md_state: MultiDomainState
    ) -> Dict[str, float]:
        """
        도메인 효과를 전투력 배수로 변환
        Returns: {unit_type: multiplier}
        """
        multipliers: Dict[str, float] = {ut.value: 1.0 for ut in UnitType}

        # 공중 우세 → 지상 화력 강화
        if md_state.air_superiority > 0.1:
            for ut in DOMAIN_UNITS["ground"]:
                multipliers[ut] *= 1 + md_state.air_superiority * 0.30

        # EW 우세 → 통신/신호 억제
        if md_state.ew_dominance > 0.1:
            multipliers["signal"] *= max(0.3, 1 - md_state.ew_dominance * 0.4)

        # EW 열세 → 아군 전투력 저하
        if md_state.ew_dominance < -0.1:
            for ut in DOMAIN_UNITS["ground"]:
                multipliers[ut] *= max(0.7, 1 + md_state.ew_dominance * 0.2)

        # 항공 우세 → 적 EW 억제 (Blue 입장)
        if md_state.air_superiority > 0.2:
            multipliers["electronic_warfare"] *= 1.2

        return multipliers

    def get_uncertainty_multiplier(self, md_state: MultiDomainState) -> float:
        """GNN 불확실성 증폭 배수 반환"""
        return md_state.uncertainty_amplification

    def generate_domain_report(self, md_state: MultiDomainState) -> str:
        """멀티 도메인 상태 보고서"""
        def bar(v):
            val = (v + 1) / 2   # [-1,1] → [0,1]
            blue = int(val * 10)
            red  = 10 - blue
            return f"Blue{'█'*blue}{'░'*red}Red"

        lines = [
            "🌐 멀티 도메인 상태",
            f"  지상 주도권: {bar(md_state.ground_initiative)} ({md_state.ground_initiative:+.2f})",
            f"  공중 우세도: {bar(md_state.air_superiority)} ({md_state.air_superiority:+.2f})",
            f"  EW 우세도:   {bar(md_state.ew_dominance)} ({md_state.ew_dominance:+.2f})",
            f"  GNN 불확실성 증폭: ×{md_state.uncertainty_amplification:.2f}",
        ]
        return "\n".join(lines)

    def domain_trend(self, window: int = 5) -> Dict[str, str]:
        """최근 N 스텝의 도메인 트렌드"""
        if len(self.history) < 2:
            return {}
        recent = self.history[-window:]
        trends = {}
        for domain in ["air_superiority", "ew_dominance", "ground_initiative"]:
            vals = [getattr(s, domain) for s in recent]
            slope = vals[-1] - vals[0]
            trends[domain] = "↑ 상승" if slope > 0.05 else "↓ 하락" if slope < -0.05 else "→ 유지"
        return trends


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine

    print("=" * 55)
    print("⑫ 멀티 도메인 확장 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    analyzer = MultiDomainAnalyzer(seed=42)
    engine = MixedLanchesterEngine(seed=42)

    print("\n[도메인 균형 현황]")
    balance = analyzer.compute_domain_balance(kg)
    for domain, data in balance.items():
        sup = data["superiority"]
        status = "Blue 우세" if sup > 0.1 else "Red 우세" if sup < -0.1 else "균형"
        print(f"  {domain:10s}: Blue={data['blue_cp']:.2f}  "
              f"Red={data['red_cp']:.2f}  → {status}({sup:+.2f})")

    print("\n[10 스텝 멀티 도메인 추적]")
    for step in range(10):
        md_state = analyzer.compute_multidomain_state(kg)
        multipliers = analyzer.apply_domain_effects(kg, md_state)
        unc_mult = analyzer.get_uncertainty_multiplier(md_state)
        print(f"  스텝 {step+1:2d}: 공중={md_state.air_superiority:+.2f}  "
              f"EW={md_state.ew_dominance:+.2f}  "
              f"지상={md_state.ground_initiative:+.2f}  "
              f"불확실성×{unc_mult:.2f}")
        engine.run_step_mixed(kg)

    print("\n" + analyzer.generate_domain_report(analyzer.history[-1]))

    trends = analyzer.domain_trend(5)
    print(f"\n[5스텝 트렌드] {trends}")
    print("\n✅ 멀티 도메인 확장 정상 동작!")
