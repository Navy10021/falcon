"""
counterfactual.py - 반사실 설명 생성 (⑧)
"병력 10% 감소 시 승률 X%p 하락" 형태 설명 자동 생성
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

from ontology.combat_schema import (
    ScenarioFactory, CombatKnowledgeGraph, ForceAlignment, UnitStatus
)
from simulator.mixed_lanchester import MixedLanchesterEngine
from simulator.resource_manager import ResourceManager


@dataclass
class CounterfactualVariable:
    name: str
    label: str
    base_value: float
    delta: float
    is_ratio: bool = True
    direction: str = "both"


@dataclass
class CounterfactualResult:
    variable: CounterfactualVariable
    direction: str
    base_win_rate: float
    cf_win_rate: float
    base_casualties: float
    cf_casualties: float
    base_steps: float
    cf_steps: float
    n_runs: int

    @property
    def win_rate_delta(self) -> float:
        return self.cf_win_rate - self.base_win_rate

    @property
    def casualty_delta(self) -> float:
        return self.cf_casualties - self.base_casualties

    @property
    def is_beneficial(self) -> bool:
        return self.win_rate_delta > 0.02 or self.casualty_delta < -2

    def natural_language(self) -> str:
        var_label = self.variable.label
        sign = "증가" if self.direction == "up" else "감소"
        delta_pct = abs(self.variable.delta) * 100

        if self.variable.is_ratio:
            trigger = f"{var_label}을 {delta_pct:.0f}% {sign}시키면"
        else:
            trigger = f"{var_label}을 {abs(self.variable.delta):.2f} {sign}시키면"

        effects = []
        wr_d = self.win_rate_delta * 100
        if abs(wr_d) >= 1.0:
            effects.append(f"승률 {abs(wr_d):.1f}%p {'상승' if wr_d > 0 else '하락'}")
        cas_d = self.casualty_delta
        if abs(cas_d) >= 1.0:
            effects.append(f"사상자 {abs(cas_d):.0f}명 {'감소' if cas_d < 0 else '증가'}")
        step_d = self.cf_steps - self.base_steps
        if abs(step_d) >= 1.0:
            effects.append(f"임무시간 {abs(step_d):.0f}스텝 {'단축' if step_d < 0 else '연장'}")
        if not effects:
            effects.append("유의미한 변화 없음")

        assessment = "✅ 권장" if self.is_beneficial else "⚠️ 비권장"
        return f"  {assessment}  {trigger} → {', '.join(effects)}"


class CounterfactualAnalyzer:
    DEFAULT_VARIABLES = [
        CounterfactualVariable("force_size", "병력 규모",  0.0, 0.10, True,  "both"),
        CounterfactualVariable("ammo_level", "탄약 수준",  0.0, 0.20, False, "up"),
        CounterfactualVariable("fuel_level", "연료 수준",  0.0, 0.20, False, "up"),
        CounterfactualVariable("morale",     "사기",       0.0, 0.15, False, "both"),
        CounterfactualVariable("experience", "훈련도",     0.0, 0.20, False, "up"),
    ]

    def __init__(self, n_runs: int = 30, max_steps: int = 25, seed: int = 0):
        self.n_runs = n_runs
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

    def _run_simulation(self, kg: CombatKnowledgeGraph, seed_offset: int = 0) -> Dict:
        wins, casualties, steps_list = [], [], []
        for run in range(self.n_runs):
            kg_c = copy.deepcopy(kg)
            engine = MixedLanchesterEngine(seed=run + seed_offset)
            rm = ResourceManager(seed=run)
            total_cas = 0
            status = "ongoing"
            for step in range(self.max_steps):
                result = engine.run_step_mixed(kg_c, resource_manager=rm)
                total_cas += result["blue_casualties"]
                status = result["mission_status"]
                if status != "ongoing":
                    break
            wins.append(1 if status == "blue_win" else 0)
            casualties.append(total_cas)
            steps_list.append(step + 1)
        return {
            "win_rate":   float(np.mean(wins)),
            "casualties": float(np.mean(casualties)),
            "steps":      float(np.mean(steps_list)),
        }

    def _apply_counterfactual(self, kg, var: CounterfactualVariable, direction: str):
        kg_cf = copy.deepcopy(kg)
        blues = [u for u in kg_cf.units.values()
                 if u.alignment == ForceAlignment.BLUE
                 and u.status != UnitStatus.DESTROYED]
        d = var.delta if direction == "up" else -var.delta
        for u in blues:
            if var.name == "force_size":
                scale = 1 + d
                u.headcount = max(5, int(u.headcount * scale))
            elif var.name == "ammo_level":
                u.capability.ammo_level = float(np.clip(u.capability.ammo_level + d, 0, 1))
            elif var.name == "fuel_level":
                u.capability.fuel_level = float(np.clip(u.capability.fuel_level + d, 0, 1))
            elif var.name == "morale":
                u.morale = float(np.clip(u.morale + d, 0, 1))
            elif var.name == "experience":
                u.experience = float(np.clip(u.experience + d, 0, 1))
        return kg_cf

    def analyze_variable(self, kg, var, base_result=None) -> List[CounterfactualResult]:
        if base_result is None:
            base_result = self._run_simulation(kg)
        directions = ["up", "down"] if var.direction == "both" else [var.direction]
        results = []
        for direction in directions:
            kg_cf = self._apply_counterfactual(kg, var, direction)
            cf_result = self._run_simulation(kg_cf, seed_offset=100)
            results.append(CounterfactualResult(
                variable=var, direction=direction,
                base_win_rate=base_result["win_rate"],
                cf_win_rate=cf_result["win_rate"],
                base_casualties=base_result["casualties"],
                cf_casualties=cf_result["casualties"],
                base_steps=base_result["steps"],
                cf_steps=cf_result["steps"],
                n_runs=self.n_runs,
            ))
        return results

    def full_analysis(self, kg, variables=None, verbose=True) -> List[CounterfactualResult]:
        if variables is None:
            variables = self.DEFAULT_VARIABLES
        if verbose:
            print("  [기준 시뮬레이션]", end=" ", flush=True)
        base = self._run_simulation(kg)
        if verbose:
            print(f"완료 (승률={base['win_rate']:.1%})")
        all_results = []
        for var in variables:
            if verbose:
                print(f"  [{var.label}] 반사실 계산...", end=" ", flush=True)
            all_results.extend(self.analyze_variable(kg, var, base))
            if verbose:
                print("완료")
        all_results.sort(
            key=lambda r: abs(r.win_rate_delta) + abs(r.casualty_delta) / 50,
            reverse=True
        )
        return all_results

    def variable_importance(self, results: List[CounterfactualResult]) -> List[Tuple[str, float]]:
        importance: Dict[str, float] = {}
        for r in results:
            key = r.variable.label
            impact = abs(r.win_rate_delta) * 2 + abs(r.casualty_delta) / 20
            importance[key] = max(importance.get(key, 0), impact)
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)

    def print_report(self, results: List[CounterfactualResult], top_n: int = 8):
        print("\n" + "=" * 58)
        print("📊 반사실 분석 보고서")
        print("=" * 58)
        if results:
            print(f"기준: 승률={results[0].base_win_rate:.1%}  "
                  f"사상자={results[0].base_casualties:.0f}명\n")
        print("[반사실 시나리오 (영향도 내림차순)]")
        for r in results[:top_n]:
            print(r.natural_language())
        importance = self.variable_importance(results)
        print("\n[변수 중요도]")
        for label, score in importance:
            bar = "█" * int(score * 15)
            print(f"  {label:12s}: {score:.3f}  [{bar}]")
        print("=" * 58)


if __name__ == "__main__":
    print("=" * 55)
    print("⑧ 반사실 설명 생성 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    analyzer = CounterfactualAnalyzer(n_runs=15, max_steps=18, seed=42)

    variables = [
        CounterfactualVariable("force_size", "병력 규모", 0.0, 0.15, True,  "both"),
        CounterfactualVariable("ammo_level", "탄약 수준", 0.0, 0.20, False, "up"),
        CounterfactualVariable("morale",     "사기",      0.0, 0.20, False, "both"),
    ]

    print("\n[반사실 분석 실행]")
    results = analyzer.full_analysis(kg, variables, verbose=True)
    analyzer.print_report(results, top_n=6)
    print("\n✅ 반사실 설명 생성 정상 동작!")
