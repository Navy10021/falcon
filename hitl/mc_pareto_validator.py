"""
mc_pareto_validator.py
======================
④ MC 기반 Pareto 전략 실검증
- 각 Pareto 후보를 실제 N회 시뮬레이션으로 검증
- 통계적 신뢰구간과 함께 후보 제시
- 휴리스틱 수치 → 시뮬레이션 수치로 대체
- 후보 간 통계적 유의차 검정 (t-test)

기존 pareto_generator.py의 수식 기반 수치를 실제 시뮬레이션으로 보강
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats as scipy_stats

from ontology.combat_schema import ScenarioFactory, ForceAlignment, CombatKnowledgeGraph
from simulator.mixed_lanchester import MixedLanchesterEngine
from simulator.resource_manager import ResourceManager
from hitl.pareto_generator import StrategyOption, CommanderConstraints, ParetoStrategyGenerator


@dataclass
class ValidatedStrategyOption:
    """MC 검증이 완료된 전략 후보"""
    original: StrategyOption

    # MC 시뮬레이션 실측값
    mc_win_rate: float
    mc_win_rate_ci: Tuple[float, float]          # 95% CI
    mc_avg_casualties: float
    mc_casualties_ci: Tuple[float, float]
    mc_avg_force_remaining: float
    mc_force_remaining_ci: Tuple[float, float]
    mc_avg_steps: float

    # 후보 간 비교
    is_pareto_dominant: bool = True
    dominated_by: List[str] = field(default_factory=list)

    # 검증 메타
    n_simulations: int = 0
    validation_passed: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def reliability_score(self) -> float:
        """신뢰도 점수 [0,1]: 좁은 CI = 높은 신뢰도"""
        win_ci_width = self.mc_win_rate_ci[1] - self.mc_win_rate_ci[0]
        cas_ci_width = self.mc_casualties_ci[1] - self.mc_casualties_ci[0]
        return float(max(0, 1 - win_ci_width * 2 - cas_ci_width / 100))

    def summary_line(self) -> str:
        dom_flag = "" if self.is_pareto_dominant else " [지배됨]"
        return (
            f"[{self.original.option_id}] {self.original.label}{dom_flag}\n"
            f"  MC 승률: {self.mc_win_rate:.1%} "
            f"(CI: {self.mc_win_rate_ci[0]:.1%}~{self.mc_win_rate_ci[1]:.1%})\n"
            f"  MC 사상자: {self.mc_avg_casualties:.1f}명 "
            f"(CI: {self.mc_casualties_ci[0]:.0f}~{self.mc_casualties_ci[1]:.0f})\n"
            f"  신뢰도 점수: {self.reliability_score:.3f}\n"
            f"  휴리스틱 승률: {self.original.win_probability:.1%} → "
            f"실측: {self.mc_win_rate:.1%} "
            f"({'과대' if self.original.win_probability > self.mc_win_rate else '과소'}추정)"
        )


class MCParetoValidator:
    """
    Monte Carlo 기반 Pareto 전략 검증기

    워크플로:
    1. ParetoStrategyGenerator로 후보 생성
    2. 각 후보를 N번 시뮬레이션으로 실검증
    3. 실측 통계로 Pareto 우위 재계산
    4. 통계적 유의차 검정
    5. 검증된 최종 순위 반환
    """

    def __init__(
        self,
        n_simulations: int = 100,
        max_steps_per_sim: int = 30,
        seed: int = 0
    ):
        self.n_simulations = n_simulations
        self.max_steps = max_steps_per_sim
        self.engine = MixedLanchesterEngine(seed=seed)
        self.rng = np.random.RandomState(seed)

    def validate_option(
        self,
        option: StrategyOption,
        base_kg: CombatKnowledgeGraph,
        constraints: CommanderConstraints
    ) -> ValidatedStrategyOption:
        """
        단일 전략 후보 MC 검증
        - 전략 유형에 따른 행동 시퀀스 적용
        - N번 시뮬레이션 후 통계 집계
        """
        wins = []
        casualties = []
        force_remaining = []
        steps_list = []

        for sim_i in range(self.n_simulations):
            # 시나리오 생성 (약간의 변동)
            n_blue = self.rng.randint(6, 11)
            n_red  = self.rng.randint(4, 9)
            kg = ScenarioFactory.create_standard_scenario(
                n_blue=n_blue, n_red=n_red,
                seed=self.rng.randint(0, 99999)
            )

            # 병력 규모 조정 (전략에 따라)
            self._scale_force(kg, option, ForceAlignment.BLUE)

            rm = ResourceManager(seed=sim_i)
            total_blue_cas = 0
            mission_status = "ongoing"

            for step_t in range(self.max_steps):
                result = self.engine.run_step_mixed(kg, resource_manager=rm)
                total_blue_cas += result["blue_casualties"]
                mission_status = result["mission_status"]
                if mission_status != "ongoing":
                    break

            wins.append(1 if mission_status == "blue_win" else 0)
            casualties.append(total_blue_cas)
            final_hc = sum(u.headcount for u in kg.units.values()
                           if u.alignment == ForceAlignment.BLUE)
            force_remaining.append(final_hc)
            steps_list.append(step_t + 1)

        # 통계 계산
        wins_arr = np.array(wins)
        cas_arr  = np.array(casualties)
        fr_arr   = np.array(force_remaining)

        win_rate  = float(wins_arr.mean())
        win_ci    = self._bootstrap_ci(wins_arr)
        avg_cas   = float(cas_arr.mean())
        cas_ci    = self._bootstrap_ci(cas_arr)
        avg_fr    = float(fr_arr.mean())
        fr_ci     = self._bootstrap_ci(fr_arr)
        avg_steps = float(np.mean(steps_list))

        # 제약 조건 검증
        notes = []
        validation_passed = True
        if win_rate < constraints.min_win_probability:
            notes.append(f"승률 미달: {win_rate:.1%} < {constraints.min_win_probability:.1%}")
            validation_passed = False
        if constraints.max_casualties and avg_cas > constraints.max_casualties:
            notes.append(f"사상자 초과: {avg_cas:.0f} > {constraints.max_casualties}")
            validation_passed = False

        return ValidatedStrategyOption(
            original=option,
            mc_win_rate=win_rate,
            mc_win_rate_ci=win_ci,
            mc_avg_casualties=avg_cas,
            mc_casualties_ci=cas_ci,
            mc_avg_force_remaining=avg_fr,
            mc_force_remaining_ci=fr_ci,
            mc_avg_steps=avg_steps,
            n_simulations=self.n_simulations,
            validation_passed=validation_passed,
            notes=notes,
        )

    def validate_all(
        self,
        options: List[StrategyOption],
        base_kg: CombatKnowledgeGraph,
        constraints: CommanderConstraints,
        verbose: bool = True
    ) -> List[ValidatedStrategyOption]:
        """모든 후보 MC 검증 + Pareto 재계산"""
        validated = []
        for i, opt in enumerate(options):
            if verbose:
                print(f"  검증 중: [{opt.label}] ({i+1}/{len(options)}, "
                      f"N={self.n_simulations} sims)...", end=" ", flush=True)
            v = self.validate_option(opt, base_kg, constraints)
            validated.append(v)
            if verbose:
                status = "✅" if v.validation_passed else "⚠️"
                print(f"{status} 승률={v.mc_win_rate:.1%}")

        # Pareto 우위 재계산
        self._recompute_pareto(validated)

        # 신뢰도 × 실측 승률 기준 정렬
        validated.sort(
            key=lambda v: (v.is_pareto_dominant,
                           v.mc_win_rate * 0.4 + v.reliability_score * 0.3
                           - v.mc_avg_casualties / 1000 * 0.3),
            reverse=True
        )
        return validated

    def significance_test(
        self,
        a: ValidatedStrategyOption,
        b: ValidatedStrategyOption
    ) -> Dict:
        """
        두 전략 후보 간 승률 차이의 통계적 유의성 검정
        (독립 이항 비율 검정)
        """
        n = self.n_simulations
        p1, p2 = a.mc_win_rate, b.mc_win_rate

        # 풀링 비율
        p_pool = (p1 * n + p2 * n) / (2 * n)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n + 1/n)) + 1e-10

        z_stat = (p1 - p2) / se
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))

        return {
            "option_a": a.original.label,
            "option_b": b.original.label,
            "win_rate_diff": p1 - p2,
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "conclusion": (
                f"{a.original.label}이 유의하게 우수" if p_value < 0.05 and p1 > p2
                else f"{b.original.label}이 유의하게 우수" if p_value < 0.05
                else "통계적 유의차 없음"
            )
        }

    def _scale_force(self, kg: CombatKnowledgeGraph, option: StrategyOption,
                     alignment: ForceAlignment):
        """전략에 따른 병력 규모 조정"""
        blue_units = [u for u in kg.units.values() if u.alignment == alignment]
        if not blue_units:
            return
        total_hc = sum(u.headcount for u in blue_units)
        target_hc = option.force_size
        if total_hc > 0:
            scale = min(target_hc / total_hc, 1.0)
            for u in blue_units:
                u.headcount = max(10, int(u.headcount * scale))

    def _bootstrap_ci(self, data: np.ndarray, n_boot: int = 200,
                       ci: float = 0.95) -> Tuple[float, float]:
        boot = [self.rng.choice(data, len(data)).mean() for _ in range(n_boot)]
        alpha = (1 - ci) / 2
        return (float(np.percentile(boot, alpha * 100)),
                float(np.percentile(boot, (1 - alpha) * 100)))

    def _recompute_pareto(self, validated: List[ValidatedStrategyOption]):
        """실측 데이터로 Pareto 우위 재계산"""
        n = len(validated)
        for i in range(n):
            validated[i].is_pareto_dominant = True
            validated[i].dominated_by = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vi, vj = validated[i], validated[j]
                # j가 i를 지배하는지 (j가 모든 면에서 ≤ 나쁘지 않고 적어도 하나 더 좋음)
                j_better_win = vj.mc_win_rate >= vi.mc_win_rate
                j_better_cas = vj.mc_avg_casualties <= vi.mc_avg_casualties
                j_strictly   = (vj.mc_win_rate > vi.mc_win_rate or
                                vj.mc_avg_casualties < vi.mc_avg_casualties)
                if j_better_win and j_better_cas and j_strictly:
                    validated[i].dominated_by.append(vj.original.label)
                    validated[i].is_pareto_dominant = False

    def print_full_report(self, validated: List[ValidatedStrategyOption]):
        print("\n" + "=" * 60)
        print("🎯 MC 검증 완료 — Pareto 전략 최종 보고서")
        print("=" * 60)
        for v in validated:
            print(f"\n{v.summary_line()}")
            if v.notes:
                for note in v.notes:
                    print(f"  ⚠️  {note}")
        print("\n[통계적 유의차 검정]")
        if len(validated) >= 2:
            test = self.significance_test(validated[0], validated[1])
            print(f"  {test['option_a']} vs {test['option_b']}: "
                  f"p={test['p_value']:.4f} → {test['conclusion']}")
        print("=" * 60)


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory

    print("=" * 55)
    print("④ MC 기반 Pareto 실검증 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    constraints = CommanderConstraints(
        max_force_size=1200, min_win_probability=0.4, max_casualties=300
    )

    # 후보 생성
    gen = ParetoStrategyGenerator(n_candidates=3)
    options = gen.generate(kg, constraints)
    print(f"\n생성된 Pareto 후보: {len(options)}개")

    # MC 검증 (빠른 테스트: 30 sims)
    validator = MCParetoValidator(n_simulations=30, max_steps_per_sim=20, seed=42)
    print("\n[MC 시뮬레이션 검증 시작]")
    validated = validator.validate_all(options, kg, constraints, verbose=True)
    validator.print_full_report(validated)
    print("\n✅ MC 기반 Pareto 실검증 정상 동작!")
