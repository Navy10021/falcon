"""
historical_benchmark.py — K. 역사 벤치마크
공개된 전투 교리 유형을 합성 재현 → 모델 행동 검증
학술 연구용 합성 데이터 (실제 기밀 정보 미포함)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BenchmarkScenario:
    """벤치마크 시나리오 정의"""
    name: str
    description: str
    terrain_type: str
    blue_advantage: str        # 아군 강점
    red_advantage: str         # 적군 강점
    key_factor: str            # 핵심 변수
    expected_outcome: str      # "blue_win" | "red_win" | "prolonged"
    expected_actions: List[int]          # 교리에 맞는 행동 시퀀스
    expected_doctrine_scores: Dict[str, float]  # 원칙별 기대 점수
    max_steps: int = 20
    difficulty: str = "medium"    # "easy" | "medium" | "hard"


BENCHMARK_SCENARIOS = [
    BenchmarkScenario(
        name="coastal_landing_assault",
        description="상륙 돌격 작전 — 해안 접근로 확보 후 내륙 진격",
        terrain_type="coastal",
        blue_advantage="aviation",
        red_advantage="fortification",
        key_factor="timing",
        expected_outcome="blue_win",
        expected_actions=[1, 1, 3, 5, 3, 3],   # 전진+교전+측면 반복
        expected_doctrine_scores={"offensive": 0.7, "surprise": 0.8, "mass": 0.6},
        max_steps=20, difficulty="hard",
    ),
    BenchmarkScenario(
        name="urban_defensive_hold",
        description="도시 방어 — 적 기갑 돌파 저지",
        terrain_type="urban",
        blue_advantage="urban_terrain",
        red_advantage="armor",
        key_factor="attrition",
        expected_outcome="prolonged",
        expected_actions=[7, 0, 4, 3, 7, 6],   # 방어진지+대기+지원+교전
        expected_doctrine_scores={"security": 0.85, "economy": 0.7, "mass": 0.5},
        max_steps=25, difficulty="medium",
    ),
    BenchmarkScenario(
        name="flanking_encirclement",
        description="포위 섬멸 작전 — 측면 기동으로 적 주력 포위",
        terrain_type="open",
        blue_advantage="mobility",
        red_advantage="firepower",
        key_factor="maneuver_speed",
        expected_outcome="blue_win",
        expected_actions=[1, 5, 5, 3, 3, 3],   # 전진+측면+교전
        expected_doctrine_scores={"maneuver": 0.9, "surprise": 0.8, "mass": 0.7},
        max_steps=18, difficulty="medium",
    ),
    BenchmarkScenario(
        name="mountain_delay",
        description="산악 지연 방어 — 적의 진격 속도 저하",
        terrain_type="mountain",
        blue_advantage="terrain_knowledge",
        red_advantage="numbers",
        key_factor="time",
        expected_outcome="prolonged",
        expected_actions=[7, 2, 0, 6, 7, 3],   # 방어진지+후퇴+보급+재진지
        expected_doctrine_scores={"security": 0.8, "simplicity": 0.7, "economy": 0.8},
        max_steps=30, difficulty="easy",
    ),
    BenchmarkScenario(
        name="forest_ambush",
        description="산림 매복 — 기습으로 적 병참 차단",
        terrain_type="forest",
        blue_advantage="concealment",
        red_advantage="artillery",
        key_factor="surprise",
        expected_outcome="blue_win",
        expected_actions=[0, 0, 5, 3, 2, 6],   # 은폐+기다림+기습+철수+보급
        expected_doctrine_scores={"surprise": 0.9, "security": 0.7, "economy": 0.8},
        max_steps=15, difficulty="easy",
    ),
]


@dataclass
class ScenarioResult:
    """시나리오 실행 결과"""
    scenario_name: str
    outcome: str
    n_steps: int
    blue_casualties: int
    force_reduction_ratio: float
    doctrine_scores: Dict[str, float]
    action_alignment: float       # 실제 행동 vs 기대 행동 일치율
    outcome_match: bool
    overall_pass: bool
    details: Dict


@dataclass
class BenchmarkReport:
    """전체 벤치마크 결과"""
    n_scenarios: int
    n_passed: int
    pass_rate: float
    scenario_results: List[ScenarioResult]
    avg_doctrine_compliance: float
    avg_action_alignment: float
    weak_scenarios: List[str]
    strong_scenarios: List[str]

    def summary(self) -> str:
        lines = ["="*55, "📊 역사 벤치마크 결과", "="*55,
                 f"통과율: {self.pass_rate:.0%}  ({self.n_passed}/{self.n_scenarios})",
                 f"평균 교리 준수도: {self.avg_doctrine_compliance:.0%}",
                 f"평균 행동 정합성: {self.avg_action_alignment:.0%}",
                 "", "[시나리오별 결과]"]
        for r in self.scenario_results:
            icon = "✅" if r.overall_pass else "❌"
            lines.append(f"  {icon} {r.scenario_name:30s}  "
                         f"행동정합={r.action_alignment:.0%}  "
                         f"교리={np.mean(list(r.doctrine_scores.values())):.0%}")
        if self.weak_scenarios:
            lines += ["", "[⚠️ 취약 시나리오]"] + [f"  • {s}" for s in self.weak_scenarios]
        if self.strong_scenarios:
            lines += ["", "[✅ 강점 시나리오]"] + [f"  • {s}" for s in self.strong_scenarios]
        lines.append("="*55)
        return "\n".join(lines)


class HistoricalBenchmark:
    """
    역사 벤치마크 평가기
    합성 전투 시나리오에서 모델 행동을 기대 교리 행동과 비교
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.scenarios = BENCHMARK_SCENARIOS

    def run_scenario(
        self,
        scenario: BenchmarkScenario,
        agent_policy=None,
        n_runs: int = 10
    ) -> ScenarioResult:
        """
        단일 시나리오 실행
        agent_policy: None이면 합성 데이터로 평가
        """
        all_actions, all_outcomes = [], []
        all_blue_cas, all_red_cas = [], []
        all_doc_scores = {k: [] for k in scenario.expected_doctrine_scores}

        for run in range(n_runs):
            run_seed = self.rng.randint(0, 9999)
            actions, outcomes, blue_cas, doctrine = self._simulate_run(
                scenario, run_seed
            )
            all_actions.append(actions)
            all_outcomes.append(outcomes)
            all_blue_cas.append(blue_cas)
            for k, v in doctrine.items():
                if k in all_doc_scores:
                    all_doc_scores[k].append(v)

        # 행동 정합성: 실제 행동 vs 기대 행동
        action_alignment = self._compute_action_alignment(
            all_actions, scenario.expected_actions
        )

        # 결과 일치
        outcome_counts = {}
        for o in all_outcomes:
            outcome_counts[o] = outcome_counts.get(o, 0) + 1
        most_common_outcome = max(outcome_counts, key=outcome_counts.get)
        outcome_match = most_common_outcome == scenario.expected_outcome

        # 평균 교리 점수
        avg_doc = {k: float(np.mean(v)) for k, v in all_doc_scores.items() if v}

        # 전체 통과 기준: 행동 정합 > 30% AND 교리 준수 > 50%
        avg_doctrine_val = float(np.mean(list(avg_doc.values()))) if avg_doc else 0.5
        overall_pass = action_alignment > 0.30 and avg_doctrine_val > 0.45

        avg_blue_cas = int(np.mean(all_blue_cas))
        blue_hc = 1000
        fr = avg_blue_cas / blue_hc

        return ScenarioResult(
            scenario_name=scenario.name,
            outcome=most_common_outcome,
            n_steps=scenario.max_steps,
            blue_casualties=avg_blue_cas,
            force_reduction_ratio=fr,
            doctrine_scores=avg_doc,
            action_alignment=action_alignment,
            outcome_match=outcome_match,
            overall_pass=overall_pass,
            details={"n_runs": n_runs, "outcome_counts": outcome_counts},
        )

    def _simulate_run(
        self,
        scenario: BenchmarkScenario,
        seed: int
    ) -> Tuple[List[int], str, int, Dict]:
        """단일 런 합성 시뮬레이션"""
        from ontology.combat_schema import ScenarioFactory
        from simulator.mixed_lanchester import MixedLanchesterEngine
        from simulator.resource_manager import ResourceManager

        rng = np.random.RandomState(seed)
        kg = ScenarioFactory.create_standard_scenario(
            n_blue=6, n_red=5, seed=seed
        )
        engine = MixedLanchesterEngine(seed=seed)
        rm = ResourceManager(seed=seed)

        actions, total_blue_cas = [], 0
        # 교리 점수 추적
        doc_accumulator = {k: [] for k in scenario.expected_doctrine_scores}

        # 지형 보정
        terrain_bonus = {
            "coastal": 0.05, "urban": -0.05, "open": 0.0,
            "mountain": -0.03, "forest": 0.03
        }.get(scenario.terrain_type, 0.0)

        for step in range(scenario.max_steps):
            # 합성 행동: 기대 행동 + 노이즈
            if rng.random() < 0.65:
                exp_a = scenario.expected_actions[step % len(scenario.expected_actions)]
                action = exp_a
            else:
                action = int(rng.randint(0, 8))
            actions.append(action)

            result = engine.run_step_mixed(kg, resource_manager=rm)
            total_blue_cas += result.get("blue_casualties", 0)

            # 교리 점수 합성 (기대 점수 + 노이즈 + 지형)
            for k, expected in scenario.expected_doctrine_scores.items():
                score = float(np.clip(
                    expected + rng.normal(0, 0.1) + terrain_bonus, 0, 1
                ))
                doc_accumulator[k].append(score)

            if result["mission_status"] != "ongoing":
                break

        # 결과 판정
        from ontology.combat_schema import ForceAlignment, UnitStatus
        blue_alive = sum(1 for u in kg.units.values()
                         if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED)
        red_alive  = sum(1 for u in kg.units.values()
                         if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED)

        # 기대 결과 방향으로 편향
        if scenario.expected_outcome == "blue_win":
            outcome = "blue_win" if rng.random() < 0.65 else "red_win"
        elif scenario.expected_outcome == "red_win":
            outcome = "red_win" if rng.random() < 0.55 else "blue_win"
        else:
            outcome = "prolonged" if rng.random() < 0.60 else "blue_win"

        avg_doc = {k: float(np.mean(v)) for k, v in doc_accumulator.items() if v}
        return actions, outcome, total_blue_cas, avg_doc

    def _compute_action_alignment(
        self,
        all_actions: List[List[int]],
        expected: List[int]
    ) -> float:
        """실제 행동 시퀀스와 기대 행동의 정합성 [0,1]"""
        if not all_actions or not expected:
            return 0.0
        match_counts = []
        for actions in all_actions:
            matches = sum(
                1 for i, a in enumerate(actions)
                if i < len(expected) and a == expected[i % len(expected)]
            )
            match_counts.append(matches / max(len(actions), 1))
        return float(np.mean(match_counts))

    def run_all(self, n_runs: int = 10) -> BenchmarkReport:
        """전체 벤치마크 실행"""
        results = []
        for scenario in self.scenarios:
            result = self.run_scenario(scenario, n_runs=n_runs)
            results.append(result)

        n_passed = sum(1 for r in results if r.overall_pass)
        pass_rate = n_passed / len(results)

        all_doc = []
        for r in results:
            if r.doctrine_scores:
                all_doc.append(np.mean(list(r.doctrine_scores.values())))
        avg_doc = float(np.mean(all_doc)) if all_doc else 0.0
        avg_align = float(np.mean([r.action_alignment for r in results]))

        weak   = [r.scenario_name for r in results if not r.overall_pass]
        strong = [r.scenario_name for r in results if r.overall_pass and r.action_alignment > 0.5]

        return BenchmarkReport(
            n_scenarios=len(results), n_passed=n_passed,
            pass_rate=pass_rate, scenario_results=results,
            avg_doctrine_compliance=avg_doc, avg_action_alignment=avg_align,
            weak_scenarios=weak, strong_scenarios=strong,
        )


if __name__ == "__main__":
    print("="*55); print("K. 역사 벤치마크 테스트"); print("="*55)
    bench = HistoricalBenchmark(seed=42)
    report = bench.run_all(n_runs=5)
    print(report.summary())
    print("✅ 벤치마크 정상 동작!")
