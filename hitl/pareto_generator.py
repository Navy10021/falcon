"""
pareto_generator.py
===================
Phase 3: Pareto 최적 전략 후보 생성기
- 다중 목적 최적화 (병력 최소화 vs 승률 vs 사상자 최소화)
- 지휘관 제약 조건 반영
- 3~5개 Pareto 최적 후보 생성
"""

from __future__ import annotations
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum


class StrategyType(Enum):
    MIN_FORCE = "minimum_force"
    BALANCED = "balanced"
    MAX_WIN_RATE = "maximum_win_rate"
    MIN_CASUALTY = "minimum_casualty"
    MIN_TIME = "minimum_time"


@dataclass
class CommanderConstraints:
    """지휘관 제약 조건"""
    # Hard Constraints (절대 조건)
    max_force_size: Optional[int] = None       # 최대 투입 병력
    min_win_probability: float = 0.5           # 최소 승률 요구사항
    max_casualties: Optional[int] = None       # 최대 허용 사상자
    max_time_steps: int = 50                   # 최대 허용 시간
    forbidden_zones: List[str] = field(default_factory=list)  # 진입 금지 구역

    # Soft Constraints (선호 조건)
    prefer_flanking: bool = False              # 측면 공격 선호
    prefer_artillery: bool = False             # 포병 지원 선호
    avoid_urban: bool = False                  # 도심 전투 회피
    min_morale_threshold: float = 0.3         # 최소 사기 임계치

    # Ethical Constraints
    roe_compliance: bool = True               # 교전 규칙 준수
    minimize_collateral: bool = True          # 부수 피해 최소화


@dataclass
class StrategyOption:
    """Pareto 전략 후보"""
    option_id: str
    strategy_type: StrategyType
    label: str                         # 예: "최소 병력", "균형", "최고 승률"
    force_size: int                    # 투입 병력 수
    win_probability: float             # 예상 승률
    expected_casualties: int           # 예상 사상자
    expected_time_steps: int           # 예상 소요 시간
    risk_level: str                    # "low", "medium", "high"
    action_sequence: List[int]         # 권고 행동 시퀀스
    trade_offs: str                    # 트레이드오프 설명
    constraint_violations: List[str] = field(default_factory=list)
    uncertainty_score: float = 0.0    # GNN 불확실성 점수


class ParetoStrategyGenerator:
    """
    Pareto 최적 전략 생성기
    
    다중 목적함수:
    1. 병력 규모 최소화 (↓)
    2. 승률 최대화 (↑)
    3. 아군 사상자 최소화 (↓)
    4. 임무 시간 최소화 (↓)
    """

    def __init__(self, n_candidates: int = 5, mc_eval_runs: int = 100):
        self.n_candidates = n_candidates
        self.mc_eval_runs = mc_eval_runs
        self._last_generated: List[StrategyOption] = []
        self.last_generation_all_violated: bool = False
        self.last_generation_note: str = ""

    def generate(
        self,
        kg,
        constraints: CommanderConstraints,
        gnn_output: Optional[Dict] = None,
        blue_agent=None
    ) -> List[StrategyOption]:
        """
        Pareto 최적 전략 후보 생성
        """
        # 전장 현황 분석
        from ontology.combat_schema import ForceAlignment, UnitStatus
        blue_units = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED]
        red_units  = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED]
        available_force   = sum(u.headcount     for u in blue_units)
        blue_combat_power = sum(u.combat_power  for u in blue_units)
        red_combat_power  = sum(u.combat_power  for u in red_units)

        uncertainty = 0.0
        if gnn_output:
            uncertainty = float(gnn_output.get("total_uncertainty", 0.0))

        # Monte Carlo 평가로 각 전략 시뮬레이션
        candidates = self._generate_candidate_strategies(
            available_force, blue_combat_power, red_combat_power, uncertainty, constraints
        )

        # Pareto 필터링 (제약 조건 위반 후보 제거)
        valid_candidates = [c for c in candidates
                           if not self._violates_hard_constraints(c, constraints)]

        self.last_generation_all_violated = False
        self.last_generation_note = ""
        if not valid_candidates:
            self.last_generation_all_violated = True
            self.last_generation_note = "모든 후보가 hard constraint를 위반하여 최소 위반 후보를 반환합니다."
            valid_candidates = self._select_min_violation_candidates(candidates)

        # P1-1: top-k 후보에 경량 MC-lite 보정 적용
        self._apply_mc_lite_calibration(valid_candidates, kg)

        # Pareto 프론트 계산
        pareto_front = self._compute_pareto_front(valid_candidates)

        self._last_generated = pareto_front[:self.n_candidates]
        return self._last_generated

    @staticmethod
    def _dynamic_win_base(blue_cp: float, red_cp: float, win_delta: float = 0.0) -> float:
        """
        전투력 비율 기반 동적 win_base 계산.

        cp_ratio = blue_cp / red_cp 에 대해 logistic 함수로 매핑.
        k=2.5 로 설정 시: ratio=1.0 → 0.50, 1.5 → 0.73, 0.7 → 0.32

        Parameters
        ----------
        blue_cp, red_cp : float
            아군/적군 전투력 합계.
        win_delta : float
            전략별 상대적 승률 조정값 (-1..+1).

        Returns
        -------
        float
            [0.10, 0.97] 로 클리핑된 예상 승률.
        """
        cp_ratio = blue_cp / max(red_cp, 1e-6)
        k = 2.5
        base = 1.0 / (1.0 + math.exp(-k * (cp_ratio - 1.0)))
        return float(np.clip(base + win_delta, 0.10, 0.97))

    def _generate_candidate_strategies(
        self, available_force: int, blue_cp: float, red_cp: float,
        uncertainty: float, constraints: CommanderConstraints
    ) -> List[StrategyOption]:
        """기본 전략 후보 생성 (win_base를 전투력 비율로 동적 계산)"""
        candidates = []
        base_force = available_force

        # win_delta: 전략별 상대적 승률 조정값 (전투력 비율 기준 baseline에서의 편차)
        strategy_configs = [
            {
                "type": StrategyType.MIN_FORCE,
                "label": "최소 병력",
                "force_ratio": 0.65,
                "win_delta": -0.08,   # 병력 절감으로 승률 소폭 하락
                "cas_ratio": 0.12,
                "time_ratio": 1.2,
                "risk": "high",
                "trade_off": "병력 절감 최우선, 승률 다소 감수"
            },
            {
                "type": StrategyType.BALANCED,
                "label": "균형",
                "force_ratio": 0.82,
                "win_delta":  0.00,   # 전투력 비율 그대로 반영
                "cas_ratio": 0.08,
                "time_ratio": 1.0,
                "risk": "medium",
                "trade_off": "병력-성과 균형 최적점"
            },
            {
                "type": StrategyType.MAX_WIN_RATE,
                "label": "최고 승률",
                "force_ratio": 0.95,
                "win_delta": +0.07,   # 전병력 투입으로 승률 상승
                "cas_ratio": 0.06,
                "time_ratio": 0.9,
                "risk": "low",
                "trade_off": "승률 최우선, 병력 여유 사용"
            },
            {
                "type": StrategyType.MIN_CASUALTY,
                "label": "최저 사상자",
                "force_ratio": 0.88,
                "win_delta": +0.02,   # 포병/간접화력 → 사상자 절감, 승률 소폭 상승
                "cas_ratio": 0.04,
                "time_ratio": 1.1,
                "risk": "low",
                "trade_off": "인명 손실 최소화 우선"
            },
            {
                "type": StrategyType.MIN_TIME,
                "label": "신속 완료",
                "force_ratio": 0.90,
                "win_delta": -0.03,   # 속전속결로 위험 증가, 승률 소폭 하락
                "cas_ratio": 0.10,
                "time_ratio": 0.7,
                "risk": "medium",
                "trade_off": "임무 조기 완료, 사상자 다소 감수"
            },
        ]

        for i, cfg in enumerate(strategy_configs):
            # 불확실성에 따른 조정
            uncertainty_adj = 1.0 + 0.2 * uncertainty
            force_ratio = min(cfg["force_ratio"] * uncertainty_adj, 1.0)
            # win_base: 전투력 비율 기반 동적 계산 + 불확실성 패널티
            win_base = self._dynamic_win_base(
                blue_cp * cfg["force_ratio"],   # 실제 투입 전투력 근사
                red_cp,
                win_delta=cfg["win_delta"],
            )
            win_prob = max(win_base - 0.05 * uncertainty, 0.10)
            cas_ratio = cfg["cas_ratio"] * uncertainty_adj

            force_size = int(base_force * force_ratio)
            casualties = int(force_size * cas_ratio)
            time_steps = int(30 * cfg["time_ratio"])

            # 지형/제약 선호도 반영
            if constraints.prefer_artillery:
                casualties = int(casualties * 0.85)  # 포병 지원 시 사상자 감소
            if constraints.avoid_urban:
                time_steps = int(time_steps * 1.1)   # 우회 시 시간 증가

            option = StrategyOption(
                option_id=f"option_{i+1}",
                strategy_type=cfg["type"],
                label=cfg["label"],
                force_size=force_size,
                win_probability=win_prob,
                expected_casualties=casualties,
                expected_time_steps=time_steps,
                risk_level=cfg["risk"],
                action_sequence=self._generate_action_sequence(cfg["type"]),
                trade_offs=cfg["trade_off"],
                uncertainty_score=uncertainty
            )
            candidates.append(option)

        return candidates

    def _generate_action_sequence(self, strategy_type: StrategyType) -> List[int]:
        """전략 유형별 권고 행동 시퀀스"""
        # 0=Reinforce,1=Advance,2=Withdraw,3=Reallocate,4=Support,5=Flank
        sequences = {
            StrategyType.MIN_FORCE:    [5, 2, 3, 0],
            StrategyType.BALANCED:     [1, 0, 4, 1],
            StrategyType.MAX_WIN_RATE: [1, 4, 1, 1],
            StrategyType.MIN_CASUALTY: [2, 4, 5, 0],
            StrategyType.MIN_TIME:     [1, 5, 1, 0],
        }
        return sequences.get(strategy_type, [0, 1, 2, 3])

    def _violates_hard_constraints(
        self, option: StrategyOption, constraints: CommanderConstraints
    ) -> bool:
        """Hard Constraint 위반 여부 확인"""
        violations = []

        if constraints.max_force_size and option.force_size > constraints.max_force_size:
            violations.append(f"병력 초과: {option.force_size} > {constraints.max_force_size}")

        if option.win_probability < constraints.min_win_probability:
            violations.append(f"승률 미달: {option.win_probability:.1%} < {constraints.min_win_probability:.1%}")

        if constraints.max_casualties and option.expected_casualties > constraints.max_casualties:
            violations.append(f"사상자 초과: {option.expected_casualties} > {constraints.max_casualties}")

        if option.expected_time_steps > constraints.max_time_steps:
            violations.append(f"시간 초과: {option.expected_time_steps} > {constraints.max_time_steps}")

        option.constraint_violations = violations
        return len(violations) > 0

    def _compute_pareto_front(self, candidates: List[StrategyOption]) -> List[StrategyOption]:
        """
        Pareto 프론트 계산
        목적: 병력↓, 승률↑, 사상자↓, 시간↓
        """
        if len(candidates) <= 1:
            return candidates

        # 정규화
        forces = np.array([c.force_size for c in candidates], dtype=float)
        wins   = np.array([c.win_probability for c in candidates])
        cas    = np.array([c.expected_casualties for c in candidates], dtype=float)
        times  = np.array([c.expected_time_steps for c in candidates], dtype=float)

        def norm(arr): return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        # 최소화 방향 통일 (승률은 1-win으로 변환)
        obj = np.stack([
            norm(forces),
            1 - norm(wins),
            norm(cas),
            norm(times)
        ], axis=1)  # [N, 4]

        # Pareto 지배 관계 계산
        n = len(candidates)
        dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                    dominated[i] = True
                    break

        pareto = [c for i, c in enumerate(candidates) if not dominated[i]]

        # Pareto 프론트 정렬 (균형 점수 기준)
        def balance_score(c):
            return (c.win_probability * 0.4
                    - c.force_size / 1000 * 0.3
                    - c.expected_casualties / 100 * 0.3)

        pareto.sort(key=balance_score, reverse=True)
        return pareto


    def _select_min_violation_candidates(self, candidates: List[StrategyOption]) -> List[StrategyOption]:
        """모든 후보가 위반인 경우 최소 위반(개수/심각도) 후보 우선 선택"""
        def violation_penalty(option: StrategyOption) -> float:
            penalty = float(len(option.constraint_violations))
            for v in option.constraint_violations:
                if "병력 초과" in v:
                    penalty += 1.5
                elif "승률 미달" in v:
                    penalty += 1.5
                elif "사상자 초과" in v:
                    penalty += 1.0
                elif "시간 초과" in v:
                    penalty += 0.8
                else:
                    penalty += 0.5
            return penalty

        ranked = sorted(candidates, key=violation_penalty)
        return ranked[:max(1, self.n_candidates)]


    def _apply_mc_lite_calibration(self, candidates: List[StrategyOption], kg) -> None:
        """상위 후보 일부에 대해 Monte Carlo-lite 보정치를 반영한다."""
        if not candidates or self.mc_eval_runs <= 0:
            return

        ranked = sorted(candidates, key=lambda c: c.win_probability, reverse=True)
        top_k = max(1, min(3, len(ranked)))
        for opt in ranked[:top_k]:
            self._calibrate_candidate_with_mc(kg, opt)

    def _calibrate_candidate_with_mc(self, kg, option: StrategyOption) -> None:
        """단일 후보에 대해 경량 N-run rollout 기반으로 핵심 지표를 보정한다."""
        from ontology.combat_schema import ForceAlignment
        from simulator.lanchester_engine import LanchesterEngine

        wins, casualties, steps = [], [], []
        n_runs = max(1, self.mc_eval_runs)

        for sim_i in range(n_runs):
            sim_kg = copy.deepcopy(kg)
            engine = LanchesterEngine(seed=sim_i)

            # 옵션 병력 규모 반영
            blue_units = [u for u in sim_kg.units.values() if u.alignment == ForceAlignment.BLUE]
            total_blue = sum(u.headcount for u in blue_units)
            if total_blue > 0:
                scale = min(option.force_size / total_blue, 1.0)
                for u in blue_units:
                    u.headcount = max(5, int(u.headcount * scale))

            mission_status = "ongoing"
            total_blue_cas = 0
            for step_t in range(max(10, option.expected_time_steps)):
                res = engine.run_step(sim_kg)
                total_blue_cas += res.blue_total_casualties
                mission_status = res.mission_status
                if mission_status != "ongoing":
                    steps.append(step_t + 1)
                    break
            else:
                steps.append(max(10, option.expected_time_steps))

            wins.append(1 if mission_status == "blue_win" else 0)
            casualties.append(total_blue_cas)

        option.win_probability = float(np.mean(wins))
        option.expected_casualties = int(np.mean(casualties))
        option.expected_time_steps = int(np.mean(steps))

    def display_options(self, options: List[StrategyOption]) -> str:
        """전략 후보 출력 텍스트 생성"""
        lines = [
            "═" * 70,
            "🎯 Pareto 최적 전략 후보안",
            "═" * 70,
        ]
        if self.last_generation_all_violated:
            lines.append(f"⚠️ {self.last_generation_note}")
        for opt in options:
            lines.append(f"\n📌 [{opt.option_id.upper()}] {opt.label}")
            lines.append(f"   투입 병력:     {opt.force_size}명")
            lines.append(f"   예상 승률:     {opt.win_probability:.1%}")
            lines.append(f"   예상 사상자:   {opt.expected_casualties}명")
            lines.append(f"   예상 소요시간: {opt.expected_time_steps} 스텝")
            lines.append(f"   위험 수준:     {opt.risk_level.upper()}")
            lines.append(f"   특성:          {opt.trade_offs}")
            if opt.constraint_violations:
                lines.append(f"   ⚠️ 제약 위반:  {', '.join(opt.constraint_violations)}")

        lines.append("\n" + "═" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    print("=== Pareto Strategy Generator Test ===")

    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    constraints = CommanderConstraints(
        max_force_size=700,
        min_win_probability=0.65,
        max_casualties=80,
        prefer_artillery=True
    )
    generator = ParetoStrategyGenerator()
    options = generator.generate(kg, constraints)
    print(generator.display_options(options))
    print(f"✅ {len(options)}개 Pareto 전략 후보 생성 완료")
