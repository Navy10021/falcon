"""
monte_carlo.py
==============
Monte Carlo 강건성 평가 (5,000+ 시나리오)
- 다양한 초기 조건에서의 전략 성능 분포 평가
- 신뢰구간 계산
- 취약점 분석
- ProcessPoolExecutor 기반 병렬화 (n_workers > 1)
"""

from __future__ import annotations
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


def _build_mc_action_pairs(kg, blue_action: int) -> Optional[list]:
    """
    N-1: 에이전트 행동 → 교전 쌍 변환 (Monte Carlo 평가용).

    BlueActionSpace 인덱스에 따라 공격 대상 우선순위(약한/강한/균형)를 결정한다.
    train.py._build_blue_action_pairs()와 동일한 로직 — 평가-학습 환경 일관성 확보.
    """
    from ontology.combat_schema import ForceAlignment, UnitStatus
    from rl_agent.blue_agent import BlueActionSpace

    # BlueActionSpace 상수 — import 기반 동기화 (하드코딩 제거)
    ADVANCE = BlueActionSpace.ADVANCE
    FLANK = BlueActionSpace.FLANK
    REINFORCE = BlueActionSpace.REINFORCE
    SUPPORT = BlueActionSpace.SUPPORT
    WITHDRAW = BlueActionSpace.WITHDRAW

    blue_units = [u for u in kg.units.values()
                  if u.alignment == ForceAlignment.BLUE
                  and u.status != UnitStatus.DESTROYED
                  and u.headcount > 0]
    red_units  = [u for u in kg.units.values()
                  if u.alignment == ForceAlignment.RED
                  and u.status != UnitStatus.DESTROYED
                  and u.headcount > 0]
    if not blue_units or not red_units:
        return None

    def pick_targets(units, mode):
        if mode == "weakest":
            return sorted(units, key=lambda u: u.headcount)
        if mode == "strongest":
            return sorted(units, key=lambda u: u.combat_power, reverse=True)
        return sorted(units, key=lambda u: u.headcount, reverse=True)

    mode_map = {
        ADVANCE: "weakest", FLANK: "weakest",
        REINFORCE: "strongest", SUPPORT: "strongest",
    }
    mode = mode_map.get(int(blue_action), "balanced")
    targets = pick_targets(red_units, mode)

    attackers = blue_units
    if int(blue_action) == WITHDRAW:
        attackers = blue_units[:max(1, len(blue_units) // 2)]

    return [(att.unit_id, targets[i % len(targets)].unit_id)
            for i, att in enumerate(attackers)]


def _mc_run_worker(args: Dict) -> "MCResult":
    """
    단일 Monte Carlo 런 실행 — ProcessPoolExecutor 워커용 최상위 함수.

    Parameters
    ----------
    args : dict
        run_id, base_seed, max_steps, fog_level_name, checkpoint_path
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from ontology.combat_schema import ScenarioFactory, ForceAlignment
    from simulator.lanchester_engine import LanchesterEngine
    from rl_agent.blue_agent import BlueAgent, build_state_vector

    run_id: int = args["run_id"]
    base_seed: int = args["base_seed"]
    max_steps: int = args["max_steps"]
    fog_level_name: Optional[str] = args.get("fog_level_name")
    checkpoint_path: Optional[str] = args.get("checkpoint_path")

    rng = np.random.RandomState(base_seed + run_id)
    n_blue = int(rng.randint(5, 13))
    n_red  = int(rng.randint(4, 10))

    engine = LanchesterEngine(seed=base_seed + run_id)
    agent  = BlueAgent()
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)

    fog_filter = None
    if fog_level_name:
        from simulator.fog_of_war import FogOfWarFilter, FogLevel
        fog_level = getattr(FogLevel, fog_level_name.upper(), None)
        if fog_level is not None:
            fog_filter = FogOfWarFilter(fog_level, seed=base_seed + run_id)

    kg = ScenarioFactory.create_standard_scenario(
        n_blue=n_blue, n_red=n_red, seed=base_seed + run_id
    )
    initial_blue_hc = sum(u.headcount for u in kg.units.values()
                          if u.alignment == ForceAlignment.BLUE)

    blue_casualties = 0
    red_casualties  = 0
    winner = "draw"
    n_steps = 0

    for _ in range(max_steps):
        if fog_filter:
            obs_kg, uncertainty = fog_filter.observe(kg, ForceAlignment.BLUE)
            state = build_state_vector(obs_kg, uncertainty_map=uncertainty)
        else:
            state = build_state_vector(kg)

        # N-1: select_action 결과를 action_pairs로 변환해 run_step에 반영
        action, _, _ = agent.select_action(state, deterministic=True)
        action_pairs = _build_mc_action_pairs(kg, action)

        step = engine.run_step(kg, action_pairs=action_pairs)
        blue_casualties += step.blue_total_casualties
        red_casualties  += step.red_total_casualties
        n_steps += 1

        if step.mission_status != "ongoing":
            winner = step.mission_status
            break

    final_blue_hc = sum(u.headcount for u in kg.units.values()
                        if u.alignment == ForceAlignment.BLUE)
    force_reduction = (initial_blue_hc - final_blue_hc) / max(initial_blue_hc, 1)

    return MCResult(
        run_id=run_id,
        winner=winner,
        blue_casualties=blue_casualties,
        red_casualties=red_casualties,
        n_steps=n_steps,
        force_reduction_ratio=force_reduction,
        initial_conditions={"n_blue": n_blue, "n_red": n_red},
    )


@dataclass
class MCResult:
    """Monte Carlo 단일 실행 결과"""
    run_id: int
    winner: str
    blue_casualties: int
    red_casualties: int
    n_steps: int
    force_reduction_ratio: float
    initial_conditions: Dict


@dataclass
class MCEvaluationReport:
    """Monte Carlo 평가 보고서"""
    n_runs: int
    blue_win_rate: float
    blue_win_rate_ci: Tuple[float, float]   # 95% CI
    avg_blue_casualties: float
    avg_blue_casualties_ci: Tuple[float, float]
    avg_force_reduction: float
    avg_steps: float
    worst_case_casualties: int
    best_case_casualties: int
    strategy_robustness: float             # [0, 1] 높을수록 강건
    failure_scenarios: List[Dict]          # 실패 시나리오 분석
    vulnerability_analysis: Dict[str, float]
    # CVaR 극단 시나리오 강건성 (장기: CVaR 기반 위험 지표)
    cvar_05: float = 0.0    # 최악 5% 시나리오의 평균 사상자 (Conditional Value at Risk)
    cvar_10: float = 0.0    # 최악 10% 시나리오의 평균 사상자
    cvar_robustness: float = 0.0  # 1 - cvar_10 / max_casualties (0→1, 높을수록 강건)
    metadata: Dict[str, object] = field(default_factory=dict)


class MonteCarloEvaluator:
    """
    Monte Carlo 강건성 평가기
    다양한 초기 조건 + 확률적 전투 결과로 전략 성능 분포 계산
    """

    def __init__(self, n_runs: int = 5000, seed: int = 0,
                 n_workers: int = 1, checkpoint_path: Optional[str] = None):
        self.n_runs = n_runs
        self.base_seed = seed
        self.n_workers = n_workers
        self.checkpoint_path = checkpoint_path

    def evaluate(
        self,
        agent=None,     # BlueAgent (순차 모드에서만 사용)
        engine=None,    # LanchesterEngine (순차 모드에서만 사용)
        fog_filter=None,
        fog_level_name: Optional[str] = None,
        scenario_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        show_progress: bool = True,
        max_steps: int = 50,
    ) -> MCEvaluationReport:
        """
        Monte Carlo 평가 실행.

        n_workers > 1 이면 ProcessPoolExecutor로 병렬 실행.
        병렬 모드에서는 각 워커가 내부에서 engine/agent를 재생성하므로
        agent / engine 인자는 checkpoint_path 로 대체된다.

        fog_level_name : fog_filter 가 None 일 때 워커에서 fog 생성용.
            FogLevel enum 멤버 이름 (예: "MODERATE").
        """
        # fog_level 이름 추출 (fog_filter 우선)
        _fog_name: Optional[str] = fog_level_name
        if fog_filter is not None and _fog_name is None:
            # fog_filter.fog_level 가 FogLevel enum 이라고 가정
            try:
                _fog_name = fog_filter.fog_level.name
            except AttributeError:
                _fog_name = None

        if self.n_workers > 1:
            return self._evaluate_parallel(
                max_steps=max_steps,
                fog_level_name=_fog_name,
                verbose=verbose,
                show_progress=show_progress,
            )
        else:
            return self._evaluate_sequential(
                agent=agent,
                engine=engine,
                fog_filter=fog_filter,
                max_steps=max_steps,
                verbose=verbose,
                show_progress=show_progress,
            )

    # ------------------------------------------------------------------
    def _evaluate_sequential(
        self, agent, engine, fog_filter, max_steps, verbose, show_progress
    ) -> MCEvaluationReport:
        from ontology.combat_schema import ScenarioFactory, ForceAlignment
        from rl_agent.blue_agent import build_state_vector

        results: List[MCResult] = []

        iterator = range(self.n_runs)
        if verbose and show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo 평가")

        for run_id in iterator:
            rng = np.random.RandomState(self.base_seed + run_id)
            n_blue = int(rng.randint(5, 13))
            n_red  = int(rng.randint(4, 10))

            kg = ScenarioFactory.create_standard_scenario(
                n_blue=n_blue, n_red=n_red, seed=self.base_seed + run_id
            )
            initial_blue_hc = sum(u.headcount for u in kg.units.values()
                                  if u.alignment == ForceAlignment.BLUE)

            blue_casualties = 0
            red_casualties = 0
            winner = "draw"
            n_steps = 0

            for _ in range(max_steps):
                if fog_filter:
                    obs_kg, uncertainty = fog_filter.observe(kg, ForceAlignment.BLUE)
                    state = build_state_vector(obs_kg, uncertainty_map=uncertainty)
                else:
                    state = build_state_vector(kg)

                # N-1: select_action 결과를 action_pairs로 변환해 run_step에 반영
                action, _, _ = agent.select_action(state, deterministic=True)
                action_pairs = _build_mc_action_pairs(kg, action)

                step = engine.run_step(kg, action_pairs=action_pairs)
                blue_casualties += step.blue_total_casualties
                red_casualties += step.red_total_casualties
                n_steps += 1

                if step.mission_status != "ongoing":
                    winner = step.mission_status
                    break

            final_blue_hc = sum(u.headcount for u in kg.units.values()
                                if u.alignment == ForceAlignment.BLUE)
            force_reduction = (initial_blue_hc - final_blue_hc) / max(initial_blue_hc, 1)

            results.append(MCResult(
                run_id=run_id,
                winner=winner,
                blue_casualties=blue_casualties,
                red_casualties=red_casualties,
                n_steps=n_steps,
                force_reduction_ratio=force_reduction,
                initial_conditions={"n_blue": n_blue, "n_red": n_red},
            ))

        return self._compute_report(results)

    # ------------------------------------------------------------------
    def _evaluate_parallel(
        self,
        max_steps: int,
        fog_level_name: Optional[str],
        verbose: bool,
        show_progress: bool,
    ) -> MCEvaluationReport:
        """ProcessPoolExecutor 기반 병렬 Monte Carlo 평가."""
        worker_args = [
            {
                "run_id":          i,
                "base_seed":       self.base_seed,
                "max_steps":       max_steps,
                "fog_level_name":  fog_level_name,
                "checkpoint_path": self.checkpoint_path,
            }
            for i in range(self.n_runs)
        ]

        results: List[MCResult] = []
        pbar = tqdm(total=self.n_runs, desc="Monte Carlo 병렬 평가") \
               if (verbose and show_progress) else None

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_mc_run_worker, a): a for a in worker_args}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    # 개별 런 실패 시 skip (전체 중단 방지)
                    if verbose:
                        print(f"  [MC worker error] {exc}")
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        results.sort(key=lambda r: r.run_id)
        return self._compute_report(results)

    @staticmethod
    def _compute_cvar(values: np.ndarray, alpha: float) -> float:
        """
        CVaR (Conditional Value at Risk) 계산.
        alpha 분위수를 초과하는 상위 (1-alpha)% 값들의 평균.
        여기서는 사상자 기준 최악 alpha% 시나리오 평균을 반환한다.
        예) alpha=0.05 → 최악 5% 시나리오의 평균 사상자.
        """
        if len(values) == 0:
            return 0.0
        threshold = np.quantile(values, 1.0 - alpha)
        tail = values[values >= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(values.max())

    def _compute_report(self, results: List[MCResult]) -> MCEvaluationReport:
        """통계 보고서 계산"""
        n = len(results)
        winners = [r.winner for r in results]
        blue_wins = sum(1 for w in winners if w == "blue_win")
        blue_win_rate = blue_wins / n

        casualties = np.array([r.blue_casualties for r in results])
        force_reductions = np.array([r.force_reduction_ratio for r in results])
        steps = np.array([r.n_steps for r in results])

        # 95% 신뢰구간 (Bootstrap)
        def bootstrap_ci(data, stat_fn=np.mean, n_boot=1000, ci=0.95):
            rng = np.random.RandomState(42)
            boot_stats = [stat_fn(rng.choice(data, len(data))) for _ in range(n_boot)]
            alpha = (1 - ci) / 2
            return (float(np.percentile(boot_stats, alpha * 100)),
                    float(np.percentile(boot_stats, (1 - alpha) * 100)))

        win_ci = bootstrap_ci([1 if r.winner == "blue_win" else 0 for r in results], n_boot=500)
        cas_ci = bootstrap_ci(casualties, n_boot=500)

        # 실패 시나리오 분석 (Blue 패배)
        failures = [r for r in results if r.winner == "red_win"]
        failure_scenarios = []
        for f in failures[:10]:  # 상위 10개
            failure_scenarios.append({
                "run_id": f.run_id,
                "initial_conditions": f.initial_conditions,
                "blue_casualties": f.blue_casualties,
                "red_casualties": f.red_casualties,
                "exchange_ratio": f.blue_casualties / max(f.red_casualties, 1),
                "n_steps": f.n_steps
            })

        # 취약점 분석
        vulnerability = self._analyze_vulnerability(results)

        # 전략 강건성 = 승률 × (1 - 사상자 변동성)
        win_array = np.array([1 if r.winner == "blue_win" else 0 for r in results])
        robustness = float(blue_win_rate * (1 - np.std(win_array)))

        # CVaR 극단 시나리오 강건성 지표
        cvar_05 = self._compute_cvar(casualties, alpha=0.05)
        cvar_10 = self._compute_cvar(casualties, alpha=0.10)
        max_cas  = float(casualties.max()) if len(casualties) > 0 else 1.0
        cvar_robustness = float(1.0 - cvar_10 / max(max_cas, 1.0))

        return MCEvaluationReport(
            n_runs=n,
            blue_win_rate=blue_win_rate,
            blue_win_rate_ci=win_ci,
            avg_blue_casualties=float(casualties.mean()),
            avg_blue_casualties_ci=cas_ci,
            avg_force_reduction=float(force_reductions.mean()),
            avg_steps=float(steps.mean()),
            worst_case_casualties=int(casualties.max()),
            best_case_casualties=int(casualties.min()),
            strategy_robustness=robustness,
            failure_scenarios=failure_scenarios,
            vulnerability_analysis=vulnerability,
            cvar_05=cvar_05,
            cvar_10=cvar_10,
            cvar_robustness=cvar_robustness,
            metadata={
                "base_seed": self.base_seed,
                "n_workers": self.n_workers,
                "checkpoint_path": self.checkpoint_path,
            },
        )

    def _analyze_vulnerability(self, results: List[MCResult]) -> Dict[str, float]:
        """초기 조건별 취약점 분석"""
        # n_blue vs n_red 비율별 승률
        ratio_wins: Dict[str, List[int]] = {}
        for r in results:
            n_b = r.initial_conditions["n_blue"]
            n_r = r.initial_conditions["n_red"]
            ratio = round(n_b / max(n_r, 1), 1)
            key = f"ratio_{ratio}"
            if key not in ratio_wins:
                ratio_wins[key] = []
            ratio_wins[key].append(1 if r.winner == "blue_win" else 0)

        return {k: float(np.mean(v)) for k, v in ratio_wins.items()
                if len(v) >= 10}

    @staticmethod
    def report_to_dict(report: MCEvaluationReport) -> Dict[str, object]:
        """고정 JSON schema(dict)로 평가 보고서를 직렬화한다."""
        return {
            "schema_version": "falcon-mc-report-v1",
            "metadata": report.metadata,
            "metrics": {
                "n_runs": report.n_runs,
                "blue_win_rate": report.blue_win_rate,
                "blue_win_rate_ci": list(report.blue_win_rate_ci),
                "avg_blue_casualties": report.avg_blue_casualties,
                "avg_blue_casualties_ci": list(report.avg_blue_casualties_ci),
                "avg_force_reduction": report.avg_force_reduction,
                "avg_steps": report.avg_steps,
                "worst_case_casualties": report.worst_case_casualties,
                "best_case_casualties": report.best_case_casualties,
                "strategy_robustness": report.strategy_robustness,
                "cvar_05": report.cvar_05,
                "cvar_10": report.cvar_10,
                "cvar_robustness": report.cvar_robustness,
            },
            "vulnerability_analysis": report.vulnerability_analysis,
            "failure_scenarios": report.failure_scenarios,
        }

    def save_report_json(self, report: MCEvaluationReport, output_path: str) -> str:
        """평가 보고서를 고정 schema JSON 파일로 저장한다."""
        payload = self.report_to_dict(report)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return output_path

    def print_report(self, report: MCEvaluationReport):
        """보고서 출력"""
        print("\n" + "═" * 60)
        print("📊 Monte Carlo 강건성 평가 보고서")
        print("═" * 60)
        print(f"총 실행 횟수: {report.n_runs:,}회")
        print(f"\n🏆 성능 지표:")
        print(f"  Blue 승률:    {report.blue_win_rate:.1%} "
              f"(95% CI: {report.blue_win_rate_ci[0]:.1%} ~ {report.blue_win_rate_ci[1]:.1%})")
        print(f"  평균 사상자:  {report.avg_blue_casualties:.1f}명 "
              f"(95% CI: {report.avg_blue_casualties_ci[0]:.1f} ~ {report.avg_blue_casualties_ci[1]:.1f})")
        print(f"  병력 절감율:  {report.avg_force_reduction:.1%}")
        print(f"  평균 소요시간: {report.avg_steps:.1f} 스텝")
        print(f"  전략 강건성:  {report.strategy_robustness:.4f}")
        print(f"\n⚠️ 극단값 및 CVaR 위험 지표:")
        print(f"  최악 사상자:  {report.worst_case_casualties}명")
        print(f"  최선 사상자:  {report.best_case_casualties}명")
        print(f"  CVaR (최악 5%): {report.cvar_05:.1f}명 평균 사상자")
        print(f"  CVaR (최악 10%): {report.cvar_10:.1f}명 평균 사상자")
        print(f"  CVaR 강건성: {report.cvar_robustness:.4f}  (1=완전강건, 0=극단적 취약)")
        print(f"\n🔍 취약점 분석 (병력 비율별 승률):")
        for k, v in sorted(report.vulnerability_analysis.items()):
            ratio_val = k.replace("ratio_", "")
            print(f"  Blue/Red={ratio_val}: {v:.1%}")
        print("═" * 60)


if __name__ == "__main__":
    print("=== Monte Carlo Evaluator Test (단축 실행) ===")
    from ontology.combat_schema import ScenarioFactory
    from simulator.lanchester_engine import LanchesterEngine
    from rl_agent.blue_agent import BlueAgent

    agent = BlueAgent()
    engine = LanchesterEngine(seed=42)
    evaluator = MonteCarloEvaluator(n_runs=50, seed=42)  # 테스트용 단축

    report = evaluator.evaluate(agent, engine, verbose=True)
    evaluator.print_report(report)
    print("✅ Monte Carlo 평가 테스트 완료!")
