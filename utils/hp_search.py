"""
hp_search.py
============
CVaR-aware 하이퍼파라미터 격자 탐색 — R15 구현

평균 승률(mean win_rate)과 꼬리 위험(CVaR@10)을 동시에 최적화하는
격자 탐색(Grid Search)을 수행한다.

설계 원칙:
  - CVaR@10 개선이 배포 게이트 조건 (단순 평균 우위만으로 채택 불가)
  - MonteCarloEvaluator를 직접 사용 → Tier A와 동일한 평가 경로 보장
  - ExperimentRegistry에 결과 자동 등록
  - Pareto 프런트(mean_win_rate vs cvar_10) 반환

사용 예시::

    search = CVaRAwareHPSearch(HPSearchConfig(n_mc_runs=50, seeds=[42, 7]))
    pareto = search.run(base_args)
    for candidate in pareto:
        print(candidate)

학술 연구용 합성 데이터
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("falcon")


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class HPSearchConfig:
    """CVaR-aware 하이퍼파라미터 탐색 설정"""

    # 탐색 파라미터 격자
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        "lr":           [1e-4, 3e-4, 1e-3],
        "clip_range":   [0.1, 0.2, 0.3],
        "entropy_coef": [0.0, 0.01, 0.05],
    })

    # 평가 설정
    n_mc_runs: int = 100        # 후보당 MC 평가 횟수 (빠른 탐색은 50, 정밀 탐색은 300+)
    seeds: List[int] = field(default_factory=lambda: [42, 7, 13])
    max_steps: int = 50         # 에피소드당 최대 스텝

    # CVaR 배포 게이트 (모두 만족해야 Pareto 후보 선정)
    min_win_rate: float = 0.45          # 최소 승률 (평균)
    max_cvar10_casualties: float = 40.0 # 최악 10% 구간 평균 사상자 상한

    # 결과 저장
    output_dir: str = "outputs/hp_search"
    save_registry: bool = True           # ExperimentRegistry에 자동 등록


# ──────────────────────────────────────────────
# 단일 후보 평가 결과
# ──────────────────────────────────────────────

@dataclass
class HPCandidate:
    """단일 HP 조합 평가 결과"""
    params: Dict[str, Any]
    mean_win_rate: float
    std_win_rate: float
    mean_cvar10: float          # 최악 10% 평균 사상자 (낮을수록 좋음)
    mean_casualties: float      # 평균 사상자
    strategy_robustness: float  # 강건성 점수 [0, 1]
    n_seeds: int
    passes_gate: bool           # CVaR 게이트 통과 여부

    def __str__(self) -> str:
        gate_tag = "✓" if self.passes_gate else "✗"
        return (
            f"[{gate_tag}] lr={self.params.get('lr', '-'):<8} "
            f"clip={self.params.get('clip_range', '-'):<5} "
            f"ent={self.params.get('entropy_coef', '-'):<6} | "
            f"WR={self.mean_win_rate:.3f}±{self.std_win_rate:.3f} "
            f"CVaR10={self.mean_cvar10:.1f} "
            f"Rob={self.strategy_robustness:.3f}"
        )


# ──────────────────────────────────────────────
# 탐색기
# ──────────────────────────────────────────────

class CVaRAwareHPSearch:
    """
    CVaR@10을 필수 게이트로 사용하는 격자 탐색기.

    각 HP 조합을 여러 seed에서 MC 평가하여 평균 성능과 꼬리 위험을 동시에 측정,
    Pareto 프런트(mean_win_rate vs cvar_10)에서 후보군을 선별한다.
    """

    def __init__(self, config: Optional[HPSearchConfig] = None):
        self.config = config or HPSearchConfig()
        self._results: List[HPCandidate] = []

    # ------------------------------------------------------------------

    def run(self, base_args) -> List[HPCandidate]:
        """
        전체 격자 탐색 실행.

        Parameters
        ----------
        base_args : argparse.Namespace 또는 duck-typed object
            train.py args와 동일한 형태. lr, clip_range, entropy_coef 등
            param_grid 키가 속성으로 존재해야 함.

        Returns
        -------
        list[HPCandidate]
            CVaR 게이트를 통과한 후보 중 Pareto 최적 후보 목록.
            게이트 통과 후보가 없으면 전체 목록을 win_rate 내림차순으로 반환.
        """
        cfg = self.config
        param_keys = list(cfg.param_grid.keys())
        param_values = list(cfg.param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(
            "[HPSearch] 탐색 시작: %d 조합 × %d seed × %d MC runs",
            len(combinations), len(cfg.seeds), cfg.n_mc_runs,
        )
        os.makedirs(cfg.output_dir, exist_ok=True)

        self._results = []

        for combo_idx, combo in enumerate(combinations):
            params = dict(zip(param_keys, combo))
            candidate = self._evaluate_params(params, base_args, combo_idx, len(combinations))
            self._results.append(candidate)
            logger.info("  [%d/%d] %s", combo_idx + 1, len(combinations), candidate)

        # 결과 저장
        self._save_results()

        # Pareto 프런트 계산 및 반환
        pareto = self._compute_pareto_front()
        logger.info("[HPSearch] 완료: 총 %d 조합 → Pareto 후보 %d개", len(self._results), len(pareto))
        return pareto

    # ------------------------------------------------------------------

    def _evaluate_params(
        self,
        params: Dict[str, Any],
        base_args,
        combo_idx: int,
        total: int,
    ) -> HPCandidate:
        """단일 HP 조합을 여러 seed에서 평가"""
        from evaluation.monte_carlo import MonteCarloEvaluator
        from rl_agent.blue_agent import BlueAgent, PPOConfig
        from simulator.lanchester_engine import LanchesterEngine

        cfg = self.config
        win_rates: List[float] = []
        cvar10s: List[float] = []
        casualties: List[float] = []
        robustness_scores: List[float] = []

        for seed in cfg.seeds:
            # HP 적용 — base_args 복사 후 param_grid 값 주입
            args_copy = copy.copy(base_args)
            for k, v in params.items():
                setattr(args_copy, k, v)

            # 에이전트/엔진 생성
            ppo_config = PPOConfig(
                lr=params.get("lr", 3e-4),
                clip_eps=params.get("clip_range", params.get("clip_eps", 0.2)),
                entropy_coef=params.get("entropy_coef", 0.01),
            )
            agent = BlueAgent(config=ppo_config)
            engine = LanchesterEngine(seed=seed)

            mc_eval = MonteCarloEvaluator(
                n_runs=cfg.n_mc_runs,
                seed=seed,
                n_workers=1,
            )
            report = mc_eval.evaluate(
                agent=agent,
                engine=engine,
                max_steps=cfg.max_steps,
                verbose=False,
                show_progress=False,
            )

            win_rates.append(report.blue_win_rate)
            cvar10s.append(report.cvar_10)
            casualties.append(report.avg_blue_casualties)
            robustness_scores.append(report.strategy_robustness)

        mean_wr = float(np.mean(win_rates))
        mean_cvar10 = float(np.mean(cvar10s))

        passes_gate = (
            mean_wr >= cfg.min_win_rate
            and mean_cvar10 <= cfg.max_cvar10_casualties
        )

        return HPCandidate(
            params=params,
            mean_win_rate=mean_wr,
            std_win_rate=float(np.std(win_rates)),
            mean_cvar10=mean_cvar10,
            mean_casualties=float(np.mean(casualties)),
            strategy_robustness=float(np.mean(robustness_scores)),
            n_seeds=len(cfg.seeds),
            passes_gate=passes_gate,
        )

    # ------------------------------------------------------------------

    def _compute_pareto_front(self) -> List[HPCandidate]:
        """
        Pareto 프런트 계산.

        목표: mean_win_rate ↑, mean_cvar10 ↓ (두 목표 동시 최적화)
        CVaR 게이트를 통과한 후보 중에서만 Pareto 최적 집합 선별.
        게이트 통과 후보가 없으면 전체에서 win_rate 내림차순 반환.
        """
        candidates = [c for c in self._results if c.passes_gate]
        if not candidates:
            logger.warning("[HPSearch] CVaR 게이트 통과 후보 없음 — 전체 결과 반환(win_rate 정렬)")
            return sorted(self._results, key=lambda c: c.mean_win_rate, reverse=True)

        pareto: List[HPCandidate] = []
        for cand in candidates:
            dominated = False
            for other in candidates:
                if other is cand:
                    continue
                # other가 cand를 지배: 두 목표 모두에서 같거나 낫고, 하나는 엄격히 우월
                if (other.mean_win_rate >= cand.mean_win_rate
                        and other.mean_cvar10 <= cand.mean_cvar10
                        and (other.mean_win_rate > cand.mean_win_rate
                             or other.mean_cvar10 < cand.mean_cvar10)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(cand)

        # win_rate 내림차순 정렬
        pareto.sort(key=lambda c: c.mean_win_rate, reverse=True)
        return pareto

    # ------------------------------------------------------------------

    def _save_results(self) -> None:
        """탐색 결과를 JSON으로 저장하고 ExperimentRegistry에 등록"""
        cfg = self.config
        results_data = []
        for c in self._results:
            results_data.append({
                "params": c.params,
                "mean_win_rate": c.mean_win_rate,
                "std_win_rate": c.std_win_rate,
                "mean_cvar10": c.mean_cvar10,
                "mean_casualties": c.mean_casualties,
                "strategy_robustness": c.strategy_robustness,
                "n_seeds": c.n_seeds,
                "passes_gate": c.passes_gate,
            })

        out_path = os.path.join(cfg.output_dir, "hp_search_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "param_grid": cfg.param_grid,
                        "n_mc_runs": cfg.n_mc_runs,
                        "seeds": cfg.seeds,
                        "min_win_rate": cfg.min_win_rate,
                        "max_cvar10_casualties": cfg.max_cvar10_casualties,
                    },
                    "results": results_data,
                    "total_combinations": len(results_data),
                    "gate_passed": sum(1 for c in self._results if c.passes_gate),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("[HPSearch] 결과 저장: %s", out_path)

        if cfg.save_registry:
            self._register_to_registry(out_path)

    def _register_to_registry(self, results_path: str) -> None:
        """ExperimentRegistry에 HP 탐색 결과 등록"""
        try:
            from utils.experiment_registry import ExperimentRegistry, RunRecord
            registry = ExperimentRegistry(root_dir=self.config.output_dir)
            best = max(
                (c for c in self._results if c.passes_gate),
                key=lambda c: c.mean_win_rate,
                default=None,
            ) or max(self._results, key=lambda c: c.mean_win_rate)

            record = RunRecord(
                run_id=f"hp_search_{os.path.basename(self.config.output_dir)}",
                phase="hp_search",
                algorithm="ppo_grid",
                seed=self.config.seeds[0] if self.config.seeds else 0,
                episodes=self.config.n_mc_runs * len(self.config.seeds),
                win_rate=best.mean_win_rate,
                checkpoint_path=results_path,
                config={"param_grid": self.config.param_grid},
            )
            registry.add(record)
            logger.info("[HPSearch] ExperimentRegistry 등록 완료")
        except Exception as exc:
            logger.warning("[HPSearch] Registry 등록 실패 (무시): %s", exc)

    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """탐색 결과 요약 테이블 문자열 반환"""
        if not self._results:
            return "(결과 없음)"
        lines = ["HP 탐색 결과 (상위 10개, win_rate 내림차순):", ""]
        sorted_results = sorted(self._results, key=lambda c: c.mean_win_rate, reverse=True)
        for c in sorted_results[:10]:
            lines.append(f"  {c}")
        return "\n".join(lines)
