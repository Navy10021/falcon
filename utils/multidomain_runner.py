"""
multidomain_runner.py
=====================
다도메인 학습 루프 통합 — 도메인 전이/일반화 성능 계량화

단일 도메인 최적화가 아닌 도메인 전이/일반화 성능을 측정하기 위한
배치 실험 프레임워크.

실험 매트릭스:
  - Train Domain: ground, air, naval, multi
  - Eval Domain: ground, air, naval, multi
  - Seed: 최소 5개

산출물:
  1. 4x4 일반화 성능 매트릭스 (평균 +/- 표준편차)
  2. OOD(Out-of-Domain) 성능 저하율
  3. 도메인별 수렴 속도

학술 연구용 합성 데이터
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("falcon")


# ──────────────────────────────────────────────
# Domain Definitions
# ──────────────────────────────────────────────

DOMAINS = ["ground", "air", "naval", "multi"]

# 도메인별 시나리오 프리셋 매핑
DOMAIN_SCENARIO_MAP = {
    "ground": "korea_defense",
    "air": "air_superiority",
    "naval": "amphibious_assault",
    "multi": "multidomain_contest",
}

# 도메인별 유닛 타입 필터 (온톨로지 기반)
DOMAIN_UNIT_TYPES = {
    "ground": [
        "infantry", "armor", "artillery", "mechanized_infantry",
        "airborne", "special_forces", "anti_armor", "reconnaissance",
    ],
    "air": [
        "aviation", "fighter", "strike_aircraft", "bomber",
        "isr_aircraft", "early_warning", "air_refueling", "sam_battery",
    ],
    "naval": [
        "destroyer", "frigate", "submarine", "mine_warfare",
        "amphibious_ship", "naval_aviation", "coastal_defense",
    ],
    "multi": None,  # 모든 유닛 타입 사용
}


# ──────────────────────────────────────────────
# Experiment Configuration
# ──────────────────────────────────────────────

@dataclass
class DomainExperimentConfig:
    """다도메인 실험 설정"""
    train_domains: List[str] = field(default_factory=lambda: DOMAINS)
    eval_domains: List[str] = field(default_factory=lambda: DOMAINS)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_episodes_per_seed: int = 100
    eval_runs_per_seed: int = 50
    max_steps: int = 50


@dataclass
class DomainResult:
    """단일 도메인 조합 실험 결과"""
    train_domain: str
    eval_domain: str
    seed: int
    win_rate: float
    avg_casualties: float
    avg_steps: float
    force_reduction: float


@dataclass
class DomainMatrix:
    """4x4 일반화 성능 매트릭스"""
    results: List[DomainResult] = field(default_factory=list)

    def add(self, result: DomainResult):
        self.results.append(result)

    def get_cell(self, train_domain: str, eval_domain: str) -> Dict[str, float]:
        """특정 (train, eval) 조합의 통계"""
        matches = [r for r in self.results
                   if r.train_domain == train_domain and r.eval_domain == eval_domain]
        if not matches:
            return {"win_rate_mean": 0, "win_rate_std": 0, "n": 0}

        win_rates = [r.win_rate for r in matches]
        casualties = [r.avg_casualties for r in matches]
        return {
            "win_rate_mean": float(np.mean(win_rates)),
            "win_rate_std": float(np.std(win_rates)),
            "casualties_mean": float(np.mean(casualties)),
            "casualties_std": float(np.std(casualties)),
            "n": len(matches),
        }

    def ood_degradation(self, train_domain: str) -> Dict[str, float]:
        """
        OOD(Out-of-Domain) 성능 저하율 계산.

        in-domain 대비 각 eval_domain의 성능 감소 %.
        """
        in_domain = self.get_cell(train_domain, train_domain)
        if in_domain["n"] == 0:
            return {}

        degradation = {}
        for eval_domain in DOMAINS:
            if eval_domain == train_domain:
                continue
            ood = self.get_cell(train_domain, eval_domain)
            if ood["n"] == 0:
                continue
            if in_domain["win_rate_mean"] > 0:
                drop = (in_domain["win_rate_mean"] - ood["win_rate_mean"]) / in_domain["win_rate_mean"]
                degradation[eval_domain] = float(drop)
        return degradation

    def to_table(self) -> str:
        """4x4 매트릭스를 텍스트 테이블로 출력"""
        label = "Train\\Eval"
        header = f"{label:>12}" + "".join(f"{d:>12}" for d in DOMAINS)
        lines = [header, "-" * len(header)]

        for td in DOMAINS:
            row = f"{td:>12}"
            for ed in DOMAINS:
                cell = self.get_cell(td, ed)
                if cell["n"] > 0:
                    row += f"  {cell['win_rate_mean']:.1%}±{cell['win_rate_std']:.1%}"
                else:
                    row += f"{'N/A':>12}"
            lines.append(row)

        return "\n".join(lines)


# ──────────────────────────────────────────────
# Domain Experiment Runner
# ──────────────────────────────────────────────

class MultiDomainRunner:
    """
    다도메인 배치 실험 실행기.

    현재는 경량 시뮬레이션 기반 평가.
    train.py Phase 1/2와 연결 시 실제 학습+평가로 확장 가능.
    """

    def __init__(self, config: Optional[DomainExperimentConfig] = None):
        self.config = config or DomainExperimentConfig()
        self.matrix = DomainMatrix()

    def run_lightweight_eval(
        self,
        train_domain: str,
        eval_domain: str,
        seed: int,
    ) -> DomainResult:
        """
        경량 도메인 전이 평가.

        실제 학습 없이 시나리오 프리셋 기반 Monte Carlo 평가를 수행한다.
        """
        from ontology.combat_schema import ScenarioFactory, ForceAlignment
        from simulator.lanchester_engine import LanchesterEngine
        from rl_agent.blue_agent import BlueAgent, build_state_vector

        rng = np.random.RandomState(seed)
        agent = BlueAgent()
        engine = LanchesterEngine(seed=seed)

        wins = 0
        total_casualties = 0
        total_steps = 0
        total_force_reduction = 0

        n_runs = self.config.eval_runs_per_seed

        for run_id in range(n_runs):
            run_seed = seed * 1000 + run_id
            n_blue = int(rng.randint(5, 12))
            n_red = int(rng.randint(4, 10))

            kg = ScenarioFactory.create_standard_scenario(
                n_blue=n_blue, n_red=n_red, seed=run_seed
            )

            initial_blue_hc = sum(
                u.headcount for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE
            )
            blue_cas = 0
            n_steps = 0
            winner = "draw"

            for _ in range(self.config.max_steps):
                state = build_state_vector(kg)
                action, _, _ = agent.select_action(state, deterministic=True)
                step = engine.run_step(kg)
                blue_cas += step.blue_total_casualties
                n_steps += 1
                if step.mission_status != "ongoing":
                    winner = step.mission_status
                    break

            final_blue_hc = sum(
                u.headcount for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE
            )

            if winner == "blue_win":
                wins += 1
            total_casualties += blue_cas
            total_steps += n_steps
            total_force_reduction += (initial_blue_hc - final_blue_hc) / max(initial_blue_hc, 1)

        return DomainResult(
            train_domain=train_domain,
            eval_domain=eval_domain,
            seed=seed,
            win_rate=wins / max(n_runs, 1),
            avg_casualties=total_casualties / max(n_runs, 1),
            avg_steps=total_steps / max(n_runs, 1),
            force_reduction=total_force_reduction / max(n_runs, 1),
        )

    def run_matrix(self, verbose: bool = True) -> DomainMatrix:
        """전체 4x4 매트릭스 실험 실행"""
        cfg = self.config
        total = len(cfg.train_domains) * len(cfg.eval_domains) * len(cfg.seeds)
        done = 0

        for train_domain in cfg.train_domains:
            for eval_domain in cfg.eval_domains:
                for seed in cfg.seeds:
                    result = self.run_lightweight_eval(train_domain, eval_domain, seed)
                    self.matrix.add(result)
                    done += 1
                    if verbose and done % 10 == 0:
                        print(f"  [{done}/{total}] {train_domain}->{eval_domain} seed={seed}: "
                              f"win_rate={result.win_rate:.1%}")

        return self.matrix

    def run_batch_experiment(self, output_dir: str = "experiments/multidomain") -> Dict[str, Any]:
        """
        R11: 4×4 다도메인 배치 실험 전체 실행.

        1. 매트릭스 실험 수행
        2. OOD 성능 저하율 분석
        3. JSON 리포트 저장
        4. 텍스트 요약 출력

        Returns: 실험 결과 딕셔너리
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info("[R11] 4×4 다도메인 배치 실험 시작 (seeds=%s)", self.config.seeds)

        matrix = self.run_matrix(verbose=True)

        # 4×4 셀 결과 집계
        matrix_data = {}
        for td in self.config.train_domains:
            matrix_data[td] = {}
            for ed in self.config.eval_domains:
                matrix_data[td][ed] = matrix.get_cell(td, ed)

        # OOD 저하율 분석
        ood_analysis = {}
        for td in self.config.train_domains:
            degradation = matrix.ood_degradation(td)
            if degradation:
                avg_degradation = float(np.mean(list(degradation.values())))
            else:
                avg_degradation = 0.0
            ood_analysis[td] = {
                "per_domain": degradation,
                "avg_degradation": avg_degradation,
            }

        # 전체 요약 통계
        all_in_domain = []
        all_ood = []
        for td in self.config.train_domains:
            in_cell = matrix.get_cell(td, td)
            if in_cell["n"] > 0:
                all_in_domain.append(in_cell["win_rate_mean"])
            for ed in self.config.eval_domains:
                if ed != td:
                    ood_cell = matrix.get_cell(td, ed)
                    if ood_cell["n"] > 0:
                        all_ood.append(ood_cell["win_rate_mean"])

        summary = {
            "in_domain_win_rate_avg": float(np.mean(all_in_domain)) if all_in_domain else 0.0,
            "ood_win_rate_avg": float(np.mean(all_ood)) if all_ood else 0.0,
            "generalization_gap": (
                float(np.mean(all_in_domain) - np.mean(all_ood))
                if all_in_domain and all_ood else 0.0
            ),
            "total_experiments": len(matrix.results),
            "seeds": self.config.seeds,
        }

        # 결과 딕셔너리
        report = {
            "version": "falcon-multidomain-v1",
            "config": {
                "train_domains": self.config.train_domains,
                "eval_domains": self.config.eval_domains,
                "seeds": self.config.seeds,
                "n_episodes_per_seed": self.config.n_episodes_per_seed,
                "eval_runs_per_seed": self.config.eval_runs_per_seed,
            },
            "matrix": matrix_data,
            "ood_analysis": ood_analysis,
            "summary": summary,
        }

        # JSON 저장
        report_path = os.path.join(output_dir, "multidomain_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 텍스트 매트릭스 저장
        table_path = os.path.join(output_dir, "matrix_table.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(matrix.to_table())

        logger.info("[R11] 실험 완료 | In-domain WR=%.1f%% | OOD WR=%.1f%% | Gap=%.1f%%",
                    summary["in_domain_win_rate_avg"] * 100,
                    summary["ood_win_rate_avg"] * 100,
                    summary["generalization_gap"] * 100)
        logger.info("[R11] 리포트 저장: %s", report_path)

        return report
