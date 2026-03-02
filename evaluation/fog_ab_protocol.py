"""
fog_ab_protocol.py
==================
Fog Curriculum A/B 측정 프로토콜.

커리큘럼 러닝(점진적 Fog 증가)의 효과를 통제 실험으로 측정한다.

두 조건(A=커리큘럼 ON, B=커리큘럼 OFF / 고정 Fog)을 비교하여
승률·보상·불확실성 처리 능력의 차이를 통계적으로 검증한다.

사용 예시::

    protocol = FogABProtocol(n_episodes=100, seed=42)
    report = protocol.run(blue_agent, engine)
    protocol.save_report(report, "fog_ab_results.json")

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
# Configuration
# ──────────────────────────────────────────────

@dataclass
class ABCondition:
    """A/B 실험 조건 정의"""
    name: str
    use_curriculum: bool = True
    fixed_fog_level: Optional[str] = None  # 커리큘럼 OFF 시 고정 Fog (CLEAR/MODERATE/HEAVY)
    description: str = ""


@dataclass
class FogABConfig:
    """A/B 실험 전체 설정"""
    n_episodes: int = 100
    max_steps_per_episode: int = 50
    seed: int = 42
    conditions: List[ABCondition] = field(default_factory=lambda: [
        ABCondition(
            name="curriculum_on",
            use_curriculum=True,
            description="점진적 Fog 커리큘럼 (CLEAR→MAXIMUM)",
        ),
        ABCondition(
            name="fixed_moderate",
            use_curriculum=False,
            fixed_fog_level="MODERATE",
            description="고정 Fog=MODERATE (통제 조건)",
        ),
    ])


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

@dataclass
class ConditionMetrics:
    """단일 조건의 측정 결과"""
    condition_name: str
    win_rate: float = 0.0
    avg_reward: float = 0.0
    avg_survival_rate: float = 0.0
    avg_episode_length: float = 0.0
    reward_std: float = 0.0
    # Fog-specific: 고Fog 구간에서의 성능
    high_fog_win_rate: float = 0.0
    # 보상 궤적 (에피소드별)
    episode_rewards: List[float] = field(default_factory=list)
    episode_wins: List[int] = field(default_factory=list)


@dataclass
class ABReport:
    """A/B 비교 보고서"""
    config: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, ConditionMetrics] = field(default_factory=dict)
    # 통계 비교
    delta_win_rate: float = 0.0
    delta_reward: float = 0.0
    effect_size_cohens_d: float = 0.0
    curriculum_benefit: bool = False
    summary: str = ""


# ──────────────────────────────────────────────
# Protocol Runner
# ──────────────────────────────────────────────

class FogABProtocol:
    """
    Fog Curriculum A/B 실험 프로토콜.

    동일한 에이전트·엔진을 사용해 두 조건(커리큘럼 ON/OFF)을 비교한다.
    """

    def __init__(self, config: Optional[FogABConfig] = None):
        self.config = config or FogABConfig()

    def _run_condition(
        self,
        condition: ABCondition,
        agent,
        engine,
    ) -> ConditionMetrics:
        """단일 조건 실행"""
        from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
        from rl_agent.blue_agent import build_state_vector, BlueActionSpace
        from simulator.fog_of_war import FogOfWarFilter, FogLevel, CurriculumScheduler

        rng = np.random.RandomState(self.config.seed)
        metrics = ConditionMetrics(condition_name=condition.name)
        curriculum = CurriculumScheduler() if condition.use_curriculum else None

        # 고정 Fog 레벨 결정
        if not condition.use_curriculum and condition.fixed_fog_level:
            fixed_level = getattr(FogLevel, condition.fixed_fog_level.upper(), FogLevel.MODERATE)
        else:
            fixed_level = FogLevel.MODERATE

        high_fog_wins = []

        for ep in range(self.config.n_episodes):
            progress = ep / max(self.config.n_episodes, 1)

            # Fog 레벨 결정
            if curriculum:
                fog_level, _ = curriculum.update(progress)
            else:
                fog_level = fixed_level

            fog_filter = FogOfWarFilter(fog_level, seed=self.config.seed + ep)

            kg = ScenarioFactory.create_standard_scenario(
                n_blue=rng.randint(5, 10),
                n_red=rng.randint(4, 8),
                seed=self.config.seed + ep,
            )
            initial_blue_hc = sum(
                u.headcount for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE
            )

            episode_reward = 0.0
            done = False
            step_count = 0
            step_result = None

            for step_t in range(self.config.max_steps_per_episode):
                if done:
                    break
                step_count += 1

                obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)
                state = build_state_vector(obs_kg, uncertainty_map=uncertainty_map)

                with __import__("torch").no_grad():
                    action, log_prob, value = agent.select_action(state)

                from train import _build_blue_action_pairs
                action_pairs = _build_blue_action_pairs(
                    kg, action, ForceAlignment, UnitStatus, BlueActionSpace,
                )
                step_result = engine.run_step(kg, action_pairs=action_pairs)
                done = (step_result.mission_status != "ongoing")

                current_hc = step_result.blue_total_headcount
                reward = agent.compute_reward(
                    step_result, initial_blue_hc, current_hc, 0.0,
                    initial_force_size=initial_blue_hc,
                )
                episode_reward += reward

            # 결과 기록
            win = 1 if step_result and "blue_win" in str(step_result.mission_status) else 0
            final_hc = step_result.blue_total_headcount if step_result else 0
            survival = final_hc / max(initial_blue_hc, 1)

            metrics.episode_rewards.append(episode_reward)
            metrics.episode_wins.append(win)

            # High fog (HEAVY 이상) 성능 추적
            if fog_level.value >= FogLevel.HEAVY.value:
                high_fog_wins.append(win)

        # 집계
        metrics.win_rate = float(np.mean(metrics.episode_wins))
        metrics.avg_reward = float(np.mean(metrics.episode_rewards))
        metrics.reward_std = float(np.std(metrics.episode_rewards))
        metrics.avg_episode_length = float(self.config.max_steps_per_episode)
        metrics.high_fog_win_rate = float(np.mean(high_fog_wins)) if high_fog_wins else 0.0

        return metrics

    def run(self, agent, engine) -> ABReport:
        """A/B 실험 실행 후 비교 보고서 반환."""
        report = ABReport(config={"n_episodes": self.config.n_episodes, "seed": self.config.seed})

        for condition in self.config.conditions:
            logger.info("[Fog A/B] Running condition: %s (%s)", condition.name, condition.description)
            metrics = self._run_condition(condition, agent, engine)
            report.conditions[condition.name] = metrics
            logger.info("  → WR=%.1f%% | AvgR=%.2f | HighFogWR=%.1f%%",
                        metrics.win_rate * 100, metrics.avg_reward, metrics.high_fog_win_rate * 100)

        # 통계 비교
        names = list(report.conditions.keys())
        if len(names) >= 2:
            a = report.conditions[names[0]]
            b = report.conditions[names[1]]
            report.delta_win_rate = a.win_rate - b.win_rate
            report.delta_reward = a.avg_reward - b.avg_reward

            # Cohen's d effect size
            pooled_std = np.sqrt((a.reward_std ** 2 + b.reward_std ** 2) / 2)
            if pooled_std > 1e-8:
                report.effect_size_cohens_d = float(report.delta_reward / pooled_std)
            else:
                report.effect_size_cohens_d = 0.0

            report.curriculum_benefit = report.delta_win_rate > 0
            report.summary = (
                f"커리큘럼 효과: WR Δ={report.delta_win_rate:+.1%}, "
                f"Reward Δ={report.delta_reward:+.2f}, "
                f"Cohen's d={report.effect_size_cohens_d:.2f} "
                f"({'유의미' if abs(report.effect_size_cohens_d) > 0.2 else '미미'})"
            )
            logger.info("[Fog A/B] %s", report.summary)

        return report

    @staticmethod
    def save_report(report: ABReport, path: str):
        """보고서를 JSON으로 저장."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "config": report.config,
            "conditions": {},
            "delta_win_rate": report.delta_win_rate,
            "delta_reward": report.delta_reward,
            "effect_size_cohens_d": report.effect_size_cohens_d,
            "curriculum_benefit": report.curriculum_benefit,
            "summary": report.summary,
        }
        for name, m in report.conditions.items():
            data["conditions"][name] = {
                "condition_name": m.condition_name,
                "win_rate": m.win_rate,
                "avg_reward": m.avg_reward,
                "reward_std": m.reward_std,
                "high_fog_win_rate": m.high_fog_win_rate,
                "n_episodes": len(m.episode_rewards),
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
