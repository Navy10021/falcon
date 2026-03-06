"""
tier_c_trainer.py
=================
Tier C → Tier B 승격 학습 루프 — RARL + NFSP 통합 훈련기.

PPO (Tier A)와 동일한 평가 프레임워크(MonteCarloEvaluator)를 사용하여
RARL과 NFSP의 성능을 정량 비교할 수 있게 한다.

train.py에서 --algorithm rarl|nfsp 옵션으로 호출 가능.

사용 예시::

    # RARL 학습
    trainer = TierCTrainer(algorithm="rarl", seed=42)
    metrics = trainer.train(episodes=500)

    # NFSP 학습
    trainer = TierCTrainer(algorithm="nfsp", seed=42)
    metrics = trainer.train(episodes=500)

    # 비교 평가
    report = trainer.evaluate_vs_baseline()

학술 연구용 합성 데이터
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("falcon")


@dataclass
class TierCConfig:
    """Tier C 훈련 설정"""
    algorithm: str = "rarl"  # rarl | nfsp
    episodes: int = 500
    max_steps: int = 50
    seed: int = 42
    lr: float = 3e-4
    log_interval: int = 50
    save_interval: int = 200
    checkpoint_dir: str = "checkpoints"
    # RARL specific
    rarl_epsilon: float = 0.05
    rarl_epsilon_decay: float = 0.999
    # NFSP specific
    nfsp_eta: float = 0.1


class TierCTrainer:
    """
    RARL/NFSP 통합 훈련기.

    Phase 1 (GNN+PPO)과 동일한 시나리오·보상·평가 인터페이스 유지.
    """

    def __init__(self, config: Optional[TierCConfig] = None):
        self.config = config or TierCConfig()
        self.agent = None
        self._init_agent()

    def _init_agent(self):
        """알고리즘별 에이전트 초기화"""
        cfg = self.config
        if cfg.algorithm == "rarl":
            from rl_agent.rarl import RARLTrainer, RARLConfig
            rarl_cfg = RARLConfig(
                epsilon=cfg.rarl_epsilon,
                epsilon_decay=cfg.rarl_epsilon_decay,
                protagonist_lr=cfg.lr,
            )
            self.agent = RARLTrainer(rarl_cfg)
        elif cfg.algorithm == "nfsp":
            from rl_agent.nfsp_agent import NFSPAgent, NFSPAgentConfig
            from rl_agent.blue_agent import PPOConfig
            nfsp_cfg = NFSPAgentConfig(
                eta=cfg.nfsp_eta,
                ppo=PPOConfig(lr=cfg.lr),
            )
            self.agent = NFSPAgent(config=nfsp_cfg)
        else:
            raise ValueError(f"Unknown Tier C algorithm: {cfg.algorithm}")

    def _select_action(
        self,
        state: np.ndarray,
        use_br: Optional[bool] = None,
    ) -> Tuple[int, float, float]:
        """알고리즘별 행동 선택 (통합 인터페이스)"""
        if self.config.algorithm == "nfsp":
            return self.agent.select_action(state, use_br=use_br)
        return self.agent.select_action(state)

    def _compute_reward(
        self, step_result, force_before: int, force_after: int,
        uncertainty: float = 0.0, initial_force_size: int = 100,
    ) -> float:
        """표준 보상 함수 (BlueAgent.compute_reward와 동일 공식)"""
        # win/loss bonus
        if "blue_win" in str(step_result.mission_status):
            reward = 10.0
        elif "red_win" in str(step_result.mission_status):
            reward = -10.0
        else:
            reward = 0.0

        # casualty penalty
        reward -= 0.1 * step_result.blue_total_casualties

        # force preservation
        if initial_force_size > 0:
            reward += 0.15 * (force_after / initial_force_size) * 10.0

        # enemy damage
        reward += 0.05 * step_result.red_total_casualties

        # survival bonus
        if force_before > 0:
            reward += 0.1 * (force_after / force_before) * 10.0

        # uncertainty penalty
        reward -= 0.05 * uncertainty

        return float(reward)

    def _store_transition(
        self, state, action, reward, next_state, done, log_prob, value, use_br=True,
    ):
        """알고리즘별 transition 저장"""
        cfg = self.config
        if cfg.algorithm == "rarl":
            self.agent.prot_buffer.add(state, action, reward, log_prob, value, done, 0.0)
        elif cfg.algorithm == "nfsp":
            self.agent.store(state, action, reward, next_state, done, log_prob, value, use_br)

    def _update_agent(self) -> Dict[str, float]:
        """알고리즘별 정책 업데이트"""
        cfg = self.config
        if cfg.algorithm == "rarl":
            return self.agent.update()
        elif cfg.algorithm == "nfsp":
            return self.agent.update()
        return {}

    def train(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        학습 루프 실행.

        Phase 1과 동일한 시나리오/보상/로깅 구조.
        """
        from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
        from simulator.lanchester_engine import LanchesterEngine
        from simulator.fog_of_war import FogOfWarFilter, CurriculumScheduler
        from rl_agent.blue_agent import build_state_vector, BlueActionSpace
        from train import _build_blue_action_pairs

        cfg = self.config
        n_episodes = episodes or cfg.episodes
        engine = LanchesterEngine(seed=cfg.seed)
        curriculum = CurriculumScheduler()
        rng = np.random.RandomState(cfg.seed)

        rewards_history = []
        win_rate_history = []

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        logger.info("[Tier C/%s] 학습 시작 | episodes=%d", cfg.algorithm.upper(), n_episodes)

        for ep in range(n_episodes):
            progress = ep / max(n_episodes, 1)
            fog_level, _ = curriculum.update(progress)
            fog_filter = FogOfWarFilter(fog_level, seed=cfg.seed + ep)

            kg = ScenarioFactory.create_standard_scenario(
                n_blue=int(rng.randint(5, 12)),
                n_red=int(rng.randint(4, 9)),
                seed=cfg.seed + ep,
            )
            initial_blue_hc = sum(
                u.headcount for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE
            )
            prev_blue_hc = initial_blue_hc
            episode_reward = 0.0
            done = False
            step_result = None

            for step_t in range(cfg.max_steps):
                if done:
                    break

                obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)
                state = build_state_vector(obs_kg, uncertainty_map=uncertainty_map)
                avg_unc = float(np.mean(list(uncertainty_map.values()))) if uncertainty_map else 0.0

                use_br = True
                if cfg.algorithm == "nfsp":
                    use_br = self.agent.should_use_br()

                action, log_prob, value = self._select_action(state, use_br=use_br)

                action_pairs = _build_blue_action_pairs(
                    kg, action, ForceAlignment, UnitStatus, BlueActionSpace,
                )
                step_result = engine.run_step(kg, action_pairs=action_pairs)
                done = step_result.mission_status != "ongoing"

                current_hc = step_result.blue_total_headcount
                reward = self._compute_reward(
                    step_result, prev_blue_hc, current_hc, avg_unc,
                    initial_force_size=initial_blue_hc,
                )

                # next state for NFSP buffer
                if not done:
                    next_obs_kg, _ = fog_filter.observe(kg, ForceAlignment.BLUE)
                    next_state = build_state_vector(next_obs_kg)
                else:
                    next_state = state  # terminal

                self._store_transition(
                    state, action, reward, next_state, done, log_prob, value, use_br,
                )

                prev_blue_hc = current_hc
                episode_reward += reward

            # 정책 업데이트
            update_metrics = self._update_agent()

            # 기록
            win = 1 if step_result and "blue_win" in str(step_result.mission_status) else 0
            rewards_history.append(episode_reward)
            win_rate_history.append(win)

            if ep % cfg.log_interval == 0 and ep > 0:
                recent_wr = np.mean(win_rate_history[-50:]) if len(win_rate_history) >= 50 else np.mean(win_rate_history)
                recent_r = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
                logger.info("EP %5d/%d | %s | WR=%.1f%% | R=%.2f",
                            ep, n_episodes, cfg.algorithm.upper(),
                            recent_wr * 100, recent_r)

            # 체크포인트
            if ep > 0 and ep % cfg.save_interval == 0:
                self._save_checkpoint(ep)

        # 최종 저장
        self._save_checkpoint("final")

        final_wr = float(np.mean(win_rate_history[-100:])) if win_rate_history else 0.0
        final_r = float(np.mean(rewards_history[-100:])) if rewards_history else 0.0

        logger.info("[Tier C/%s] 학습 완료 | 최종 WR=%.1f%% | R=%.2f",
                    cfg.algorithm.upper(), final_wr * 100, final_r)

        return {
            "algorithm": cfg.algorithm,
            "episodes": n_episodes,
            "final_win_rate": final_wr,
            "final_avg_reward": final_r,
            "rewards_history": rewards_history,
            "win_rate_history": win_rate_history,
        }

    def _save_checkpoint(self, label):
        """체크포인트 저장"""
        cfg = self.config
        path = os.path.join(cfg.checkpoint_dir, f"{cfg.algorithm}_{label}.pt")
        if hasattr(self.agent, "save"):
            self.agent.save(path)
        else:
            # RARL: save protagonist state dict
            torch.save({
                "protagonist": self.agent.protagonist.state_dict(),
                "adversary": self.agent.adversary.state_dict(),
            }, path)
        logger.info("  체크포인트 저장: %s", path)

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        BlueAgent 호환 인터페이스 — MonteCarloEvaluator에서 직접 사용 가능.

        MonteCarloEvaluator._evaluate_sequential()은 agent.select_action(state)를
        호출하므로, TierCTrainer를 동일 평가 경로에서 사용하기 위해 공개 메서드로 노출.
        """
        return self._select_action(state)

    def evaluate_vs_baseline(self, n_runs: int = 50) -> Dict[str, Any]:
        """
        PPO 베이스라인 대비 Tier C 에이전트 성능 비교 평가.

        MonteCarloEvaluator를 사용하여 Tier C와 기본 PPO 에이전트를
        동일한 평가 경로(동일 엔진·시나리오 분포·지표 계산)로 평가한다.
        이전의 내부 경량 루프 대신 표준 MC 평가 프레임워크를 사용하여
        Tier A (PPO)와의 비교 신뢰도를 보장한다.
        """
        from evaluation.monte_carlo import MonteCarloEvaluator
        from simulator.lanchester_engine import LanchesterEngine
        from rl_agent.blue_agent import BlueAgent

        engine = LanchesterEngine(seed=self.config.seed)
        ppo_agent = BlueAgent()

        mc_eval = MonteCarloEvaluator(
            n_runs=n_runs,
            seed=self.config.seed,
            n_workers=1,
        )

        tier_c_report = mc_eval.evaluate(
            agent=self,
            engine=engine,
            max_steps=self.config.max_steps,
            verbose=False,
            show_progress=False,
        )
        ppo_report = mc_eval.evaluate(
            agent=ppo_agent,
            engine=engine,
            max_steps=self.config.max_steps,
            verbose=False,
            show_progress=False,
        )

        comparison = {
            "tier_c": {
                "win_rate":          tier_c_report.blue_win_rate,
                "win_rate_ci":       tier_c_report.blue_win_rate_ci,
                "avg_casualties":    tier_c_report.avg_blue_casualties,
                "strategy_robustness": tier_c_report.strategy_robustness,
                "cvar_10":           tier_c_report.cvar_10,
            },
            "ppo": {
                "win_rate":          ppo_report.blue_win_rate,
                "win_rate_ci":       ppo_report.blue_win_rate_ci,
                "avg_casualties":    ppo_report.avg_blue_casualties,
                "strategy_robustness": ppo_report.strategy_robustness,
                "cvar_10":           ppo_report.cvar_10,
            },
            "delta_win_rate":    tier_c_report.blue_win_rate - ppo_report.blue_win_rate,
            "delta_cvar_10":     tier_c_report.cvar_10 - ppo_report.cvar_10,
            "tier_c_algorithm":  self.config.algorithm,
            "n_runs":            n_runs,
        }
        return comparison

    @staticmethod
    def save_comparison_report(report: Dict, path: str):
        """비교 평가 보고서 JSON 저장"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
