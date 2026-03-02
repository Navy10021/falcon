"""
reward_shaper.py
================
Explainability → Reward Feedback 연결

Counterfactual 분석 결과를 보상 보정(reward shaping)에 반영하여
설명 가능성이 정책 개선에 직접 기여하도록 한다.

핵심 아이디어:
  1. Counterfactual 분석에서 "놓친 기회"를 식별
  2. 해당 행동에 대한 보상 보정항(shaping bonus)을 계산
  3. 경험 재생(experience replay) 우선순위를 조정

이 모듈은 학습 루프에 직접 삽입되지 않고, 에피소드 종료 후
보상을 소급 보정하는 포스트-프로세싱 방식으로 작동한다.

학술 연구용 합성 데이터
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Counterfactual Reward Shaping
# ──────────────────────────────────────────────

@dataclass
class ShapingSignal:
    """단일 스텝의 보상 보정 신호"""
    step: int
    action_taken: int
    best_counterfactual_action: int
    regret: float         # 최적 행동 대비 손실 (>0 이면 놓친 기회)
    shaping_bonus: float  # 보정 보상 (다음 학습에서 적용)
    explanation: str      # 보정 이유 (자연어)


@dataclass
class RewardShapingConfig:
    """보상 보정 설정"""
    shaping_scale: float = 0.1        # 보정 보상의 스케일
    regret_threshold: float = 0.05    # 이 이상의 regret만 보정
    max_bonus: float = 1.0            # 최대 보정 보상
    decay_factor: float = 0.95        # 보정 강도 점진적 감소 (에피소드 증가 시)
    priority_boost: float = 2.0       # 높은 regret 경험의 replay 우선순위 배수


class CounterfactualRewardShaper:
    """
    Counterfactual 기반 보상 보정기.

    에피소드 종료 후 각 스텝의 보상을 소급 보정한다.
    """

    def __init__(self, config: Optional[RewardShapingConfig] = None):
        self.config = config or RewardShapingConfig()
        self.episode_count = 0
        self.total_shaped_steps = 0
        self.cumulative_regret = 0.0
        self._signals_history: List[List[ShapingSignal]] = []

    def compute_shaping(
        self,
        episode_actions: List[int],
        episode_rewards: List[float],
        counterfactual_results: List[Dict],
    ) -> List[ShapingSignal]:
        """
        에피소드의 행동-보상 시퀀스에 대해 counterfactual 보정 계산.

        Parameters
        ----------
        episode_actions : list[int]
            에피소드에서 선택한 행동 시퀀스
        episode_rewards : list[float]
            에피소드에서 받은 보상 시퀀스
        counterfactual_results : list[dict]
            각 스텝의 counterfactual 분석 결과.
            각 dict는 {action_idx: expected_return} 형식의 추정.

        Returns
        -------
        list[ShapingSignal]
        """
        cfg = self.config
        signals = []
        decay = cfg.decay_factor ** self.episode_count

        for step, (action, reward) in enumerate(zip(episode_actions, episode_rewards)):
            if step >= len(counterfactual_results):
                break

            cf = counterfactual_results[step]
            if not cf:
                continue

            # 최적 counterfactual 행동 찾기
            best_action = max(cf, key=cf.get)
            best_return = cf[best_action]
            actual_return = cf.get(action, reward)

            # regret 계산
            regret = max(0, best_return - actual_return)

            if regret < cfg.regret_threshold:
                continue

            # 보정 보상 계산
            raw_bonus = cfg.shaping_scale * regret * decay
            bonus = min(raw_bonus, cfg.max_bonus)

            # 설명 생성
            from rl_agent.blue_agent import BlueActionSpace
            action_names = BlueActionSpace.ACTION_NAMES
            taken_name = action_names[action] if action < len(action_names) else f"Action-{action}"
            best_name = action_names[best_action] if best_action < len(action_names) else f"Action-{best_action}"

            explanation = (
                f"Step {step}: '{taken_name}' 선택 → regret={regret:.3f}. "
                f"'{best_name}'이 더 나았을 것 (예상 개선: {regret:.1%})"
            )

            signals.append(ShapingSignal(
                step=step,
                action_taken=action,
                best_counterfactual_action=best_action,
                regret=regret,
                shaping_bonus=bonus,
                explanation=explanation,
            ))

            self.total_shaped_steps += 1
            self.cumulative_regret += regret

        self.episode_count += 1
        self._signals_history.append(signals)
        return signals

    def apply_shaping(
        self,
        rewards: List[float],
        signals: List[ShapingSignal],
    ) -> List[float]:
        """
        보정 신호를 보상 시퀀스에 적용.

        보정 보상은 해당 스텝의 보상에 **가산**된다.
        음수 보정(패널티)은 적용하지 않는다.
        """
        shaped_rewards = list(rewards)
        for sig in signals:
            if 0 <= sig.step < len(shaped_rewards):
                # 해당 스텝에서 놓친 기회만큼 추가 보상 (다음 학습 시 적용)
                # 원래 보상은 유지하고, 보정은 양수만 가산
                shaped_rewards[sig.step] += sig.shaping_bonus
        return shaped_rewards

    def compute_replay_priorities(
        self,
        signals: List[ShapingSignal],
        n_steps: int,
    ) -> np.ndarray:
        """
        경험 재생 우선순위 계산.

        높은 regret을 가진 스텝에 더 높은 우선순위를 부여한다.
        """
        cfg = self.config
        priorities = np.ones(n_steps, dtype=np.float32)

        for sig in signals:
            if 0 <= sig.step < n_steps:
                priorities[sig.step] *= cfg.priority_boost * (1 + sig.regret)

        # 정규화
        priorities /= priorities.sum()
        return priorities

    def get_summary(self) -> Dict[str, float]:
        """보정 통계 요약"""
        return {
            "episodes_processed": self.episode_count,
            "total_shaped_steps": self.total_shaped_steps,
            "cumulative_regret": self.cumulative_regret,
            "avg_regret_per_episode": (
                self.cumulative_regret / max(self.episode_count, 1)
            ),
            "shaping_rate": (
                self.total_shaped_steps / max(self.episode_count * 50, 1)
            ),
        }

    def get_top_regrets(self, n: int = 5) -> List[ShapingSignal]:
        """가장 높은 regret을 가진 보정 신호 top-N"""
        all_signals = [s for episode in self._signals_history for s in episode]
        return sorted(all_signals, key=lambda s: s.regret, reverse=True)[:n]
