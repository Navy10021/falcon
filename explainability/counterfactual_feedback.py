"""
counterfactual_feedback.py
==========================
R13: Counterfactual 분석 → 보상 피드백 연결

학습 루프에 통합되는 경량 counterfactual 분석 모듈.
에피소드 종료 후 각 스텝에서 "다른 행동을 선택했다면?"에 대한
예상 수익을 시뮬레이션하여 CounterfactualRewardShaper에 전달한다.

통합 흐름:
    1. 에피소드 실행 중 각 스텝의 (state, action, reward, kg_snapshot) 기록
    2. 에피소드 종료 후 각 스텝에서 6가지 대안 행동의 1-step 시뮬레이션
    3. CounterfactualRewardShaper.compute_shaping() → 보정 신호 생성
    4. apply_shaping() → PPO 버퍼 보상에 보정 반영
    5. 다음 PPO 업데이트에서 보정된 보상으로 학습

학술 연구용 합성 데이터
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from explainability.reward_shaper import (
    CounterfactualRewardShaper,
    RewardShapingConfig,
    ShapingSignal,
)

logger = logging.getLogger("falcon")


@dataclass
class StepSnapshot:
    """에피소드 내 단일 스텝의 상태 기록"""
    step: int
    action: int
    reward: float
    kg_snapshot: object  # deepcopy of CombatKnowledgeGraph (lightweight)
    state: object = None  # optional numpy state vector
    fog_filter: object = None  # optional FogOfWarFilter reference


@dataclass
class CounterfactualFeedbackConfig:
    """R13 Counterfactual 피드백 설정"""
    enabled: bool = True
    n_actions: int = 6  # BlueActionSpace.N_ACTIONS
    max_cf_steps: int = 3  # counterfactual 시뮬레이션 깊이 (1-step 빠른 평가)
    cf_discount: float = 0.99
    shaping_config: Optional[RewardShapingConfig] = None
    warmup_episodes: int = 10  # 이 이후부터 CF 보정 적용
    cf_interval: int = 1  # 몇 에피소드마다 CF 분석 수행 (1=매 에피소드)
    log_top_regrets: int = 3  # 에피소드별 상위 N개 regret 로깅


class CounterfactualFeedbackLoop:
    """
    학습 루프에 통합되는 Counterfactual 피드백 매니저.

    사용법::

        cf_loop = CounterfactualFeedbackLoop()

        for ep in range(episodes):
            cf_loop.begin_episode()

            for step_t in range(max_steps):
                # ... 행동 선택 및 시뮬레이션 ...
                cf_loop.record_step(step_t, action, reward, kg_snapshot)

            # 에피소드 종료 후 CF 분석 및 보상 보정
            shaped_rewards = cf_loop.end_episode(
                episode_rewards, engine, ForceAlignment, UnitStatus, BlueActionSpace
            )
            # shaped_rewards를 PPO 버퍼에 반영
    """

    def __init__(self, config: Optional[CounterfactualFeedbackConfig] = None):
        self.config = config or CounterfactualFeedbackConfig()
        shaping_cfg = self.config.shaping_config or RewardShapingConfig()
        self.shaper = CounterfactualRewardShaper(config=shaping_cfg)
        self._step_buffer: List[StepSnapshot] = []
        self._episode_count = 0
        self._total_shaped_episodes = 0
        self._cumulative_win_rate_delta = 0.0

    def begin_episode(self):
        """에피소드 시작 시 버퍼 초기화"""
        self._step_buffer = []

    def record_step(
        self,
        step: int,
        action: int,
        reward: float,
        kg_snapshot: object,
    ):
        """각 스텝의 상태를 기록 (kg_snapshot은 deepcopy)"""
        self._step_buffer.append(StepSnapshot(
            step=step,
            action=action,
            reward=reward,
            kg_snapshot=kg_snapshot,
        ))

    def end_episode(
        self,
        episode_rewards: List[float],
        engine,
        ForceAlignment,
        UnitStatus,
        BlueActionSpace,
        build_action_pairs_fn=None,
    ) -> List[float]:
        """
        에피소드 종료 후 counterfactual 분석 수행 및 보상 보정.

        Parameters
        ----------
        episode_rewards : list[float]
            원래 에피소드 보상 시퀀스
        engine : LanchesterEngine
            시뮬레이션 엔진 (counterfactual rollout용)
        ForceAlignment, UnitStatus, BlueActionSpace
            ontology 열거형
        build_action_pairs_fn : callable, optional
            _build_blue_action_pairs 함수 참조

        Returns
        -------
        list[float]
            보정된 보상 시퀀스 (warmup 전이면 원본 그대로)
        """
        self._episode_count += 1

        # warmup 기간이거나 interval 아닌 에피소드면 원본 반환
        if self._episode_count <= self.config.warmup_episodes:
            return list(episode_rewards)
        if self._episode_count % self.config.cf_interval != 0:
            return list(episode_rewards)

        # counterfactual 결과 생성
        cf_results = self._compute_counterfactual_returns(
            engine, ForceAlignment, UnitStatus, BlueActionSpace,
            build_action_pairs_fn,
        )

        if not cf_results:
            return list(episode_rewards)

        # 행동/보상 시퀀스 추출
        actions = [snap.action for snap in self._step_buffer]
        rewards = [snap.reward for snap in self._step_buffer]

        # CounterfactualRewardShaper로 보정 신호 계산
        signals = self.shaper.compute_shaping(actions, rewards, cf_results)

        if signals:
            self._total_shaped_episodes += 1

            # 상위 regret 로깅
            if self.config.log_top_regrets > 0:
                top = sorted(signals, key=lambda s: s.regret, reverse=True)[
                    : self.config.log_top_regrets
                ]
                for sig in top:
                    logger.debug(
                        "[CF] ep=%d %s", self._episode_count, sig.explanation
                    )

        # 보정 적용
        shaped = self.shaper.apply_shaping(list(episode_rewards), signals)
        return shaped

    def _compute_counterfactual_returns(
        self,
        engine,
        ForceAlignment,
        UnitStatus,
        BlueActionSpace,
        build_action_pairs_fn,
    ) -> List[Dict]:
        """
        각 스텝에서 6가지 대안 행동의 1-step 시뮬레이션.

        Returns
        -------
        list[dict]
            각 스텝에 대해 {action_idx: expected_return} 형식
        """
        n_actions = self.config.n_actions
        cf_results = []

        for snap in self._step_buffer:
            step_cf: Dict[int, float] = {}

            for alt_action in range(n_actions):
                try:
                    # KG 스냅샷 복원
                    kg_copy = copy.deepcopy(snap.kg_snapshot)

                    # 대안 행동으로 1-step 시뮬레이션
                    if build_action_pairs_fn is not None:
                        action_pairs = build_action_pairs_fn(
                            kg_copy, alt_action, ForceAlignment, UnitStatus, BlueActionSpace
                        )
                        result = engine.run_step(kg_copy, action_pairs=action_pairs)
                    else:
                        result = engine.run_step(kg_copy)

                    # 즉각 보상 추정 (단순화: blue_cas - red_cas 기반)
                    blue_cas = result.blue_total_casualties
                    red_cas = result.red_total_casualties
                    win_bonus = (
                        10.0 if result.mission_status == "blue_win"
                        else -10.0 if result.mission_status == "red_win"
                        else 0.0
                    )

                    # heuristic return estimate
                    cf_return = (
                        win_bonus
                        - blue_cas * 0.5
                        + red_cas * 0.3
                    )
                    step_cf[alt_action] = cf_return

                except Exception:
                    # 시뮬레이션 실패 시 해당 행동 스킵
                    continue

            cf_results.append(step_cf)

        return cf_results

    def apply_to_buffer(
        self,
        buffer,
        shaped_rewards: List[float],
        original_rewards: List[float],
    ):
        """
        PPO 버퍼의 보상을 보정된 값으로 갱신.

        buffer.rewards 텐서의 마지막 N개 항목을 보정값으로 교체.
        """
        n = len(shaped_rewards)
        if not hasattr(buffer, "rewards") or buffer.rewards is None:
            return

        import torch
        rewards_tensor = buffer.rewards
        buf_len = len(rewards_tensor)

        # 버퍼 끝에서 n개 항목 교체
        start_idx = max(0, buf_len - n)
        for i, (shaped, orig) in enumerate(zip(shaped_rewards, original_rewards)):
            idx = start_idx + i
            if idx < buf_len:
                rewards_tensor[idx] = shaped

    def get_summary(self) -> Dict[str, float]:
        """CF 피드백 통계 요약"""
        shaper_summary = self.shaper.get_summary()
        return {
            "cf_episodes_total": self._episode_count,
            "cf_episodes_shaped": self._total_shaped_episodes,
            "cf_shaping_ratio": (
                self._total_shaped_episodes / max(self._episode_count, 1)
            ),
            **shaper_summary,
        }

    def get_top_regrets(self, n: int = 5) -> List[ShapingSignal]:
        """전체 학습에서 가장 높은 regret을 가진 신호 반환"""
        return self.shaper.get_top_regrets(n)
