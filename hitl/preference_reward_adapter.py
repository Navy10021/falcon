"""
preference_reward_adapter.py
============================
P3-3: HITL 선호도 → PPO 보상 함수 어댑터

CommanderPreferenceLearner의 선호도 가중치를
BlueAgent 보상 계수(PPO Config 레벨)로 변환한다.

사용 예:
    adapter = PreferenceRewardAdapter.from_file("checkpoints/hitl_preferences.json")
    scaled_config = adapter.adapt_ppo_config(base_ppo_config)
    reward = blue_agent.compute_reward(..., config=scaled_config)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class RewardScale:
    """
    보상 계수 스케일 팩터.

    각 항목은 BlueAgent 기본 계수(W_*)에 곱할 배율.
    1.0 = 기본값 유지, >1.0 = 강화, <1.0 = 약화.
    """
    force_save_scale:    float = 1.0   # W_FORCE_SAVE 배율 (병력 절감)
    casualty_scale:      float = 1.0   # W_CASUALTY 배율 (사상자 패널티)
    win_scale:           float = 1.0   # W_WIN 배율 (승리 보상)
    force_ratio_scale:   float = 1.0   # W_FORCE_RATIO 배율 (잔존 비율)


class PreferenceRewardAdapter:
    """
    CommanderPreferenceLearner 선호도 → BlueAgent 보상 스케일 변환기.

    preference_weights 3개 값을 정규화해 각 보상 항목의 가중치를
    0.5× ~ 2.0× 범위에서 선형 스케일링한다.

    스케일 공식 (w ∈ [0.1, 0.6], normalized):
        scale = 0.5 + 1.5 * w_normalized
        (w_normalized = 0 → scale 0.5, w_normalized = 1 → scale 2.0)
    """

    _MIN_SCALE = 0.5
    _MAX_SCALE = 2.0

    def __init__(self, preference_weights: dict):
        """
        Parameters
        ----------
        preference_weights : dict
            {"force_size_weight": float, "win_rate_weight": float, "casualty_weight": float}
        """
        self.preference_weights = preference_weights
        self._reward_scale = self._compute_scale()

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "PreferenceRewardAdapter":
        """저장된 선호도 JSON에서 어댑터 생성."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"선호도 파일을 찾을 수 없습니다: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls(data["preference_weights"])

    @classmethod
    def uniform(cls) -> "PreferenceRewardAdapter":
        """균등 선호도(기본값) 어댑터."""
        return cls({"force_size_weight": 0.33, "win_rate_weight": 0.33, "casualty_weight": 0.34})

    # ------------------------------------------------------------------
    def _normalize(self, key: str) -> float:
        """
        preference_weights 값 → [0, 1] 정규화.
        실제 범위는 [0.1, 0.6] (학습기 clip 기준)이므로 선형 맵핑.
        """
        val = self.preference_weights.get(key, 0.33)
        val_norm = (val - 0.10) / (0.60 - 0.10)
        return float(max(0.0, min(1.0, val_norm)))

    def _to_scale(self, norm: float) -> float:
        return self._MIN_SCALE + (self._MAX_SCALE - self._MIN_SCALE) * norm

    def _compute_scale(self) -> RewardScale:
        force_norm    = self._normalize("force_size_weight")
        win_norm      = self._normalize("win_rate_weight")
        casualty_norm = self._normalize("casualty_weight")

        return RewardScale(
            force_save_scale  = self._to_scale(force_norm),
            casualty_scale    = self._to_scale(casualty_norm),
            win_scale         = self._to_scale(win_norm),
            force_ratio_scale = self._to_scale(force_norm),  # 병력 관련
        )

    @property
    def reward_scale(self) -> RewardScale:
        return self._reward_scale

    # ------------------------------------------------------------------
    def compute_scaled_reward(
        self,
        base_reward: float,
        win_reward: float,
        force_reward: float,
        casualty_penalty: float,
        survival_bonus: float,
        enemy_damage: float,
        doctrine_bonus: float,
        uncertainty_penalty: float,
    ) -> float:
        """
        기본 보상 항목들에 선호도 스케일을 적용해 최종 보상 계산.

        BlueAgent.compute_reward() 에서 각 항목을 분리해 전달하거나,
        스케일을 적용한 delta를 더하는 방식으로 사용 가능.
        """
        s = self._reward_scale
        scaled = (
            win_reward         * s.win_scale
            - casualty_penalty * s.casualty_scale
            + force_reward     * s.force_save_scale
            + survival_bonus   * s.force_ratio_scale
            + enemy_damage
            + doctrine_bonus
            - uncertainty_penalty
        )
        return float(scaled)

    def summary(self) -> str:
        """어댑터 스케일 요약 문자열."""
        s = self._reward_scale
        return (
            f"PreferenceRewardAdapter | "
            f"force×{s.force_save_scale:.2f} "
            f"casualty×{s.casualty_scale:.2f} "
            f"win×{s.win_scale:.2f}"
        )
