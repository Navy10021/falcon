"""
preference_learner.py
=====================
Phase 3: 지휘관 선호도 학습
- 선택 패턴 누적 → 선호도 벡터 업데이트
- PPO 상태에 선호도 포함
- 맞춤형 전략 후보 생성
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import json
import os

from hitl.pareto_generator import StrategyOption, StrategyType


@dataclass
class SelectionRecord:
    """지휘관 선택 기록"""
    scenario_id: int
    options_presented: List[str]   # option_id 목록
    selected_option_id: str
    selected_type: str
    force_size: int
    win_probability: float
    expected_casualties: int
    feedback_rating: Optional[int] = None  # 1~5


class CommanderPreferenceLearner:
    """
    지휘관 선호도 학습기
    선택 패턴 → 선호도 벡터 → 맞춤 가중치
    """

    def __init__(self, commander_id: str = "default"):
        self.commander_id = commander_id
        self.records: List[SelectionRecord] = []
        
        # 선호도 가중치 벡터 (초기값: 균등)
        self.preference_weights = {
            "force_size_weight": 0.33,       # 병력 절감 선호도
            "win_rate_weight": 0.33,         # 승률 선호도
            "casualty_weight": 0.34,         # 사상자 최소화 선호도
        }
        
        # 전략 유형 선호 빈도
        self.type_frequency: Dict[str, int] = {t.value: 0 for t in StrategyType}
        self.adoption_rate = 0.0  # AI 추천 채택률

    def record_selection(self, record: SelectionRecord):
        """지휘관 선택 기록 및 선호도 업데이트"""
        self.records.append(record)
        self.type_frequency[record.selected_type] = \
            self.type_frequency.get(record.selected_type, 0) + 1
        self._update_weights(record)
        self._update_adoption_rate(record)

    def _update_weights(self, record: SelectionRecord, lr: float = 0.05):
        """
        온라인 학습으로 선호도 가중치 업데이트
        선택된 전략의 특성을 강화
        """
        selected_type = record.selected_type
        
        # 전략 유형별 가중치 조정
        if selected_type == StrategyType.MIN_FORCE.value:
            self.preference_weights["force_size_weight"] = min(
                self.preference_weights["force_size_weight"] + lr, 0.6)
        elif selected_type == StrategyType.MIN_CASUALTY.value:
            self.preference_weights["casualty_weight"] = min(
                self.preference_weights["casualty_weight"] + lr, 0.6)
        elif selected_type == StrategyType.MAX_WIN_RATE.value:
            self.preference_weights["win_rate_weight"] = min(
                self.preference_weights["win_rate_weight"] + lr, 0.6)

        # 정규화
        total = sum(self.preference_weights.values())
        for k in self.preference_weights:
            self.preference_weights[k] /= total

    def _update_adoption_rate(self, record: SelectionRecord):
        """AI 추천 채택률 업데이트 (HITL 신뢰도 지표)"""
        # 균형 전략(Option 2)이 AI 추천으로 가정
        adopted = record.selected_option_id in ["option_2", "option_3"]
        n = len(self.records)
        self.adoption_rate = (self.adoption_rate * (n - 1) + float(adopted)) / n

    def get_preference_vector(self) -> np.ndarray:
        """PPO 상태에 포함할 선호도 벡터 (5차원)"""
        total_selections = max(sum(self.type_frequency.values()), 1)
        dominant_type_idx = np.argmax(list(self.type_frequency.values()))
        
        return np.array([
            self.preference_weights["force_size_weight"],
            self.preference_weights["win_rate_weight"],
            self.preference_weights["casualty_weight"],
            dominant_type_idx / len(StrategyType),
            self.adoption_rate,
        ], dtype=np.float32)

    def get_personalized_ranking(self, options: List[StrategyOption]) -> List[StrategyOption]:
        """지휘관 선호도 기반 전략 재정렬"""
        w = self.preference_weights
        
        def personal_score(opt: StrategyOption) -> float:
            force_score   = (1 - opt.force_size / 1000) * w["force_size_weight"]
            win_score     = opt.win_probability          * w["win_rate_weight"]
            casualty_score = (1 - opt.expected_casualties / 200) * w["casualty_weight"]
            return force_score + win_score + casualty_score

        return sorted(options, key=personal_score, reverse=True)

    def get_insight_report(self) -> str:
        """지휘관 선호도 분석 리포트"""
        if not self.records:
            return "선택 기록이 없습니다."

        dominant = max(self.type_frequency, key=self.type_frequency.get)
        dominant_count = self.type_frequency[dominant]
        total = len(self.records)

        lines = [
            f"📊 지휘관 선호도 분석 ({self.commander_id})",
            f"   총 선택 횟수: {total}회",
            f"   가장 선호하는 전략: {dominant} ({dominant_count}/{total}회)",
            f"   AI 추천 채택률: {self.adoption_rate:.1%}",
            f"   병력 절감 선호: {self.preference_weights['force_size_weight']:.1%}",
            f"   승률 우선 선호: {self.preference_weights['win_rate_weight']:.1%}",
            f"   사상자 최소 선호: {self.preference_weights['casualty_weight']:.1%}",
        ]
        return "\n".join(lines)

    def save(self, path: str):
        data = {
            "commander_id": self.commander_id,
            "preference_weights": self.preference_weights,
            "type_frequency": self.type_frequency,
            "adoption_rate": self.adoption_rate,
            "n_records": len(self.records)
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.preference_weights = data["preference_weights"]
        self.type_frequency = data["type_frequency"]
        self.adoption_rate = data["adoption_rate"]


if __name__ == "__main__":
    print("=== Commander Preference Learner Test ===")
    learner = CommanderPreferenceLearner("commander_alpha")

    # 시뮬레이션된 선택 기록
    for i, strategy_type in enumerate([
        StrategyType.BALANCED, StrategyType.MIN_CASUALTY,
        StrategyType.BALANCED, StrategyType.MIN_FORCE,
        StrategyType.BALANCED
    ]):
        record = SelectionRecord(
            scenario_id=i,
            options_presented=["option_1", "option_2", "option_3"],
            selected_option_id="option_2",
            selected_type=strategy_type.value,
            force_size=800,
            win_probability=0.81,
            expected_casualties=6
        )
        learner.record_selection(record)

    print(learner.get_insight_report())
    pref_vec = learner.get_preference_vector()
    print(f"\n선호도 벡터: {pref_vec}")
    print("✅ 선호도 학습 테스트 완료!")
