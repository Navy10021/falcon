"""
bandit_preference.py
====================
⑦ 선호도 Bandit 학습
- Thompson Sampling 기반 Contextual Bandit
- 지휘관의 전략 선택 패턴을 확률 분포로 모델링
- Beta 분포 업데이트 (선택/미선택)
- LinUCB: 문맥(전장상황) 의존 선호도

기존 preference_learner.py의 빈도 기반 → 확률 분포 기반 업그레이드
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────
# 전략 문맥 벡터
# ──────────────────────────────────────────────

@dataclass
class StrategyContext:
    """전략 후보의 문맥 특성 벡터"""
    option_id: str
    strategy_type: str
    force_ratio: float       # 투입병력 / 가용병력 [0,1]
    win_probability: float
    casualty_ratio: float
    time_pressure: float     # [0=여유, 1=긴박]
    uncertainty: float
    risk_level: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.force_ratio, self.win_probability,
            self.casualty_ratio, self.time_pressure,
            self.uncertainty, self.risk_level,
        ], dtype=np.float32)


# ──────────────────────────────────────────────
# Beta Bandit 팔
# ──────────────────────────────────────────────

class BetaBanditArm:
    """단일 전략 유형의 Beta 분포 선호도 모델"""

    def __init__(self, strategy_type: str, alpha0: float = 1.0, beta0: float = 1.0):
        self.strategy_type = strategy_type
        self.alpha = alpha0
        self.beta  = beta0

    def sample(self, rng: np.random.RandomState) -> float:
        return float(rng.beta(self.alpha, self.beta))

    def update(self, selected: bool, reward: float = 1.0):
        if selected:
            self.alpha += reward
        else:
            self.beta  += max(0, 1 - reward)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self) -> float:
        a, b = self.alpha, self.beta
        n = a + b
        return float(np.sqrt(a * b / (n**2 * (n + 1))))

    @property
    def confidence(self) -> float:
        return min((self.alpha + self.beta - 2) / 20.0, 1.0)

    def summary(self) -> str:
        return (f"{self.strategy_type:25s}: 선호도={self.mean:.3f}"
                f"±{self.std:.3f}  (α={self.alpha:.1f}, β={self.beta:.1f},"
                f"  신뢰={self.confidence:.2f})")


# ──────────────────────────────────────────────
# LinUCB 팔 (문맥 의존)
# ──────────────────────────────────────────────

class LinUCBArm:
    """Linear UCB: 문맥 벡터 기반 선호도 추정"""

    def __init__(self, strategy_type: str, context_dim: int = 6,
                 alpha: float = 0.5):
        self.strategy_type = strategy_type
        self.alpha = alpha
        self.d = context_dim
        self.A = np.eye(context_dim, dtype=np.float32)
        self.b = np.zeros((context_dim, 1), dtype=np.float32)
        self.n_updates = 0

    def ucb_score(self, context: np.ndarray) -> float:
        x = context.reshape(-1, 1)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean_pred = float(np.dot(x.flatten(), theta.flatten()))
        explore   = float(self.alpha * np.sqrt((x.T @ A_inv @ x).item()))
        return mean_pred + explore

    def update(self, context: np.ndarray, reward: float):
        x = context.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x
        self.n_updates += 1

    @property
    def theta(self) -> np.ndarray:
        return (np.linalg.inv(self.A) @ self.b).flatten()


# ──────────────────────────────────────────────
# 통합 Bandit 선호도 학습기
# ──────────────────────────────────────────────

@dataclass
class SelectionEvent:
    scenario_id: int
    selected_type: str
    all_types: List[str]
    context: StrategyContext
    outcome_reward: float = 1.0


class BanditPreferenceLearner:
    """
    Beta Bandit + LinUCB 결합 선호도 학습기
    - Beta Bandit: 전략 유형별 전반적 선호도
    - LinUCB:      전장 문맥에 따른 선호도 조정
    """

    STRATEGY_TYPES = [
        "minimum_force", "balanced", "maximum_win_rate",
        "minimum_casualty", "minimum_time"
    ]

    def __init__(self, commander_id: str = "default", seed: int = 42):
        self.commander_id = commander_id
        self.rng = np.random.RandomState(seed)
        self.history: List[SelectionEvent] = []
        self.n_selections = 0
        self.exploration_rate = 0.20

        self.beta_arms: Dict[str, BetaBanditArm] = {
            st: BetaBanditArm(st) for st in self.STRATEGY_TYPES
        }
        self.linucb_arms: Dict[str, LinUCBArm] = {
            st: LinUCBArm(st, context_dim=6) for st in self.STRATEGY_TYPES
        }

    def record_selection(self, event: SelectionEvent):
        """베이지안 업데이트"""
        self.history.append(event)
        self.n_selections += 1
        ctx_vec = event.context.to_vector()

        for st in event.all_types:
            selected = (st == event.selected_type)
            reward   = event.outcome_reward if selected else 0.0
            if st in self.beta_arms:
                self.beta_arms[st].update(selected, reward)
            if selected and st in self.linucb_arms:
                self.linucb_arms[st].update(ctx_vec, event.outcome_reward)

        self.exploration_rate = max(0.05, 0.20 - self.n_selections * 0.005)

    def _score_option(self, ctx: StrategyContext, method: str = "combined") -> float:
        st = ctx.strategy_type
        if st not in self.beta_arms:
            return 0.5

        if method == "beta":
            return self.beta_arms[st].sample(self.rng)
        elif method == "linucb":
            return float(np.clip(self.linucb_arms[st].ucb_score(ctx.to_vector()), 0, 2))
        else:
            beta_s  = self.beta_arms[st].sample(self.rng)
            ucb_s   = float(np.clip(self.linucb_arms[st].ucb_score(ctx.to_vector()), 0, 1))
            conf    = self.beta_arms[st].confidence
            return conf * beta_s + (1 - conf) * ucb_s

    def rank_options(
        self, contexts: List[StrategyContext], method: str = "combined"
    ) -> List[Tuple[StrategyContext, float]]:
        scored = [(ctx, self._score_option(ctx, method)) for ctx in contexts]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def recommend(self, contexts: List[StrategyContext]) -> StrategyContext:
        """Thompson Sampling + ε-greedy 추천"""
        if not contexts:
            raise ValueError("contexts 비어있음")
        ranked = self.rank_options(contexts)
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice([c for c, _ in ranked])
        return ranked[0][0]

    def get_preference_profile(self) -> Dict[str, Dict]:
        profile = {}
        for st, arm in self.beta_arms.items():
            profile[st] = {
                "mean_preference": arm.mean,
                "std": arm.std,
                "confidence": arm.confidence,
                "n_selected": int(arm.alpha - 1),
                "context_weights": self.linucb_arms[st].theta.tolist(),
            }
        return profile

    def get_top_preference(self) -> str:
        return max(self.beta_arms, key=lambda k: self.beta_arms[k].mean)

    def adoption_rate(self) -> float:
        if not self.history:
            return 0.0
        top = self.get_top_preference()
        return sum(1 for e in self.history if e.selected_type == top) / len(self.history)

    def get_insight_report(self) -> str:
        if self.n_selections == 0:
            return "데이터 부족"
        top = self.get_top_preference()
        arm = self.beta_arms[top]
        feature_names = ["병력절감", "승률", "사상자최소", "시간절약", "불확실성", "위험감수"]
        theta = self.linucb_arms[top].theta
        top_feat = feature_names[int(np.argmax(np.abs(theta)))]
        return (f"총 {self.n_selections}회 | 최선호={top} "
                f"(선호도={arm.mean:.3f}) | 주결정요인={top_feat} | "
                f"탐색률={self.exploration_rate:.2f}")

    def print_arms(self):
        print("\n[Bandit 팔 상태 (선호도 내림차순)]")
        for st, arm in sorted(self.beta_arms.items(),
                               key=lambda x: x[1].mean, reverse=True):
            bar = "█" * int(arm.mean * 20)
            print(f"  {arm.summary()}")
            print(f"    [{bar:<20s}]")


def simulate_commander_choices(
    learner: BanditPreferenceLearner,
    n_episodes: int = 50,
    true_preference: str = "minimum_casualty",
    seed: int = 0
) -> List[float]:
    """지휘관 행동 시뮬레이션 → 에피소드별 예측 정확도"""
    rng = np.random.RandomState(seed)
    accuracies = []

    for ep in range(n_episodes):
        n_opts = rng.randint(2, 5)
        contexts = []
        for i in range(n_opts):
            st = rng.choice(learner.STRATEGY_TYPES)
            ctx = StrategyContext(
                option_id=f"opt_{i}", strategy_type=st,
                force_ratio=float(rng.uniform(0.4, 1.0)),
                win_probability=float(rng.uniform(0.5, 0.95)),
                casualty_ratio=float(rng.uniform(0.01, 0.15)),
                time_pressure=float(rng.uniform(0.1, 0.9)),
                uncertainty=float(rng.uniform(0.1, 0.7)),
                risk_level=float(rng.uniform(0.1, 0.8)),
            )
            contexts.append(ctx)

        # 지휘관: 70% 확률로 true_preference 선택
        true_opts = [c for c in contexts if c.strategy_type == true_preference]
        if true_opts and rng.random() < 0.70:
            selected = true_opts[0]
        else:
            selected = rng.choice(contexts)

        event = SelectionEvent(
            scenario_id=ep, selected_type=selected.strategy_type,
            all_types=[c.strategy_type for c in contexts],
            context=selected, outcome_reward=float(rng.uniform(0.5, 1.0)),
        )
        learner.record_selection(event)

        ranked = learner.rank_options(contexts)
        top_pred = ranked[0][0].strategy_type if ranked else ""
        accuracies.append(float(top_pred == true_preference))

    return accuracies


if __name__ == "__main__":
    print("=" * 55)
    print("⑦ Bandit 선호도 학습 테스트")
    print("=" * 55)

    learner = BanditPreferenceLearner("commander_A", seed=42)
    TRUE_PREF = "minimum_casualty"

    print(f"\n[50 에피소드 시뮬레이션 — 진짜 선호: {TRUE_PREF}]")
    accuracies = simulate_commander_choices(learner, 50, TRUE_PREF, seed=42)

    for start in range(0, 50, 10):
        chunk = accuracies[start:start+10]
        acc = np.mean(chunk)
        bar = "█" * int(acc * 20)
        print(f"  EP {start+1:2d}~{start+10:2d}: 정확도={acc:.2f} [{bar:<20s}]")

    print(f"\n최종 정확도 (마지막 10회): {np.mean(accuracies[-10:]):.2f}")
    print(f"채택률: {learner.adoption_rate():.2f}")
    print(f"인사이트: {learner.get_insight_report()}")
    learner.print_arms()
    print("\n✅ Bandit 선호도 학습 정상 동작!")
