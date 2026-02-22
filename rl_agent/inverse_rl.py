"""
inverse_rl.py — B. 역강화학습 (Maximum Entropy IRL)
전투 교범의 '좋은 전투' 사례 시퀀스로부터 보상 함수를 자동 추출
학술 연구용 합성 데이터
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── 데이터 구조 ──────────────────────────────

@dataclass
class Demonstration:
    """전문가 (교리) 데모 에피소드"""
    episode_id: str
    state_sequence:  List[np.ndarray]   # [T, state_dim]
    action_sequence: List[int]          # [T]
    outcome: str                        # "win" | "loss" | "draw"
    doctrine_label: str                 # 어떤 교리 전술인지

    @property
    def length(self) -> int:
        return len(self.action_sequence)


@dataclass
class LearnedReward:
    """학습된 보상 함수 파라미터"""
    weights: np.ndarray      # 피처별 가중치 [feature_dim]
    feature_names: List[str]
    training_loss: List[float]
    n_demos: int

    def compute(self, state: np.ndarray) -> float:
        """상태 → 스칼라 보상"""
        features = self._extract_features(state)
        return float(np.dot(self.weights, features))

    def _extract_features(self, state: np.ndarray) -> np.ndarray:
        """상태 벡터에서 피처 추출 (가중치 차원 맞춤)"""
        if len(state) >= len(self.weights):
            return state[:len(self.weights)]
        # 패딩
        padded = np.zeros(len(self.weights))
        padded[:len(state)] = state
        return padded

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """가중치 상위 피처"""
        idx = np.argsort(np.abs(self.weights))[::-1][:n]
        return [(self.feature_names[i], float(self.weights[i])) for i in idx]

    def summary(self) -> str:
        lines = ["📊 학습된 보상 함수 (IRL 결과)"]
        lines.append(f"  데모 수: {self.n_demos}개 | 피처 수: {len(self.weights)}개")
        lines.append("  [상위 5개 피처]")
        for name, w in self.top_features(5):
            bar = "+" * int(abs(w) * 10) if w > 0 else "-" * int(abs(w) * 10)
            lines.append(f"    {name:30s}: {w:+.3f}  [{bar}]")
        lines.append(f"  최종 학습 손실: {self.training_loss[-1]:.4f}")
        return "\n".join(lines)


# ── 피처 추출기 ──────────────────────────────

FEATURE_NAMES = [
    # 전투력 피처
    "blue_force_ratio",         # Blue/전체 전투력 비율
    "casualty_efficiency",      # Red 사상 / Blue 사상
    "force_size_penalty",       # 병력 규모 (작을수록 절감)
    # 교리 피처
    "mass_score",               # 전투력 집중도
    "offensive_score",          # 공세 지수
    "security_score",           # 방호 수준
    "maneuver_score",           # 기동 활용도
    # 자원 피처
    "ammo_level",               # 탄약 수준
    "fuel_level",               # 연료 수준
    # 시간 피처
    "time_pressure",            # 남은 시간 비율
    # 기동 피처
    "flanking_bonus",           # 측면 기동 성공
    "envelopment_bonus",        # 포위 달성
]

STATE_DIM = len(FEATURE_NAMES)


def extract_features(state_dict: Dict) -> np.ndarray:
    """전투 상태 딕셔너리 → 피처 벡터"""
    f = np.zeros(STATE_DIM, dtype=np.float32)
    f[0]  = float(state_dict.get("blue_force_ratio", 0.5))
    f[1]  = float(state_dict.get("casualty_efficiency", 1.0))
    f[2]  = float(state_dict.get("force_size_penalty", 0.5))
    f[3]  = float(state_dict.get("mass_score", 0.5))
    f[4]  = float(state_dict.get("offensive_score", 0.5))
    f[5]  = float(state_dict.get("security_score", 0.5))
    f[6]  = float(state_dict.get("maneuver_score", 0.5))
    f[7]  = float(state_dict.get("ammo_level", 0.8))
    f[8]  = float(state_dict.get("fuel_level", 0.8))
    f[9]  = float(state_dict.get("time_pressure", 0.5))
    f[10] = float(state_dict.get("flanking_bonus", 0.0))
    f[11] = float(state_dict.get("envelopment_bonus", 0.0))
    return np.clip(f, -2.0, 2.0)


# ── 합성 데모 생성기 ─────────────────────────

class DemonstrationGenerator:
    """
    합성 전문가 데모 생성 (실제 교범 데이터 대용)
    교리 전술 유형별로 다른 행동 패턴 생성
    """

    DOCTRINE_TACTICS = {
        "mass_attack": {
            "description": "전투력 집중 공격",
            "preferred_actions": [1, 3, 3, 3, 4],   # 전진+교전 반복
            "state_bias": {"mass_score": 0.8, "offensive_score": 0.9, "force_size_penalty": 0.3},
        },
        "flanking_maneuver": {
            "description": "측면 기동 공격",
            "preferred_actions": [1, 5, 5, 3, 3],   # 전진+측면+교전
            "state_bias": {"maneuver_score": 0.9, "flanking_bonus": 0.8, "security_score": 0.6},
        },
        "economy_of_force": {
            "description": "전력 절약 방어",
            "preferred_actions": [7, 0, 4, 6, 7],   # 방어진지+대기+지원+보급
            "state_bias": {"security_score": 0.9, "force_size_penalty": 0.7, "ammo_level": 0.9},
        },
        "envelopment": {
            "description": "포위 섬멸",
            "preferred_actions": [1, 5, 5, 3, 3],   # 측면 양방향
            "state_bias": {"envelopment_bonus": 0.9, "maneuver_score": 0.8, "mass_score": 0.7},
        },
        "delay_action": {
            "description": "지연 방어",
            "preferred_actions": [7, 2, 0, 6, 7],   # 방어+후퇴+보급
            "state_bias": {"security_score": 0.8, "time_pressure": 0.7, "fuel_level": 0.8},
        },
    }

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def generate(self, n_per_tactic: int = 20, episode_len: int = 15) -> List[Demonstration]:
        demos = []
        for tactic_name, tactic in self.DOCTRINE_TACTICS.items():
            for ep in range(n_per_tactic):
                states, actions = [], []
                for t in range(episode_len):
                    # 상태: 교리 편향 + 노이즈
                    state_dict = {k: self.rng.uniform(0.3, 0.7) for k in FEATURE_NAMES}
                    for k, v in tactic["state_bias"].items():
                        state_dict[k] = float(np.clip(v + self.rng.normal(0, 0.1), 0, 1))
                    state_dict["time_pressure"] = 1.0 - t / episode_len
                    states.append(extract_features(state_dict))

                    # 행동: 선호 행동에서 확률적 선택
                    preferred = tactic["preferred_actions"]
                    if self.rng.random() < 0.75:
                        action = preferred[t % len(preferred)]
                    else:
                        action = int(self.rng.randint(0, 8))
                    actions.append(action)

                outcome = "win" if tactic_name != "delay_action" else "draw"
                demos.append(Demonstration(
                    episode_id=f"{tactic_name}_{ep:03d}",
                    state_sequence=states,
                    action_sequence=actions,
                    outcome=outcome,
                    doctrine_label=tactic_name,
                ))
        return demos


# ── Max-Entropy IRL ───────────────────────────

class MaxEntropyIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning
    (Ziebart et al., 2008)

    핵심 아이디어:
    - 전문가 데모의 피처 기댓값을 매칭하는 보상 함수 학습
    - 최대 엔트로피 원칙: 불필요한 가정 최소화
    - 학습된 보상 함수로 PPO 에이전트의 보상 대체 가능

    수식:
    P(τ) ∝ exp(R(τ))   where R(τ) = Σ w · φ(s_t)
    ∇w = E_demo[φ] - E_policy[φ]   (피처 기댓값 차이)
    """

    def __init__(
        self,
        feature_dim: int = STATE_DIM,
        lr: float = 0.01,
        n_iter: int = 200,
        reg_lambda: float = 0.01,
        seed: int = 0
    ):
        self.feature_dim = feature_dim
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.rng = np.random.RandomState(seed)
        # 보상 가중치 초기화
        self.weights = self.rng.normal(0, 0.1, feature_dim).astype(np.float32)

    def _compute_demo_feature_expectation(
        self, demos: List[Demonstration]
    ) -> np.ndarray:
        """데모 피처 기댓값 μ_D = E_demo[φ(s)]"""
        all_features = []
        for demo in demos:
            for state in demo.state_sequence:
                feats = state[:self.feature_dim]
                if len(feats) < self.feature_dim:
                    feats = np.pad(feats, (0, self.feature_dim - len(feats)))
                all_features.append(feats)
        return np.mean(all_features, axis=0).astype(np.float32)

    def _compute_policy_feature_expectation(
        self, demos: List[Demonstration], gamma: float = 0.99
    ) -> np.ndarray:
        """
        현재 보상 함수 하에서의 정책 피처 기댓값 (근사치)
        실제 구현: Value Iteration + 상태 방문 빈도
        단순화: 소프트맥스 정책으로 근사
        """
        all_weighted = []
        for demo in demos:
            for t, state in enumerate(demo.state_sequence):
                feats = state[:self.feature_dim]
                if len(feats) < self.feature_dim:
                    feats = np.pad(feats, (0, self.feature_dim - len(feats)))
                # 현재 가중치로 상태 가치 계산
                reward = float(np.dot(self.weights, feats))
                # 시간 할인
                weight = (gamma ** t) * np.exp(reward * 0.1)
                all_weighted.append(feats * weight)

        if not all_weighted:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # 정규화
        total = sum(np.exp(float(np.dot(self.weights, s[:self.feature_dim])) * 0.1)
                    for demo in demos for s in demo.state_sequence)
        return (np.sum(all_weighted, axis=0) / (total + 1e-8)).astype(np.float32)

    def train(
        self,
        demos: List[Demonstration],
        verbose: bool = False
    ) -> LearnedReward:
        """IRL 학습 루프"""
        # 데모 피처 기댓값 (고정)
        mu_demo = self._compute_demo_feature_expectation(demos)

        losses = []
        lr = self.lr

        for iteration in range(self.n_iter):
            # 현재 정책 피처 기댓값
            mu_policy = self._compute_policy_feature_expectation(demos)

            # 그래디언트: 데모 - 정책 기댓값 (+ L2 정규화)
            grad = mu_demo - mu_policy - self.reg_lambda * self.weights

            # 그래디언트 어센트 (보상 최대화)
            self.weights += lr * grad

            # 손실: 피처 기댓값 차이의 L2 노름
            loss = float(np.linalg.norm(mu_demo - mu_policy))
            losses.append(loss)

            # 학습률 감소
            if (iteration + 1) % 50 == 0:
                lr *= 0.8

            if verbose and (iteration + 1) % 50 == 0:
                print(f"    Iter {iteration+1:3d}: loss={loss:.4f}  "
                      f"|w|={np.linalg.norm(self.weights):.3f}")

        return LearnedReward(
            weights=self.weights.copy(),
            feature_names=FEATURE_NAMES[:self.feature_dim],
            training_loss=losses,
            n_demos=len(demos),
        )

    def compute_reward(self, state: np.ndarray) -> float:
        """학습된 보상 함수 적용"""
        feats = state[:self.feature_dim]
        if len(feats) < self.feature_dim:
            feats = np.pad(feats, (0, self.feature_dim - len(feats)))
        return float(np.dot(self.weights, feats))


# ── 보상 함수 분석기 ─────────────────────────

class RewardFunctionAnalyzer:
    """
    학습된 보상 함수 해석 및 비교
    IRL 결과의 의미를 군사 교리 언어로 설명
    """

    FEATURE_DOCTRINE_MAP = {
        "blue_force_ratio":     ("전투력 비율",     "MASS, OFFENSIVE"),
        "casualty_efficiency":  ("교전 효율",        "ECONOMY, OFFENSIVE"),
        "force_size_penalty":   ("병력 절감",        "ECONOMY"),
        "mass_score":           ("전투력 집중",       "MASS"),
        "offensive_score":      ("공세 지수",        "OFFENSIVE"),
        "security_score":       ("방호 수준",        "SECURITY"),
        "maneuver_score":       ("기동 활용",        "MANEUVER"),
        "ammo_level":           ("탄약 수준",        "SECURITY"),
        "fuel_level":           ("연료 수준",        "SECURITY, ECONOMY"),
        "time_pressure":        ("시간 압박",        "SIMPLICITY"),
        "flanking_bonus":       ("측면 기동",        "SURPRISE, MANEUVER"),
        "envelopment_bonus":    ("포위 달성",        "MASS, MANEUVER"),
    }

    def interpret(self, reward: LearnedReward) -> str:
        """보상 함수를 교리 언어로 해석"""
        lines = ["🔍 IRL 학습 결과 해석 (교리 관점)"]

        # 상위 3개 양/음 피처
        pos_idx = np.argsort(reward.weights)[::-1][:3]
        neg_idx = np.argsort(reward.weights)[:3]

        lines.append("\n  [AI가 중요하게 학습한 가치 (양의 가중치)]")
        for i in pos_idx:
            name = reward.feature_names[i]
            w = reward.weights[i]
            kr, doctrine = self.FEATURE_DOCTRINE_MAP.get(name, (name, "N/A"))
            lines.append(f"    +{w:.3f}  {kr} → [{doctrine}] 원칙")

        lines.append("\n  [AI가 회피하도록 학습한 요소 (음의 가중치)]")
        for i in neg_idx:
            if reward.weights[i] < 0:
                name = reward.feature_names[i]
                w = reward.weights[i]
                kr, doctrine = self.FEATURE_DOCTRINE_MAP.get(name, (name, "N/A"))
                lines.append(f"    {w:.3f}  {kr} → [{doctrine}] 원칙")

        # 수렴 분석
        if len(reward.training_loss) > 10:
            initial = reward.training_loss[0]
            final   = reward.training_loss[-1]
            lines.append(f"\n  [학습 수렴]  초기 손실={initial:.4f} → 최종={final:.4f}"
                         f"  ({(1-final/initial)*100:.1f}% 개선)")

        return "\n".join(lines)

    def compare_with_handcrafted(
        self,
        learned: LearnedReward,
        handcrafted_weights: Optional[np.ndarray] = None
    ) -> str:
        """학습된 보상 vs 수작업 보상 비교"""
        if handcrafted_weights is None:
            # 기본 수작업 보상: Win, Casualty, ForceSize
            handcrafted_weights = np.zeros(len(learned.weights))
            handcrafted_weights[0] = 1.0   # blue_force_ratio (Win 대용)
            handcrafted_weights[2] = -0.5  # force_size_penalty
            handcrafted_weights[1] = 0.8   # casualty_efficiency

        cosine_sim = float(
            np.dot(learned.weights, handcrafted_weights) /
            (np.linalg.norm(learned.weights) * np.linalg.norm(handcrafted_weights) + 1e-8)
        )
        lines = [
            f"📊 학습된 보상 vs 수작업 보상 비교",
            f"  코사인 유사도: {cosine_sim:.3f}",
            f"  해석: {'유사한 가치 함수 학습됨' if cosine_sim > 0.7 else '다른 가치 함수 — 새로운 인사이트'}",
        ]
        return "\n".join(lines)


# ── P3-4: IRL 보상 로더 ─────────────────────────────────────────────

class IRLRewardLoader:
    """
    P3-4: data/irl_demos_summary.json에서 사전 학습된 IRL 보상 가중치를 로드하고
    실시간 전투 상태에 적용해 IRL 보너스를 계산한다.

    사용 예:
        irl_loader = IRLRewardLoader.from_file("data/irl_demos_summary.json")
        bonus = irl_loader.compute_irl_bonus(kg, step_result, maneuver_result)
    """

    _IRL_BONUS_SCALE = 0.5   # 전체 IRL 보상 배율 (PPO 보상에 더해지는 비율)

    def __init__(self, learned_reward: "LearnedReward"):
        self.learned_reward = learned_reward

    @classmethod
    def from_file(cls, path: str) -> "IRLRewardLoader":
        """
        JSON 파일에서 사전 학습된 IRL 가중치 로드.

        Parameters
        ----------
        path : str
            data/irl_demos_summary.json 경로
        """
        import json, os
        if not os.path.isfile(path):
            raise FileNotFoundError(f"IRL 데이터 파일 없음: {path}")
        with open(path) as f:
            data = json.load(f)

        raw_weights = data.get("learned_reward_weights", {})
        weights_arr = np.array(
            [raw_weights.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float32,
        )
        # 단위 정규화
        norm = np.linalg.norm(weights_arr)
        if norm > 0:
            weights_arr = weights_arr / norm

        learned = LearnedReward(
            weights=weights_arr,
            feature_names=FEATURE_NAMES,
            training_loss=[data.get("final_training_loss", 0.0)],
            n_demos=data.get("n_demos", 0),
        )
        return cls(learned)

    def _build_state_dict(self, kg, step_result, maneuver_result: Optional[Dict] = None) -> Dict:
        """
        CombatKnowledgeGraph + BattleStep + ManeuverResult → 12차원 피처 딕셔너리 변환.
        """
        from ontology.combat_schema import ForceAlignment, UnitStatus

        blue = [u for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED]
        red  = [u for u in kg.units.values()
                if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED]

        blue_cp  = sum(u.combat_power for u in blue)
        red_cp   = sum(u.combat_power for u in red)
        total_cp = blue_cp + red_cp + 1e-8

        blue_cas = step_result.blue_total_casualties
        red_cas  = step_result.red_total_casualties

        avg_ammo = float(np.mean([u.capability.ammo_level for u in blue])) if blue else 0.5
        avg_fuel = float(np.mean([u.capability.fuel_level  for u in blue])) if blue else 0.5

        high_cp = [u for u in blue if u.combat_power > 2.0]
        mass_score = len(high_cp) / max(len(blue), 1)

        n_engageable  = (maneuver_result or {}).get("n_engageable", 0)
        flanking_cnt  = (maneuver_result or {}).get("flanking_units", 0)
        envelop_flag  = float((maneuver_result or {}).get("enveloping", False))

        return {
            "blue_force_ratio":     blue_cp / total_cp,
            "casualty_efficiency":  red_cas  / max(blue_cas, 1),
            "force_size_penalty":   1.0 - len(blue) / max(len(blue) + len(red), 1),
            "mass_score":           mass_score,
            "offensive_score":      float(np.clip(blue_cp / (red_cp + 1e-8) / 2, 0, 1)),
            "security_score":       (avg_ammo + avg_fuel) / 2,
            "maneuver_score":       float(np.clip(n_engageable * 0.1, 0, 1)),
            "ammo_level":           avg_ammo,
            "fuel_level":           avg_fuel,
            "time_pressure":        0.5,   # 스텝 진행도 미연동 (근사치)
            "flanking_bonus":       float(flanking_cnt > 0),
            "envelopment_bonus":    envelop_flag,
        }

    def compute_irl_bonus(self, kg, step_result, maneuver_result: Optional[Dict] = None) -> float:
        """
        현재 전투 상태에서 IRL 보상 보너스를 계산한다.

        Returns
        -------
        float
            IRL 보너스 값 (PPO 보상에 더해짐). 양수 = 교리 준수 행동 강화.
        """
        state_dict = self._build_state_dict(kg, step_result, maneuver_result)
        feature_vec = extract_features(state_dict)
        raw_reward  = self.learned_reward.compute(feature_vec)
        return float(raw_reward * self._IRL_BONUS_SCALE)


if __name__ == "__main__":
    print("="*55)
    print("B. 역강화학습 (Max-Entropy IRL) 테스트")
    print("="*55)

    gen   = DemonstrationGenerator(seed=42)
    demos = gen.generate(n_per_tactic=10, episode_len=12)
    print(f"\n데모 생성: {len(demos)}개 에피소드 "
          f"({len(set(d.doctrine_label for d in demos))}개 전술 유형)")

    irl = MaxEntropyIRL(feature_dim=STATE_DIM, lr=0.02, n_iter=150, seed=42)
    print("\n[IRL 학습 중...]")
    reward = irl.train(demos, verbose=True)
    print(reward.summary())

    analyzer = RewardFunctionAnalyzer()
    print(analyzer.interpret(reward))
    print(analyzer.compare_with_handcrafted(reward))

    # 샘플 상태에 보상 적용
    sample = extract_features({"blue_force_ratio":0.7,"mass_score":0.8,"flanking_bonus":0.6})
    print(f"\n샘플 상태 보상: {irl.compute_reward(sample):.4f}")
    print("✅ IRL 정상 동작!")
