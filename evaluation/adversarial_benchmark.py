"""
adversarial_benchmark.py
========================
적대적 강건성 벤치마크

Blue 에이전트가 다양한 적대적 조건에서도
얼마나 안정적으로 작동하는지 정량 평가한다.

평가 축:
  1. Observation Perturbation  — Red의 기만 공격에 대한 내성
  2. Domain Shift              — 시뮬레이션 파라미터 변화 내성
  3. Opponent Diversity        — 다양한 Red 전략 대응 능력
  4. Robustness Score          — 종합 강건성 점수 (0–1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rl_agent.red_agent import RedAgent, RedActionSpace
from simulator.adversarial_scenario import DomainRandomizer, DomainSample


# ──────────────────────────────────────────────
# 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class PerturbationResult:
    """단일 perturbation 조건 평가 결과"""
    perturbation_type: str
    epsilon: float
    n_episodes: int
    win_rate: float
    avg_reward: float
    action_consistency: float   # 원본 행동과 perturbed 행동 일치율


@dataclass
class DomainShiftResult:
    """도메인 변화 평가 결과"""
    domain_sample: DomainSample
    win_rate: float
    avg_reward: float
    force_reduction_rate: float


@dataclass
class AdversarialBenchmarkReport:
    """벤치마크 종합 보고서"""
    baseline_win_rate: float
    perturbation_results: List[PerturbationResult] = field(default_factory=list)
    domain_shift_results: List[DomainShiftResult] = field(default_factory=list)
    opponent_diversity_score: float = 0.0
    robustness_score: float = 0.0
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Adversarial Benchmark Report",
            "=" * 60,
            f"Baseline Win Rate        : {self.baseline_win_rate:.3f}",
            f"Robustness Score         : {self.robustness_score:.3f}",
            f"Opponent Diversity Score : {self.opponent_diversity_score:.3f}",
            f"Elapsed                  : {self.elapsed_sec:.1f}s",
            "",
            "[Perturbation Tests]",
        ]
        for r in self.perturbation_results:
            drop = self.baseline_win_rate - r.win_rate
            lines.append(
                f"  {r.perturbation_type:10s} ε={r.epsilon:.2f}"
                f"  win={r.win_rate:.3f} (drop={drop:+.3f})"
                f"  consistency={r.action_consistency:.3f}"
            )
        if self.domain_shift_results:
            lines.append("")
            lines.append("[Domain Shift Tests]")
            for r in self.domain_shift_results:
                lines.append(
                    f"  force_scale={r.domain_sample.force_scale:.2f}"
                    f"  cp_scale={r.domain_sample.combat_power_scale:.2f}"
                    f"  win={r.win_rate:.3f}"
                    f"  frr={r.force_reduction_rate:.3f}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 벤치마크 실행기
# ──────────────────────────────────────────────

class AdversarialBenchmark:
    """
    Blue 에이전트 적대적 강건성 평가기.

    실제 전투 시뮬레이션 루프 없이도 동작하도록
    Blue 에이전트의 `select_action` 인터페이스만 의존한다.

    Parameters
    ----------
    blue_agent : any
        `select_action(state, deterministic=True)` 메서드를 가진 에이전트.
    red_agent : RedAgent
        perturbation 생성에 사용할 Red 에이전트.
    state_dim : int
        상태벡터 차원 (기본 128).
    seed : int
        난수 시드.
    """

    PERTURBATION_CONFIGS: List[Tuple[str, float]] = [
        ("gaussian",  0.05),
        ("gaussian",  0.10),
        ("gaussian",  0.20),
        ("sign",      0.05),
        ("sign",      0.10),
        ("uniform",   0.10),
        ("zero_out",  0.10),
        ("zero_out",  0.20),
    ]

    def __init__(
        self,
        blue_agent,
        red_agent: Optional[RedAgent] = None,
        state_dim: int = 128,
        seed: int = 42,
    ):
        self.blue_agent = blue_agent
        self.red_agent = red_agent or RedAgent()
        self.state_dim = state_dim
        self.rng = np.random.RandomState(seed)
        self.domain_randomizer = DomainRandomizer(seed=seed)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _mock_states(self, n: int) -> np.ndarray:
        """재현 가능한 의사 상태벡터 n개 생성"""
        rng = np.random.RandomState(0)
        return rng.randn(n, self.state_dim).astype(np.float32)

    def _mock_win(self, action: int, rng: np.random.RandomState) -> Tuple[bool, float]:
        """
        행동 → (승리 여부, 보상) 결정론적 근사.
        실제 환경 없이 행동만으로 간단히 평가 (벤치마크 전용).
        """
        # 행동 0(Reinforce)·1(Advance)·5(Flank) 공격적 = 승률 높음
        win_prob = {0: 0.62, 1: 0.68, 2: 0.45, 3: 0.55, 4: 0.58, 5: 0.65}.get(action, 0.5)
        win = rng.random() < win_prob
        reward = 5.0 if win else -3.0
        return win, reward

    # ------------------------------------------------------------------
    # 1. Perturbation 테스트
    # ------------------------------------------------------------------

    def evaluate_perturbation(
        self,
        n_episodes: int = 200,
        configs: Optional[List[Tuple[str, float]]] = None,
    ) -> List[PerturbationResult]:
        """
        다양한 타입·강도의 perturbation 하에서 Blue 에이전트 평가.
        """
        configs = configs or self.PERTURBATION_CONFIGS
        states = self._mock_states(n_episodes)
        results: List[PerturbationResult] = []

        # 원본 행동 기록 (baseline)
        baseline_actions = []
        for s in states:
            action, _, _ = self.blue_agent.select_action(s, deterministic=True)
            baseline_actions.append(action)

        for p_type, epsilon in configs:
            ep_rng = np.random.RandomState(int(epsilon * 1000) + hash(p_type) % 10000)
            wins, rewards, same_action = [], [], []

            for i, s in enumerate(states):
                perturbed = self.red_agent.compute_observation_perturbation(
                    s, perturbation_type=p_type, epsilon=epsilon,
                    rng=np.random.RandomState(i)
                )
                action, _, _ = self.blue_agent.select_action(perturbed, deterministic=True)
                win, reward = self._mock_win(action, ep_rng)
                wins.append(win)
                rewards.append(reward)
                same_action.append(int(action == baseline_actions[i]))

            results.append(PerturbationResult(
                perturbation_type=p_type,
                epsilon=epsilon,
                n_episodes=n_episodes,
                win_rate=float(np.mean(wins)),
                avg_reward=float(np.mean(rewards)),
                action_consistency=float(np.mean(same_action)),
            ))

        return results

    # ------------------------------------------------------------------
    # 2. Domain Shift 테스트
    # ------------------------------------------------------------------

    def evaluate_domain_shift(
        self,
        n_samples: int = 10,
        n_episodes_per_sample: int = 50,
    ) -> List[DomainShiftResult]:
        """
        서로 다른 도메인 파라미터 조건에서 Blue 에이전트 평가.
        """
        samples = self.domain_randomizer.sample_batch(n_samples)
        states = self._mock_states(n_episodes_per_sample)
        results: List[DomainShiftResult] = []

        for sample in samples:
            ep_rng = np.random.RandomState(sample.seed % (2 ** 31))

            # 도메인 파라미터로 상태 스케일 (간소화: force_scale 적용)
            scaled_states = states * sample.force_scale * sample.combat_power_scale

            wins, rewards, frrs = [], [], []
            for s in scaled_states:
                action, _, _ = self.blue_agent.select_action(s, deterministic=True)
                win, reward = self._mock_win(action, ep_rng)
                # 병력 감소율: supply 방해 비례 패널티
                frr = self.rng.uniform(0.1, 0.5) * (1 + sample.supply_disruption_prob)
                wins.append(win)
                rewards.append(reward)
                frrs.append(frr)

            results.append(DomainShiftResult(
                domain_sample=sample,
                win_rate=float(np.mean(wins)),
                avg_reward=float(np.mean(rewards)),
                force_reduction_rate=float(np.mean(frrs)),
            ))

        return results

    # ------------------------------------------------------------------
    # 3. 종합 강건성 점수
    # ------------------------------------------------------------------

    @staticmethod
    def compute_robustness_score(
        baseline_win_rate: float,
        perturbation_results: List[PerturbationResult],
        domain_shift_results: List[DomainShiftResult],
    ) -> float:
        """
        강건성 점수 [0, 1] 계산.

        공식:
          score = w1 * (1 - perturbation_drop) + w2 * domain_win_mean + w3 * consistency_mean

        w1=0.4, w2=0.4, w3=0.2
        """
        if not perturbation_results:
            return 0.0

        drops = [max(0.0, baseline_win_rate - r.win_rate) for r in perturbation_results]
        avg_drop = float(np.mean(drops))
        perturbation_robustness = max(0.0, 1.0 - avg_drop / max(baseline_win_rate, 1e-6))

        consistencies = [r.action_consistency for r in perturbation_results]
        avg_consistency = float(np.mean(consistencies))

        if domain_shift_results:
            domain_win_mean = float(np.mean([r.win_rate for r in domain_shift_results]))
        else:
            domain_win_mean = baseline_win_rate

        score = 0.4 * perturbation_robustness + 0.4 * domain_win_mean + 0.2 * avg_consistency
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 4. 전체 벤치마크 실행
    # ------------------------------------------------------------------

    def run(
        self,
        n_perturbation_episodes: int = 200,
        n_domain_samples: int = 10,
        n_domain_episodes: int = 50,
        baseline_episodes: int = 200,
    ) -> AdversarialBenchmarkReport:
        """
        전체 벤치마크를 순서대로 실행하고 보고서를 반환.
        """
        t0 = time.time()

        # 베이스라인 평가
        states = self._mock_states(baseline_episodes)
        base_wins = []
        base_rng = np.random.RandomState(999)
        for s in states:
            action, _, _ = self.blue_agent.select_action(s, deterministic=True)
            win, _ = self._mock_win(action, base_rng)
            base_wins.append(win)
        baseline_win_rate = float(np.mean(base_wins))

        # Perturbation 평가
        perturbation_results = self.evaluate_perturbation(
            n_episodes=n_perturbation_episodes
        )

        # Domain Shift 평가
        domain_shift_results = self.evaluate_domain_shift(
            n_samples=n_domain_samples,
            n_episodes_per_sample=n_domain_episodes,
        )

        # 다양한 상대 다양성 점수 (ELO 분산 기반 근사)
        action_dist = np.zeros(6)
        for s in states:
            action, _, _ = self.blue_agent.select_action(s, deterministic=False)
            action_dist[action] += 1
        action_dist /= action_dist.sum() + 1e-8
        opponent_diversity_score = float(-np.sum(action_dist * np.log(action_dist + 1e-8)) / np.log(6))

        # 강건성 점수
        robustness_score = self.compute_robustness_score(
            baseline_win_rate, perturbation_results, domain_shift_results
        )

        return AdversarialBenchmarkReport(
            baseline_win_rate=baseline_win_rate,
            perturbation_results=perturbation_results,
            domain_shift_results=domain_shift_results,
            opponent_diversity_score=opponent_diversity_score,
            robustness_score=robustness_score,
            elapsed_sec=time.time() - t0,
        )


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    _project_root = os.path.dirname(os.path.dirname(__file__))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from rl_agent.blue_agent import ActorCritic, BlueActionSpace
    from rl_agent.red_agent import RedAgent

    class _DummyBlue:
        """테스트용 더미 Blue 에이전트"""
        def __init__(self):
            self.net = ActorCritic(n_actions=BlueActionSpace.N_ACTIONS)
        def select_action(self, state, deterministic=False):
            import torch
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = self.net(s)
                if deterministic:
                    action = int(logits.argmax().item())
                    lp = float(torch.log_softmax(logits, -1).max().item())
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())
                    lp = float(dist.log_prob(torch.tensor(action)).item())
            return action, lp, float(value.item())

    print("=== Adversarial Benchmark 테스트 ===")
    blue = _DummyBlue()
    red = RedAgent()
    bench = AdversarialBenchmark(blue_agent=blue, red_agent=red, seed=42)

    report = bench.run(
        n_perturbation_episodes=100,
        n_domain_samples=5,
        n_domain_episodes=30,
        baseline_episodes=100,
    )
    print(report.summary())
    print("✅ adversarial_benchmark.py 정상 동작!")
