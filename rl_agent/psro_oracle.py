"""
psro_oracle.py
==============
Policy Space Response Oracles (PSRO) + α-Rank 통합 구현

PSRO 알고리즘 (Lanctot et al., NeurIPS 2017):
  반복마다 현재 전략 풀의 payoff 행렬에서 내쉬 혼합 전략을 계산하고
  그에 대한 Best Response를 전략 풀에 추가하여 수렴.

α-Rank (Omidshafiei et al., Science Advances 2019):
  내쉬 균형이 없거나 순환이 발생하는 게임에서도
  진화 게임 이론의 Markov Chain 정적 분포로 전략 순위 산출.

LeagueManager와 연동:
  - PSROOracle이 LeagueAgent 스냅샷을 전략 원소로 관리
  - 대전 스케줄러가 α-Rank 분포 기반 상대 샘플링으로 교체 가능
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rl_agent.league_selfplay import LeagueAgent, AgentRole


# ──────────────────────────────────────────────
# Payoff 행렬 관리
# ──────────────────────────────────────────────

class PayoffMatrix:
    """
    n×n 전략 payoff 행렬.

    M[i, j] = 전략 i가 전략 j를 상대로 한 평균 승률 (0~1).
    새 전략 추가 시 행·열 확장, 기존 값 보존.

    사용::
        pm = PayoffMatrix()
        pm.add_strategy("s0")
        pm.add_strategy("s1")
        pm.update("s0", "s1", result=0.7)
        print(pm.get("s0", "s1"))  # 0.7
    """

    def __init__(self):
        self._ids: List[str] = []
        self._idx: Dict[str, int] = {}
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._counts: np.ndarray = np.empty((0, 0), dtype=np.int32)

    # ------------------------------------------------------------------
    def add_strategy(self, strategy_id: str, default_payoff: float = 0.5):
        """새 전략을 행렬에 추가 (행·열 확장)."""
        if strategy_id in self._idx:
            return
        n = len(self._ids)
        self._ids.append(strategy_id)
        self._idx[strategy_id] = n

        # 행렬 크기 확장
        new_matrix = np.full((n + 1, n + 1), default_payoff, dtype=np.float64)
        new_counts  = np.zeros((n + 1, n + 1), dtype=np.int32)
        if n > 0:
            new_matrix[:n, :n] = self._matrix
            new_counts[:n, :n] = self._counts
        np.fill_diagonal(new_matrix, 0.5)
        self._matrix = new_matrix
        self._counts = new_counts

    def update(self, id_a: str, id_b: str, result: float):
        """
        id_a vs id_b 경기 결과 반영 (온라인 평균).
        result: 1=a 승, 0=a 패, 0.5=무
        """
        i, j = self._idx[id_a], self._idx[id_b]
        n_ij = self._counts[i, j] + 1
        self._matrix[i, j] = (self._matrix[i, j] * self._counts[i, j] + result) / n_ij
        self._matrix[j, i] = 1.0 - self._matrix[i, j]
        self._counts[i, j] += 1
        self._counts[j, i] += 1

    def get(self, id_a: str, id_b: str) -> float:
        return float(self._matrix[self._idx[id_a], self._idx[id_b]])

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    @property
    def strategy_ids(self) -> List[str]:
        return list(self._ids)

    @property
    def n(self) -> int:
        return len(self._ids)


# ──────────────────────────────────────────────
# α-Rank 계산기
# ──────────────────────────────────────────────

class AlphaRankCalculator:
    """
    payoff 행렬로부터 α-Rank 정적 분포 계산.

    두 가지 방법:
      - "power"    : 거듭제곱법 (Power Iteration) — 빠르고 안정적
      - "eigen"    : 고유벡터 — 소규모 행렬에 정확
    """

    def __init__(self, alpha: float = 10.0, method: str = "power"):
        """
        Parameters
        ----------
        alpha : float
            선택 압력. 클수록 강한 전략 선호. 일반적으로 10~100.
        method : str
            "power" | "eigen"
        """
        self.alpha = alpha
        self.method = method

    def compute(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """
        payoff 행렬 M (n×n) → 정적 분포 π (n,).

        전이 행렬 T[i,j] = σ(α·(M[j,i] - M[i,j])) for i≠j.
        """
        n = len(payoff_matrix)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # 전이 확률 행렬 구성
        T = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                advantage = payoff_matrix[j, i] - payoff_matrix[i, j]
                # Fermi 선택 함수 (시그모이드)
                T[i, j] = 1.0 / (1.0 + np.exp(-self.alpha * advantage))
            # 대각선 = 자기 자신으로의 전이
            row_off = T[i].sum()
            T[i, i] = max(0.0, (n - 1) - row_off)
            T[i] /= (n - 1)   # 정규화: 각 행의 합 = 1

        if self.method == "power":
            return self._power_iteration(T)
        else:
            return self._eigen(T)

    @staticmethod
    def _power_iteration(T: np.ndarray, max_iter: int = 2000, tol: float = 1e-10) -> np.ndarray:
        n = len(T)
        pi = np.ones(n, dtype=np.float64) / n
        for _ in range(max_iter):
            pi_new = pi @ T
            pi_new /= pi_new.sum() + 1e-15
            if np.max(np.abs(pi_new - pi)) < tol:
                break
            pi = pi_new
        return pi_new / pi_new.sum()

    @staticmethod
    def _eigen(T: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eig(T.T)
        # 최소 절댓값 고유값에 해당하는 고유벡터 = 정적 분포
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = np.abs(stat)
        return stat / stat.sum()


# ──────────────────────────────────────────────
# PSRO Oracle
# ──────────────────────────────────────────────

@dataclass
class PSROConfig:
    """PSRO 하이퍼파라미터"""
    alpha: float = 10.0          # α-Rank 선택 압력
    alpha_rank_method: str = "power"  # "power" | "eigen"
    max_strategies: int = 50     # 최대 전략 풀 크기
    min_games_per_pair: int = 3  # payoff 신뢰 최소 경기 수
    mixture_type: str = "alpha_rank"  # "alpha_rank" | "nash" | "uniform"


class PSROOracle:
    """
    Policy Space Response Oracles (PSRO) — 전략 풀 + payoff 행렬 관리.

    LeagueManager의 PFSP 대전 스케줄러와 교체하거나 병행하여 사용.

    주요 인터페이스::

        oracle = PSROOracle()
        oracle.add_strategy(agent)          # 새 전략 등록
        oracle.record_result(a, b, 0.7)     # 경기 결과 기록
        ranking = oracle.alpha_rank()       # {agent_id: score}
        opponent = oracle.sample_opponent(focal_id)
    """

    def __init__(self, config: Optional[PSROConfig] = None, seed: int = 0):
        self.config = config or PSROConfig()
        self.payoff = PayoffMatrix()
        self.strategies: Dict[str, LeagueAgent] = {}
        self.alpha_calc = AlphaRankCalculator(
            alpha=self.config.alpha,
            method=self.config.alpha_rank_method,
        )
        self.rng = np.random.RandomState(seed)
        self._iteration = 0

    # ------------------------------------------------------------------
    # 전략 풀 관리
    # ------------------------------------------------------------------

    def add_strategy(self, agent: LeagueAgent):
        """새 전략(에이전트 스냅샷)을 풀에 등록."""
        if len(self.strategies) >= self.config.max_strategies:
            self._prune_weakest()
        self.strategies[agent.agent_id] = agent
        self.payoff.add_strategy(agent.agent_id)

    def _prune_weakest(self):
        """α-Rank 점수 최하위 전략 제거 (풀 크기 제한)."""
        scores = self.alpha_rank()
        if not scores:
            return
        weakest_id = min(scores, key=lambda k: scores[k])
        self.strategies.pop(weakest_id, None)
        # PayoffMatrix는 rebuild (id 제거 후 재구성)
        remaining = list(self.strategies.keys())
        new_pm = PayoffMatrix()
        for sid in remaining:
            new_pm.add_strategy(sid)
        for i, a in enumerate(remaining):
            for j, b in enumerate(remaining):
                if i != j:
                    val = self.payoff.get(a, b)
                    new_pm.update(a, b, val)
        self.payoff = new_pm

    # ------------------------------------------------------------------
    # 경기 결과 기록
    # ------------------------------------------------------------------

    def record_result(self, id_a: str, id_b: str, result: float):
        """
        id_a vs id_b 경기 결과 기록.
        result: 1=a 승, 0=a 패, 0.5=무
        """
        if id_a not in self.strategies or id_b not in self.strategies:
            return
        self.payoff.update(id_a, id_b, result)
        # LeagueAgent ELO·기록도 함께 업데이트
        self.strategies[id_a].record_result(id_b, result)
        self.strategies[id_b].record_result(id_a, 1.0 - result)
        self.strategies[id_a].update_elo(self.strategies[id_b].elo_rating, result)
        self.strategies[id_b].update_elo(self.strategies[id_a].elo_rating, 1.0 - result)

    # ------------------------------------------------------------------
    # α-Rank 계산
    # ------------------------------------------------------------------

    def alpha_rank(self) -> Dict[str, float]:
        """
        현재 payoff 행렬로 α-Rank 정적 분포 계산.

        Returns
        -------
        Dict[str, float]
            {agent_id: score} — 합이 1.0인 확률 분포
        """
        ids = self.payoff.strategy_ids
        if not ids:
            return {}
        pi = self.alpha_calc.compute(self.payoff.matrix)
        return {sid: float(pi[i]) for i, sid in enumerate(ids)}

    def nash_mixture(self) -> Dict[str, float]:
        """
        균등 혼합 전략 (내쉬 근사).
        ELO 소프트맥스 기반 가중 혼합.
        """
        ids = list(self.strategies.keys())
        if not ids:
            return {}
        elos = np.array([self.strategies[sid].elo_rating for sid in ids], dtype=np.float64)
        weights = np.exp((elos - elos.max()) / 400.0)
        weights /= weights.sum()
        return {sid: float(weights[i]) for i, sid in enumerate(ids)}

    # ------------------------------------------------------------------
    # 상대 샘플링
    # ------------------------------------------------------------------

    def sample_opponent(self, focal_id: str) -> Optional[LeagueAgent]:
        """
        mixture_type 설정에 따라 상대를 샘플링.

        focal_id 자신은 제외하고 샘플링.
        """
        ids = [sid for sid in self.strategies if sid != focal_id]
        if not ids:
            return None

        if self.config.mixture_type == "alpha_rank":
            scores = self.alpha_rank()
            weights = np.array([scores.get(sid, 1e-6) for sid in ids], dtype=np.float64)
        elif self.config.mixture_type == "nash":
            nm = self.nash_mixture()
            weights = np.array([nm.get(sid, 1e-6) for sid in ids], dtype=np.float64)
        else:
            weights = np.ones(len(ids), dtype=np.float64)

        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()
        chosen_id = self.rng.choice(ids, p=weights)
        return self.strategies[chosen_id]

    # ------------------------------------------------------------------
    # PSRO 반복
    # ------------------------------------------------------------------

    def psro_iteration(self) -> Tuple[str, str]:
        """
        PSRO 반복 1회 — 학습할 focal과 상대 쌍 반환.
        호출자가 이 쌍으로 Best Response를 학습하고
        결과를 `record_result`로 기록해야 한다.
        """
        ids = list(self.strategies.keys())
        if len(ids) < 2:
            return ids[0], ids[0]

        # α-Rank 분포로 focal 선택
        scores = self.alpha_rank()
        # focal: 점수가 낮은(개선 여지가 큰) 전략 우선
        inv_weights = np.array([1.0 / (scores.get(sid, 1e-6) + 1e-9) for sid in ids])
        inv_weights /= inv_weights.sum()
        focal_id = self.rng.choice(ids, p=inv_weights)

        opponent = self.sample_opponent(focal_id)
        opponent_id = opponent.agent_id if opponent else focal_id
        self._iteration += 1
        return focal_id, opponent_id

    # ------------------------------------------------------------------
    # 리포트
    # ------------------------------------------------------------------

    def print_ranking(self, top_k: int = 10):
        scores = self.alpha_rank()
        if not scores:
            print("[PSROOracle] 전략 없음")
            return
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        print(f"\n[PSRO α-Rank 순위 (α={self.config.alpha})]")
        for rank, (sid, score) in enumerate(ranked[:top_k], 1):
            agent = self.strategies.get(sid)
            elo = agent.elo_rating if agent else 0.0
            wr  = agent.win_rate   if agent else 0.0
            print(f"  {rank:2d}. {sid:30s}  score={score:.4f}  ELO={elo:.0f}  WR={wr:.3f}")
        print(f"  총 전략 수: {self.payoff.n}  PSRO 반복: {self._iteration}")


    @property
    def payoff_matrix(self) -> PayoffMatrix:
        """Legacy alias for payoff matrix."""
        return self.payoff

    def add_initial_strategies(self, strategy_ids: List[str]):
        """Legacy helper to register strategy ids with default LeagueAgent metadata."""
        for sid in strategy_ids:
            if sid in self.strategies:
                continue
            self.add_strategy(LeagueAgent(agent_id=sid, role=AgentRole.MAIN))

    def compute_nash_mix(self) -> Dict[str, float]:
        """Legacy alias for nash_mixture."""
        return self.nash_mixture()

    @property
    def n_strategies(self) -> int:
        return len(self.strategies)


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PSRO Oracle 테스트 ===")

    oracle = PSROOracle(PSROConfig(alpha=10.0, mixture_type="alpha_rank"), seed=42)
    rng = np.random.RandomState(42)

    # 가상 전략 8개 등록
    for i in range(8):
        agent = LeagueAgent(
            agent_id=f"strategy_{i}",
            role=AgentRole.MAIN,
            elo_rating=1200.0 + rng.randn() * 50,
        )
        oracle.add_strategy(agent)

    # 가상 경기 40회
    ids = list(oracle.strategies.keys())
    for _ in range(40):
        a_id, b_id = rng.choice(ids, 2, replace=False)
        result = float(rng.beta(2, 2))
        oracle.record_result(a_id, b_id, result)

    oracle.print_ranking()

    # PSRO 반복 5회
    print("\n[PSRO 반복]")
    for step in range(5):
        focal, opp = oracle.psro_iteration()
        result = float(rng.beta(2, 2))
        oracle.record_result(focal, opp, result)
        print(f"  step {step+1}: {focal} vs {opp}  result={result:.3f}")

    print("\n✅ psro_oracle.py 정상 동작!")
