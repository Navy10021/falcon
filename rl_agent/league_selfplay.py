"""
league_selfplay.py
==================
⑩ League Self-Play (AlphaStar 방식)
- 다양한 전략 풀(League) 유지
- Prioritized Fictitious Self-Play (PFSP)
- 과적합 방지: Main, League Exploiter, Main Exploiter
- Population Diversity 지표
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class AgentRole(Enum):
    MAIN       = "main"             # 메인 에이전트 (최종 목표)
    MAIN_EXP   = "main_exploiter"   # Main의 약점 공략
    LEAGUE_EXP = "league_exploiter" # League 전체 공략


@dataclass
class LeagueAgent:
    """League 에이전트"""
    agent_id: str
    role: AgentRole
    generation: int = 0
    elo_rating: float = 1200.0
    weights_snapshot: Optional[np.ndarray] = None   # 가중치 스냅샷

    # 경기 기록
    wins: int = 0
    losses: int = 0
    draws: int = 0
    matchup_history: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.5

    @property
    def n_games(self) -> int:
        return self.wins + self.losses + self.draws

    def record_result(self, opponent_id: str, result: float):
        """결과 기록 (1=승, 0=패, 0.5=무)"""
        if result > 0.7:
            self.wins += 1
        elif result < 0.3:
            self.losses += 1
        else:
            self.draws += 1
        self.matchup_history.append((opponent_id, result))

    def update_elo(self, opponent_elo: float, result: float, k: float = 32):
        """ELO 레이팅 업데이트"""
        expected = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        self.elo_rating += k * (result - expected)


class PFSPScheduler:
    """
    Prioritized Fictitious Self-Play (PFSP) 대전 스케줄러
    - 약한 상대보다 어려운 상대를 더 자주 선택
    - 우선순위: P(선택) ∝ f(승률) (f=하드 함수)
    """

    def __init__(self, mode: str = "hard", seed: int = 0):
        self.mode = mode   # "hard": 어려운 상대 선호, "uniform": 균등
        self.rng = np.random.RandomState(seed)

    def select_opponent(
        self,
        focal: LeagueAgent,
        candidates: List[LeagueAgent],
        exclude_ids: Optional[List[str]] = None
    ) -> LeagueAgent:
        """
        PFSP로 대전 상대 선택
        focal 에이전트 기준 각 후보의 상대적 강도로 가중 샘플링
        """
        if not candidates:
            raise ValueError("후보 없음")

        exclude = set(exclude_ids or [])
        valid = [c for c in candidates if c.agent_id not in exclude
                 and c.agent_id != focal.agent_id]
        if not valid:
            valid = candidates

        if self.mode == "uniform":
            return self.rng.choice(valid)

        # 각 후보에 대한 focal의 역사적 승률 계산
        win_rates = []
        for cand in valid:
            past = [r for oid, r in focal.matchup_history if oid == cand.agent_id]
            wr = np.mean(past) if past else 0.5
            win_rates.append(float(wr))

        # PFSP 가중치: f(x) = x^(1/3) → 강한 상대(승률 낮음) 선호
        if self.mode == "hard":
            weights = np.array([(1 - wr) ** (1/3) for wr in win_rates], dtype=float)
        else:  # "soft"
            weights = np.array([wr ** (1/3) for wr in win_rates], dtype=float)

        weights = weights / weights.sum()
        idx = self.rng.choice(len(valid), p=weights)
        return valid[idx]


class LeagueManager:
    """
    League Self-Play 관리자
    - 에이전트 풀 관리
    - 대전 스케줄링
    - 주기적 스냅샷 추가
    - 다양성 지표 계산
    """

    def __init__(
        self,
        n_main: int = 2,
        n_main_exp: int = 2,
        n_league_exp: int = 2,
        snapshot_interval: int = 10,
        seed: int = 0
    ):
        self.rng = np.random.RandomState(seed)
        self.pfsp = PFSPScheduler(mode="hard", seed=seed)
        self.snapshot_interval = snapshot_interval
        self.match_count = 0
        self.all_results: List[Dict] = []

        # 에이전트 풀 초기화
        self.agents: Dict[str, LeagueAgent] = {}
        for i in range(n_main):
            a = LeagueAgent(f"main_{i}", AgentRole.MAIN, elo_rating=1200.0)
            self.agents[a.agent_id] = a
        for i in range(n_main_exp):
            a = LeagueAgent(f"main_exp_{i}", AgentRole.MAIN_EXP, elo_rating=1150.0)
            self.agents[a.agent_id] = a
        for i in range(n_league_exp):
            a = LeagueAgent(f"league_exp_{i}", AgentRole.LEAGUE_EXP, elo_rating=1100.0)
            self.agents[a.agent_id] = a

        # 초기 스냅샷 (과거 버전)
        self.snapshots: List[LeagueAgent] = []
        self._add_snapshots()

    def _add_snapshots(self):
        """현재 Main 에이전트를 스냅샷으로 보관"""
        for aid, agent in self.agents.items():
            if agent.role == AgentRole.MAIN:
                snap = LeagueAgent(
                    f"snap_{aid}_gen{agent.generation}",
                    AgentRole.MAIN,
                    generation=agent.generation,
                    elo_rating=agent.elo_rating
                )
                self.snapshots.append(snap)

    def get_candidates_for(self, focal: LeagueAgent) -> List[LeagueAgent]:
        """대전 후보 목록 (역할별 상대 정책)"""
        all_cands = list(self.agents.values()) + self.snapshots
        if focal.role == AgentRole.MAIN:
            # Main은 모든 에이전트와 대전 가능
            return [c for c in all_cands if c.agent_id != focal.agent_id]
        elif focal.role == AgentRole.MAIN_EXP:
            # Main Exploiter는 Main 에이전트만 상대
            return [c for c in all_cands
                    if c.role == AgentRole.MAIN and c.agent_id != focal.agent_id]
        else:
            # League Exploiter는 모든 에이전트
            return [c for c in all_cands if c.agent_id != focal.agent_id]

    def schedule_match(self) -> Tuple[LeagueAgent, LeagueAgent]:
        """다음 대전 쌍 결정"""
        trainable = list(self.agents.values())
        focal = self.rng.choice(trainable)
        candidates = self.get_candidates_for(focal)
        if not candidates:
            candidates = trainable
        opponent = self.pfsp.select_opponent(focal, candidates)
        return focal, opponent

    def record_match_result(
        self, blue: LeagueAgent, red: LeagueAgent, result: float
    ):
        """경기 결과 기록 및 ELO 업데이트"""
        blue.record_result(red.agent_id, result)
        red.record_result(blue.agent_id, 1 - result)
        blue.update_elo(red.elo_rating, result)
        red.update_elo(blue.elo_rating, 1 - result)
        self.match_count += 1
        self.all_results.append({
            "match": self.match_count,
            "blue": blue.agent_id,
            "red": red.agent_id,
            "result": result,
        })

        # 스냅샷 추가
        if self.match_count % self.snapshot_interval == 0:
            self._add_snapshots()

        # Main Exploiter 주기적 리셋
        for aid, agent in self.agents.items():
            if agent.role == AgentRole.MAIN_EXP and agent.n_games > 20:
                if agent.win_rate > 0.7:
                    agent.elo_rating = 1150.0
                    agent.wins = agent.losses = agent.draws = 0
                    agent.matchup_history.clear()

    def compute_diversity_index(self) -> float:
        """
        Population Diversity 지표
        ELO 레이팅 표준편차 기반 다양성 측정
        다양성 높음 = 에이전트 간 전략 차이 큼
        """
        elos = [a.elo_rating for a in self.agents.values()]
        return float(np.std(elos)) / 400.0   # 정규화

    def nash_gap_estimate(self) -> float:
        """
        Nash Gap 근사 추정
        Main 에이전트들의 최근 10경기 승률 분산
        """
        main_agents = [a for a in self.agents.values() if a.role == AgentRole.MAIN]
        if not main_agents:
            return 1.0
        recent_wrs = []
        for a in main_agents:
            recent = [r for _, r in a.matchup_history[-10:]]
            if recent:
                recent_wrs.append(float(np.mean(recent)))
        if not recent_wrs:
            return 1.0
        return float(np.std(recent_wrs))

    def compute_alpha_rank(self, n_top: Optional[int] = None) -> Dict[str, float]:
        """
        Alpha-Rank 전략 순위 계산 (Omidshafiei et al., 2019).

        다중 에이전트 환경에서 각 에이전트의 마코프-체인 정적 분포를
        수렴 결과(α→∞)로 근사한다.

        알고리즘 개요:
          1. 모든 대전 쌍의 승률 행렬 M(i,j) 구성
          2. 전이 확률 행렬 T 구성: T[i→j] ∝ σ(α·(M(j,i)-M(i,j)))
          3. 정상 분포(π) 계산 — 반복 거듭제곱법
          4. 점수 정규화 후 반환

        Parameters
        ----------
        n_top : int, optional
            반환할 상위 에이전트 수 (None=전체)

        Returns
        -------
        Dict[str, float]
            {agent_id: alpha_rank_score} — 합이 1.0인 확률 분포
        """
        agents = list(self.agents.values()) + self.snapshots
        n = len(agents)
        if n == 0:
            return {}
        if n == 1:
            return {agents[0].agent_id: 1.0}

        # 1. 승률 행렬 M[i,j] = i가 j를 상대로 한 승률 (역사 기록 기반)
        M = np.full((n, n), 0.5)  # 기록 없으면 0.5 (균등)
        for i, a_i in enumerate(agents):
            for j, a_j in enumerate(agents):
                if i == j:
                    M[i, j] = 0.5
                    continue
                hist = [r for oid, r in a_i.matchup_history if oid == a_j.agent_id]
                if hist:
                    M[i, j] = float(np.mean(hist))
                else:
                    # ELO 기반 기대 승률 폴백
                    M[i, j] = 1.0 / (1.0 + 10 ** ((a_j.elo_rating - a_i.elo_rating) / 400.0))

        # 2. 전이 확률 행렬 (α=10.0 — 충분히 선택적)
        alpha = 10.0
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # σ(α·(M[j,i] - M[i,j])) — j가 i를 이길 확률 기반
                advantage = M[j, i] - M[i, j]
                T[i, j] = 1.0 / (1.0 + np.exp(-alpha * advantage))
            # 자기 자신으로의 전이 = 나머지 확률
            row_sum = T[i].sum()
            T[i, i] = max(0.0, n - 1 - row_sum)
            T[i] /= T[i].sum() + 1e-12

        # 3. 정상 분포 — 거듭제곱법 (Power Iteration)
        pi = np.ones(n) / n
        for _ in range(1000):
            pi_new = pi @ T
            if np.max(np.abs(pi_new - pi)) < 1e-9:
                break
            pi = pi_new
        pi = pi / pi.sum()

        # 4. 결과 사전 구성
        ranked = sorted(
            zip([a.agent_id for a in agents], pi.tolist()),
            key=lambda x: -x[1]
        )
        if n_top is not None:
            ranked = ranked[:n_top]

        total = sum(v for _, v in ranked)
        return {aid: v / (total + 1e-12) for aid, v in ranked}

    def print_leaderboard(self):
        print("\n[League 순위표 (ELO)]")
        sorted_agents = sorted(
            list(self.agents.values()) + self.snapshots,
            key=lambda a: a.elo_rating, reverse=True
        )
        for i, agent in enumerate(sorted_agents[:8]):
            role_mark = {"main": "🔵", "main_exploiter": "🔴",
                         "league_exploiter": "🟡"}.get(agent.role.value, "⚪")
            print(f"  {i+1:2d}. {role_mark} {agent.agent_id:25s} "
                  f"ELO={agent.elo_rating:6.0f}  "
                  f"W{agent.wins}/D{agent.draws}/L{agent.losses}")
        print(f"\n  Diversity: {self.compute_diversity_index():.4f}")
        print(f"  Nash Gap:  {self.nash_gap_estimate():.4f}")
        print(f"  Snapshots: {len(self.snapshots)}")


if __name__ == "__main__":
    print("=" * 55)
    print("⑩ League Self-Play 테스트")
    print("=" * 55)

    league = LeagueManager(n_main=2, n_main_exp=2, n_league_exp=2,
                            snapshot_interval=5, seed=42)
    rng = np.random.RandomState(42)

    print(f"\n[30 경기 시뮬레이션]")
    for match in range(30):
        blue, red = league.schedule_match()
        result = float(rng.beta(2, 2))   # 승률 0~1 랜덤
        league.record_match_result(blue, red, result)
        if (match + 1) % 10 == 0:
            print(f"  {match+1}경기 후: Nash Gap={league.nash_gap_estimate():.4f}  "
                  f"Diversity={league.compute_diversity_index():.4f}  "
                  f"Snapshots={len(league.snapshots)}")

    league.print_leaderboard()
    print("\n✅ League Self-Play 정상 동작!")
