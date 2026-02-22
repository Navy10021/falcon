"""
blue_agent.py
=============
Phase 1+2: Blue (아군) PPO 에이전트
- 불확실성 인식 상태 벡터 (GNN 출력 포함)
- 최소 병력으로 임무 달성 최적화
- Curriculum Learning 지원
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 상태/행동 공간 정의
# ──────────────────────────────────────────────

@dataclass
class BlueActionSpace:
    """Blue 에이전트 행동 공간"""
    REINFORCE = 0     # 현재 위치 고수 + 공격
    ADVANCE = 1       # 전진 공격
    WITHDRAW = 2      # 후퇴 (병력 보존)
    REALLOCATE = 3    # 병력 재배치
    SUPPORT = 4       # 인접 아군 지원
    FLANK = 5         # 측면 우회 기동

    N_ACTIONS = 6
    ACTION_NAMES = ["Reinforce", "Advance", "Withdraw", "Reallocate", "Support", "Flank"]


STATE_DIM = 64  # 전체 상태 벡터 차원


def build_state_vector(
    kg,
    gnn_extension: Optional[np.ndarray] = None,
    temporal_features: Optional[np.ndarray] = None,
    uncertainty_map: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    PPO 상태 벡터 구성 (64차원)
    
    기본 전장 상태 (40D) + GNN 불확실성 (6D) + 시계열 트렌드 (8D) + 패딩 (10D)
    """
    stats = kg.get_stats()

    # 기본 전장 상태 (40D)
    from ontology.combat_schema import ForceAlignment, UnitStatus

    blue_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
    red_units  = [u for u in kg.units.values() if u.alignment == ForceAlignment.RED]

    n_blue_active = sum(1 for u in blue_units if u.status != UnitStatus.DESTROYED)
    n_red_active  = sum(1 for u in red_units  if u.status != UnitStatus.DESTROYED)
    blue_hc = max(sum(u.headcount for u in blue_units), 1)
    red_hc  = max(sum(u.headcount for u in red_units),  1)
    blue_cp = sum(u.combat_power for u in blue_units)
    red_cp  = sum(u.combat_power for u in red_units)
    blue_morale = np.mean([u.morale for u in blue_units]) if blue_units else 0.5

    avg_unc = np.mean(list(uncertainty_map.values())) if uncertainty_map else 0.0

    base_state = np.array([
        # 병력 현황 (10D)
        n_blue_active / 10.0,
        n_red_active  / 10.0,
        blue_hc / 1000.0,
        red_hc  / 1000.0,
        blue_cp,
        red_cp,
        blue_cp / max(red_cp, 1e-6),  # 전력 비율
        blue_morale,
        avg_unc,                        # 전장 불확실성
        float(n_blue_active > n_red_active),  # 병력 우세 여부

        # 상태별 유닛 수 (Blue) (4D)
        sum(1 for u in blue_units if u.status.value == "active")    / max(n_blue_active, 1),
        sum(1 for u in blue_units if u.status.value == "degraded")  / max(n_blue_active, 1),
        sum(1 for u in blue_units if u.status.value == "critical")  / max(n_blue_active, 1),
        sum(1 for u in blue_units if u.status.value == "destroyed") / max(len(blue_units), 1),

        # 상태별 유닛 수 (Red) (4D)
        sum(1 for u in red_units if u.status.value == "active")    / max(n_red_active, 1),
        sum(1 for u in red_units if u.status.value == "degraded")  / max(n_red_active, 1),
        sum(1 for u in red_units if u.status.value == "critical")  / max(n_red_active, 1),
        sum(1 for u in red_units if u.status.value == "destroyed") / max(len(red_units), 1),

        # 유닛 유형 구성 Blue (8D)
        *[sum(1 for u in blue_units if u.unit_type.value == t) / max(len(blue_units), 1)
          for t in ["infantry", "armor", "artillery", "aviation", "engineer", "signal", "logistics", "electronic_warfare"]],

        # 유닛 유형 구성 Red (8D)
        *[sum(1 for u in red_units if u.unit_type.value == t) / max(len(red_units), 1)
          for t in ["infantry", "armor", "artillery", "aviation", "engineer", "signal", "logistics", "electronic_warfare"]],

        # 지형 정보 (6D)
        *[sum(1 for c in kg.terrain_cells.values() if c.terrain_type.value == t) / max(len(kg.terrain_cells), 1)
          for t in ["open", "urban", "forest", "mountain", "river", "coastal"]],
    ], dtype=np.float32)

    # GNN 불확실성 확장 (6D)
    if gnn_extension is None:
        gnn_extension = np.zeros(6, dtype=np.float32)

    # 시계열 트렌드 (8D)
    if temporal_features is None:
        temporal_features = np.zeros(8, dtype=np.float32)

    # 결합 및 패딩
    combined = np.concatenate([base_state[:40], gnn_extension[:6], temporal_features[:8]])
    # 64차원으로 패딩
    result = np.zeros(STATE_DIM, dtype=np.float32)
    result[:len(combined)] = combined[:STATE_DIM]
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = np.clip(result, -10.0, 10.0).astype(np.float32)
    return result


# ──────────────────────────────────────────────
# PPO Actor-Critic 네트워크
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    """공유 백본 Actor-Critic 네트워크"""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = BlueActionSpace.N_ACTIONS,
                 hidden_dim: int = 256):
        super().__init__()

        # 공유 특성 추출기
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Actor (정책)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Linear(128, n_actions)
        )

        # Critic (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Actor 마지막 레이어: 작은 초기값
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        logits = self.actor(feat)
        value  = self.critic(feat)
        return logits, value

    def get_action(self, state: np.ndarray,
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """
        상태 → (행동, 로그 확률, 가치)
        """
        safe_state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        x = torch.tensor(safe_state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = self(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1).item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action)).item()
        return int(action), float(log_prob), float(value.squeeze())


# ──────────────────────────────────────────────
# PPO 알고리즘
# ──────────────────────────────────────────────

@dataclass
class PPOConfig:
    """PPO 하이퍼파라미터"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 256
    uncertainty_penalty_coef: float = 0.1   # 불확실성 패널티 가중치


@dataclass
class RolloutBuffer:
    """PPO 롤아웃 버퍼"""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)

    def clear(self):
        for attr in ['states', 'actions', 'rewards', 'log_probs', 'values', 'dones', 'uncertainties']:
            setattr(self, attr, [])

    def add(self, state, action, reward, log_prob, value, done, uncertainty=0.0):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.uncertainties.append(uncertainty)

    def __len__(self):
        return len(self.states)


class BlueAgent:
    """
    Blue PPO 에이전트
    - 불확실성 인식 보상 함수
    - GAE (Generalized Advantage Estimation)
    """

    def __init__(self, config: Optional[PPOConfig] = None, device: str = "cpu"):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.network = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()
        self.total_steps = 0
        self.episode_count = 0

        # 훈련 기록
        self.training_logs: List[Dict] = []

    # ── 보상 계수 상수 ───────────────────────────────────────────────────
    _W_WIN          =  10.0   # 승리 보상
    _W_LOSS         = -10.0   # 패배 패널티
    _W_CASUALTY     =   0.05  # 아군 사상자당 패널티
    _W_FORCE_SAVE   =   0.15  # 스텝별 병력 절감 보상 (이전 0.01 → 0.15, 15× 상향)
    _W_FORCE_RATIO  =   2.0   # 임무 종료 시 잔존 병력 비율 보너스 스케일
    _W_ENEMY_DMG    =   0.03  # 적 사상자 보상 (이전 0.02 → 0.03)
    _W_DOCTRINE     =   1.0   # P3-5: 교리 준수 보너스 스케일
    # ────────────────────────────────────────────────────────────────────

    def compute_reward(
        self,
        step_result,            # BattleStep
        force_size_before: int,
        force_size_after: int,
        uncertainty: float = 0.0,
        config: Optional[PPOConfig] = None,
        initial_force_size: int = 0,    # 에피소드 시작 병력 (종료 보너스 계산용)
        doctrine_score: float = 0.0,    # P3-5: DoctrineCompliance.total_score [0,1]
    ) -> float:
        """
        보상 함수 (P2-6 보상 스케일 재조정, P3-5 교리 준수도 추가):

          R = w_win × WinSignal
            - w_cas  × BlueCasualties
            + w_save × ForceReduction        ← 핵심: 15× 상향
            + w_ratio× SurvivalRatioBonus    ← 임무 종료 시 잔존 비율 보너스
            + w_edm  × RedCasualties
            + w_doc  × DoctrineScore         ← P3-5: 교리 준수 보너스 [-0.5, +0.5]
            - UncertaintyPenalty
        """
        cfg = config or self.config

        # 1. 승/패 보상
        win_reward = 0.0
        terminal = step_result.mission_status != "ongoing"
        if step_result.mission_status == "blue_win":
            win_reward = self._W_WIN
        elif step_result.mission_status == "red_win":
            win_reward = -self._W_WIN

        # 2. 아군 사상자 패널티
        casualty_penalty = step_result.blue_total_casualties * self._W_CASUALTY

        # 3. 스텝별 병력 절감 보상 (핵심 목표 — 대폭 상향)
        force_reduction = force_size_before - force_size_after
        force_reward = self._W_FORCE_SAVE * force_reduction if force_reduction > 0 else 0.0

        # 4. 임무 종료 시 잔존 병력 비율 보너스
        #    최종 병력 / 초기 병력 비율이 높을수록 큰 보상
        survival_bonus = 0.0
        if terminal and initial_force_size > 0:
            survival_ratio = force_size_after / initial_force_size
            survival_bonus = self._W_FORCE_RATIO * survival_ratio

        # 5. 적 사상자 보상
        enemy_damage = step_result.red_total_casualties * self._W_ENEMY_DMG

        # 6. P3-5: 교리 준수 보너스 (score=0.5 → 중립, >0.5 보너스, <0.5 패널티)
        doctrine_bonus = self._W_DOCTRINE * (doctrine_score - 0.5)

        # 7. 불확실성 패널티 (불확실한 상황에서 과감한 병력 운용 시 패널티)
        uncertainty_penalty = 0.0
        if uncertainty > 0.5 and force_reduction > 0:
            uncertainty_penalty = (cfg.uncertainty_penalty_coef
                                   * uncertainty * force_reduction * 0.05)

        reward = (win_reward
                  - casualty_penalty
                  + force_reward
                  + survival_bonus
                  + enemy_damage
                  + doctrine_bonus
                  - uncertainty_penalty)

        return float(reward)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        return self.network.get_action(state, deterministic)

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation"""
        cfg = self.config
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            done_mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + cfg.gamma * next_val * done_mask - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * done_mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """PPO 업데이트"""
        cfg = self.config
        buf = self.buffer

        if len(buf) == 0:
            return {}

        # GAE 계산
        advantages, returns = self.compute_gae(
            buf.rewards, buf.values, buf.dones, next_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 텐서 변환
        states_np = np.nan_to_num(np.array(buf.states), nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        states_t   = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        actions_t  = torch.tensor(buf.actions,             dtype=torch.long).to(self.device)
        old_lp_t   = torch.tensor(buf.log_probs,          dtype=torch.float32).to(self.device)
        adv_t      = torch.tensor(advantages,             dtype=torch.float32).to(self.device)
        returns_t  = torch.tensor(returns,                dtype=torch.float32).to(self.device)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        n = len(buf)

        for _ in range(cfg.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, cfg.batch_size):
                batch_idx = torch.tensor(idx[start:start + cfg.batch_size])

                logits, values_pred = self.network(states_t[batch_idx])
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                values_pred = torch.nan_to_num(values_pred, nan=0.0, posinf=1e3, neginf=-1e3)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_t[batch_idx]).exp()
                adv_b = adv_t[batch_idx]

                # Clipped surrogate loss
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(values_pred.squeeze(), returns_t[batch_idx])

                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy)

                self.optimizer.zero_grad()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += entropy.item()

        self.buffer.clear()
        self.total_steps += n
        return metrics

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)


if __name__ == "__main__":
    print("=== Blue PPO Agent Test ===")
    from ontology.combat_schema import ScenarioFactory

    agent = BlueAgent()
    print(f"네트워크 파라미터: {sum(p.numel() for p in agent.network.parameters()):,}")

    kg = ScenarioFactory.create_standard_scenario(seed=42)
    state = build_state_vector(kg)
    print(f"상태 벡터 차원: {state.shape}")

    action, log_prob, value = agent.select_action(state)
    action_name = BlueActionSpace.ACTION_NAMES[action]
    print(f"선택 행동: {action_name} (log_prob={log_prob:.4f}, value={value:.4f})")
    print("✅ Blue Agent 테스트 완료!")
