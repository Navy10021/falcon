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
import os
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


STATE_DIM = 128  # 전체 상태 벡터 차원 (확장판: 64 → 128)

# Branch / Domain 레이블 (순서 고정)
_BRANCHES = ["army", "navy", "air_force", "marine_corps", "strategic", "space", "cyber"]
_DOMAINS  = ["land", "sea", "air", "subsurface", "cyber", "space", "strategic"]


def build_state_vector(
    kg,
    gnn_extension: Optional[np.ndarray] = None,
    temporal_features: Optional[np.ndarray] = None,
    uncertainty_map: Optional[Dict[str, float]] = None,
    c2_quality: float = 1.0,           # 지휘통제 품질 [0,1]
    joint_fires_available: float = 0.0, # 합동화력 가용 여부 [0,1]
    intel_manager=None,                 # IntelligenceManager (optional)
) -> np.ndarray:
    """
    PPO 상태 벡터 구성 (128차원)

    구성:
      병력 현황 (10D) + Blue 상태분포 (4D) + Red 상태분포 (4D)
    + Blue 군종 구성 (7D) + Red 군종 구성 (7D)
    + Blue 도메인 구성 (7D) + Red 도메인 구성 (7D)
    + 지형 정보 (6D) + C2·합동화력 (4D)
    = 56D 기본
    + 해상·사이버 도메인 (6D)           ← V9: naval/cyber 확장
    +  GNN 불확실성 (8D) + 시계열 트렌드 (8D)
    = 78D → 128D 패딩

    해상·사이버 6D (V9):
      naval_blue_fraction   : Blue 해상 유닛 생존 비율
      naval_red_fraction    : Red 해상 유닛 생존 비율 (위협 지표)
      salvo_intensity       : 해상 화력 교환 강도 (전체 해상 전투력 정규화)
      cyber_degradation     : Blue 사이버 전력 저하도 (손실 비율)
      ew_suppression        : Red EW 유닛 능동 비율 (Blue 억제 수준)
      naval_threat_level    : Red 해상 전투력 / Blue 해상 전투력 비율

    intel_manager 제공 시:
      - gnn_extension 자동 계산 (정보 온톨로지 8D 특성)
      - uncertainty_map 자동 계산 (FogOfWar 불확실도)
    """
    from ontology.combat_schema import ForceAlignment, UnitStatus

    # intel_manager 자동 연동 (정보 온톨로지)
    if intel_manager is not None:
        if gnn_extension is None:
            try:
                gnn_extension = intel_manager.get_gnn_state_features(
                    ForceAlignment.BLUE, kg
                )
            except Exception:
                pass
        if uncertainty_map is None:
            try:
                uncertainty_map = intel_manager.get_uncertainty_map(
                    ForceAlignment.BLUE, kg
                )
            except Exception:
                pass

    blue_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
    red_units  = [u for u in kg.units.values() if u.alignment == ForceAlignment.RED]

    n_blue = max(len(blue_units), 1)
    n_red  = max(len(red_units),  1)
    n_blue_active = sum(1 for u in blue_units if u.status != UnitStatus.DESTROYED)
    n_red_active  = sum(1 for u in red_units  if u.status != UnitStatus.DESTROYED)
    blue_hc = max(sum(u.headcount for u in blue_units), 1)
    red_hc  = max(sum(u.headcount for u in red_units),  1)
    blue_cp = sum(u.combat_power for u in blue_units)
    red_cp  = sum(u.combat_power for u in red_units)
    blue_morale = np.mean([u.morale for u in blue_units]) if blue_units else 0.5

    avg_unc = np.mean(list(uncertainty_map.values())) if uncertainty_map else 0.0

    # ── 10D: 전장 종합 현황 ──────────────────────────────
    force_state = np.array([
        n_blue_active / 10.0,
        n_red_active  / 10.0,
        blue_hc / 10000.0,            # 대규모 부대 단위로 정규화
        red_hc  / 10000.0,
        float(np.clip(blue_cp, 0, 20)),
        float(np.clip(red_cp,  0, 20)),
        float(np.clip(blue_cp / max(red_cp, 1e-6), 0, 5)),  # 전력 비율
        blue_morale,
        avg_unc,
        float(n_blue_active > n_red_active),
    ], dtype=np.float32)

    # ── 4D: Blue 상태 분포 ────────────────────────────────
    blue_status = np.array([
        sum(1 for u in blue_units if u.status.value == "active")    / n_blue,
        sum(1 for u in blue_units if u.status.value == "degraded")  / n_blue,
        sum(1 for u in blue_units if u.status.value == "critical")  / n_blue,
        sum(1 for u in blue_units if u.status.value == "destroyed") / n_blue,
    ], dtype=np.float32)

    # ── 4D: Red 상태 분포 ─────────────────────────────────
    red_status = np.array([
        sum(1 for u in red_units if u.status.value == "active")    / n_red,
        sum(1 for u in red_units if u.status.value == "degraded")  / n_red,
        sum(1 for u in red_units if u.status.value == "critical")  / n_red,
        sum(1 for u in red_units if u.status.value == "destroyed") / n_red,
    ], dtype=np.float32)

    # ── 7D: Blue 군종(branch) 구성 ────────────────────────
    blue_branch = np.array([
        sum(1 for u in blue_units if hasattr(u, "branch") and u.branch.value == b) / n_blue
        for b in _BRANCHES
    ], dtype=np.float32)

    # ── 7D: Red 군종(branch) 구성 ─────────────────────────
    red_branch = np.array([
        sum(1 for u in red_units if hasattr(u, "branch") and u.branch.value == b) / n_red
        for b in _BRANCHES
    ], dtype=np.float32)

    # ── 7D: Blue 도메인 구성 ──────────────────────────────
    blue_domain = np.array([
        sum(1 for u in blue_units if hasattr(u, "domain") and u.domain.value == d) / n_blue
        for d in _DOMAINS
    ], dtype=np.float32)

    # ── 7D: Red 도메인 구성 ───────────────────────────────
    red_domain = np.array([
        sum(1 for u in red_units if hasattr(u, "domain") and u.domain.value == d) / n_red
        for d in _DOMAINS
    ], dtype=np.float32)

    # ── 6D: 지형 구성 ─────────────────────────────────────
    n_terrain = max(len(kg.terrain_cells), 1)
    terrain_state = np.array([
        sum(1 for c in kg.terrain_cells.values() if c.terrain_type.value == t) / n_terrain
        for t in ["open", "urban", "forest", "mountain", "river", "coastal"]
    ], dtype=np.float32)

    # ── 4D: C2·합동화력 정보 ─────────────────────────────
    c2_state = np.array([
        float(np.clip(c2_quality, 0, 1)),
        float(np.clip(joint_fires_available, 0, 1)),
        # Blue 잔존 전투력 비율 (사이버·전자전 유닛)
        sum(u.combat_power for u in blue_units
            if hasattr(u, "domain") and u.domain.value in ("cyber",)) / max(blue_cp, 1e-6),
        # Red 사이버·EW 위협 수준
        sum(u.combat_power for u in red_units
            if hasattr(u, "domain") and u.domain.value in ("cyber",)) / max(red_cp, 1e-6),
    ], dtype=np.float32)

    # 기본 56D 결합
    base_state = np.concatenate([
        force_state,    # 10
        blue_status,    # 4
        red_status,     # 4
        blue_branch,    # 7
        red_branch,     # 7
        blue_domain,    # 7
        red_domain,     # 7
        terrain_state,  # 6
        c2_state,       # 4
    ])  # = 56D

    # ── V9: 6D 해상·사이버 도메인 확장 ──────────────────────
    _naval_domains = {"sea", "subsurface"}
    _blue_naval = [u for u in blue_units
                   if hasattr(u, "domain") and u.domain.value in _naval_domains]
    _red_naval  = [u for u in red_units
                   if hasattr(u, "domain") and u.domain.value in _naval_domains]
    _blue_cyber = [u for u in blue_units
                   if hasattr(u, "domain") and u.domain.value == "cyber"]
    _red_ew     = [u for u in red_units
                   if hasattr(u, "domain") and u.domain.value in ("cyber", "electronic_warfare")]

    # naval_blue_fraction: Blue 해상 유닛 생존 비율
    _nb_total  = max(len(_blue_naval), 1)
    _nb_active = sum(1 for u in _blue_naval if u.status != UnitStatus.DESTROYED)
    naval_blue_frac = _nb_active / _nb_total

    # naval_red_fraction: Red 해상 유닛 생존 비율 (위협 지표)
    _nr_total  = max(len(_red_naval), 1)
    _nr_active = sum(1 for u in _red_naval if u.status != UnitStatus.DESTROYED)
    naval_red_frac = _nr_active / _nr_total if _red_naval else 0.0

    # salvo_intensity: 해상 화력 교환 강도 (전체 해상 전투력 정규화)
    _blue_naval_cp = sum(u.combat_power for u in _blue_naval)
    _red_naval_cp  = sum(u.combat_power for u in _red_naval)
    salvo_intensity = float(np.clip(
        (_blue_naval_cp + _red_naval_cp) / max(blue_cp + red_cp, 1e-6), 0.0, 1.0
    ))

    # cyber_degradation: Blue 사이버 전력 저하도 (파괴/손실 비율)
    _bc_total = max(len(_blue_cyber), 1)
    _bc_lost  = sum(1 for u in _blue_cyber if u.status == UnitStatus.DESTROYED)
    cyber_degradation = _bc_lost / _bc_total if _blue_cyber else 0.0

    # ew_suppression: Red EW 유닛 능동 비율 (Blue에 대한 전자전 억제)
    _rew_active = sum(1 for u in _red_ew if u.status != UnitStatus.DESTROYED)
    ew_suppression = _rew_active / max(n_red, 1)

    # naval_threat_level: Red 해상 전투력 / Blue 해상 전투력
    naval_threat_level = float(np.clip(
        _red_naval_cp / max(_blue_naval_cp, 1e-6), 0.0, 5.0
    ) / 5.0)  # 0~1로 정규화

    naval_cyber_state = np.array([
        naval_blue_frac,
        naval_red_frac,
        salvo_intensity,
        cyber_degradation,
        ew_suppression,
        naval_threat_level,
    ], dtype=np.float32)  # 6D

    # GNN 불확실성 확장 (8D)
    if gnn_extension is None:
        gnn_extension = np.zeros(8, dtype=np.float32)
    gnn_ext = np.zeros(8, dtype=np.float32)
    n_gnn = min(len(gnn_extension), 8)
    gnn_ext[:n_gnn] = gnn_extension[:n_gnn]

    # 시계열 트렌드 (8D)
    if temporal_features is None:
        temporal_features = np.zeros(8, dtype=np.float32)
    temp_feat = np.zeros(8, dtype=np.float32)
    n_temp = min(len(temporal_features), 8)
    temp_feat[:n_temp] = temporal_features[:n_temp]

    # 78D → 128D 패딩
    combined = np.concatenate([base_state, naval_cyber_state, gnn_ext, temp_feat])  # 78D
    result = np.zeros(STATE_DIM, dtype=np.float32)
    result[:len(combined)] = combined[:STATE_DIM]
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    result = np.clip(result, -10.0, 10.0).astype(np.float32)
    return result


# ──────────────────────────────────────────────
# PPO Actor-Critic 네트워크
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    """공유 백본 Actor-Critic 네트워크 (STATE_DIM=128 지원)"""

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
    # 하위 호환성: 일부 경로(hp_search/구버전 설정)에서 clip_range를 전달함
    clip_range: Optional[float] = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 256
    uncertainty_penalty_coef: float = 0.1   # 불확실성 패널티 가중치

    def __post_init__(self):
        # clip_range가 주어지면 clip_eps와 동기화
        if self.clip_range is not None:
            self.clip_eps = float(self.clip_range)
        else:
            self.clip_range = float(self.clip_eps)


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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
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
