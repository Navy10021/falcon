"""
mappo.py
========
⑥ 유닛별 분산 행동 공간 — Multi-Agent PPO (MAPPO)
- 각 유닛이 독립적으로 행동 선택
- 공유 Critic (Centralized Critic) + 분산 Actor
- CTDE (Centralized Training, Decentralized Execution)
- 유닛 유형별 특화 행동 마스크

아키텍처:
  각 유닛 i: 지역 관측 o_i → Actor_i → 행동 a_i
  Centralized:  전역 상태 S → Critic → V(S)
  
학술 연구용 합성 데이터
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────
# 유닛별 행동 공간
# ──────────────────────────────────────────────

UNIT_ACTION_MASKS = {
    # 유닛 유형: 허용 행동 인덱스 리스트
    # 행동: 0=대기, 1=전진, 2=후퇴, 3=교전, 4=지원, 5=측면, 6=보급요청, 7=방어진지
    "infantry":           [0, 1, 2, 3, 4, 5, 7],
    "armor":              [0, 1, 2, 3, 5],
    "artillery":          [0, 3, 7],             # 포병은 이동 제한
    "aviation":           [0, 1, 3, 5],
    "engineer":           [0, 1, 2, 7],
    "signal":             [0, 7],                 # 통신은 지원만
    "logistics":          [0, 1, 2, 6],           # 병참은 보급 특화
    "electronic_warfare": [0, 3, 7],
}

N_ACTIONS = 8
ACTION_NAMES = ["대기", "전진", "후퇴", "교전", "지원", "측면기동", "보급요청", "방어진지구축"]

UNIT_OBS_DIM = 24   # 유닛 지역 관측 차원
GLOBAL_STATE_DIM = 64   # 전역 상태 차원 (Centralized Critic)


# ──────────────────────────────────────────────
# Numpy 기반 경량 Actor (단일 유닛)
# ──────────────────────────────────────────────

class NumpyUnitActor:
    """
    단일 유닛용 경량 Actor (numpy 구현)
    지역 관측 → 행동 확률 분포
    """

    def __init__(self, obs_dim: int = UNIT_OBS_DIM,
                 n_actions: int = N_ACTIONS,
                 hidden_dim: int = 64, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_actions = n_actions
        self.W1 = rng.randn(obs_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, n_actions).astype(np.float32) * 0.01
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """관측 → 행동 로짓"""
        h = np.maximum(0, obs @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        if action_mask is not None:
            # 마스크된 행동 → -inf
            logits = np.where(action_mask, logits, -1e9)
        return logits

    def sample_action(
        self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float]:
        """확률적 행동 샘플링 → (action, log_prob)"""
        logits = self.forward(obs, action_mask)
        logits -= logits.max()   # 수치 안정성
        probs = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
        action = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob

    def greedy_action(
        self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> int:
        logits = self.forward(obs, action_mask)
        if action_mask is not None:
            logits = np.where(action_mask, logits, -1e9)
        return int(np.argmax(logits))


# ──────────────────────────────────────────────
# PyTorch 기반 학습 가능 Actor / Critic
# ──────────────────────────────────────────────

if TORCH_AVAILABLE:
    import torch.nn.functional as _F  # noqa: E402

    class TorchUnitActor(nn.Module):
        """
        단일 유닛용 학습 가능 Actor (torch.nn.Module).
        NumpyUnitActor 대비 실제 PPO gradient 업데이트 지원.
        유닛 유형 간 파라미터 공유 가능.
        """

        def __init__(self, obs_dim: int = UNIT_OBS_DIM, n_actions: int = N_ACTIONS,
                     hidden_dim: int = 64):
            super().__init__()
            self.n_actions = n_actions
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(self, obs: torch.Tensor,
                    action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            logits = self.net(obs)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask, -1e9)
            return logits

        def select_action(
            self, obs: np.ndarray,
            action_mask: Optional[np.ndarray] = None
        ) -> Tuple[int, float]:
            obs_t  = torch.from_numpy(obs).float()
            mask_t = torch.from_numpy(action_mask) if action_mask is not None else None
            with torch.no_grad():
                dist = torch.distributions.Categorical(
                    logits=self.forward(obs_t, mask_t))
                action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())

        def greedy_action(
            self, obs: np.ndarray,
            action_mask: Optional[np.ndarray] = None
        ) -> int:
            obs_t  = torch.from_numpy(obs).float()
            mask_t = torch.from_numpy(action_mask) if action_mask is not None else None
            with torch.no_grad():
                return int(self.forward(obs_t, mask_t).argmax().item())

        def evaluate_actions(
            self,
            obs_batch: torch.Tensor,
            actions_batch: torch.Tensor,
            masks_batch: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """PPO 업데이트용 (log_prob, entropy) 계산"""
            dist = torch.distributions.Categorical(
                logits=self.forward(obs_batch, masks_batch))
            return dist.log_prob(actions_batch), dist.entropy()

    class TorchCentralizedCritic(nn.Module):
        """CTDE 중앙화된 Critic — 전역 상태 → 가치 함수 (backprop 가능)."""

        def __init__(self, state_dim: int = GLOBAL_STATE_DIM, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state).squeeze(-1)

        def value(self, global_state: np.ndarray) -> float:
            with torch.no_grad():
                return float(self.forward(torch.from_numpy(global_state).float()).item())


class NumpyCentralizedCritic:
    """
    중앙화된 Critic (전역 상태 → 가치 함수)
    CTDE 패러다임: 훈련 시 전역 상태 접근, 실행 시 지역 관측만 사용
    """

    def __init__(self, state_dim: int = GLOBAL_STATE_DIM,
                 hidden_dim: int = 128, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, 64).astype(np.float32) * 0.1
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = rng.randn(64, 1).astype(np.float32) * 0.01
        self.b3 = np.zeros(1, dtype=np.float32)

    def value(self, global_state: np.ndarray) -> float:
        h = np.maximum(0, global_state @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        v = float((h @ self.W3 + self.b3).squeeze())
        return v


# ──────────────────────────────────────────────
# MAPPO 에이전트 (다중 유닛 관리)
# ──────────────────────────────────────────────

@dataclass
class UnitObservation:
    """유닛 지역 관측"""
    unit_id: str
    unit_type: str
    obs_vector: np.ndarray       # [UNIT_OBS_DIM]
    action_mask: np.ndarray      # [N_ACTIONS] bool


@dataclass
class UnitTransition:
    """단일 유닛 전환 기록 (버퍼용)"""
    unit_id: str
    observation: np.ndarray
    action: int
    log_prob: float
    reward: float
    global_value: float
    done: bool
    action_mask: np.ndarray
    unit_type: str = "infantry"   # HAPPO 순차 업데이트용 유닛 유형


class MAPPOManager:
    """
    MAPPO 다중 에이전트 관리자
    - 각 유닛 유형별 Actor 공유 (파라미터 공유)
    - 중앙화된 단일 Critic
    - 유닛별 행동 마스크 적용
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            # Torch 기반 Actor — 유닛 유형별 파라미터 공유 (실제 gradient 학습)
            self.type_actors: Dict[str, "TorchUnitActor"] = {  # type: ignore[assignment]
                unit_type: TorchUnitActor()
                for unit_type in UNIT_ACTION_MASKS.keys()
            }
            self.critic: "TorchCentralizedCritic" = TorchCentralizedCritic()  # type: ignore[assignment]
            # 통합 옵티마이저 (전체 Actor 파라미터 + Critic)
            _actor_params = [p for a in self.type_actors.values() for p in a.parameters()]
            self.actor_optimizer: Optional[torch.optim.Adam] = torch.optim.Adam(
                _actor_params, lr=3e-4)
            self.critic_optimizer: Optional[torch.optim.Adam] = torch.optim.Adam(
                self.critic.parameters(), lr=1e-3)
        else:
            self.type_actors = {
                unit_type: NumpyUnitActor(seed=seed + i)  # type: ignore[assignment]
                for i, unit_type in enumerate(UNIT_ACTION_MASKS.keys())
            }
            self.critic = NumpyCentralizedCritic(seed=seed + 100)  # type: ignore[assignment]
            self.actor_optimizer = None
            self.critic_optimizer = None
        self.buffers: Dict[str, List[UnitTransition]] = {}
        self.episode_count = 0

    def build_unit_observation(self, unit, kg, uncertainty_map=None) -> UnitObservation:
        """
        단일 유닛의 지역 관측 벡터 구성 (24차원)
        """
        from ontology.combat_schema import ForceAlignment, UnitStatus

        cap = unit.capability
        enemies = [u for u in kg.units.values()
                   if u.alignment != unit.alignment
                   and u.status != UnitStatus.DESTROYED]
        friends = [u for u in kg.units.values()
                   if u.alignment == unit.alignment
                   and u.status != UnitStatus.DESTROYED
                   and u.unit_id != unit.unit_id]

        # 가장 가까운 적
        if enemies:
            dists = [unit.position.distance_to(e.position) for e in enemies]
            nearest_enemy = enemies[np.argmin(dists)]
            min_dist = min(dists)
            nearest_enemy_cp = nearest_enemy.combat_power
        else:
            min_dist = 99.0
            nearest_enemy_cp = 0.0

        unc = uncertainty_map.get(unit.unit_id, 0.5) if uncertainty_map else 0.5

        obs = np.array([
            # 자신 상태 (8D)
            cap.firepower, cap.mobility, cap.protection,
            cap.ammo_level, cap.fuel_level, cap.comms_quality,
            unit.morale, unit.experience,

            # 병력 현황 (4D)
            unit.headcount / 200.0,
            unit.combat_power,
            float(len(friends)) / 10.0,
            float(len(enemies)) / 10.0,

            # 근거리 적 정보 (4D)
            min_dist / 30.0,
            nearest_enemy_cp,
            float(len([e for e in enemies if unit.position.distance_to(e.position) <= 5.0])),
            cap.range_km / 50.0,

            # 불확실성 (2D)
            unc,
            float(unit.status.value == "active"),

            # 위치 정규화 (2D)
            unit.position.x / 30.0,
            unit.position.y / 30.0,

            # 아군 지원 (2D)
            float(len([f for f in friends
                       if unit.position.distance_to(f.position) <= 5.0])),
            float(any(f.unit_type.value == "logistics" for f in friends)),

            # 패딩 (2D)
            0.0, 0.0,
        ], dtype=np.float32)[:UNIT_OBS_DIM]

        # 행동 마스크
        allowed = UNIT_ACTION_MASKS.get(unit.unit_type.value, list(range(N_ACTIONS)))
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[allowed] = True

        # 연료 고갈 시 이동 불가
        if cap.fuel_level < 0.05:
            mask[1] = False  # 전진 불가
            mask[5] = False  # 측면기동 불가

        # 탄약 고갈 시 교전 불가
        if cap.ammo_level < 0.05:
            mask[3] = False  # 교전 불가

        return UnitObservation(unit.unit_id, unit.unit_type.value, obs, mask)

    def select_actions(
        self, kg, uncertainty_map=None, deterministic: bool = False
    ) -> Dict[str, Tuple[int, float, str]]:
        """
        전체 유닛의 행동 동시 선택
        Returns: {unit_id: (action, log_prob, action_name)}
        """
        from ontology.combat_schema import ForceAlignment, UnitStatus

        actions = {}
        for uid, unit in kg.units.items():
            if (unit.alignment != ForceAlignment.BLUE
                    or unit.status == UnitStatus.DESTROYED):
                continue

            unit_obs = self.build_unit_observation(unit, kg, uncertainty_map)
            actor = self.type_actors.get(unit.unit_type.value,
                                          list(self.type_actors.values())[0])

            # TorchUnitActor / NumpyUnitActor 공통 인터페이스
            if deterministic:
                action   = actor.greedy_action(unit_obs.obs_vector, unit_obs.action_mask)
                log_prob = 0.0
            else:
                action, log_prob = actor.select_action(
                    unit_obs.obs_vector, unit_obs.action_mask
                )

            actions[uid] = (action, log_prob, ACTION_NAMES[action])

        return actions

    def compute_joint_reward(
        self,
        step_result: Dict,
        actions: Dict[str, Tuple[int, float, str]],
        force_before: int,
        force_after: int
    ) -> Dict[str, float]:
        """
        유닛별 개별 보상 계산
        - 팀 승리/패배 공유 보상
        - 개인 기여도 보상 (적 사상자 기여)
        - 보급 요청 성공 보상
        """
        rewards = {}
        mission_reward = 0.0
        if step_result.get("mission_status") == "blue_win":
            mission_reward = 5.0
        elif step_result.get("mission_status") == "red_win":
            mission_reward = -5.0

        blue_cas = step_result.get("blue_casualties", 0)
        casualty_penalty = blue_cas * 0.05

        for uid, (action, lp, action_name) in actions.items():
            reward = mission_reward - casualty_penalty

            # 보급 요청 보너스
            if action == 6:    # 보급요청
                reward += 0.5

            # 방어 진지 구축 보너스 (불확실성 높을 때)
            if action == 7:    # 방어진지
                reward += 0.3

            rewards[uid] = float(reward)

        return rewards

    def store_transitions(
        self,
        observations: Dict[str, UnitObservation],
        actions: Dict[str, Tuple[int, float, str]],
        rewards: Dict[str, float],
        global_value: float,
        dones: Dict[str, bool]
    ):
        """경험 버퍼에 전환 저장"""
        for uid in actions:
            if uid not in observations:
                continue
            obs = observations[uid]
            action, log_prob, _ = actions[uid]
            reward = rewards.get(uid, 0.0)
            done = dones.get(uid, False)

            if uid not in self.buffers:
                self.buffers[uid] = []
            self.buffers[uid].append(UnitTransition(
                unit_id=uid,
                observation=obs.obs_vector,
                action=action,
                log_prob=log_prob,
                reward=reward,
                global_value=global_value,
                done=done,
                action_mask=obs.action_mask,
                unit_type=obs.unit_type,   # HAPPO 순차 업데이트용
            ))

    def get_action_distribution_stats(self, kg) -> Dict[str, Dict]:
        """유닛별 행동 분포 통계"""
        from ontology.combat_schema import ForceAlignment, UnitStatus
        stats = {}
        for uid, unit in kg.units.items():
            if unit.alignment != ForceAlignment.BLUE or unit.status == UnitStatus.DESTROYED:
                continue
            allowed = UNIT_ACTION_MASKS.get(unit.unit_type.value, list(range(N_ACTIONS)))
            stats[uid] = {
                "unit_type": unit.unit_type.value,
                "n_allowed_actions": len(allowed),
                "allowed_actions": [ACTION_NAMES[a] for a in allowed],
            }
        return stats

    def update(
        self,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        gamma: float = 0.99,
    ) -> Dict:
        """
        PPO gradient 업데이트 (TORCH_AVAILABLE 시에만 실행).

        전체 버퍼의 전환 데이터를 통합하여 n_epochs 동안
        clip-PPO로 Actor / Critic 파라미터를 갱신한다.
        """
        if not TORCH_AVAILABLE or not self.buffers:
            return {}

        all_trans = [t for buf in self.buffers.values() for t in buf]
        if not all_trans:
            return {}

        obs_t    = torch.tensor(np.stack([t.observation   for t in all_trans]), dtype=torch.float32)
        acts_t   = torch.tensor([t.action                 for t in all_trans], dtype=torch.long)
        old_lp_t = torch.tensor([t.log_prob               for t in all_trans], dtype=torch.float32)
        masks_t  = torch.tensor(np.stack([t.action_mask   for t in all_trans]), dtype=torch.bool)

        rets_t = self._compute_returns(all_trans, gamma)
        vals_t = torch.tensor([t.global_value for t in all_trans], dtype=torch.float32)
        advs_t = rets_t - vals_t
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        # 공통 Actor (유닛 유형별 구분은 Phase 2 작업으로 연기)
        actor = list(self.type_actors.values())[0]

        actor_losses, critic_losses = [], []
        for _ in range(n_epochs):
            # ── Actor 업데이트 ──────────────────────────
            new_lp, entropy = actor.evaluate_actions(obs_t, acts_t, masks_t)
            ratio  = torch.exp(new_lp - old_lp_t)
            surr1  = ratio * advs_t
            surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs_t
            a_loss = (-torch.min(surr1, surr2) - entropy_coef * entropy).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            self.actor_optimizer.step()
            actor_losses.append(a_loss.item())

            # ── Critic 업데이트 ─────────────────────────
            # 전역 상태 없을 때: 관측 평균으로 대체 (근사)
            gs_t   = obs_t.mean(dim=1, keepdim=True).expand(-1, GLOBAL_STATE_DIM)
            values = self.critic.forward(gs_t.float())
            c_loss = value_coef * torch.nn.functional.mse_loss(values, rets_t)
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            critic_losses.append(c_loss.item())

        self.clear_buffers()
        self.episode_count += 1
        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "n_transitions": len(all_trans),
        }

    def _compute_returns(
        self, transitions: List[UnitTransition], gamma: float = 0.99
    ) -> "torch.Tensor":
        """Monte Carlo 누적 수익 계산 (할인 γ)"""
        returns, G = [], 0.0
        for t in reversed(transitions):
            G = t.reward + gamma * G * (1.0 - float(t.done))
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def happo_update(
        self,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        gamma: float = 0.99,
    ) -> Dict:
        """
        HAPPO (Heterogeneous-Agent PPO) 순차 업데이트.

        Zhong et al., ICLR 2022:
          "Heterogeneous-Agent Trust Region Policy Optimisation"

        동시 업데이트(기존 update()) 대신 유닛 유형을 순서대로 업데이트하여
        결합 단조 개선(joint monotonic improvement)을 보장한다.

        업데이트 순서 (지원→전투):
          logistics → signal → engineer → infantry → armor → artillery
          → aviation → electronic_warfare

        각 유닛 유형 i 업데이트 시:
          - 이전 유형들의 누적 중요도 비율(importance weight accumulation) 반영
          - loss = -E[∏_{k<i} r_k(θ) · r_i(θ) · A_i]

        Returns
        -------
        Dict
            유닛 유형별 손실 및 전체 통계
        """
        if not TORCH_AVAILABLE or not self.buffers:
            return {}

        all_trans = [t for buf in self.buffers.values() for t in buf]
        if not all_trans:
            return {}

        rets_t = self._compute_returns(all_trans, gamma)
        vals_t = torch.tensor([t.global_value for t in all_trans], dtype=torch.float32)
        advs_t = rets_t - vals_t
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        obs_t    = torch.tensor(np.stack([t.observation for t in all_trans]), dtype=torch.float32)
        acts_t   = torch.tensor([t.action               for t in all_trans], dtype=torch.long)
        old_lp_t = torch.tensor([t.log_prob             for t in all_trans], dtype=torch.float32)
        masks_t  = torch.tensor(np.stack([t.action_mask for t in all_trans]), dtype=torch.bool)

        # 유닛 유형별 인덱스 매핑 (unit_type 필드 사용)
        type_to_indices: Dict[str, List[int]] = {}
        for idx, trans in enumerate(all_trans):
            type_to_indices.setdefault(trans.unit_type, []).append(idx)

        # HAPPO 순차 업데이트 순서 (지원 → 전투)
        update_order = [
            "logistics", "signal", "engineer",
            "infantry", "armor", "artillery", "aviation", "electronic_warfare",
        ]

        # 누적 중요도 비율 (∏_{k<i} r_k) — 전체 샘플에 대해 유지
        accumulated_ratio = torch.ones(len(all_trans), dtype=torch.float32)

        type_losses: Dict[str, float] = {}
        critic_losses_all: List[float] = []

        for unit_type in update_order:
            if unit_type not in self.type_actors:
                continue
            indices = type_to_indices.get(unit_type, [])
            if not indices:
                accumulated_ratio = accumulated_ratio.detach()
                continue

            actor = self.type_actors[unit_type]
            if not isinstance(actor, TorchUnitActor):
                accumulated_ratio = accumulated_ratio.detach()
                continue

            idx_t = torch.tensor(indices, dtype=torch.long)
            type_actor_losses: List[float] = []

            for _ in range(n_epochs):
                new_lp, entropy = actor.evaluate_actions(
                    obs_t[idx_t], acts_t[idx_t], masks_t[idx_t]
                )
                ratio_i = torch.exp(new_lp - old_lp_t[idx_t])

                # 누적 비율 적용 (이전 유형들의 비율 곱)
                acc_ratio_i = accumulated_ratio[idx_t].detach() * ratio_i

                adv_i = advs_t[idx_t]
                surr1 = acc_ratio_i * adv_i
                surr2 = (
                    torch.clamp(acc_ratio_i, 1 - clip_eps, 1 + clip_eps) * adv_i
                )
                actor_loss = (-torch.min(surr1, surr2) - entropy_coef * entropy).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                self.actor_optimizer.step()
                type_actor_losses.append(actor_loss.item())

            # 이 유형의 비율로 누적 갱신 (다음 유형 업데이트에 반영)
            with torch.no_grad():
                new_lp_final, _ = actor.evaluate_actions(
                    obs_t[idx_t], acts_t[idx_t], masks_t[idx_t]
                )
                ratio_final = torch.exp(new_lp_final - old_lp_t[idx_t])
                accumulated_ratio = accumulated_ratio.clone()
                accumulated_ratio[idx_t] = accumulated_ratio[idx_t] * ratio_final

            type_losses[unit_type] = float(np.mean(type_actor_losses)) if type_actor_losses else 0.0

            # Critic 업데이트 (유닛 유형마다 1회)
            gs_t   = obs_t.mean(dim=1, keepdim=True).expand(-1, GLOBAL_STATE_DIM)
            values = self.critic.forward(gs_t.float())
            c_loss = value_coef * torch.nn.functional.mse_loss(values, rets_t)
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            critic_losses_all.append(c_loss.item())

        self.clear_buffers()
        self.episode_count += 1

        result = {
            "happo_critic_loss": float(np.mean(critic_losses_all)) if critic_losses_all else 0.0,
            "n_transitions": len(all_trans),
            "n_types_updated": len(type_losses),
        }
        result.update({f"happo_actor_{k}": v for k, v in type_losses.items()})
        return result

    def clear_buffers(self):
        self.buffers.clear()


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory, ForceAlignment

    print("=" * 55)
    print("⑥ Multi-Agent PPO (MAPPO) 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=42)
    manager = MAPPOManager(seed=42)

    # 행동 마스크 확인
    print("\n[유닛 유형별 허용 행동]")
    for unit_type, allowed in UNIT_ACTION_MASKS.items():
        action_names = [ACTION_NAMES[a] for a in allowed]
        print(f"  {unit_type:20s}: {action_names}")

    # 전체 유닛 행동 선택
    print("\n[유닛별 행동 선택]")
    actions = manager.select_actions(kg)
    for uid, (action, lp, name) in list(actions.items())[:4]:
        unit = kg.units[uid]
        print(f"  {uid:8s} ({unit.unit_type.value:15s}) → {name:15s} (lp={lp:.3f})")

    # 행동 분포 통계
    dist_stats = manager.get_action_distribution_stats(kg)
    total_action_space = sum(s["n_allowed_actions"] for s in dist_stats.values())
    print(f"\n총 유닛 수: {len(actions)}")
    print(f"평균 허용 행동 수: {total_action_space/max(len(dist_stats),1):.1f}")
    print(f"전체 결합 행동 공간 크기: {N_ACTIONS}^{len(actions)} = 매우 큼 → 분산으로 해결")

    print("\n✅ MAPPO 정상 동작!")
