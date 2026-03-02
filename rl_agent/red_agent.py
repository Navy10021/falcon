"""
red_agent.py
============
Phase 2: Red (적군) PPO 에이전트
- 비대칭 보상 함수 (Blue와 반대 목적)
- 기만 전술 (Deception) 행동 포함
- Self-Play 학습 지원
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field

from rl_agent.blue_agent import ActorCritic, RolloutBuffer, PPOConfig
import torch.nn.functional as F


@dataclass
class RedActionSpace:
    """Red 에이전트 행동 공간"""
    AMBUSH = 0          # 매복 → Blue 이동 시 기습
    COUNTER_ATTACK = 1  # 측면 역습
    DECEPTION = 2       # 허위 신호 → GNN 혼란
    FORTIFY = 3         # 방어 거점 구축
    RETREAT_REGROUP = 4 # 철수 후 재편성
    ARTILLERY_STRIKE = 5 # 포병 집중 타격

    N_ACTIONS = 6
    ACTION_NAMES = ["Ambush", "Counter-Attack", "Deception", "Fortify", "Retreat&Regroup", "Artillery Strike"]


STATE_DIM = 64


class RedAgent:
    """
    Red PPO 에이전트 (적 AI)
    목표: Blue의 임무 저지, 최대 피해 유발
    
    보상: R_R = +Damage + λ4×Blue_Casualty - λ5×Red_Loss
    """

    def __init__(self, config: Optional[PPOConfig] = None, device: str = "cpu"):
        self.config = config or PPOConfig(
            lr=2e-4,
            entropy_coef=0.02,   # 탐색 강화 (다양한 전술)
            uncertainty_penalty_coef=0.0  # Red는 불확실성 활용
        )
        self.device = torch.device(device)
        self.network = ActorCritic(n_actions=RedActionSpace.N_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()
        self.total_steps = 0
        self.deception_count = 0   # 기만 전술 사용 횟수

    def compute_reward(
        self,
        step_result,
        blue_casualties: int,
        red_casualties: int,
        damage_inflicted: float,
        action_taken: int
    ) -> float:
        """
        Red 보상 함수
        R_R = +Damage_Inflicted + λ4×Blue_Casualty - λ5×Red_Loss
        """
        # 임무 저지 보상
        mission_reward = 0.0
        if step_result.mission_status == "red_win":
            mission_reward = 10.0
        elif step_result.mission_status == "blue_win":
            mission_reward = -10.0

        # Blue 사상자 보상 (주 목표)
        blue_casualty_reward = blue_casualties * 0.08

        # Red 손실 패널티
        red_loss_penalty = red_casualties * 0.04

        # 피해 유발 보상
        damage_reward = damage_inflicted * 0.1

        # 기만 전술 보너스 (탐색 장려)
        deception_bonus = 0.3 if action_taken == RedActionSpace.DECEPTION else 0.0

        reward = (mission_reward
                  + blue_casualty_reward
                  - red_loss_penalty
                  + damage_reward
                  + deception_bonus)

        if action_taken == RedActionSpace.DECEPTION:
            self.deception_count += 1

        return float(reward)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        return self.network.get_action(state, deterministic)

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        """PPO 업데이트 (Blue Agent와 동일 알고리즘)"""
        cfg = self.config
        buf = self.buffer
        if len(buf) == 0:
            return {}

        advantages, returns = self.compute_gae(
            buf.rewards, buf.values, buf.dones, next_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t  = torch.tensor(np.array(buf.states),  dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(buf.actions,            dtype=torch.long).to(self.device)
        old_lp_t  = torch.tensor(buf.log_probs,         dtype=torch.float32).to(self.device)
        adv_t     = torch.tensor(advantages,            dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns,               dtype=torch.float32).to(self.device)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        n = len(buf)

        for _ in range(cfg.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, cfg.batch_size):
                batch_idx = torch.tensor(idx[start:start + cfg.batch_size])
                logits, values_pred = self.network(states_t[batch_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()
                ratio = (new_lp - old_lp_t[batch_idx]).exp()
                adv_b = adv_t[batch_idx]
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(values_pred.squeeze(), returns_t[batch_idx])
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += entropy.item()

        self.buffer.clear()
        self.total_steps += n
        return metrics

    def compute_observation_perturbation(
        self,
        obs: np.ndarray,
        perturbation_type: str = "gaussian",
        epsilon: float = 0.1,
        rng: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Blue 관측 벡터에 적대적 perturbation 주입.

        Red가 DECEPTION 행동을 선택했을 때 Blue 상태벡터를
        왜곡하여 잘못된 의사결정을 유도한다.

        Parameters
        ----------
        obs : np.ndarray
            Blue 에이전트의 원본 관측 벡터 (STATE_DIM,)
        perturbation_type : str
            "gaussian" — 가우시안 노이즈 (센서 오류 모사)
            "sign"     — FGSM 방식 부호 공격
            "uniform"  — 균등 노이즈
            "zero_out" — 랜덤 차원 제로화 (통신 두절 모사)
        epsilon : float
            perturbation 크기 (l∞ 기준)
        rng : optional np.ndarray
            재현용 RandomState (None이면 내부 rng 사용)

        Returns
        -------
        np.ndarray
            perturbation이 적용된 관측 벡터 (동일 shape)
        """
        _rng = rng if rng is not None else np.random.RandomState(self.total_steps)
        perturbed = obs.copy().astype(np.float32)

        if perturbation_type == "gaussian":
            noise = _rng.randn(*obs.shape).astype(np.float32) * epsilon
            perturbed = perturbed + noise

        elif perturbation_type == "sign":
            # FGSM-style: 부호만 사용하여 최대 l∞ 공격
            grad_sign = np.sign(_rng.randn(*obs.shape)).astype(np.float32)
            perturbed = perturbed + epsilon * grad_sign

        elif perturbation_type == "uniform":
            noise = _rng.uniform(-epsilon, epsilon, size=obs.shape).astype(np.float32)
            perturbed = perturbed + noise

        elif perturbation_type == "zero_out":
            # 랜덤하게 epsilon 비율의 차원을 0으로 설정 (통신 두절 효과)
            n_zero = max(1, int(len(obs) * epsilon))
            zero_idx = _rng.choice(len(obs), size=n_zero, replace=False)
            perturbed[zero_idx] = 0.0

        else:
            raise ValueError(f"Unknown perturbation_type: {perturbation_type!r}")

        # 벡터 범위 클리핑 (상태벡터 정규화 범위 유지)
        perturbed = np.clip(perturbed, -10.0, 10.0)
        return perturbed

    def apply_deception(self, fog_filter) -> bool:
        """기만 전술 실행 → Fog Filter에 노이즈 증가 요청"""
        if hasattr(fog_filter, 'noise'):
            fog_filter.noise.deception_prob = min(
                fog_filter.noise.deception_prob + 0.15, 0.6
            )
            return True
        return False

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "deception_count": self.deception_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.deception_count = ckpt.get("deception_count", 0)


if __name__ == "__main__":
    print("=== Red PPO Agent Test ===")
    from ontology.combat_schema import ScenarioFactory
    from rl_agent.blue_agent import build_state_vector

    agent = RedAgent()
    print(f"Red 네트워크 파라미터: {sum(p.numel() for p in agent.network.parameters()):,}")

    kg = ScenarioFactory.create_standard_scenario(seed=42)
    state = build_state_vector(kg)
    action, log_prob, value = agent.select_action(state)
    action_name = RedActionSpace.ACTION_NAMES[action]
    print(f"Red 선택 행동: {action_name} (log_prob={log_prob:.4f})")
    print("✅ Red Agent 테스트 완료!")
