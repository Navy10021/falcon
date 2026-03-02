"""
nfsp_agent.py
=============
Neural Fictitious Self-Play (NFSP) 에이전트

BR(Best Response) + AS(Average Strategy) 이중 네트워크 구조로
근사 내쉬 균형에 수렴을 보장 (Heinrich & Silver, NIPS 2016).

구성:
  NFSPAgent     — BR(PPO) + AS(지도학습) 이중 정책 에이전트
  NFSPTrainer   — 두 에이전트 간 NFSP 학습 루프

학습 흐름:
  1. η 확률로 BR 정책 실행 → RL 버퍼 + SL 버퍼에 경험 저장
  2. (1-η) 확률로 AS 정책 실행 → RL 버퍼에만 경험 저장
  3. RL 버퍼로 BR 네트워크 PPO 업데이트
  4. SL 버퍼로 AS 네트워크 CrossEntropy 지도학습
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_agent.blue_agent import ActorCritic, PPOConfig, RolloutBuffer
from rl_agent.nfsp_buffer import NFSPBufferPair, NFSPConfig, Experience


# ──────────────────────────────────────────────
# NFSP 에이전트
# ──────────────────────────────────────────────

@dataclass
class NFSPAgentConfig:
    """NFSP 에이전트 하이퍼파라미터"""
    # 정책 혼합
    eta: float = 0.1               # BR 선택 확률
    # PPO (BR 네트워크)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # SL (AS 네트워크)
    sl_lr: float = 1e-3
    sl_batch_size: int = 128
    sl_n_epochs: int = 2
    # 버퍼
    buffer: NFSPConfig = field(default_factory=lambda: NFSPConfig(
        eta=0.1, rl_capacity=50_000, sl_capacity=200_000
    ))
    # 학습 시작 최소 크기
    min_rl_size: int = 256
    min_sl_size: int = 256


class NFSPAgent:
    """
    Neural Fictitious Self-Play 에이전트.

    두 개의 독립 네트워크를 유지한다:
      - br_network  (Best Response)  : PPO RL 업데이트
      - as_network  (Average Strategy): 지도학습(SL) 업데이트

    사용::
        agent = NFSPAgent(n_actions=6)
        use_br = agent.should_use_br()
        action, lp, val = agent.select_action(state, use_br=use_br)
        agent.store(state, action, reward, next_state, done, use_br)
        agent.update()
    """

    def __init__(
        self,
        n_actions: int = 6,
        state_dim: int = 128,
        config: Optional[NFSPAgentConfig] = None,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.config = config or NFSPAgentConfig()
        self.device = torch.device(device)

        # BR 네트워크 (PPO 업데이트)
        self.br_network = ActorCritic(n_actions=n_actions).to(self.device)
        self.br_optimizer = optim.Adam(
            self.br_network.parameters(), lr=self.config.ppo.lr
        )
        self.br_buffer = RolloutBuffer()

        # AS 네트워크 (지도학습 업데이트)
        self.as_network = ActorCritic(n_actions=n_actions).to(self.device)
        self.as_optimizer = optim.Adam(
            self.as_network.parameters(), lr=self.config.sl_lr
        )

        # NFSP 이중 버퍼
        self.nfsp_buffer = NFSPBufferPair(
            config=NFSPConfig(
                eta=self.config.eta,
                rl_capacity=self.config.buffer.rl_capacity,
                sl_capacity=self.config.buffer.sl_capacity,
            )
        )

        self.total_steps = 0
        self._last_use_br = True

    # ------------------------------------------------------------------
    # 행동 선택
    # ------------------------------------------------------------------

    def should_use_br(self) -> bool:
        """η 확률로 BR 정책 선택"""
        return self.nfsp_buffer.should_use_rl()

    def select_action(
        self,
        state: np.ndarray,
        use_br: bool = True,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        상태 → (action, log_prob, value)

        Parameters
        ----------
        use_br : bool
            True → BR 네트워크 사용 (Best Response)
            False → AS 네트워크 사용 (Average Strategy)
        """
        network = self.br_network if use_br else self.as_network
        self._last_use_br = use_br
        return network.get_action(state, deterministic)

    # ------------------------------------------------------------------
    # 경험 저장
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
        use_br: bool,
    ):
        """
        BR 스텝이면 두 버퍼 모두, AS 스텝이면 RL 버퍼만 저장.
        RolloutBuffer는 BR 경험만 누적 (PPO 업데이트용).
        """
        exp = Experience(state, action, reward, next_state, done)
        self.nfsp_buffer.add(exp, is_rl_step=use_br)

        if use_br:
            self.br_buffer.states.append(state)
            self.br_buffer.actions.append(action)
            self.br_buffer.rewards.append(reward)
            self.br_buffer.log_probs.append(log_prob)
            self.br_buffer.values.append(value)
            self.br_buffer.dones.append(done)

        self.total_steps += 1

    # ------------------------------------------------------------------
    # PPO 업데이트 (BR 네트워크)
    # ------------------------------------------------------------------

    def update_br(self, next_value: float = 0.0) -> Dict[str, float]:
        """BR 네트워크 PPO 클리핑 업데이트."""
        cfg = self.config.ppo
        buf = self.br_buffer
        if len(buf) < self.config.min_rl_size:
            return {}

        # GAE 이득 계산
        n = len(buf)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else buf.values[t + 1]
            mask = 0.0 if buf.dones[t] else 1.0
            delta = buf.rewards[t] + cfg.gamma * next_val * mask - buf.values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + buf.values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t  = torch.tensor(np.array(buf.states),  dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(buf.actions,           dtype=torch.long).to(self.device)
        old_lp_t  = torch.tensor(buf.log_probs,         dtype=torch.float32).to(self.device)
        adv_t     = torch.tensor(advantages,            dtype=torch.float32).to(self.device)
        rets_t    = torch.tensor(returns,               dtype=torch.float32).to(self.device)

        metrics = {"br_policy_loss": 0.0, "br_value_loss": 0.0, "br_entropy": 0.0}
        for _ in range(cfg.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, cfg.batch_size):
                b_idx = torch.tensor(idx[start:start + cfg.batch_size])
                logits, values_pred = self.br_network(states_t[b_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(actions_t[b_idx])
                entropy = dist.entropy().mean()
                ratio = (new_lp - old_lp_t[b_idx]).exp()
                adv_b = adv_t[b_idx]
                surr = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                ).mean()
                vl = F.mse_loss(values_pred.squeeze(), rets_t[b_idx])
                loss = surr + cfg.value_coef * vl - cfg.entropy_coef * entropy
                self.br_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.br_network.parameters(), cfg.max_grad_norm)
                self.br_optimizer.step()
                metrics["br_policy_loss"] += surr.item()
                metrics["br_value_loss"]  += vl.item()
                metrics["br_entropy"]     += entropy.item()

        buf.clear()
        return metrics

    # ------------------------------------------------------------------
    # SL 업데이트 (AS 네트워크)
    # ------------------------------------------------------------------

    def update_as(self) -> Dict[str, float]:
        """
        AS 네트워크 지도학습 업데이트.
        SL 버퍼에서 (state, action) 배치를 샘플하여 CrossEntropy 손실로 학습.
        """
        sl_buf = self.nfsp_buffer.sl_buffer
        if not sl_buf.is_ready(self.config.min_sl_size):
            return {}

        total_loss = 0.0
        n_batches = 0

        for _ in range(self.config.sl_n_epochs):
            states, actions, _, _, _ = sl_buf.sample_arrays(self.config.sl_batch_size)
            states_t  = torch.tensor(states,  dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)

            logits, _ = self.as_network(states_t)
            loss = F.cross_entropy(logits, actions_t)

            self.as_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.as_network.parameters(), 0.5)
            self.as_optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return {"as_sl_loss": total_loss / max(n_batches, 1)}

    # ------------------------------------------------------------------
    # 통합 업데이트
    # ------------------------------------------------------------------

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """BR + AS 업데이트를 동시에 실행하고 통합 지표 반환."""
        metrics: Dict[str, float] = {}
        metrics.update(self.update_br(next_value))
        metrics.update(self.update_as())
        return metrics

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "br_network": self.br_network.state_dict(),
            "as_network": self.as_network.state_dict(),
            "br_optimizer": self.br_optimizer.state_dict(),
            "as_optimizer": self.as_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.br_network.load_state_dict(ckpt["br_network"])
        self.as_network.load_state_dict(ckpt["as_network"])
        self.br_optimizer.load_state_dict(ckpt["br_optimizer"])
        self.as_optimizer.load_state_dict(ckpt["as_optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)

    @property
    def buffer_stats(self) -> Dict[str, int]:
        return self.nfsp_buffer.stats()


# ──────────────────────────────────────────────
# NFSP 학습 루프 (간소화)
# ──────────────────────────────────────────────

class NFSPTrainer:
    """
    두 NFSPAgent(Blue, Red)가 자기 대전하는 간소화 학습 루프.

    실제 전투 환경 연동 없이 단위 테스트 가능하도록
    더미 상태·보상을 사용.
    """

    def __init__(
        self,
        blue: NFSPAgent,
        red: NFSPAgent,
        n_actions: int = 6,
        state_dim: int = 128,
        seed: int = 0,
    ):
        self.blue = blue
        self.red = red
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rng = np.random.RandomState(seed)
        self.episode_count = 0

    def run_episode(self, n_steps: int = 50) -> Dict[str, float]:
        """
        단일 에피소드 실행.
        더미 환경: 상태 랜덤 생성, 보상은 행동 기반 근사.
        """
        blue_use_br = self.blue.should_use_br()
        red_use_br  = self.red.should_use_br()

        blue_rewards, red_rewards = [], []

        for step in range(n_steps):
            state = self.rng.randn(self.state_dim).astype(np.float32)
            done  = (step == n_steps - 1)

            b_action, b_lp, b_val = self.blue.select_action(state, use_br=blue_use_br)
            r_action, r_lp, r_val = self.red.select_action(state,  use_br=red_use_br)

            # 더미 보상: 공격적 행동 시 상대 불리 (간단 모델)
            b_reward = float(self.rng.normal(0.5 - r_action * 0.05, 0.3))
            r_reward = -b_reward + self.rng.normal(0, 0.1)

            next_state = self.rng.randn(self.state_dim).astype(np.float32)

            self.blue.store(state, b_action, b_reward, next_state, done, b_lp, b_val, blue_use_br)
            self.red.store( state, r_action, r_reward, next_state, done, r_lp, r_val, red_use_br)

            blue_rewards.append(b_reward)
            red_rewards.append(r_reward)

        self.episode_count += 1
        return {
            "blue_episode_reward": float(np.sum(blue_rewards)),
            "red_episode_reward":  float(np.sum(red_rewards)),
            "blue_use_br": float(blue_use_br),
            "red_use_br":  float(red_use_br),
        }

    def train(self, n_episodes: int = 10) -> List[Dict[str, float]]:
        """n_episodes 동안 학습 루프 실행."""
        logs = []
        for ep in range(n_episodes):
            ep_log = self.run_episode()
            blue_metrics = self.blue.update()
            red_metrics  = self.red.update()
            ep_log.update({f"blue_{k}": v for k, v in blue_metrics.items()})
            ep_log.update({f"red_{k}":  v for k, v in red_metrics.items()})
            logs.append(ep_log)
        return logs


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== NFSP Agent 테스트 ===")

    blue_cfg = NFSPAgentConfig(eta=0.1)
    red_cfg  = NFSPAgentConfig(eta=0.1)

    blue_agent = NFSPAgent(n_actions=6, state_dim=128, config=blue_cfg)
    red_agent  = NFSPAgent(n_actions=6, state_dim=128, config=red_cfg)

    br_params = sum(p.numel() for p in blue_agent.br_network.parameters())
    as_params = sum(p.numel() for p in blue_agent.as_network.parameters())
    print(f"BR 네트워크 파라미터: {br_params:,}")
    print(f"AS 네트워크 파라미터: {as_params:,}")

    trainer = NFSPTrainer(blue_agent, red_agent, n_actions=6, state_dim=128, seed=42)

    print("\n[5 에피소드 학습]")
    logs = trainer.train(n_episodes=5)
    for ep, log in enumerate(logs, 1):
        br_flag = "BR" if log["blue_use_br"] else "AS"
        print(f"  ep {ep}: Blue({br_flag}) reward={log['blue_episode_reward']:+.2f}  "
              f"Blue buf={blue_agent.buffer_stats}")

    print("\n✅ nfsp_agent.py 정상 동작!")
