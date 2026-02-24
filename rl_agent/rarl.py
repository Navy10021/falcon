"""
rarl.py
=======
Robust Adversarial Reinforcement Learning (RARL) + SA-PPO 구현

참조 논문:
  - Pinto et al., "Robust Adversarial Reinforcement Learning" (ICML 2017)
  - Zhang et al., "Robust Deep Reinforcement Learning against Adversarial Perturbations
    on State Observations" (NeurIPS 2020)  — SA-PPO (State-Adversarial PPO)

핵심 아이디어:
  미니맥스 최적화로 정책 강건성 보장:
    max_θ min_ν E[Σ rₜ | π_θ, ν_φ]

  Protagonist(Blue)는 최대화, Adversary는 관측 공간의
  ε-ball 내에서 교란(perturbation)을 최적화.

구성 요소:
  ObsAdversary   — 관측 벡터에 적대적 노이즈를 생성하는 네트워크
  RARLTrainer    — Protagonist ↔ Adversary 교대 업데이트
  sa_ppo_loss()  — SA-PPO 정규화 항 (KL 제약 관측 강건성)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_agent.blue_agent import ActorCritic, PPOConfig, RolloutBuffer, STATE_DIM


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class RARLConfig:
    """RARL 하이퍼파라미터"""
    # 교란 크기 제한
    epsilon: float = 0.05          # ε-ball 반경 (관측 벡터 정규화 기준)
    epsilon_decay: float = 1.0     # 매 업데이트마다 epsilon에 곱함 (어닐링)
    epsilon_min: float = 0.01      # 최소 ε

    # Protagonist (Blue PPO)
    protagonist_lr: float = 3e-4
    protagonist_n_epochs: int = 4
    protagonist_clip_eps: float = 0.2
    protagonist_entropy_coef: float = 0.01
    protagonist_value_coef: float = 0.5
    protagonist_gae_lambda: float = 0.95
    protagonist_gamma: float = 0.99
    protagonist_batch_size: int = 64
    protagonist_max_grad_norm: float = 0.5

    # Adversary 업데이트 주기 (Protagonist k회당 Adversary m회)
    adversary_update_freq: int = 1   # Protagonist 업데이트 k회 당
    adversary_update_steps: int = 1  # Adversary 업데이트 횟수

    # SA-PPO 정규화
    sa_ppo_kl_coef: float = 0.1     # SA-PPO KL 패널티 가중치
    sa_ppo_enabled: bool = True

    # 훈련 시 교란 확률
    train_perturb_prob: float = 0.2  # 훈련 스텝 중 교란 적용 확률


# ──────────────────────────────────────────────
# 관측 교란 유틸리티
# ──────────────────────────────────────────────

def add_adversarial_obs_noise(
    obs: np.ndarray,
    epsilon: float = 0.05,
    mode: str = "uniform",
    key_dims: Optional[List[int]] = None,
) -> np.ndarray:
    """
    관측 벡터에 적대적 노이즈 추가 (SA-PPO 방식).

    Parameters
    ----------
    obs : np.ndarray
        원본 관측 벡터
    epsilon : float
        교란 강도 (ε-ball 반경)
    mode : str
        "uniform"  — 균등 분포 노이즈
        "gaussian" — 가우시안 노이즈
        "targeted" — 핵심 차원 집중 교란 (적 거리·전투력 관련)
    key_dims : List[int], optional
        "targeted" 모드에서 교란할 차원 인덱스
    """
    if mode == "uniform":
        noise = np.random.uniform(-epsilon, epsilon, obs.shape)
    elif mode == "gaussian":
        noise = np.random.normal(0.0, epsilon / 2.0, obs.shape)
    elif mode == "targeted":
        noise = np.zeros_like(obs)
        dims = key_dims or [9, 10, 11, 12]   # 적 거리·전투력 관련 차원
        noise[dims] = np.random.uniform(-epsilon * 3, epsilon * 3, len(dims))
    else:
        noise = np.zeros_like(obs)
    return np.clip((obs + noise.astype(np.float32)), -10.0, 10.0)


# ──────────────────────────────────────────────
# Adversary 네트워크
# ──────────────────────────────────────────────

class ObsAdversary(nn.Module):
    """
    관측 공간 적대적 교란 생성기.

    입력:  관측 벡터 [state_dim]
    출력:  교란 벡터 [state_dim]  (tanh * ε로 클리핑)

    Adversary는 Protagonist의 성능을 낮추는 방향으로
    ε-ball 내 교란을 학습한다.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),  # 출력 ∈ [-1, 1]
        )

    def forward(self, obs: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        """교란 벡터 생성 (크기 ε 이내)"""
        delta = self.net(obs) * epsilon
        return obs + delta  # 교란된 관측

    def perturb_np(self, obs: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
        """NumPy 인터페이스"""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            perturbed = self.forward(obs_t, epsilon)
        return perturbed.squeeze(0).numpy()


# ──────────────────────────────────────────────
# SA-PPO 강건성 손실
# ──────────────────────────────────────────────

def sa_ppo_loss(
    policy: ActorCritic,
    obs_clean: torch.Tensor,
    obs_perturbed: torch.Tensor,
    kl_coef: float = 0.1,
) -> torch.Tensor:
    """
    SA-PPO (State-Adversarial PPO) 정규화 항.

    원본 관측과 교란 관측에서의 정책 분포 KL 발산을
    패널티로 추가하여 관측 교란에 강건한 정책 학습.

    L = L_PPO + kl_coef * KL(π(·|o) || π(·|o_adv))
    """
    with torch.no_grad():
        logits_clean, _ = policy(obs_clean)
        dist_clean = torch.distributions.Categorical(logits=logits_clean)

    logits_perturbed, _ = policy(obs_perturbed)
    dist_perturbed = torch.distributions.Categorical(logits=logits_perturbed)

    # KL(clean || perturbed): 교란 후 정책이 크게 변하지 않도록
    kl = torch.distributions.kl_divergence(dist_clean, dist_perturbed)
    return kl_coef * kl.mean()


# ──────────────────────────────────────────────
# RARLTrainer
# ──────────────────────────────────────────────

class RARLTrainer:
    """
    RARL 학습기 — Protagonist(Blue) ↔ Adversary 교대 업데이트.

    학습 흐름:
      1. Protagonist가 (교란된) 관측으로 에피소드 실행 → 버퍼 수집
      2. Protagonist PPO 업데이트 (+ SA-PPO KL 패널티)
      3. Adversary가 Protagonist 손실 반대 방향으로 업데이트 (최소화)
      4. ε 어닐링
    """

    def __init__(
        self,
        cfg: RARLConfig,
        n_actions: int = 6,
        state_dim: int = STATE_DIM,
        device: str = "cpu",
    ):
        self.cfg       = cfg
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.device    = torch.device(device)

        # Protagonist (Blue 에이전트)
        self.protagonist = ActorCritic(state_dim=state_dim, n_actions=n_actions).to(self.device)
        self.prot_opt    = optim.Adam(self.protagonist.parameters(), lr=cfg.protagonist_lr)
        self.prot_buffer = RolloutBuffer()

        # Adversary (관측 교란기)
        self.adversary  = ObsAdversary(state_dim=state_dim).to(self.device)
        self.adv_opt    = optim.Adam(self.adversary.parameters(), lr=3e-4)

        self._epsilon        = cfg.epsilon
        self._update_count   = 0
        self._episode_count  = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    # ------------------------------------------------------------------
    # 교란 적용
    # ------------------------------------------------------------------

    def perturb_obs(self, obs: np.ndarray, use_adversary: bool = True) -> np.ndarray:
        """
        관측에 교란 적용.
        use_adversary=True → ObsAdversary 네트워크로 교란
        use_adversary=False → 균등 노이즈로 교란 (기준선)
        """
        if use_adversary:
            return self.adversary.perturb_np(obs, self._epsilon)
        else:
            return add_adversarial_obs_noise(obs, self._epsilon, mode="uniform")

    def maybe_perturb(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        """훈련 중 perturb_prob 확률로 교란 적용"""
        if training and np.random.random() < self.cfg.train_perturb_prob:
            return self.perturb_obs(obs)
        return obs

    # ------------------------------------------------------------------
    # Protagonist 행동 선택
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, training: bool = True
    ) -> Tuple[int, float, float]:
        """Protagonist 행동 선택 (훈련 시 교란 관측 사용)"""
        perturbed = self.maybe_perturb(obs, training)
        return self.protagonist.get_action(perturbed, deterministic=not training)

    def select_action_clean(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """교란 없이 행동 선택 (평가용)"""
        return self.protagonist.get_action(obs, deterministic=True)

    # ------------------------------------------------------------------
    # PPO 업데이트 (Protagonist)
    # ------------------------------------------------------------------

    def update_protagonist(self, next_value: float = 0.0) -> Dict[str, float]:
        """Protagonist PPO + SA-PPO KL 패널티 업데이트."""
        buf = self.prot_buffer
        cfg = self.cfg
        n   = len(buf)
        if n == 0:
            return {}

        # GAE 이득 계산
        advantages = np.zeros(n, dtype=np.float32)
        G = 0.0
        for t in reversed(range(n)):
            nv   = next_value if t == n - 1 else buf.values[t + 1]
            mask = 0.0 if buf.dones[t] else 1.0
            delta = buf.rewards[t] + cfg.protagonist_gamma * nv * mask - buf.values[t]
            G = delta + cfg.protagonist_gamma * cfg.protagonist_gae_lambda * mask * G
            advantages[t] = G

        returns    = advantages + np.array(buf.values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t   = torch.tensor(np.array(buf.states),  dtype=torch.float32).to(self.device)
        acts_t  = torch.tensor(buf.actions,            dtype=torch.long).to(self.device)
        olp_t   = torch.tensor(buf.log_probs,          dtype=torch.float32).to(self.device)
        adv_t   = torch.tensor(advantages,             dtype=torch.float32).to(self.device)
        rets_t  = torch.tensor(returns,                dtype=torch.float32).to(self.device)

        # SA-PPO용 교란 관측 미리 생성
        if cfg.sa_ppo_enabled:
            with torch.no_grad():
                obs_perturbed = self.adversary(obs_t, self._epsilon)
        else:
            obs_perturbed = None

        policy_losses, value_losses, kl_losses = [], [], []
        for _ in range(cfg.protagonist_n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, cfg.protagonist_batch_size):
                b = perm[start:start + cfg.protagonist_batch_size]

                logits, values_pred = self.protagonist(obs_t[b])
                dist    = torch.distributions.Categorical(logits=logits)
                new_lp  = dist.log_prob(acts_t[b])
                entropy = dist.entropy().mean()
                ratio   = (new_lp - olp_t[b]).exp()
                adv_b   = adv_t[b]

                surr = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - cfg.protagonist_clip_eps, 1 + cfg.protagonist_clip_eps) * adv_b
                ).mean()
                vl = F.mse_loss(values_pred.squeeze(), rets_t[b])

                loss = surr + cfg.protagonist_value_coef * vl - cfg.protagonist_entropy_coef * entropy

                # SA-PPO KL 패널티
                if obs_perturbed is not None:
                    kl = sa_ppo_loss(self.protagonist, obs_t[b], obs_perturbed[b], cfg.sa_ppo_kl_coef)
                    loss = loss + kl
                    kl_losses.append(kl.item())

                self.prot_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.protagonist.parameters(), cfg.protagonist_max_grad_norm)
                self.prot_opt.step()
                policy_losses.append(surr.item())
                value_losses.append(vl.item())

        buf.clear()
        return {
            "rarl_protagonist_policy_loss": float(np.mean(policy_losses)),
            "rarl_protagonist_value_loss":  float(np.mean(value_losses)),
            "rarl_sa_kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
        }

    # ------------------------------------------------------------------
    # Adversary 업데이트 (관측 교란 최적화)
    # ------------------------------------------------------------------

    def update_adversary(self, obs_batch: np.ndarray) -> Dict[str, float]:
        """
        Adversary 업데이트: Protagonist의 예상 보상을 낮추도록 교란 최적화.

        Adversary 손실 = Protagonist의 행동 가치 평균 (최대화 방향 반전)
        """
        obs_t = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        obs_perturbed = self.adversary(obs_t, self._epsilon)

        with torch.no_grad():
            _, values_clean = self.protagonist(obs_t)

        _, values_perturbed = self.protagonist(obs_perturbed)

        # Adversary는 Protagonist 가치를 최소화
        adv_loss = values_perturbed.mean()  # 최소화 → gradient 하강
        self.adv_opt.zero_grad()
        adv_loss.backward()
        nn.utils.clip_grad_norm_(self.adversary.parameters(), 0.5)
        self.adv_opt.step()

        return {
            "rarl_adversary_loss": adv_loss.item(),
            "rarl_value_clean": values_clean.mean().item(),
            "rarl_value_perturbed": values_perturbed.mean().detach().item(),
        }

    # ------------------------------------------------------------------
    # 통합 업데이트
    # ------------------------------------------------------------------

    def update(
        self, obs_for_adversary: Optional[np.ndarray] = None, next_value: float = 0.0
    ) -> Dict[str, float]:
        """Protagonist + Adversary 교대 업데이트."""
        metrics: Dict[str, float] = {}

        # Protagonist 업데이트
        metrics.update(self.update_protagonist(next_value))

        # Adversary 업데이트 (주기적)
        if obs_for_adversary is not None and (
            self._update_count % self.cfg.adversary_update_freq == 0
        ):
            for _ in range(self.cfg.adversary_update_steps):
                metrics.update(self.update_adversary(obs_for_adversary))

        # ε 어닐링
        self._epsilon = max(
            self.cfg.epsilon_min,
            self._epsilon * self.cfg.epsilon_decay,
        )
        metrics["rarl_epsilon"] = self._epsilon
        self._update_count += 1
        return metrics

    # ------------------------------------------------------------------
    # 강건성 평가
    # ------------------------------------------------------------------

    def evaluate_robustness(
        self,
        obs_batch: np.ndarray,
        n_trials: int = 20,
    ) -> Dict[str, float]:
        """
        관측 교란에 대한 정책 강건성 측정.

        Returns
        -------
        Dict
            "action_consistency" : 교란 전후 동일 행동 선택 비율
            "logit_stability"    : 교란 전후 로짓 KL 발산 평균
        """
        obs_t = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits_clean, _ = self.protagonist(obs_t)
            actions_clean   = logits_clean.argmax(dim=-1)

        consistent_counts = []
        kl_divs = []

        for _ in range(n_trials):
            eps = np.random.uniform(0.01, self._epsilon)
            obs_perturbed = torch.tensor(
                add_adversarial_obs_noise(obs_batch, epsilon=eps), dtype=torch.float32
            ).to(self.device)
            with torch.no_grad():
                logits_pert, _ = self.protagonist(obs_perturbed)
                actions_pert   = logits_pert.argmax(dim=-1)

            consistent_counts.append((actions_clean == actions_pert).float().mean().item())
            dist_c = torch.distributions.Categorical(logits=logits_clean)
            dist_p = torch.distributions.Categorical(logits=logits_pert)
            kl_divs.append(torch.distributions.kl_divergence(dist_c, dist_p).mean().item())

        return {
            "action_consistency": float(np.mean(consistent_counts)),
            "logit_stability_kl": float(np.mean(kl_divs)),
        }


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RARL + SA-PPO 테스트 ===")

    cfg = RARLConfig(
        epsilon=0.05,
        sa_ppo_enabled=True,
        sa_ppo_kl_coef=0.1,
        train_perturb_prob=0.2,
    )
    trainer = RARLTrainer(cfg, n_actions=6, state_dim=128, device="cpu")

    n_prot  = sum(p.numel() for p in trainer.protagonist.parameters())
    n_adv   = sum(p.numel() for p in trainer.adversary.parameters())
    print(f"Protagonist 파라미터: {n_prot:,}")
    print(f"Adversary 파라미터  : {n_adv:,}")
    print(f"초기 ε              : {trainer.epsilon:.3f}")

    rng = np.random.RandomState(42)

    print("\n[에피소드 실행 + 업데이트]")
    obs_history = []
    for step in range(64):
        obs = rng.randn(128).astype(np.float32)
        action, lp, val = trainer.select_action(obs, training=True)
        reward = float(rng.normal(0.5, 0.3))
        done   = (step == 63)
        obs_history.append(obs)
        trainer.prot_buffer.states.append(obs)
        trainer.prot_buffer.actions.append(action)
        trainer.prot_buffer.log_probs.append(lp)
        trainer.prot_buffer.rewards.append(reward)
        trainer.prot_buffer.values.append(val)
        trainer.prot_buffer.dones.append(done)

    obs_batch = np.stack(obs_history[:16])
    metrics = trainer.update(obs_for_adversary=obs_batch)
    for k, v in metrics.items():
        print(f"  {k:40s}: {v:.4f}")

    # 강건성 평가
    print("\n[강건성 평가]")
    eval_obs = rng.randn(32, 128).astype(np.float32)
    robustness = trainer.evaluate_robustness(eval_obs, n_trials=10)
    for k, v in robustness.items():
        print(f"  {k:40s}: {v:.4f}")

    print("\n✅ rarl.py 정상 동작!")
