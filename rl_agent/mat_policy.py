"""
mat_policy.py
=============
Multi-Agent Transformer (MAT) 정책 구현

Wen et al., "Multi-Agent Transformer" (NeurIPS 2022)
https://arxiv.org/abs/2205.14953

핵심 아이디어:
  Transformer 인코더-디코더 구조로 다중 에이전트 결합 행동을
  자기회귀적(auto-regressive)으로 분해:

    π(a₁,...,aₙ | o) = ∏ᵢ πᵢ(aᵢ | a_{<i}, o)

  - Encoder: 모든 에이전트 관측 o → 컨텍스트 memory
  - Decoder: memory + 이전 행동 → 현재 에이전트 행동 logit

FALCON 통합:
  - MAPPOManager.select_actions()를 MATPolicy.select_actions()로 교체 가능
  - 유닛 수 가변(n_blue 동적 변화)에 자연스럽게 대응
  - 유닛 유형 임베딩으로 이종 에이전트(heterogeneous) 지원
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

UNIT_TYPES = [
    "infantry", "armor", "artillery", "aviation",
    "engineer", "signal", "logistics", "electronic_warfare",
]
N_UNIT_TYPES = len(UNIT_TYPES)
UNIT_TYPE_IDX: Dict[str, int] = {t: i for i, t in enumerate(UNIT_TYPES)}


@dataclass
class MATConfig:
    """MAT 하이퍼파라미터"""
    obs_dim: int = 24          # UNIT_OBS_DIM (mappo.py와 일치)
    n_actions: int = 8         # N_ACTIONS (mappo.py와 일치)
    d_model: int = 128         # Transformer 임베딩 차원
    n_heads: int = 4           # 멀티-헤드 어텐션 헤드 수
    n_enc_layers: int = 2      # 인코더 레이어 수
    n_dec_layers: int = 2      # 디코더 레이어 수
    dropout: float = 0.0
    max_agents: int = 32       # 에이전트 최대 수 (positional emb용)
    # PPO 하이퍼파라미터
    lr: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


# ──────────────────────────────────────────────
# MATEncoder
# ──────────────────────────────────────────────

class MATEncoder(nn.Module):
    """
    멀티-에이전트 관측 인코더.

    입력:  obs_batch [B, N, obs_dim]
           type_ids  [B, N]  (유닛 유형 인덱스)
    출력:  memory    [B, N, d_model]  (각 에이전트 컨텍스트 표현)
    """

    def __init__(self, cfg: MATConfig):
        super().__init__()
        self.obs_embed  = nn.Linear(cfg.obs_dim, cfg.d_model)
        self.type_embed = nn.Embedding(N_UNIT_TYPES + 1, cfg.d_model, padding_idx=N_UNIT_TYPES)
        self.pos_embed  = nn.Embedding(cfg.max_agents, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_enc_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        obs_batch: torch.Tensor,      # [B, N, obs_dim]
        type_ids: torch.Tensor,       # [B, N]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N] bool (True=pad)
    ) -> torch.Tensor:
        B, N, _ = obs_batch.shape
        pos = torch.arange(N, device=obs_batch.device).unsqueeze(0).expand(B, N)

        x = self.obs_embed(obs_batch) + self.type_embed(type_ids) + self.pos_embed(pos)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)  # [B, N, d_model]


# ──────────────────────────────────────────────
# MATDecoder
# ──────────────────────────────────────────────

class MATDecoder(nn.Module):
    """
    자기회귀적 행동 디코더.

    에이전트 순서 i에 대해:
      tgt = <BOS>, a_1, ..., a_{i-1}
      → 디코더 → 행동 logit for aᵢ

    훈련 시: teacher-forcing (한 번의 forward로 모든 행동 logit 계산)
    추론 시: 순차 생성 (auto-regressive)

    입력:  memory        [B, N, d_model]
           prev_actions  [B, N]  (0=<BOS>, 1~n_actions=실제 행동+1)
    출력:  logits        [B, N, n_actions]
           values        [B, N]
    """

    def __init__(self, cfg: MATConfig):
        super().__init__()
        # +1: <BOS> 토큰 (인덱스 0)
        self.action_embed = nn.Embedding(cfg.n_actions + 1, cfg.d_model)
        self.pos_embed    = nn.Embedding(cfg.max_agents, cfg.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer  = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_dec_layers)
        self.norm         = nn.LayerNorm(cfg.d_model)
        self.action_head  = nn.Linear(cfg.d_model, cfg.n_actions)
        self.value_head   = nn.Linear(cfg.d_model, 1)

    def _causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """인과 마스크: 미래 행동을 볼 수 없음"""
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        memory: torch.Tensor,         # [B, N, d_model]
        prev_actions: torch.Tensor,   # [B, N]  (이전 행동 시프트; 0=BOS)
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = memory.shape
        pos = torch.arange(N, device=memory.device).unsqueeze(0).expand(B, N)
        tgt = self.action_embed(prev_actions) + self.pos_embed(pos)
        causal = self._causal_mask(N, memory.device)

        out = self.transformer(
            tgt,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = self.norm(out)  # [B, N, d_model]
        logits = self.action_head(out)   # [B, N, n_actions]
        values = self.value_head(out).squeeze(-1)  # [B, N]
        return logits, values


# ──────────────────────────────────────────────
# MATPolicy
# ──────────────────────────────────────────────

class MATPolicy(nn.Module):
    """
    Multi-Agent Transformer 통합 정책.

    API:
        policy = MATPolicy(cfg)
        # 추론
        actions, log_probs, values = policy.select_actions(obs, type_ids, masks)
        # 학습
        logits, values = policy(obs, type_ids, prev_actions)
    """

    def __init__(self, cfg: MATConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = MATEncoder(cfg)
        self.decoder = MATDecoder(cfg)

    def forward(
        self,
        obs: torch.Tensor,              # [B, N, obs_dim]
        type_ids: torch.Tensor,         # [B, N]
        prev_actions: torch.Tensor,     # [B, N]  (teacher-forcing)
        pad_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.encoder(obs, type_ids, key_padding_mask=pad_mask)
        logits, values = self.decoder(memory, prev_actions, memory_key_padding_mask=pad_mask)
        return logits, values   # [B, N, n_actions], [B, N]

    @torch.no_grad()
    def select_actions(
        self,
        obs: torch.Tensor,          # [B, N, obs_dim]
        type_ids: torch.Tensor,     # [B, N]
        action_masks: Optional[torch.Tensor] = None,  # [B, N, n_actions] bool (True=허용)
        pad_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        자기회귀 행동 샘플링.

        Returns
        -------
        actions   [B, N]
        log_probs [B, N]
        values    [B, N]
        """
        B, N = obs.shape[:2]
        memory = self.encoder(obs, type_ids, key_padding_mask=pad_mask)

        # auto-regressive 생성
        prev_actions = torch.zeros(B, N, dtype=torch.long, device=obs.device)  # 초기: all BOS
        actions   = torch.zeros(B, N, dtype=torch.long,  device=obs.device)
        log_probs = torch.zeros(B, N, dtype=torch.float, device=obs.device)
        values_all= torch.zeros(B, N, dtype=torch.float, device=obs.device)

        for i in range(N):
            # i번째 에이전트 행동 결정
            # prev_actions[:, :i+1]: BOS + 이전 행동들
            logits_i, vals_i = self.decoder(memory, prev_actions, memory_key_padding_mask=pad_mask)
            logit_i = logits_i[:, i, :]   # [B, n_actions]
            val_i   = vals_i[:, i]         # [B]

            # 행동 마스크 적용
            if action_masks is not None:
                mask_i = action_masks[:, i, :]  # [B, n_actions]
                logit_i = logit_i.masked_fill(~mask_i, -1e9)

            dist = torch.distributions.Categorical(logits=logit_i)
            if deterministic:
                a_i = logit_i.argmax(dim=-1)
            else:
                a_i = dist.sample()

            lp_i = dist.log_prob(a_i)
            actions[:, i]    = a_i
            log_probs[:, i]  = lp_i
            values_all[:, i] = val_i

            # 다음 단계를 위해 prev_actions 업데이트 (a+1: BOS=0 예약)
            if i + 1 < N:
                prev_actions = prev_actions.clone()
                prev_actions[:, i + 1] = a_i + 1

        return actions, log_probs, values_all


# ──────────────────────────────────────────────
# MAT 전환 버퍼
# ──────────────────────────────────────────────

@dataclass
class MATTransition:
    """MAT 학습용 단일 전환 (B=1 episode step)"""
    obs: np.ndarray          # [N, obs_dim]
    type_ids: np.ndarray     # [N]
    actions: np.ndarray      # [N]
    log_probs: np.ndarray    # [N]
    values: np.ndarray       # [N]
    rewards: np.ndarray      # [N]
    dones: np.ndarray        # [N]
    action_masks: Optional[np.ndarray] = None  # [N, n_actions]
    pad_mask: Optional[np.ndarray] = None      # [N]  True=pad


class MATRolloutBuffer:
    """MAT 에피소드 전환 버퍼"""

    def __init__(self):
        self.transitions: List[MATTransition] = []

    def add(self, t: MATTransition):
        self.transitions.append(t)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


# ──────────────────────────────────────────────
# MATTrainer (PPO 업데이트)
# ──────────────────────────────────────────────

class MATTrainer:
    """
    Multi-Agent Transformer PPO 학습기.

    MATPolicy를 buffer에 쌓인 에피소드 전환으로 업데이트한다.
    CTDE 패러다임: 훈련 시 모든 에이전트 관측에 접근.
    """

    def __init__(self, cfg: MATConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)
        self.policy = MATPolicy(cfg).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.buffer = MATRolloutBuffer()
        self.episode_count = 0

    # ------------------------------------------------------------------
    # 행동 선택 (NumPy 인터페이스)
    # ------------------------------------------------------------------

    def select_actions_np(
        self,
        obs: np.ndarray,             # [N, obs_dim]
        type_ids: np.ndarray,        # [N]
        action_masks: Optional[np.ndarray] = None,  # [N, n_actions]
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy → 행동 선택 → NumPy 반환 (inference용)"""
        obs_t     = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        type_t    = torch.tensor(type_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        masks_t   = None
        if action_masks is not None:
            masks_t = torch.tensor(action_masks, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            actions, log_probs, values = self.policy.select_actions(
                obs_t, type_t, masks_t, deterministic=deterministic
            )

        return (
            actions.squeeze(0).cpu().numpy(),
            log_probs.squeeze(0).cpu().numpy(),
            values.squeeze(0).cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # PPO 업데이트
    # ------------------------------------------------------------------

    def update(self, next_values: Optional[np.ndarray] = None) -> Dict[str, float]:
        """버퍼 데이터로 MAT PPO 업데이트."""
        if len(self.buffer) == 0:
            return {}

        T = len(self.buffer)
        cfg = self.cfg

        # GAE 이득 계산
        advantages = []
        G = np.zeros_like(self.buffer.transitions[0].values)
        for t in reversed(range(T)):
            tr = self.buffer.transitions[t]
            mask = 1.0 - tr.dones.astype(np.float32)
            nv = next_values if (t == T - 1 and next_values is not None) else (
                self.buffer.transitions[t + 1].values if t + 1 < T else np.zeros_like(tr.values)
            )
            delta = tr.rewards + cfg.gamma * nv * mask - tr.values
            G = delta + cfg.gamma * cfg.gae_lambda * mask * G
            advantages.insert(0, G)

        # 텐서 변환
        def stack(attr):
            return np.stack([getattr(tr, attr) for tr in self.buffer.transitions])

        obs_np   = stack("obs")       # [T, N, obs_dim]
        type_np  = stack("type_ids")  # [T, N]
        acts_np  = stack("actions")   # [T, N]
        olp_np   = stack("log_probs") # [T, N]
        rets_np  = stack("values") + np.array(advantages)  # [T, N]
        adv_np   = np.array(advantages)  # [T, N]

        # 정규화
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        # 마스크 처리
        first_mask = self.buffer.transitions[0].action_masks
        has_masks  = first_mask is not None
        masks_np   = stack("action_masks") if has_masks else None

        TN = T  # 배치 크기는 T (각 스텝이 하나의 배치 원소)

        obs_t   = torch.tensor(obs_np,  dtype=torch.float32, device=self.device)
        type_t  = torch.tensor(type_np, dtype=torch.long,    device=self.device)
        acts_t  = torch.tensor(acts_np, dtype=torch.long,    device=self.device)
        olp_t   = torch.tensor(olp_np,  dtype=torch.float32, device=self.device)
        rets_t  = torch.tensor(rets_np, dtype=torch.float32, device=self.device)
        adv_t   = torch.tensor(adv_np,  dtype=torch.float32, device=self.device)
        masks_t = (torch.tensor(masks_np, dtype=torch.bool, device=self.device)
                   if has_masks else None)

        # teacher-forcing prev_actions: [T, N] (BOS=0, 이후 실제 행동+1)
        prev_acts_t = torch.zeros_like(acts_t)
        prev_acts_t[:, 1:] = acts_t[:, :-1] + 1   # 시프트

        policy_losses, value_losses, entropy_vals = [], [], []
        for _ in range(cfg.n_epochs):
            idx = torch.randperm(TN)
            for start in range(0, TN, cfg.batch_size):
                b = idx[start:start + cfg.batch_size]
                logits, values_pred = self.policy(
                    obs_t[b], type_t[b], prev_acts_t[b]
                )   # [B, N, n_actions], [B, N]

                if masks_t is not None:
                    logits = logits.masked_fill(~masks_t[b], -1e9)

                dist  = torch.distributions.Categorical(logits=logits)
                new_lp= dist.log_prob(acts_t[b])   # [B, N]
                entropy = dist.entropy()            # [B, N]

                ratio = (new_lp - olp_t[b]).exp()
                adv_b = adv_t[b]
                surr = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                ).mean()
                vl = F.mse_loss(values_pred, rets_t[b])
                loss = surr + cfg.value_coef * vl - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                policy_losses.append(surr.item())
                value_losses.append(vl.item())
                entropy_vals.append(entropy.mean().item())

        self.buffer.clear()
        self.episode_count += 1
        return {
            "mat_policy_loss": float(np.mean(policy_losses)),
            "mat_value_loss":  float(np.mean(value_losses)),
            "mat_entropy":     float(np.mean(entropy_vals)),
        }


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Multi-Agent Transformer (MAT) 테스트 ===")

    cfg = MATConfig(obs_dim=24, n_actions=8, d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=2)
    trainer = MATTrainer(cfg, device="cpu")

    n_params = sum(p.numel() for p in trainer.policy.parameters())
    print(f"MAT 파라미터 수: {n_params:,}")

    # 더미 에피소드 실행
    N = 6   # 에이전트 6명
    rng = np.random.RandomState(42)

    print("\n[에피소드 실행 (T=20 스텝)]")
    for step in range(20):
        obs      = rng.randn(N, 24).astype(np.float32)
        type_ids = rng.randint(0, 8, size=N)
        masks    = rng.rand(N, 8) > 0.3   # 일부 행동 차단

        actions, log_probs, values = trainer.select_actions_np(obs, type_ids, masks)
        rewards = rng.randn(N).astype(np.float32)
        dones   = np.zeros(N, dtype=np.float32)
        if step == 19:
            dones[:] = 1.0

        trainer.buffer.add(MATTransition(
            obs=obs, type_ids=type_ids,
            actions=actions, log_probs=log_probs, values=values,
            rewards=rewards, dones=dones, action_masks=masks,
        ))

    metrics = trainer.update()
    print(f"  MAT Policy Loss : {metrics.get('mat_policy_loss', 0):.4f}")
    print(f"  MAT Value Loss  : {metrics.get('mat_value_loss', 0):.4f}")
    print(f"  MAT Entropy     : {metrics.get('mat_entropy', 0):.4f}")

    # 추론 테스트 (자기회귀)
    print("\n[자기회귀 추론 테스트 (N=8 에이전트)]")
    obs8     = rng.randn(8, 24).astype(np.float32)
    type8    = rng.randint(0, 8, size=8)
    a, lp, v = trainer.select_actions_np(obs8, type8)
    for i in range(8):
        tname = UNIT_TYPES[type8[i]]
        print(f"  agent {i} ({tname:18s}) → action={a[i]}  lp={lp[i]:.3f}  val={v[i]:.3f}")

    print("\n✅ mat_policy.py 정상 동작!")
