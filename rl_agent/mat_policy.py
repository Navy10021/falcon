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
    # legacy aliases
    n_agents: Optional[int] = None
    n_layers: Optional[int] = None

    def __post_init__(self):
        if self.n_layers is not None:
            self.n_enc_layers = int(self.n_layers)
            self.n_dec_layers = int(self.n_layers)
        if self.n_agents is not None:
            self.max_agents = max(int(self.n_agents), int(self.max_agents))


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
        type_ids: Optional[torch.Tensor] = None,         # [B, N]
        prev_actions: Optional[torch.Tensor] = None,     # [B, N]  (teacher-forcing)
        pad_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_agents = obs.shape[:2]
        if type_ids is None:
            type_ids = torch.full((bsz, n_agents), N_UNIT_TYPES, dtype=torch.long, device=obs.device)
        if prev_actions is None:
            prev_actions = torch.zeros((bsz, n_agents), dtype=torch.long, device=obs.device)
        memory = self.encoder(obs, type_ids, key_padding_mask=pad_mask)
        logits, values = self.decoder(memory, prev_actions, memory_key_padding_mask=pad_mask)
        return logits, values   # [B, N, n_actions], [B, N]

    @torch.no_grad()
    def select_actions(
        self,
        obs: torch.Tensor,          # [B, N, obs_dim]
        type_ids: Optional[torch.Tensor] = None,     # [B, N]
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
        legacy_call = type_ids is None
        if type_ids is None:
            type_ids = torch.full((B, N), N_UNIT_TYPES, dtype=torch.long, device=obs.device)
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

        if legacy_call:
            return actions[0].detach().cpu().tolist()  # type: ignore[return-value]
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
        """버퍼 데이터로 MAT PPO 업데이트.

        에피소드 내 에이전트 수(N)가 가변적인 경우를 안전하게 처리:
          - N이 일정한 연속 세그먼트에서만 GAE 부트스트랩 적용
          - N이 달라지거나 done=True 경계에서 G를 0으로 리셋
        """
        if len(self.buffer) == 0:
            return {}

        T = len(self.buffer)
        cfg = self.cfg

        # ── GAE 이득 계산 (가변 N 안전 처리) ──────────────────────────
        advantages_list: List[np.ndarray] = [None] * T  # type: ignore[list-item]
        G = np.zeros(1, dtype=np.float32)

        for t in reversed(range(T)):
            tr   = self.buffer.transitions[t]
            N_t  = len(tr.values)
            mask = 1.0 - tr.dones.astype(np.float32)  # [N_t]

            # 부트스트랩 값: 다음 스텝이 없거나 N이 달라지면 0으로 리셋
            if t == T - 1:
                nv = next_values if (next_values is not None and len(next_values) == N_t) \
                     else np.zeros(N_t, dtype=np.float32)
            else:
                tr_next = self.buffer.transitions[t + 1]
                if len(tr_next.values) == N_t:
                    nv = tr_next.values
                else:
                    nv = np.zeros(N_t, dtype=np.float32)  # N 불일치: G 리셋

            # G 크기가 N_t와 다르면 리셋
            if G.shape != (N_t,):
                G = np.zeros(N_t, dtype=np.float32)

            delta = tr.rewards + cfg.gamma * nv * mask - tr.values
            G = delta + cfg.gamma * cfg.gae_lambda * mask * G
            advantages_list[t] = G.copy()

        # ── 텐서 변환 (가변 N → 평탄화) ──────────────────────────────
        # 각 스텝을 독립적인 배치 원소로 처리 (step-wise batching)
        obs_list   = [tr.obs       for tr in self.buffer.transitions]  # list of [N_t, obs_dim]
        type_list  = [tr.type_ids  for tr in self.buffer.transitions]  # list of [N_t]
        act_list   = [tr.actions   for tr in self.buffer.transitions]
        olp_list   = [tr.log_probs for tr in self.buffer.transitions]
        val_list   = [tr.values    for tr in self.buffer.transitions]
        adv_list   = advantages_list
        mask_list  = [tr.action_masks for tr in self.buffer.transitions]

        has_masks = mask_list[0] is not None

        # 모든 N이 동일하면 np.stack, 아니면 각 step을 개별 처리
        all_n = [tr.obs.shape[0] for tr in self.buffer.transitions]
        uniform_n = len(set(all_n)) == 1

        if uniform_n:
            obs_np  = np.stack(obs_list)      # [T, N, obs_dim]
            type_np = np.stack(type_list)     # [T, N]
            acts_np = np.stack(act_list)
            olp_np  = np.stack(olp_list)
            val_np  = np.stack(val_list)
            adv_np  = np.stack(adv_list)
            masks_np= np.stack(mask_list) if has_masks else None
            rets_np = val_np + adv_np
        else:
            # 가변 N: 에이전트 차원을 평탄화하여 [T*N_avg, ...] 형태로 변환
            # 각 스텝은 크기가 다를 수 있으므로 스텝을 배치로 처리
            # 편의상 패딩하여 max_N에 맞춤
            max_n = max(all_n)
            obs_dim = obs_list[0].shape[1]
            n_act   = act_list[0].shape[0]

            obs_np   = np.zeros((T, max_n, obs_dim), dtype=np.float32)
            type_np  = np.zeros((T, max_n), dtype=np.int64)
            acts_np  = np.zeros((T, max_n), dtype=np.int64)
            olp_np   = np.zeros((T, max_n), dtype=np.float32)
            val_np   = np.zeros((T, max_n), dtype=np.float32)
            adv_np   = np.zeros((T, max_n), dtype=np.float32)
            rets_np  = np.zeros((T, max_n), dtype=np.float32)
            pad_mask_np = np.ones((T, max_n), dtype=bool)   # True=pad
            masks_np = np.zeros((T, max_n, self.cfg.n_actions), dtype=bool) if has_masks else None

            for t, tr in enumerate(self.buffer.transitions):
                n = all_n[t]
                obs_np[t, :n]    = obs_list[t]
                type_np[t, :n]   = type_list[t]
                acts_np[t, :n]   = act_list[t]
                olp_np[t, :n]    = olp_list[t]
                val_np[t, :n]    = val_list[t]
                adv_np[t, :n]    = adv_list[t]
                rets_np[t, :n]   = val_list[t] + adv_list[t]
                pad_mask_np[t, :n] = False
                if has_masks and mask_list[t] is not None:
                    masks_np[t, :n] = mask_list[t]

        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        TN = T  # 배치 크기는 T

        obs_t   = torch.tensor(obs_np,  dtype=torch.float32, device=self.device)
        type_t  = torch.tensor(type_np, dtype=torch.long,    device=self.device)
        acts_t  = torch.tensor(acts_np, dtype=torch.long,    device=self.device)
        olp_t   = torch.tensor(olp_np,  dtype=torch.float32, device=self.device)
        rets_t  = torch.tensor(rets_np, dtype=torch.float32, device=self.device)
        adv_t   = torch.tensor(adv_np,  dtype=torch.float32, device=self.device)
        masks_t = (torch.tensor(masks_np, dtype=torch.bool, device=self.device)
                   if has_masks else None)

        # 가변 N 패딩 마스크 (True=유효 에이전트)
        valid_t = None
        if not uniform_n:
            valid_t = torch.tensor(~pad_mask_np, dtype=torch.bool, device=self.device)  # [T, N_max]

        # teacher-forcing prev_actions: [T, N] (BOS=0, 이후 실제 행동+1)
        prev_acts_t = torch.zeros_like(acts_t)
        prev_acts_t[:, 1:] = acts_t[:, :-1] + 1   # 시프트

        # Transformer 인코더에 전달할 패딩 마스크 [T, N_max] (True=패드)
        enc_pad_mask_t = None if uniform_n else torch.tensor(pad_mask_np, dtype=torch.bool, device=self.device)

        policy_losses, value_losses, entropy_vals = [], [], []
        for _ in range(cfg.n_epochs):
            idx = torch.randperm(TN)
            for start in range(0, TN, cfg.batch_size):
                b = idx[start:start + cfg.batch_size]
                pad_b = enc_pad_mask_t[b] if enc_pad_mask_t is not None else None

                logits, values_pred = self.policy(
                    obs_t[b], type_t[b], prev_acts_t[b], pad_mask=pad_b
                )   # [B, N, n_actions], [B, N]

                if masks_t is not None:
                    logits = logits.masked_fill(~masks_t[b], -1e9)

                dist  = torch.distributions.Categorical(logits=logits)
                new_lp= dist.log_prob(acts_t[b])   # [B, N]
                entropy = dist.entropy()            # [B, N]

                # 유효 에이전트만 손실 계산
                if valid_t is not None:
                    valid_b = valid_t[b]  # [B, N]
                    new_lp  = new_lp[valid_b]
                    entropy = entropy[valid_b]
                    values_pred = values_pred[valid_b]
                    rets_b  = rets_t[b][valid_b]
                    adv_b   = adv_t[b][valid_b]
                    olp_b   = olp_t[b][valid_b]
                else:
                    rets_b  = rets_t[b]
                    adv_b   = adv_t[b]
                    olp_b   = olp_t[b]

                ratio = (new_lp - olp_b).exp()
                surr = -torch.min(
                    ratio * adv_b,
                    ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                ).mean()
                vl = F.mse_loss(values_pred, rets_b)
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
