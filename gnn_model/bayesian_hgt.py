"""
bayesian_hgt.py
===============
Phase 1: Uncertainty-Aware GNN
MC Dropout 기반 Bayesian 이종 그래프 변환기 (HGT)
- 단일 숫자 예측 → 확률 분포 (평균 + 분산) 출력
- Epistemic / Aleatoric 불확실성 분리
- PPO 상태 벡터에 불확실성 정보 포함
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Graph Attention Layer (수동 구현 - 외부 의존 없음)
# ──────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    """
    단순화된 Graph Attention Layer
    PyTorch Geometric 없이 순수 PyTorch로 구현
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim
        self.dropout = dropout

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

        # 어텐션 가중치 저장 (Explainability용)
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_dim] 노드 특성
        adj: [N, N] 인접 행렬
        """
        N = x.size(0)

        Q = self.W_q(x).view(N, self.n_heads, self.head_dim)
        K = self.W_k(x).view(N, self.n_heads, self.head_dim)
        V = self.W_v(x).view(N, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = torch.einsum('ihd,jhd->ijh', Q, K) / scale  # [N, N, heads]

        # 마스킹 (인접하지 않은 노드 제거)
        if adj is not None:
            mask = (adj == 0).unsqueeze(-1).expand_as(attn)
            attn = attn.masked_fill(mask, -1e9)

        attn_weights = F.softmax(attn, dim=1)   # [N, N, heads]
        attn_weights = self.drop(attn_weights)

        # 어텐션 가중치 저장
        self.last_attention_weights = attn_weights.detach()

        # 어텐션 집계
        out = torch.einsum('ijh,jhd->ihd', attn_weights, V)  # [N, heads, head_dim]
        out = out.reshape(N, self.out_dim)
        out = self.W_o(out)
        out = self.norm(out + x if x.size(-1) == out.size(-1) else out)
        return out


# ──────────────────────────────────────────────
# Bayesian HGT-GNN (MC Dropout)
# ──────────────────────────────────────────────

class BayesianHGT(nn.Module):
    """
    MC Dropout 기반 Bayesian Heterogeneous Graph Transformer
    
    출력:
    - casualty_mean:     예측 사상자 수 (평균)
    - casualty_log_var:  예측 사상자 수 (로그 분산) → Aleatoric uncertainty
    - risk_mean:         위험 점수 [0, 1] (평균)
    - risk_log_var:      위험 점수 (로그 분산)
    
    MC Dropout으로 N번 forward → Epistemic uncertainty 계산
    """

    def __init__(
        self,
        node_in_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        mc_samples: int = 20
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout

        # 입력 임베딩
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 그래프 어텐션 레이어 스택
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 피드포워드 레이어
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(n_layers)
        ])

        # 그래프 풀링
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 예측 헤드 (평균 + 로그 분산)
        head_dim = hidden_dim // 2
        # 사상자 예측 헤드
        self.casualty_head_mean = nn.Sequential(
            nn.Linear(head_dim, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.casualty_head_logvar = nn.Sequential(
            nn.Linear(head_dim, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # 위험 점수 헤드
        self.risk_head_mean = nn.Sequential(
            nn.Linear(head_dim, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.risk_head_logvar = nn.Sequential(
            nn.Linear(head_dim, 64), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def _single_forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """단일 forward pass"""
        h = self.input_proj(x)

        # 그래프 어텐션 레이어
        for gat, ff in zip(self.gat_layers, self.ff_layers):
            h_gat = gat(h, adj)
            h = h + ff(h_gat)

        # 풀링 (평균 풀링 + 최대 풀링 결합)
        h_mean = h.mean(dim=0, keepdim=True)   # [1, hidden]
        h_max  = h.max(dim=0).values.unsqueeze(0)  # [1, hidden]
        h_pool = ((h_mean + h_max) / 2)

        h_proj = self.pool_proj(h_pool)  # [1, head_dim]

        return {
            "casualty_mean":   F.softplus(self.casualty_head_mean(h_proj)).squeeze(),
            "casualty_logvar": self.casualty_head_logvar(h_proj).squeeze(),
            "risk_mean":       self.risk_head_mean(h_proj).squeeze(),
            "risk_logvar":     self.risk_head_logvar(h_proj).squeeze(),
        }

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """단일 예측 (훈련 시 사용)"""
        return self._single_forward(x, adj)

    def predict_with_uncertainty(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        MC Dropout으로 불확실성 정량화
        훈련 모드로 전환해 Dropout 활성화 후 복원
        """
        was_training = self.training
        self.train()  # Dropout 활성화

        casualty_means, casualty_logvars = [], []
        risk_means, risk_logvars = [], []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                out = self._single_forward(x, adj)
                casualty_means.append(out["casualty_mean"].unsqueeze(0))
                casualty_logvars.append(out["casualty_logvar"].unsqueeze(0))
                risk_means.append(out["risk_mean"].unsqueeze(0))
                risk_logvars.append(out["risk_logvar"].unsqueeze(0))

        self.train(was_training)  # 호출 전 모드로 복원

        cas_means_t = torch.stack(casualty_means)  # [S, 1]
        risk_means_t = torch.stack(risk_means)

        # Epistemic uncertainty: 예측 평균의 분산 (모델 불확실성)
        epistemic_cas  = cas_means_t.var(dim=0)
        epistemic_risk = risk_means_t.var(dim=0)

        # Aleatoric uncertainty: 예측 로그 분산의 평균 (데이터 노이즈)
        cas_logvars_t = torch.stack(casualty_logvars)
        risk_logvars_t = torch.stack(risk_logvars)
        aleatoric_cas  = torch.exp(torch.clamp(cas_logvars_t, min=-10.0, max=10.0)).mean(dim=0)
        aleatoric_risk = torch.exp(torch.clamp(risk_logvars_t, min=-10.0, max=10.0)).mean(dim=0)

        return {
            # 예측값
            "casualty_mean":       torch.clamp(cas_means_t.mean(dim=0), min=0.0),
            "casualty_std":        cas_means_t.std(dim=0),
            "risk_mean":           risk_means_t.mean(dim=0),
            "risk_std":            risk_means_t.std(dim=0),
            # 불확실성 분해
            "epistemic_uncertainty": (epistemic_cas + epistemic_risk) / 2,
            "aleatoric_uncertainty": (aleatoric_cas + aleatoric_risk) / 2,
            "total_uncertainty":     ((epistemic_cas + aleatoric_cas) +
                                      (epistemic_risk + aleatoric_risk)) / 2,
            # 95% 신뢰구간
            "casualty_ci_low":  torch.clamp(cas_means_t.mean(dim=0) - 1.96 * cas_means_t.std(dim=0), min=0.0),
            "casualty_ci_high": torch.clamp(cas_means_t.mean(dim=0) + 1.96 * cas_means_t.std(dim=0), min=0.0),
        }

    def get_attention_weights(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """어텐션 가중치 반환 (Explainability용)"""
        if not self.gat_layers:
            return None
        layer = self.gat_layers[layer_idx]
        return layer.last_attention_weights

    def compute_ppo_state_extension(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> np.ndarray:
        """
        PPO 상태 벡터 확장 부분 반환
        [casualty_mean, casualty_std, risk_mean, risk_std,
         epistemic_uncertainty, aleatoric_uncertainty] → 6차원
        """
        out = self.predict_with_uncertainty(x, adj)
        return np.array([
            out["casualty_mean"].item(),
            out["casualty_std"].item(),
            out["risk_mean"].item(),
            out["risk_std"].item(),
            out["epistemic_uncertainty"].item(),
            out["aleatoric_uncertainty"].item(),
        ], dtype=np.float32)


# ──────────────────────────────────────────────
# Loss Function
# ──────────────────────────────────────────────

class NegativeLogLikelihoodLoss(nn.Module):
    """
    가우시안 NLL 손실 함수 (불확실성 학습)
    L = 0.5 * (log σ² + (y - μ)² / σ²)
    """

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var) + 1e-6
        loss = 0.5 * (log_var + (target - mean) ** 2 / var)
        return loss.mean()


class CombatGNNLoss(nn.Module):
    """결합 손실 함수 (사상자 + 위험도 + NLL)"""

    def __init__(self, casualty_weight: float = 1.0, risk_weight: float = 0.5):
        super().__init__()
        self.nll = NegativeLogLikelihoodLoss()
        self.casualty_weight = casualty_weight
        self.risk_weight = risk_weight

    def forward(self, pred: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        cas_loss  = self.nll(pred["casualty_mean"], pred["casualty_logvar"],
                              targets["blue_casualties"])
        risk_loss = self.nll(pred["risk_mean"], pred["risk_logvar"],
                              targets["risk_score"])
        return self.casualty_weight * cas_loss + self.risk_weight * risk_loss


# ──────────────────────────────────────────────
# Training Helper
# ──────────────────────────────────────────────

def build_adjacency_matrix(edge_index: np.ndarray, n_nodes: int) -> torch.Tensor:
    """엣지 인덱스 → 인접 행렬 변환"""
    adj = torch.zeros(n_nodes, n_nodes)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    # 자기 연결 추가
    adj += torch.eye(n_nodes)
    return adj


def prepare_graph_tensors(kg) -> Tuple[torch.Tensor, torch.Tensor]:
    """KG → (node_features, adjacency_matrix) 텐서 변환"""
    node_feats = kg.get_node_features()
    edge_idx, _ = kg.get_adjacency_info()
    n_nodes = len(node_feats)

    x = torch.tensor(node_feats, dtype=torch.float32)
    adj = build_adjacency_matrix(edge_idx, n_nodes)
    return x, adj


if __name__ == "__main__":
    print("=== Bayesian HGT-GNN Test ===")
    from ontology.combat_schema import ScenarioFactory

    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    x, adj = prepare_graph_tensors(kg)
    print(f"입력 노드 특성: {x.shape}, 인접 행렬: {adj.shape}")

    model = BayesianHGT(node_in_dim=32, hidden_dim=128, n_layers=2, mc_samples=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {n_params:,}")

    # 단일 예측
    model.train()
    out = model(x, adj)
    print(f"\n단일 예측:")
    print(f"  사상자 평균: {out['casualty_mean'].item():.2f}")
    print(f"  위험도 평균: {out['risk_mean'].item():.4f}")

    # 불확실성 정량화
    unc_out = model.predict_with_uncertainty(x, adj)
    print(f"\nMC Dropout 예측 (N={model.mc_samples}):")
    print(f"  사상자: {unc_out['casualty_mean'].item():.2f} ± {unc_out['casualty_std'].item():.2f}")
    print(f"  95% CI: [{unc_out['casualty_ci_low'].item():.1f}, {unc_out['casualty_ci_high'].item():.1f}]")
    print(f"  Epistemic 불확실성: {unc_out['epistemic_uncertainty'].item():.4f}")
    print(f"  Aleatoric 불확실성: {unc_out['aleatoric_uncertainty'].item():.4f}")

    # PPO 상태 확장 벡터
    state_ext = model.compute_ppo_state_extension(x, adj)
    print(f"\nPPO 상태 확장 벡터 (6D): {state_ext}")

    # 어텐션 가중치
    _ = model(x, adj)
    attn = model.get_attention_weights()
    if attn is not None:
        print(f"\n어텐션 가중치 형태: {attn.shape}")
        print("✅ Bayesian HGT-GNN 테스트 완료!")
