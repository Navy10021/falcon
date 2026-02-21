"""
temporal_gnn.py
===============
⑤ Temporal GNN: LSTM + GNN 결합
- GNN이 각 타임스텝의 그래프 임베딩 생성
- LSTM이 임베딩 시계열을 처리해 시간 의존성 학습
- 사상자 예측에 역사적 전투 흐름 반영
- 슬라이딩 윈도우 시퀀스 입력

아키텍처:
  [t-W, ..., t-1, t] 그래프 시퀀스
          ↓ (각 스텝)
      GNN Encoder → 노드 임베딩 → 그래프 임베딩 벡터
          ↓ (시퀀스)
      LSTM (window_size steps)
          ↓
      예측 헤드 (사상자, 위험도, 임무완료 확률)

학술 연구용 합성 데이터
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────
# Numpy-only fallback (torch 없는 환경)
# ──────────────────────────────────────────────

class NumpyGraphEncoder:
    """
    PyTorch 없이 동작하는 경량 그래프 인코더
    선형 변환 + 평균 풀링
    """

    def __init__(self, node_in_dim: int = 32, embed_dim: int = 64, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(node_in_dim, embed_dim).astype(np.float32) * 0.1
        self.b = np.zeros(embed_dim, dtype=np.float32)
        self.embed_dim = embed_dim

    def encode(self, node_features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        node_features: [N, node_in_dim]
        adj:           [N, N]
        Returns: [embed_dim] 그래프 임베딩
        """
        # 그래프 어텐션 (단순 정규화 인접행렬 기반)
        deg = adj.sum(axis=1, keepdims=True) + 1e-8
        agg = (adj / deg) @ node_features  # 이웃 평균 집계

        # 선형 변환 + ReLU
        h = np.maximum(0, agg @ self.W + self.b)  # [N, embed_dim]

        # 평균 + 최대 풀링 결합
        mean_pool = h.mean(axis=0)
        max_pool  = h.max(axis=0)
        return (mean_pool + max_pool) / 2  # [embed_dim]


class NumpyLSTMCell:
    """경량 LSTM 셀 (numpy 구현)"""

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim
        scale = 0.1

        # 입력 게이트, 망각 게이트, 셀 게이트, 출력 게이트
        total_in = input_dim + hidden_dim
        self.Wf = rng.randn(total_in, hidden_dim).astype(np.float32) * scale
        self.Wi = rng.randn(total_in, hidden_dim).astype(np.float32) * scale
        self.Wc = rng.randn(total_in, hidden_dim).astype(np.float32) * scale
        self.Wo = rng.randn(total_in, hidden_dim).astype(np.float32) * scale

        self.bf = np.ones(hidden_dim, dtype=np.float32)   # 망각 게이트 편향=1 (초기)
        self.bi = np.zeros(hidden_dim, dtype=np.float32)
        self.bc = np.zeros(hidden_dim, dtype=np.float32)
        self.bo = np.zeros(hidden_dim, dtype=np.float32)

    def step(
        self, x: np.ndarray, h: np.ndarray, c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """단일 LSTM 스텝"""
        xh = np.concatenate([x, h])
        f  = self._sigmoid(xh @ self.Wf + self.bf)
        i  = self._sigmoid(xh @ self.Wi + self.bi)
        g  = np.tanh(xh @ self.Wc + self.bc)
        o  = self._sigmoid(xh @ self.Wo + self.bo)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    @staticmethod
    def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


class NumpyTemporalGNN:
    """
    PyTorch 없는 Temporal GNN (numpy 기반)
    GNN Encoder + LSTM 시계열 처리
    """

    def __init__(
        self,
        node_in_dim: int = 32,
        gnn_embed_dim: int = 64,
        lstm_hidden_dim: int = 128,
        window_size: int = 8,
        seed: int = 0
    ):
        rng = np.random.RandomState(seed)
        self.window_size   = window_size
        self.lstm_hidden   = lstm_hidden_dim
        self.gnn_embed_dim = gnn_embed_dim

        # 모듈 초기화
        self.gnn_encoder = NumpyGraphEncoder(node_in_dim, gnn_embed_dim, seed)
        self.lstm_cell   = NumpyLSTMCell(gnn_embed_dim, lstm_hidden_dim, seed + 1)

        # 예측 헤드 (선형)
        self.W_cas  = rng.randn(lstm_hidden_dim, 1).astype(np.float32) * 0.1
        self.W_risk = rng.randn(lstm_hidden_dim, 1).astype(np.float32) * 0.1
        self.W_done = rng.randn(lstm_hidden_dim, 1).astype(np.float32) * 0.1

        # 슬라이딩 윈도우 버퍼
        self.embedding_buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self.h = np.zeros(lstm_hidden_dim, dtype=np.float32)
        self.c = np.zeros(lstm_hidden_dim, dtype=np.float32)

    def encode_graph(self, node_features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """현재 타임스텝 그래프 임베딩"""
        return self.gnn_encoder.encode(node_features, adj)

    def update_sequence(self, graph_embedding: np.ndarray):
        """슬라이딩 윈도우 업데이트 + LSTM 상태 갱신"""
        self.embedding_buffer.append(graph_embedding)
        self.h, self.c = self.lstm_cell.step(graph_embedding, self.h, self.c)

    def predict(self) -> Dict[str, float]:
        """
        현재 LSTM 상태로 예측
        Returns: 사상자, 위험도, 임무완료 확률
        """
        h = self.h
        casualty_pred = float((h @ self.W_cas).squeeze())
        risk_pred     = float(1 / (1 + np.exp(-(h @ self.W_risk).squeeze())))  # sigmoid
        done_pred     = float(1 / (1 + np.exp(-(h @ self.W_done).squeeze())))  # sigmoid

        # 불확실성: 버퍼가 찰수록 낮아짐
        buffer_ratio = len(self.embedding_buffer) / self.window_size
        uncertainty  = 1.0 - buffer_ratio * 0.7

        return {
            "casualty_mean":   abs(casualty_pred) * 20,
            "risk_mean":       risk_pred,
            "mission_done_prob": done_pred,
            "temporal_uncertainty": uncertainty,
            "buffer_fill": len(self.embedding_buffer),
            "window_size":   self.window_size,
        }

    def step(self, node_features: np.ndarray, adj: np.ndarray) -> Dict[str, float]:
        """그래프 입력 → 임베딩 → LSTM 갱신 → 예측 (원스톱)"""
        emb = self.encode_graph(node_features, adj)
        self.update_sequence(emb)
        return self.predict()

    def reset_state(self):
        """에피소드 시작 시 LSTM 상태 초기화"""
        self.h = np.zeros(self.lstm_hidden, dtype=np.float32)
        self.c = np.zeros(self.lstm_hidden, dtype=np.float32)
        self.embedding_buffer.clear()

    def get_temporal_state_vector(self) -> np.ndarray:
        """
        PPO 상태 벡터 확장용 (8차원)
        [casualty_mean, risk_mean, mission_done_prob, temporal_uncertainty,
         buffer_fill_ratio, h_mean, h_std, trend]
        """
        pred = self.predict()
        h = self.h

        # 임베딩 트렌드 (버퍼 처음 vs 마지막)
        if len(self.embedding_buffer) >= 2:
            buf_list = list(self.embedding_buffer)
            trend = float(np.mean(buf_list[-1] - buf_list[0]))
        else:
            trend = 0.0

        return np.array([
            min(pred["casualty_mean"] / 100, 1.0),
            pred["risk_mean"],
            pred["mission_done_prob"],
            pred["temporal_uncertainty"],
            pred["buffer_fill"] / self.window_size,
            float(np.mean(np.abs(h))),
            float(np.std(h)),
            np.clip(trend, -1, 1),
        ], dtype=np.float32)


# ──────────────────────────────────────────────
# PyTorch 버전 (torch 있을 때)
# ──────────────────────────────────────────────

if TORCH_AVAILABLE:
    class TemporalGNNTorch(nn.Module):
        """
        PyTorch 기반 Temporal GNN
        GNN → LSTM → 예측 헤드
        """

        def __init__(
            self,
            node_in_dim: int = 32,
            gnn_embed_dim: int = 64,
            lstm_hidden_dim: int = 128,
            window_size: int = 8,
            n_gnn_layers: int = 2,
            dropout: float = 0.2
        ):
            super().__init__()
            self.window_size = window_size
            self.lstm_hidden = lstm_hidden_dim

            # GNN 인코더
            layers = [nn.Linear(node_in_dim, gnn_embed_dim), nn.GELU()]
            for _ in range(n_gnn_layers - 1):
                layers += [nn.Linear(gnn_embed_dim, gnn_embed_dim), nn.GELU()]
            self.gnn_encoder = nn.Sequential(*layers)
            self.pool_proj = nn.Linear(gnn_embed_dim * 2, gnn_embed_dim)

            # LSTM (시퀀스 처리)
            self.lstm = nn.LSTM(
                input_size=gnn_embed_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )

            # 예측 헤드들
            self.head_casualty = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 2)   # [mean, log_var]
            )
            self.head_risk = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )
            self.head_done = nn.Sequential(
                nn.Linear(lstm_hidden_dim, 32), nn.GELU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )

        def encode_graph(self, x: 'torch.Tensor', adj: 'torch.Tensor') -> 'torch.Tensor':
            """[N, D] → [embed_dim]"""
            h = self.gnn_encoder(x)       # [N, embed_dim]
            # 정규화 집계
            deg = adj.sum(dim=1, keepdim=True) + 1e-8
            h_agg = (adj / deg) @ h
            mean_p = h_agg.mean(dim=0)
            max_p  = h_agg.max(dim=0).values
            pooled = self.pool_proj(torch.cat([mean_p, max_p], dim=-1))
            return pooled  # [embed_dim]

        def forward(
            self,
            graph_sequence: 'List[Tuple[torch.Tensor, torch.Tensor]]'
        ) -> Dict[str, 'torch.Tensor']:
            """
            graph_sequence: list of (node_features, adj) per timestep
            """
            embeddings = []
            for x, adj in graph_sequence:
                emb = self.encode_graph(x, adj)
                embeddings.append(emb)

            seq = torch.stack(embeddings).unsqueeze(0)   # [1, T, embed_dim]
            lstm_out, _ = self.lstm(seq)
            h_last = lstm_out[:, -1, :]                  # [1, hidden]

            cas_out  = self.head_casualty(h_last)
            risk_out = self.head_risk(h_last)
            done_out = self.head_done(h_last)

            return {
                "casualty_mean":   cas_out[:, 0],
                "casualty_logvar": cas_out[:, 1],
                "risk_prob":       risk_out.squeeze(),
                "mission_done_prob": done_out.squeeze(),
            }


# ──────────────────────────────────────────────
# Temporal GNN 학습 루프 헬퍼
# ──────────────────────────────────────────────

@dataclass
class TemporalDataPoint:
    """Temporal GNN 학습 데이터 포인트"""
    graph_sequence: List[np.ndarray]   # 그래프 임베딩 시퀀스
    target_casualties: float
    target_risk: float
    target_done: bool


class TemporalDataCollector:
    """에피소드 실행 중 시퀀스 데이터 수집"""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.episode_embeddings: List[np.ndarray] = []
        self.episode_labels: List[Dict] = []

    def add_step(self, embedding: np.ndarray, label: Dict):
        self.episode_embeddings.append(embedding)
        self.episode_labels.append(label)

    def get_sequences(self) -> List[TemporalDataPoint]:
        """슬라이딩 윈도우로 학습 시퀀스 추출"""
        sequences = []
        W = self.window_size
        for t in range(W, len(self.episode_embeddings)):
            seq   = self.episode_embeddings[t-W:t]
            label = self.episode_labels[t]
            sequences.append(TemporalDataPoint(
                graph_sequence=seq,
                target_casualties=label.get("casualties", 0),
                target_risk=label.get("risk", 0.5),
                target_done=label.get("done", False),
            ))
        return sequences

    def reset(self):
        self.episode_embeddings.clear()
        self.episode_labels.clear()


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine

    print("=" * 55)
    print("⑤ Temporal GNN (LSTM + GNN) 테스트")
    print("=" * 55)

    tgnn = NumpyTemporalGNN(
        node_in_dim=32, gnn_embed_dim=64,
        lstm_hidden_dim=128, window_size=8, seed=42
    )

    engine = MixedLanchesterEngine(seed=42)
    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    collector = TemporalDataCollector(window_size=8)

    print("\n[15 스텝 시퀀스 수집 & 예측]")
    for step in range(15):
        node_feats = kg.get_node_features()
        edge_idx, edge_weights = kg.get_adjacency_info()
        n_nodes = len(node_feats)
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        if edge_idx.shape[1] > 0:
            adj[edge_idx[0], edge_idx[1]] = 1.0
        np.fill_diagonal(adj, 1.0)

        pred = tgnn.step(node_feats, adj)
        emb  = tgnn.gnn_encoder.encode(node_feats, adj)
        collector.add_step(emb, {
            "casualties": pred["casualty_mean"],
            "risk": pred["risk_mean"],
            "done": step > 10,
        })

        result = engine.run_step_mixed(kg)
        print(f"  스텝 {step+1:2d}: 사상자예측={pred['casualty_mean']:.1f}  "
              f"위험도={pred['risk_mean']:.3f}  "
              f"버퍼={pred['buffer_fill']}/{pred['window_size']}  "
              f"불확실성={pred['temporal_uncertainty']:.3f}")

    temporal_state = tgnn.get_temporal_state_vector()
    print(f"\n시계열 상태 벡터 (8D): {temporal_state.round(3)}")

    sequences = collector.get_sequences()
    print(f"\n수집된 학습 시퀀스: {len(sequences)}개")

    if TORCH_AVAILABLE:
        print("\n✅ PyTorch Temporal GNN 사용 가능")
    else:
        print("\nℹ️  PyTorch 미설치 → Numpy 폴백 모드")

    print("\n✅ Temporal GNN 정상 동작!")
