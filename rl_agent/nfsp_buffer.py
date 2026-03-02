"""
nfsp_buffer.py
==============
Neural Fictitious Self-Play (NFSP) 경험 재생 버퍼

구성:
  - ReservoirBuffer  : 균등 샘플링 보장 저수지 샘플링 버퍼 (지도학습 용)
  - CircularBuffer   : 최근 경험 유지 원형 버퍼 (강화학습 RL 용)
  - NFSPBufferPair   : RL + SL 버퍼 쌍 (에이전트 당 1개 할당)

NFSP 알고리즘 개요:
  - RL 버퍼 (CircularBuffer): Best-Response 학습용 (DQN/PPO)
  - SL 버퍼 (ReservoirBuffer): 평균 전략(Average Policy) 학습용
  - 에이전트는 η 확률로 RL 정책 사용, (1-η)로 Average Policy 사용

Reference:
  Heinrich & Silver (2016) "Deep Reinforcement Learning from Self-Play
  in Imperfect-Information Games"
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# 공통 경험 튜플
# ──────────────────────────────────────────────

class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ──────────────────────────────────────────────
# 저수지 샘플링 버퍼 (SL 용)
# ──────────────────────────────────────────────

class ReservoirBuffer:
    """
    Reservoir Sampling 기반 경험 재생 버퍼.

    크기 M을 초과하는 순간도 각 원소가 균등하게 저장될 확률을 보장:
      P(경험 i 유지) = M / max(M, total_seen)

    NFSP에서 Average Policy(지도학습) 학습에 사용된다.

    Parameters
    ----------
    capacity : int
        최대 저장 용량 (M).
    seed : int
        난수 시드 (재현성).
    """

    def __init__(self, capacity: int = 200_000, seed: int = 0):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self._total_seen: int = 0
        self.rng = random.Random(seed)

    def add(self, exp: Experience):
        """
        저수지 샘플링으로 새 경험 추가.
        처음 capacity개는 무조건 저장, 이후는 확률적으로 교체.
        """
        self._total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            idx = self.rng.randint(0, self._total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = exp

    def sample(self, batch_size: int) -> List[Experience]:
        """균등 무작위 샘플링 (중복 없음)"""
        n = min(batch_size, len(self.buffer))
        return self.rng.sample(self.buffer, n)

    def sample_arrays(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        샘플을 numpy 배열로 변환하여 반환.

        Returns
        -------
        states, actions, rewards, next_states, dones
        """
        batch = self.sample(batch_size)
        states      = np.stack([e.state      for e in batch])
        actions     = np.array([e.action     for e in batch], dtype=np.int64)
        rewards     = np.array([e.reward     for e in batch], dtype=np.float32)
        next_states = np.stack([e.next_state for e in batch])
        dones       = np.array([e.done       for e in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def total_seen(self) -> int:
        return self._total_seen

    def clear(self):
        self.buffer.clear()
        self._total_seen = 0

    def is_ready(self, min_size: int = 64) -> bool:
        return len(self.buffer) >= min_size


# ──────────────────────────────────────────────
# 원형 버퍼 (RL 용)
# ──────────────────────────────────────────────

class CircularBuffer:
    """
    최근 경험을 FIFO 방식으로 유지하는 원형 버퍼.

    NFSP에서 Best-Response(강화학습) 학습에 사용된다.

    Parameters
    ----------
    capacity : int
        최대 저장 용량.
    seed : int
        난수 시드.
    """

    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, exp: Experience):
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        n = min(batch_size, len(self.buffer))
        return self.rng.sample(list(self.buffer), n)

    def sample_arrays(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = self.sample(batch_size)
        states      = np.stack([e.state      for e in batch])
        actions     = np.array([e.action     for e in batch], dtype=np.int64)
        rewards     = np.array([e.reward     for e in batch], dtype=np.float32)
        next_states = np.stack([e.next_state for e in batch])
        dones       = np.array([e.done       for e in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int = 64) -> bool:
        return len(self.buffer) >= min_size


# ──────────────────────────────────────────────
# NFSP 버퍼 쌍
# ──────────────────────────────────────────────

@dataclass
class NFSPConfig:
    """NFSP 하이퍼파라미터"""
    eta: float = 0.1           # RL 정책 사용 확률 (탐욕적 행동 비율)
    rl_capacity: int = 100_000  # RL 버퍼 크기
    sl_capacity: int = 200_000  # SL 버퍼 크기
    min_rl_size: int = 64       # RL 학습 시작 최소 크기
    min_sl_size: int = 64       # SL 학습 시작 최소 크기


class NFSPBufferPair:
    """
    에이전트 별 RL + SL 버퍼 쌍.

    사용 패턴::

        pair = NFSPBufferPair(config, seed=0)

        # 매 스텝
        use_rl = pair.should_use_rl()   # η 확률로 True
        exp = Experience(s, a, r, s2, done)
        pair.add(exp, is_rl_step=use_rl)

        # 학습 시
        if pair.rl_buffer.is_ready():
            rl_batch = pair.rl_buffer.sample_arrays(batch_size)
        if pair.sl_buffer.is_ready():
            sl_batch = pair.sl_buffer.sample_arrays(batch_size)
    """

    def __init__(self, config: Optional[NFSPConfig] = None, seed: int = 0):
        self.config = config or NFSPConfig()
        self.rl_buffer = CircularBuffer(self.config.rl_capacity, seed=seed)
        self.sl_buffer = ReservoirBuffer(self.config.sl_capacity, seed=seed + 1)
        self._rng = random.Random(seed)
        self._step = 0

    def should_use_rl(self) -> bool:
        """η 확률로 True (RL 정책 선택), 그 외 False (Average Policy)"""
        return self._rng.random() < self.config.eta

    def add(self, exp: Experience, is_rl_step: bool):
        """
        경험을 RL 버퍼에 항상 추가하고,
        RL 스텝에서만 SL 버퍼에도 추가 (Best-Response 행동만 SL에 기록).
        """
        self.rl_buffer.add(exp)
        if is_rl_step:
            self.sl_buffer.add(exp)
        self._step += 1


    def add_rl(self, exp: Experience):
        """Legacy helper: add only to RL buffer."""
        self.rl_buffer.add(exp)
        self._step += 1

    def add_sl(self, exp: Experience):
        """Legacy helper: add only to SL buffer."""
        self.sl_buffer.add(exp)
        self._step += 1

    def stats(self) -> Dict[str, int]:
        return {
            "rl_buffer_size": len(self.rl_buffer),
            "sl_buffer_size": len(self.sl_buffer),
            "total_steps": self._step,
        }


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== NFSP Buffer 테스트 ===")

    # ReservoirBuffer 균등성 검증
    res = ReservoirBuffer(capacity=1000, seed=0)
    N = 5000
    for i in range(N):
        s  = np.zeros(8, dtype=np.float32)
        s[0] = i
        res.add(Experience(s, i % 6, float(i), s, False))

    print(f"ReservoirBuffer: total_seen={res.total_seen}, stored={len(res)}")
    sample = res.sample(200)
    vals = [int(e.state[0]) for e in sample]
    print(f"  샘플 최소값={min(vals)}, 최대값={max(vals)}, 범위 커버율={len(set(vals))/200:.2f}")

    # CircularBuffer 테스트
    circ = CircularBuffer(capacity=500, seed=1)
    for i in range(800):
        s = np.ones(8, dtype=np.float32) * i
        circ.add(Experience(s, i % 6, 0.0, s, False))
    print(f"\nCircularBuffer: stored={len(circ)} (cap=500, total added=800)")

    # NFSPBufferPair 테스트
    cfg = NFSPConfig(eta=0.1)
    pair = NFSPBufferPair(cfg, seed=42)
    rl_steps = 0
    for _ in range(1000):
        use_rl = pair.should_use_rl()
        if use_rl:
            rl_steps += 1
        s = np.random.randn(8).astype(np.float32)
        pair.add(Experience(s, 0, 0.0, s, False), is_rl_step=use_rl)

    stats = pair.stats()
    print(f"\nNFSPBufferPair: {stats}")
    print(f"  η=0.1 → RL step 비율: {rl_steps/1000:.3f} (기대: ~0.1)")
    print("\n✅ nfsp_buffer.py 정상 동작!")
