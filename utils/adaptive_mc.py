"""
adaptive_mc.py
==============
적응형 MC Dropout — 불확실성 수렴 시 조기 중단

GNN의 MC Dropout forward pass 횟수를 고정(15회)이 아닌
적응적으로 조절하여 대규모 시나리오에서의 성능 병목을 해소한다.

원리:
  - 불확실성의 이동 평균이 수렴하면 조기 중단
  - 최소 n_min 회 보장, 최대 n_max 회로 제한
  - 수렴 기준: 연속 patience 회의 표준편차 변화가 threshold 미만

사용 예시::

    amc = AdaptiveMCDropout(n_min=5, n_max=30, patience=3, threshold=0.01)
    mean, std = amc.run(forward_fn, input_data)

학술 연구용 합성 데이터
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class AdaptiveMCConfig:
    """적응형 MC Dropout 설정"""
    n_min: int = 5       # 최소 forward pass 횟수
    n_max: int = 30      # 최대 forward pass 횟수
    patience: int = 3    # 수렴 판정 연속 횟수
    threshold: float = 0.01  # 표준편차 변화 임계값
    warmup: int = 3      # 수렴 판정 시작 전 최소 실행 횟수


class AdaptiveMCDropout:
    """
    적응형 MC Dropout 실행기.

    forward_fn을 반복 호출하되, 불확실성이 수렴하면 조기 중단한다.
    """

    def __init__(self, config: Optional[AdaptiveMCConfig] = None):
        self.config = config or AdaptiveMCConfig()
        self._last_n_runs = 0
        self._last_converged = False

    def run(
        self,
        forward_fn: Callable[[], np.ndarray],
        aggregate: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        적응형 MC Dropout 실행.

        Parameters
        ----------
        forward_fn : Callable
            MC Dropout 활성 상태에서 forward pass를 수행하는 함수.
            호출 시마다 다른 dropout mask로 다른 결과를 반환해야 함.
        aggregate : str
            "mean" (기본) — 평균과 표준편차 반환

        Returns
        -------
        (mean, std) : tuple of np.ndarray
            MC Dropout 평균과 불확실성(표준편차)
        """
        cfg = self.config
        results: List[np.ndarray] = []
        converge_count = 0
        prev_std = None

        for i in range(cfg.n_max):
            result = forward_fn()
            results.append(np.asarray(result, dtype=np.float64))

            # 최소 횟수 미달
            if i + 1 < cfg.n_min:
                continue

            # warmup 미달
            if i + 1 < cfg.warmup:
                continue

            # 현재 표준편차 계산
            stacked = np.stack(results, axis=0)
            curr_std = np.mean(np.std(stacked, axis=0))

            # 수렴 판정
            if prev_std is not None:
                delta = abs(curr_std - prev_std)
                if delta < cfg.threshold:
                    converge_count += 1
                else:
                    converge_count = 0

                if converge_count >= cfg.patience:
                    self._last_converged = True
                    break

            prev_std = curr_std

        self._last_n_runs = len(results)
        stacked = np.stack(results, axis=0)
        mean = np.mean(stacked, axis=0).astype(np.float32)
        std = np.std(stacked, axis=0).astype(np.float32)
        return mean, std

    @property
    def last_n_runs(self) -> int:
        """마지막 실행에서의 실제 forward pass 횟수"""
        return self._last_n_runs

    @property
    def last_converged(self) -> bool:
        """마지막 실행에서 조기 수렴했는지 여부"""
        return self._last_converged

    def efficiency_ratio(self) -> float:
        """효율성 비율: 1 - (실제 횟수 / 최대 횟수)"""
        if self.config.n_max == 0:
            return 0.0
        return 1.0 - self._last_n_runs / self.config.n_max
