"""
metrics.py
==========
통합 성능 지표 계산
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class TrainingMetrics:
    episode: int
    phase: str
    blue_win_rate: float
    avg_blue_casualties: float
    avg_force_reduction: float
    avg_steps: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    nash_gap: float = 0.0
    gnn_ece: float = 0.0


def compute_force_reduction_rate(
    initial_force: int, final_force: int
) -> float:
    return (initial_force - final_force) / max(initial_force, 1)


def compute_exchange_ratio(blue_cas: int, red_cas: int) -> float:
    """사상 교환비 (높을수록 Blue 불리)"""
    return blue_cas / max(red_cas, 1)


def compute_mission_efficiency(
    win: bool, casualties: int, force_used: int, time_steps: int
) -> float:
    """임무 효율 지수 [0, 1]"""
    if not win:
        return 0.0
    casualty_factor = max(0, 1 - casualties / max(force_used, 1))
    time_factor = max(0, 1 - time_steps / 50)
    return 0.5 * casualty_factor + 0.5 * time_factor


def moving_average(data: List[float], window: int = 50) -> np.ndarray:
    return np.convolve(data, np.ones(window) / window, mode='valid')


def print_metrics(metrics: TrainingMetrics):
    print(f"EP {metrics.episode:5d} | Phase={metrics.phase} | "
          f"WR={metrics.blue_win_rate:.1%} | "
          f"Cas={metrics.avg_blue_casualties:.1f} | "
          f"Force↓={metrics.avg_force_reduction:.1%} | "
          f"Nash={metrics.nash_gap:.4f}")
