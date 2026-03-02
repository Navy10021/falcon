"""
experiment_tracker.py
=====================
FALCON 실험 추적 인프라 — TensorBoard + JSON 이중 로깅

학습/평가 메트릭을 TensorBoard와 JSON 파일에 동시 기록하며,
이상 탐지(NaN, 발산, 급격한 성능 저하)를 자동 수행한다.

사용 예시::

    tracker = ExperimentTracker(log_dir="runs/exp_001", run_id="abc123")
    tracker.log_episode(ep=10, metrics={"win_rate": 0.65, "loss": 0.12})
    tracker.log_evaluation(step=100, report=mc_report)
    tracker.check_anomalies()  # NaN/발산 자동 감지
    tracker.close()

학술 연구용 합성 데이터
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("falcon")


# ──────────────────────────────────────────────
# Anomaly Detection
# ──────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    """탐지된 이상 이벤트"""
    step: int
    metric_name: str
    anomaly_type: str  # "nan", "inf", "divergence", "performance_drop"
    value: float
    threshold: float
    message: str


@dataclass
class AnomalyDetectorConfig:
    """이상 탐지 설정"""
    check_nan: bool = True
    check_inf: bool = True
    divergence_threshold: float = 100.0
    performance_drop_window: int = 50
    performance_drop_ratio: float = 0.3  # 30% 이상 급락 시 경고


class AnomalyDetector:
    """학습 메트릭 이상 탐지기"""

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self.history: Dict[str, List[float]] = {}
        self.anomalies: List[AnomalyEvent] = []

    def check(self, step: int, metrics: Dict[str, float]) -> List[AnomalyEvent]:
        """메트릭 검사 후 이상 이벤트 반환"""
        events = []
        for name, value in metrics.items():
            # 히스토리 기록
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

            # NaN 체크
            if self.config.check_nan and (np.isnan(value) if isinstance(value, float) else False):
                evt = AnomalyEvent(step, name, "nan", value, 0,
                                   f"NaN detected in {name} at step {step}")
                events.append(evt)

            # Inf 체크
            if self.config.check_inf and (np.isinf(value) if isinstance(value, float) else False):
                evt = AnomalyEvent(step, name, "inf", value, 0,
                                   f"Inf detected in {name} at step {step}")
                events.append(evt)

            # 발산 체크
            if isinstance(value, (int, float)) and abs(value) > self.config.divergence_threshold:
                evt = AnomalyEvent(step, name, "divergence", value,
                                   self.config.divergence_threshold,
                                   f"{name}={value:.2f} exceeds threshold "
                                   f"{self.config.divergence_threshold}")
                events.append(evt)

            # 성능 급락 체크
            hist = self.history[name]
            w = self.config.performance_drop_window
            if len(hist) > w and name in ("win_rate", "blue_win_rate"):
                recent_avg = np.mean(hist[-w // 2:])
                prev_avg = np.mean(hist[-w:-w // 2])
                if prev_avg > 0 and (prev_avg - recent_avg) / prev_avg > self.config.performance_drop_ratio:
                    evt = AnomalyEvent(
                        step, name, "performance_drop", recent_avg, prev_avg,
                        f"{name} dropped {(prev_avg - recent_avg)/prev_avg:.1%}: "
                        f"{prev_avg:.3f} -> {recent_avg:.3f}",
                    )
                    events.append(evt)

        self.anomalies.extend(events)
        return events


# ──────────────────────────────────────────────
# Experiment Tracker
# ──────────────────────────────────────────────

class ExperimentTracker:
    """
    TensorBoard + JSON 이중 실험 추적기.

    TensorBoard가 설치되어 있으면 활성화, 아니면 JSON만 사용.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        run_id: str = "default",
        enable_tensorboard: bool = True,
        anomaly_config: Optional[AnomalyDetectorConfig] = None,
    ):
        self.log_dir = os.path.join(log_dir, run_id)
        self.run_id = run_id
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard writer (optional)
        self._tb_writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(self.log_dir)
            except ImportError:
                warnings.warn(
                    "TensorBoard not available. Install with: pip install tensorboard",
                    stacklevel=2,
                )

        # JSON 로그
        self._json_log_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._json_file = open(self._json_log_path, "a", encoding="utf-8")

        # 이상 탐지기
        self.anomaly_detector = AnomalyDetector(anomaly_config)

        # 메트릭 캐시
        self._episode_metrics: List[Dict] = []
        self._eval_metrics: List[Dict] = []

    def log_episode(self, ep: int, metrics: Dict[str, float], phase: str = "train"):
        """
        에피소드 메트릭 기록.

        Parameters
        ----------
        ep : int
            에피소드 번호
        metrics : dict
            {metric_name: value} 형식
        phase : str
            "train" | "eval" | "selfplay"
        """
        # TensorBoard 기록
        if self._tb_writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._tb_writer.add_scalar(f"{phase}/{key}", val, ep)

        # JSON 기록
        record = {"type": "episode", "ep": ep, "phase": phase, **metrics}
        self._json_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._json_file.flush()

        # 이상 탐지
        anomalies = self.anomaly_detector.check(ep, metrics)
        for a in anomalies:
            logger.warning("[ANOMALY] %s", a.message)

        self._episode_metrics.append(record)

    def log_evaluation(self, step: int, report, phase: str = "eval"):
        """
        평가 보고서 메트릭 기록.

        report는 MCEvaluationReport 또는 dict 형식.
        """
        if isinstance(report, dict):
            metrics = report
        else:
            # MCEvaluationReport 형식
            metrics = {
                "win_rate": getattr(report, "blue_win_rate", 0),
                "avg_casualties": getattr(report, "avg_blue_casualties", 0),
                "avg_force_reduction": getattr(report, "avg_force_reduction", 0),
                "strategy_robustness": getattr(report, "strategy_robustness", 0),
                "cvar_05": getattr(report, "cvar_05", 0),
                "cvar_10": getattr(report, "cvar_10", 0),
            }

        # TensorBoard
        if self._tb_writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._tb_writer.add_scalar(f"{phase}/{key}", val, step)

        # JSON
        record = {"type": "evaluation", "step": step, "phase": phase, **metrics}
        self._json_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._json_file.flush()

        self._eval_metrics.append(record)

    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """하이퍼파라미터 기록"""
        if self._tb_writer is not None and metrics:
            try:
                self._tb_writer.add_hparams(hparams, metrics)
            except Exception:
                pass

        record = {"type": "hparams", **hparams}
        self._json_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._json_file.flush()

    def check_anomalies(self) -> List[AnomalyEvent]:
        """누적된 이상 이벤트 조회"""
        return self.anomaly_detector.anomalies

    def get_metric_history(self, metric_name: str) -> List[float]:
        """특정 메트릭의 히스토리 조회"""
        return self.anomaly_detector.history.get(metric_name, [])

    def close(self):
        """리소스 정리"""
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._json_file and not self._json_file.closed:
            self._json_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def metrics_path(self) -> str:
        return self._json_log_path
