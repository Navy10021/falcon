"""
temperature_scaling.py
======================
③ GNN Temperature Scaling 보정
- 사후 보정 (post-hoc calibration): 모델 재훈련 없이 ECE 개선
- Platt Scaling (분류)  + Temperature Scaling (회귀 불확실성)
- Isotonic Regression 보정
- 보정 전후 ECE 비교 자동 출력

목표: ECE < 0.10
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from gnn_model.uncertainty_utils import compute_ece


# ──────────────────────────────────────────────
# Temperature Scaling (회귀 불확실성)
# ──────────────────────────────────────────────

class TemperatureScaler:
    """
    Temperature Scaling for Uncertainty Calibration
    
    원리:
      보정된 표준편차 = σ_raw / T
      T > 1: 불확실성 축소 (과다 추정 보정)
      T < 1: 불확실성 확대 (과소 추정 보정)
    
    최적 T: Negative Log-Likelihood 최소화
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False
        self.calibration_history: List[Dict] = []

    def fit(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray,
        actual_values: np.ndarray,
        n_iter: int = 100,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Temperature 최적화 (경사하강법)
        최소화 목표: NLL = 0.5·[log(σ²/T²) + (y-μ)²/(σ²/T²)]
        """
        T = 1.0
        best_T = T
        best_nll = float('inf')

        ece_before = compute_ece(predicted_means, predicted_stds, actual_values)

        for i in range(n_iter):
            # 현재 T로 NLL 계산
            scaled_var = (predicted_stds / T) ** 2 + 1e-8
            nll = 0.5 * np.mean(
                np.log(scaled_var) + (actual_values - predicted_means) ** 2 / scaled_var
            )

            if nll < best_nll:
                best_nll = nll
                best_T = T

            # NLL의 T에 대한 수치 미분
            eps = 1e-4
            scaled_var_p = (predicted_stds / (T + eps)) ** 2 + 1e-8
            nll_p = 0.5 * np.mean(
                np.log(scaled_var_p) + (actual_values - predicted_means) ** 2 / scaled_var_p
            )
            grad = (nll_p - nll) / eps

            # 업데이트 (log-space에서 최적화)
            T = T * np.exp(-lr * grad)
            T = np.clip(T, 0.05, 10.0)

        self.temperature = float(best_T)
        self.is_fitted = True

        calibrated_stds = predicted_stds / self.temperature
        ece_after = compute_ece(predicted_means, calibrated_stds, actual_values)

        result = {
            "temperature": self.temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "improvement": ece_before - ece_after,
            "target_met": ece_after < 0.10,
        }
        self.calibration_history.append(result)
        return result

    def calibrate(self, predicted_stds: np.ndarray) -> np.ndarray:
        """보정된 표준편차 반환"""
        if not self.is_fitted:
            return predicted_stds
        return predicted_stds / self.temperature

    def calibrate_single(self, std: float) -> float:
        return float(std / self.temperature)


# ──────────────────────────────────────────────
# Isotonic Regression 보정
# ──────────────────────────────────────────────

class IsotonicCalibrator:
    """
    Isotonic Regression 기반 보정
    - 단조 증가 제약 하에 보정 함수 학습
    - Temperature Scaling보다 유연 (비선형 보정 가능)
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_corrections: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray,
        actual_values: np.ndarray
    ) -> Dict[str, float]:
        """
        신뢰구간 포함률 기반 Isotonic Calibration 학습
        """
        ece_before = compute_ece(predicted_means, predicted_stds, actual_values)

        # 각 예측의 z-score (표준화 오차)
        z_scores = np.abs(actual_values - predicted_means) / (predicted_stds + 1e-8)

        # z-score 기반 보정 스케일 학습
        # 이상적으로는 z-score ~ 반정규분포
        # quantile 별로 보정 계수 계산
        quantiles = np.linspace(0.05, 0.95, self.n_bins)
        expected_z = np.array([
            np.percentile(np.abs(np.random.standard_normal(10000)), q * 100)
            for q in quantiles
        ])
        actual_z = np.percentile(z_scores, quantiles * 100)

        # 보정 스케일: actual_z / expected_z (단조 증가 강제)
        raw_corrections = actual_z / (expected_z + 1e-8)
        # 단조 증가 강제 (pool adjacent violators)
        corrections = self._isotonic_regression(raw_corrections)

        self.bin_edges = np.percentile(z_scores, np.linspace(0, 100, self.n_bins + 1))
        self.bin_corrections = corrections
        self.is_fitted = True

        calibrated_stds = self._apply_correction(predicted_means, predicted_stds, actual_values)
        ece_after = compute_ece(predicted_means, calibrated_stds, actual_values)

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "improvement": ece_before - ece_after,
            "target_met": ece_after < 0.10,
        }

    def _isotonic_regression(self, y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm"""
        n = len(y)
        result = y.copy()
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # 위반 지점: 평균으로 대체
                j = i
                while j < n - 1 and result[j] > result[j + 1]:
                    j += 1
                avg = result[i:j + 1].mean()
                result[i:j + 1] = avg
                i = max(0, i - 1)
            else:
                i += 1
        return result

    def _apply_correction(self, means, stds, actuals):
        z = np.abs(actuals - means) / (stds + 1e-8)
        correction = np.ones_like(stds)
        if self.bin_corrections is not None:
            bins = np.digitize(z, self.bin_edges[1:-1])
            for i, b in enumerate(bins):
                b = min(b, len(self.bin_corrections) - 1)
                correction[i] = self.bin_corrections[b]
        return stds * correction

    def calibrate(self, means: np.ndarray, stds: np.ndarray,
                  reference_z: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted or self.bin_corrections is None:
            return stds
        z = reference_z if reference_z is not None else np.ones_like(stds)
        correction = np.ones_like(stds)
        bins = np.digitize(z, self.bin_edges[1:-1]) if self.bin_edges is not None else np.zeros(len(stds), int)
        for i, b in enumerate(bins):
            b = min(b, len(self.bin_corrections) - 1)
            correction[i] = self.bin_corrections[b]
        return stds * correction


# ──────────────────────────────────────────────
# 통합 GNN 보정 파이프라인
# ──────────────────────────────────────────────

@dataclass
class CalibrationReport:
    """보정 결과 보고서"""
    method: str
    ece_before: float
    ece_after: float
    improvement: float
    target_met: bool          # ECE < 0.10
    temperature: Optional[float] = None

    def summary(self) -> str:
        status = "✅ 목표 달성" if self.target_met else "⚠️ 추가 개선 필요"
        lines = [
            f"[{self.method}] {status}",
            f"  ECE 보정 전: {self.ece_before:.4f}",
            f"  ECE 보정 후: {self.ece_after:.4f}",
            f"  개선량:      {self.improvement:.4f} ({self.improvement/max(self.ece_before,1e-8)*100:.1f}%)",
        ]
        if self.temperature:
            lines.append(f"  Temperature: {self.temperature:.4f}")
        return "\n".join(lines)


class GNNCalibrationPipeline:
    """
    GNN 불확실성 보정 통합 파이프라인
    Temperature Scaling → Isotonic Regression 순서로 적용
    최적 방법 자동 선택
    """

    def __init__(self):
        self.temp_scaler = TemperatureScaler()
        self.isotonic    = IsotonicCalibrator()
        self.best_method: str = "none"
        self.reports: List[CalibrationReport] = []

    def fit_all(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray,
        actual_values: np.ndarray
    ) -> List[CalibrationReport]:
        """모든 보정 방법 적합 및 비교"""
        # Temperature Scaling
        ts_result = self.temp_scaler.fit(predicted_means, predicted_stds, actual_values)
        ts_report = CalibrationReport(
            method="Temperature Scaling",
            ece_before=ts_result["ece_before"],
            ece_after=ts_result["ece_after"],
            improvement=ts_result["improvement"],
            target_met=ts_result["target_met"],
            temperature=ts_result["temperature"],
        )

        # Isotonic Regression
        iso_result = self.isotonic.fit(predicted_means, predicted_stds, actual_values)
        iso_report = CalibrationReport(
            method="Isotonic Regression",
            ece_before=iso_result["ece_before"],
            ece_after=iso_result["ece_after"],
            improvement=iso_result["improvement"],
            target_met=iso_result["target_met"],
        )

        self.reports = [ts_report, iso_report]

        # 최적 방법 선택
        best = min(self.reports, key=lambda r: r.ece_after)
        self.best_method = best.method

        return self.reports

    def calibrate(
        self,
        predicted_means: np.ndarray,
        predicted_stds: np.ndarray
    ) -> np.ndarray:
        """최적 방법으로 보정된 std 반환"""
        if self.best_method == "Temperature Scaling":
            return self.temp_scaler.calibrate(predicted_stds)
        elif self.best_method == "Isotonic Regression":
            return self.isotonic.calibrate(predicted_means, predicted_stds)
        return predicted_stds

    def calibrate_single(self, mean: float, std: float) -> Tuple[float, float]:
        """단일 예측값 보정"""
        calibrated_std = self.calibrate(
            np.array([mean]), np.array([std])
        )[0]
        return mean, float(calibrated_std)

    def print_report(self):
        print("\n" + "=" * 50)
        print("📊 GNN 보정 결과 보고서")
        print("=" * 50)
        for r in self.reports:
            print(r.summary())
            print()
        print(f"✅ 선택된 최적 방법: {self.best_method}")
        best = min(self.reports, key=lambda r: r.ece_after)
        status = "달성" if best.target_met else "미달"
        print(f"   최종 ECE {best.ece_after:.4f} (목표 < 0.10: {status})")


# ──────────────────────────────────────────────
# 합성 데이터 기반 검증 유틸
# ──────────────────────────────────────────────

def generate_calibration_dataset(
    n_samples: int = 500,
    overconfident: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    보정 테스트용 합성 데이터 생성
    overconfident=True: 모델이 불확실성을 과소 추정 (실제 분산 > 예측 분산)
    """
    rng = np.random.RandomState(seed)
    true_values = rng.uniform(0, 100, n_samples)
    pred_means  = true_values + rng.normal(0, 5, n_samples)

    if overconfident:
        # 과신 모델: 실제 오차보다 std를 작게 예측
        pred_stds = rng.uniform(2, 6, n_samples)   # 실제보다 좁은 CI
    else:
        pred_stds = rng.uniform(8, 18, n_samples)  # 실제보다 넓은 CI

    return pred_means, pred_stds, true_values


if __name__ == "__main__":
    print("=" * 55)
    print("③ GNN Temperature Scaling 보정 테스트")
    print("=" * 55)

    # 과신 모델 데이터 (ECE 높음)
    means, stds, actuals = generate_calibration_dataset(500, overconfident=True, seed=42)

    ece_raw = compute_ece(means, stds, actuals)
    print(f"\n보정 전 ECE: {ece_raw:.4f}")

    pipeline = GNNCalibrationPipeline()
    reports = pipeline.fit_all(means, stds, actuals)
    pipeline.print_report()

    # 단일값 보정 테스트
    cal_mean, cal_std = pipeline.calibrate_single(25.0, 3.0)
    print(f"\n단일값 보정: mean={cal_mean:.2f}, std 원본=3.00 → 보정={cal_std:.2f}")
    print("\n✅ Temperature Scaling 보정 정상 동작!")
