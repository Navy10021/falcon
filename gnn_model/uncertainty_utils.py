"""
uncertainty_utils.py
====================
불확실성 보정 및 분석 유틸리티
- ECE (Expected Calibration Error) 계산
- 불확실성 시각화
- Epistemic / Aleatoric 분리 분석
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_ece(
    predicted_means: np.ndarray,
    predicted_stds: np.ndarray,
    actual_values: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) 계산
    잘 보정된 모델: ECE < 0.1
    
    신뢰구간 포함률 기반 ECE 계산
    """
    confidence_levels = np.linspace(0.1, 0.99, n_bins)
    calibration_errors = []

    for conf in confidence_levels:
        z = abs(np.percentile(np.random.standard_normal(10000), (1 - conf) / 2 * 100))
        lower = predicted_means - z * predicted_stds
        upper = predicted_means + z * predicted_stds
        coverage = np.mean((actual_values >= lower) & (actual_values <= upper))
        calibration_errors.append(abs(coverage - conf))

    return float(np.mean(calibration_errors))


def decompose_uncertainty(
    mc_predictions: np.ndarray,      # [n_samples, n_mc]
    predicted_variances: np.ndarray   # [n_samples] (aleatoric)
) -> Dict[str, np.ndarray]:
    """
    불확실성 분해 (Epistemic + Aleatoric)
    
    Total = Epistemic + Aleatoric
    Epistemic: MC 샘플 간 분산 (모델 불확실성, 데이터로 줄일 수 있음)
    Aleatoric: 데이터 고유 노이즈 (줄일 수 없음)
    """
    epistemic = np.var(mc_predictions, axis=1)   # 모델 불확실성
    aleatoric = predicted_variances               # 데이터 불확실성
    total = epistemic + aleatoric

    return {
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total": total,
        "epistemic_ratio": epistemic / (total + 1e-8),
        "aleatoric_ratio": aleatoric / (total + 1e-8),
    }


def risk_action_mapping(
    uncertainty_score: float,
    base_threshold: float = 0.3
) -> Dict[str, object]:
    """
    불확실성 점수 → 행동 권고 매핑
    
    불확실성 높음 → 보수적 전략 (병력 유지)
    불확실성 낮음 → 적극적 전략 (최소 병력)
    """
    if uncertainty_score < base_threshold * 0.5:
        return {
            "strategy": "aggressive",
            "force_multiplier": 0.75,   # 병력 25% 절감
            "description": "불확실성 낮음 → 최소 병력 투입 가능",
            "confidence": "high"
        }
    elif uncertainty_score < base_threshold:
        return {
            "strategy": "balanced",
            "force_multiplier": 0.90,
            "description": "불확실성 중간 → 표준 병력 투입",
            "confidence": "medium"
        }
    elif uncertainty_score < base_threshold * 2:
        return {
            "strategy": "conservative",
            "force_multiplier": 1.10,
            "description": "불확실성 높음 → 여유 병력 유지",
            "confidence": "low"
        }
    else:
        return {
            "strategy": "hold",
            "force_multiplier": 1.25,
            "description": "불확실성 매우 높음 → 추가 정찰 후 행동",
            "confidence": "very_low"
        }


def plot_uncertainty_calibration(
    predicted_means: np.ndarray,
    predicted_stds: np.ndarray,
    actual_values: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """보정 플롯 생성"""
    confidence_levels = np.linspace(0.05, 0.99, 20)
    coverages = []

    for conf in confidence_levels:
        z = abs(np.percentile(np.random.standard_normal(10000), (1 - conf) / 2 * 100))
        lower = predicted_means - z * predicted_stds
        upper = predicted_means + z * predicted_stds
        coverage = np.mean((actual_values >= lower) & (actual_values <= upper))
        coverages.append(coverage)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 보정 곡선
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    ax.plot(confidence_levels, coverages, 'b-o', markersize=4, label='Model Calibration')
    ax.fill_between(confidence_levels, confidence_levels, coverages,
                    alpha=0.2, color='red', label='Calibration Error')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('Uncertainty Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ece = compute_ece(predicted_means, predicted_stds, actual_values)
    ax.set_title(f'Calibration Curve (ECE = {ece:.4f})')

    # 예측 분포 vs 실제
    ax2 = axes[1]
    idx = np.argsort(actual_values)
    x_axis = np.arange(len(actual_values))
    ax2.fill_between(x_axis,
                     predicted_means[idx] - 2 * predicted_stds[idx],
                     predicted_means[idx] + 2 * predicted_stds[idx],
                     alpha=0.3, label='95% CI')
    ax2.plot(x_axis, predicted_means[idx], 'b-', label='Predicted Mean')
    ax2.plot(x_axis, actual_values[idx], 'r.', markersize=3, label='Actual')
    ax2.set_xlabel('Sample Index (sorted)')
    ax2.set_ylabel('Casualty Count')
    ax2.set_title('Prediction Intervals vs Actual Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_uncertainty_decomposition(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """불확실성 분해 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {'epistemic': '#2196F3', 'aleatoric': '#FF9800', 'total': '#9C27B0'}

    total = epistemic + aleatoric
    ax = axes[0]
    ax.hist(epistemic, bins=30, color=colors['epistemic'], alpha=0.7, label='Epistemic')
    ax.hist(aleatoric, bins=30, color=colors['aleatoric'], alpha=0.7, label='Aleatoric')
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(epistemic, aleatoric, alpha=0.5, s=10, c=total, cmap='viridis')
    ax2.set_xlabel('Epistemic Uncertainty')
    ax2.set_ylabel('Aleatoric Uncertainty')
    ax2.set_title('Epistemic vs Aleatoric')

    ax3 = axes[2]
    ratios = epistemic / (total + 1e-8)
    ax3.hist(ratios, bins=30, color=colors['total'], alpha=0.7)
    ax3.axvline(0.5, color='red', linestyle='--', label='50% split')
    ax3.set_xlabel('Epistemic Ratio')
    ax3.set_ylabel('Count')
    ax3.set_title('Epistemic Ratio Distribution')
    ax3.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("=== Uncertainty Utils Test ===")
    rng = np.random.RandomState(42)

    n = 200
    true_vals = rng.uniform(0, 100, n)
    pred_means = true_vals + rng.normal(0, 5, n)
    pred_stds = rng.uniform(3, 15, n)

    ece = compute_ece(pred_means, pred_stds, true_vals)
    print(f"✅ ECE: {ece:.4f} ({'GOOD' if ece < 0.1 else 'NEEDS IMPROVEMENT'})")

    mc_preds = rng.normal(pred_means[:, None], pred_stds[:, None], (n, 20))
    decomp = decompose_uncertainty(mc_preds, pred_stds**2)
    print(f"✅ 평균 Epistemic: {decomp['epistemic'].mean():.2f}")
    print(f"✅ 평균 Aleatoric: {decomp['aleatoric'].mean():.2f}")

    for unc_score in [0.05, 0.2, 0.5, 0.8]:
        action = risk_action_mapping(unc_score)
        print(f"  불확실성={unc_score:.2f} → {action['strategy']}: {action['description']}")
