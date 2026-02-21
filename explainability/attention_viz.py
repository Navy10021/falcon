"""
attention_viz.py
================
GNN 어텐션 가중치 시각화 + 결정 근거 생성
Explainability Layer
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def generate_decision_report(
    unit_id: str,
    unit_type: str,
    terrain: str,
    attention_scores: Dict[str, float],
    uncertainty: Dict[str, float],
    recommended_action: str,
    casualty_prediction: Dict
) -> str:
    """
    GNN 어텐션 기반 자연어 결정 근거 생성
    """
    top_factors = sorted(attention_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    lines = [
        f"🔍 결정 근거 보고서: {unit_id}",
        "─" * 50,
        f"유닛 유형: {unit_type}",
        f"현재 지형: {terrain}",
        f"권고 행동: {recommended_action}",
        "",
        "주요 판단 근거 (어텐션 가중치):",
    ]

    for factor, score in top_factors:
        bar = "█" * int(score * 20)
        lines.append(f"  {factor:30s} {bar} ({score:.3f})")

    lines.extend([
        "",
        f"불확실성 분석:",
        f"  Epistemic (모델 불확실성): {uncertainty.get('epistemic', 0):.3f}",
        f"  Aleatoric (데이터 노이즈): {uncertainty.get('aleatoric', 0):.3f}",
    ])

    ep = uncertainty.get('epistemic', 0)
    al = uncertainty.get('aleatoric', 0)
    total_unc = ep + al
    if total_unc > 0.6:
        unc_label = "매우 높음 → 추가 정찰 권고"
    elif total_unc > 0.3:
        unc_label = "중간 → 보수적 전략 고려"
    else:
        unc_label = "낮음 → 적극적 행동 가능"
    lines.append(f"  전체 불확실성 수준: {unc_label}")

    if casualty_prediction:
        mean = casualty_prediction.get("mean", 0)
        std  = casualty_prediction.get("std",  0)
        ci_l = mean - 1.96 * std
        ci_h = mean + 1.96 * std
        lines.extend([
            "",
            f"사상자 예측:",
            f"  예상: {mean:.1f}명 ± {std:.1f}명",
            f"  95% CI: [{max(0, ci_l):.1f}, {ci_h:.1f}]명",
        ])

    return "\n".join(lines)


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    node_labels: List[str],
    title: str = "GNN Attention Weights",
    save_path: Optional[str] = None
) -> plt.Figure:
    """어텐션 가중치 히트맵 시각화"""
    n = min(len(node_labels), attention_weights.shape[0], 15)
    weights = attention_weights[:n, :n]
    labels = node_labels[:n]

    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Source Node")
    ax.set_ylabel("Target Node")

    # 값 표시
    for i in range(n):
        for j in range(n):
            val = weights[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_training_curves(
    episode_rewards: List[float],
    win_rates: List[float],
    casualty_rates: List[float],
    nash_gaps: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """훈련 곡선 시각화"""
    n_plots = 4 if nash_gaps else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    def smooth(data, w=20):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w) / w, mode='valid').tolist()

    plots = [
        (episode_rewards, "Episode Reward",    "Reward",    "#2196F3"),
        (win_rates,       "Blue Win Rate",     "Win Rate",  "#4CAF50"),
        (casualty_rates,  "Avg Blue Casualties","Count",    "#F44336"),
    ]
    if nash_gaps:
        plots.append((nash_gaps, "Nash Gap", "Nash Gap", "#9C27B0"))

    for ax, (data, title, ylabel, color) in zip(axes, plots):
        smoothed = smooth(data)
        ax.plot(data, color=color, alpha=0.3, linewidth=0.5)
        ax.plot(range(len(smoothed)), smoothed, color=color, linewidth=2, label='Smoothed')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if title == "Nash Gap":
            ax.axhline(0.05, color='red', linestyle='--', label='Convergence threshold')
            ax.legend(fontsize=8)

    plt.suptitle("AI Combat Optimization — Training Progress", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    print("=== Explainability Module Test ===")

    # 결정 근거 보고서
    report = generate_decision_report(
        unit_id="blue_3",
        unit_type="armor",
        terrain="urban",
        attention_scores={
            "Urban terrain": 0.71,
            "Enemy artillery support": 0.68,
            "Low ammo state": 0.54,
            "Adjacent infantry unit": 0.42,
            "River obstacle": 0.31,
        },
        uncertainty={"epistemic": 0.42, "aleatoric": 0.29},
        recommended_action="Withdraw",
        casualty_prediction={"mean": 23.5, "std": 8.2}
    )
    print(report)

    # 어텐션 히트맵
    n = 8
    weights = np.random.dirichlet(np.ones(n), n)
    labels = [f"unit_{i}" for i in range(n)]
    fig = plot_attention_heatmap(weights, labels, save_path="/tmp/attention_test.png")
    print(f"\n✅ 어텐션 히트맵 저장: /tmp/attention_test.png")

    # 훈련 곡선
    ep = 200
    fig2 = plot_training_curves(
        episode_rewards=list(np.random.cumsum(np.random.randn(ep))),
        win_rates=list(np.clip(np.cumsum(np.random.randn(ep) * 0.01) + 0.5, 0, 1)),
        casualty_rates=list(np.abs(np.random.randn(ep) * 10 + 20)),
        nash_gaps=list(np.exp(-np.arange(ep) / 100) + np.random.randn(ep) * 0.01),
        save_path="/tmp/training_curves_test.png"
    )
    print("✅ 훈련 곡선 저장: /tmp/training_curves_test.png")
    print("✅ Explainability 모듈 테스트 완료!")
