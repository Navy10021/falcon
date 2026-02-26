from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_episode_plot(path: Path, metrics_rows: Iterable[Dict[str, float]]) -> None:
    rows: List[Dict[str, float]] = list(metrics_rows)
    episodes = [int(row["episode"]) for row in rows]
    friendly_losses = [float(row["friendly_loss"]) for row in rows]
    enemy_losses = [float(row["enemy_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, friendly_losses, marker="o", label="friendly_loss")
    ax.plot(episodes, enemy_losses, marker="s", label="enemy_loss")
    ax.set_title("Episode Loss Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
