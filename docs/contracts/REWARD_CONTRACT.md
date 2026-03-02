# FALCON Reward Contract

**Version**: 1.0
**Last Updated**: 2026-03-02
**Source of Truth**: `rl_agent/blue_agent.py:BlueAgent.compute_reward()`

---

## 1. Reward Components

FALCON's reward function is a weighted sum of 7 components.
All components MUST stay within their documented ranges.

| # | Component | Symbol | Output Range | Default Weight | Description |
|---|-----------|--------|-------------|----------------|-------------|
| 1 | **Win/Loss Signal** | `win_reward` | {-10, 0, +10} | 1.0 | Terminal: +10 (blue_win), -10 (red_win), 0 (ongoing/draw) |
| 2 | **Casualty Penalty** | `casualty_penalty` | [0, +inf) | 0.05 per casualty | `blue_total_casualties * W_CASUALTY`. Always subtracted. |
| 3 | **Force Saving** | `force_reward` | [0, +inf) | 0.15 per unit saved | `max(0, force_before - force_after) * W_FORCE_SAVE` |
| 4 | **Survival Bonus** | `survival_bonus` | [0, 2.0] | 2.0 (scale) | Terminal only: `W_FORCE_RATIO * (force_after / initial_force)` |
| 5 | **Enemy Damage** | `enemy_damage` | [0, +inf) | 0.03 per casualty | `red_total_casualties * W_ENEMY_DMG` |
| 6 | **Doctrine Bonus** | `doctrine_bonus` | [-0.5, +0.5] | 1.0 (scale) | `W_DOCTRINE * (doctrine_score - 0.5)` |
| 7 | **Uncertainty Penalty** | `uncertainty_penalty` | [0, +inf) | 0.1 (coef) | Active only when `uncertainty > 0.5 AND force_reduction > 0` |

## 2. Composite Formula

```
R = win_reward
  - casualty_penalty
  + force_reward
  + survival_bonus
  + enemy_damage
  + doctrine_bonus
  - uncertainty_penalty
```

## 3. Weight Constants (BlueAgent class attributes)

| Constant | Value | Tuning History |
|----------|-------|---------------|
| `_W_WIN` | 10.0 | Original |
| `_W_LOSS` | -10.0 | Original |
| `_W_CASUALTY` | 0.05 | Original |
| `_W_FORCE_SAVE` | 0.15 | v2: 0.01 -> 0.15 (15x increase) |
| `_W_FORCE_RATIO` | 2.0 | Original |
| `_W_ENEMY_DMG` | 0.03 | v2: 0.02 -> 0.03 |
| `_W_DOCTRINE` | 1.0 | Added in Phase 3 |
| `uncertainty_penalty_coef` | 0.1 | PPOConfig default |

## 4. Phase-specific Behavior

| Phase | Active Components | Notes |
|-------|------------------|-------|
| **Phase 1** (GNN+PPO) | 1-5, 7 | Doctrine bonus inactive (score=0.0 default) |
| **Phase 2** (Self-Play) | 1-5, 7 | Symmetric: blue/red each use own reward |
| **Phase 3** (HITL) | N/A | No RL update; Pareto evaluation only |
| **Phase 4** (Preference) | 1-7 | All active; IRL bonus via preference adapter |

## 5. Boundary Constraints (enforced by tests)

1. `win_reward` MUST be exactly one of {-10, 0, +10}
2. `casualty_penalty` MUST be >= 0
3. `force_reward` MUST be >= 0
4. `survival_bonus` MUST be in [0, W_FORCE_RATIO] and only nonzero at terminal
5. `enemy_damage` MUST be >= 0
6. `doctrine_bonus` MUST be in [-0.5, +0.5]
7. `uncertainty_penalty` MUST be >= 0

## 6. Scale Normalization Guide

For future reward shaping, normalize each component to [-1, +1] before weighting:
- `norm_win = win_reward / W_WIN`  -> {-1, 0, +1}
- `norm_casualty = -min(casualty / 100, 1.0)` -> [-1, 0]
- `norm_force = min(force_reward / 5.0, 1.0)` -> [0, 1]
- `norm_survival = survival_bonus / W_FORCE_RATIO` -> [0, 1]
- `norm_enemy = min(enemy_damage / 5.0, 1.0)` -> [0, 1]
- `norm_doctrine = doctrine_bonus / 0.5` -> [-1, +1]
- `norm_uncertainty = -min(uncertainty_penalty / 1.0, 1.0)` -> [-1, 0]
