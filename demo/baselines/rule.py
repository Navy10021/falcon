from __future__ import annotations


def rule_policy_action(blue_strength: float, red_strength: float) -> str:
    if blue_strength > red_strength * 1.12:
        return "assault"
    if blue_strength < red_strength * 0.90:
        return "fallback"
    return "hold"
