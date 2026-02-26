from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceProfile:
    name: str
    loss_weight: float
    time_weight: float
    mission_weight: float


def get_preference_profile(mode: str) -> PreferenceProfile:
    profiles = {
        "loss_min": PreferenceProfile("loss_min", loss_weight=0.65, time_weight=0.15, mission_weight=0.20),
        "time_min": PreferenceProfile("time_min", loss_weight=0.20, time_weight=0.65, mission_weight=0.15),
        "balanced": PreferenceProfile("balanced", loss_weight=0.4, time_weight=0.3, mission_weight=0.3),
    }
    if mode not in profiles:
        raise ValueError(f"Unknown preference mode: {mode}")
    return profiles[mode]


def score_action(*, expected_loss: float, expected_time: float, mission_gain: float, profile: PreferenceProfile) -> float:
    loss_score = 1.0 - expected_loss
    time_score = 1.0 - expected_time
    return (
        profile.loss_weight * loss_score
        + profile.time_weight * time_score
        + profile.mission_weight * mission_gain
    )
