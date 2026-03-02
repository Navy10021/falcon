"""
Reward contract tests — enforce documented component boundaries.

These tests validate the REWARD_CONTRACT.md specification:
all 7 reward components must stay within documented ranges.
"""

import numpy as np
import pytest

from rl_agent.blue_agent import BlueAgent, BlueActionSpace, PPOConfig
from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus


# ── helpers ──────────────────────────────────────────────

def _make_step_result(mission_status="ongoing", blue_cas=10, red_cas=5):
    """Minimal step result stub matching BattleStep interface."""
    class _Step:
        def __init__(self, ms, bc, rc):
            self.mission_status = ms
            self.blue_total_casualties = bc
            self.red_total_casualties = rc
    return _Step(mission_status, blue_cas, red_cas)


@pytest.fixture
def agent():
    return BlueAgent()


# ── 1. win_reward boundary ───────────────────────────────

def test_win_reward_blue_win(agent):
    """Win signal must be +W_WIN on blue_win."""
    step = _make_step_result("blue_win", blue_cas=0, red_cas=0)
    r = agent.compute_reward(step, 100, 100, initial_force_size=100)
    # win_reward=+10, all other terms near zero → reward >= 10
    assert r >= agent._W_WIN


def test_win_reward_red_win(agent):
    """Loss signal must be -W_WIN on red_win."""
    step = _make_step_result("red_win", blue_cas=0, red_cas=0)
    r = agent.compute_reward(step, 100, 100, initial_force_size=100)
    # win_reward=-10, survival_bonus positive → reward <= -W_WIN + survival
    assert r <= -agent._W_WIN + agent._W_FORCE_RATIO + 0.1


def test_win_reward_ongoing(agent):
    """Ongoing step has no win signal."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    r = agent.compute_reward(step, 100, 100)
    # No big terminal bonus, small terms only
    assert abs(r) < 5.0


# ── 2. casualty_penalty non-negative ─────────────────────

def test_casualty_penalty_nonneg(agent):
    """Casualty penalty must only reduce reward."""
    step_0 = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    step_100 = _make_step_result("ongoing", blue_cas=100, red_cas=0)
    r0 = agent.compute_reward(step_0, 100, 100)
    r100 = agent.compute_reward(step_100, 100, 100)
    assert r0 >= r100  # more casualties → lower reward


# ── 3. force_reward non-negative ─────────────────────────

def test_force_reward_nonneg(agent):
    """Force saving reward must be >= 0."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    # force_before > force_after → positive
    r_save = agent.compute_reward(step, 100, 90)
    r_none = agent.compute_reward(step, 100, 100)
    assert r_save >= r_none


def test_force_reward_zero_on_increase(agent):
    """Force saving reward must be 0 when force increases (shouldn't happen, but safe)."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    r = agent.compute_reward(step, 90, 100)
    # force_before < force_after → no force reward
    r2 = agent.compute_reward(step, 100, 100)
    assert r <= r2 + 0.01  # at most marginal difference from other terms


# ── 4. survival_bonus terminal only ─────────────────────

def test_survival_bonus_only_terminal(agent):
    """Survival bonus must be 0 during ongoing steps."""
    step_ongoing = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    step_win = _make_step_result("blue_win", blue_cas=0, red_cas=0)
    r_ongoing = agent.compute_reward(step_ongoing, 100, 100, initial_force_size=100)
    r_win = agent.compute_reward(step_win, 100, 100, initial_force_size=100)
    # terminal has big win_reward (+10) + survival_bonus
    assert r_win > r_ongoing


def test_survival_bonus_bounded(agent):
    """Survival bonus must be in [0, W_FORCE_RATIO]."""
    step = _make_step_result("blue_win", blue_cas=0, red_cas=0)
    # Compare with same force_before=force_after to isolate survival bonus
    # Full survival (ratio=1.0) → bonus=W_FORCE_RATIO
    r_full = agent.compute_reward(step, 100, 100, initial_force_size=100)
    # Half survival (ratio=0.5) → bonus=0.5*W_FORCE_RATIO
    r_half = agent.compute_reward(step, 50, 50, initial_force_size=100)
    assert r_full > r_half


# ── 5. enemy_damage non-negative ─────────────────────────

def test_enemy_damage_nonneg(agent):
    """More enemy casualties → higher reward."""
    step_0 = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    step_50 = _make_step_result("ongoing", blue_cas=0, red_cas=50)
    r0 = agent.compute_reward(step_0, 100, 100)
    r50 = agent.compute_reward(step_50, 100, 100)
    assert r50 >= r0


# ── 6. doctrine_bonus in [-0.5, +0.5] ────────────────────

def test_doctrine_bonus_positive(agent):
    """High doctrine score (>0.5) → positive bonus."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    r_high = agent.compute_reward(step, 100, 100, doctrine_score=1.0)
    r_low = agent.compute_reward(step, 100, 100, doctrine_score=0.0)
    assert r_high > r_low


def test_doctrine_bonus_neutral_at_half(agent):
    """Doctrine score=0.5 → bonus=0."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    r_half = agent.compute_reward(step, 100, 100, doctrine_score=0.5)
    r_zero = agent.compute_reward(step, 100, 100, doctrine_score=0.0)
    # At 0.5, doctrine contribution is 0; at 0.0 it's -0.5
    assert r_half > r_zero


def test_doctrine_bonus_bounded_range(agent):
    """Doctrine bonus must be within [-0.5, +0.5] for score in [0,1]."""
    bonus_min = agent._W_DOCTRINE * (0.0 - 0.5)  # -0.5
    bonus_max = agent._W_DOCTRINE * (1.0 - 0.5)  # +0.5
    assert bonus_min == pytest.approx(-0.5)
    assert bonus_max == pytest.approx(0.5)


# ── 7. uncertainty_penalty non-negative ──────────────────

def test_uncertainty_penalty_nonneg(agent):
    """Uncertainty penalty must only reduce reward."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    r_low = agent.compute_reward(step, 100, 90, uncertainty=0.0)
    r_high = agent.compute_reward(step, 100, 90, uncertainty=0.9)
    assert r_low >= r_high


def test_uncertainty_penalty_zero_when_no_reduction(agent):
    """No force reduction → no uncertainty penalty."""
    step = _make_step_result("ongoing", blue_cas=0, red_cas=0)
    r0 = agent.compute_reward(step, 100, 100, uncertainty=0.0)
    r1 = agent.compute_reward(step, 100, 100, uncertainty=0.9)
    # Both should be equal: no force reduction → no penalty triggered
    assert r0 == pytest.approx(r1)


# ── 8. composite sanity checks ───────────────────────────

def test_reward_finite(agent):
    """Reward must always be finite."""
    for ms in ["ongoing", "blue_win", "red_win"]:
        for unc in [0.0, 0.5, 1.0]:
            step = _make_step_result(ms, blue_cas=50, red_cas=30)
            r = agent.compute_reward(
                step, 200, 150, uncertainty=unc,
                initial_force_size=200, doctrine_score=0.7,
            )
            assert np.isfinite(r), f"Non-finite reward: {r}"


def test_reward_weights_match_constants(agent):
    """Weight constants must match documented contract values."""
    assert agent._W_WIN == 10.0
    assert agent._W_LOSS == -10.0
    assert agent._W_CASUALTY == 0.05
    assert agent._W_FORCE_SAVE == 0.15
    assert agent._W_FORCE_RATIO == 2.0
    assert agent._W_ENEMY_DMG == 0.03
    assert agent._W_DOCTRINE == 1.0


def test_action_space_integrity():
    """BlueActionSpace constants must be self-consistent."""
    assert BlueActionSpace.N_ACTIONS == 6
    assert len(BlueActionSpace.ACTION_NAMES) == BlueActionSpace.N_ACTIONS
    # Values must be contiguous 0..N-1
    vals = [
        BlueActionSpace.REINFORCE, BlueActionSpace.ADVANCE,
        BlueActionSpace.WITHDRAW, BlueActionSpace.REALLOCATE,
        BlueActionSpace.SUPPORT, BlueActionSpace.FLANK,
    ]
    assert sorted(vals) == list(range(BlueActionSpace.N_ACTIONS))
