"""
Smoke tests for Tier C algorithms.

Validates that each algorithm can:
1. Instantiate without errors
2. Complete a forward pass
3. Not crash on basic operations

These are NOT training convergence tests — just "does it run?" checks.
"""
import numpy as np
import pytest
import torch


# ── MAT (Multi-Agent Transformer) ────────────────────────

def test_mat_instantiate():
    """MATPolicy should instantiate with default config."""
    from rl_agent.mat_policy import MATPolicy, MATConfig
    config = MATConfig()
    policy = MATPolicy(config)
    assert policy is not None


def test_mat_forward_pass():
    """MATPolicy encoder+decoder forward pass should produce valid logits."""
    from rl_agent.mat_policy import MATPolicy, MATConfig
    config = MATConfig(n_agents=3, obs_dim=24, n_actions=6, d_model=32, n_heads=2, n_layers=1)
    policy = MATPolicy(config)

    # Batch of observations: [batch=1, n_agents=3, obs_dim=24]
    obs = torch.randn(1, 3, 24)
    with torch.no_grad():
        result = policy(obs)
    # result should have logits for each agent
    assert result is not None
    # Check output shape contains agent actions
    if isinstance(result, tuple):
        logits = result[0]
    else:
        logits = result
    assert logits.shape[-1] == 6  # n_actions


def test_mat_select_actions():
    """MATPolicy.select_actions should return valid action indices."""
    from rl_agent.mat_policy import MATPolicy, MATConfig
    config = MATConfig(n_agents=4, obs_dim=24, n_actions=6, d_model=32, n_heads=2, n_layers=1)
    policy = MATPolicy(config)

    obs = torch.randn(1, 4, 24)
    with torch.no_grad():
        actions = policy.select_actions(obs)
    assert len(actions) == 4
    assert all(0 <= a < 6 for a in actions)


# ── RARL (Robust Adversarial RL) ─────────────────────────

def test_rarl_instantiate():
    """RARLTrainer should instantiate with default config."""
    from rl_agent.rarl import RARLTrainer, RARLConfig
    config = RARLConfig()
    trainer = RARLTrainer(config)
    assert trainer is not None
    assert trainer.adversary is not None


def test_rarl_adversary_forward():
    """ObsAdversary should produce perturbations within epsilon."""
    from rl_agent.rarl import ObsAdversary, RARLConfig
    config = RARLConfig(epsilon=0.1)
    adv = ObsAdversary(state_dim=128, config=config)

    state = torch.randn(1, 128)
    with torch.no_grad():
        perturbed = adv(state)

    delta = (perturbed - state).abs().max().item()
    assert delta <= config.epsilon + 1e-6, f"Perturbation {delta} exceeds epsilon {config.epsilon}"


def test_rarl_protagonist_step():
    """RARL protagonist should select valid actions on perturbed observations."""
    from rl_agent.rarl import RARLTrainer, RARLConfig
    config = RARLConfig(epsilon=0.05)
    trainer = RARLTrainer(config)

    state = np.random.randn(128).astype(np.float32)
    action, log_prob, value = trainer.select_action(state)
    assert 0 <= action < 6
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


# ── NFSP (Neural Fictitious Self-Play) ───────────────────

def test_nfsp_instantiate():
    """NFSPAgent should instantiate with dual networks."""
    from rl_agent.nfsp_agent import NFSPAgent, NFSPAgentConfig
    config = NFSPAgentConfig()
    agent = NFSPAgent(config)
    assert agent.br_network is not None
    assert agent.as_network is not None


def test_nfsp_select_action():
    """NFSPAgent should select actions via BR or AS policy."""
    from rl_agent.nfsp_agent import NFSPAgent, NFSPAgentConfig
    config = NFSPAgentConfig(eta=0.5)
    agent = NFSPAgent(config)

    state = np.random.randn(128).astype(np.float32)
    action, mode = agent.select_action(state)
    assert 0 <= action < 6
    assert mode in ("br", "as")


def test_nfsp_buffer_add():
    """NFSP buffers should accept experience tuples."""
    from rl_agent.nfsp_buffer import NFSPBufferPair, NFSPConfig, Experience
    config = NFSPConfig()
    buffers = NFSPBufferPair(config)

    exp = Experience(
        state=np.random.randn(128).astype(np.float32),
        action=2,
        reward=1.0,
        next_state=np.random.randn(128).astype(np.float32),
        done=False,
    )
    buffers.add_rl(exp)
    buffers.add_sl(exp)
    assert len(buffers.rl_buffer) == 1
    assert len(buffers.sl_buffer) == 1


# ── PSRO (Policy Space Response Oracles) ─────────────────

def test_psro_payoff_matrix():
    """PayoffMatrix should track strategies and update payoffs."""
    from rl_agent.psro_oracle import PayoffMatrix
    pm = PayoffMatrix()
    pm.add_strategy("s0")
    pm.add_strategy("s1")
    pm.update("s0", "s1", result=0.7)
    assert pm.get("s0", "s1") == pytest.approx(0.7, abs=0.01)


def test_psro_oracle_instantiate():
    """PSROOracle should instantiate and manage strategy pool."""
    from rl_agent.psro_oracle import PSROOracle
    oracle = PSROOracle(seed=42)
    assert oracle is not None
    assert oracle.payoff_matrix is not None


def test_psro_nash_mix():
    """PSROOracle should compute Nash mixture (even trivial)."""
    from rl_agent.psro_oracle import PSROOracle
    oracle = PSROOracle(seed=42)
    # Add initial strategies
    oracle.add_initial_strategies(["s0", "s1"])
    oracle.payoff_matrix.update("s0", "s1", 0.6)
    oracle.payoff_matrix.update("s1", "s0", 0.4)

    mix = oracle.compute_nash_mix()
    assert len(mix) == 2
    assert abs(sum(mix.values()) - 1.0) < 1e-6


# ── HierarchicalRL ───────────────────────────────────────

def test_hierarchical_instantiate():
    """HierarchicalRLCoordinator should instantiate."""
    from rl_agent.hierarchical_rl import HierarchicalRLCoordinator
    coord = HierarchicalRLCoordinator(decision_interval=3, seed=42)
    assert coord is not None
    assert coord.manager is not None


def test_hierarchical_operational_state():
    """Operational state builder should produce valid 20D vector."""
    from rl_agent.hierarchical_rl import HierarchicalRLCoordinator
    from ontology.combat_schema import ScenarioFactory
    coord = HierarchicalRLCoordinator(seed=42)
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=42)

    op_state = coord.build_operational_state(kg, step=0, max_steps=20)
    vec = op_state.to_vector()
    assert vec.shape == (20,)
    assert np.all(np.isfinite(vec))


def test_hierarchical_step():
    """Coordinator.step should return dict with objective and actions."""
    from rl_agent.hierarchical_rl import HierarchicalRLCoordinator
    from ontology.combat_schema import ScenarioFactory
    coord = HierarchicalRLCoordinator(decision_interval=1, seed=42)
    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=3, seed=42)

    result = coord.step(kg, step=0, max_steps=10)
    assert "operational_objective" in result
    assert "n_blue_acting" in result
    assert result["n_blue_acting"] >= 0
