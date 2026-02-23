"""Numerical stability smoke tests for BlueAgent/GNN outputs."""
import numpy as np

from ontology.combat_schema import ScenarioFactory, ForceAlignment
from simulator.fog_of_war import FogOfWarFilter, FogLevel
from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors
from rl_agent.blue_agent import BlueAgent, build_state_vector


def main():
    # 1) build_state_vector should sanitize NaN/Inf extensions
    kg = ScenarioFactory.create_standard_scenario(seed=42)
    bad_ext = np.array([np.nan, np.inf, -np.inf, 1e30, -1e30, 0.0], dtype=np.float32)
    state = build_state_vector(kg, gnn_extension=bad_ext)
    assert np.isfinite(state).all(), "state vector contains non-finite values"

    # 2) BlueAgent action selection should not crash with non-finite input
    agent = BlueAgent()
    action, log_prob, value = agent.select_action(state)
    assert isinstance(action, int)
    assert np.isfinite(log_prob), "log_prob is non-finite"
    assert np.isfinite(value), "value is non-finite"

    # 3) GNN uncertainty path should remain finite
    fog = FogOfWarFilter(FogLevel.MODERATE, seed=42)
    obs_kg, _ = fog.observe(kg, ForceAlignment.BLUE)
    x, adj = prepare_graph_tensors(obs_kg)
    gnn = BayesianHGT(node_in_dim=128, hidden_dim=64, n_layers=2, mc_samples=5)
    out = gnn.predict_with_uncertainty(x, adj)
    for k, v in out.items():
        val = float(v.item())
        assert np.isfinite(val), f"{k} is non-finite"
        if "casualty" in k:
            assert val >= 0.0, f"{k} should be non-negative"

    print("✅ numerical stability tests passed")


def test_numerical_stability():
    """pytest: GNN/BlueAgent 수치 안정성 검증."""
    main()


if __name__ == "__main__":
    main()
