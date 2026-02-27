<div align="center">

# 🦅 FALCON: Force-Adaptive Learning for Combat Optimization Network

### Ontology-driven decision support with Bayesian GNN, Adversarial RL, and HITL command constraints

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**FALCON is an experimentation framework for uncertainty-aware, robust battlefield decision recommendation.**

</div>

---

## Overview

FALCON integrates four core ideas into a single experimentation pipeline:

1. **Ontology-first state representation** for units, missions, command relations, and constraints.
2. **Bayesian graph reasoning** for confidence-aware situation understanding under partial observability.
3. **Adversarial reinforcement learning** for robust policy search against deceptive or worst-case opponents.
4. **Human-in-the-loop (HITL)** reranking under commander preferences and ROE-like constraints.

This repository is designed for research and prototyping workflows such as:
- robust policy comparison in simulated combat settings,
- uncertainty/risk-aware action recommendation,
- reproducible evaluation with Monte Carlo rollouts and benchmark suites.

---

## Key Capabilities

- 🧠 **Knowledge-structured combat modeling** via ontology modules.
- 🌫️ **Uncertainty-aware inference** via Bayesian GNN components.
- ⚔️ **Adversarial RL suite** including self-play and robust optimization variants.
- 👨‍✈️ **Commander-centered decision support** through HITL preference/constraint modules.
- 📊 **Evaluation stack** for scenario suites, Monte Carlo robustness, and metrics reporting.
- 🧾 **Reproducibility-focused demo pipeline** that generates artifacts and reports.

---

## Technical Stack

FALCON is designed not as a concept-only proposal, but as an end-to-end stack for rapidly iterating through practical experimentation and validation loops.

### Core Runtime & ML

- **Python 3.10+**: Unifies experiment code, simulator logic, and evaluation pipelines.
- **PyTorch 2.x**: Implements Bayesian GNNs, policy networks, and training loops.
- **NumPy / SciPy**: Provides numerical computing, distribution sampling, and statistical evaluation.
- **scikit-learn**: Supports calibration and auxiliary analysis/meta-modeling tasks.

### RL & Decision Intelligence

- **Gymnasium**: Standardizes interfaces for experimentation-ready battlefield environments.
- **Stable-Baselines3**: Enables quick PPO-family baseline establishment.
- **Custom Adversarial RL Modules**: Supports RARL, self-play, NFSP, MAPPO, and league/PSRO extensions.

### Knowledge / Graph / Ontology Layer

- **Domain Ontology Modules (`ontology/`)**: Structure unit types, missions, command relationships, and rules of engagement.
- **Graph-centric Reasoning (`gnn_model/`)**: Perform relation-based inference under partial observability.
- **Constraint Encoding (ROE/HITL)**: Directly inject commander intent and constraints into policy selection.

### Evaluation, Ops, and Reproducibility

- **YAML Config System (`configs/`)**: Declaratively manages scenario, evaluation, and training settings.
- **Monte Carlo & Benchmark Suite (`evaluation/`, `demo/evaluation/`)**: Validates robustness and generalization performance.
- **PyTest-based Regression Gates (`tests/`)**: Enforces staged contract tests and numerical stability checks.
- **TensorBoard / Rich / Plotly / Matplotlib / Seaborn**: Covers training tracking, reporting, and visualization.

---

## Why FALCON: Differentiation & Feasibility

### 1) Differentiation

1. **Single-pipeline integration of Ontology + Bayesian GNN + Adversarial RL + HITL**  
   While many approaches remain rule-based, pure RL, or narrowly focused on isolated situation assessment, FALCON links knowledge representation, uncertainty modeling, robust learning, and commander constraints in one flow.

2. **Design centered on trustworthy recommendations rather than raw accuracy**  
   It jointly evaluates probabilistic confidence (uncertainty), policy robustness, and ROE compliance for operationally realistic decision support.

3. **Architecture that balances experimentability and explainability**  
   Reproducible scenario/seed-based evaluation is paired with AAR/interpretability modules so teams can trace why recommendations were produced.

### 2) Feasibility

1. **Already modular architecture**  
   The system is decomposed into `ontology`, `gnn_model`, `rl_agent`, `simulator`, `hitl`, and `evaluation`, enabling parallel development and staged upgrades.

2. **Supports a progressive adoption strategy**  
   Teams can validate performance and stability incrementally using a path such as Rule baseline → RL baseline → adversarial/self-play → HITL reranking.

3. **Built-in validation-friendly experiment system**  
   CLI entrypoints, test suites, and Monte Carlo evaluation make model-performance claims repeatable.

4. **Prototype-friendly even in CPU-only environments**  
   Dependency and design choices support lightweight experiments, lowering infrastructure overhead during early adoption.

---

## Repository Structure

```text
falcon/
├── demo/              # Packaged demo/evaluation/reporting entrypoints
├── ontology/          # Domain schema, constraints, and ontology logic
├── gnn_model/         # Bayesian / uncertainty-aware GNN components
├── rl_agent/          # RL agents, adversarial/self-play training modules
├── simulator/         # Combat dynamics and simulation environments
├── hitl/              # Human-in-the-loop preference and constraint handling
├── evaluation/        # Benchmarks and Monte Carlo evaluation utilities
├── tests/             # Regression, contract, smoke, and phase tests
├── demo.py            # Top-level demo runner
└── evaluate.py        # Top-level evaluation runner
```

---

## Quick Start

### 1) Environment setup

```bash
git clone https://github.com/Navy10021/falcon
cd falcon

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run demo pipeline

```bash
python demo.py
```

### 3) Run packaged demo entrypoint

```bash
python -m demo.demo --scenario urban_defense --seed 42 --policy rule --out runs/demo_urban
```

### 4) Run evaluation

```bash
python evaluate.py --fast
python -m demo.evaluate --suite small --mc 20 --seed 42 --output-dir runs/eval
```

---

## Typical Workflow

1. Select or define a scenario and baseline policy.
2. Execute demo rollouts to generate trajectories/artifacts.
3. Run evaluation suites (fast or full) with Monte Carlo sampling.
4. Compare results with robustness and mission-effectiveness metrics.
5. Iterate on ontology, policy, uncertainty modeling, and HITL constraints.

---

## Development & Testing

```bash
# optional developer dependencies
pip install -r requirements-dev.txt

# run tests
pytest -q
```

If you are contributing algorithmic changes, include:
- reproducible run commands,
- seed/config information,
- before/after metric evidence.

---

## Documentation

- Korean extended documentation: [`README_KOR.md`](README_KOR.md)
- Demo package details: [`demo/DEMO_README.md`](demo/DEMO_README.md)

---

## Contributing

Contributions are welcome for:
- model quality and stability improvements,
- robustness/evaluation extensions,
- explainability and HITL UX improvements,
- test coverage and reproducibility tooling.

Please open an issue or PR with clear context and validation steps.

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
