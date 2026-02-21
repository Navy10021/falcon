<div align="center">

# 🦅 FALCON: AI Combat Optimization Framework
### **F**orce-**A**daptive **L**earning for **C**ombat **O**ptimization **N**etwork

*Ontology-driven Bayesian GNN + Adversarial RL + Human-in-the-Loop Decision Support*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[Overview](#-overview) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Training--evaluation](#-training--evaluation) · [Ethics](#-ethics)

</div>

---

## ⚡ Problem Statement

Battlefield decisions are made under severe uncertainty (partial observation, deception, time pressure). Traditional optimization methods are brittle in this setting.

**FALCON** is an end-to-end research framework that answers:

> *"What is the minimum force needed to complete the mission while maintaining acceptable risk?"*

---

## 🎯 Overview

FALCON integrates four major capabilities in one pipeline:

1. **Combat ontology + simulator** for structured, doctrine-aware battlefield state.
2. **Bayesian GNN** for predictive uncertainty (epistemic + aleatoric).
3. **Adversarial RL (Blue vs Red)** for robust strategy learning.
4. **HITL (Human-in-the-Loop)** for commander-constrained, explainable decision support.

| Challenge | FALCON Approach |
|---|---|
| Fog of war and noisy observations | Bayesian uncertainty estimation over graph state |
| Brittle fixed-strategy behavior | Blue–Red self-play and adversarial pressure |
| Black-box recommendations | Doctrine-language explainability and AAR |
| Ethical autonomy concerns | Commander remains final decision authority (HITL) |

---

## 🏗️ Architecture

```text
Ontology/Scenario -> Simulator/Fog -> Bayesian GNN -> RL Agents (Blue/Red)
                                     -> HITL Pareto options + preference learning
                                     -> Explainability + Evaluation
```

### Main modules

- `ontology/`: knowledge graph schema, doctrine encoding, scenario generation
- `simulator/`: Lanchester-based engines, fog-of-war, maneuvers, resources
- `gnn_model/`: Bayesian HGT and uncertainty utilities
- `rl_agent/`: PPO agents and self-play trainer
- `hitl/`: constraint parsing, Pareto strategy generation, preference learning
- `explainability/`: attention-based reports, AAR utilities
- `evaluation/`: Monte Carlo and historical benchmark evaluation
- `visualization/`: dashboard and charting components

---

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/falcon-combat-ai.git
cd falcon-combat-ai

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run full demo

```bash
python demo.py
```

> Note: `demo.py` runs with randomly initialized agents by default. Low win rates are expected unless trained checkpoints are loaded.

---

## 🧠 Training / 📊 Evaluation

### Training

```bash
# Phase 1: Bayesian GNN + Blue PPO baseline
python train.py --phase 1 --episodes 1000

# Phase 2: self-play trainer
python train.py --phase 2 --episodes 5000

# Phase 3: HITL loop
python train.py --phase 3 --episodes 2000 --hitl
```

### Evaluation

```bash
# Monte Carlo robustness
python evaluate.py --monte-carlo 5000 --fog-level moderate

# Historical benchmark scenarios
python evaluate.py --benchmark historical --benchmark-runs 10
```

---

## 🧪 Tests

```bash
PYTHONPATH=. python tests/test_tier1.py
PYTHONPATH=. python tests/test_tier2.py
PYTHONPATH=. python tests/test_tier3.py
PYTHONPATH=. python tests/test_phase1_immediate.py
PYTHONPATH=. python tests/test_phase2_short.py
PYTHONPATH=. python tests/test_phase3_medium.py
```

---

## 📈 Interpreting Demo Metrics

If you see very low Blue win rates in demo output (for example ~4%), this is generally **expected** in current default settings because:

- Agents in `demo.py` are instantiated without loading trained checkpoints.
- Monte Carlo evaluation in demo uses this untrained policy.
- The Red side and environment randomness can dominate early behavior.

For meaningful policy quality checks, run training first and evaluate with a checkpoint.

---

## ⚖️ Ethics & Scope

- This repository is for **research and education** on synthetic combat simulation.
- It does **not** use classified data.
- It does **not** grant lethal autonomy to AI systems.
- HITL design is intended to preserve meaningful human control.

---

## 📄 License

MIT License.
