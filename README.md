<div align="center">

# 🦅 FALCON
### **F**orce-**A**daptive **L**earning for **C**ombat **O**ptimization **N**etwork

*Ontology-Driven GNN + Adversarial Reinforcement Learning for Minimum-Force Warfare*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-251%2F251_✓-22C55E?style=flat-square)](tests/)
[![Purpose](https://img.shields.io/badge/Purpose-Defense_AI_Competition-003087?style=flat-square)](https://ideamaicon.kr)

**"Achieve the same combat effect. With 25% fewer troops."**

[Overview](#-overview) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [Results](#-results) · [Ethics](#-ethics)

</div>

---

## ⚡ The Problem in One Sentence

A battlefield commander must make life-or-death decisions under **severe information uncertainty** — incomplete intel, adaptive enemies, and time pressure. Traditional optimization breaks down. FALCON doesn't.

---

## 🎯 Overview

FALCON is an end-to-end AI decision-support framework that answers a single question:

> *"What is the minimum force needed to accomplish this mission, and how confident should we be?"*

It combines four tightly integrated modules — Bayesian GNN, Adversarial Self-Play RL, Human-in-the-Loop control, and doctrine-aware explainability — into a unified pipeline that learns **robust, minimum-force tactics** without sacrificing combat effectiveness.

| Challenge | FALCON's Solution |
|-----------|-------------------|
| Fog of War / uncertainty | Bayesian GNN quantifies *epistemic* + *aleatoric* uncertainty |
| Fixed-scenario brittle tactics | Red vs. Blue Self-Play → Robust Minimax strategies |
| Black-box AI decisions | Doctrine-grounded auto-explanation (ADP 9 principles) |
| Full autonomy ethical risk | Human-in-the-Loop — commander always holds final authority |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      FALCON PIPELINE                          │
│                                                                │
│  🗂️ Ontology  ──▶  ⚔️  Simulator  ──▶  🕸️  Bayesian GNN      │
│  Combat knowledge     Lanchester physics   Uncertainty-aware   │
│  (KG: 14 units,       Fog of War          prediction dist.     │
│   120 edges,          curriculum          N(μ, σ²)            │
│   9 ADP principles)                            │               │
│                                                ▼               │
│  📊 Evaluation  ◀──  🎯 HITL       ◀──  🤖  RL Agents         │
│  Monte Carlo          Pareto options       Blue PPO            │
│  5,000 runs           Natural language     ⬆️  ⬇️              │
│  Benchmark            Commander lock-in    Red PPO (Adversarial)│
│       │                    │                                   │
│       └──────────── 💡 Explainability ───────────────────────  │
│                    Auto-AAR + Attention viz + Counterfactuals  │
└────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

<details>
<summary><b>🗂️ Ontology Layer — Combat Knowledge as Code</b></summary>

Every battlefield concept is encoded as a structured knowledge graph. The AI *understands* warfare, not just numbers.

```python
from ontology.combat_schema import ScenarioFactory
from ontology.doctrine_encoder import DoctrineEncoder

kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6)
# → 14 units | 120 edges | terrain, capability, relationship fully structured

doctrine = DoctrineEncoder()
compliance = doctrine.evaluate(kg, step=5, step_result=result)
# → mass: 0.82  offensive: 0.71  security: 0.65  (9 ADP principles, auto-scored)
```

**ADP 9 Principles auto-evaluated in real-time:**
`MASS` · `OBJECTIVE` · `OFFENSIVE` · `SECURITY` · `SURPRISE` · `ECONOMY OF FORCE` · `MANEUVER` · `UNITY OF COMMAND` · `SIMPLICITY`
</details>

<details>
<summary><b>🕸️ Bayesian GNN — Uncertainty Is a First-Class Citizen</b></summary>

Standard GNNs output a single point estimate. Under Fog of War, that's dangerous. FALCON's Bayesian HGT-GNN outputs a **distribution**:

```
Standard GNN:    "Expected casualties = 23"
FALCON GNN:      "Casualties ~ N(23, 8²)   95% CI = [7, 39]"
                  → High uncertainty  →  Agent plays conservatively
                  → Low  uncertainty  →  Agent pushes minimum-force option
```

- **MC-Dropout** over Heterogeneous Graph Transformer (HGT)
- Epistemic / Aleatoric uncertainty decomposition
- Temperature scaling post-hoc calibration (ECE < 0.1 target)
- Node types: `Unit` · `Terrain` · `Objective` · `Threat`
- Edge types: `supports` · `threatens` · `occupies` · `observes`
</details>

<details>
<summary><b>🤖 Self-Play RL — The Red Agent Makes Blue Unbreakable</b></summary>

FALCON trains two adversarial PPO agents simultaneously:

| Agent | Objective | Action Space |
|-------|-----------|-------------|
| 🔵 Blue | Accomplish mission · minimize force | Reinforce · Withdraw · Reallocate |
| 🔴 Red | Maximize Blue's casualties · deny objective | Ambush · Counter · **Deception** · Fortify |

Red's `Deception` action injects false signals → Blue's GNN uncertainty spikes → Blue must learn to act robustly under manipulation. **This is where Phase 1 and Phase 2 create compounding value.**

**4-Phase Training Schedule:**
```
Phase 2-A  │  Red fixed (rule-based), Blue trains   →  Base strategy
Phase 2-B  │  Blue fixed, Red adapts               →  Adversarial pressure
Phase 2-C  │  Alternating updates (10-epoch cycle)  →  Nash convergence
Phase 2-D  │  Population-based League              →  Overfitting prevention
```

Result: **Robust Minimax strategies** — tactics that hold even when the enemy plays optimally.
</details>

<details>
<summary><b>🎯 HITL — The Commander Is Always in Control</b></summary>

The commander types in plain language. FALCON handles the rest.

```
Input:  "Seize Hill 3 within 2 hours. Keep casualties under 50."

Parsed:
  ✅ HARD constraint  →  max_time_steps = 12
  ✅ HARD constraint  →  max_casualties = 50
  ✅ Intent           →  SEIZE_OBJECTIVE  (confidence 87%)
```

**Pareto-optimal strategy candidates presented to commander:**

| Option | Force | Win Rate | Casualties | Profile |
|--------|-------|----------|------------|---------|
| A — Minimum Force | 65 troops | 74% | 8 | Max force savings |
| **B — Balanced** ⭐ | **80 troops** | **81%** | **6** | **Sweet spot** |
| C — Max Win Rate | 95 troops | 89% | 4 | Certainty-first |
| D — Min Casualties | 88 troops | 83% | 3 | Casualty-averse |

The commander selects. FALCON learns the commander's preference style over time via a contextual bandit, generating increasingly personalized proposals.
</details>

<details>
<summary><b>💡 Explainability — "Why this strategy?" in doctrine language</b></summary>

**Auto After-Action Review (AAR) output:**
```
📋 Auto AAR — 🏆 WIN   (Overall Score: 81%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Friendly losses: 187 (18%)   Enemy losses: 342   Duration: 15 steps

⚡ Critical Turning Points:
  📈 Step  7: Win probability 51% → 67% surge  (flank maneuver effect)
  📈 Step 10: Successful flank → 89 enemies suppressed

✅ Best Decision:
  Step 5 [Flank Maneuver]: SURPRISE + MANEUVER principles applied

💡 Improvement Recommendations:
  1. Establish proactive resupply routine before ammo reaches 30%
```

GNN attention weights are visualized per decision, showing *which terrain features / threat edges* drove each recommendation.
</details>

---

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/falcon-combat-ai.git
cd falcon-combat-ai

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Run the full pipeline demo

```bash
python demo.py
```

```
══════════════ 🦅 FALCON — Combat AI System v2.0 ══════════════

STEP 1  Ontology & Scenario Init
  ✅ Knowledge Graph: 14 units, 120 edges

STEP 2  Fog of War
  ✅ Observed: 9/14 units (64.3%)   Uncertainty: Medium

STEP 3  Bayesian GNN Inference
  ✅ Casualty forecast: 23.4 ± 8.1   Epistemic uncertainty: 0.32

STEP 4  PPO Blue Agent
  ✅ Action: Flank Maneuver   Doctrine compliance: 78%

STEP 5  HITL Pareto Generation
  ⭐ Recommended: Option B — 80 troops, 81% win rate

STEP 6  Monte Carlo Validation (1,000 runs)
  ✅ Win rate: 79.2% ± 3.1%   Force reduction: 22%
```

### Training

```bash
# Phase 1: Uncertainty-Aware GNN + Blue Agent baseline
python train.py --phase 1 --episodes 1000

# Phase 2: Self-Play (Blue ↔ Red adversarial)
python train.py --phase 2 --episodes 5000 --self-play

# Phase 3: HITL integration
python train.py --phase 3 --episodes 2000 --hitl
```

### Evaluation

```bash
python evaluate.py --monte-carlo 5000        # Robustness
python evaluate.py --benchmark historical    # 5 tactical scenarios
```

### Full Test Suite (251 tests)

```bash
PYTHONPATH=. python tests/test_tier1.py              # Ontology · Simulator   21/21
PYTHONPATH=. python tests/test_tier2.py              # GNN · RL               30/30
PYTHONPATH=. python tests/test_tier3.py              # HITL · Evaluation      39/39
PYTHONPATH=. python tests/test_phase1_immediate.py   # Dashboard · NL         32/32
PYTHONPATH=. python tests/test_phase2_short.py       # Maneuver · Doctrine    53/53
PYTHONPATH=. python tests/test_phase3_medium.py      # IRL · AAR · Benchmark  76/76
```

---

## 📊 Results

| Metric | Target | Meaning |
|--------|--------|---------|
| **Force reduction** | **15–25%** | Same combat effect · fewer troops |
| **Win rate delta** | **< ±3%** | Performance cost is minimal |
| **Uncertainty calibration** | **ECE < 0.1** | GNN predictions are trustworthy |
| **Self-Play convergence** | **Nash Gap < 0.05** | Adversarial strategies stabilize |
| **HITL adoption rate** | **> 70%** | Commanders trust the recommendations |

### Historical Benchmark Scenarios

| Scenario | Terrain | Key Variable | Difficulty |
|----------|---------|-------------|------------|
| Amphibious Assault | Coastal | Timing | Hard |
| Urban Defense | City | Attrition | Medium |
| Encirclement | Open | Maneuver speed | Medium |
| Mountain Delay | Highland | Time | Easy |
| Forest Ambush | Forest | Surprise | Easy |

---

## 📁 Repository Structure

```
falcon-combat-ai/
├── 🗂️  ontology/            Combat knowledge ontology
├── ⚔️   simulator/           Battle environment (Lanchester engine, Fog of War)
├── 🕸️  gnn_model/            Bayesian HGT-GNN + uncertainty utils
├── 🤖  rl_agent/             PPO · MAPPO · IRL · Self-Play trainer
├── 🎯  hitl/                 Human-in-the-Loop · Pareto generator · NL interface
├── 💡  explainability/       Auto-AAR · Attention viz · Counterfactual analysis
├── 📊  evaluation/           Monte Carlo · Historical benchmarks · Metrics
├── 📺  visualization/        Plotly.js real-time dashboard
├── 🧪  tests/                251-test suite
├── demo.py                   End-to-end pipeline demo
├── train.py                  Training script
└── evaluate.py               Evaluation script
```

---

## 🔬 Research Contributions

**Contribution 1 — Uncertainty-Aware Combat GNN**
The first integrated framework that quantifies battlefield uncertainty via Bayesian GNN and embeds the prediction distribution directly into PPO agent state, enabling risk-adaptive strategy learning under Fog of War.

**Contribution 2 — Adversarial Combat Self-Play**
A Blue–Red PPO Self-Play system where Red's `Deception` actions directly amplify Blue's GNN epistemic uncertainty, producing Robust Minimax minimum-force strategies that remain effective against an optimally playing adversary.

**Contribution 3 — Doctrine-Aware HITL**
A single interface integrating ADP-9 doctrine auto-evaluation, Pareto strategy generation, and commander preference learning — satisfying Meaningful Human Control requirements under IHL.

**Contribution 4 — End-to-End Explainable Pipeline**
Doctrine-language explanations generated across the full Ontology → GNN → RL → HITL chain, enabling commanders to audit every AI recommendation in terms they already understand.

---

## 🗺️ Roadmap

```
v2.0  NOW ──────────────────────────── ✅ Complete
  Bayesian GNN + Uncertainty-Aware RL
  Blue ↔ Red Self-Play
  HITL + Natural Language Interface
  Auto-AAR + Historical Benchmark

v3.0  NEAR-TERM ─────────────────────── 🔲 Planned
  Real training data integration (anonymized)
  3D battlefield visualization
  Coalition RL (multi-national forces)
  LLM-powered natural language command interface

v4.0  LONG-TERM ─────────────────────── 🔲 Research
  Digital twin battlefield environment
  Drone swarm coordination
  Satellite ISR open-data integration
```

---

## ⚖️ Ethics

> FALCON operates exclusively on **fully synthetic, simulated combat data** for **academic research purposes only**. No classified or operational information is involved.

- **No lethal autonomy**: FALCON never makes final engagement decisions. The human commander always retains authority.
- **IHL-compliant**: The HITL architecture satisfies the "Meaningful Human Control" requirement central to international autonomous weapons discussions.
- **Research-only**: Intended solely as a training simulation decision-support tool.

---

## 📄 License

MIT License — Free for academic research, education, and non-commercial use.

---

<div align="center">

**🦅 FALCON — Force-Adaptive Learning for Combat Optimization Network**

*"Saving force is saving lives."*

[![Tests](https://img.shields.io/badge/Tests-251%2F251_✓-22C55E?style=flat-square)](tests/)
[![Modules](https://img.shields.io/badge/Modules-8-3B82F6?style=flat-square)](.)
[![Code](https://img.shields.io/badge/Code-13%2C800+_lines-8B5CF6?style=flat-square)](.)
[![Stars](https://img.shields.io/github/stars/your-username/falcon-combat-ai?style=flat-square&color=FFD700)](.)

</div>
