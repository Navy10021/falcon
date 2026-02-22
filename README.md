<div align="center">

# 🦅 FALCON: AI Combat Optimization Framework
### **F**orce-**A**daptive **L**earning for **C**ombat **O**ptimization **N**etwork

*Ontology-driven Bayesian GNN + Adversarial RL + Human-in-the-Loop Decision Support*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Code](https://img.shields.io/badge/Code-20%2C000+_lines-8B5CF6?style=flat-square)](.)
[![Ontology](https://img.shields.io/badge/UnitTypes-42_종-F59E0B?style=flat-square)](ontology/)

[Overview](#-overview) · [Architecture](#-architecture) · [Ontology](#-ontology-layer-expanded) · [Quickstart](#-quickstart) · [Training](#-training--evaluation) · [Ethics](#-ethics)

</div>

---

## ⚡ Problem Statement

Battlefield decisions are made under severe uncertainty — partial observation, deception, information asymmetry, and time pressure. Traditional optimization methods are brittle in this setting.

**FALCON** is an end-to-end research framework that answers:

> *"What is the minimum force needed to complete the mission while maintaining acceptable risk?"*

---

## 🎯 Overview

FALCON integrates six major capabilities in one pipeline:

1. **Multi-domain combat ontology** — 42 unit types, 7 branches, C2 structures, ROE, and intelligence fusion.
2. **Bayesian GNN** — Epistemic + aleatoric uncertainty prediction over heterogeneous graphs.
3. **Adversarial RL (Blue vs Red)** — Robust strategy learning through self-play.
4. **HITL (Human-in-the-Loop)** — Commander-constrained, explainable decision support.
5. **Combat dynamics** — Ammunition, BDA, supply chains, and electromagnetic spectrum.
6. **Intelligence Ontology** — HUMINT/SIGINT/IMINT/MASINT fusion with Fog-of-War modeling.

| Challenge | FALCON Approach |
|---|---|
| Fog of war and noisy observations | Intelligence fusion (8-source) + Bayesian FogOfWar → GNN uncertainty |
| Brittle fixed-strategy behavior | Blue–Red self-play and adversarial pressure |
| Black-box recommendations | Doctrine-language explainability and AAR |
| Ethical autonomy concerns | ROE checker + Commander final authority (HITL) |
| Multi-domain complexity | 42 UnitTypes across Land/Sea/Air/Cyber/Space/Subsurface |

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FALCON END-TO-END PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

[0] Intelligence Layer  ← NEW
    ontology/intelligence.py
    HUMINT/SIGINT/IMINT/MASINT → INTELReport → FogOfWarState
    ISRAssetModel + IntelligenceFusionEngine (Dempster-Shafer)
            │  isr_quality / uncertainty_map / gnn_extension(8D)
            ▼
[1] Knowledge & Scenario Layer
    ontology/combat_schema     (42 UnitType · 7 Branch · 7 Domain)
    ontology/joint_operations  (C2 · CommandStructure · JointFires)
    ontology/scenario_presets  (5 presets: KoreaDefense·AirSup·Urban·Multi·CyberEW)
    ontology/roe_ethics        (ROE · EthicalConstraintChecker · 5 presets)
    ontology/doctrine_encoder  (ADP 9 principles)
            │
            ▼
[2] Simulation & Dynamics Layer
    simulator/mixed_lanchester   (42-type Lanchester, aviation support)
    simulator/combat_dynamics    (AmmoModel · BDA · SupplyMgr · ElectroMagEnv)
    simulator/fog_of_war         (Partial observation + curriculum)
    simulator/maneuver_engine    (A* · LOS · flanking)
    simulator/resource_manager   (Ammo · fuel · maintenance)
            │
            ▼
[3] Uncertainty Modeling Layer
    gnn_model/bayesian_hgt   (MC-Dropout HGT, node_in_dim=128)
    ├─ casualty_mean / casualty_ci
    ├─ epistemic uncertainty  ← EMS gnn_unc_multiplier applied
    └─ aleatoric uncertainty  ← FogOfWar node weights applied
            │
            ▼
[4] Decision Policy Layer
    rl_agent/blue_agent  STATE_DIM=128 (branch·domain·C2·intel·FoW)
    rl_agent/red_agent   ⇄  self_play_trainer (Phase A→D curriculum)
    rl_agent/mappo       (multi-agent)
    rl_agent/hierarchical_rl / league_selfplay / inverse_rl
            │
            ├──────────────────────────────────────────┐
            ▼                                          ▼
[5A] Human-in-the-Loop                       [5B] Explainability
     hitl/constraint_parser                       explainability/attention_viz
     hitl/pareto_generator                        explainability/auto_aar
     hitl/preference_learner                      counterfactual analysis
     hitl/realtime_replanner
            │                                          │
            └────────────────────┬─────────────────────┘
                                 ▼
[6] Evaluation & Reporting
    evaluation/monte_carlo + historical_benchmark + robustness metrics
```

---

## 🗺️ Ontology Layer (Expanded)

The ontology layer has been substantially expanded from the original 8-type schema to a comprehensive multi-domain military ontology.

### Unit Types — 42 kinds across all branches

| Domain | Types |
|--------|-------|
| **Land (Army)** | Infantry, Armor, Artillery, Mechanized Infantry, Airborne, Special Forces, Anti-Armor, Air Defense Artillery, Reconnaissance, NBC Defense, Military Police, Engineer, Logistics |
| **Land (Marine)** | Marine Infantry, Amphibious Assault |
| **Maritime (Navy)** | Destroyer, Frigate, Submarine, Mine Warfare, Amphibious Ship, Naval Aviation, Coastal Defense |
| **Air (Air Force)** | Fighter, Strike Aircraft, Bomber, ISR Aircraft, Tanker, Transport, AEW, Naval Aviation |
| **Missile & UAS** | Ballistic Missile, Cruise Missile, Anti-Ship Missile, SAM Battery, Rocket Artillery, UAV Recon, UAV Strike, Loitering Munition, UGV, USV |
| **Strategic / Cyber / Space** | Cyber Unit, Space Asset, PSYOPS, Strategic Missile, Electronic Warfare, Signal |

### New Ontology Modules

| Module | Contents |
|--------|----------|
| `joint_operations.py` | C2Link, CommandStructure, JointFiresRequest, JointOperationsManager |
| `scenario_presets.py` | 5 realistic scenario presets with C2 auto-build |
| `roe_ethics.py` | RulesOfEngagement (IHL-based), EthicalConstraintChecker, ROEManager |
| `intelligence.py` | INTELReport, FogOfWarState, ISRAssetModel (16 types), IntelligenceFusionEngine, IntelligenceManager |
| `military_units.py` | Realistic Korean military order-of-battle constants |

### State Vector — 128D

```
force_state(10) + blue_status(4) + red_status(4)
+ blue_branch(7) + red_branch(7)     ← branch composition
+ blue_domain(7) + red_domain(7)     ← domain composition
+ terrain_state(6) + c2_state(4)     ← C2/JointFires
= 56D base
+ gnn_extension(8)                   ← Intel: coverage/freshness/fusion/FoW/SIGINT/IMINT/HUMINT/score
+ temporal_features(8)
= 72D → padded to 128D
```

---

## 🔬 Key Modules

### `ontology/intelligence.py` — Intelligence Ontology

```python
from ontology.intelligence import IntelligenceManager
from ontology.scenario_presets import load_scenario

kg, c2_mgr = load_scenario("korea_defense", seed=42, with_c2=True)

intel = IntelligenceManager(seed=42)
intel.initialize_from_kg(kg)

# 3-step ISR collection
for _ in range(3):
    intel.step(kg, ForceAlignment.BLUE, ems_jamming=0.1)

# Pipeline integration outputs
isr_quality  = intel.get_isr_quality(ForceAlignment.BLUE, kg)      # → CombatDynamicsManager
unc_map      = intel.get_uncertainty_map(ForceAlignment.BLUE, kg)  # → build_state_vector
gnn_features = intel.get_gnn_state_features(ForceAlignment.BLUE, kg)  # 8D gnn_extension
node_weights = intel.get_node_uncertainty_weights(ForceAlignment.BLUE, kg)  # [1.0–2.0×]
```

### `ontology/roe_ethics.py` — Rules of Engagement

```python
from ontology.roe_ethics import ROEManager

roe = ROEManager(preset_name="conventional_war")  # or peacekeeping/counterinsurgency/...
approved, penalty, reasons = roe.check_engagement(
    attacker=blue_unit, target=red_unit, kg=kg,
    military_advantage=0.6, blast_radius_km=2.0
)
# penalty → BlueAgent.compute_reward() deduction
```

### `ontology/scenario_presets.py` — Realistic Scenarios

```python
from ontology.scenario_presets import load_scenario

kg, c2_mgr = load_scenario("korea_defense")      # 16 Blue + 10 Red, 100km map
kg, c2_mgr = load_scenario("air_superiority")    # 8 Blue + 7 Red, 200km map
kg, c2_mgr = load_scenario("urban_warfare")      # 7 Blue + 6 Red, 10km city
kg, c2_mgr = load_scenario("multidomain_contest") # 15 Blue + 10 Red, all 5 domains
kg, c2_mgr = load_scenario("cyber_ew")           # 7 Blue + 6 Red, EW-heavy
```

### Full pipeline with all integrations

```python
from rl_agent.blue_agent import build_state_vector
from simulator.combat_dynamics import CombatDynamicsManager

# C2 quality
c2_quality = c2_mgr.get_overall_c2_quality(ForceAlignment.BLUE, kg)
jf_avail   = c2_mgr.get_joint_fires_available(ForceAlignment.BLUE, kg)

# Build 128D state (intel auto-integrated)
state = build_state_vector(
    kg,
    c2_quality=c2_quality,
    joint_fires_available=jf_avail,
    intel_manager=intel,   # ← gnn_extension + uncertainty_map auto-computed
)  # shape=(128,)

# Combat dynamics with intel auto-step
dyn = CombatDynamicsManager()
dyn.initialize_from_kg(kg)
result = dyn.step_update(kg, step_result, intel_manager=intel)
# result keys: bda_list, supply_summary, ems_state, gnn_unc_multiplier, isr_quality
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/falcon-combat-ai.git
cd falcon-combat-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run full demo

```bash
python demo.py
```

### Generate training data

```bash
python generate_data.py                    # 100 scenarios (~2s)
python generate_data.py --scenarios 500    # large-scale
python generate_data.py --quick            # 10 scenarios (CI)
```

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
python evaluate.py --monte-carlo 5000 --fog-level moderate --max-steps 50

# Fast smoke test
python evaluate.py --fast --no-progress

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

## 📁 Project Structure

```
falcon/
├── ontology/               Multi-domain combat knowledge ontology
│   ├── combat_schema.py        42 UnitType · 7 Branch · Capability(19-field)
│   ├── military_units.py       Korean military OOB constants
│   ├── joint_operations.py     C2 structures + joint fires
│   ├── scenario_presets.py     5 realistic scenario presets
│   ├── roe_ethics.py           IHL-based ROE + ethical constraint checker
│   ├── intelligence.py         8-source intel fusion + FogOfWar ← NEW
│   ├── multidomain.py          Domain synergy/suppression effects
│   ├── doctrine_encoder.py     ADP 9-principle auto-evaluation
│   └── temporal_extension.py   Time-series state tracking
├── simulator/              Physical combat environment
│   ├── mixed_lanchester.py     42-type Lanchester engagement engine
│   ├── combat_dynamics.py      BDA · ammo · supply · EMS ← NEW
│   ├── fog_of_war.py           Partial observation + curriculum
│   ├── maneuver_engine.py      A* path · LOS · flanking
│   ├── lanchester_engine.py    Base Lanchester ODE solver
│   └── resource_manager.py     Ammo · fuel · maintenance
├── gnn_model/              Bayesian uncertainty modeling
│   ├── bayesian_hgt.py         MC-Dropout HGT (node_in_dim=128)
│   ├── temporal_gnn.py         Time-series GNN
│   ├── uncertainty_utils.py    Epistemic/aleatoric decomposition
│   └── temperature_scaling.py  Calibration (ECE minimization)
├── rl_agent/               Reinforcement learning agents
│   ├── blue_agent.py           PPO Blue (STATE_DIM=128)
│   ├── red_agent.py            Adversarial Red agent
│   ├── self_play_trainer.py    Phase A→D curriculum
│   ├── mappo.py                Multi-agent PPO
│   ├── hierarchical_rl.py      Strategic→tactical→action HRL
│   ├── inverse_rl.py           Max-Entropy IRL from doctrine demos
│   └── league_selfplay.py      Population-based league
├── hitl/                   Human-in-the-Loop
│   ├── natural_language_interface.py   NL command → structured constraints
│   ├── pareto_generator.py             Multi-objective Pareto candidates
│   ├── constraint_parser.py            Hard/Soft/Ethical constraint parsing
│   ├── mc_pareto_validator.py          Monte Carlo Pareto validation
│   ├── preference_learner.py           Commander preference modeling
│   ├── bandit_preference.py            Bandit-based preference learning
│   ├── preference_reward_adapter.py    Preference → reward integration
│   └── realtime_replanner.py           Real-time replan trigger
├── explainability/         Explainability tools
│   ├── auto_aar.py             Automatic After-Action Review generation
│   ├── attention_viz.py        GNN attention weight visualization
│   └── counterfactual.py       Counterfactual "what-if" analysis
├── evaluation/             Evaluation framework
│   ├── monte_carlo.py          5,000-run robustness evaluation
│   ├── historical_benchmark.py 5-scenario tactical benchmark
│   └── metrics.py              Unified performance metrics
├── visualization/          Dashboard & charts
│   └── realtime_dashboard.py   Plotly real-time battle dashboard
├── tests/                  Test suite
├── docs/reports/           Analysis reports
│   ├── FALCON_COMPREHENSIVE_ANALYSIS.md    v1 analysis (2026-02-21)
│   ├── FALCON_COMPREHENSIVE_ANALYSIS_v2.md v2 analysis (2026-02-22) ← NEW
│   └── ONTOLOGY_DEVELOPMENT_ROADMAP.md     Ontology roadmap
├── train.py                Training entry point
├── evaluate.py             Evaluation entry point
├── demo.py                 End-to-end demo
└── generate_data.py        Data generation + visualization
```

---

## 📈 Key Metrics

| Metric | Target | Approach |
|--------|--------|----------|
| Force reduction | **15–25%** | Pareto optimization + ForceSize penalty |
| Win rate (vs baseline) | **±3%** | Robust Minimax self-play |
| GNN calibration | **ECE < 0.1** | Temperature Scaling |
| ISR detection rate | **70–100%** | Intel Ontology (3 steps) |
| Self-play convergence | **Nash Gap < 0.05** | League self-play |
| HITL adoption | **> 70%** | Preference learning + Pareto quality |

---

## ⚖️ Ethics & Scope

- This repository is for **research and education** on synthetic combat simulation.
- It does **not** use classified data.
- It does **not** grant lethal autonomy to AI systems — HITL preserves **meaningful human control**.
- `roe_ethics.py` implements IHL principles (proportionality, distinction, military necessity).
- ROE penalty is deducted from the RL reward function to enforce ethical constraints during learning.

---

## 📄 License

MIT License.

---

<div align="center">

**FALCON v3.0** — Multi-Domain Combat AI Research Framework

*42 UnitTypes · 128D State · 8-source Intel · IHL-compliant ROE · 5 Scenario Presets*

[![Code](https://img.shields.io/badge/Python-20%2C000+_lines-8B5CF6?style=flat-square)](.)
[![Ontology](https://img.shields.io/badge/Ontology-STEP_1--9_+_Intel-F59E0B?style=flat-square)](ontology/)

</div>
