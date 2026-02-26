<div align="center">

# FALCON: Force-Adaptive Learning for Combat Optimization Network

### *Ontology-Driven Bayesian GNN + Adversarial RL + Human-in-the-Loop Combat Decision Support*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![LOC](https://img.shields.io/badge/Code-26%2C500+_lines-8B5CF6?style=flat-square)](.)
[![UnitTypes](https://img.shields.io/badge/UnitTypes-42_kinds-F59E0B?style=flat-square)](ontology/combat_schema.py)
[![Scenarios](https://img.shields.io/badge/Scenarios-5_presets-3B82F6?style=flat-square)](ontology/scenario_presets.py)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-16A34A?style=flat-square)](https://github.com)

[Overview](#-overview) · [Architecture](#-system-architecture) · [What's New](#-whats-new--adversarial-rl-p1p2p3) · [Quickstart](#-quickstart) · [Modules](#-module-reference) · [Training](#-training--evaluation) · [Ethics](#-ethics--scope)

</div>

---

## Problem Statement

Modern battlefield decisions demand precise judgment under extreme uncertainty — partial observation, adversarial deception, information asymmetry, and time pressure. Classical optimization methods are brittle. Rule-based systems fail to generalize. Pure data-driven RL collapses against adaptive adversaries.

**FALCON** is an end-to-end research framework that addresses:

> *"How can we achieve the same combat effect with fewer forces — while remaining robust against adaptive adversaries and explainable to human commanders?"*

| Challenge | FALCON's Approach |
|-----------|-------------------|
| Fog of War / noisy intel | 8-source ISR fusion (DS theory) → Bayesian FogOfWar → GNN uncertainty |
| Brittle fixed-strategy behavior | Adversarial RL: RARL, NFSP, PSRO, League Self-Play |
| Generalization gap (sim-to-real) | ACCEL curriculum + DomainRandomizer |
| Multi-agent coordination | MAT (autoregressive joint action) + HAPPO sequential update |
| Black-box AI recommendations | Doctrine-aware explainability + Auto-AAR |
| Lethal autonomy concerns | IHL ROE checker + HITL final authority |
| Multi-domain complexity | 42 UnitTypes across Land / Sea / Air / Cyber / Space / UAS |

---

## Overview

FALCON integrates nine major capabilities in one coherent pipeline:

1. **Multi-domain combat ontology** — 42 unit types, 7 branches, C2 hierarchies, ROE, and intelligence ontology.
2. **Bayesian GNN** — Epistemic + aleatoric uncertainty quantification over heterogeneous combat graphs.
3. **Adversarial RL suite** — RARL, NFSP, PSRO + α-Rank, MAT, HAPPO, League Self-Play, ACCEL.
4. **HITL decision support** — Commander-constrained Pareto optimization with preference learning.
5. **Combat dynamics** — Lanchester attrition, BDA, ammunition, supply chains, electromagnetic spectrum.
6. **Intelligence ontology** — 8-source HUMINT/SIGINT/IMINT/MASINT fusion with Dempster-Shafer theory.
7. **Hierarchical RL** — Strategic → tactical → action decomposition.
8. **Inverse RL** — Reward extraction from doctrine demonstrations.
9. **Explainability** — Auto After-Action Review, GNN attention visualization, counterfactual analysis.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       FALCON END-TO-END PIPELINE                         │
└──────────────────────────────────────────────────────────────────────────┘

[0] Intelligence Layer         ontology/intelligence.py
    8-source ISR (HUMINT/SIGINT/IMINT/MASINT/OSINT/TECHINT/CYBER/SPACE)
    → Dempster-Shafer fusion → FogOfWarState → 8D GNN extension
           │
           ▼
[1] Knowledge & Scenario Layer
    ontology/combat_schema      42 UnitType · 7 Branch · 7 Domain · 19-field Capability
    ontology/joint_operations   C2Link · CommandStructure · JointFiresRequest
    ontology/scenario_presets   5 presets (KoreaDefense · AirSup · Urban · Multi · CyberEW)
    ontology/roe_ethics         IHL-based ROE · EthicalConstraintChecker
    ontology/multidomain        Domain synergy/suppression effects
           │
           ▼
[2] Simulation & Dynamics Layer
    simulator/mixed_lanchester   42-type Lanchester engagement engine
    simulator/combat_dynamics    BDA · ammo · supply · ElectromagneticEnv
    simulator/fog_of_war         Partial observation + curriculum
    simulator/maneuver_engine    A* path · LOS · flanking judgment
    simulator/adversarial_scenario   DomainRandomizer + ACCEL curriculum   
           │
           ▼
[3] Uncertainty Modeling Layer
    gnn_model/bayesian_hgt       MC-Dropout HGT (node_in_dim=128)
    gnn_model/temporal_gnn       Combat trajectory prediction
    gnn_model/uncertainty_utils  Epistemic / aleatoric decomposition
    gnn_model/temperature_scaling  ECE minimization calibration
           │
           ▼
[4] Adversarial RL Layer                                    ← EXPANDED P1/P2/P3
    ┌── P1: Population Methods ──────────────────────────────────────────────┐
    │   rl_agent/psro_oracle     PSRO + α-Rank (NeurIPS 2017 / SA 2019)      │
    │   rl_agent/nfsp_agent      NFSP BR+AS dual network (NIPS 2016)         │
    │   rl_agent/nfsp_buffer     ReservoirBuffer + CircularBuffer            │
    │   rl_agent/league_selfplay PFSP · ELO · snapshot league                │
    └────────────────────────────────────────────────────────────────────────┘
    ┌── P2: Multi-Agent Coordination ────────────────────────────────────────┐
    │   rl_agent/mat_policy      MAT autoregressive joint action (NeurIPS 22)│
    │   rl_agent/mappo           MAPPO + HAPPO sequential update             │
    └────────────────────────────────────────────────────────────────────────┘
    ┌── P3: Robustness & Curriculum ─────────────────────────────────────────┐
    │   rl_agent/rarl            RARL + SA-PPO (ICML 2017 / NeurIPS 2020)    │
    │   simulator/adversarial_scenario  ACCEL + DomainRandomizer             │
    └────────────────────────────────────────────────────────────────────────┘
    rl_agent/blue_agent         PPO Blue (STATE_DIM=128)
    rl_agent/red_agent          Adversarial Red agent
    rl_agent/self_play_trainer  Phase A→D curriculum
    rl_agent/hierarchical_rl    Strategic→tactical→action HRL
    rl_agent/inverse_rl         Max-Entropy IRL from doctrine demos
           │
           ├──────────────────────────────────────────┐
           ▼                                          ▼
[5A] Human-in-the-Loop                       [5B] Explainability
     hitl/constraint_parser                       explainability/auto_aar
     hitl/pareto_generator                        explainability/attention_viz
     hitl/preference_learner                      explainability/counterfactual
     hitl/bandit_preference
     hitl/realtime_replanner
           │                                          │
           └──────────────────┬───────────────────────┘
                              ▼
[6] Evaluation & Reporting
    evaluation/monte_carlo           5,000-run robustness
    evaluation/historical_benchmark  5-scenario tactical benchmark
    evaluation/adversarial_benchmark Adversarial robustness metrics
    evaluation/metrics               Unified performance metrics
```

---

## What's New — Adversarial RL (P1/P2/P3)

FALCON v4.0 introduces a comprehensive adversarial RL suite built on top of the existing multi-agent framework.

### P1: Population-Based Self-Play

#### `rl_agent/psro_oracle.py` — PSRO + α-Rank

Policy Space Response Oracles (Lanctot et al., NeurIPS 2017) + α-Rank evolutionary strategy ranking (Omidshafiei et al., Science Advances 2019).

```python
from rl_agent.psro_oracle import PSROOracle, PSROConfig
from rl_agent.league_selfplay import LeagueManager

oracle = PSROOracle(PSROConfig(alpha=10.0, mixture_type='alpha_rank'), seed=0)
league = LeagueManager(n_main=2, n_main_exp=2, n_league_exp=2, seed=0)

for agent in list(league.agents.values()) + league.snapshots:
    oracle.add_strategy(agent)

oracle.record_result(focal_id, opponent_id, result=0.72)

scores = oracle.alpha_rank()           # {agent_id: float}, sums to 1.0
focal_id, opp_id = oracle.psro_iteration()  # select next training pair
```

Key classes: `PayoffMatrix`, `AlphaRankCalculator`, `PSROOracle`

#### `rl_agent/nfsp_agent.py` + `rl_agent/nfsp_buffer.py` — NFSP

Neural Fictitious Self-Play (Heinrich & Silver, NIPS 2016). Dual BR (Best Response) + AS (Average Strategy) networks with reservoir sampling buffer for approximate Nash convergence.

```python
from rl_agent.nfsp_agent import NFSPAgent, NFSPAgentConfig, NFSPTrainer

blue = NFSPAgent(n_actions=6, state_dim=128, config=NFSPAgentConfig(eta=0.2))

use_br = blue.should_use_br()                           # η-probability of BR
action, log_prob, value = blue.select_action(state, use_br=use_br)
blue.store(state, action, reward, next_state, done, log_prob, value, use_br)
metrics = blue.update()  # PPO for BR network, CrossEntropy for AS network
```

Buffer internals:
- `CircularBuffer`: Recent experience replay for the BR network (RL)
- `ReservoirBuffer`: Uniform-random reservoir sampling for the AS network (supervised)

#### `rl_agent/league_selfplay.py` — League Self-Play

Population-based league with ELO rating, PFSP (Prioritized Fictitious Self-Play) scheduling, and automatic agent snapshotting.

```python
from rl_agent.league_selfplay import LeagueManager, AgentRole

league = LeagueManager(n_main=2, n_main_exp=2, n_league_exp=2, seed=0)
candidates = league.get_candidates_for(league.agents["main_0"])
opponent = league.pfsp.sample(focal, candidates)        # win-rate-weighted sampling
league.record_match(focal_id, opponent_id, result=0.65)
league.maybe_add_snapshot()                             # periodic Main snapshot
```

Roles: `MAIN` (trains vs all), `MAIN_EXP` (exploits Main), `LEAGUE_EXP` (exploits full league)

---

### P2: Multi-Agent Coordination

#### `rl_agent/mat_policy.py` — Multi-Agent Transformer (MAT)

Wen et al., NeurIPS 2022. Autoregressive joint action decoding — each agent conditions on the previous agents' actions.

```
π(a₁,...,aₙ | o) = ∏ᵢ πᵢ(aᵢ | a₁,...,aᵢ₋₁, o)

Encoder:  o₁,...,oₙ → context memory   [Transformer Encoder]
Decoder:  memory + prev_actions → logits  [Transformer Decoder, auto-regressive]
```

Handles **variable numbers of agents** (units destroyed mid-episode) via padding + valid_mask.

```python
from rl_agent.mat_policy import MATConfig, MATTrainer, MATTransition

trainer = MATTrainer(MATConfig(obs_dim=128, n_actions=8, d_model=128, n_heads=8))
actions, log_probs, values = trainer.select_actions_np(obs_np, type_ids, action_masks)
trainer.buffer.add(MATTransition(obs, type_ids, actions, log_probs, values, rewards, dones, masks))
metrics = trainer.update()   # PPO + GAE with variable-N safe handling
```

Key classes: `MATEncoder`, `MATDecoder`, `MATPolicy`, `MATTrainer`

#### `rl_agent/mappo.py` — MAPPO + HAPPO

Multi-Agent PPO with **HAPPO** (Heterogeneous-Agent PPO) sequential update by unit type. `UnitTransition` now carries `unit_type` so HAPPO correctly groups updates by `infantry`, `armor`, `artillery`, etc. (not by `blue`/`red` alignment).

```python
manager = MAPPOManager(seed=42)
actions = manager.select_actions(kg)
manager.store_transitions(obs_dict, actions, rewards, global_value, dones)
result = manager.happo_update()
# → {happo_actor_infantry: loss, happo_actor_armor: loss, n_types_updated: 5}
```

---

### P3: Robustness & Curriculum

#### `rl_agent/rarl.py` — RARL + SA-PPO

Robust Adversarial RL (Pinto et al., ICML 2017) combined with State-Adversarial PPO (Zhang et al., NeurIPS 2020). Minimax optimization:

```
max_θ min_ν  E[Σ rₜ | π_θ(protagonist), ν_φ(obs_adversary)]
```

```python
from rl_agent.rarl import RARLTrainer, RARLConfig, add_adversarial_obs_noise

trainer = RARLTrainer(RARLConfig(epsilon=0.05, sa_ppo_enabled=True), n_actions=6, state_dim=128)
action, log_prob, value = trainer.select_action(obs, training=True)
metrics = trainer.update(obs_for_adversary=obs_batch)
# → rarl_protagonist_policy_loss, rarl_adversary_loss, rarl_sa_kl_loss, rarl_epsilon

rob = trainer.evaluate_robustness(obs_batch, n_trials=20)
# → action_consistency, logit_stability_kl

# Three noise injection modes for offline evaluation
add_adversarial_obs_noise(obs, epsilon=0.05, mode='uniform')   # ±ε uniform
add_adversarial_obs_noise(obs, epsilon=0.05, mode='gaussian')  # σ = ε/3
add_adversarial_obs_noise(obs, epsilon=0.05, mode='targeted')  # gradient-like
```

Key classes: `ObsAdversary`, `RARLTrainer`

#### `simulator/adversarial_scenario.py` — ACCEL + DomainRandomizer

Automatic Curriculum through Environment Learning (Dennis et al., NeurIPS 2020) with domain randomization.

```python
from simulator.adversarial_scenario import DomainRandomizer, ACCELScheduler, ACCELConfig

rand  = DomainRandomizer(seed=42)
accel = ACCELScheduler(randomizer=rand, cfg=ACCELConfig(
    target_success_min=0.3, target_success_max=0.7,
), seed=0)

accel.initialize(n_initial=30)

rec = accel.sample()                                     # sample from buffer
kg  = ScenarioFactory.create_standard_scenario(seed=rec.base_seed)
kg  = rand.apply(kg, rec.domain_sample)                  # apply randomized params

accel.update(rec, blue_won=True, regret=0.8)             # update success rate
accel.mutate_and_add(rec, n=5)                           # neighbor scenarios
accel.prune_easy_hard()                                  # remove trivial/impossible
```

`DomainSample` randomizes: `force_scale`, `combat_power_scale`, `supply_disruption_prob`, `terrain_mobility`, `intel_delay_min`, `c2_quality`, `weather_effect`, `red_reinforcement_ratio`

Key classes: `DomainRandomizer`, `DomainSample`, `AdversarialCurriculum`, `ACCELScheduler`

---

## Quick Start

### Installation

```bash
git clone https://github.com/Navy10021/falcon.git
cd falcon
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### End-to-End Demo

```bash
python demo.py
```

### One-Command Artifact Demo (CPU-only)

```bash
python -m falcon.demo --scenario urban_defense --seed 42 --out runs/demo_urban
```

Fixed output artifacts:

- `runs/demo_urban/summary.json`
- `runs/demo_urban/metrics.csv`
- `runs/demo_urban/fig_episode.png`
- `runs/demo_urban/aar.html`

### Generate Training Data

```bash
python generate_data.py                    # 100 scenarios (default)
python generate_data.py --scenarios 500    # large-scale
python generate_data.py --quick            # 10 scenarios (CI smoke)
```

### Interactive Notebook

```bash
jupyter notebook notebook/FALCON.ipynb
```

---

## Module Reference

### `ontology/` — Combat Knowledge Ontology

| Module | Contents |
|--------|----------|
| `combat_schema.py` | 42 `UnitType`, 7 `BranchType`, 7 `DomainType`, `Capability` (19 fields), `Unit`, `CombatKnowledgeGraph`, `ScenarioFactory` |
| `military_units.py` | Korean military order-of-battle constants (headcount by echelon/type) |
| `joint_operations.py` | `C2Link`, `CommandStructure`, `JointFiresRequest`, `JointOperationsManager` |
| `scenario_presets.py` | 5 scenario presets — Korea Defense, Air Superiority, Urban Warfare, Multi-Domain, Cyber/EW |
| `roe_ethics.py` | `RulesOfEngagement`, `EthicalConstraintChecker`, `ROEManager` (IHL: proportionality, distinction, necessity) |
| `intelligence.py` | `INTELReport`, `FogOfWarState`, `ISRAssetModel` (16 ISR types), `IntelligenceFusionEngine` (DS theory), `IntelligenceManager` |
| `multidomain.py` | Domain synergy/suppression effect matrix |
| `doctrine_encoder.py` | ADP 9-principle automatic evaluation |
| `temporal_extension.py` | Time-series state tracking |

#### 42 UnitTypes by Domain

| Domain | Unit Types |
|--------|-----------|
| **Land (Army)** | infantry, armor, artillery, mechanized_infantry, airborne, special_forces, anti_armor, air_defense_artillery, reconnaissance, nbc_defense, military_police, engineer, logistics |
| **Land (Marine)** | marine_infantry, amphibious_assault |
| **Maritime** | destroyer, frigate, submarine, mine_warfare, amphibious_ship, naval_aviation, coastal_defense |
| **Air** | fighter, strike_aircraft, bomber, isr_aircraft, tanker, transport, aew_aircraft |
| **Missile / UAS** | ballistic_missile, cruise_missile, anti_ship_missile, sam_battery, rocket_artillery, uav_recon, uav_strike, loitering_munition, ugv, usv |
| **Strategic / Cyber** | cyber_unit, space_asset, psyops, strategic_missile, electronic_warfare, signal |

#### 5 Scenario Presets

```python
from ontology.scenario_presets import load_scenario

kg, c2_mgr = load_scenario("korea_defense")       # 16 Blue vs 10 Red | 100 km map
kg, c2_mgr = load_scenario("air_superiority")     #  8 Blue vs  7 Red | 200 km map
kg, c2_mgr = load_scenario("urban_warfare")       #  7 Blue vs  6 Red |  10 km city
kg, c2_mgr = load_scenario("multidomain_contest") # 15 Blue vs 10 Red | 150 km, all domains
kg, c2_mgr = load_scenario("cyber_ew")            #  7 Blue vs  6 Red |  80 km, EW-heavy
```

---

### `simulator/` — Combat Environment

| Module | Purpose |
|--------|---------|
| `mixed_lanchester.py` | 42-type heterogeneous Lanchester engagement engine with aviation support |
| `combat_dynamics.py` | `AmmoConsumptionModel` · `BDAEngine` · `SupplyStatusManager` · `ElectromagneticEnv` |
| `adversarial_scenario.py` | `DomainRandomizer` + `ACCELScheduler` — curriculum & domain randomization |
| `fog_of_war.py` | Partial observation, detection modeling, curriculum-based visibility |
| `maneuver_engine.py` | A* pathfinding, LOS calculation, flanking maneuver judgment |
| `naval_engine.py` | Maritime engagement mechanics |
| `cyber_effects.py` | Cyber/EW attack & defense capability effects |
| `missile_model.py` | Ballistic / cruise / anti-ship missile engagement model |
| `weather_model.py` | Weather degradation on unit capability and movement |
| `resource_manager.py` | Ammunition, fuel, maintenance lifecycle |
| `lanchester_engine.py` | Base Lanchester ODE solver (square / linear / mixed laws) |

---

### `gnn_model/` — Bayesian Uncertainty GNN

```
gnn_model/
├── bayesian_hgt.py        MC-Dropout HGT (node_in_dim=128, heterogeneous graph)
├── temporal_gnn.py        Time-series combat trajectory GNN
├── uncertainty_utils.py   Epistemic / aleatoric uncertainty decomposition
└── temperature_scaling.py ECE-minimization calibration
```

128D Node Feature Vector:
```
force_state(10) + blue_status(4) + red_status(4)
+ blue_branch(7) + red_branch(7) + blue_domain(7) + red_domain(7)
+ terrain(6) + c2_state(4)                              = 56D base
+ gnn_extension(8)  [ISR coverage, freshness, DS fusion score,
                     FoW uncertainty, SIGINT%, IMINT%, HUMINT%, intel score]
+ temporal_features(8)
= 72D → zero-padded to 128D
```

---

### `rl_agent/` — Reinforcement Learning Agents

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `blue_agent.py` | PPO protagonist (STATE_DIM=128) | Schulman et al., 2017 |
| `red_agent.py` | PPO adversarial Red | — |
| `self_play_trainer.py` | Phase A→D curriculum self-play | — |
| `mappo.py` | MAPPO + HAPPO sequential update | Yu et al., 2022 |
| `hierarchical_rl.py` | Strategic → tactical → action HRL | — |
| `inverse_rl.py` | Max-Entropy IRL from doctrine | Ziebart et al., 2008 |
| `league_selfplay.py` | League + PFSP + ELO | Vinyals et al., 2019 |
| `psro_oracle.py` | PSRO + α-Rank | Lanctot 2017; Omidshafiei 2019 |
| `nfsp_agent.py` | NFSP BR + AS dual network | Heinrich & Silver, 2016 |
| `nfsp_buffer.py` | Reservoir + Circular buffer | — |
| `mat_policy.py` | Multi-Agent Transformer | Wen et al., NeurIPS 2022 |
| `rarl.py` | RARL + SA-PPO | Pinto 2017; Zhang et al., 2020 |

#### State Vector (STATE_DIM = 128)

```python
from rl_agent.blue_agent import build_state_vector

state = build_state_vector(
    kg,
    c2_quality=c2_mgr.get_overall_c2_quality(ForceAlignment.BLUE, kg),
    joint_fires_available=c2_mgr.get_joint_fires_available(ForceAlignment.BLUE, kg),
    intel_manager=intel_mgr,   # auto-computes 8D GNN extension from ISR fusion
)  # → shape (128,)
```

#### Reward Function

```
R = w₁·Win − w₂·Casualty − w₃·ForceSize − w₄·UncertaintyPenalty
  − w₅·DoctrineViolation − w₆·ROE_Penalty
```

---

### `hitl/` — Human-in-the-Loop

```
hitl/
├── natural_language_interface.py   NL command → structured constraints
├── pareto_generator.py             Multi-objective Pareto candidate generation
├── constraint_parser.py            Hard / Soft / Ethical constraint parsing
├── mc_pareto_validator.py          Monte Carlo Pareto validation
├── preference_learner.py           Commander preference pattern learning
├── bandit_preference.py            Bandit-based adaptive preference learning
├── preference_reward_adapter.py    Preference → RL reward integration
└── realtime_replanner.py           Real-time replan trigger
```

Example commander interaction:

```
Input: "Seize Hill 303 within 2 hours. Limit casualties to 50."

Parsed:
  HARD: max_time_steps = 12,  max_casualties = 50
  Intent: SEIZE_OBJECTIVE (confidence 87%)

Pareto Candidates:
  Option A — Minimal Force:  65 troops | Win 74% | Casualties 8
  Option B — Balanced:       80 troops | Win 81% | Casualties 6   ← recommended
  Option C — High Assurance: 95 troops | Win 89% | Casualties 4
```

---

### `explainability/` — Interpretable AI

| Module | Function |
|--------|----------|
| `auto_aar.py` | Automatic After-Action Review — decision turning points, best decisions, improvement recommendations in doctrine language |
| `attention_viz.py` | GNN attention weight heatmap — which units and edges most influenced the decision |
| `counterfactual.py` | "What if X had been different?" counterfactual scenario analysis |

---

### `evaluation/` — Performance Validation

```bash
# Monte Carlo robustness (5,000 runs)
python evaluate.py --monte-carlo 5000 --fog-level moderate --max-steps 50

# Historical tactical benchmark
python evaluate.py --benchmark historical --benchmark-runs 10

# Fast smoke test
python evaluate.py --fast --no-progress
```

| Metric | Target |
|--------|--------|
| Force reduction vs baseline | **15–25%** |
| Win rate delta | **within ±3%** |
| GNN calibration (ECE) | **< 0.1** |
| ISR detection rate (3 steps) | **70–100%** |
| Self-play Nash Gap | **< 0.05** |
| HITL plan adoption rate | **> 70%** |

---

## Training & Evaluation

### Training Phases

```bash
# Phase 1 — Bayesian GNN + Blue PPO baseline
python train.py --phase 1 --episodes 1000

# Phase 2 — Adversarial self-play with league
python train.py --phase 2 --episodes 5000

# Phase 3 — HITL integration loop
python train.py --phase 3 --episodes 2000 --hitl
```

### YAML Configuration

```
configs/
├── default.yaml              global defaults (lr, gamma, batch_size, …)
├── phase1.yaml               GNN + PPO hyperparameters
├── phase2.yaml               self-play settings
├── phase3.yaml               HITL settings
├── evaluation.yaml           evaluation settings
└── scenarios/                per-scenario parameter overrides
    ├── korea_defense.yaml
    ├── air_superiority.yaml
    ├── urban_warfare.yaml
    ├── multidomain_contest.yaml
    └── cyber_ew.yaml
```

---

## Test Suite

```bash
PYTHONPATH=. python tests/test_tier1.py
PYTHONPATH=. python tests/test_tier2.py
PYTHONPATH=. python tests/test_tier3.py
PYTHONPATH=. python tests/test_phase1_immediate.py
PYTHONPATH=. python tests/test_phase2_short.py
PYTHONPATH=. python tests/test_phase3_medium.py
PYTHONPATH=. python tests/test_new_simulators.py
PYTHONPATH=. python tests/test_numerical_stability.py
```

CI runs automatically on all `main` and `claude/**` branches via GitHub Actions.

---

## Project Structure

```
falcon/
├── ontology/                    Multi-domain combat knowledge ontology (4,922 LOC)
│   ├── combat_schema.py             42 UnitType · 7 Branch · Capability (19 fields)
│   ├── military_units.py            Korean military OOB constants
│   ├── joint_operations.py          C2 structures + joint fires
│   ├── scenario_presets.py          5 realistic scenario presets
│   ├── roe_ethics.py                IHL-based ROE + ethical checker
│   ├── intelligence.py              8-source intel fusion + FogOfWar
│   ├── multidomain.py               Domain synergy/suppression
│   ├── doctrine_encoder.py          ADP 9-principle evaluation
│   └── temporal_extension.py        Time-series state tracking
├── simulator/                   Physical combat environment (4,937 LOC)
│   ├── mixed_lanchester.py          42-type Lanchester engine
│   ├── combat_dynamics.py           BDA · ammo · supply · EMS
│   ├── adversarial_scenario.py      ACCEL + DomainRandomizer       
│   ├── fog_of_war.py                Partial observation
│   ├── maneuver_engine.py           A* path · LOS · flanking
│   ├── naval_engine.py              Maritime engagement
│   ├── cyber_effects.py             Cyber/EW effects
│   ├── missile_model.py             Missile engagement
│   └── weather_model.py             Weather effects
├── gnn_model/                   Bayesian uncertainty GNN (1,385 LOC)
│   ├── bayesian_hgt.py              MC-Dropout HGT (node_in_dim=128)
│   ├── temporal_gnn.py              Time-series GNN
│   ├── uncertainty_utils.py         Epistemic/aleatoric decomposition
│   └── temperature_scaling.py       ECE calibration
├── rl_agent/                    Reinforcement learning agents (5,633 LOC)
│   ├── blue_agent.py                PPO Blue (STATE_DIM=128)
│   ├── red_agent.py                 Adversarial Red agent
│   ├── self_play_trainer.py         Phase A→D curriculum
│   ├── mappo.py                     MAPPO + HAPPO sequential update
│   ├── hierarchical_rl.py           Strategic→tactical→action HRL
│   ├── inverse_rl.py                Max-Entropy IRL
│   ├── league_selfplay.py           League + PFSP + ELO
│   ├── psro_oracle.py               PSRO + α-Rank              
│   ├── nfsp_agent.py                NFSP BR+AS dual network    
│   ├── nfsp_buffer.py               Reservoir + Circular buffer 
│   ├── mat_policy.py                Multi-Agent Transformer    
│   └── rarl.py                      RARL + SA-PPO             
├── hitl/                        Human-in-the-Loop (2,270 LOC)
│   ├── natural_language_interface.py   NL → structured constraints
│   ├── pareto_generator.py              Multi-objective Pareto
│   ├── constraint_parser.py             Hard/Soft/Ethical parsing
│   ├── mc_pareto_validator.py           Monte Carlo validation
│   ├── preference_learner.py            Commander preference
│   ├── bandit_preference.py             Bandit-based preference
│   ├── preference_reward_adapter.py     Preference → reward
│   └── realtime_replanner.py            Real-time replan
├── explainability/              Explainability tools (920 LOC)
│   ├── auto_aar.py                  Auto After-Action Review
│   ├── attention_viz.py             GNN attention visualization
│   └── counterfactual.py            Counterfactual what-if
├── evaluation/                  Evaluation framework (1,259 LOC)
│   ├── monte_carlo.py               5,000-run robustness
│   ├── historical_benchmark.py      5-scenario benchmark
│   ├── adversarial_benchmark.py     Adversarial robustness
│   └── metrics.py                   Unified metrics
├── visualization/               Plotly real-time dashboard (799 LOC)
├── tests/                       Test suite — 9 modules (1,939 LOC)
├── configs/                     YAML configuration files
├── docs/reports/                Technical analysis reports
├── data/                        Generated training data
├── checkpoints/                 Training checkpoints & logs
├── notebook/FALCON.ipynb        Interactive Jupyter notebook
├── train.py                     Training entry point (806 LOC)
├── evaluate.py                  Evaluation entry point (116 LOC)
├── demo.py                      End-to-end demo (374 LOC)
└── generate_data.py             Data generation (925 LOC)

Total: ~26,500 lines of Python
```

---

## Research Contributions

**1. Uncertainty-Aware Combat GNN**
First integrated framework quantifying battlefield uncertainty via Bayesian MC-Dropout HGT and feeding epistemic/aleatoric uncertainty directly into the PPO agent's 128D state vector.

**2. Multi-Source Intelligence → GNN Uncertainty**
8-source ISR integration (HUMINT/SIGINT/IMINT/MASINT/OSINT/TECHINT/CYBER/SPACE) using Dempster-Shafer belief theory, mapping FogOfWar state to GNN node uncertainty weights [1.0–2.0×].

**3. Comprehensive Adversarial RL Suite**
Single framework combining RARL + SA-PPO (observation robustness), NFSP (Nash convergence), PSRO + α-Rank (population strategy ranking), MAT (autoregressive multi-agent coordination), HAPPO (unit-type sequential update), and ACCEL (automatic difficulty curriculum).

**4. Doctrine-Aware HITL + IHL-Compliant ROE**
ADP 9-principle automatic evaluation + IHL ROE verification (proportionality / distinction / military necessity) + Pareto commander decision support, all within a single interface that preserves meaningful human control.

**5. End-to-End Explainable Multi-Domain Pipeline**
42 UnitTypes × 5 domains × 5 realistic scenarios with continuous explainability from GNN attention weights through doctrine-language Auto-AAR reports.

---

## Ethics & Scope

- This repository is for **research and educational use only** with fully synthetic simulation data.
- It does **not** use classified or real operational data.
- **No autonomous lethal decision-making**: The AI proposes candidate plans; the human commander retains final authority (Meaningful Human Control).
- `roe_ethics.py` enforces IHL principles (proportionality, distinction, military necessity) as an RL training penalty, not only as a post-hoc check.
- ROE penalty `w₆` is subtracted from the reward signal so the policy learns to avoid violations during training itself.

---

## License

[MIT License](LICENSE) — Free for academic research, education, and non-commercial use.

---

<div align="center">

**FALCON v4.0** — Multi-Domain Combat AI Research Framework

*42 UnitTypes · 128D State Vector · 8-source ISR Fusion · PSRO + NFSP + MAT + RARL + ACCEL · IHL-compliant ROE · 5 Scenario Presets*

[![Python LOC](https://img.shields.io/badge/Python-26%2C500+_lines-8B5CF6?style=flat-square)](.)
[![Ontology](https://img.shields.io/badge/Ontology-9_modules-F59E0B?style=flat-square)](ontology/)
[![Adversarial RL](https://img.shields.io/badge/Adversarial_RL-P1%2FP2%2FP3-EF4444?style=flat-square)](rl_agent/)

</div>
