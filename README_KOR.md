<div align="center">

# 🦅 FALCON : Force-multiplying Adaptive Learning & Cognitive Operation Network

### *다영역 온톨로지 기반 Bayesian GNN + 적대적 강화학습 + Human-in-the-Loop 전투 의사결정 지원 플랫폼*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![LOC](https://img.shields.io/badge/코드-26%2C500+_줄-8B5CF6?style=flat-square)](.)
[![UnitTypes](https://img.shields.io/badge/UnitType-42종-F59E0B?style=flat-square)](ontology/combat_schema.py)
[![Scenarios](https://img.shields.io/badge/시나리오-5종_프리셋-3B82F6?style=flat-square)](ontology/scenario_presets.py)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-16A34A?style=flat-square)](https://github.com)

*"동일한 전투 효과를 유지하면서 최소한의 병력으로 임무를 달성하는 AI 의사결정 프레임워크"*

</div>

---

## 1. 문제 정의

현대 전장에서 병력 손실 최소화와 전투 효율 극대화는 상충 관계에 있습니다. 지휘관은 매 순간 **안개전쟁(Fog of War)** 속에서 불완전한 정보로 병력을 운용해야 하며, 잘못된 판단은 곧 인명 피해로 직결됩니다.

기존 방법론의 한계와 FALCON의 해결책:

| 기존 한계 | FALCON의 해결책 |
|-----------|----------------|
| 안개전쟁 / 불완전 정보 | 8종 ISR 융합(DS 이론) → Bayesian FogOfWar → GNN 불확실성 정량화 |
| 고정 전술의 취약성 | 적대적 RL: RARL, NFSP, PSRO, League Self-Play |
| 시뮬레이션-현실 간 격차 | ACCEL 커리큘럼 + DomainRandomizer |
| 다중 에이전트 조율 | MAT(자기회귀 결합행동) + HAPPO 순차 업데이트 |
| AI 의사결정의 불투명성 | 교리 기반 Explainability + 자동 전투 후 분석(AAR) |
| 완전 자율화의 윤리 문제 | IHL 기반 ROE 검증기 + HITL 최종 결정권 |
| 다영역 복잡성 | 42종 UnitType — 육/해/공/사이버/우주/무인체계 |

---

## 2. 시스템 개요

FALCON은 9개 핵심 기능을 하나의 통합 파이프라인으로 연결합니다:

1. **다영역 전투 온톨로지** — 42종 유닛, 7개 군종, C2 계층구조, ROE, 정보 온톨로지
2. **Bayesian GNN** — 이종(heterogeneous) 전투 그래프에서 인식론적·우연적 불확실성 정량화
3. **적대적 RL 스위트** — RARL, NFSP, PSRO+α-Rank, MAT, HAPPO, League Self-Play, ACCEL
4. **HITL 의사결정 지원** — 지휘관 제약 기반 Pareto 최적화 + 선호도 학습
5. **전투 역학** — 란체스터 소모, BDA, 탄약, 보급망, 전자기 스펙트럼
6. **정보 온톨로지** — Dempster-Shafer 이론 기반 HUMINT/SIGINT/IMINT/MASINT 8종 융합
7. **계층적 RL** — 전략 → 전술 → 행동 분해
8. **역강화학습** — 교범 시연 데이터에서 보상함수 자동 추출
9. **설명 가능성** — 자동 전후분석(AAR), GNN 어텐션 시각화, 반사실 분석

---

## 3. 전체 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       FALCON 엔드-투-엔드 파이프라인                           │
└──────────────────────────────────────────────────────────────────────────┘

[0] 정보 계층              ontology/intelligence.py
    8종 ISR (HUMINT/SIGINT/IMINT/MASINT/OSINT/TECHINT/CYBER/SPACE)
    → Dempster-Shafer 융합 → FogOfWarState → 8D GNN 확장
           │
           ▼
[1] 지식·시나리오 계층
    ontology/combat_schema     42 UnitType · 7 Branch · 7 Domain · 19-field Capability
    ontology/joint_operations  C2Link · CommandStructure · JointFiresRequest
    ontology/scenario_presets  5 프리셋 (한반도 방어·공중우세·시가전·다영역·사이버/전자전)
    ontology/roe_ethics        IHL 기반 ROE · EthicalConstraintChecker
    ontology/multidomain       도메인 시너지/억제 효과
           │
           ▼
[2] 시뮬레이션·역학 계층
    simulator/mixed_lanchester   42종 혼합 란체스터 교전 엔진
    simulator/combat_dynamics    BDA · 탄약 · 보급 · ElectromagneticEnv
    simulator/fog_of_war         부분 관측 + 커리큘럼
    simulator/maneuver_engine    A* 경로 · LOS · 기동 판정
    simulator/adversarial_scenario   DomainRandomizer + ACCEL 커리큘럼   
           │
           ▼
[3] 불확실성 모델링 계층
    gnn_model/bayesian_hgt       MC-Dropout HGT (node_in_dim=128)
    gnn_model/temporal_gnn       전투 추이 예측
    gnn_model/uncertainty_utils  인식론적/우연적 불확실성 분리
    gnn_model/temperature_scaling ECE 최소화 보정
           │
           ▼
[4] 적대적 RL 계층                                           
    ┌── P1: 집단 기반 자기 대전 ─────────────────────────────────────────────┐
    │   rl_agent/psro_oracle     PSRO + α-Rank (NeurIPS 2017 / SA 2019) 
    │   rl_agent/nfsp_agent      NFSP BR+AS 이중 네트워크 (NIPS 2016)       
    │   rl_agent/nfsp_buffer     ReservoirBuffer + CircularBuffer        
    │   rl_agent/league_selfplay PFSP · ELO · 스냅샷 리그               
    └───────────────────────────────────────────────────────────────────┘
    ┌── P2: 다중 에이전트 조율 ────────────────────────────────────────────┐
    │   rl_agent/mat_policy      MAT 자기회귀 결합행동 (NeurIPS 2022)     
    │   rl_agent/mappo           MAPPO + HAPPO 순차 업데이트              
    └─────────────────────────────────────────────────────────────────┘
    ┌── P3: 강건성·커리큘럼 ──────────────────────────────────────────────────┐
    │   rl_agent/rarl            RARL + SA-PPO (ICML 2017 / NeurIPS 2020)
    │   simulator/adversarial_scenario  ACCEL + DomainRandomizer         
    └─────────────────────────────────────────────────────────────────────┘
    rl_agent/blue_agent         PPO Blue (STATE_DIM=128)
    rl_agent/red_agent          적군 PPO
    rl_agent/self_play_trainer  Phase A→D 커리큘럼
    rl_agent/hierarchical_rl    전략→전술→행동 HRL
    rl_agent/inverse_rl         Max-Entropy IRL
           │
           ├──────────────────────────────────────────┐
           ▼                                          ▼
[5A] Human-in-the-Loop                       [5B] 설명 가능성
     hitl/constraint_parser                       explainability/auto_aar
     hitl/pareto_generator                        explainability/attention_viz
     hitl/preference_learner                      explainability/counterfactual
     hitl/bandit_preference
     hitl/realtime_replanner
           │                                          │
           └──────────────────┬───────────────────────┘
                              ▼
[6] 평가·보고
    evaluation/monte_carlo           5,000회 강건성 평가
    evaluation/historical_benchmark  5개 전술 시나리오 벤치마크
    evaluation/adversarial_benchmark 적대적 강건성 지표
```

---

## 4. 신규 기능 — 적대적 RL 개선 

FALCON v4.0은 기존 다중 에이전트 프레임워크 위에 포괄적인 적대적 RL 스위트를 추가합니다.

### P1: 집단 기반 자기 대전

#### `rl_agent/psro_oracle.py` — PSRO + α-Rank

Policy Space Response Oracles (Lanctot et al., NeurIPS 2017) + α-Rank 진화 게임 이론 전략 순위화 (Omidshafiei et al., Science Advances 2019).

```python
from rl_agent.psro_oracle import PSROOracle, PSROConfig
from rl_agent.league_selfplay import LeagueManager

oracle = PSROOracle(PSROConfig(alpha=10.0, mixture_type='alpha_rank'), seed=0)
league = LeagueManager(n_main=2, n_main_exp=2, n_league_exp=2, seed=0)

for agent in list(league.agents.values()) + league.snapshots:
    oracle.add_strategy(agent)

oracle.record_result(focal_id, opponent_id, result=0.72)

scores = oracle.alpha_rank()           # {agent_id: float}, 합계=1.0
focal_id, opp_id = oracle.psro_iteration()  # 다음 학습 쌍 선택
```

핵심 클래스: `PayoffMatrix`, `AlphaRankCalculator`, `PSROOracle`

#### `rl_agent/nfsp_agent.py` + `rl_agent/nfsp_buffer.py` — NFSP

Neural Fictitious Self-Play (Heinrich & Silver, NIPS 2016). BR(최선 응답) + AS(평균 전략) 이중 네트워크로 근사 내쉬 균형 수렴 보장.

```python
from rl_agent.nfsp_agent import NFSPAgent, NFSPAgentConfig

blue = NFSPAgent(n_actions=6, state_dim=128, config=NFSPAgentConfig(eta=0.2))

use_br = blue.should_use_br()                           # η=0.2 확률로 BR 선택
action, log_prob, value = blue.select_action(state, use_br=use_br)
blue.store(state, action, reward, next_state, done, log_prob, value, use_br)
metrics = blue.update()  # BR→PPO 업데이트, AS→CrossEntropy 지도학습
```

버퍼 구조:
- `CircularBuffer`: BR 네트워크 학습용 최근 경험 재생 (RL)
- `ReservoirBuffer`: AS 네트워크 학습용 균등 샘플링 저수지 (지도학습)

#### `rl_agent/league_selfplay.py` — League Self-Play

ELO 레이팅, PFSP(우선순위 허구적 자기 대전) 스케줄링, 자동 스냅샷 기반 집단 리그.

```python
from rl_agent.league_selfplay import LeagueManager, AgentRole

league = LeagueManager(n_main=2, n_main_exp=2, n_league_exp=2, seed=0)
candidates = league.get_candidates_for(league.agents["main_0"])
opponent   = league.pfsp.sample(focal, candidates)  # 승률 가중 샘플링
league.record_match(focal_id, opponent_id, result=0.65)
league.maybe_add_snapshot()                          # 주기적 Main 스냅샷
```

역할: `MAIN`(전체 대전), `MAIN_EXP`(Main 집중 공략), `LEAGUE_EXP`(전체 리그 공략)

---

### P2: 다중 에이전트 조율

#### `rl_agent/mat_policy.py` — Multi-Agent Transformer (MAT)

Wen et al., NeurIPS 2022. 자기회귀 결합행동 디코딩 — 각 에이전트가 이전 에이전트들의 행동을 조건으로 사용.

```
π(a₁,...,aₙ | o) = ∏ᵢ πᵢ(aᵢ | a₁,...,aᵢ₋₁, o)

Encoder:  o₁,...,oₙ → 컨텍스트 메모리  [Transformer Encoder]
Decoder:  메모리 + 이전_행동 → 행동 logit  [Transformer Decoder, 자기회귀]
```

전투 중 유닛 파괴로 인한 **가변 에이전트 수(N)**를 패딩 + valid_mask로 안전하게 처리.

```python
from rl_agent.mat_policy import MATConfig, MATTrainer, MATTransition

trainer = MATTrainer(MATConfig(obs_dim=128, n_actions=8, d_model=128, n_heads=8))
actions, log_probs, values = trainer.select_actions_np(obs_np, type_ids, action_masks)
trainer.buffer.add(MATTransition(obs, type_ids, actions, log_probs, values, rewards, dones, masks))
metrics = trainer.update()  # PPO + 가변 N 안전 GAE 계산
```

핵심 클래스: `MATEncoder`, `MATDecoder`, `MATPolicy`, `MATTrainer`

#### `rl_agent/mappo.py` — MAPPO + HAPPO

유닛 유형별 순차 업데이트 보장하는 **HAPPO(Heterogeneous-Agent PPO)** 통합. `UnitTransition`에 `unit_type` 필드를 추가하여 `infantry`, `armor`, `artillery` 등 실제 유닛 유형으로 HAPPO 그룹 업데이트를 수행 (이전의 잘못된 `blue`/`red` 그룹화 수정).

```python
manager = MAPPOManager(seed=42)
actions = manager.select_actions(kg)
manager.store_transitions(obs_dict, actions, rewards, global_value, dones)
result  = manager.happo_update()
# → {happo_actor_infantry: loss, happo_actor_armor: loss, n_types_updated: 5}
```

---

### P3: 강건성·커리큘럼

#### `rl_agent/rarl.py` — RARL + SA-PPO

Robust Adversarial RL (Pinto et al., ICML 2017) + State-Adversarial PPO (Zhang et al., NeurIPS 2020). 미니맥스 최적화:

```
max_θ min_ν  E[Σ rₜ | π_θ(프로타고니스트), ν_φ(관측_적대자)]
```

`ObsAdversary` 네트워크가 ε-ball 내에서 관측 벡터를 교란하여 프로타고니스트 성능을 최대한 저하시키도록 학습.

```python
from rl_agent.rarl import RARLTrainer, RARLConfig, add_adversarial_obs_noise

trainer = RARLTrainer(RARLConfig(epsilon=0.05, sa_ppo_enabled=True), n_actions=6, state_dim=128)
action, log_prob, value = trainer.select_action(obs, training=True)
metrics = trainer.update(obs_for_adversary=obs_batch)
# → rarl_protagonist_policy_loss, rarl_adversary_loss, rarl_sa_kl_loss, rarl_epsilon

rob = trainer.evaluate_robustness(obs_batch, n_trials=20)
# → action_consistency (행동 일관성), logit_stability_kl (로짓 안정성)

# 오프라인 평가용 3종 노이즈 주입
add_adversarial_obs_noise(obs, epsilon=0.05, mode='uniform')   # ±ε 균일
add_adversarial_obs_noise(obs, epsilon=0.05, mode='gaussian')  # σ = ε/3 가우시안
add_adversarial_obs_noise(obs, epsilon=0.05, mode='targeted')  # 기울기 기반
```

#### `simulator/adversarial_scenario.py` — ACCEL + DomainRandomizer

Automatic Curriculum through Environment Learning (Dennis et al., NeurIPS 2020) + 도메인 무작위화.

```python
from simulator.adversarial_scenario import DomainRandomizer, ACCELScheduler, ACCELConfig

rand  = DomainRandomizer(seed=42)
accel = ACCELScheduler(randomizer=rand, cfg=ACCELConfig(
    target_success_min=0.3,   # 목표 난이도 구간
    target_success_max=0.7,
), seed=0)

accel.initialize(n_initial=30)

rec = accel.sample()                                      # 버퍼에서 시나리오 샘플링
kg  = ScenarioFactory.create_standard_scenario(seed=rec.base_seed)
kg  = rand.apply(kg, rec.domain_sample)                   # 무작위 파라미터 적용

accel.update(rec, blue_won=True, regret=0.8)              # 성공률 업데이트
accel.mutate_and_add(rec, n=5)                            # 인근 시나리오 생성
accel.prune_easy_hard()                                   # 너무 쉽거나 어려운 제거
```

`DomainSample` 무작위화 대상: `force_scale`, `combat_power_scale`, `supply_disruption_prob`, `terrain_mobility`, `intel_delay_min`, `c2_quality`, `weather_effect`, `red_reinforcement_ratio`

---

## 5. 빠른 시작

### 설치

```bash
git clone https://github.com/Navy10021/falcon.git
cd falcon
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 전체 파이프라인 데모

```bash
python demo.py
```

### 훈련 온톨로지 데이터 생성

```bash
python generate_data.py                    # 기본 100개 시나리오
python generate_data.py --scenarios 500    # 대규모
python generate_data.py --quick            # 10개 (CI 연기 테스트)
```

### 인터랙티브 노트북

```bash
jupyter notebook notebook/FALCON.ipynb
```

---

## 6. 모듈별 상세 설명

### `ontology/` — 전투 지식 온톨로지 (4,922 LOC)

| 모듈 | 내용 |
|------|------|
| `combat_schema.py` | 42 `UnitType`, 7 `BranchType`, 7 `DomainType`, `Capability`(19 필드), `Unit`, `CombatKnowledgeGraph`, `ScenarioFactory` |
| `military_units.py` | 한국군 현실 편제 병력 규모 상수 테이블 |
| `joint_operations.py` | `C2Link`, `CommandStructure`, `JointFiresRequest`, `JointOperationsManager` |
| `scenario_presets.py` | 5개 시나리오 프리셋 — 한반도 방어, 공중우세, 시가전, 다영역, 사이버/전자전 |
| `roe_ethics.py` | `RulesOfEngagement`, `EthicalConstraintChecker`, `ROEManager` — IHL 비례성·식별성·필요성 |
| `intelligence.py` | `INTELReport`, `FogOfWarState`, `ISRAssetModel`(16종), `IntelligenceFusionEngine`(DS이론), `IntelligenceManager` |
| `multidomain.py` | 도메인 시너지/억제 효과 행렬 |
| `doctrine_encoder.py` | ADP 9대 원칙 자동 평가 |
| `temporal_extension.py` | 시계열 상태 추적 |

#### 42종 UnitType 분류

| 군종·도메인 | 유닛 종류 |
|------------|----------|
| **육군** | infantry, armor, artillery, mechanized_infantry, airborne, special_forces, anti_armor, air_defense_artillery, reconnaissance, nbc_defense, military_police, engineer, logistics |
| **해병대** | marine_infantry, amphibious_assault |
| **해군** | destroyer, frigate, submarine, mine_warfare, amphibious_ship, naval_aviation, coastal_defense |
| **공군** | fighter, strike_aircraft, bomber, isr_aircraft, tanker, transport, aew_aircraft |
| **미사일·드론** | ballistic_missile, cruise_missile, anti_ship_missile, sam_battery, rocket_artillery, uav_recon, uav_strike, loitering_munition, ugv, usv |
| **전략·사이버·우주** | cyber_unit, space_asset, psyops, strategic_missile, electronic_warfare, signal |

#### 5종 시나리오 프리셋

```python
from ontology.scenario_presets import load_scenario

kg, c2_mgr = load_scenario("korea_defense")       # Blue 16 vs Red 10 | 100km
kg, c2_mgr = load_scenario("air_superiority")     # Blue  8 vs Red  7 | 200km
kg, c2_mgr = load_scenario("urban_warfare")       # Blue  7 vs Red  6 |  10km
kg, c2_mgr = load_scenario("multidomain_contest") # Blue 15 vs Red 10 | 150km, 전 도메인
kg, c2_mgr = load_scenario("cyber_ew")            # Blue  7 vs Red  6 |  80km, EW 중심
```

---

### `simulator/` — 전투 환경 시뮬레이터 (4,937 LOC)

| 모듈 | 역할 |
|------|------|
| `mixed_lanchester.py` | 42종 이종 란체스터 교전 엔진 (항공 지원 포함) |
| `combat_dynamics.py` | `AmmoConsumptionModel` · `BDAEngine` · `SupplyStatusManager` · `ElectromagneticEnv` |
| `adversarial_scenario.py` | `DomainRandomizer` + `ACCELScheduler` — 커리큘럼 & 도메인 무작위화 |
| `fog_of_war.py` | 부분 관측, 탐지 모델링, 커리큘럼 가시성 |
| `maneuver_engine.py` | A* 경로 탐색, LOS 계산, 우회 기동 판정 |
| `naval_engine.py` | 해상 교전 역학 |
| `cyber_effects.py` | 사이버/전자전 공격·방어 능력 효과 |
| `missile_model.py` | 탄도/순항/대함 미사일 교전 모델 |
| `weather_model.py` | 날씨 영향 (기동력·화력 감쇄) |
| `resource_manager.py` | 탄약·연료·정비 수명주기 |

---

### `gnn_model/` — Bayesian 불확실성 GNN (1,385 LOC)

```
gnn_model/
├── bayesian_hgt.py        MC-Dropout HGT (node_in_dim=128, 이종 그래프)
├── temporal_gnn.py        시계열 전투 추이 예측
├── uncertainty_utils.py   인식론적/우연적 불확실성 분리
└── temperature_scaling.py ECE 최소화 보정
```

128차원 노드 특성 벡터:
```
병력현황(10) + Blue상태(4) + Red상태(4)
+ Blue군종(7) + Red군종(7) + Blue도메인(7) + Red도메인(7)
+ 지형(6) + C2·합동화력(4)                          = 56D 기본
+ GNN 확장(8D) [ISR커버리지, 정보신선도, DS융합점수,
                FoW불확실도, SIGINT%, IMINT%, HUMINT%, 정보우위점수]
+ 시계열 특징(8D)
= 72D → 제로패딩 → 128D
```

---

### `rl_agent/` — 강화학습 에이전트 (5,633 LOC)

| 모듈 | 알고리즘 | 참조 논문 |
|------|---------|---------|
| `blue_agent.py` | PPO 프로타고니스트 (STATE_DIM=128) | Schulman et al., 2017 |
| `red_agent.py` | PPO 적군 | — |
| `self_play_trainer.py` | Phase A→D 커리큘럼 자기 대전 | — |
| `mappo.py` | MAPPO + HAPPO 순차 업데이트 | Yu et al., 2022 |
| `hierarchical_rl.py` | 전략→전술→행동 HRL | — |
| `inverse_rl.py` | 교범 기반 Max-Entropy IRL | Ziebart et al., 2008 |
| `league_selfplay.py` | 리그 + PFSP + ELO | Vinyals et al., 2019 |
| `psro_oracle.py` | PSRO + α-Rank | Lanctot 2017; Omidshafiei 2019 |
| `nfsp_agent.py` | NFSP BR+AS 이중 네트워크 | Heinrich & Silver, 2016 |
| `nfsp_buffer.py` | ReservoirBuffer + CircularBuffer | — |
| `mat_policy.py` | Multi-Agent Transformer | Wen et al., NeurIPS 2022 |
| `rarl.py` | RARL + SA-PPO | Pinto 2017; Zhang et al., 2020 |

#### 보상 함수

```
R = w₁·승리 − w₂·사상자 − w₃·병력규모 − w₄·불확실성_패널티
  − w₅·교리_위반 − w₆·ROE_패널티
```

---

### `hitl/` — Human-in-the-Loop (2,270 LOC)

```
hitl/
├── natural_language_interface.py   자연어 명령 → 구조화 제약
├── pareto_generator.py             다목적 Pareto 후보 생성
├── constraint_parser.py            Hard/Soft/Ethical 제약 파서
├── mc_pareto_validator.py          Monte Carlo Pareto 검증
├── preference_learner.py           지휘관 선호도 패턴 학습
├── bandit_preference.py            Bandit 알고리즘 적응형 선호도
├── preference_reward_adapter.py    선호도 → RL 보상 연동
└── realtime_replanner.py           실시간 계획 재수립
```

지휘관 자연어 명령 예시:

```
입력: "303고지를 반드시 2시간 이내에 점령하라. 사상자는 50명 이내로 제한."

해석:
  HARD: max_time_steps = 12,  max_casualties = 50
  의도: SEIZE_OBJECTIVE (신뢰도 87%)

Pareto 후보:
  A — 최소 병력:  65명 | 승률 74% | 사상자 8명
  B — 균형형 :    80명 | 승률 81% | 사상자 6명   ← 권장
  C — 최고 승률:  95명 | 승률 89% | 사상자 4명
```

---

### `explainability/` — 설명 가능한 AI (920 LOC)

| 모듈 | 기능 |
|------|------|
| `auto_aar.py` | 자동 전후분석(AAR) — 결정적 전환점, 최선 결정, 교리 언어 개선 권고 |
| `attention_viz.py` | GNN 어텐션 가중치 히트맵 — 의사결정에 영향을 준 유닛/엣지 시각화 |
| `counterfactual.py` | "만약 X였다면?" 반사실 시나리오 분석 |

---

### `evaluation/` — 성능 검증 (1,259 LOC)

```bash
# Monte Carlo 강건성 (5,000회)
python evaluate.py --monte-carlo 5000 --fog-level moderate --max-steps 50

# 전술 시나리오 벤치마크
python evaluate.py --benchmark historical --benchmark-runs 10

# 빠른 연기 테스트
python evaluate.py --fast --no-progress
```

| 지표 | 목표치 |
|------|--------|
| 병력 감축률 | **15–25%** |
| 승률 변화폭 | **±3% 이내** |
| GNN 보정 (ECE) | **< 0.1** |
| ISR 탐지율 (3스텝) | **70–100%** |
| Self-Play Nash Gap | **< 0.05** |
| HITL 계획 채택률 | **> 70%** |

---

## 7. 훈련 및 평가

### 훈련 단계

```bash
# Phase 1 — Bayesian GNN + Blue PPO 기준선
python train.py --phase 1 --episodes 1000

# Phase 2 — 리그 기반 적대적 자기 대전
python train.py --phase 2 --episodes 5000

# Phase 3 — HITL 통합 루프
python train.py --phase 3 --episodes 2000 --hitl
```

### YAML 설정

```
configs/
├── default.yaml              전역 기본값 (lr, gamma, batch_size 등)
├── phase1.yaml               GNN + PPO 하이퍼파라미터
├── phase2.yaml               자기 대전 설정
├── phase3.yaml               HITL 설정
├── evaluation.yaml           평가 설정
└── scenarios/                시나리오별 파라미터 오버라이드
    ├── korea_defense.yaml
    ├── air_superiority.yaml
    ├── urban_warfare.yaml
    ├── multidomain_contest.yaml
    └── cyber_ew.yaml
```

---

## 8. 테스트

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

CI는 `main` 및 `claude/**` 브랜치 푸시 시 GitHub Actions를 통해 자동 실행됩니다.

---

## 9. 프로젝트 구조

```
falcon/
├── ontology/                    다영역 전투 지식 온톨로지 (4,922 LOC)
│   ├── combat_schema.py             42 UnitType · 7 Branch · Capability(19 필드)
│   ├── military_units.py            한국군 현실 편제 상수
│   ├── joint_operations.py          C2 구조 + 합동화력
│   ├── scenario_presets.py          5개 현실 시나리오 프리셋
│   ├── roe_ethics.py                IHL 기반 ROE + 윤리 검증기
│   ├── intelligence.py              8종 정보 융합 + FogOfWar
│   ├── multidomain.py               도메인 시너지/억제
│   ├── doctrine_encoder.py          ADP 9대 원칙 평가
│   └── temporal_extension.py        시계열 상태 추적
├── simulator/                   전투 환경 시뮬레이터 (4,937 LOC)
│   ├── mixed_lanchester.py          42종 혼합 란체스터 엔진
│   ├── combat_dynamics.py           BDA · 탄약 · 보급 · EMS
│   ├── adversarial_scenario.py      ACCEL + DomainRandomizer      ← 신규 P3
│   ├── fog_of_war.py                부분 관측
│   ├── maneuver_engine.py           A* · LOS · 기동 판정
│   ├── naval_engine.py              해상 교전
│   ├── cyber_effects.py             사이버/EW 효과
│   ├── missile_model.py             미사일 교전
│   └── weather_model.py             날씨 효과
├── gnn_model/                   Bayesian 불확실성 GNN (1,385 LOC)
│   ├── bayesian_hgt.py              MC-Dropout HGT (node_in_dim=128)
│   ├── temporal_gnn.py              시계열 GNN
│   ├── uncertainty_utils.py         인식론적/우연적 분리
│   └── temperature_scaling.py       ECE 보정
├── rl_agent/                    강화학습 에이전트 (5,633 LOC)
│   ├── blue_agent.py                PPO Blue (STATE_DIM=128)
│   ├── red_agent.py                 적군 PPO
│   ├── self_play_trainer.py         Phase A→D 커리큘럼
│   ├── mappo.py                     MAPPO + HAPPO 순차 업데이트
│   ├── hierarchical_rl.py           전략→전술→행동 HRL
│   ├── inverse_rl.py                Max-Entropy IRL
│   ├── league_selfplay.py           리그 + PFSP + ELO
│   ├── psro_oracle.py               PSRO + α-Rank               ← 신규 P1
│   ├── nfsp_agent.py                NFSP BR+AS 이중 네트워크    ← 신규 P1
│   ├── nfsp_buffer.py               Reservoir + Circular 버퍼   ← 신규 P1
│   ├── mat_policy.py                Multi-Agent Transformer     ← 신규 P2
│   └── rarl.py                      RARL + SA-PPO              ← 신규 P3
├── hitl/                        Human-in-the-Loop (2,270 LOC)
│   ├── natural_language_interface.py   자연어 → 구조화 제약
│   ├── pareto_generator.py              다목적 Pareto
│   ├── constraint_parser.py             Hard/Soft/Ethical 파서
│   ├── mc_pareto_validator.py           Monte Carlo 검증
│   ├── preference_learner.py            지휘관 선호도 학습
│   ├── bandit_preference.py             Bandit 선호도
│   ├── preference_reward_adapter.py     선호도 → 보상
│   └── realtime_replanner.py            실시간 재수립
├── explainability/              설명 가능성 (920 LOC)
│   ├── auto_aar.py                  자동 전후분석(AAR)
│   ├── attention_viz.py             GNN 어텐션 시각화
│   └── counterfactual.py            반사실 분석
├── evaluation/                  성능 평가 (1,259 LOC)
│   ├── monte_carlo.py               5,000회 강건성
│   ├── historical_benchmark.py      5개 시나리오 벤치마크
│   ├── adversarial_benchmark.py     적대적 강건성
│   └── metrics.py                   통합 지표
├── visualization/               Plotly 실시간 전투 대시보드 (799 LOC)
├── tests/                       테스트 스위트 — 9개 모듈 (1,939 LOC)
├── configs/                     YAML 설정 파일
├── docs/reports/                기술 분석 보고서
├── data/                        생성 훈련 데이터
├── checkpoints/                 훈련 체크포인트 & 로그
├── notebook/FALCON.ipynb        인터랙티브 주피터 노트북
├── train.py                     훈련 진입점 (806 LOC)
├── evaluate.py                  평가 진입점 (116 LOC)
├── demo.py                      엔드-투-엔드 데모 (374 LOC)
└── generate_data.py             데이터 생성 (925 LOC)

총계: ~26,500 줄의 Python 코드
```

---

## 10. 핵심 연구 기여 (5대 Novelty)

**기여 1. 불확실성 인식 전투 GNN**
Bayesian MC-Dropout HGT로 전장 불확실성을 정량화하고 인식론적·우연적 불확실성을 PPO 에이전트의 128D 상태 벡터에 직접 통합하는 최초의 통합 프레임워크.

**기여 2. 다중 출처 정보 융합 → GNN 불확실성**
8종 ISR(HUMINT/SIGINT/IMINT/MASINT/OSINT/TECHINT/CYBER/SPACE)을 Dempster-Shafer 신념 이론으로 융합하여 FogOfWar 상태를 GNN 노드 불확실성 가중치[1.0–2.0×]로 연결하는 통합 모델.

**기여 3. 포괄적 적대적 RL 스위트**
단일 프레임워크에서 RARL+SA-PPO(관측 강건성), NFSP(근사 내쉬 수렴), PSRO+α-Rank(집단 전략 순위화), MAT(자기회귀 다중 에이전트 조율), HAPPO(유닛 유형별 순차 업데이트), ACCEL(자동 난이도 커리큘럼) 통합.

**기여 4. 교리 기반 HITL + IHL 준수 ROE**
ADP 9대 원칙 자동 평가 + IHL ROE(비례성·식별성·군사적 필요성) 검증 + Pareto 지휘관 의사결정 지원 — Meaningful Human Control을 보장하는 단일 인터페이스.

**기여 5. 엔드-투-엔드 설명 가능 다영역 파이프라인**
42종 UnitType × 5개 도메인 × 5개 현실 시나리오에서 GNN 어텐션 가중치부터 교리 언어 자동 AAR 보고서까지 전 구간 연속 설명 가능성 제공.

---

## 11. 예상 성과

| 지표 | 목표치 | 달성 방법 |
|------|--------|---------|
| 병력 감축률 | **15–25%** | Pareto 최적화 + ForceSize 패널티 |
| 승률 유지 | **±3% 이내** | Robust Minimax Self-Play |
| GNN 불확실성 보정 | **ECE < 0.1** | Temperature Scaling |
| ISR 탐지율 | **70–100%** | 8종 정보 융합 (3스텝) |
| Self-Play 수렴 | **Nash Gap < 0.05** | League + PSRO + NFSP |
| HITL 채택률 | **> 70%** | 선호도 학습 + Pareto 품질 |
| 적대적 강건성 | **행동 일관성 > 0.8** | RARL + SA-PPO + ACCEL |

---

## 12. 개발 로드맵

```
현재 (v4.0) ─────────────── 완성
  ✅ Bayesian GNN + 128D 상태벡터
  ✅ Self-Play (Blue ↔ Red), League + PFSP
  ✅ PSRO + α-Rank (P1)
  ✅ NFSP BR+AS 이중 네트워크 (P1)
  ✅ MAPPO + HAPPO 유닛 유형별 업데이트 (P2)
  ✅ Multi-Agent Transformer (P2)
  ✅ RARL + SA-PPO 관측 강건성 (P3)
  ✅ ACCEL + DomainRandomizer 커리큘럼 (P3)
  ✅ HITL + 자연어 인터페이스
  ✅ 42종 UnitType 온톨로지
  ✅ 정보 온톨로지 (HUMINT/SIGINT/IMINT 등 8종)
  ✅ ROE + IHL 윤리 검증기
  ✅ 5개 현실 시나리오 프리셋

Phase 5 (단기 계획) ──────── 예정
  □ 비식별화된 실제 훈련 데이터 연동
  □ 심리전·인지전 작전 온톨로지
  □ 기후·환경 요소 온톨로지
  □ 연합작전 온톨로지
  □ LLM 기반 자연어 지휘 고도화
  □ 드론 스웜 조율 알고리즘
```

---

## 13. 면책 조항 및 윤리

> 본 시스템은 **완전히 합성된 가상 전투 데이터**를 사용하며, **학술 연구 및 AI 방법론 연구 목적으로만** 활용됩니다. 실제 작전 정보나 기밀 정보를 포함하지 않습니다.

- **자율 무기 반대**: AI가 최종 교전 결정을 내리지 않습니다. 인간 지휘관이 항상 최종 결정권을 보유합니다 (Meaningful Human Control).
- **IHL 준수**: ROE 검증기가 비례성·식별성·군사적 필요성 원칙을 자동 적용합니다.
- **훈련 중 강제**: ROE 패널티(`w₆`)가 보상 함수에 포함되어 학습 과정에서부터 윤리적 행동이 강제됩니다.
- **연구 목적 한정**: 훈련 시뮬레이션 보조 도구로만 사용되어야 합니다.

---

## 14. 라이선스

[MIT License](LICENSE) — 학술 연구, 교육, 비상업적 목적으로 자유롭게 사용 가능합니다.

---

<div align="center">

**FALCON v4.0** — 다영역 전투 AI 연구 프레임워크

*42종 UnitType · 128D 상태벡터 · 8종 ISR 융합 · PSRO + NFSP + MAT + RARL + ACCEL · IHL 준수 ROE · 5종 시나리오 프리셋*

[![Python LOC](https://img.shields.io/badge/Python-26%2C500+_줄-8B5CF6?style=flat-square)](.)
[![Ontology](https://img.shields.io/badge/온톨로지-9_모듈-F59E0B?style=flat-square)](ontology/)
[![Adversarial RL](https://img.shields.io/badge/적대적_RL-P1%2FP2%2FP3-EF4444?style=flat-square)](rl_agent/)

</div>
