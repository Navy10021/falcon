# 🧠 FALCON AI Combat Optimization System v3.0

<div align="center">

**병력 절감을 위한 AI 기술 전장 활용 방안**

*다영역 온톨로지 기반 Bayesian GNN + 정보 융합 + 강화학습 + HITL 전투 최적화 프레임워크*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Code](https://img.shields.io/badge/코드-20%2C000+_줄-8B5CF6?style=flat-square)](.)
[![UnitTypes](https://img.shields.io/badge/UnitType-42종-F59E0B?style=flat-square)](ontology/)
[![Scenarios](https://img.shields.io/badge/시나리오-5종_프리셋-3B82F6?style=flat-square)](ontology/scenario_presets.py)

*"동일한 전투 효과를 유지하면서 최소한의 병력으로 임무를 달성하는 AI 의사결정 프레임워크"*

</div>

---

## 1. 🎯 프로젝트 소개

### 문제 정의

현대 전장에서 병력 손실 최소화와 전투 효율 극대화는 상충 관계에 놓입니다. 지휘관은 매 순간 **안개전쟁(Fog of War)** 속에서 불완전한 정보로 병력을 운용해야 하며, 잘못된 판단은 곧 인명 피해로 직결됩니다.

| 기존 한계 | FALCON의 해결책 |
|-----------|----------------|
| 인간 지휘관의 인지 한계 | Bayesian GNN으로 전장 불확실성 정량화 |
| 고정 시나리오에 취약한 전술 | Self-Play RL로 적응형 적 대응 전략 학습 |
| AI 결정의 불투명성 | 교리 기반 Explainability + 자동 전후분석(AAR) |
| 완전 자율화의 윤리 문제 | Human-in-the-Loop + IHL 기반 ROE 검증기 |
| 단편적인 전장 인식 | 8종 정보 수집 출처 융합 → FogOfWar 불확실도 모델링 |

---

## 2. ⚡ TL;DR

```
8종 정보 출처(HUMINT/SIGINT/IMINT 등) → INTELReport 생성 → FogOfWar 불확실도 감소
    ↓
전장 온톨로지 (42종 UnitType · 7개 군종 · 5개 도메인 · C2 구조 · ROE)
    ↓
Bayesian GNN (node_in_dim=128) → 불확실성 분포 예측
    ↓
PPO 에이전트 (STATE_DIM=128) → 최소 병력 전술 Self-Play 학습
    ↓
Pareto 후보 전략 생성 → 지휘관 HITL 최종 선택
    ↓
Monte Carlo 강건성 검증 (5,000 runs) + 교리 준수 자동 평가
```

**핵심 수치:**

| 지표 | 목표치 | 달성 근거 |
|------|--------|-----------|
| 병력 감축률 | **15~25%** | Pareto 최적화 + ForceSize 패널티 |
| 승률 유지 | **±3% 이내** | Robust Minimax Self-Play |
| ISR 탐지율 | **70~100%** | 8종 정보 융합 (3스텝) |
| 불확실성 보정 | **ECE < 0.1** | Temperature Scaling |
| Self-Play 수렴 | **Nash Gap < 0.05** | League 자기 대전 |
| HITL 채택률 | **> 70%** | 선호도 학습 + Pareto 품질 |

---

## 3. 🔬 모듈별 기술 설명

### 3-1. 🗂️ Ontology Layer — 다영역 전장 지식 구조화

FALCON의 온톨로지 계층은 초기 8종 유닛에서 **42종 + 7개 군종 + 7개 도메인**으로 대폭 확장되었습니다.

```
ontology/
├── combat_schema.py        # 42 UnitType · 7 Branch · 7 Domain · 19-field Capability
├── military_units.py       # 한국군 현실 편제 병력 규모 테이블
├── joint_operations.py     # C2구조 · 합동화력 요청 · JointOperationsManager
├── scenario_presets.py     # 5개 현실 시나리오 프리셋 (한반도·공중·시가·다영역·사이버)
├── roe_ethics.py           # 교전규칙(ROE) · 비례성/식별성/필요성 검증기
├── intelligence.py         # 정보 온톨로지 (8종 출처 · DS 융합 · FogOfWar) ← 신규
├── multidomain.py          # 도메인 시너지/억제 효과
├── doctrine_encoder.py     # ADP 9대 원칙 자동 평가
└── temporal_extension.py   # 시계열 상태 추적
```

#### 42종 UnitType 분류

| 군종 | 유닛 종류 |
|------|----------|
| **육군** | 보병·기갑·포병·기계화보병·공수·특수전·대전차·방공포병·수색·화생방·헌병·공병·군수 |
| **해병대** | 해병보병·상륙돌격 |
| **해군** | 구축함·호위함·잠수함·기뢰전·상륙함·함재항공·해안방어 |
| **공군** | 전투기·공격기·폭격기·ISR기·급유기·수송기·조기경보기 |
| **미사일·드론** | 탄도미사일·순항미사일·대함미사일·SAM·다연장로켓·UAV정찰·UAV타격·배회탄약·UGV·USV |
| **전략·사이버·우주** | 사이버전·우주자산·심리전·전략미사일·전자전·신호 |

#### 시나리오 프리셋 5종

```python
from ontology.scenario_presets import load_scenario

kg, c2_mgr = load_scenario("korea_defense")       # 한반도 방어: Blue 16 vs Red 10, 100km
kg, c2_mgr = load_scenario("air_superiority")     # 공중전: Blue 8 vs Red 7, 200km
kg, c2_mgr = load_scenario("urban_warfare")       # 시가전: Blue 7 vs Red 6, 10km
kg, c2_mgr = load_scenario("multidomain_contest") # 다영역: Blue 15 vs Red 10, 150km
kg, c2_mgr = load_scenario("cyber_ew")            # 사이버EW: Blue 7 vs Red 6, 80km
```

---

### 3-2. 🔍 Intelligence Layer — 정보 온톨로지 ← 신규

```
ontology/intelligence.py
```

8종 정보 수집 출처를 통합하여 안개전쟁 불확실도를 체계적으로 관리합니다.

```
수집 출처:  HUMINT · SIGINT · IMINT · MASINT · OSINT · TECHINT · CYBER · SPACE
신뢰도:     NATO STANAG 2511 기준 (A~F 등급 × 1~6 확실도)
융합 방식:  Dempster-Shafer 이론 기반 다중출처 융합
```

```python
intel = IntelligenceManager(seed=42)
intel.initialize_from_kg(kg)

# 3스텝 ISR 수집 → FogOfWar 불확실도 감소
for _ in range(3):
    intel.step(kg, ForceAlignment.BLUE, ems_jamming=0.15)

# 파이프라인 연동값 출력
isr_quality   = intel.get_isr_quality(...)        # → CombatDynamicsManager (BDA 신뢰도)
unc_map       = intel.get_uncertainty_map(...)    # → build_state_vector (불확실도 맵)
gnn_features  = intel.get_gnn_state_features(...) # 8D → GNN 확장 슬롯
node_weights  = intel.get_node_uncertainty_weights(...) # [1.0~2.0×] GNN 불확실성 배수
```

**EMS 재밍 효과 예시 (cyber_ew 시나리오):**

| 재밍 강도 | 보고서 수 | ISR 품질 |
|---------|---------|--------|
| 0.0 (없음) | 13건 | 0.698 |
| 0.3 (약) | 8건 | 0.567 |
| 0.6 (중) | 8건 | 0.567 |
| 0.9 (강) | 6건 | 0.561 |

---

### 3-3. 📜 ROE / Ethics Layer — 교전규칙 검증기

```
ontology/roe_ethics.py
```

국제인도법(IHL) 기반 3대 원칙 자동 검증 + RL 보상 연동:

- **비례성 (Proportionality)**: 군사적 이득 / 민간피해 비율 ≥ 기준치
- **식별성 (Distinction)**: 전투원 vs 비전투원 구분
- **군사적 필요성 (Military Necessity)**: 최소 필요 무력 사용

```python
roe = ROEManager(preset_name="conventional_war")
approved, penalty, reasons = roe.check_engagement(attacker, target, kg)
# penalty → BlueAgent.compute_reward() 감점 반영
```

**임무별 ROE 프리셋 5종:**
`peacekeeping` · `conventional_war` · `counterinsurgency` · `counterterrorism` · `humanitarian`

---

### 3-4. ⚔️ Simulator Layer — 현실 전투 역학

```
simulator/
├── mixed_lanchester.py    # 42종 UnitType 혼합 란체스터 전투
├── combat_dynamics.py     # BDA · 탄약소모 · 보급관리 · 전자기환경 ← 신규
├── fog_of_war.py          # 부분 관측 + 커리큘럼
├── maneuver_engine.py     # A* 경로 · LOS · 기동 판정
└── resource_manager.py    # 자원 관리
```

**`combat_dynamics.py` 신규 기능:**
- `AmmoConsumptionModel`: 42종 UnitType 교전당 탄약 소모
- `BDAEngine`: ISR 품질 기반 전투피해평가 (confidence 반영)
- `SupplyStatusManager`: 보급로 차단 → 전투력 감쇄 (×1.0/0.75/0.5/0.3)
- `ElectromagneticEnv`: EW 활동 → GPS 저하 + GNN 불확실성 배수 (1.0~2.0×)

---

### 3-5. 🕸️ GNN Layer — 불확실성 인식 전장 예측

```
gnn_model/
├── bayesian_hgt.py        # MC-Dropout HGT (node_in_dim=128)
├── temporal_gnn.py        # 시계열 전투 추이 예측
├── uncertainty_utils.py   # Epistemic/Aleatoric 불확실성 분리
└── temperature_scaling.py # ECE 최소화 보정
```

**정보 온톨로지 → GNN 연동:**
- `FogOfWar.overall_uncertainty` → 노드 에피스테믹 불확실성 가중치 [1.0~2.0×]
- `ElectromagneticEnv.gnn_unc_multiplier()` → EW 환경 불확실성 배수
- `IntelligenceManager.get_gnn_state_features()` → GNN 확장 슬롯 8D

---

### 3-6. 🤖 RL Agent Layer — 최소 병력 전략 학습

```
rl_agent/
├── blue_agent.py          # PPO Blue (STATE_DIM=128)
├── red_agent.py           # 적군 PPO (Adversarial)
├── self_play_trainer.py   # Phase A→D 커리큘럼
├── mappo.py               # Multi-Agent PPO
├── hierarchical_rl.py     # 계층적 RL (전략→전술→행동)
├── inverse_rl.py          # Max-Entropy IRL (교범→보상 자동 추출)
└── league_selfplay.py     # Population-based League
```

**확장된 상태 벡터 (STATE_DIM = 128):**

```
기본 56D: 병력현황(10) + Blue상태(4) + Red상태(4)
         + Blue군종(7) + Red군종(7) + Blue도메인(7) + Red도메인(7)
         + 지형(6) + C2·합동화력(4)
GNN 확장 8D: [ISR커버리지, 정보신선도, 융합신뢰도, FOW불확실도,
              SIGINT비율, IMINT비율, HUMINT비율, 정보우위점수]
시계열 8D: 전투 추이 트렌드
패딩 → 128D
```

**보상 함수:**
```
R = w₁·Win - w₂·Casualty - w₃·ForceSize - w₄·UncertaintyPenalty
  - w₅·DoctrineViolation - w₆·ROE_Penalty  ← ROE 감점 신규 추가
```

---

### 3-7. 🎯 HITL Layer — 인간-AI 협력

```
hitl/
├── natural_language_interface.py  # 자연어 명령 → 구조화 제약
├── pareto_generator.py            # Pareto 최적 전략 후보 생성
├── constraint_parser.py           # Hard/Soft/Ethical 제약 파서
├── mc_pareto_validator.py         # Monte Carlo Pareto 검증
├── preference_learner.py          # 지휘관 선호도 패턴 학습
├── bandit_preference.py           # Bandit 알고리즘 선호도 학습
├── preference_reward_adapter.py   # 선호도 → 보상 연동
└── realtime_replanner.py          # 실시간 계획 재수립
```

**지휘관 자연어 명령 예시:**
```
입력: "3번 고지를 반드시 2시간 이내에 점령하라. 사상자는 50명 이내로 제한."

해석:
  ✅ HARD: max_time_steps = 12
  ✅ HARD: max_casualties = 50
  ✅ Intent: SEIZE_OBJECTIVE (신뢰도 87%)

Pareto 후보:
  A — 최소 병력: 65명 | 승률 74% | 사상자 8명
  B — 균형 ⭐ :  80명 | 승률 81% | 사상자 6명
  C — 최고 승률: 95명 | 승률 89% | 사상자 4명
```

---

### 3-8. 💡 Explainability Layer — 설명 가능한 AI

```
explainability/
├── auto_aar.py      # 자동 전투 후 분석 (결정적 전환점·최선결정·개선권고)
├── attention_viz.py # GNN 어텐션 가중치 시각화
└── counterfactual.py # "만약 X였다면?" 반사실 분석
```

---

### 3-9. 📊 Evaluation Layer — 성능 검증

```
evaluation/
├── monte_carlo.py          # 5,000+ run 강건성 평가
├── historical_benchmark.py # 5개 전술 시나리오 벤치마크
└── metrics.py              # 통합 성능 지표
```

---

## 4. 🔄 전체 파이프라인

```
┌──────────────────────────────────────────────────────────────┐
│                    FALCON v3.0 파이프라인                     │
│                                                              │
│  🔍 Intel      →  🗂️ Ontology  →  ⚔️ Simulator  →  🕸️ GNN  │
│  8종 ISR 수집     42종 유닛KG    란체스터+BDA+EMS   Bayesian   │
│  FogOfWar 업데이트  C2·ROE·시나리오  탄약·보급관리   불확실성   │
│                                          ↓                   │
│  📊 Evaluation  ←  🎯 HITL      ←  🤖 RL Agent             │
│  MC 강건성 검증    Pareto+제약    PPO(128D)+Self-Play         │
│                          ↓                                   │
│                    💡 Explainability                          │
│                    AAR + 어텐션 + 반사실                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 🚀 실행 가이드

### 설치

```bash
git clone https://github.com/your-username/falcon-combat-ai.git
cd falcon-combat-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### 전체 파이프라인 데모

```bash
python demo.py
```

### 온톨로지 데이터 생성 (훈련 전 필수)

```bash
python generate_data.py                    # 기본 100개
python generate_data.py --scenarios 500    # 대규모
python generate_data.py --quick            # 빠른 검증 (10개)
```

### 훈련

```bash
python train.py --phase 1 --episodes 1000   # GNN + Blue PPO
python train.py --phase 2 --episodes 5000   # Self-Play
python train.py --phase 3 --episodes 2000 --hitl  # HITL 통합
```

### 평가

```bash
python evaluate.py --monte-carlo 5000
python evaluate.py --benchmark historical
python evaluate.py --fast --no-progress
```

### 테스트

```bash
PYTHONPATH=. python tests/test_tier1.py
PYTHONPATH=. python tests/test_tier2.py
PYTHONPATH=. python tests/test_tier3.py
PYTHONPATH=. python tests/test_phase1_immediate.py
PYTHONPATH=. python tests/test_phase2_short.py
PYTHONPATH=. python tests/test_phase3_medium.py
```

---

## 6. 📁 프로젝트 구조

```
falcon/
├── 🔍 ontology/intelligence.py      정보 온톨로지 (신규)
├── 🗂️ ontology/                     다영역 전장 지식 온톨로지
│   ├── combat_schema.py                42 UnitType · 19-field Capability
│   ├── joint_operations.py             C2 구조 + 합동화력
│   ├── scenario_presets.py             5개 현실 시나리오 프리셋
│   └── roe_ethics.py                   IHL 기반 ROE + 윤리 검증기
├── ⚔️ simulator/                    전투 환경 시뮬레이터
│   ├── mixed_lanchester.py             42종 혼합 란체스터 엔진
│   └── combat_dynamics.py              BDA · 탄약 · 보급 · EMS (신규)
├── 🕸️ gnn_model/                   Bayesian HGT-GNN (node_in_dim=128)
├── 🤖 rl_agent/blue_agent.py        PPO Blue (STATE_DIM=128)
├── 🎯 hitl/                         Human-in-the-Loop
├── 💡 explainability/               AAR · 어텐션 · 반사실 분석
├── 📊 evaluation/                   Monte Carlo · 벤치마크
├── 📺 visualization/                실시간 전투 대시보드
├── 🧪 tests/                        테스트 스위트
└── docs/reports/                    분석 보고서
    ├── FALCON_COMPREHENSIVE_ANALYSIS.md    v1 (2026-02-21)
    ├── FALCON_COMPREHENSIVE_ANALYSIS_v2.md v2 (2026-02-22) ← 신규
    └── ONTOLOGY_DEVELOPMENT_ROADMAP.md     온톨로지 로드맵
```

---

## 7. 📈 예상 성과 및 로드맵

### 핵심 성과

| 목표 지표 | 수치 | 달성 근거 |
|-----------|------|-----------|
| **병력 감축률** | **15~25%** | Pareto 최적화 + ForceSize 패널티 |
| **승률 유지** | **±3%** | Robust Minimax Self-Play |
| **불확실성 보정** | **ECE < 0.1** | Temperature Scaling |
| **HITL 채택률** | **> 70%** | 선호도 학습 + Pareto 품질 |

### 정성적 가치
- **즉각 활용 가능**: 훈련 시나리오 계획에 AI 보조 의사결정 즉시 적용
- **교리 준수 자동화**: ADP 9대 원칙 위반 실시간 경보
- **투명한 AI**: "왜 이 전략인가"를 교리 언어로 자동 설명
- **인명 보호**: 최소 병력 원칙 + ROE 감점으로 윤리적 행동 강제
- **정보 우위**: 8종 ISR 융합 → GNN 불확실성 감소 → 정밀 의사결정

### 로드맵

```
현재 (v3.0) ────────────────── 완성
  ✅ Bayesian GNN + 128D 상태벡터
  ✅ Self-Play (Blue ↔ Red)
  ✅ HITL + 자연어 인터페이스
  ✅ 42종 UnitType 온톨로지 (STEP 1-9 완성)
  ✅ 정보 온톨로지 (HUMINT/SIGINT/IMINT 등 8종)
  ✅ ROE + IHL 윤리 검증기
  ✅ 5개 현실 시나리오 프리셋

Phase 4 (계획) ─────────────── 단기
  🔲 실제 훈련 데이터 연동 (비식별화)
  🔲 심리전·영향력 작전 온톨로지 (6-2)
  🔲 기후/환경 요소 온톨로지 (6-3)
  🔲 연합작전 온톨로지 (6-4)
  🔲 LLM 기반 자연어 지휘 고도화

Phase 5 (장기) ─────────────── 중장기
  🔲 디지털 트윈 전장 환경
  🔲 드론 스웜 조율 알고리즘
  🔲 위성 ISR 공개 데이터 연동
  🔲 하이브리드전 온톨로지 (6-5)
```

---

## 8. 🏅 핵심 연구 기여 (5대 Novelty)

**기여 1. Uncertainty-Aware Combat GNN**
전장 불확실성을 Bayesian GNN으로 정량화하고 PPO 에이전트 상태에 통합하는 최초의 통합 프레임워크

**기여 2. Intelligence Fusion → GNN Uncertainty**
HUMINT/SIGINT/IMINT 등 8종 정보 출처를 Dempster-Shafer 융합하여 FogOfWar를 GNN 에피스테믹 불확실성으로 연결하는 최초의 통합 모델

**기여 3. Adversarial Combat Self-Play**
"적이 최적으로 반응할 때도 유효한" Robust Minimax 최소 병력 전략 — Red의 Deception → Blue의 GNN 불확실성 증가 → 강건한 전략 자연 학습

**기여 4. Doctrine-Aware HITL + IHL ROE**
ADP 9대 원칙 자동 평가 + IHL 기반 ROE 3원칙(비례성/식별성/필요성) 검증 + Pareto 생성 + Meaningful Human Control을 단일 인터페이스로 통합

**기여 5. End-to-End Explainable Multi-Domain Pipeline**
온톨로지 → GNN → RL → HITL 전 구간에서 교리 언어 기반 자동 설명 생성; 42종 유닛 × 5개 도메인 × 5개 시나리오 완전 연동

---

## 9. ⚠️ 면책 조항 & 윤리

> 본 시스템은 **완전히 합성된 가상 전투 데이터**를 사용하며, **학술 연구 및 AI 방법론 연구 목적으로만** 활용됩니다. 실제 작전 정보나 기밀 정보를 포함하지 않습니다.

- **자율 무기 반대**: AI가 최종 교전 결정을 내리지 않습니다. 인간 지휘관이 항상 최종 결정권을 보유합니다.
- **IHL 준수**: ROE 검증기가 비례성·식별성·군사적 필요성 원칙을 자동 적용합니다.
- **연구 목적 한정**: 훈련 시뮬레이션 보조 도구로만 사용되어야 합니다.
- **ROE 패널티**: RL 보상 함수에 ROE 위반 감점을 포함하여 학습 중에도 윤리적 행동을 강제합니다.

---

## 10. 📄 라이선스

MIT License — 학술 연구, 교육, 비상업적 목적으로 자유롭게 사용 가능합니다.

---

<div align="center">

**🧠 FALCON AI Combat Optimization System v3.0**

*"병력을 아끼는 것이 곧 사람을 아끼는 것이다"*

[![Code](https://img.shields.io/badge/코드-20%2C000+_줄-8B5CF6?style=flat-square)](.)
[![Ontology](https://img.shields.io/badge/온톨로지-STEP_1--9_+_Intel-F59E0B?style=flat-square)](ontology/)
[![Scenarios](https://img.shields.io/badge/시나리오-5종_현실_프리셋-3B82F6?style=flat-square)](ontology/scenario_presets.py)
[![UnitTypes](https://img.shields.io/badge/UnitType-42종-22C55E?style=flat-square)](ontology/combat_schema.py)

</div>
