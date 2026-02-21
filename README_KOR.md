# 🧠 AI Combat Optimization System v2.0

<div align="center">

**병력 절감을 위한 AI 기술 전장 활용 방안**

*Ontology-Driven GNN + Reinforcement Learning for Minimum-Force Combat Optimization*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-251%2F251_✓-22C55E?style=flat-square)](tests/)
[![Research](https://img.shields.io/badge/Purpose-국방_AI_경진대회-003087?style=flat-square)](https://ideamaicon.kr)

*"동일한 전투 효과를 유지하면서 최소한의 병력으로 임무를 달성하는 AI 의사결정 프레임워크"*

</div>

---

## 1. 🎯 Project Introduction

### 문제 정의: 왜 AI 전투 최적화인가?

현대 전장에서 병력 손실 최소화와 전투 효율 극대화는 상충 관계에 놓입니다. 지휘관은 매 순간 **불완전한 정보(Fog of War)** 속에서 병력 운용을 결정해야 하며, 잘못된 판단은 곧 인명 피해로 직결됩니다.

| 기존 한계 | 본 시스템의 해결책 |
|-----------|-------------------|
| 인간 지휘관의 인지 한계 | Bayesian GNN으로 전장 불확실성 정량화 |
| 고정 시나리오에 최적화된 취약한 전술 | Self-Play RL로 적응형 적 대응 전략 학습 |
| AI 결정의 불투명성 | 교리 기반 Explainability + 자동 전후분석 |
| 완전 자율화의 윤리 문제 | Human-in-the-Loop으로 지휘관 최종 통제권 보장 |

### 경진대회 주제 적합성

> **"병력 절감을 위한 AI 기술 전장 활용 방안"** (2026 제1차 국방 AI 활용 아이디어 경연대회)

- **병력 절감**: 동일 전투 효과 유지 하에 **15~25% 병력 감축** 달성
- **AI 기술 전장 활용**: 온톨로지 → GNN → RL → HITL 완전 End-to-End 파이프라인
- **실용적 AI**: 지휘관이 통제 가능한 설명 가능한 AI 의사결정 지원

---

## 2. ⚡ TL;DR

```
전장 온톨로지로 전투 지식을 구조화
    → Bayesian GNN이 불확실한 상황에서 위험도를 예측
    → PPO 에이전트가 최소 병력 전술을 Self-Play로 학습
    → Pareto 후보 전략을 생성해 지휘관에게 제시
    → 지휘관이 최종 선택 → AI가 선호도 학습 → 점점 더 맞춤화
    → Monte Carlo 강건성 검증 + 교리 준수 자동 평가
```

**핵심 수치:**

| 지표 | 목표치 | 의미 |
|------|--------|------|
| 병력 감축률 | **15~25%** | 동일 효과로 더 적은 병력 |
| 승률 유지 | **±3% 이내** | 성능 타협 최소화 |
| 불확실성 보정 | **ECE < 0.1** | GNN 예측 신뢰도 |
| Self-Play 수렴 | **Nash Gap < 0.05** | 적응형 전략 안정화 |
| HITL 채택률 | **> 70%** | 지휘관 AI 신뢰도 |

---

## 3. 🔬 Background — 모듈별 기술 설명

### 3-1. 🗂️ Ontology Layer — 전장 지식 구조화

```
ontology/
├── combat_schema.py        # 유닛·지형·능력치 클래스 + KnowledgeGraph
├── doctrine_encoder.py     # ADP 9대 원칙 준수도 자동 평가
├── multidomain.py          # 지상·공중·전자전 복합 도메인
└── temporal_extension.py   # 시계열 상태 추적
```

전장 온톨로지는 이 시스템의 언어입니다. 실제 군사 교리를 Python 클래스로 인코딩하여 AI가 전투 개념을 이해합니다.

```python
from ontology.combat_schema import ScenarioFactory
from ontology.doctrine_encoder import DoctrineEncoder

kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6)
# → 유닛 14개, 엣지 120개, 지형/능력치/관계 완전 구조화

doctrine = DoctrineEncoder()
compliance = doctrine.evaluate(kg, step=5, step_result=result)
# → mass: 0.82  offensive: 0.71  security: 0.65  (9개 원칙 자동 평가)
```

**ADP 9대 원칙** 실시간 자동 평가:
`MASS` · `OBJECTIVE` · `OFFENSIVE` · `SECURITY` · `SURPRISE` · `ECONOMY` · `MANEUVER` · `UNITY` · `SIMPLICITY`

---

### 3-2. ⚔️ Simulator Layer — 물리적 전투 환경

```
simulator/
├── lanchester_engine.py    # 란체스터 법칙 기반 교전 시뮬레이터
├── mixed_lanchester.py     # 혼합 전력 교전 (보병/기갑/항공/포병)
├── fog_of_war.py           # 부분 관측 환경 + Curriculum 스케줄러
├── maneuver_engine.py      # A* 경로 계획 + LOS + 측면기동 판정
└── resource_manager.py     # 탄약·연료·정비 자원 관리
```

**Fog of War 커리큘럼 (점진적 난이도 증가):**
```
완전관측 (0~20%) → 부분관측 (20~50%) → 강한안개 (50~80%) → 최대불확실 (80~100%)
```

**지형 기동 엔진 핵심 기능:**
- **A* 경로 계획**: 지형 이동 비용 고려 최적 경로 탐색
- **가시선(LOS) 체크**: Bresenham 직선 기반 실시간 계산
- **기동 판정**: 측면각도 > 45° → 측면기동, 포위커버리지 > 60% → 포위기동

---

### 3-3. 🕸️ GNN Layer — 전장 상황 인식

```
gnn_model/
├── bayesian_hgt.py          # MC-Dropout Heterogeneous Graph Transformer
├── temporal_gnn.py          # 시계열 GNN (전투 추이 예측)
├── uncertainty_utils.py     # Epistemic/Aleatoric 불확실성 분해
└── temperature_scaling.py   # 보정 후처리 (ECE 최소화)
```

기존 GNN이 단일 값을 출력하는 것과 달리, **Bayesian HGT-GNN**은 예측 분포를 출력합니다:

```
기존 GNN:   "예상 사상자 = 23명"
Bayesian:   "사상자 ~ N(23, 8²),  95% CI = [7, 39]"
             → 불확실성 높음 → 에이전트가 보수적 전략 선택
             → 불확실성 낮음 → 에이전트가 공격적 전략 선택
```

이종 그래프 노드 타입: `Unit` · `Terrain` · `Objective` · `Threat`
엣지 타입: `supports` · `threatens` · `occupies` · `observes`

---

### 3-4. 🤖 RL Agent Layer — 최소 병력 전략 학습

```
rl_agent/
├── blue_agent.py           # 아군 PPO 에이전트 (불확실성 상태 통합)
├── red_agent.py            # 적군 PPO 에이전트 (Adversarial)
├── self_play_trainer.py    # Self-Play 교대 학습 루프
├── mappo.py                # Multi-Agent PPO (MAPPO)
├── hierarchical_rl.py      # 계층적 RL (전략→전술→행동)
├── inverse_rl.py           # Max-Entropy IRL (교범→보상 자동 추출)
└── league_selfplay.py      # Population-based League 학습
```

**보상 함수:**
```
R = w₁·Win - w₂·Casualty - w₃·ForceSize - w₄·UncertaintyPenalty - w₅·DoctrineViolation
```

**Self-Play 학습 4단계:**

| 단계 | 설정 | 목적 |
|------|------|------|
| Phase 2-A | Red 고정(룰베이스), Blue만 학습 | 기본 전략 습득 |
| Phase 2-B | Blue 고정, Red만 학습 | 적응형 적 학습 |
| Phase 2-C | 교대 업데이트 (10 epoch) | Nash Equilibrium 수렴 |
| Phase 2-D | Population-based League | 과적합 방지 |

**역강화학습(IRL):** 전문가 교범 시범 데이터에서 보상 함수를 자동 추출합니다. "AI가 교범을 읽고 가치관을 스스로 형성"합니다.

---

### 3-5. 🎯 HITL Layer — 인간-AI 협력

```
hitl/
├── natural_language_interface.py  # 자연어 명령 → 구조화 제약
├── pareto_generator.py            # Pareto 최적 전략 후보 생성
├── constraint_parser.py           # Hard/Soft 제약 파서
├── mc_pareto_validator.py         # Monte Carlo Pareto 검증
├── bandit_preference.py           # Bandit 알고리즘 선호도 학습
├── preference_learner.py          # 지휘관 의사결정 패턴 학습
└── realtime_replanner.py          # 실시간 계획 재수립
```

지휘관은 자연어로 명령합니다:

```
입력: "3번 고지를 반드시 2시간 이내에 점령하라. 사상자는 50명 이내로 제한한다."

해석:
  ✅ HARD: max_time_steps = 12 (1시간=6스텝)
  ✅ HARD: max_casualties = 50
  ✅ Intent: SEIZE_OBJECTIVE (신뢰도 87%)
```

**Pareto 전략 후보:**

| 옵션 | 병력 | 승률 | 사상자 | 특성 |
|------|------|------|--------|------|
| A — 최소 병력 | 65명 | 74% | 8명 | 병력 절감 최우선 |
| **B — 균형** ⭐ | **80명** | **81%** | **6명** | **최적 균형점** |
| C — 최고 승률 | 95명 | 89% | 4명 | 확실성 우선 |

> **Meaningful Human Control**: AI가 최종 결정권을 갖지 않아 국제 인도법(IHL) 요건을 충족합니다.

---

### 3-6. 💡 Explainability Layer — 설명 가능한 AI

```
explainability/
├── auto_aar.py             # 자동 전투 후 분석 (After-Action Review)
├── attention_viz.py        # GNN 어텐션 가중치 시각화
└── counterfactual.py       # 반사실적 분석 ("만약 X였다면?")
```

**자동 AAR 출력 예시:**
```
📋 자동 전투 후 분석 — 🏆 WIN  (종합 점수 81%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
아군 손실: 187명(18%)  적군 손실: 342명  총 15스텝

⚡ 결정적 전환점:
  📈 스텝  7: 승률 51%→67% 급등 (측면기동 효과)
  📈 스텝 10: 측면기동 성공 → 적 89명 제압

✅ 최선의 결정:
  스텝 5 [측면기동]: SURPRISE·MANEUVER 원칙 적용

💡 개선 권고:
  1. 탄약 30% 도달 전 선제 보급 요청 루틴 수립 권장
```

---

### 3-7. 📊 Evaluation Layer — 성능 검증

```
evaluation/
├── monte_carlo.py          # 5,000+ run 강건성 평가
├── historical_benchmark.py # 5개 전술 시나리오 벤치마크
└── metrics.py              # 통합 성능 지표
```

**역사 벤치마크 시나리오:**

| 시나리오 | 지형 | 핵심 변수 | 난이도 |
|----------|------|-----------|--------|
| 상륙 돌격 | 해안 | 타이밍 | Hard |
| 도시 방어 | 시가지 | 소모전 | Medium |
| 포위 섬멸 | 평야 | 기동속도 | Medium |
| 산악 지연 | 산악 | 시간 | Easy |
| 산림 매복 | 산림 | 기습 | Easy |

---

## 4. 🔄 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────┐
                  FULL SYSTEM PIPELINE                    
                                                          
  🗂️ Ontology  ──▶  ⚔️ Simulator  ──▶  🕸️ GNN           
  전장 지식 구조화    전투 환경·Fog of War  상황인식·불확실성   
                                          │              
  📊 Evaluation  ◀── 🎯 HITL      ◀── 🤖 RL Agent       
  Monte Carlo 검증   인간-AI 협력      Self-Play 최적화    
                          │                              
                    💡 Explainability                     
                    AAR + 어텐션 시각화                    
└─────────────────────────────────────────────────────────┘
```

**데이터 흐름:**
```
1. 전장 온톨로지 → Knowledge Graph (이종 그래프)
2. Fog of War 필터 → 부분 관측 상태 벡터
3. Bayesian HGT-GNN → (예측값, 불확실성) 분포
4. PPO 상태 = [전투정보] + [GNN예측분포] + [교리준수도 9D] + [지휘관선호]
5. Blue PPO ↔ Red PPO Self-Play → Robust Minimax 전략
6. Pareto Generator → 3~5개 최적 후보 전략
7. HITL → 지휘관 선택 + 선호도 학습
8. Monte Carlo (5,000 runs) → 강건성 검증
9. Auto AAR → 전후 분석 리포트
```

---

## 5. 🚀 Quickstart

> **실행 순서**: 설치 → 데모 → **데이터 생성** → 훈련 → 평가
> 데이터 생성(`generate_data.py`)은 훈련 전 반드시 실행해야 합니다.

### Step 0. 설치

```bash
git clone https://github.com/your-username/ai-combat-optimization.git
cd ai-combat-optimization

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Step 1. 전체 파이프라인 데모

시스템의 End-to-End 흐름을 빠르게 확인합니다.

```bash
python demo.py
```

```
══════════ 🧠 AI Combat Optimization System v2.0 ══════════

STEP 1: 전장 온톨로지 & 시나리오 생성
  ✅ Knowledge Graph: 14개 유닛, 엣지 120개

STEP 2: Fog of War (부분 관측 환경)
  ✅ 관측 유닛: 9/14 (64.3%)  불확실성: 중간

STEP 3: Bayesian GNN 불확실성 예측
  ✅ 사상자 예측: 23.4명 ± 8.1  불확실성: 0.32

STEP 4: PPO Blue Agent 최적화
  ✅ 선택 행동: 측면기동  교리 준수도: 78%

STEP 5: HITL Pareto 전략 생성
  ⭐ 추천: Option B — 병력 80명, 승률 81%

STEP 6: Monte Carlo 강건성 검증 (1,000 runs)
  ✅ 승률: 79.2% ± 3.1%  병력 감축: 22%
```

---

### Step 2. 온톨로지 데이터 생성 & 시각화 ← **훈련 전 필수**

전장 시나리오·에피소드·IRL 데모 데이터를 생성하고, 통계를 인터랙티브 HTML로 시각화합니다.

```bash
# 기본 (100개 시나리오, ~2초)
python generate_data.py

# 대규모 학습용 (500개 시나리오)
python generate_data.py --scenarios 500 --irl-demos 50

# 빠른 검증 (10개, CI/테스트용)
python generate_data.py --quick
```

```
╔══════════════════════════════════════════════╗
║  🧠 AI Combat Optimization — 데이터 생성기  ║
╚══════════════════════════════════════════════╝
  시나리오: 100개 | IRL 데모: 20/전술 | 시드: 42

[1/4] 전장 시나리오 생성 (100개)...
  ✅ 시나리오 100개 → data/scenarios.json

[2/4] 시뮬레이션 에피소드 생성 (100개)...
  ✅ 에피소드 100개   Blue 승률: 52%  (Win:52 Loss:37 Draw:11)

[3/4] IRL 전문가 데모 생성...
  ✅ IRL 데모 100개 (전술 5종)  학습 손실: 0.128261

[4/4] 통계 계산 → data/data_stats.json
[+]  시각화 → data/ontology_stats.html  ← 브라우저로 열기
```

**생성 파일:**

| 파일 | 내용 | 용도 |
|------|------|------|
| `data/scenarios.json` | 100개 시나리오 KG | GNN 학습 입력 |
| `data/episodes.json` | 전투 롤아웃 | RL 경험 버퍼 |
| `data/irl_demos_summary.json` | IRL 데모 요약 | 보상 함수 추출 |
| `data/data_stats.json` | 전체 통계 | 리포트용 |
| **`data/ontology_stats.html`** | **인터랙티브 시각화** | **브라우저 열기** |

**시각화 대시보드 포함 차트 (9종):**
- 유닛 타입 분포 (Blue vs Red 그룹 바)
- 전력비(Blue/Red) 분포 히스토그램
- 유닛 능력치 레이더 차트 (화력·기동력·방호력·탄약·연료)
- 시나리오 0번 유닛 초기 배치도 (전장 지도)
- 전술 유형별 Blue 승률 (8개 전술)
- 헤드카운트 추이 (샘플 5개 에피소드 시계열)
- 에피소드 결과 도넛 차트
- 병력 감소율 분포 + 목표 구간 표시
- IRL 학습된 보상 가중치 (12개 피처)

---

### Step 3. 단계별 훈련

```bash
# Phase 1: Uncertainty-Aware GNN + Blue Agent
python train.py --phase 1 --episodes 1000

# Phase 2: Self-Play (Blue + Red)
python train.py --phase 2 --episodes 5000 --self-play

# Phase 3: HITL 통합
python train.py --phase 3 --episodes 2000 --hitl
```

### Step 4. 평가

```bash
# Monte Carlo 강건성
python evaluate.py --monte-carlo 5000

# 역사 벤치마크
python evaluate.py --benchmark historical
```

### Step 5. 전체 테스트 (251개)

```bash
PYTHONPATH=. python tests/test_tier1.py        # 온톨로지·시뮬레이터 21/21
PYTHONPATH=. python tests/test_tier2.py        # GNN·RL 30/30
PYTHONPATH=. python tests/test_tier3.py        # HITL·평가 39/39
PYTHONPATH=. python tests/test_phase1_immediate.py   # 대시보드·NL 32/32
PYTHONPATH=. python tests/test_phase2_short.py       # 기동·교리 53/53
PYTHONPATH=. python tests/test_phase3_medium.py      # IRL·AAR·벤치마크 76/76
```

### Step 6. 실시간 전투 대시보드

```bash
python -c "
from visualization.realtime_dashboard import run_and_render
out = run_and_render(n_blue=8, n_red=6, max_steps=20, output_path='dashboard.html')
print(f'브라우저에서 열기: {out}')
"
```

---

## 6. 📈 Expected Impact & Roadmap

### 예상 성과

| 목표 지표 | 수치 | 달성 근거 |
|-----------|------|-----------|
| **병력 감축률** | **15~25%** | Pareto 최적화 + ForceSize 패널티 |
| **승률 유지** | **±3%** | Robust Minimax Self-Play |
| **불확실성 보정** | **ECE < 0.1** | Temperature Scaling |
| **HITL 채택률** | **> 70%** | 선호도 학습 + Pareto 품질 |

### 정성적 가치
- **즉각 활용**: 훈련 시나리오 계획에 AI 보조 의사결정 즉시 적용
- **교리 준수 자동화**: 9대 원칙 위반 실시간 경보로 훈련 교관 부담 감소
- **투명한 AI**: "왜 이 전략인가"를 교리 언어로 자동 설명
- **인명 보호**: 최소 병력 원칙으로 불필요한 인명 손실 사전 방지

### 로드맵

```
현재 (v2.0) ────────────────── 완성
  ✅ Bayesian GNN + Uncertainty-Aware RL
  ✅ Self-Play (Blue ↔ Red)
  ✅ HITL + Natural Language Interface
  ✅ Auto AAR + Historical Benchmark

Phase 4 (계획) ─────────────── 단기
  🔲 실제 훈련 데이터 연동 (비식별화)
  🔲 3D 전장 시각화
  🔲 다국적군 협동 작전 (Coalition RL)
  🔲 LLM 기반 자연어 지휘 고도화

Phase 5 (장기) ─────────────── 중장기
  🔲 디지털 트윈 전장 환경
  🔲 드론 스웜 조율 알고리즘
  🔲 위성 ISR 공개 데이터 연동
```

---

## 7. 📁 프로젝트 구조

```
ai-combat-optimization/
├── 🗂️ ontology/            전장 지식 온톨로지
├── ⚔️ simulator/           전투 환경 시뮬레이터
├── 🕸️ gnn_model/           Bayesian HGT-GNN
├── 🤖 rl_agent/            PPO·MAPPO·IRL·Self-Play
├── 🎯 hitl/                Human-in-the-Loop
├── 💡 explainability/      AAR·어텐션·반사실 분석
├── 📊 evaluation/          Monte Carlo·벤치마크
├── 📺 visualization/       Plotly.js 실시간 대시보드
├── 🧪 tests/               테스트 스위트 (251개)
├── generate_data.py         온톨로지 데이터 생성 + 통계 시각화
├── demo.py                 전체 파이프라인 데모
├── train.py                훈련 스크립트
└── evaluate.py             평가 스크립트
```

---

## 8. 🏅 핵심 연구 기여 (4대 Novelty)

**기여 1. Uncertainty-Aware Combat GNN**
전장 불확실성을 Bayesian GNN으로 정량화하고, PPO 에이전트 상태에 통합하여 리스크 적응 전략을 학습하는 최초의 통합 프레임워크

**기여 2. Adversarial Combat Self-Play**
"적이 최적으로 반응할 때도 유효한" Robust Minimax 최소 병력 전략 — Red의 Deception 행동 → Blue의 GNN 불확실성 증가 → 강건한 전략 자연 학습

**기여 3. Doctrine-Aware HITL**
ADP 9대 원칙 자동 평가 + Pareto 생성 + Meaningful Human Control 충족을 단일 인터페이스로 통합

**기여 4. End-to-End Explainable Pipeline**
온톨로지 → GNN → RL → HITL 전 구간에서 교리 언어 기반 자동 설명 생성

---

## 9. ⚠️ 면책 조항 & 윤리

> 본 시스템은 **완전히 합성된 가상 전투 데이터**를 사용하며, **학술 연구 및 AI 방법론 연구 목적으로만** 활용됩니다. 실제 작전 정보나 기밀 정보를 포함하지 않습니다.

- **자율 무기 반대**: AI가 최종 교전 결정을 내리지 않습니다. 인간 지휘관이 항상 최종 결정권을 보유합니다.
- **IHL 준수**: HITL 구조는 "의미 있는 인간 통제(Meaningful Human Control)" 요건을 충족합니다.
- **연구 목적 한정**: 훈련 시뮬레이션 보조 도구로만 사용되어야 합니다.

---

## 10. 📄 라이선스

MIT License — 학술 연구, 교육, 비상업적 목적으로 자유롭게 사용 가능합니다.

---

<div align="center">

**🧠 AI Combat Optimization System v2.0**

*"병력을 아끼는 것이 곧 사람을 아끼는 것이다"*

[![Tests](https://img.shields.io/badge/Tests-251%2F251_✓-22C55E?style=flat-square)](tests/)
[![Modules](https://img.shields.io/badge/Modules-8개-3B82F6?style=flat-square)](.)
[![Code](https://img.shields.io/badge/Code-13%2C800+_lines-8B5CF6?style=flat-square)](.)

</div>
