# FALCON 프로젝트 종합 심층 분석 보고서 v4

**작성일**: 2026-03-02
**분석 기준**: 현재 코드베이스 전체 (104 소스 파일, ~28,000 LoC)
**참고 문서**: v1 (2026-02-21), v2 (2026-02-22), v3 (2026-02-25)
**분석 관점**: AI 전문가 × 국방 전문가 × 팔란티어 핵심 기술자 관점의 삼중 교차 분석
**분석 목표**: v3 이후 진전 사항을 반영하고, 현재 구조를 유지한 상태에서 **운영 전개 가능한 수준의 신뢰도**를 확보하기 위한 심층 진단 및 실행 가능한 발전 방안 제시

---

## 0. Executive Summary

FALCON(Force-multiplying Adaptive Learning & Cognitive Operation Network)은 **온톨로지→시뮬레이션→GNN→RL→HITL→평가**로 이어지는 6계층 end-to-end 파이프라인을 갖춘 군사 AI 의사결정 지원 플랫폼이다. v3 이후 다음 세 가지 영역에서 실질적 진전이 확인되었다.

1. **v3 핵심 지적사항 대부분 해소**: `setup_logger` 핸들러 중복 방지, `run_manifest.json` 자동 생성, MAPPO 실험 엔트리(`--algorithm mappo`) 완성, Phase 2 PPO vs MAPPO 비교 리포트 자동화
2. **평가 인프라 성숙**: MC report JSON schema 고정(`falcon-mc-report-v1`), 병렬 워커 안정화, CVaR 꼬리위험 지표 내장, Historical Benchmark 모듈 추가
3. **HITL 파이프라인 심화**: MC Pareto Validator(실측 기반 후보 검증), Bandit Preference Learner, Real-time Replanner, Natural Language Interface 등 5개 하위 모듈이 Phase 3 루프에 구조적으로 통합

그러나 **아키텍처가 성숙할수록 새로운 종류의 리스크가 등장**한다. v4 분석은 "모듈 부재"가 아니라 **"모듈 간 의미론적 정합성"**, **"운영 전개 시 표면화될 비기능적 취약점"**, **"연구→운용 전환 격차"**에 집중한다.

### v4 핵심 결론

| # | 결론 | 근거 |
|---|------|------|
| 1 | **6계층 아키텍처는 변경 없이 유지해야 한다** | 계층 분리가 실험 독립성과 모듈 테스트 가능성의 기반 |
| 2 | **성과를 제한하는 병목은 "기능 부재"가 아니라 "계약(Contract) 부재"이다** | Phase 간 보상 함수 7개 구성 요소의 의미/스케일이 암묵적이며 문서화되지 않음 |
| 3 | **운영 전개를 가로막는 최대 장벽은 "관측 가능성(Observability)"이다** | 실험 추적, 메트릭 대시보드, 이상 탐지가 수동적 |
| 4 | **향후 12주 전략은 "신뢰성 심화→관측 가능성 확보→전투 실험 확장" 순서가 최적** | 기반 미확보 시 고도화된 알고리즘(PSRO, MAT 등)의 효과 검증 자체가 불가 |

---

## 1. 분석 관점 및 방법

### 1-1. 삼중 교차 분석 프레임워크

본 v4 분석은 단일 관점이 아닌, 세 전문 영역의 렌즈를 교차 적용한다.

| 관점 | 핵심 질문 | 평가 기준 |
|------|----------|----------|
| **AI 전문가** | 학습 파이프라인이 이론적으로 건전하고 실험적으로 재현 가능한가? | 보상 설계, 수렴 보장, 불확실성 정량화, 실험 통제 |
| **국방 전문가** | 군사 교리·작전 개념과 정합하며, 지휘관이 신뢰할 수 있는 산출물을 제공하는가? | 온톨로지 충실도, ROE 준수, 교리 정렬, HITL 해석 가능성 |
| **팔란티어 기술자** | 분산 데이터 환경에서 운영 가능하고, 대규모 배치를 견딜 수 있는 엔지니어링 품질인가? | 모듈 결합도, 구성 관리, 관측 가능성, CI/CD, 보안 |

### 1-2. 분석 방법론

1. **정적 코드 분석**: 104개 Python 소스 파일(~28,000 LoC) 전수 조사 — 설계 패턴, 의존성 그래프, 복잡도, 보안 취약점
2. **파이프라인 추적 분석**: `train.py` → Phase 1/2/3/4 → `evaluate.py` → `demo.py`의 데이터 흐름을 엔드-투-엔드로 추적
3. **계약 검증**: Phase 간 입출력 인터페이스의 의미론적 일관성 검증
4. **v1~v3 이행 감사**: 이전 보고서에서 제기된 총 33개 항목의 이행 상태 확인
5. **테스트 품질 평가**: 17개 테스트 파일, 137+ 테스트 함수의 커버리지 및 계약 테스트 적합성 분석

### 1-3. 코드베이스 통계 (v4 기준)

| 지표 | 값 |
|------|-----|
| Python 소스 파일 | 104개 (테스트 17개 포함) |
| 총 코드 라인 | ~28,000 LoC |
| 핵심 모듈 | 10개 (ontology, simulator, gnn_model, rl_agent, hitl, evaluation, explainability, visualization, demo, utils) |
| 데이터클래스 | 70+ 클래스 |
| Enum 정의 | 15+ (42 UnitType, 7 Branch, 16 Echelon, 7 Domain 등) |
| RL 알고리즘 | 8+ (PPO, MAPPO, MAT, RARL, NFSP, PSRO, HierarchicalRL, InverseRL) |
| 시뮬레이터 엔진 | 11개 (Lanchester, Mixed, Combat Dynamics, Naval, Missile, Cyber, Weather, Maneuver, Fog, Resource, Adversarial) |
| 설정 파일 | 5 YAML + 6 시나리오 프리셋 |
| CI/CD | GitHub Actions (lint + test) |

---

## 2. 현재 아키텍처 성숙도 진단

### 2-1. 아키텍처 총평

FALCON은 v3 시점의 "핵심 경로가 연결된 통합형 연구 플랫폼" 단계를 넘어, **"실험 비교가 가능한 반복 연구 인프라"** 수준에 진입했다. 특히 다음 진전이 의미 있다:

- **PPO vs MAPPO 비교 리포트 자동 생성** (`_write_phase2_comparison_report`): 동일 seed 기반 알고리즘 비교가 단일 명령어로 가능
- **Run Manifest 표준화**: `run_manifest.json`에 git SHA, config hash, seed, phase, checkpoint 경로가 자동 기록
- **MC Report Schema 고정**: `falcon-mc-report-v1` 스키마로 평가 결과의 구조적 비교 가능
- **Logger 핸들러 중복 방지**: `setup_logger()`에서 기존 핸들러 정리 후 재등록 (`train.py:33-35`)

### 2-2. 계층별 성숙도 매트릭스 (v4 업데이트)

| 계층 | v3 등급 | v4 등급 | 변화 | 핵심 진전 | 잔존 과제 |
|------|---------|---------|------|----------|----------|
| **Ontology** | ★★★★★ | ★★★★★ | → | 42 UnitType + 7 Branch + 7 Domain 체계 안정 | 도메인 간 상호작용 규칙의 시뮬레이터 반영률 |
| **Simulator** | ★★★★☆ | ★★★★½ | ↑ | Naval/Cyber/Weather/Missile 엔진 추가, Mixed Lanchester 포위 효과 | 신규 엔진들의 학습 루프 통합률 낮음 |
| **GNN** | ★★★★☆ | ★★★★½ | ↑ | Temporal GNN + LSTM 하이브리드, numpy fallback 지원 | Calibration 자동 스케줄링 부재 |
| **RL** | ★★★½☆ | ★★★★☆ | ↑ | MAPPO 엔트리 개통, Phase 2 비교 리포트, Phase 4 선호도 재학습 | 8개 알고리즘 중 학습 루프 연결은 PPO/MAPPO/Self-Play만 완성 |
| **HITL** | ★★★½☆ | ★★★★☆ | ↑ | MC Pareto Validator, Bandit Preference, NL Interface, Replanner 통합 | 실시간 지휘관 인터랙션 UX 부재 |
| **Evaluation** | ★★★½☆ | ★★★★½ | ↑ | JSON schema 고정, CVaR 내장, Historical Benchmark, 병렬 워커 안정화 | 실험 간 통합 대시보드 부재 |
| **Explainability** | ★★★☆☆ | ★★★½☆ | ↑ | Auto AAR 구조화, Counterfactual 분석, Attention Viz | 학습 루프와의 피드백 연결 부재 |
| **CI/CD** | ★★★☆☆ | ★★★½☆ | ↑ | GitHub Actions 안정화, pre-commit 설정 | 커버리지 리포트, 성능 회귀 게이트 부재 |

### 2-3. v3 대비 주요 진전 사항 검증

#### v3 체크리스트 이행 상태

| v3 항목 | 상태 | 근거 |
|---------|------|------|
| Self-Play `prev_blue_hc` 정렬 | ✅ 해소 | `self_play_trainer.py:58-62`에서 스텝 차분 추적 |
| `setup_logger` 핸들러 중복 방지 | ✅ 해소 | `train.py:33-35`에서 기존 핸들러 clear 후 재등록 |
| `run_manifest.json` 표준 저장 | ✅ 해소 | `train.py:775-795` `_write_run_manifest()` 구현 |
| Pareto MC-lite 보정 연결 | ✅ 해소 | `mc_pareto_validator.py` 모듈 — 후보별 N회 시뮬레이션 검증 |
| Phase contract 문서 + CI 계약 테스트 | ⚠️ 부분 | `test_phase_contract.py` 존재하나 4개 테스트로 제한적 |
| MAPPO 실험 엔트리 1차 개통 | ✅ 해소 | `train.py:973-974` `--algorithm mappo` + 비교 리포트 자동화 |

**이행률: 5/6 완료, 1/6 부분 이행** — v3 대비 실질적 개선이 확인됨.

---

## 3. 심층 리스크 분석 (현재 구조 유지 전제)

> v4의 리스크 분석은 "기능 부재"가 아닌 **"성숙한 시스템 특유의 구조적 취약점"**에 집중한다. 아키텍처 변경 없이 효과적으로 해소 가능한 항목만 포함한다.

### 3-1. P0 (즉시, 0-2주) — 신뢰성 위협

#### P0-1. 보상 함수 7개 구성 요소의 **명세 부재**

**현상**: Phase 1/4의 보상 함수는 7개 구성 요소(win_bonus, force_reduction, casualty_penalty, doctrine_bonus, uncertainty_penalty, irl_bonus, supply_penalty)를 결합한다. 그러나:

- 각 구성 요소의 **스케일 범위**가 문서화되지 않음
- `survival_bonus`의 스케일링이 Phase에 따라 다름 (Phase 1: 단순 비율, Phase 4: preference adapter 적용)
- `irl_bonus`의 기여도가 IRL 학습 상태에 따라 0~무한으로 변동 가능

**영향**: Phase 간 정책 전이 시 보상 의미 드리프트 → 학습 불안정 또는 성능 저하
**정량 지표**: Phase 1 → Phase 2 → Phase 4 전이 시 win_rate 분산이 ±15% 이상 증가할 수 있음

**개선안**:
```
docs/contracts/REWARD_CONTRACT.md:
  - 구성 요소별 출력 범위 명세 (e.g., win_bonus ∈ [-1, +10])
  - Phase별 가중치 기본값 테이블
  - 스케일 정규화 규칙 (e.g., 모든 구성 요소를 [-1, +1]로 정규화 후 가중 합산)
```

#### P0-2. Monte Carlo 평가와 학습 루프의 **행동 해석 불일치**

**현상**: `evaluation/monte_carlo.py:31`의 `_build_mc_action_pairs`는 `BlueActionSpace` 상수를 **하드코딩**(`ADVANCE=1, FLANK=5` 등)하여 `blue_agent.py`의 enum과 동기화한다. 현재는 값이 일치하지만:

- `BlueActionSpace`에 새 행동을 추가하면 MC 평가가 **자동으로 깨짐**
- 두 위치 간 **컴파일 타임 연결이 없음** (import 대신 상수 복사)

**영향**: 행동 공간 변경 시 평가 루프가 잘못된 행동 매핑을 수행하되, 에러 없이 실행됨 (silent failure)

**개선안**: MC 모듈에서 `from rl_agent.blue_agent import BlueActionSpace`를 직접 import하거나, 공유 상수 모듈(`utils/action_space.py`) 도입

#### P0-3. 테스트 격리 불완전 — `sys.path.insert(0, ...)` 전역 오염

**현상**: 15개 이상의 모듈에서 `sys.path.insert(0, os.path.dirname(__file__))` 패턴 사용. 이는:
- 테스트 실행 시 **모듈 해석 순서를 비결정적**으로 만듦
- 패키지 설치 환경(`pip install -e .`)과 개발 환경에서 **다른 모듈이 로드**될 수 있음

**영향**: CI에서 통과하지만 로컬에서 실패하거나, 그 반대 상황 발생 가능

**개선안**: `conftest.py`에서 한 번만 `sys.path` 조정하고, 소스 모듈에서는 `sys.path` 조작 제거. `pyproject.toml`에 `[tool.pytest.ini_options] pythonpath = ["."]` 설정

### 3-2. P1 (단기, 2-6주) — 실험 품질 위협

#### P1-1. **학습 안 된 알고리즘의 데드코드 위험**

**현상**: 8개 RL 알고리즘 중 학습 루프에 완전 연결된 것은 3개(PPO, MAPPO, Self-Play)뿐. 나머지 5개(MAT, RARL, NFSP, PSRO, HierarchicalRL)는:

| 알고리즘 | 소스 코드 | train.py 연결 | 테스트 커버리지 |
|---------|----------|--------------|---------------|
| MAT (Multi-Agent Transformer) | ✅ 587 lines | ❌ 미연결 | ❌ 없음 |
| RARL (Robust Adversarial RL) | ✅ 496 lines | ❌ 미연결 | ❌ 없음 |
| NFSP (Neural Fictitious Self-Play) | ✅ 442 lines | ❌ 미연결 | ❌ 없음 |
| PSRO (Policy Space Response Oracle) | ✅ 405 lines | ❌ 미연결 | ❌ 없음 |
| HierarchicalRL | ✅ 474 lines | ❌ 미연결 | ❌ 없음 |

**영향**:
- 총 2,404 LoC(전체 RL 모듈의 ~42%)가 실험적으로 검증 불가 상태
- 코드 부패(code rot) 위험 — 다른 모듈 리팩토링 시 호환성 보장 불가
- 외부 리뷰어/사용자에게 **기능 과대 표현** 인상

**개선안**: 각 알고리즘에 대해 최소한의 smoke test(`test_algorithm_smoke.py`)를 추가하여 forward/backward pass가 깨지지 않았음을 보장

#### P1-2. **시뮬레이터 엔진 확장의 학습 루프 미반영**

**현상**: v3 이후 추가된 시뮬레이터 엔진들(Naval, Missile, Cyber, Weather)이 실제 학습 루프에서 사용되지 않음.

| 엔진 | 소스 | Phase 1 사용 | Phase 2 사용 | 평가 사용 |
|------|------|-------------|-------------|----------|
| LanchesterEngine | 522 lines | ✅ | ✅ | ✅ |
| MixedLanchesterEngine | 608 lines | ⚠️ 부분 | ❌ | ❌ |
| ManeuverEngine | 420 lines | ✅ | ✅ | ❌ |
| NavalEngine | 556 lines | ❌ | ❌ | ❌ |
| MissileModel | 520 lines | ❌ | ❌ | ❌ |
| CyberEffectsEngine | 484 lines | ❌ | ❌ | ❌ |
| WeatherModel | 461 lines | ❌ | ❌ | ❌ |
| ResourceManager | 430 lines | ⚠️ Phase 4만 | ❌ | ❌ |

**영향**: 2,451 LoC의 도메인별 시뮬레이터가 온톨로지와 함께 존재하지만 RL이 활용하지 못함

**개선안**: `SimulatorComposer` 패턴 도입 — 설정 파일에서 활성 엔진을 선택하면 학습 루프가 자동으로 해당 엔진을 조합

#### P1-3. **Fog of War Curriculum과 학습 성능의 상관관계 미측정**

**현상**: `CurriculumScheduler`가 Fog 레벨을 점진적으로 높이지만, Fog 레벨 변경이 정책 성능에 미치는 영향을 **정량적으로 추적하지 않음**

**영향**: Curriculum의 효과를 입증하지 못하면 불필요한 복잡성일 수 있음

**개선안**: Fog 레벨별 win_rate/casualty를 별도 메트릭으로 기록하고, curriculum 적용 vs 고정 Fog 간 A/B 비교 프로토콜 추가

### 3-3. P2 (중기, 6-12주) — 운영 전개성 위협

#### P2-1. **관측 가능성(Observability) 인프라 부재**

**현상**: 현재 로깅은 파일 기반(`run_*.log`)이며, 메트릭 추적은 JSON 파일 기반이다. 다음이 부재:

- **실험 간 비교 대시보드**: 동일 시드에서 하이퍼파라미터 변경 시 메트릭 차이를 시각적으로 비교하는 도구 없음
- **이상 탐지**: 학습 중 NaN, 발산, 갑작스러운 성능 저하를 자동 감지하는 메커니즘 없음
- **실험 레지스트리**: 과거 실행 결과를 구조적으로 검색/비교하는 시스템 없음

**영향**: 연구자가 실험 결과를 수동으로 비교해야 하며, 회귀를 놓칠 수 있음. Palantir Foundry와 같은 운영 플랫폼 연동 시 메트릭 파이프라인 재설계 필요.

**개선안 (단계적)**:
1. (2주) TensorBoard 로깅 활성화 — 이미 의존성에 포함(`tensorboard>=2.13.0`)되어 있으나 실제 로깅 코드 미구현
2. (4주) MLflow/W&B 연동 어댑터 — `utils/experiment_tracker.py` 추가
3. (8주) Plotly 기반 실험 비교 대시보드 — 기존 `visualization/realtime_dashboard.py` 확장

#### P2-2. **대규모 시나리오 배치의 메모리/성능 제약**

**현상**:
- Monte Carlo 5,000회 평가 시 `ProcessPoolExecutor`가 프로세스별로 전체 모델을 로드
- `generate_data.py`에서 100+ 시나리오 생성 시 메모리 사용량 선형 증가
- 대규모 지식 그래프(40+ 유닛)에서 `BayesianHGT`의 MC Dropout 15회 forward pass가 병목

**영향**: 학술 실험(소규모)에서는 무방하나, 운영 배치(대규모 배치 평가)에서 병목 발생

**정량 추정**:
- 유닛 20개 시나리오: MC 1,000회 ≈ 5분 (1 worker)
- 유닛 40개 시나리오: MC 1,000회 ≈ 20분+ (추정, quadratic scaling)

**개선안**:
- MC Dropout 샘플 수를 적응적으로 조절 (불확실성 수렴 시 조기 중단)
- 프로세스 풀 대신 `SharedMemory` 기반 에이전트 공유

#### P2-3. **보안 경계면(Security Boundary) 미정의**

**현상**: 코드베이스에 하드코딩된 시크릿이나 직접적인 취약점은 없으나:

- `--checkpoint` 경로에 대한 **입력 검증 부재** — 임의 파일 경로 주입 가능
- YAML 설정 로더의 fallback 파서가 PyYAML 규격보다 **느슨한 파싱** 수행
- `ProcessPoolExecutor` 워커가 임의 Python 코드를 실행하는 구조 — 악의적 체크포인트 역직렬화 시 코드 실행 가능 (`torch.load`의 pickle 위험)

**영향**: 연구 환경에서는 낮은 위험이나, 운영 환경 배치 시 보안 감사 필요

**개선안**:
- `torch.load(weights_only=True)` 강제 적용 (PyTorch 2.0+ 지원)
- 체크포인트 경로 allowlist 또는 서명 검증 추가
- YAML 로더에서 `yaml.safe_load()` 사용 확인 (현재 적용 중이나 fallback 파서에는 미적용)

### 3-4. P3 (장기, 12주+) — 연구 확장성 위협

#### P3-1. **다도메인 시나리오의 학습 루프 완전 통합 미완**

온톨로지(42 UnitType, 7 Domain)와 시나리오 프리셋(6종: korea_defense, urban_warfare, amphibious_assault, air_superiority, cyber_ew, multidomain_contest)은 풍부하나, 학습 루프는 **지상 도메인 중심**으로 구성되어 있다.

**영향**: 합동작전(Joint Operations) 시나리오의 RL 학습이 구조적으로 제한됨

#### P3-2. **설명 가능성(Explainability) → 학습 피드백 루프 부재**

`explainability/` 모듈(Auto AAR, Counterfactual, Attention Viz)이 사후 분석 도구로만 기능하며, 학습 과정에 피드백을 제공하지 않는다.

**영향**: 설명 가능성이 "보고용"에 머물고, 정책 개선에 기여하지 못함

**잠재적 개선**: Counterfactual 분석 결과를 보상 보정(reward shaping) 또는 경험 재생(experience replay) 우선순위에 반영

#### P3-3. **실시간 지휘관 인터랙션 인터페이스 부재**

`hitl/` 모듈이 프로그래밍 인터페이스(Python API)로만 존재하며, 지휘관이 직접 상호작용할 수 있는 **웹 기반 UI 또는 대화형 인터페이스**가 없다.

**영향**: HITL 파이프라인의 실전 유용성 검증 불가

---

## 4. 개선/발전 제안 (구조 유지형)

### 4-1. 설계 원칙 (v4 강화)

| 원칙 | 설명 | v3 대비 변화 |
|------|------|-------------|
| **계약 우선(Contract-First)** | 모듈 추가보다 인터페이스 계약 명세 우선 | 신규 |
| **관측 우선(Observe-First)** | 기능 개발 전 측정 지표와 대시보드 먼저 구축 | 신규 |
| **점진적 통합(Incremental Integration)** | 8개 RL 알고리즘을 한꺼번에 연결하지 않고, smoke test → 학습 루프 → 평가 순서로 점진 통합 | v3 계승 |
| **구조 보존(Architecture Preservation)** | 6계층 분리 유지, 모듈 간 결합도 최소화 | v3 계승 |
| **실측 기반(Evidence-Based)** | 휴리스틱 추정을 시뮬레이션 실측으로 대체 | v3 계승 |

### 4-2. 핵심 개선안 상세

#### 개선안 1: Phase 보상 계약(Reward Contract) 명세

**목표**: 7개 보상 구성 요소의 의미, 스케일, Phase별 가중치를 명세하여 정책 전이 안정성 확보

**산출물**: `docs/contracts/REWARD_CONTRACT.md` + `tests/test_reward_contract.py`

**보상 구성 요소 명세 (안)**:

| 구성 요소 | 출력 범위 | Phase 1 가중치 | Phase 2 가중치 | Phase 4 가중치 | 설명 |
|-----------|----------|---------------|---------------|---------------|------|
| `win_bonus` | {-1, +10} | 1.0 | 1.0 | 1.0 | 임무 성공/실패 |
| `force_reduction` | [-1, +1] | 0.5 | 0.5 | 0.5 | 병력 효율 |
| `casualty_penalty` | [-5, 0] | 0.3 | 0.3 | 0.3×pref | 아군 피해 |
| `doctrine_bonus` | [0, +2] | 0.2 | 0.0 | 0.2 | 교리 준수 |
| `uncertainty_penalty` | [-1, 0] | 0.1 | 0.0 | 0.1 | GNN 불확실성 |
| `irl_bonus` | [-1, +1] | 0.0 | 0.0 | 0.3 | IRL 보상 |
| `supply_penalty` | [-1, 0] | 0.1 | 0.0 | 0.1 | 보급 부족 |

**계약 테스트 예시**:
```python
def test_reward_components_bounded():
    """모든 보상 구성 요소가 명세된 범위 내에 있는지 검증"""
    reward = compute_reward(state, action, next_state, phase=1)
    assert -1 <= reward.win_bonus <= 10
    assert -1 <= reward.force_reduction <= 1
    assert -5 <= reward.casualty_penalty <= 0
```

#### 개선안 2: 알고리즘 연결 상태 계층화 (Maturity Tiers)

**목표**: 8개 RL 알고리즘의 연결 상태를 명시적으로 계층화하여 사용자 혼동 방지

| Tier | 정의 | 알고리즘 | 요구 사항 |
|------|------|---------|----------|
| **Tier A: Production** | 학습 + 평가 + 비교 리포트 완비 | PPO, Self-Play | 유지 |
| **Tier B: Experimental** | 학습 루프 연결, smoke test 통과 | MAPPO | smoke test 강화 |
| **Tier C: Prototype** | 소스 코드 존재, forward pass 검증 | MAT, RARL, NFSP, PSRO, HierarchicalRL | smoke test 추가 필요 |

**실행**: 각 Tier C 알고리즘에 `test_{algorithm}_smoke.py` 추가 (forward + backward pass, 3 에피소드 학습 안정성)

#### 개선안 3: 시뮬레이터 통합 엔진(Simulator Composer)

**목표**: 설정 파일에서 활성 엔진을 선택하면 학습 루프가 자동으로 해당 엔진을 조합

**설계 (안)**:
```yaml
# configs/scenario_engines.yaml
engines:
  combat: "lanchester"          # lanchester | mixed_lanchester | combat_dynamics
  movement: "maneuver"          # maneuver | null
  effects:                      # 복수 선택 가능
    - "fog_of_war"
    - "weather"
    - "cyber"
  resources: "resource_manager" # resource_manager | null
```

**구현**: `simulator/composer.py` — 설정 기반으로 엔진 인스턴스를 조합하고, `run_step(kg)` 인터페이스를 통일

#### 개선안 4: TensorBoard 실험 추적 활성화

**목표**: 이미 의존성에 포함된 TensorBoard를 활용하여 학습 메트릭 실시간 추적

**구현 범위**:
```python
# utils/tb_logger.py
from torch.utils.tensorboard import SummaryWriter

class FalconTBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_episode(self, ep: int, metrics: dict):
        for key, val in metrics.items():
            self.writer.add_scalar(f"train/{key}", val, ep)

    def log_evaluation(self, step: int, report: MCEvaluationReport):
        self.writer.add_scalar("eval/win_rate", report.blue_win_rate, step)
        self.writer.add_scalar("eval/cvar_10", report.cvar_10, step)
```

#### 개선안 5: 보안 강화 — 체크포인트 안전 로딩

**목표**: `torch.load` pickle 역직렬화 공격 방지

**변경 사항**: 모든 `torch.load()` 호출에 `weights_only=True` 추가

```python
# Before (위험)
state_dict = torch.load(checkpoint_path)

# After (안전)
state_dict = torch.load(checkpoint_path, weights_only=True)
```

**적용 대상**: `blue_agent.py`, `red_agent.py`, `evaluate.py` 내 모든 체크포인트 로드 코드

---

## 5. 우선순위별 실행 로드맵

### Phase I: 신뢰성 심화 (0-4주)

| ID | 작업 | 복잡도 | 영향도 | 소요 추정 |
|----|------|--------|--------|----------|
| **R1** | 보상 계약 명세(REWARD_CONTRACT.md) 작성 + 계약 테스트 | 중 | 높음 | 1주 |
| **R2** | MC 평가 행동 상수 → import 기반 동기화 | 낮 | 높음 | 2일 |
| **R3** | `sys.path.insert` 제거 → `pyproject.toml` pythonpath 설정 | 중 | 중 | 3일 |
| **R4** | `torch.load(weights_only=True)` 전역 적용 | 낮 | 중 | 1일 |
| **R5** | Tier C 알고리즘 smoke test 5개 추가 | 중 | 중 | 1주 |

**완료 기준 (DoD)**:
- 보상 구성 요소별 범위 테스트 전수 통과
- MC 평가와 학습 루프의 행동 매핑 100% 일치 (자동 검증)
- `pytest` 실행 시 모든 알고리즘의 기본 forward pass 성공
- 보안 감사 항목 0건 (체크포인트 로딩)

### Phase II: 관측 가능성 확보 (4-8주)

| ID | 작업 | 복잡도 | 영향도 | 소요 추정 |
|----|------|--------|--------|----------|
| **R6** | TensorBoard 실험 추적 통합 | 중 | 높음 | 1주 |
| **R7** | 시뮬레이터 통합 엔진(Composer) 프로토타입 | 높 | 높음 | 2주 |
| **R8** | Phase contract 문서 확장 + 계약 테스트 강화 (8개 → 20개) | 중 | 높음 | 1주 |
| **R9** | Fog Curriculum 효과 A/B 측정 프로토콜 | 중 | 중 | 1주 |
| **R10** | 실험 레지스트리(run manifest 인덱스 + 검색) | 중 | 중 | 1주 |

**완료 기준 (DoD)**:
- TensorBoard에서 학습 곡선(win_rate, loss, entropy) 실시간 확인 가능
- 설정 파일 하나로 엔진 조합 변경 가능
- Phase contract 위반 시 테스트 실패로 자동 차단
- Fog curriculum의 win_rate 개선 효과가 정량 리포트로 확인

### Phase III: 전투 실험 확장 (8-16주)

| ID | 작업 | 복잡도 | 영향도 | 소요 추정 |
|----|------|--------|--------|----------|
| **R11** | 다도메인 시나리오 배치 실험 (4×4 매트릭스) | 높 | 높음 | 3주 |
| **R12** | Tier C → Tier B 승격: RARL + NFSP 학습 루프 연결 | 높 | 높음 | 3주 |
| **R13** | Explainability → 보상 피드백 연결 (Counterfactual reward shaping) | 높 | 중 | 2주 |
| **R14** | HITL 웹 인터페이스 프로토타입 (Plotly Dash 기반) | 높 | 높음 | 3주 |
| **R15** | CVaR-aware 하이퍼파라미터 탐색 자동화 | 높 | 중 | 2주 |

**완료 기준 (DoD)**:
- 4×4 다도메인 일반화 성능 매트릭스 확보
- RARL/NFSP가 PPO와 동일 평가 프레임워크에서 비교 가능
- Counterfactual 분석 결과가 보상에 반영되어 win_rate ≥ 2% 개선 확인
- 지휘관이 웹 브라우저에서 Pareto 후보를 선택하고 제약을 입력할 수 있음

### 로드맵 시각화

```
Week:  0    2    4    6    8    10   12   14   16
       |----|----|----|----|----|----|----|----|
Phase I ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  신뢰성 심화
  R1   ████
  R2   ██
  R3   ███
  R4   █
  R5   ████
Phase II     ░░░░████████████░░░░░░░░░░░░░░░░░░  관측 가능성
  R6              ████
  R7              ████████
  R8              ████
  R9              ████
  R10                 ████
Phase III                    ░░░░████████████████  전투 실험
  R11                             ████████████
  R12                             ████████████
  R13                                 ████████
  R14                             ████████████
  R15                                 ████████
```

---

## 6. 최종 결론

### 6-1. 현재 위치 진단

FALCON은 군사 AI 의사결정 지원 시스템으로서 **아키텍처적으로 성숙하고, 핵심 파이프라인이 작동하며, 평가 인프라가 구조화된 상태**에 도달했다. 이는 다음 수치로 입증된다:

- **6계층 완전 연결**: Ontology → Simulator → GNN → RL → HITL → Evaluation
- **4단계 학습 파이프라인**: Phase 1(GNN+PPO) → Phase 2(Self-Play) → Phase 3(HITL) → Phase 4(선호도 재학습)
- **통계적 평가 기반**: Monte Carlo 5,000+ run, Bootstrap 95% CI, CVaR 꼬리위험
- **v3 지적사항 이행률**: 83% (5/6 완료)

### 6-2. 핵심 격차

그러나 **"작동하는 프로토타입"에서 "신뢰할 수 있는 실험 플랫폼"으로의 전환**에는 다음 격차가 남아 있다:

| 격차 유형 | 구체적 내용 | 해소 난이도 |
|----------|-----------|-----------|
| **계약 격차** | 보상 함수 7개 구성 요소의 명세 부재, Phase 간 전이 시 의미 드리프트 | 낮음 (문서화 + 테스트) |
| **통합 격차** | 11개 시뮬레이터 중 3개만 학습 루프 연결, 8개 RL 알고리즘 중 3개만 완전 연결 | 중간 (점진적 통합) |
| **관측 격차** | 실험 추적 수동적, 메트릭 대시보드 부재, 이상 탐지 없음 | 중간 (도구 연동) |
| **인터페이스 격차** | HITL이 Python API로만 존재, 지휘관 직접 상호작용 불가 | 높음 (UI 개발) |

### 6-3. 전략적 권고

v4 단계의 전략은 **"더 많은 알고리즘을 추가하는 것"이 아니라, "이미 존재하는 강력한 구조의 신뢰도와 관측 가능성을 확보하는 것"**이다.

구체적으로:

1. **즉시(0-4주)**: 보상 계약 명세, 행동 상수 동기화, 보안 강화 — **기존 코드의 내적 일관성 확보**
2. **단기(4-8주)**: TensorBoard 연동, 시뮬레이터 통합, 테스트 강화 — **실험의 관측 가능성 확보**
3. **중기(8-16주)**: 다도메인 실험, 알고리즘 승격, HITL UI — **연구 범위 확장과 실전 유용성 검증**

이 순서를 따르면, FALCON은 16주 내에 **연구 논문의 실험 기반**이자 **운영 환경 배치의 프로토타입**으로 동시에 기능할 수 있는 수준에 도달할 수 있다.

### 6-4. Palantir Gotham/Foundry 통합 관점의 추가 권고

운영 환경에서 Palantir 플랫폼과의 통합을 고려할 때, 다음 사전 조건이 필요하다:

| 통합 영역 | 현재 상태 | 필요 조건 |
|----------|----------|----------|
| **데이터 파이프라인** | JSON/CSV 파일 기반 | Object 기반 데이터 모델 + REST API |
| **온톨로지 매핑** | Python enum/dataclass | OWL/RDF 또는 Foundry Ontology 매핑 |
| **실험 추적** | run_manifest.json | Foundry Pipeline 연동 가능한 메타데이터 스키마 |
| **의사결정 지원** | Python API | Gotham 위젯 또는 웹 대시보드 |
| **보안** | 연구 환경 수준 | 분류 등급별 데이터 격리, RBAC |

---

## Appendix A. v4 우선 작업 체크리스트

### 즉시 (0-4주)
- [ ] `docs/contracts/REWARD_CONTRACT.md` 작성 + 7개 구성 요소 범위 명세
- [ ] `tests/test_reward_contract.py` — 보상 구성 요소 범위 검증 테스트
- [ ] `evaluation/monte_carlo.py` — BlueActionSpace import 기반 동기화
- [ ] 전역 `sys.path.insert` 제거 → `pyproject.toml` pythonpath 설정
- [ ] `torch.load(weights_only=True)` 전역 적용
- [ ] Tier C 알고리즘 smoke test 5개 추가 (MAT, RARL, NFSP, PSRO, HierarchicalRL)

### 단기 (4-8주)
- [ ] `utils/tb_logger.py` — TensorBoard 실험 추적 통합
- [ ] `simulator/composer.py` — 설정 기반 엔진 조합기 프로토타입
- [ ] Phase contract 테스트 확장 (4개 → 20개)
- [ ] Fog Curriculum A/B 측정 프로토콜
- [ ] 실험 레지스트리 인덱스 + 검색 기능

### 중기 (8-16주)
- [ ] 4×4 다도메인 일반화 실험
- [ ] RARL + NFSP Tier B 승격 (학습 루프 연결 + 평가)
- [ ] Counterfactual → reward shaping 피드백 연결
- [ ] HITL 웹 인터페이스 프로토타입 (Plotly Dash)
- [ ] CVaR-aware 하이퍼파라미터 탐색 자동화

---

## Appendix B. 계층별 코드 품질 상세 평가

### B-1. 설계 패턴 사용 현황

| 패턴 | 적용 위치 | 품질 |
|------|----------|------|
| **Factory** | `ScenarioFactory`, `DomainRandomizer` | ★★★★★ |
| **Strategy** | `BlueAgent`/`RedAgent` 행동 선택, `UnitActionMasks` | ★★★★☆ |
| **Observer** | `CommanderPreferenceLearner`, `BattleHistory` | ★★★★☆ |
| **Template Method** | `LanchesterEngine.calculate_engagement`, Training loops | ★★★★☆ |
| **Adapter** | `FogOfWarFilter.observe`, `build_state_vector` | ★★★★☆ |
| **Builder** | `CombatKnowledgeGraph` 단계적 구성 | ★★★★☆ |
| **State** | `UnitStatus`, `FogLevel` enum 기반 상태 전이 | ★★★★★ |

### B-2. 코드 품질 지표

| 지표 | 값 | 평가 |
|------|-----|------|
| `from __future__ import annotations` 사용률 | 69/104 (66%) | 양호 |
| `@dataclass` 활용 | 70+ 클래스 | 우수 |
| 타입 힌트 적용률 | ~80% | 양호 |
| `try/except` 방어적 코딩 | 41개 블록 | 충분 |
| `np.clip` 수치 안정성 | 전역 적용 | 우수 |
| 하드코딩된 시크릿 | 0건 | 안전 |

### B-3. 의존성 건전성

| 의존성 | 버전 | 위험도 | 비고 |
|--------|------|--------|------|
| `torch>=2.0.0` | 주요 | 낮음 | LTS 지원 |
| `numpy>=1.24.0` | 주요 | 낮음 | 안정 |
| `gymnasium>=0.29.0` | 주요 | 낮음 | Farama Foundation 관리 |
| `stable-baselines3>=2.0.0` | 보조 | 중간 | 커뮤니티 프로젝트 |
| `plotly>=5.15.0` | 시각화 | 낮음 | Dash 확장 가능 |
| `networkx>=3.1` | 주요 | 낮음 | 학술 표준 |

순환 의존성: **0건** — 모듈 간 단방향 의존 그래프 유지

---

## Appendix C. KPI 프레임워크 (v4 확장)

### 5축 KPI 체계

| 축 | 지표 | 목표값 | 현재 추정값 |
|----|------|--------|-----------|
| **임무 성과** | win_rate | ≥ 60% | ~55% (MC 500 run 기준) |
| | mission_time | ≤ 30 steps | ~35 steps |
| **비용 효율** | blue_casualties / initial_force | ≤ 20% | ~25% |
| | force_reduction | ≥ 20% | ~18% |
| **강건성** | CVaR@10 (worst 10% win_rate) | ≥ 40% | 측정 필요 |
| | strategy_robustness | ≥ 0.30 | ~0.25 |
| **지휘 적합성** | doctrine_compliance | ≥ 80% | ~75% |
| | HITL adoption_rate | ≥ 60% | 시뮬레이션 기반 측정 필요 |
| **시스템 품질** | test_pass_rate | 100% | ~95% |
| | CI pipeline 소요 시간 | ≤ 10분 | ~5분 |

---

## Appendix D. v1~v4 진화 요약

| 버전 | 날짜 | 핵심 초점 | 발견 항목 | 해결률 |
|------|------|----------|----------|--------|
| **v1** | 2026-02-21 | 버그 + 기능 결함 | 15개 (P0-P3) | → v2에서 80% 해소 |
| **v2** | 2026-02-22 | 잔존 결함 + 신규 연결 이슈 | 3 잔존 + 5 신규 | → v3에서 75% 해소 |
| **v3** | 2026-02-25 | 구조 유지 + 연결 일관성 | 6개 개선안 (P0-P2) | → v4에서 83% 해소 |
| **v4** | 2026-03-02 | 계약/관측/통합 격차 | 15개 개선안 (P0-P3) | 실행 대기 |

**궤적 해석**: v1(버그 헌팅) → v2(연결 수리) → v3(일관성 확보) → v4(계약·관측·확장)로 **분석 관점이 저수준에서 고수준으로 이동**하고 있으며, 이는 코드베이스의 성숙도가 실질적으로 향상되고 있음을 의미한다.

---

*본 보고서는 AI 전문가, 국방 전문가, 팔란티어 핵심 기술자 관점의 삼중 교차 분석을 기반으로 작성되었습니다.*
*FALCON v2.0.0 | 104 소스 파일 | ~28,000 LoC | 2026-03-02*
