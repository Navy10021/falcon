# FALCON 프로젝트 종합 분석 보고서

**작성일**: 2026-02-21
**분석 범위**: 전체 코드베이스 (ontology / simulator / gnn_model / rl_agent / hitl / explainability / evaluation / tests / docs)

---

## 1. 프로젝트 개요

FALCON(**F**orce-**A**daptive **L**earning for **C**ombat **O**ptimization **N**etwork)은 전장 의사결정 지원을 위한 AI 연구 프레임워크다. 핵심 질문은 다음과 같다:

> *"임무를 달성하면서 허용 가능한 위험을 유지하는 최소 병력은?"*

### 아키텍처 6계층 요약

| 계층 | 모듈 | 역할 |
|------|------|------|
| 1 | `ontology/` | 전장 지식 그래프 스키마, 교리 인코딩, 시나리오 생성 |
| 2 | `simulator/` | 란체스터 엔진, 안개 전쟁, 기동/자원 관리 |
| 3 | `gnn_model/` | Bayesian HGT-GNN, 불확실성 정량화 |
| 4 | `rl_agent/` | Blue/Red PPO, Self-Play, MAPPO, League, IRL |
| 5 | `hitl/` | 제약 파서, Pareto 생성기, 선호도 학습 |
| 5' | `explainability/` | 어텐션 시각화, 자동 AAR, 반사실 분석 |
| 6 | `evaluation/` | Monte Carlo, 역사 벤치마크, 지표 계산 |

---

## 2. 강점 분석

### 2-1. 아키텍처 설계
- **계층 분리가 명확**하다. 온톨로지 → 시뮬레이터 → GNN → RL → HITL 흐름이 단방향으로 잘 정의되어 있다.
- `CombatKnowledgeGraph`를 중심 데이터 구조로 사용해 모든 계층이 동일 인터페이스를 공유한다.
- `ScenarioFactory`로 재현 가능한 시나리오 생성이 가능하다.

### 2-2. 불확실성 모델링
- MC Dropout 기반 Bayesian HGT-GNN이 **Epistemic/Aleatoric 불확실성을 분리**해 정량화한다.
- `predict_with_uncertainty()`가 95% 신뢰구간까지 반환한다.
- PPO 상태 벡터에 불확실성 6차원을 포함해 **에이전트가 불확실성을 인식하며 행동**한다.

### 2-3. 강화학습 설계
- Phase A→D 커리큘럼(룰기반 Red → Blue 고정 → 교대 → Population)이 **점진적 난이도 상승**을 구현한다.
- GAE + Clipped Surrogate Loss + Entropy Regularization의 표준 PPO가 구현되어 있다.
- `MAPPO`로 유닛별 분산 행동 공간 확장, `LeagueSelfPlay`로 AlphaStar 방식 다양성 유지가 준비되어 있다.
- 역강화학습(`inverse_rl.py`)으로 교리 시연에서 보상 함수를 자동 추출하는 기능도 구현되어 있다.

### 2-4. HITL 설계
- `CommanderConstraints`의 Hard / Soft / Ethical 3-tier 제약 구조가 실사용에 가깝다.
- Pareto 다목적 최적화(병력↓, 승률↑, 사상자↓, 시간↓)로 지휘관에게 명확한 트레이드오프를 제시한다.
- `MCParetoValidator`로 Pareto 후보를 Monte Carlo로 사후 검증하는 구조가 있다.

### 2-5. 설명 가능성
- `AutoAARGenerator`가 에피소드 로그에서 **결정적 전환점, 최선 결정, 놓친 기회, 개선 권고**를 자동 생성한다.
- 어텐션 가중치를 저장(`last_attention_weights`)해 GNN 설명 가능성의 기반이 마련되어 있다.

---

## 3. 문제점 및 개선 사항 (단계별)

---

### PHASE 0 — 즉시 수정 (버그 / 논리 오류)

#### P0-1. `train.py` Phase 1 보상 계산 오류

**위치**: `train.py:102`

```python
# 현재 (문제)
reward = blue_agent.compute_reward(
    step_result,
    initial_blue_hc,           # ← 에피소드 시작값 고정
    step_result.blue_total_headcount,
    avg_unc
)
```

`compute_reward(step_result, force_size_before, force_size_after, ...)`의 `force_size_before`에 **에피소드 전체 시작 병력이 고정**으로 들어간다. 매 스텝마다 직전 스텝 대비 절감이 아닌 에피소드 시작 대비 절감을 계산하므로, `self_play_trainer.py`와 보상 의미가 다르다.

**개선**: 직전 스텝 병력을 `prev_blue_hc` 변수로 추적해 전달.

---

#### P0-2. `monte_carlo.py` Red 사상자 항상 0

**위치**: `monte_carlo.py:120`

```python
red_casualties=sum(...) * 0,  # placeholder
```

`* 0`이 남아 있어 `MCResult.red_casualties`가 항상 0이다. 취약점 분석에서 교환비(Exchange Ratio) 계산이 불가능하다.

**개선**: placeholder 제거, 누적 `red_casualties` 값으로 대체.

---

#### P0-3. `resource_manager.py` 유닛 타입 키 오타

**위치**: `resource_manager.py:47`

```python
FUEL_CONSUMPTION_RATE = {
    "electronics_warfare": 0.02,   # ← 오타
    # 정상: "electronic_warfare"
}
```

`UnitType.ELECTRONIC_WARFARE = "electronic_warfare"`와 키가 불일치해 전자전 유닛의 연료 소모가 적용되지 않는다.

**개선**: `"electronics_warfare"` → `"electronic_warfare"` 수정.

---

#### P0-4. `self_play_trainer.py` 변수 스코프 위험

**위치**: `self_play_trainer.py:250`

```python
winner = step_result.mission_status if 'step_result' in dir() else "draw"
```

`dir()`을 통한 변수 존재 확인은 파이썬 안티패턴이다. 루프가 0회 실행되면 `step_result`가 미정의 상태다.

**개선**: 루프 전에 `step_result = None` 초기화 후 `if step_result is not None` 조건으로 변경.

---

#### P0-5. `BayesianHGT` 훈련/추론 모드 미복원

**위치**: `bayesian_hgt.py:202`

```python
def predict_with_uncertainty(self, x, adj):
    self.train()  # Dropout 활성화 (모드 변경)
    with torch.no_grad():
        ...
    # 함수 종료 시 eval() 복원 없음
```

`self.train()` 호출 후 복원하지 않아 호출자의 eval 모드가 깨진다. 이후 GNN 배치 학습에서 드롭아웃이 이중 적용될 수 있다.

**개선**: 진입 시 `was_training = self.training` 저장, 종료 시 `self.train(was_training)` 복원.

---

#### P0-6. `blue_agent.py` 중복 import

**위치**: `blue_agent.py` 마지막 줄

```python
# 순환 import 방지
import torch.nn.functional as F   # ← 파일 상단에 이미 임포트됨
```

실제 순환 import를 해결하지 못하고 코드 혼란만 야기한다.

**개선**: 중복 import 제거.

---

### PHASE 1 — 단기 보완 (기능 완성도)

#### P1-1. Phase 1에서 에이전트 행동이 시뮬레이션에 미반영

**위치**: `train.py:98`

```python
action, log_prob, value = blue_agent.select_action(state)
step_result = engine.run_step(kg)   # action이 전달되지 않음
```

PPO가 행동을 선택하지만 `run_step()`은 자동 교전(`_auto_pair_units`)으로 실행된다. 에이전트 행동과 환경 결과가 **완전히 분리**되어 있어 PPO가 의미 없는 상태-행동 쌍에서 학습한다. `self_play_trainer.py`에는 `_build_action_pairs()`로 연결되어 있어 **Phase 간 구현 불일치**가 존재한다.

**개선**: `train.py` Phase 1에도 `_build_action_pairs()` 동일 로직 적용.

---

#### P1-2. GNN `risk_score` 타깃 상수 고정

**위치**: `train.py:125`

```python
targets = {"blue_casualties": bt, "risk_score": torch.tensor(0.5)}
```

위험도 헤드의 타깃이 항상 0.5로 고정되어 **위험도 헤드가 전혀 학습되지 않는다**. GNN 출력 6차원 중 `risk_mean`, `risk_std`가 무의미한 값이 된다.

**개선**: `risk_score` 타깃을 교전 결과 기반(예: `blue_casualties / (blue_casualties + red_casualties + 1)`)으로 동적 계산.

---

#### P1-3. `ScenarioFactory` 전역 시드 오염

**위치**: `ontology/combat_schema.py:354`

```python
if seed is not None:
    np.random.seed(seed)   # 전역 상태 변경
```

병렬 실행 또는 연속 시나리오 생성 시 다른 모듈의 난수 생성기에 영향을 미친다.

**개선**: `rng = np.random.RandomState(seed)` 인스턴스를 생성해 해당 rng만 사용.

---

#### P1-4. MAPPO와 League Self-Play 연결 미완

- `rl_agent/mappo.py`의 `MAPPOManager`: `NumpyUnitActor`가 역전파 없는 numpy 구현이라 실제 학습 불가. `train.py` 어느 Phase에서도 호출되지 않는다.
- `rl_agent/league_selfplay.py`의 `PFSPScheduler`: Phase D에서 여전히 `np.random.choice(self.blue_population)` 단순 랜덤 선택을 사용한다.

**개선**:
1. `NumpyUnitActor` → `torch.nn.Module` 기반으로 업그레이드
2. `train.py --phase 2 --mappo` 옵션으로 MAPPO 학습 루프 연결
3. Phase D에서 `PFSPScheduler.select_opponent()` 호출

---

#### P1-5. 테스트가 pytest 미사용, CI 미연결

모든 테스트(`tests/`)가 `results.append()` + `print()` 방식의 커스텀 검증 프레임으로 작성되어 있다. CI 자동화가 불가능하다.

**개선**:
1. pytest를 dev 의존성으로 추가 (`requirements.txt` 주석 해제)
2. `check()` 헬퍼를 `assert` 문으로 전환
3. GitHub Actions 워크플로 추가: `pytest tests/ -v`

---

#### P1-6. `data/` 디렉토리가 코드와 연결되지 않음

`data/episodes.json`, `data/scenarios.json`, `data/irl_demos_summary.json`이 존재하지만 어느 코드에서도 로드하지 않는다.

**개선**: `inverse_rl.py`의 `InverseRLTrainer`가 `data/irl_demos_summary.json`을 로드하는 데이터 로더 구현.

---

### PHASE 2 — 중기 발전 (시스템 품질)

#### P2-1. 설정 분산 문제 (YAML 도입)

`train.py`, `evaluate.py`, `demo.py`의 기본값이 각자 하드코딩되어 있다. 실험 재현 시 매번 CLI 인자를 수동으로 맞춰야 한다.

**개선**:
```
configs/
  default.yaml
  phase1.yaml / phase2.yaml / phase3.yaml
  evaluation.yaml
```
CLI 인자는 override만 담당. 실행 시 최종 config snapshot을 `checkpoints/{run_id}/config.yaml`에 자동 저장.

---

#### P2-2. 로깅 체계 미흡

`print()` 중심 로그는 장기 실험 추적, 원격 서버 실행, 자동화된 분석에 한계가 있다.

**개선**:
- `structlog` 또는 표준 `logging` 모듈로 전환
- `run_id = {phase}_{seed}_{timestamp}` 형식으로 실험 식별자 통일
- 에피소드 지표를 `jsonl` 포맷으로 파일 저장 (TensorBoard 병행)

---

#### P2-3. 재현성 설정 불완전

**위치**: `utils/reproducibility.py`

`torch.backends.cudnn.deterministic = True` 설정이 누락되어 GPU 환경에서 완전한 재현성이 보장되지 않는다.

**개선**:
```python
def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

---

#### P2-4. Monte Carlo 평가 병렬화 미지원

5,000회 Monte Carlo는 단일 스레드 실행 시 시간이 과다 소요된다.

**개선**: `concurrent.futures.ProcessPoolExecutor`를 사용해 시나리오 단위 병렬화. `--fast`(100회) / `--full`(5,000회) 프리셋 분리.

---

#### P2-5. Pareto 전략 후보가 실제 시뮬레이션 기반 아님

**위치**: `pareto_generator.py:133~218`

`force_ratio`, `win_base`, `cas_ratio` 등이 **하드코딩된 상수**다. 실제 KG 상태(`blue_combat_power` 대비 `red_combat_power`)가 결과에 반영되지 않는다.

**개선**:
1. 각 전략 후보에 대해 축약 Monte Carlo 시뮬레이션(N=50~200) 실행
2. `MCParetoValidator`를 생성 단계에 통합
3. 전투력 비율 기반으로 `win_base` 동적 계산

---

#### P2-6. 보상 스케일 불균형

**위치**: `blue_agent.py:279~304`

| 항목 | 크기 |
|------|------|
| 승리/패배 | ±10.0 |
| 사상자 패널티 | casualties × 0.05 ≈ 1~5 |
| 병력 절감 보상 | reduction × 0.01 ≈ 매우 작음 |
| 적 피해 보상 | red_casualties × 0.02 |

승리/패배 보상이 다른 항을 압도해 에이전트가 **병력 절감보다 승패에만 집중**하는 경향이 생긴다. 핵심 목표인 "최소 병력으로 임무 달성"이 보상에 충분히 반영되지 않는다.

**개선**: 보상 스케일 재조정 또는 정규화. 병력 절감 보상 가중치 대폭 상향 권장.

---

### PHASE 3 — 장기 발전 (연구 확장)

#### P3-1. 공간 이동 모델 부재

`LanchesterEngine`은 거리 기반으로 교전 여부를 결정하지만 `run_step()`에서 유닛의 `position`을 업데이트하지 않는다. `maneuver_engine.py`가 별도 존재하나 메인 엔진과 통합되지 않았다. ADVANCE, FLANK, WITHDRAW 행동이 공간적 효과를 전혀 갖지 않는다.

**개선**: `ManeuverEngine`을 `LanchesterEngine`과 결합해 매 스텝마다 유닛 위치 갱신.

---

#### P3-2. GNN 노드 특성 동적 업데이트 미구현

유닛 상태(headcount, status, morale)가 시뮬레이션 중 변경되지만, `CombatKnowledgeGraph.graph`의 노드 `features` 속성이 초기화 시점 값으로 고정된다. GNN이 매 스텝 동일한 초기 그래프를 입력받는다.

**개선**: `engine.run_step()` 이후 `kg.update_node_features()` 메서드를 호출해 변경된 유닛 상태를 그래프 노드에 동기화.

---

#### P3-3. HITL 선호도 학습 → RL 정책 반영 루프 미완성

Phase 3에서 지휘관 선호도를 학습하지만 RL 정책(Phase 1/2)에 피드백하는 루프가 없다. "HITL → 선호도 학습 → 정책 재학습" 사이클이 완성되지 않았다.

**개선**:
1. `CommanderPreferenceLearner`의 가중치를 PPO 보상 함수에 반영하는 어댑터 구현
2. `train.py --phase 4 --preference-model checkpoints/hitl_preferences.json` 옵션으로 선호도 반영 재학습 지원

---

#### P3-4. 역강화학습 파이프라인 미완성

`inverse_rl.py`가 구현되어 있으나 실제 교리 데모 데이터(`data/irl_demos_summary.json`)와의 연결이 없다. IRL로 학습된 보상 가중치를 Blue 에이전트에 반영하는 코드도 없다.

**개선**: 데이터 로더 → IRL 학습 → 보상 함수 반영의 전체 파이프라인 완성.

---

#### P3-5. 교리 준수도가 RL 학습에 미반영

`ontology/doctrine_encoder.py`가 구현되어 있고 AAR에서 `doctrine_score`가 표시되지만, Blue 에이전트의 보상 함수에는 전혀 포함되지 않는다.

**개선**: `BlueAgent.compute_reward()`에 `doctrine_compliance_bonus` 항을 추가하고 `DoctrineEncoder`에서 실시간 교리 점수를 계산해 보상 신호에 반영.

---

#### P3-6. 다도메인(Multi-Domain) 통합 미활용

`ontology/multidomain.py`가 구현되어 있으나 `CombatKnowledgeGraph`나 `ScenarioFactory`와 연결되지 않았다.

**개선**: `MultidomanScenarioFactory` 구현 후 `train.py --domain {ground,air,naval,multi}` 옵션으로 연동.

---

## 4. 코드 품질 세부 지적

| 항목 | 위치 | 내용 |
|------|------|------|
| 잔차 연결 조건부 생략 | `bayesian_hgt.py:77` | `x.size(-1) == out.size(-1)` 불일치 시 조용히 생략됨. 명시적 프로젝션 레이어 또는 assert 필요 |
| 강건성 점수 수식 | `monte_carlo.py:166` | `win_rate * (1 - std(0/1 배열))` 은 승률에 종속적. CVaR 기반 대체 권장 |
| FogOfWar O(N²) 비용 | `fog_of_war.py:121` | 매 스텝 `build_spatial_relations()` 호출. 캐시 또는 incremental 업데이트 필요 |
| `setup.py` 구식 패키징 | `setup.py` | `pyproject.toml` (PEP 517/518) 전환 권장 |
| 타입 힌트 불완전 | `pareto_generator.py`, `monte_carlo.py` 등 | `kg` 파라미터가 타입 없이 전달. mypy 정적 분석 목표로 점진적 보완 |

---

## 5. 우선순위별 실행 로드맵

### 즉시 (1~2일) — 버그 수정

| 파일 | 위치 | 수정 내용 |
|------|------|-----------|
| `evaluation/monte_carlo.py` | L120 | `* 0` 제거, 실제 red 사상자 집계 |
| `simulator/resource_manager.py` | L47 | `electronics_warfare` → `electronic_warfare` |
| `ontology/combat_schema.py` | L354 | `np.random.seed()` → `RandomState` 인스턴스 사용 |
| `rl_agent/self_play_trainer.py` | L250 | `dir()` 패턴 → `None` 초기화 방식 |
| `rl_agent/blue_agent.py` | 말미 | 중복 `import F` 제거 |
| `gnn_model/bayesian_hgt.py` | L202 | 훈련 모드 복원 로직 추가 |

### 단기 (1~2주) — 기능 완성

| 항목 | 기대 효과 |
|------|-----------|
| Phase 1 보상에서 `prev_blue_hc` 추적 | 훈련 신뢰성 향상 |
| GNN `risk_score` 타깃 동적 계산 | GNN 학습 효과 실질화 |
| Phase 1 행동-시뮬레이션 연결 | PPO 학습 의미 확보 |
| pytest 도입 + GitHub Actions CI | 회귀 자동 감지 |
| 재현성 설정 완성 (`cudnn.deterministic`) | 실험 신뢰성 확보 |

### 중기 (1~2개월) — 구조 고도화

| 항목 | 기대 효과 |
|------|-----------|
| YAML 설정 체계 통합 | 실험 관리 효율화 |
| Monte Carlo 병렬화 | 평가 속도 5~10× 향상 |
| Pareto 후보 실시뮬레이션 기반화 | HITL 신뢰도 향상 |
| MAPPO + League PFSP 통합 | 다에이전트 전략 다양성 |
| ManeuverEngine 통합 | 공간 기동 현실화 |
| GNN 노드 동적 업데이트 | 예측 정확도 향상 |

### 장기 (3개월 이상) — 연구 확장

| 항목 | 기대 효과 |
|------|-----------|
| IRL ↔ RL 파이프라인 완성 | 교리 기반 자동 보상 설계 |
| HITL 선호도 → RL 재학습 루프 | 진정한 Human-in-the-Loop 완성 |
| 교리 준수도 → 보상 직접 연결 | 교리 준수 학습 가능화 |
| 다도메인 시나리오 통합 | 연구 확장성 |

---

## 6. 종합 평가

| 항목 | 평가 | 비고 |
|------|------|------|
| 아키텍처 설계 | ★★★★☆ | 계층 분리 우수, 모듈 간 연결 일부 미완 |
| 시뮬레이터 물리 모델 | ★★★☆☆ | 란체스터 구현 양호, 공간 이동 부재 |
| GNN / 불확실성 모델 | ★★★★☆ | Bayesian 설계 탄탄, 모드 관리 버그 존재 |
| RL 학습 파이프라인 | ★★★☆☆ | Phase 설계 우수, 행동-환경 연결 취약 |
| HITL 인터페이스 | ★★★☆☆ | 제약 구조 우수, Pareto 예측 신뢰성 미흡 |
| 설명 가능성 | ★★★★☆ | AAR 완성도 높음, 어텐션 연결 필요 |
| 코드 품질 | ★★★☆☆ | 타입 힌트 양호, 테스트 프레임워크 미비 |
| 문서화 | ★★★★☆ | README 충실, 코드-문서 정합성 개선 여지 |

**전반적 진단**: FALCON은 연구 프레임워크로서 아이디어와 구조가 탄탄하며 각 모듈의 독립 완성도가 높다. 핵심 과제는 **모듈 간 연결의 완성**이다. 특히 ① 행동이 시뮬레이션에 실질 반영, ② GNN 학습의 실질화, ③ HITL 루프 완성, ④ 테스트 자동화가 우선 해결되어야 연구 결과의 신뢰성이 확보된다.
