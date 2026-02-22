# FALCON 프로젝트 종합 분석 보고서 v2

**작성일**: 2026-02-22
**분석 기준 브랜치**: `claude/falcon-analysis-report-Km9hV`
**이전 보고서**: `FALCON_COMPREHENSIVE_ANALYSIS.md` (2026-02-21)
**분석 범위**: 전체 코드베이스 — `ontology / simulator / gnn_model / rl_agent / hitl / explainability / evaluation / utils / train.py`

---

## 0. 요약 (Executive Summary)

v1 분석(2026-02-21) 이후 **15개 P0~P3 개선 항목 중 12개가 구현 완료**되었다. 특히 Phase 4 전체 파이프라인(HITL 선호도 → RL 재학습), ManeuverEngine 통합, IRL 보상 로더, 교리 준수도 보상 연결 등 핵심 장기 과제가 해결되었다. 현재 코드베이스는 학술 연구 프레임워크로서 **구조적 완성도가 높다**.

그럼에도 3개의 버그와 3개의 미완성 기능, 그리고 이번 분석에서 신규로 발견된 5개의 문제가 잔존한다.

| 구분 | v1 개수 | 해결됨 | 잔존 |
|------|---------|--------|------|
| P0 즉시 수정 버그 | 6 | 5 | 1 |
| P1 단기 기능 완성 | 6 | 5 | 1 |
| P2 중기 시스템 품질 | 6 | 5 | 1 |
| P3 장기 연구 확장 | 6 | 6 | 0 |
| **신규 발견 (v2)** | — | — | **5** |

---

## 1. 프로젝트 아키텍처 재검토

### 1-1. 계층 구조 (v2 기준)

```
┌────────────────────────────────────────────────────────────────┐
│                    FALCON v2 아키텍처                          │
├────────────┬───────────────────────────────────────────────────┤
│ 계층 1     │ ontology/                                         │
│            │  combat_schema.py   42종 UnitType, BranchType     │
│            │  doctrine_encoder.py 교리 준수도 평가 (신규 연동) │
│            │  multidomain.py     다도메인 스키마 (미연결)       │
├────────────┼───────────────────────────────────────────────────┤
│ 계층 2     │ simulator/                                        │
│            │  lanchester_engine.py  교전 엔진                  │
│            │  maneuver_engine.py    공간 기동 엔진 (★신규 연동)│
│            │  fog_of_war.py         커리큘럼 FOW 필터           │
│            │  resource_manager.py   탄약/연료 소모              │
│            │  combat_dynamics.py    BDA/EMS/보급 통합 관리자   │
│            │  mixed_lanchester.py   다도메인 교전 (미연결)      │
├────────────┼───────────────────────────────────────────────────┤
│ 계층 3     │ gnn_model/                                        │
│            │  bayesian_hgt.py  MC Dropout HGT-GNN              │
│            │  (노드 동적 업데이트 연동 ★신규)                   │
├────────────┼───────────────────────────────────────────────────┤
│ 계층 4     │ rl_agent/                                         │
│            │  blue_agent.py    PPO + 교리 보너스 (★신규)       │
│            │  red_agent.py     Red PPO                         │
│            │  self_play_trainer.py Phase A~D 커리큘럼          │
│            │  mappo.py         MAPPO (미연결)                  │
│            │  league_selfplay.py   League (미연결)             │
│            │  inverse_rl.py    MaxEntropy IRL + IRLRewardLoader│
├────────────┼───────────────────────────────────────────────────┤
│ 계층 5     │ hitl/                                             │
│            │  pareto_generator.py    동적 win_base (★신규)     │
│            │  preference_learner.py  지휘관 선호도 학습         │
│            │  preference_reward_adapter.py  ★신규: 선호도→보상 │
├────────────┼───────────────────────────────────────────────────┤
│ 계층 5'    │ explainability/                                   │
│            │  auto_aar_generator.py  자동 AAR 생성             │
│            │  attention_viz.py       어텐션 시각화             │
├────────────┼───────────────────────────────────────────────────┤
│ 계층 6     │ evaluation/                                       │
│            │  monte_carlo.py         ProcessPool 병렬화 (★신규)│
│            │  historical_benchmark.py 역사적 벤치마크          │
└────────────┴───────────────────────────────────────────────────┘
          ↑
  train.py — Phase 1/2/3/4 통합 학습 루프 (★Phase 4 신규)
  utils/  — reproducibility.py (cudnn.deterministic ★신규)
           config_loader.py (YAML 설정 ★신규)
```

### 1-2. 학습 파이프라인 전체 흐름 (v2)

```
Phase 1: GNN + Blue PPO 기초 학습
  ScenarioFactory → FogOfWarFilter → BayesianHGT → BlueAgent(PPO)
  → ManeuverEngine(위치갱신) → LanchesterEngine(교전)
  → kg.update_node_features() → DoctrineEncoder(교리평가)
  → compute_reward(prev_hc, doctrine_score) → PPO update

Phase 2: Self-Play
  SelfPlayTrainer → Blue vs Red PPO (Phase A→D 커리큘럼)
  → Nash Gap 수렴 판단

Phase 3: HITL
  ParetoStrategyGenerator(dynamic_win_base) → CommanderPreferenceLearner
  → 선호도 JSON 저장

Phase 4: 선호도 반영 재학습 ★신규
  PreferenceRewardAdapter.from_file() + IRLRewardLoader.from_file()
  → scaled_reward로 Phase 1 루프 재실행
```

---

## 2. v1 → v2 변경사항 추적

### ✅ 해결된 P0 버그

#### P0-1. train.py Phase 1 보상 계산 오류 → **해결**

```python
# v2 수정 후 (train.py:227, 282~288)
prev_blue_hc = initial_blue_hc      # 에피소드 시작 전 초기화
...
reward = blue_agent.compute_reward(
    step_result,
    prev_blue_hc,                   # ← 직전 스텝 병력 (수정됨)
    step_result.blue_total_headcount,
    avg_unc, ...
)
prev_blue_hc = step_result.blue_total_headcount   # 스텝마다 갱신
```

#### P0-2. monte_carlo.py Red 사상자 항상 0 → **해결**

```python
# v2 수정 후 (monte_carlo.py:80)
red_casualties += step.red_total_casualties   # * 0 제거됨
```

#### P0-3. resource_manager.py 유닛 타입 키 오타 → **해결**

`FUEL_CONSUMPTION_RATE`의 `"electronics_warfare"` → `"electronic_warfare"`로 수정 확인됨 (L47).

#### P0-5. BayesianHGT 훈련/추론 모드 미복원 → **해결**

```python
# v2 수정 후 (bayesian_hgt.py:202~216)
def predict_with_uncertainty(self, x, adj):
    was_training = self.training
    self.train()
    with torch.no_grad():
        ...
    self.train(was_training)   # ← 호출 전 모드로 복원
```

#### P0-6. blue_agent.py 중복 import → **해결**

파일 말미 중복 `import torch.nn.functional as F` 제거 확인됨.

---

### ✅ 해결된 P1 기능 완성

#### P1-1. Phase 1 행동-시뮬레이션 연결 → **해결**

```python
# v2 train.py:265~266
action_pairs = _build_blue_action_pairs(kg, action, ...)
step_result = engine.run_step(kg, action_pairs=action_pairs)
```

`_build_blue_action_pairs()`가 행동 유형(ADVANCE/FLANK/WITHDRAW 등)에 따라 공격 대상 우선순위(weakest/strongest/balanced)를 결정한 뒤 시뮬레이션에 반영한다.

#### P1-2. GNN risk_score 타깃 동적 계산 → **해결**

```python
# v2 train.py:297~302
blue_cas = float(step_result.blue_total_casualties)
red_cas  = float(step_result.red_total_casualties)
risk_score = blue_cas / (blue_cas + red_cas + 1.0)  # [0, 1]
gnn_batch_targets.append({
    "blue_casualties": torch.tensor(blue_cas, ...),
    "risk_score":      torch.tensor(risk_score, ...),  # 동적 계산
})
```

#### P1-6. data/ 연결 → **해결** (IRLRewardLoader)

```python
# inverse_rl.py:401~432
@classmethod
def from_file(cls, path: str) -> "IRLRewardLoader":
    with open(path) as f:
        data = json.load(f)
    raw_weights = data.get("learned_reward_weights", {})
    ...
```

`train_phase4()`에서 `data/irl_demos_summary.json` 자동 로드.

---

### ✅ 해결된 P2 시스템 품질

#### P2-1. 설정 분산 (YAML 도입) → **해결**

`utils/config_loader.py` + `configs/` 디렉토리. `train.py`에서 `--config configs/phase1.yaml`로 오버라이드 가능.

#### P2-2. 로깅 체계 → **해결**

```python
# train.py:27~50
logger = logging.getLogger("falcon")
setup_logger(checkpoint_dir, run_id)
# 콘솔(INFO) + 파일(DEBUG) 분리, run_id 타임스탬프 기반
```

#### P2-3. 재현성 설정 완성 → **해결**

```python
# utils/reproducibility.py
def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

#### P2-4. Monte Carlo 병렬화 → **해결**

```python
# monte_carlo.py:278
with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
    futures = {executor.submit(_mc_run_worker, a): a for a in worker_args}
```

`_mc_run_worker()`가 top-level 함수로 선언되어 pickle 직렬화 가능.

#### P2-5. Pareto 후보 전투력 기반화 → **부분 해결**

```python
# pareto_generator.py:129~151
@staticmethod
def _dynamic_win_base(blue_cp, red_cp, win_delta=0.0):
    cp_ratio = blue_cp / max(red_cp, 1e-6)
    k = 2.5
    base = 1.0 / (1.0 + math.exp(-k * (cp_ratio - 1.0)))
    return float(np.clip(base + win_delta, 0.10, 0.97))
```

전투력 비율 기반 logistic 함수로 `win_base`를 동적 계산한다. 그러나 **실제 Monte Carlo 시뮬레이션 N회 실행으로 검증하는 단계는 아직 미구현**이다.

#### P2-6. 보상 스케일 재조정 → **해결**

```python
# blue_agent.py:356~358
_W_WIN        =  10.0
_W_FORCE_SAVE =   0.15   # 이전 0.01 → 15× 상향
_W_DOCTRINE   =   1.0    # 교리 준수 보너스 신규 추가
```

---

### ✅ 해결된 P3 장기 연구 확장

#### P3-1. 공간 이동 모델 통합 → **해결**

```python
# train.py:259~262
blue_targets = _build_blue_maneuver_targets(kg, action, ...)
maneuver_result = maneuver_engine.run_maneuver_step(kg, blue_targets=blue_targets)
```

`_build_blue_maneuver_targets()`가 행동 유형별 목표 좌표를 계산하고 `ManeuverEngine`이 실제 유닛 위치를 갱신한다. ADVANCE(적 중심 전진), FLANK(90도 측면 좌표), WITHDRAW(후방 이동), SUPPORT(최근접 아군 방향)로 분기.

#### P3-2. GNN 노드 동적 업데이트 → **해결**

```python
# train.py:270
kg.update_node_features()   # 교전 후 변경된 유닛 상태 → 그래프 노드 동기화
```

#### P3-3. HITL 선호도 → RL 정책 반영 루프 → **해결**

Phase 4 전체 파이프라인이 `train_phase4()`로 완성되었다. `PreferenceRewardAdapter.from_file()`이 선호도 JSON을 로드하고, 각 보상 항목(승리/병력절감/사상자/교리 등)에 선호도 배율을 적용한다.

#### P3-4. IRL 파이프라인 → **해결**

`IRLRewardLoader.from_file("data/irl_demos_summary.json")`으로 사전 학습된 IRL 가중치를 로드하고, `compute_irl_bonus(kg, step_result, maneuver_result)`로 교전 결과를 12차원 피처로 변환해 IRL 보너스를 계산한다.

#### P3-5. 교리 준수도 → 보상 반영 → **해결**

```python
# train.py:279~286
compliance = doctrine_encoder.evaluate(kg, step_t, step_dict)
reward = blue_agent.compute_reward(
    ..., doctrine_score=compliance.total_score
)
# blue_agent.py:411
doctrine_bonus = self._W_DOCTRINE * (doctrine_score - 0.5)   # [-0.5, +0.5]
```

#### P3-6. 다도메인 시나리오 → **부분 해결**

`ontology/multidomain.py`의 스키마는 완성되어 있으나, `ScenarioFactory`와의 연결 및 `train.py --domain` 옵션은 아직 미구현이다.

---

## 3. 잔존 문제 (v1에서 미해결)

### R-1. MAPPO NumpyUnitActor 역전파 불가 (P1-4 잔존)

**위치**: `rl_agent/mappo.py:59~102`

`NumpyUnitActor`는 순수 NumPy 행렬 연산으로 구현되어 있어 역전파(backpropagation)가 불가능하다. `MAPPOManager`는 `train.py`의 어느 Phase에서도 호출되지 않는다.

```python
class NumpyUnitActor:
    def __init__(self, obs_dim, n_actions, hidden_dim=64, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(obs_dim, hidden_dim) * 0.1   # ← numpy, 그래디언트 없음
        self.W2 = rng.randn(hidden_dim, n_actions) * 0.01
```

`NumpyCentralizedCritic`도 동일하게 역전파가 불가능하다.

**잔존 이유**: MAPPO를 `torch.nn.Module`로 전환하는 작업이 복잡해 후순위로 밀린 것으로 판단.

---

### R-2. Pareto 후보가 실제 시뮬레이션으로 검증되지 않음 (P2-5 부분 잔존)

**위치**: `hitl/pareto_generator.py:153~252`

`win_base`가 logistic 함수로 동적 계산되는 점은 개선되었으나, 각 Pareto 후보를 실제 시뮬레이션 N회로 평가하는 `MCParetoValidator` 통합은 미완성이다. `mc_eval_runs=100` 파라미터가 생성자에 있으나 실제 사용되지 않는다.

```python
class ParetoStrategyGenerator:
    def __init__(self, n_candidates=5, mc_eval_runs=100):
        self.mc_eval_runs = mc_eval_runs   # ← 설정되지만 어디서도 사용 안 됨
```

---

### R-3. 다도메인 ScenarioFactory 미연결 (P3-6 부분 잔존)

**위치**: `ontology/multidomain.py`

`Multidomain` 스키마가 존재하나 `ScenarioFactory`와 연동되지 않아 `train.py`에서 `--domain` 옵션으로 다도메인 시나리오를 생성할 수 없다.

---

## 4. 신규 발견 문제 (v2 신규)

### N-1. monte_carlo.py 워커에서 에이전트 행동이 시뮬레이션에 미반영

**위치**: `monte_carlo.py:76~78`, `_mc_run_worker():68~85`

Monte Carlo 평가 루프에서 `agent.select_action()` 반환값이 `engine.run_step()`에 전달되지 않는다. 행동이 무시되고 자동 교전(`_auto_pair_units`)으로 실행된다.

```python
# monte_carlo.py:76~78 (문제)
agent.select_action(state, deterministic=True)   # 반환값 버림
step = engine.run_step(kg)                       # action_pairs 없음
```

이는 train.py Phase 1이 수정된 것과 달리 evaluation 루프는 수정이 적용되지 않은 상태다. **훈련된 에이전트를 평가할 때 행동이 실제로 반영되지 않으므로 평가 결과의 신뢰성이 낮다.**

**개선**: `agent.select_action()` 반환값을 `_build_blue_action_pairs()`에 전달 후 `engine.run_step(kg, action_pairs=pairs)`로 연결.

---

### N-2. self_play_trainer.py ManeuverEngine 미연동

**위치**: `rl_agent/self_play_trainer.py`

Phase 2(Self-Play)에서는 `ManeuverEngine`이 사용되지 않는다. Phase 1에서 ManeuverEngine으로 학습한 에이전트가 Phase 2에서 공간 이동 없이 학습하게 되어 **Phase 간 환경 불일치(distribution shift)**가 발생한다.

```python
# self_play_trainer.py: ManeuverEngine 임포트 없음
from simulator.lanchester_engine import LanchesterEngine   # 교전만
from simulator.fog_of_war import FogOfWarFilter            # FOW만
# ManeuverEngine 없음 ← 문제
```

**개선**: `SelfPlayTrainer` 내부에 `ManeuverEngine`을 추가하고 Phase 1과 동일한 `_build_blue_maneuver_targets()` 로직 적용.

---

### N-3. IRLRewardLoader time_pressure 피처 미연동

**위치**: `rl_agent/inverse_rl.py:471`

```python
def _build_state_dict(self, kg, step_result, maneuver_result=None) -> Dict:
    return {
        ...
        "time_pressure": 0.5,   # ← 스텝 진행도 미연동, 항상 0.5 고정
        ...
    }
```

`time_pressure`는 IRL 피처 중 하나인데 실제 스텝 번호(`step_t / max_steps`)가 아닌 상수 0.5로 고정되어 있다. IRL 보너스 계산 시 시간 압박 효과가 반영되지 않는다.

**개선**: `compute_irl_bonus()`에 `step_t`, `max_steps` 파라미터를 추가하고 `time_pressure = 1.0 - step_t / max_steps`로 동적 계산.

---

### N-4. PhaseA~D 경계에서 GNN 체크포인트 미전달

**위치**: `rl_agent/self_play_trainer.py` + `train.py:347~368`

Phase 2 (`train_phase2`)는 `SelfPlayTrainer`를 생성할 때 Phase 1에서 학습한 GNN 체크포인트를 로드하는 경로를 전달하지 않는다.

```python
# train.py:353~361
def train_phase2(args):
    config = SelfPlayConfig(
        total_episodes=args.episodes, ...
    )
    trainer = SelfPlayTrainer(config)
    # ← Phase 1 GNN 체크포인트 경로 미전달
```

결과적으로 Phase 2 에이전트는 GNN 불확실성 없이 고정 상태 벡터로만 학습하게 된다.

**개선**: `SelfPlayConfig`에 `gnn_checkpoint_path` 필드 추가 후 `SelfPlayTrainer`에서 BayesianHGT를 로드해 상태 벡터 생성에 활용.

---

### N-5. 보상 스케일 — survival_bonus 이중 계산 위험

**위치**: `train.py:589~598` (Phase 4)

Phase 4에서 `survival_bonus_raw`가 계산될 때, `adapter.compute_scaled_reward()`에도 동일한 항목이 독립적으로 스케일링된다. 단일 에피소드 종료 시 `force_ratio × _W_FORCE_RATIO` 보너스와 `PreferenceRewardAdapter`의 `survival_scale`이 중복 적용될 수 있다.

```python
# train.py:596~617
survival_bonus_raw = (blue_agent._W_FORCE_RATIO
                      * step_result.blue_total_headcount / max(initial_blue_hc, 1)
                      if terminal and initial_blue_hc > 0 else 0.0)
...
reward = adapter.compute_scaled_reward(
    ...
    survival_bonus=survival_bonus_raw,   # 선호도 배율 추가로 곱해짐
    ...
)
```

선호도 가중치가 1.0보다 크면 종료 스텝에서 보상 스파이크가 과도하게 발생한다.

**개선**: 종료 스텝 보너스에 별도 클리핑(`max(survival_bonus, 5.0)`) 추가 또는 `adapter`에서 survival_bonus를 스케일링하지 않도록 분리.

---

## 5. 코드 품질 심층 분석

### 5-1. 모듈별 완성도 매트릭스

| 모듈 | 기능 완성 | 테스트 | 파이프라인 연결 | 전체 |
|------|----------|--------|----------------|------|
| `ontology/combat_schema.py` | ★★★★★ | ★★★☆☆ | ★★★★☆ | **★★★★☆** |
| `ontology/doctrine_encoder.py` | ★★★★☆ | ★★★☆☆ | ★★★★☆ | **★★★★☆** |
| `simulator/lanchester_engine.py` | ★★★★☆ | ★★★☆☆ | ★★★★★ | **★★★★☆** |
| `simulator/maneuver_engine.py` | ★★★★☆ | ★★☆☆☆ | ★★★★☆ | **★★★☆☆** |
| `simulator/combat_dynamics.py` | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | **★★★☆☆** |
| `gnn_model/bayesian_hgt.py` | ★★★★★ | ★★★☆☆ | ★★★★☆ | **★★★★☆** |
| `rl_agent/blue_agent.py` | ★★★★★ | ★★★☆☆ | ★★★★★ | **★★★★★** |
| `rl_agent/self_play_trainer.py` | ★★★★☆ | ★★☆☆☆ | ★★★★☆ | **★★★☆☆** |
| `rl_agent/mappo.py` | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ | **★★☆☆☆** |
| `rl_agent/inverse_rl.py` | ★★★★☆ | ★★★☆☆ | ★★★★☆ | **★★★★☆** |
| `hitl/pareto_generator.py` | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | **★★★☆☆** |
| `hitl/preference_reward_adapter.py` | ★★★★☆ | ★★☆☆☆ | ★★★★★ | **★★★★☆** |
| `evaluation/monte_carlo.py` | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | **★★★☆☆** |

### 5-2. 세부 코드 품질 지적 (신규 포함)

| 항목 | 위치 | 내용 |
|------|------|------|
| MC 평가 행동 미반영 | `monte_carlo.py:76` | `select_action()` 결과 버려짐 (N-1) |
| time_pressure 고정값 | `inverse_rl.py:471` | `0.5` 하드코딩 (N-3) |
| ManeuverEngine 미사용 | `self_play_trainer.py` | Phase 2 환경 불일치 (N-2) |
| GNN 체크포인트 미전달 | `train.py:353` | Phase 1→2 연속성 단절 (N-4) |
| survival_bonus 이중 스케일 | `train.py:596` | 종료 스텝 보상 스파이크 (N-5) |
| FogOfWar O(N²) 비용 | `fog_of_war.py:121` | 매 스텝 공간관계 재빌드 (v1 잔존) |
| 강건성 수식 문제 | `monte_carlo.py:336` | `win_rate × (1 - std(0/1))` CVaR 대체 권장 |
| 타입 힌트 불완전 | `pareto_generator.py` 등 | `kg` 파라미터 타입 미지정 |
| 테스트 프레임워크 | `tests/` 전체 | pytest 미사용, CI 미연결 |
| 잔차 연결 조건부 생략 | `bayesian_hgt.py:77` | 불일치 시 조용히 생략됨 |

---

## 6. 아키텍처 심층 분석 — 주요 모듈

### 6-1. BayesianHGT — 불확실성 정량화 품질

**강점**:
- MC Dropout(N=20회 forward) + Epistemic/Aleatoric 분리 구조가 탄탄하다.
- `predict_with_uncertainty()` → 95% CI까지 반환.
- `was_training` 패턴으로 모드 관리 버그 수정 완료.
- 어텐션 가중치 저장(`last_attention_weights`)으로 Explainability 기반 마련.

**구조적 한계**:
```python
# bayesian_hgt.py:77
out = self.norm(out + x if x.size(-1) == out.size(-1) else out)
```
잔차 연결이 차원 불일치 시 조용히 생략된다. 첫 번째 레이어(in_dim=128 → hidden_dim=128)에서는 문제없으나, 모델 하이퍼파라미터 변경 시(hidden_dim ≠ node_in_dim) 잔차 없이 학습되어 수렴이 느려질 수 있다.

**GNN 풀링 방식**:
```python
h_pool = ((h_mean + h_max) / 2)   # Mean+Max pooling
```
Mean+Max 절충 풀링은 적절하나, 전장의 소수 고전투력 유닛이 평균에 묻힐 수 있다. Attention Pooling으로 중요 유닛 가중치를 높이면 예측 정밀도 향상 가능.

### 6-2. BlueAgent 보상 함수 — 설계 분석

```
R = +10.0 × WinSignal              # 승/패 (±10)
  - 0.05  × BlueCasualties         # 아군 사상자 패널티
  + 0.15  × ForceReduction         # 병력 절감 (15× 상향)
  + 2.0   × SurvivalRatio (종료시) # 잔존 병력 비율 보너스
  + 0.03  × RedCasualties          # 적 피해
  + 1.0   × (DoctrineScore - 0.5)  # 교리 준수 [-0.5, +0.5]
  - UncertaintyPenalty             # 불확실성 패널티
```

**개선된 점**: 병력 절감 가중치가 15배 상향되어 "최소 병력 임무 달성"이라는 핵심 목표가 보상에 더 잘 반영된다.

**잔존 불균형**: 승리 보상(±10.0)이 여전히 가장 큰 항목이다. 누적 episode에서 병력 절감 보상(`0.15 × 수십 병사 × 수십 스텝 ≈ 수십`)과 비교할 때 동등한 수준이 되었으나, 1스텝 단위 절감은 여전히 승패 보상에 비해 미약할 수 있다.

**DoctrineEncoder 통합 효과**: `compliance.total_score ∈ [0, 1]`이 매 스텝 계산되어 보상에 반영된다. score=1.0이면 +0.5, score=0.0이면 -0.5의 보너스가 발생한다. 교리 일관성 학습이 가능해졌다.

### 6-3. combat_dynamics.py — 신규 통합 모듈 분석

`CombatDynamicsManager`는 4개의 하위 모듈(AmmoConsumptionModel, BDAEngine, SupplyStatusManager, ElectromagneticEnv)을 통합한다. 설계 자체는 우수하지만 **현재 train.py에서 사용되지 않는다**.

```python
# combat_dynamics.py:408~425 — 파이프라인 연동 주석이 있으나 미사용
class CombatDynamicsManager:
    """
    파이프라인 연동:
      1. 시뮬레이터 스텝 완료 후 step_update() 호출   ← train.py에 없음
      2. BDA 생성 → 로그 기록                         ← 미연결
      3. 탄약/연료 소모 → RL 보상 함수의 보급 패널티  ← 미연결
      4. EMS 상태 → GNN 불확실성 증폭에 반영          ← 미연결
    """
```

`combat_dynamics.py`의 `get_supply_penalty()`가 train.py 보상 함수에 추가되면 탄약/연료 고갈 시 RL 에이전트가 보급을 의식하는 행동을 학습할 수 있다.

### 6-4. MAPPO — 학습 가능성 한계

```python
# mappo.py:59~73
class NumpyUnitActor:
    def __init__(self, obs_dim=24, n_actions=8, hidden_dim=64, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(obs_dim, hidden_dim).astype(np.float32) * 0.1
        self.W2 = rng.randn(hidden_dim, n_actions).astype(np.float32) * 0.01
```

현재 MAPPO는 랜덤 초기화된 numpy 행렬로 행동을 선택하며, **어떠한 기울기도 계산하지 않는다**. 구조 자체(유닛별 Actor, 중앙화 Critic, 행동 마스크)는 올바르게 설계되어 있어 `torch.nn.Module`로의 전환만 이루어지면 즉시 실용적 MAPPO가 된다.

유닛별 관측 차원 24D, 전역 상태 64D, 행동 공간 8종 + 타입별 마스크 구조는 보존 가치가 높다.

---

## 7. 수치 분석 — 코드베이스 규모

| 항목 | 수치 |
|------|------|
| 총 Python 파일 수 | 35+ |
| 추정 총 코드 라인 | 20,000+ |
| UnitType 종류 | 42종 |
| RL 학습 Phase | 4단계 (Phase 1~4) |
| 불확실성 모델 차원 | 6D (PPO 상태 확장) |
| Pareto 목적함수 | 4개 (병력↓, 승률↑, 사상자↓, 시간↓) |
| IRL 피처 차원 | 12D (교리 기반) |
| MAPPO 행동 공간 | 8종 (타입별 마스크) |
| Monte Carlo 기본 실행 수 | 5,000회 |
| GNN MC Dropout 샘플 | 20회 |

---

## 8. 우선순위별 실행 로드맵 (v2)

### 즉시 (1~3일) — 신뢰성 복원

| 항목 | 위치 | 수정 내용 | 난이도 |
|------|------|-----------|--------|
| MC 평가 행동 연결 | `monte_carlo.py:76` | `select_action()` → `action_pairs` 생성 → `run_step(action_pairs=...)` | 하 |
| IRL time_pressure 동적화 | `inverse_rl.py:471` | `step_t / max_steps` 파라미터 전달 | 하 |
| survival_bonus 이중 스케일 | `train.py:596` | `adapter`에서 survival_bonus 클리핑 또는 분리 | 하 |

### 단기 (1~2주) — 일관성 확보

| 항목 | 기대 효과 |
|------|-----------|
| SelfPlayTrainer ManeuverEngine 추가 | Phase 1↔2 환경 일관성 확보 |
| Phase 1→2 GNN 체크포인트 전달 | 연속 학습 효과 보존 |
| pytest 전환 + GitHub Actions CI | 회귀 자동 감지 |
| CombatDynamicsManager train.py 연결 | 탄약/보급 패널티 학습 |

### 중기 (1~2개월) — 완성도 강화

| 항목 | 기대 효과 |
|------|-----------|
| MAPPO `torch.nn.Module` 전환 | 실질적 다에이전트 학습 가능화 |
| Pareto MCParetoValidator 실시뮬레이션 연결 | HITL 신뢰도 향상 |
| `Multidomain ScenarioFactory` 연결 | 공-해-지 합동 시나리오 학습 |
| GNN Attention Pooling 도입 | 예측 정밀도 향상 |

### 장기 (3개월 이상) — 연구 확장

| 항목 | 기대 효과 |
|------|-----------|
| League Self-Play PFSP 완성 | 전략 다양성 AlphaStar 수준 |
| CVaR 기반 강건성 지표 | 극단적 실패 시나리오 감지 |
| FogOfWar 공간 인덱스 최적화 | O(N²) → O(N log N) |

---

## 9. 종합 평가 (v2)

| 항목 | v1 평가 | v2 평가 | 변화 |
|------|---------|---------|------|
| 아키텍처 설계 | ★★★★☆ | ★★★★★ | ↑ Phase 4 완성 |
| 시뮬레이터 물리 모델 | ★★★☆☆ | ★★★★☆ | ↑ ManeuverEngine 통합 |
| GNN / 불확실성 모델 | ★★★★☆ | ★★★★☆ | → 버그 수정 완료 |
| RL 학습 파이프라인 | ★★★☆☆ | ★★★★☆ | ↑ 행동-환경 연결 완성 |
| HITL 인터페이스 | ★★★☆☆ | ★★★★☆ | ↑ 선호도→RL 루프 완성 |
| 설명 가능성 | ★★★★☆ | ★★★★☆ | → 유지 |
| 코드 품질 | ★★★☆☆ | ★★★☆☆ | → 테스트 프레임워크 미비 |
| 문서화 | ★★★★☆ | ★★★★★ | ↑ README 충실화 |
| 평가 신뢰성 | ★★★☆☆ | ★★★☆☆ | → MC 평가 행동 미반영 잔존 |

**v2 전반적 진단**: FALCON은 v1 대비 **핵심 파이프라인 연결 작업이 대부분 완성**되었다. Phase 4 재학습 루프, ManeuverEngine 통합, IRL/HITL 선호도 반영이 실현되어 연구 프레임워크로서의 가치가 크게 높아졌다. 현재 가장 시급한 과제는 **평가 루프(monte_carlo.py)의 행동 미반영 버그** 수정으로, 이것이 해결되어야 학습된 에이전트의 성능 수치를 신뢰할 수 있다. 그 다음 우선순위는 Phase 2의 ManeuverEngine 미연동과 MAPPO의 실질적 학습 가능화다.

---

## Appendix A. 파일별 핵심 API 참조

| 파일 | 핵심 클래스/함수 | 설명 |
|------|----------------|------|
| `ontology/combat_schema.py` | `ScenarioFactory`, `CombatKnowledgeGraph` | 시나리오 생성, 그래프 빌드 |
| `ontology/doctrine_encoder.py` | `DoctrineEncoder.evaluate()` | 매 스텝 교리 준수도 반환 |
| `simulator/lanchester_engine.py` | `LanchesterEngine.run_step()` | 교전 시뮬레이션 1스텝 |
| `simulator/maneuver_engine.py` | `ManeuverEngine.run_maneuver_step()` | 유닛 위치 갱신 |
| `simulator/combat_dynamics.py` | `CombatDynamicsManager.step_update()` | BDA/EMS/보급 통합 |
| `gnn_model/bayesian_hgt.py` | `BayesianHGT.predict_with_uncertainty()` | MC Dropout 불확실성 |
| `rl_agent/blue_agent.py` | `BlueAgent.compute_reward()`, `build_state_vector()` | PPO 보상, 상태 128D |
| `rl_agent/inverse_rl.py` | `IRLRewardLoader.compute_irl_bonus()` | IRL 교리 보너스 |
| `hitl/pareto_generator.py` | `ParetoStrategyGenerator.generate()` | Pareto 5후보 생성 |
| `hitl/preference_reward_adapter.py` | `PreferenceRewardAdapter.compute_scaled_reward()` | 선호도 보상 배율 |
| `evaluation/monte_carlo.py` | `MonteCarloEvaluator.evaluate()` | 병렬 MC 강건성 평가 |
| `train.py` | `train_phase1~4()` | Phase별 학습 루프 |

---

## Appendix B. 발견 버그 요약 (즉시 수정 대상)

```python
# N-1: monte_carlo.py — 행동 미반영 (즉시 수정)
# 현재
agent.select_action(state, deterministic=True)
step = engine.run_step(kg)

# 수정안
action, _, _ = agent.select_action(state, deterministic=True)
# action_pairs = _build_blue_action_pairs(kg, action, ...)  # 유틸 필요
step = engine.run_step(kg)  # → engine.run_step(kg, action_pairs=pairs)

# N-3: inverse_rl.py — time_pressure 고정 (즉시 수정)
# 현재
"time_pressure": 0.5,

# 수정안
"time_pressure": 1.0 - (step_t / max(max_steps, 1)),  # 파라미터 추가 필요

# N-5: train.py Phase 4 — survival_bonus 이중 스케일 (즉시 수정)
# 수정안: PreferenceRewardAdapter에서 survival_bonus 클리핑
survival_bonus_scaled = min(survival_bonus_raw * adapter.survival_scale, 5.0)
```
