# FALCON 적대적 RL 알고리즘 분석 및 개선 로드맵

**작성일**: 2026-02-23
**분석 브랜치**: `claude/falcon-analysis-report-Km9hV`
**전 보고서**: `FALCON_COMPREHENSIVE_ANALYSIS_v2.md`
**목적**: 최신 적대적 강화학습(Adversarial RL) 알고리즘 조사 및 FALCON 적용 방안 도출

---

## 0. 요약 (Executive Summary)

FALCON의 현행 적대적 학습 체계(Red PPO + League Self-Play + PFSP)는 AlphaStar(2019) 시대의 설계를 기반으로 한다. 2020~2025년 사이 다음의 핵심 발전이 있었으며, 이를 FALCON에 순차 통합하면 **더 강인하고 다양한 전략을 가진 Red 에이전트**를 생성할 수 있다.

| 우선순위 | 알고리즘 | 핵심 효과 | 구현 난도 |
|----------|---------|-----------|-----------|
| **P0** | Adversarial Policy (Gleave et al.) | 기존 Blue 에이전트의 취약점 자동 탐색 | 중 |
| **P0** | PAIRED / REPAIRED | 환경 자체를 적대적으로 생성 (커리큘럼) | 중 |
| **P1** | PSRO + α-Rank | 내쉬 근사 + 전략 다양성 보장 | 중상 |
| **P1** | NFSP (Neural Fictitious Self-Play) | 근사 내쉬 균형 수렴 보장 | 중 |
| **P2** | HAPPO / HATRPO | 이종 에이전트 이론적 수렴 보장 | 중상 |
| **P2** | Multi-Agent Transformer (MAT) | 유닛 간 어텐션 기반 협력 | 높음 |
| **P3** | RARL (Robust Adversarial RL) | 미니맥스 강건성 보장 | 중 |
| **P3** | NeuPL (Neural Population Learning) | 오픈-엔디드 전략 진화 | 높음 |

---

## 1. 현행 FALCON 적대적 학습 체계 진단

### 1-1. 현행 구성요소

```
rl_agent/
  red_agent.py         PPO 기반 Red 에이전트 (6가지 행동: Ambush/Counter-Attack/Deception/Fortify/Retreat/Artillery)
  league_selfplay.py   AlphaStar 방식 League (Main / Main-Exploiter / League-Exploiter + PFSP)
  self_play_trainer.py Phase A~D 커리큘럼 자기 대전
```

### 1-2. 식별된 한계

| 번호 | 한계 | 영향 |
|------|------|------|
| L-1 | Red 에이전트가 단순 PPO — 전략 다양성 부족 | Blue가 단일 Red 스타일에 과적합 |
| L-2 | PFSP 우선순위가 ELO 기반만 — 전략 공간 탐색 미흡 | 미발견 취약점 다수 잔존 |
| L-3 | 환경 파라미터(지형·날씨·병력비) 고정 | 훈련-평가 분포 이동 |
| L-4 | 내쉬 균형 수렴 미보장 | 순환 학습(cycling) 위험 |
| L-5 | 이종 유닛 역할(포병/기갑/보병) 간 이론적 수렴 보장 없음 | 일부 유닛 유형 과소학습 |
| L-6 | 적대적 관측 교란(Adversarial Observation Perturbation) 미구현 | GNN 취약성 미평가 |

---

## 2. 최신 적대적 RL 알고리즘 분류 및 분석

### 2-1. 적대적 에이전트 학습 (Adversarial Agent Training)

#### 2-1-1. Adversarial Policy (Gleave et al., NeurIPS 2020)

**핵심 아이디어**: 학습된 상대(Blue) 에이전트의 관측 공간에 비정형적 행동(adversarial perturbation)을 주입해 **무작위처럼 보이는 이상 행동으로 상대를 조종**한다.

```
Red 에이전트가 직접 싸우지 않고 → Blue가 이상 행동을 하도록 유도
예: 특정 위치에 대기 → Blue의 주의를 분산 → 주력 방향 돌파
```

**FALCON 적용 방안**:
- `RedAgent`에 `AdversarialPolicyMode` 추가
- Blue의 GNN 관측 벡터에 소규모 교란을 가하는 Red 행동 공간 확장
- `fog_of_war.py`의 `deception_prob`와 연동하여 전자전(EW) 기만 효과 모델링

**구현 스켈레톤**:
```python
class AdversarialPolicyRed(RedAgent):
    """Blue의 관측 공간을 교란하는 적대적 정책 Red"""

    def compute_adversarial_perturbation(
        self, blue_obs: np.ndarray, epsilon: float = 0.05
    ) -> np.ndarray:
        """FGSM 방식 관측 교란 (gradient 없이 근사)"""
        noise = np.random.uniform(-epsilon, epsilon, blue_obs.shape)
        return blue_obs + noise

    def select_action_adversarial(
        self, state: np.ndarray, blue_obs: np.ndarray
    ) -> Tuple[int, float, float]:
        """일반 PPO 행동 + 적대적 교란 행동 동시 반환"""
        action, lp, val = super().select_action(state)
        perturbed_obs = self.compute_adversarial_perturbation(blue_obs)
        return action, lp, val, perturbed_obs
```

---

#### 2-1-2. PAIRED (Dennis et al., NeurIPS 2020) & REPAIRED (Jiang et al., 2021)

**핵심 아이디어**: Protagonist(Blue), Antagonist(Red), Environment Generator 세 에이전트가 동시에 학습. Generator는 Antagonist가 이기되 Protagonist에게 불가능하지 않은 환경을 생성하여 **최대 후회(Regret) 최소화 커리큘럼**을 자동 구성한다.

```
목표: max_{env} [V(Antagonist, env) - V(Protagonist, env)]
     subject to: Protagonist의 성공률 > θ (너무 어렵지 않아야 함)
```

**FALCON 적용 방안**:
- `simulator/scenario_generator.py` (신규) 구현
- 매 훈련 에피소드마다 `ScenarioFactory` 파라미터를 적대적으로 변형
  - 지형 이점(terrain_advantage) 조정
  - Red 병력비 동적 조정
  - 날씨/가시거리 커리큘럼 강화
- `self_play_trainer.py`의 Phase B~C에 PAIRED 루프 삽입

**시나리오 생성 파라미터 공간**:
```yaml
scenario_adversarial_params:
  n_red_units: [3, 12]         # Red 병력 수
  terrain_advantage: [0.2, 0.9] # 지형 이점 (1.0 = Blue 유리)
  fog_density: [0.1, 0.8]       # 안개 농도
  blue_ammo_ratio: [0.4, 1.0]   # 아군 초기 탄약
  weather_penalty: [0.0, 0.5]   # 기상 패널티
```

---

### 2-2. 전략 공간 균형 (Strategy Space Equilibrium)

#### 2-2-1. PSRO (Policy Space Response Oracles, Lanctot et al., NeurIPS 2017; 이후 α-Rank 결합 2019~2024 지속 발전)

**핵심 아이디어**: Double Oracle의 확장. 매 반복마다 현재 전략 집합에 대한 **최선 응답(Best Response)**을 추가하고, **내쉬 균형 혼합 전략** 위에서 샘플링한다.

```
반복:
  1. 현재 전략 집합 π = {π_1, ..., π_k} 에 대해 payoff 행렬 M 계산
  2. M에서 내쉬 균형 σ* 계산 (또는 α-Rank 계산)
  3. σ* 위에서 샘플링한 상대에 대한 Best Response π_{k+1} 추가
  4. π에 π_{k+1} 추가 → 수렴 또는 최대 반복 도달까지 반복
```

**α-Rank (2019)**와의 결합:
- 내쉬 균형 대신 **진화 게임 이론**의 정적 분포 사용
- 순환 방지: 내쉬 균형이 없는 게임에서도 수렴 보장
- FALCON에서: ELO 대신 α-Rank 레이팅으로 전략 다양성 평가

**FALCON 적용 방안**:
- `LeagueManager`의 `pfsp_scheduler`를 PSRO 오라클로 교체
- `nash_gap_estimate()`를 실제 α-Rank 계산으로 업그레이드
- 각 Blue/Red 스냅샷을 PSRO 전략 집합 원소로 취급

**구현 스켈레톤**:
```python
class PSROOracle:
    """PSRO 전략 집합 + α-Rank 레이팅"""

    def __init__(self):
        self.strategy_pool: List[LeagueAgent] = []
        self.payoff_matrix: np.ndarray = np.empty((0, 0))

    def compute_alpha_rank(self, alpha: float = 1e-2) -> np.ndarray:
        """
        α-Rank 고정 분포 계산
        payoff 행렬 M에서 Markov Chain 전이 행렬 구성 후
        정적 분포(stationary distribution) 계산
        """
        n = len(self.strategy_pool)
        if n < 2:
            return np.ones(n) / max(n, 1)

        # Markov Chain 전이 행렬 T
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    delta = self.payoff_matrix[i, j] - self.payoff_matrix[j, i]
                    T[i, j] = alpha * delta / (1 - np.exp(-alpha * delta + 1e-10))

        # 정규화 → 고유벡터 계산
        T -= np.diag(T.sum(axis=1))
        # 정적 분포
        eigvals, eigvecs = np.linalg.eig(T.T)
        stat = eigvecs[:, np.argmin(np.abs(eigvals))].real
        return np.abs(stat) / np.abs(stat).sum()

    def select_opponent_psro(self, alpha_rank: np.ndarray) -> LeagueAgent:
        """α-Rank 분포로 상대 샘플링"""
        idx = np.random.choice(len(self.strategy_pool), p=alpha_rank)
        return self.strategy_pool[idx]
```

---

#### 2-2-2. NFSP (Neural Fictitious Self-Play, Heinrich & Silver, 2016; 이후 지속 발전)

**핵심 아이디어**: 각 에이전트가 두 개의 네트워크를 유지:
1. **Best Response (BR) 네트워크**: 현재 상대 정책의 평균에 최적 응답 (RL)
2. **Average Strategy (AS) 네트워크**: 지금까지의 자신 정책 평균 학습 (지도학습)

η 확률로 BR, (1-η)로 AS를 사용하여 행동 → **내쉬 균형에 수렴** 보장.

**FALCON 적용 방안**:
- `RedAgent`와 `BlueAgent` 모두에 AS 네트워크 추가 (파라미터 2배)
- `RolloutBuffer` 외에 **Reservoir Sampling Buffer** 추가 (과거 전 정책 균등 샘플링)
- `self_play_trainer.py` Phase B에서 NFSP 루프 구현

**구현 스켈레톤**:
```python
class NFSPAgent:
    """Neural Fictitious Self-Play 에이전트"""

    def __init__(self, n_actions: int, anticipatory_param: float = 0.1):
        self.br_network  = ActorCritic(n_actions=n_actions)   # Best Response
        self.as_network  = ActorCritic(n_actions=n_actions)   # Average Strategy
        self.eta = anticipatory_param                          # BR 선택 확률
        self.reservoir_buffer = ReservoirBuffer(capacity=200_000)
        self.sl_optimizer = optim.Adam(self.as_network.parameters(), lr=1e-3)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        if np.random.random() < self.eta:
            return self.br_network.get_action(state)    # Best Response
        else:
            return self.as_network.get_action(state)    # Average Strategy

    def update_average_strategy(self, batch_states, batch_actions):
        """지도학습으로 평균 정책 업데이트"""
        logits, _ = self.as_network(batch_states)
        loss = F.cross_entropy(logits, batch_actions)
        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()
```

---

### 2-3. 이종 다중 에이전트 이론 (Heterogeneous Multi-Agent Theory)

#### 2-3-1. HAPPO (Heterogeneous-Agent Proximal Policy Optimization, Zhong et al., ICLR 2022)

**핵심 아이디어**: 이종 에이전트(보병/기갑/포병 등)의 **순차적 업데이트(Sequential Update Scheme)**로 단조 개선(Monotonic Improvement) 이론적 보장.

기존 MAPPO의 문제:
```
동시 업데이트 시 에이전트 A의 업데이트가 에이전트 B에게 불리한 영향
→ 협력 보상이 있어도 수렴 불안정
```

HAPPO 해법:
```
에이전트 1 업데이트 → 중요도 비율 계산 → 에이전트 2 업데이트 (이전 비율 반영) → ...
→ 결합 단조 개선 보장
```

**FALCON 적용 방안**:
- `MAPPOManager.update()` 내부 루프를 유닛 유형 순서별 순차 업데이트로 변경
- 유닛 유형 순서: `logistics → signal → engineer → infantry → armor → artillery → aviation → electronic_warfare`
  (지원 → 전투 순서로 업데이트, 전투 유닛이 지원 유닛 정책의 변화를 반영)
- **중요도 비율 곱(Importance Weight Accumulation)** 구현

**수식**:
$$L^{\text{HAPPO}}_i(\theta_i) = \mathbb{E}\left[\frac{\pi_i(a_i|o_i)}{\pi_i^{\text{old}}(a_i|o_i)} \cdot \prod_{j < i} \frac{\pi_j(a_j|o_j)}{\pi_j^{\text{old}}(a_j|o_j)} \cdot A_i\right]$$

---

#### 2-3-2. Multi-Agent Transformer (MAT, Wen et al., 2022)

**핵심 아이디어**: Transformer의 인코더-디코더 구조를 다중 에이전트에 적용. **에이전트 간 어텐션**으로 결합 행동 확률을 자기회귀적(auto-regressive)으로 분해:

$$\pi(a_1, ..., a_n | o) = \prod_{i=1}^{n} \pi_i(a_i | a_{<i}, o)$$

**FALCON 장점**:
- 유닛 수 가변(n_blue 동적 변화)에 자연스럽게 대응
- 아군 유닛 간 암묵적 통신(implicit communication) 내재화
- 기존 `NumpyUnitActor` × N 방식 대비 협력 행동 품질 향상

**구현 스켈레톤 (torch)**:
```python
class MATEncoder(nn.Module):
    """멀티-에이전트 상태 인코더"""
    def __init__(self, obs_dim=24, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(obs_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        # obs_batch: [batch, n_agents, obs_dim]
        x = self.embed(obs_batch)
        return self.transformer(x)  # [batch, n_agents, d_model]


class MATDecoder(nn.Module):
    """자기회귀적 행동 디코더"""
    def __init__(self, n_actions=8, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions + 1, d_model)  # +1 for <BOS>
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.action_head = nn.Linear(d_model, n_actions)

    def forward(self, memory: torch.Tensor,
                prev_actions: torch.Tensor) -> torch.Tensor:
        tgt = self.action_embed(prev_actions)
        out = self.transformer(tgt, memory)
        return self.action_head(out)  # [batch, n_agents, n_actions]
```

---

### 2-4. 강건성 보장 (Robustness Guarantees)

#### 2-4-1. RARL (Robust Adversarial Reinforcement Learning, Pinto et al., ICML 2017; SA-PPO 등 후속 발전)

**핵심 아이디어**: 미니맥스 최적화로 정책 강건성 보장:
$$\max_\theta \min_\phi \mathbb{E}[\sum_t r_t | \pi_\theta, \nu_\phi]$$

Protagonist(Blue)는 최대화, Adversary(Red)는 정책 공간이 아닌 **외란(perturbation) 공간**을 최적화.

**SA-PPO (State-Adversarial PPO, 2020)** 확장:
- 관측 벡터에 적대적 노이즈를 가하는 내부 최적화
- KL 제약으로 교란 크기 제한
- 학습된 정책이 관측 교란에 강건함 보장

**FALCON 적용 방안**:
- `blue_agent.py`의 `select_action`에 관측 강건화 옵션 추가
- 훈련 시 20% 확률로 관측 벡터에 ε-ball 내 적대적 노이즈 추가
- `fog_of_war.py`의 불확실성과 통합: 실제 전장 불확실성 = 적대적 교란 시뮬레이션

**구현**:
```python
def add_adversarial_obs_noise(
    obs: np.ndarray, epsilon: float = 0.05, mode: str = "uniform"
) -> np.ndarray:
    """관측 벡터에 적대적 노이즈 추가 (SA-PPO 방식)"""
    if mode == "uniform":
        noise = np.random.uniform(-epsilon, epsilon, obs.shape)
    elif mode == "gaussian":
        noise = np.random.normal(0, epsilon / 2, obs.shape)
    return np.clip(obs + noise.astype(np.float32), -10.0, 10.0)
```

---

#### 2-4-2. Domain Randomization + Adversarial Curriculum

**핵심 아이디어**: 훈련 환경 파라미터를 무작위로 또는 적대적으로 변형하여, 특정 환경 설정에 과적합되지 않도록 한다.

**FALCON 커리큘럼 확장안**:

```
Phase A: Fixed scenario (현재 구현)
Phase B: Domain Randomization - 환경 파라미터 무작위 변형
Phase C: PAIRED - 적대적 환경 생성 (Regret 최대화)
Phase D: REPAIRED - 안전 제약 추가 (너무 어려운 시나리오 제외)
Phase E: Open-Ended - 제한 없는 시나리오 공간 탐색
```

**변형 파라미터 목록**:
```python
DOMAIN_RAND_PARAMS = {
    # 전장 파라미터
    "n_red_units":        (3, 15),
    "terrain_advantage":  (0.2, 0.9),
    "fog_density":        (0.05, 0.9),

    # 아군 초기 상태
    "blue_ammo_ratio":    (0.5, 1.0),
    "blue_fuel_ratio":    (0.4, 1.0),
    "blue_morale":        (0.5, 1.0),

    # Red 전술 파라미터
    "red_aggressiveness": (0.3, 1.0),  # Red 행동 성향
    "red_coordination":   (0.0, 1.0),  # Red 협력도
}
```

---

### 2-5. 개방형 학습 (Open-Ended Learning)

#### 2-5-1. NeuPL (Neural Population Learning, Marris et al., NeurIPS 2022)

**핵심 아이디어**: 전략 풀(Population)을 확장하며 **내쉬 균형에 수렴하는 새 에이전트를 연속적으로 추가**. AlphaStar League보다 더 엄밀한 수렴 이론을 제공한다.

```
NeuPL vs AlphaStar League:
- AlphaStar: 경험적 스냅샷 + PFSP → 경험적 수렴
- NeuPL: Conditional Policy + Best Response → 이론적 수렴
```

**핵심 기법**: **Conditional Policy**
- 에이전트가 자신의 행동 공간이 아닌 **상대 정보를 조건으로** 행동
- 단일 네트워크가 전체 전략 집합을 표현 (파라미터 효율)
- `agent_id` 임베딩을 상태에 추가하여 역할 구분

**FALCON 적용 방안**:
- `LeagueManager` 스냅샷 추가 시 `agent_id` 임베딩 함께 저장
- Red 에이전트 네트워크에 `opponent_embedding` 입력 차원 추가

---

#### 2-5-2. ACCEL (Adaptive Curriculum via Emergent Complexity, Parker-Holder et al., NeurIPS 2022)

**핵심 아이디어**:
1. 버퍼에서 에이전트가 가장 어렵게 느끼는 환경 `e` 샘플링
2. `e`를 소폭 변형하여 새 환경 후보 생성
3. 에이전트의 성능 변화 측정 → 학습 신호가 가장 큰 환경 유지

```
ALPo (Active Learning for Policy Optimization) 관점:
편의(Regret)가 높은 시나리오에 더 많은 학습 시간 배분
```

**FALCON 적용 방안**:
- `self_play_trainer.py` Phase C에 ACCEL 루프 삽입
- 시나리오를 Scenario Buffer에 저장 후 성과 추적
- 성공률 30~70% 범위의 시나리오 우선 샘플링 (중간 난이도 학습)

---

## 3. FALCON 단기 구현 로드맵

### 3-1. 즉시 구현 가능 (P0, 1~2일)

#### 3-1-1. 관측 적대적 교란 (`red_agent.py` 확장)

```python
# red_agent.py 추가 메서드

def compute_observation_perturbation(
    self,
    target_obs: np.ndarray,
    epsilon: float = 0.05,
    deception_mode: str = "uniform"
) -> np.ndarray:
    """
    Blue 관측 벡터에 적대적 교란 추가 (전자전/기만 모델링)

    - 실제 전장에서 전자전(EW)이 아군 센서/통신을 교란하는 것을 모델링
    - epsilon: 교란 강도 (기본 5%)
    - deception_mode: "uniform" | "gaussian" | "targeted"
    """
    if deception_mode == "uniform":
        noise = np.random.uniform(-epsilon, epsilon, target_obs.shape)
    elif deception_mode == "gaussian":
        noise = np.random.normal(0.0, epsilon / 2.0, target_obs.shape)
    elif deception_mode == "targeted":
        # 가장 중요한 관측 차원(적 거리, 전투력)에 집중 교란
        noise = np.zeros_like(target_obs)
        key_dims = [9, 10, 11, 12]  # nearest_enemy 관련 차원
        noise[key_dims] = np.random.uniform(-epsilon * 3, epsilon * 3, len(key_dims))
    else:
        noise = np.zeros_like(target_obs)

    return np.clip(target_obs + noise.astype(np.float32), -10.0, 10.0)
```

#### 3-1-2. 도메인 랜덤화 시나리오 생성기 (`simulator/` 확장)

```python
# simulator/adversarial_scenario.py (신규 파일)

from dataclasses import dataclass
import numpy as np

@dataclass
class ScenarioDomainParams:
    """도메인 랜덤화 파라미터"""
    n_red_units:       int   = 5
    terrain_advantage: float = 0.5
    fog_density:       float = 0.3
    blue_ammo_ratio:   float = 1.0
    blue_fuel_ratio:   float = 1.0
    red_aggressiveness: float = 0.5


class DomainRandomizer:
    """시나리오 도메인 랜덤화 (PAIRED 전 단계)"""

    PARAM_RANGES = {
        "n_red_units":        (3, 12),
        "terrain_advantage":  (0.2, 0.9),
        "fog_density":        (0.05, 0.85),
        "blue_ammo_ratio":    (0.5, 1.0),
        "blue_fuel_ratio":    (0.4, 1.0),
        "red_aggressiveness": (0.3, 0.9),
    }

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.history: list = []

    def sample(self) -> ScenarioDomainParams:
        """균등 분포에서 파라미터 샘플링"""
        p = {}
        for k, (lo, hi) in self.PARAM_RANGES.items():
            if k == "n_red_units":
                p[k] = int(self.rng.randint(int(lo), int(hi) + 1))
            else:
                p[k] = float(self.rng.uniform(lo, hi))
        params = ScenarioDomainParams(**p)
        self.history.append(params)
        return params

    def sample_adversarial(
        self, agent_win_rate: float, regret_weight: float = 0.7
    ) -> ScenarioDomainParams:
        """
        PAIRED 방식 적대적 샘플링
        agent_win_rate가 높을수록 → 더 어려운 파라미터 생성
        """
        difficulty = agent_win_rate * regret_weight
        p = {}
        for k, (lo, hi) in self.PARAM_RANGES.items():
            if k in ("n_red_units", "red_aggressiveness"):
                # Red 강화 (난이도 상승)
                val = lo + (hi - lo) * difficulty
            elif k in ("blue_ammo_ratio", "blue_fuel_ratio", "terrain_advantage"):
                # Blue 약화 (난이도 상승)
                val = hi - (hi - lo) * difficulty
            else:
                val = float(self.rng.uniform(lo, hi))

            if k == "n_red_units":
                p[k] = int(round(val))
            else:
                p[k] = float(np.clip(val, lo, hi))

        params = ScenarioDomainParams(**p)
        self.history.append(params)
        return params
```

---

### 3-2. 단기 구현 (P1, 1~2주)

#### 3-2-1. NFSP 레저버 버퍼 + 평균 전략 네트워크

```python
# rl_agent/nfsp_buffer.py (신규)

import numpy as np
from collections import deque
from typing import Optional, Tuple

class ReservoirBuffer:
    """
    저수지 샘플링(Reservoir Sampling) 기반 버퍼
    - 과거 전 에피소드의 (state, action) 쌍을 균등 확률로 유지
    - NFSP 평균 전략 네트워크 학습에 사용
    - 메모리: O(capacity), 샘플링 시간: O(1)
    """

    def __init__(self, capacity: int = 200_000, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.RandomState(seed)
        self.buffer = []
        self._total_seen = 0

    def add(self, state: np.ndarray, action: int):
        """저수지 샘플링으로 추가"""
        self._total_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((state.copy(), action))
        else:
            idx = self.rng.randint(0, self._total_seen)
            if idx < self.capacity:
                self.buffer[idx] = (state.copy(), action)

    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            return None
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        states  = np.stack([self.buffer[i][0] for i in indices])
        actions = np.array([self.buffer[i][1] for i in indices])
        return states, actions

    def __len__(self) -> int:
        return len(self.buffer)
```

#### 3-2-2. α-Rank 기반 LeagueManager 업그레이드

현행 `nash_gap_estimate()`를 실제 α-Rank 계산으로 교체:

```python
# league_selfplay.py에 추가

def compute_alpha_rank(
    self, alpha: float = 1e-2
) -> Tuple[np.ndarray, float]:
    """
    α-Rank 정적 분포 계산

    payoff 행렬 M[i,j] = 에이전트 i vs j 의 i 승률 추정
    """
    agents = list(self.agents.values())
    n = len(agents)
    if n < 2:
        return np.ones(n) / max(n, 1), 0.0

    # payoff 행렬 구성
    M = np.full((n, n), 0.5)
    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i == j:
                continue
            past = [r for oid, r in ai.matchup_history if oid == aj.agent_id]
            if past:
                M[i, j] = float(np.mean(past))

    # Markov Chain 전이 행렬 T
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rho = M[i, j] - M[j, i]
            if abs(rho) < 1e-10:
                T[i, j] = 1.0 / n
            else:
                T[i, j] = (1.0 - np.exp(-alpha * rho)) / (
                    1.0 - np.exp(-alpha * n * rho) + 1e-10
                )

    # 행 정규화
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / (row_sums + 1e-10)
    np.fill_diagonal(T, 1.0 - T.sum(axis=1) + T.diagonal())

    # 정적 분포: 고유값 1에 해당하는 고유벡터
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx_stat = np.argmin(np.abs(eigvals - 1.0))
    stat_dist = np.abs(eigvecs[:, idx_stat].real)
    stat_dist = stat_dist / (stat_dist.sum() + 1e-10)

    # α-Rank 엔트로피 (다양성 지표)
    entropy = float(-np.sum(stat_dist * np.log(stat_dist + 1e-10)))

    return stat_dist, entropy
```

---

### 3-3. 중기 구현 (P2, 1~4주)

#### 3-3-1. HAPPO 순차적 업데이트 (`mappo.py` 업그레이드)

현재 `MAPPOManager.update()`의 동시 업데이트를 HAPPO 방식으로 교체:

```python
# mappo.py update() 메서드 HAPPO 버전

UPDATE_ORDER = [
    "logistics", "signal", "engineer",
    "infantry", "armor", "artillery",
    "aviation", "electronic_warfare"
]

def update_happo(
    self, clip_eps: float = 0.2,
    value_coef: float = 0.5, entropy_coef: float = 0.01,
    n_epochs: int = 4, gamma: float = 0.99,
) -> Dict:
    """
    HAPPO: 이종 에이전트 순차적 업데이트
    중요도 비율 누적으로 단조 개선 보장
    """
    if not TORCH_AVAILABLE or not self.buffers:
        return {}

    # 유닛 유형별 버퍼 분리
    type_buffers: Dict[str, list] = {ut: [] for ut in UPDATE_ORDER}
    for uid, trans_list in self.buffers.items():
        # uid → unit_type 역조회가 필요 (UnitTransition에 unit_type 추가 권장)
        ut = uid.split("_")[0] if "_" in uid else "infantry"
        if ut in type_buffers:
            type_buffers[ut].extend(trans_list)

    cumulative_ratio = None  # 이전 유닛 유형의 중요도 비율 누적
    all_metrics = {"actor_loss": 0.0, "n_updated": 0}

    for unit_type in UPDATE_ORDER:
        trans = type_buffers.get(unit_type, [])
        if not trans:
            continue

        actor = self.type_actors.get(unit_type)
        if not (TORCH_AVAILABLE and hasattr(actor, 'evaluate_actions')):
            continue

        obs_t    = torch.tensor(np.stack([t.observation  for t in trans]), dtype=torch.float32)
        acts_t   = torch.tensor([t.action                for t in trans], dtype=torch.long)
        old_lp_t = torch.tensor([t.log_prob              for t in trans], dtype=torch.float32)
        masks_t  = torch.tensor(np.stack([t.action_mask  for t in trans]), dtype=torch.bool)

        rets_t = self._compute_returns(trans, gamma)
        vals_t = torch.tensor([t.global_value for t in trans], dtype=torch.float32)
        advs_t = (rets_t - vals_t)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        for _ in range(n_epochs):
            new_lp, entropy = actor.evaluate_actions(obs_t, acts_t, masks_t)
            ratio = torch.exp(new_lp - old_lp_t)

            # HAPPO: 이전 유닛 누적 비율 반영
            if cumulative_ratio is not None and len(cumulative_ratio) == len(ratio):
                effective_ratio = ratio * cumulative_ratio.detach()
            else:
                effective_ratio = ratio

            surr1 = effective_ratio * advs_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs_t
            a_loss = (-torch.min(surr1, surr2) - entropy_coef * entropy).mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            self.actor_optimizer.step()

            all_metrics["actor_loss"] += a_loss.item()

        # 다음 유닛을 위해 현재 비율 누적
        with torch.no_grad():
            new_lp_final, _ = actor.evaluate_actions(obs_t, acts_t, masks_t)
            cumulative_ratio = torch.exp(new_lp_final - old_lp_t)

        all_metrics["n_updated"] += 1

    self.clear_buffers()
    return all_metrics
```

---

## 4. 평가 지표 확장

### 4-1. 현행 평가 지표

현재 `evaluation/metrics.py`의 주요 지표:
- `win_rate` — 승률
- `casualty_exchange_ratio` — 사상자 교환 비율
- `mission_success_rate` — 임무 달성률

### 4-2. 적대적 학습 전용 추가 지표

| 지표 | 정의 | 목표값 |
|------|------|--------|
| **Nash Gap** | `max_{i} max_{a} Q_i(a, π_{-i}) - V_i(π)` | < 0.05 |
| **Strategy Diversity** (α-Rank 엔트로피) | `-Σ p_i log p_i` | > 1.5 |
| **Exploitability** | 최선 응답에 대한 평균 승률 | < 0.55 |
| **Robustness Score** | ε-교란 하에서의 성능 저하율 | < 10% |
| **Curriculum Progress** | PAIRED regret 감소율 | > 0 |
| **Population Coverage** | 전략 집합이 커버하는 행동 공간 비율 | > 0.8 |

### 4-3. 적대적 견고성 벤치마크

```python
# evaluation/adversarial_benchmark.py (신규 스켈레톤)

class AdversarialRobustnessBenchmark:
    """
    학습된 Blue 에이전트의 적대적 견고성 평가

    1. 관측 노이즈 견고성: ε-ball 내 교란에도 정책 유지
    2. 전략 다양성: 여러 Red 전략에 대한 평균 성능
    3. 분포 이동 견고성: 훈련과 다른 시나리오 파라미터
    """

    def evaluate_obs_robustness(
        self, agent, scenarios: list, epsilons: list = [0.01, 0.05, 0.1]
    ) -> dict:
        """관측 교란 견고성 평가"""
        results = {}
        for eps in epsilons:
            win_rates = []
            for scen in scenarios:
                # 정상 vs 교란 관측으로 에이전트 평가
                normal_wr = self._evaluate_single(agent, scen, obs_noise=0.0)
                noisy_wr  = self._evaluate_single(agent, scen, obs_noise=eps)
                win_rates.append(normal_wr - noisy_wr)  # 성능 저하
            results[f"eps_{eps}"] = float(np.mean(win_rates))
        return results

    def evaluate_strategy_coverage(
        self, agent, red_strategies: list
    ) -> dict:
        """다양한 Red 전략에 대한 Blue 성능"""
        win_rates = {}
        for red_strat in red_strategies:
            wr = self._evaluate_against(agent, red_strat)
            win_rates[red_strat.strategy_name] = wr
        return {
            "per_strategy": win_rates,
            "mean": float(np.mean(list(win_rates.values()))),
            "min": float(np.min(list(win_rates.values()))),
            "exploitability": 1.0 - float(np.min(list(win_rates.values()))),
        }
```

---

## 5. 장기 연구 방향 (P3, 1~3개월)

### 5-1. Foundation Model 기반 Red 에이전트

최신 트렌드(2024~2025): 대규모 언어 모델(LLM)을 Red 에이전트의 **전략 플래너**로 활용:

```
LLM Planner → 고수준 전략 지시 → RL Worker → 저수준 행동 실행
```

- **FALCON 적용**: `red_agent.py`의 6가지 행동 선택을 LLM 계획기가 상위 목표 설정
- `hitl/natural_language_interface.py`와 연동하여 자연어 전술 명령 처리
- Chain-of-Thought 전술 추론 로깅 → `explainability/auto_aar.py`와 통합

### 5-2. 연합 학습 기반 분산 League (Federated League)

- 다수의 독립적 FALCON 인스턴스가 자체 League를 학습
- **연합 집계(FedAvg)** 방식으로 에이전트 파라미터 공유
- 각 인스턴스가 서로 다른 지역/시나리오 특화 → 더 일반화된 에이전트 생성

### 5-3. 인과 강화학습 (Causal RL for Adversarial Robustness)

- 전장 결과의 **인과 그래프** 추론 (`ontology/` 활용)
- 반사실적 추론(counterfactual)으로 Red 취약점 탐색
- `explainability/counterfactual.py`와 직접 통합 가능

---

## 6. 구현 우선순위 요약

```
[즉시, P0 — 1~2일]
  □ simulator/adversarial_scenario.py — DomainRandomizer 구현
  □ red_agent.py — compute_observation_perturbation() 추가
  □ evaluation/adversarial_benchmark.py 스켈레톤 생성

[단기, P1 — 1~2주]
  □ rl_agent/nfsp_buffer.py — ReservoirBuffer 구현
  □ league_selfplay.py — compute_alpha_rank() 교체
  □ self_play_trainer.py — Phase B에 DomainRandomizer 연동

[중기, P2 — 1~4주]
  □ mappo.py — update_happo() 구현 (HAPPO 순차 업데이트)
  □ league_selfplay.py — PSRO 오라클 통합
  □ simulator/adversarial_scenario.py — PAIRED 루프 구현
  □ rl_agent/nfsp_agent.py — NFSPAgent 클래스 구현

[장기, P3 — 1~3개월]
  □ gnn_model/ — Multi-Agent Transformer 인코더-디코더 통합
  □ rl_agent/ — NeuPL Conditional Policy 구현
  □ evaluation/ — AdversarialRobustnessBenchmark 완성
```

---

## 7. 참고 문헌 (Research References)

| 알고리즘 | 논문 | 연도 | 적용 모듈 |
|----------|------|------|-----------|
| Adversarial Policy | Gleave et al., ICLR 2020 | 2020 | red_agent.py |
| PAIRED | Dennis et al., NeurIPS 2020 | 2020 | self_play_trainer.py |
| REPAIRED | Jiang et al., NeurIPS 2021 | 2021 | self_play_trainer.py |
| PSRO | Lanctot et al., NeurIPS 2017 | 2017 | league_selfplay.py |
| α-Rank | Omidshafiei et al., Sci Rep 2019 | 2019 | league_selfplay.py |
| NFSP | Heinrich & Silver, NIPS 2016 | 2016 | rl_agent/nfsp_buffer.py |
| RARL | Pinto et al., ICML 2017 | 2017 | blue_agent.py |
| SA-PPO | Zhang et al., ICLR 2020 | 2020 | blue_agent.py |
| HAPPO | Zhong et al., ICLR 2022 | 2022 | mappo.py |
| MAT | Wen et al., NeurIPS 2022 | 2022 | gnn_model/ |
| NeuPL | Marris et al., NeurIPS 2022 | 2022 | league_selfplay.py |
| ACCEL | Parker-Holder et al., NeurIPS 2022 | 2022 | self_play_trainer.py |
| FCP | Strouse et al., NeurIPS 2021 | 2021 | league_selfplay.py |

---

*본 보고서는 학술 연구 목적의 FALCON 시뮬레이터 개선을 위해 작성되었습니다.*
