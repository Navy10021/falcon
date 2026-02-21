"""
hierarchical_rl.py
==================
⑨ 계층적 RL — 작전(Operation) ↔ 전술(Tactical) 이중 계층
- 상위 에이전트(Operation): 임무 목표 설정 (1~5 스텝 단위)
- 하위 에이전트(Tactical): 구체적 유닛 행동 (매 스텝)
- 상위 목표가 하위 보상 형성 (reward shaping)
- Manager-Worker 패턴 (Option Framework)

학술 연구용 합성 데이터
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


# ──────────────────────────────────────────────
# 작전 목표 (상위 계층 행동 공간)
# ──────────────────────────────────────────────

class OperationalObjective(Enum):
    """상위 에이전트 행동: 작전 목표"""
    SEIZE_OBJECTIVE    = "seize"       # 목표 지점 점령
    DESTROY_FORCE      = "destroy"     # 적 전력 격멸
    HOLD_POSITION      = "hold"        # 진지 고수
    DELAY_ENEMY        = "delay"       # 적 지연
    ENVELOP_FLANK      = "envelop"     # 포위 기동
    WITHDRAW_PRESERVE  = "withdraw"    # 전력 보존 철수

N_OP_ACTIONS = len(OperationalObjective)
OP_ACTION_NAMES = [o.value for o in OperationalObjective]

# 작전 목표별 전술 행동 우선순위 (MAPPO 행동 인덱스 가중치)
# 0=대기, 1=전진, 2=후퇴, 3=교전, 4=지원, 5=측면, 6=보급, 7=방어
OBJECTIVE_TACTICAL_WEIGHTS: Dict[str, List[float]] = {
    "seize":    [0.0, 2.0, 0.0, 1.5, 1.0, 1.5, 0.0, 0.0],
    "destroy":  [0.0, 1.0, 0.0, 3.0, 0.5, 1.5, 0.0, 0.0],
    "hold":     [1.0, 0.0, 0.0, 1.5, 1.5, 0.0, 1.0, 3.0],
    "delay":    [1.0, 0.0, 1.0, 1.0, 0.5, 1.5, 0.5, 1.5],
    "envelop":  [0.0, 1.0, 0.0, 1.5, 1.0, 3.0, 0.0, 0.5],
    "withdraw": [0.0, 0.0, 3.0, 0.5, 1.0, 1.0, 2.0, 1.0],
}


# ──────────────────────────────────────────────
# 상위 에이전트 (Operational Manager)
# ──────────────────────────────────────────────

@dataclass
class OperationalState:
    """작전 수준 상태 벡터 (20차원)"""
    blue_strength_ratio: float       # Blue 전투력 / 초기 전투력
    red_strength_ratio: float
    blue_headcount_ratio: float
    red_headcount_ratio: float
    mission_progress: float          # 임무 달성도 [0,1]
    time_elapsed_ratio: float        # 경과 시간 / 최대 시간
    avg_blue_ammo: float
    avg_blue_fuel: float
    encirclement_risk: float         # 포위 위험도
    tactical_uncertainty: float
    terrain_advantage: float
    initiative: float                # 전술 주도권
    # 이전 작전 목표 (one-hot 6D)
    prev_objective: List[float] = field(default_factory=lambda: [0.0]*6)
    # 패딩 2D
    pad: List[float] = field(default_factory=lambda: [0.0]*2)

    def to_vector(self) -> np.ndarray:
        base = np.array([
            self.blue_strength_ratio, self.red_strength_ratio,
            self.blue_headcount_ratio, self.red_headcount_ratio,
            self.mission_progress, self.time_elapsed_ratio,
            self.avg_blue_ammo, self.avg_blue_fuel,
            self.encirclement_risk, self.tactical_uncertainty,
            self.terrain_advantage, self.initiative,
        ], dtype=np.float32)
        return np.concatenate([base, self.prev_objective, self.pad])


class NumpyOperationalAgent:
    """
    작전 수준 에이전트 (Manager)
    - 더 느린 주기로 작전 목표 결정 (K 전술 스텝마다)
    - 상태: 전장 전체 요약 (20D)
    - 행동: OperationalObjective (6종)
    """

    STATE_DIM = 20

    def __init__(self, hidden_dim: int = 64, decision_interval: int = 5, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.decision_interval = decision_interval   # K 스텝마다 목표 갱신
        self.step_count = 0
        self.current_objective = OperationalObjective.SEIZE_OBJECTIVE
        self.objective_history: List[OperationalObjective] = []

        # 경량 정책 네트워크
        self.W1 = rng.randn(self.STATE_DIM, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, N_OP_ACTIONS).astype(np.float32) * 0.01
        self.b2 = np.zeros(N_OP_ACTIONS, dtype=np.float32)
        self.temperature = 1.0   # Softmax 온도 (탐색 조절)

    def select_objective(
        self, state: OperationalState, force_update: bool = False
    ) -> OperationalObjective:
        """
        작전 목표 선택 (decision_interval 스텝마다 갱신)
        force_update=True: 즉시 갱신
        """
        self.step_count += 1
        if not force_update and self.step_count % self.decision_interval != 1:
            return self.current_objective

        sv = state.to_vector()
        h = np.maximum(0, sv @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        # Softmax 샘플링
        logits = logits / self.temperature
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        idx = int(np.random.choice(N_OP_ACTIONS, p=probs))

        self.current_objective = list(OperationalObjective)[idx]
        self.objective_history.append(self.current_objective)
        return self.current_objective

    def get_tactical_weights(self) -> np.ndarray:
        """현재 작전 목표 → 전술 행동 가중치 벡터"""
        weights = OBJECTIVE_TACTICAL_WEIGHTS.get(
            self.current_objective.value,
            [1.0] * 8
        )
        w = np.array(weights, dtype=np.float32)
        return w / w.sum()   # 정규화

    def compute_operational_reward(
        self,
        step_result: Dict,
        prev_state: OperationalState,
        curr_state: OperationalState
    ) -> float:
        """작전 수준 보상 계산"""
        obj = self.current_objective

        if obj == OperationalObjective.SEIZE_OBJECTIVE:
            # 임무 진행도 증가 보상
            reward = (curr_state.mission_progress - prev_state.mission_progress) * 10

        elif obj == OperationalObjective.DESTROY_FORCE:
            # 적 전력 감소 보상
            reward = (prev_state.red_strength_ratio - curr_state.red_strength_ratio) * 8

        elif obj == OperationalObjective.HOLD_POSITION:
            # 아군 전력 유지 보상
            reward = (curr_state.blue_strength_ratio - prev_state.blue_strength_ratio) * 5
            reward += 0.1 if curr_state.blue_strength_ratio > 0.6 else 0.0

        elif obj == OperationalObjective.DELAY_ENEMY:
            # 시간 버티기 보상
            reward = curr_state.time_elapsed_ratio * 0.5

        elif obj == OperationalObjective.ENVELOP_FLANK:
            # 포위 성공 시 보상
            reward = curr_state.encirclement_risk * 3  # 적 포위 위험 = 아군 성공

        elif obj == OperationalObjective.WITHDRAW_PRESERVE:
            # 전력 보존 보상
            reward = curr_state.blue_headcount_ratio * 4

        else:
            reward = 0.0

        # 공통 패널티: 아군 과도 손실
        if curr_state.blue_strength_ratio < 0.3:
            reward -= 3.0
        if step_result.get("mission_status") == "blue_win":
            reward += 10.0
        elif step_result.get("mission_status") == "red_win":
            reward -= 10.0

        return float(reward)


# ──────────────────────────────────────────────
# 하위 에이전트 (Tactical Worker)
# ──────────────────────────────────────────────

class NumpyTacticalAgent:
    """
    전술 수준 에이전트 (Worker)
    - 매 스텝 행동 선택 (MAPPO와 동일 구조)
    - 상위 작전 목표를 상태에 포함
    - 작전 목표 가중치로 행동 유도
    """

    STATE_DIM = 32   # 전술 상태 (유닛 지역 관측 24D + 작전 목표 8D)

    def __init__(self, n_actions: int = 8, hidden_dim: int = 64, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_actions = n_actions
        self.W1 = rng.randn(self.STATE_DIM, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, n_actions).astype(np.float32) * 0.01
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    def select_action(
        self,
        local_obs: np.ndarray,           # 24D 지역 관측
        tactical_weights: np.ndarray,    # 8D 작전 목표 가중치
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float]:
        """
        전술 행동 선택
        작전 목표 가중치로 행동 logit 조정
        """
        # 상태: 지역 관측 + 작전 목표 가중치
        state = np.concatenate([local_obs, tactical_weights])
        if len(state) < self.STATE_DIM:
            state = np.pad(state, (0, self.STATE_DIM - len(state)))
        elif len(state) > self.STATE_DIM:
            state = state[:self.STATE_DIM]

        h = np.maximum(0, state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        # 작전 목표 가중치 반영 (logit 조정)
        tw = tactical_weights[:self.n_actions] if len(tactical_weights) >= self.n_actions \
             else np.pad(tactical_weights, (0, self.n_actions - len(tactical_weights)))
        logits = logits + np.log(tw + 1e-8)

        if action_mask is not None:
            logits = np.where(action_mask, logits, -1e9)

        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        return action, float(np.log(probs[action] + 1e-8))

    def compute_intrinsic_reward(
        self,
        tactical_action: int,
        tactical_weights: np.ndarray
    ) -> float:
        """
        내재적 보상: 전술 행동이 작전 목표와 얼마나 일치하는가
        """
        if tactical_action < len(tactical_weights):
            return float(tactical_weights[tactical_action]) * 0.5
        return 0.0


# ──────────────────────────────────────────────
# 계층적 RL 코디네이터
# ──────────────────────────────────────────────

class HierarchicalRLCoordinator:
    """
    Manager(작전) + Worker(전술) 코디네이터
    CTDE: 작전 에이전트는 전체 상태, 전술 에이전트는 지역 관측
    """

    def __init__(self, decision_interval: int = 5, seed: int = 0):
        self.manager = NumpyOperationalAgent(
            decision_interval=decision_interval, seed=seed
        )
        # 유닛 유형별 전술 에이전트 (파라미터 공유)
        from ontology.combat_schema import UnitType
        self.workers: Dict[str, NumpyTacticalAgent] = {
            ut.value: NumpyTacticalAgent(seed=seed + i)
            for i, ut in enumerate(UnitType)
        }
        self.total_steps = 0
        self.episode_objectives: List[str] = []
        self.op_rewards: List[float] = []
        self.tac_rewards: List[float] = []

    def build_operational_state(self, kg, step: int, max_steps: int) -> OperationalState:
        """Knowledge Graph → 작전 수준 상태"""
        from ontology.combat_schema import ForceAlignment, UnitStatus

        blue_units = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.BLUE]
        red_units  = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.RED]

        blue_active = [u for u in blue_units if u.status != UnitStatus.DESTROYED]
        red_active  = [u for u in red_units  if u.status != UnitStatus.DESTROYED]

        init_blue_cp = sum(u.combat_power for u in blue_units) + 1e-8
        curr_blue_cp = sum(u.combat_power for u in blue_active)
        init_red_cp  = sum(u.combat_power for u in red_units) + 1e-8
        curr_red_cp  = sum(u.combat_power for u in red_active)

        init_blue_hc = sum(u.headcount for u in blue_units) + 1
        curr_blue_hc = sum(u.headcount for u in blue_active)
        init_red_hc  = sum(u.headcount for u in red_units) + 1
        curr_red_hc  = sum(u.headcount for u in red_active)

        avg_ammo = float(np.mean([u.capability.ammo_level for u in blue_active])) if blue_active else 0.0
        avg_fuel = float(np.mean([u.capability.fuel_level  for u in blue_active])) if blue_active else 0.0

        # 작전 목표 one-hot
        prev_obj_idx = OP_ACTION_NAMES.index(self.manager.current_objective.value) \
                       if self.manager.current_objective else 0
        prev_obj_oh = [0.0] * N_OP_ACTIONS
        prev_obj_oh[prev_obj_idx] = 1.0

        return OperationalState(
            blue_strength_ratio=float(curr_blue_cp / init_blue_cp),
            red_strength_ratio=float(curr_red_cp / init_red_cp),
            blue_headcount_ratio=float(curr_blue_hc / init_blue_hc),
            red_headcount_ratio=float(curr_red_hc / init_red_hc),
            mission_progress=float(1 - curr_red_cp / init_red_cp),
            time_elapsed_ratio=float(step / max(max_steps, 1)),
            avg_blue_ammo=avg_ammo,
            avg_blue_fuel=avg_fuel,
            encirclement_risk=float(np.random.uniform(0.1, 0.5)),
            tactical_uncertainty=float(np.random.uniform(0.2, 0.6)),
            terrain_advantage=float(np.random.uniform(0.3, 0.7)),
            initiative=float(curr_blue_cp / max(curr_red_cp, 1e-8) / 3),
            prev_objective=prev_obj_oh,
        )

    def step(self, kg, step: int, max_steps: int) -> Dict:
        """
        계층적 RL 단일 스텝

        1. Manager: 작전 목표 갱신 (K스텝마다)
        2. Workers: 각 유닛 전술 행동 선택
        3. 환경 실행
        4. 보상 분리 계산
        """
        from rl_agent.mappo import MAPPOManager, ACTION_NAMES
        from ontology.combat_schema import ForceAlignment, UnitStatus
        from simulator.mixed_lanchester import MixedLanchesterEngine

        # 1. 작전 목표 결정
        op_state = self.build_operational_state(kg, step, max_steps)
        objective = self.manager.select_objective(op_state)
        tactical_weights = self.manager.get_tactical_weights()

        # 2. 전술 행동 선택 (각 유닛)
        unit_actions = {}
        for uid, unit in kg.units.items():
            if unit.alignment != ForceAlignment.BLUE or unit.status == UnitStatus.DESTROYED:
                continue
            worker = self.workers.get(unit.unit_type.value,
                                       list(self.workers.values())[0])
            # 간단한 지역 관측 (24D padding)
            local_obs = np.random.randn(24).astype(np.float32) * 0.1
            from rl_agent.mappo import UNIT_ACTION_MASKS, N_ACTIONS
            allowed = UNIT_ACTION_MASKS.get(unit.unit_type.value, list(range(N_ACTIONS)))
            mask = np.zeros(N_ACTIONS, dtype=bool)
            mask[allowed] = True

            action, lp = worker.select_action(local_obs, tactical_weights, mask)
            intr_reward = worker.compute_intrinsic_reward(action, tactical_weights)
            unit_actions[uid] = {
                "action": action,
                "log_prob": lp,
                "action_name": ACTION_NAMES[action],
                "intrinsic_reward": intr_reward,
            }

        self.total_steps += 1
        self.episode_objectives.append(objective.value)

        return {
            "operational_objective": objective.value,
            "tactical_weights": tactical_weights.tolist(),
            "unit_actions": unit_actions,
            "n_blue_acting": len(unit_actions),
        }

    def get_stats(self) -> Dict:
        """학습 통계"""
        obj_counts: Dict[str, int] = {}
        for obj in self.episode_objectives:
            obj_counts[obj] = obj_counts.get(obj, 0) + 1
        return {
            "total_steps": self.total_steps,
            "objective_distribution": obj_counts,
            "decision_interval": self.manager.decision_interval,
            "n_objective_changes": len(self.manager.objective_history),
        }


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory

    print("=" * 55)
    print("⑨ 계층적 RL (작전-전술 계층) 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    coord = HierarchicalRLCoordinator(decision_interval=3, seed=42)

    print("\n[15 스텝 계층적 RL 실행]")
    prev_obj = None
    for step in range(15):
        result = coord.step(kg, step, max_steps=20)
        obj = result["operational_objective"]
        changed = "🔄" if obj != prev_obj else "  "
        print(f"  스텝 {step+1:2d}: {changed} 작전목표=[{obj:15s}] "
              f"전술유닛={result['n_blue_acting']}개")
        prev_obj = obj

    stats = coord.get_stats()
    print(f"\n[통계]")
    print(f"  총 스텝: {stats['total_steps']}")
    print(f"  목표 변경 횟수: {stats['n_objective_changes']}")
    print(f"  목표 분포: {stats['objective_distribution']}")
    print("\n✅ 계층적 RL 정상 동작!")
