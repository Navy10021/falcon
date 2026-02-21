"""
mappo.py
========
⑥ 유닛별 분산 행동 공간 — Multi-Agent PPO (MAPPO)
- 각 유닛이 독립적으로 행동 선택
- 공유 Critic (Centralized Critic) + 분산 Actor
- CTDE (Centralized Training, Decentralized Execution)
- 유닛 유형별 특화 행동 마스크

아키텍처:
  각 유닛 i: 지역 관측 o_i → Actor_i → 행동 a_i
  Centralized:  전역 상태 S → Critic → V(S)
  
학술 연구용 합성 데이터
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────
# 유닛별 행동 공간
# ──────────────────────────────────────────────

UNIT_ACTION_MASKS = {
    # 유닛 유형: 허용 행동 인덱스 리스트
    # 행동: 0=대기, 1=전진, 2=후퇴, 3=교전, 4=지원, 5=측면, 6=보급요청, 7=방어진지
    "infantry":           [0, 1, 2, 3, 4, 5, 7],
    "armor":              [0, 1, 2, 3, 5],
    "artillery":          [0, 3, 7],             # 포병은 이동 제한
    "aviation":           [0, 1, 3, 5],
    "engineer":           [0, 1, 2, 7],
    "signal":             [0, 7],                 # 통신은 지원만
    "logistics":          [0, 1, 2, 6],           # 병참은 보급 특화
    "electronic_warfare": [0, 3, 7],
}

N_ACTIONS = 8
ACTION_NAMES = ["대기", "전진", "후퇴", "교전", "지원", "측면기동", "보급요청", "방어진지구축"]

UNIT_OBS_DIM = 24   # 유닛 지역 관측 차원
GLOBAL_STATE_DIM = 64   # 전역 상태 차원 (Centralized Critic)


# ──────────────────────────────────────────────
# Numpy 기반 경량 Actor (단일 유닛)
# ──────────────────────────────────────────────

class NumpyUnitActor:
    """
    단일 유닛용 경량 Actor (numpy 구현)
    지역 관측 → 행동 확률 분포
    """

    def __init__(self, obs_dim: int = UNIT_OBS_DIM,
                 n_actions: int = N_ACTIONS,
                 hidden_dim: int = 64, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_actions = n_actions
        self.W1 = rng.randn(obs_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, n_actions).astype(np.float32) * 0.01
        self.b2 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """관측 → 행동 로짓"""
        h = np.maximum(0, obs @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        if action_mask is not None:
            # 마스크된 행동 → -inf
            logits = np.where(action_mask, logits, -1e9)
        return logits

    def sample_action(
        self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float]:
        """확률적 행동 샘플링 → (action, log_prob)"""
        logits = self.forward(obs, action_mask)
        logits -= logits.max()   # 수치 안정성
        probs = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
        action = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob

    def greedy_action(
        self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> int:
        logits = self.forward(obs, action_mask)
        if action_mask is not None:
            logits = np.where(action_mask, logits, -1e9)
        return int(np.argmax(logits))


class NumpyCentralizedCritic:
    """
    중앙화된 Critic (전역 상태 → 가치 함수)
    CTDE 패러다임: 훈련 시 전역 상태 접근, 실행 시 지역 관측만 사용
    """

    def __init__(self, state_dim: int = GLOBAL_STATE_DIM,
                 hidden_dim: int = 128, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, 64).astype(np.float32) * 0.1
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = rng.randn(64, 1).astype(np.float32) * 0.01
        self.b3 = np.zeros(1, dtype=np.float32)

    def value(self, global_state: np.ndarray) -> float:
        h = np.maximum(0, global_state @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        v = float((h @ self.W3 + self.b3).squeeze())
        return v


# ──────────────────────────────────────────────
# MAPPO 에이전트 (다중 유닛 관리)
# ──────────────────────────────────────────────

@dataclass
class UnitObservation:
    """유닛 지역 관측"""
    unit_id: str
    unit_type: str
    obs_vector: np.ndarray       # [UNIT_OBS_DIM]
    action_mask: np.ndarray      # [N_ACTIONS] bool


@dataclass
class UnitTransition:
    """단일 유닛 전환 기록 (버퍼용)"""
    unit_id: str
    observation: np.ndarray
    action: int
    log_prob: float
    reward: float
    global_value: float
    done: bool
    action_mask: np.ndarray


class MAPPOManager:
    """
    MAPPO 다중 에이전트 관리자
    - 각 유닛 유형별 Actor 공유 (파라미터 공유)
    - 중앙화된 단일 Critic
    - 유닛별 행동 마스크 적용
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        # 유닛 유형별 Actor (파라미터 공유)
        self.type_actors: Dict[str, NumpyUnitActor] = {
            unit_type: NumpyUnitActor(seed=seed + i)
            for i, unit_type in enumerate(UNIT_ACTION_MASKS.keys())
        }
        self.critic = NumpyCentralizedCritic(seed=seed + 100)
        self.buffers: Dict[str, List[UnitTransition]] = {}
        self.episode_count = 0

    def build_unit_observation(self, unit, kg, uncertainty_map=None) -> UnitObservation:
        """
        단일 유닛의 지역 관측 벡터 구성 (24차원)
        """
        from ontology.combat_schema import ForceAlignment, UnitStatus

        cap = unit.capability
        enemies = [u for u in kg.units.values()
                   if u.alignment != unit.alignment
                   and u.status != UnitStatus.DESTROYED]
        friends = [u for u in kg.units.values()
                   if u.alignment == unit.alignment
                   and u.status != UnitStatus.DESTROYED
                   and u.unit_id != unit.unit_id]

        # 가장 가까운 적
        if enemies:
            dists = [unit.position.distance_to(e.position) for e in enemies]
            nearest_enemy = enemies[np.argmin(dists)]
            min_dist = min(dists)
            nearest_enemy_cp = nearest_enemy.combat_power
        else:
            min_dist = 99.0
            nearest_enemy_cp = 0.0

        unc = uncertainty_map.get(unit.unit_id, 0.5) if uncertainty_map else 0.5

        obs = np.array([
            # 자신 상태 (8D)
            cap.firepower, cap.mobility, cap.protection,
            cap.ammo_level, cap.fuel_level, cap.comms_quality,
            unit.morale, unit.experience,

            # 병력 현황 (4D)
            unit.headcount / 200.0,
            unit.combat_power,
            float(len(friends)) / 10.0,
            float(len(enemies)) / 10.0,

            # 근거리 적 정보 (4D)
            min_dist / 30.0,
            nearest_enemy_cp,
            float(len([e for e in enemies if unit.position.distance_to(e.position) <= 5.0])),
            cap.range_km / 50.0,

            # 불확실성 (2D)
            unc,
            float(unit.status.value == "active"),

            # 위치 정규화 (2D)
            unit.position.x / 30.0,
            unit.position.y / 30.0,

            # 아군 지원 (2D)
            float(len([f for f in friends
                       if unit.position.distance_to(f.position) <= 5.0])),
            float(any(f.unit_type.value == "logistics" for f in friends)),

            # 패딩 (2D)
            0.0, 0.0,
        ], dtype=np.float32)[:UNIT_OBS_DIM]

        # 행동 마스크
        allowed = UNIT_ACTION_MASKS.get(unit.unit_type.value, list(range(N_ACTIONS)))
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[allowed] = True

        # 연료 고갈 시 이동 불가
        if cap.fuel_level < 0.05:
            mask[1] = False  # 전진 불가
            mask[5] = False  # 측면기동 불가

        # 탄약 고갈 시 교전 불가
        if cap.ammo_level < 0.05:
            mask[3] = False  # 교전 불가

        return UnitObservation(unit.unit_id, unit.unit_type.value, obs, mask)

    def select_actions(
        self, kg, uncertainty_map=None, deterministic: bool = False
    ) -> Dict[str, Tuple[int, float, str]]:
        """
        전체 유닛의 행동 동시 선택
        Returns: {unit_id: (action, log_prob, action_name)}
        """
        from ontology.combat_schema import ForceAlignment, UnitStatus

        actions = {}
        for uid, unit in kg.units.items():
            if (unit.alignment != ForceAlignment.BLUE
                    or unit.status == UnitStatus.DESTROYED):
                continue

            unit_obs = self.build_unit_observation(unit, kg, uncertainty_map)
            actor = self.type_actors.get(unit.unit_type.value,
                                          list(self.type_actors.values())[0])

            if deterministic:
                action = actor.greedy_action(unit_obs.obs_vector, unit_obs.action_mask)
                log_prob = 0.0
            else:
                action, log_prob = actor.sample_action(
                    unit_obs.obs_vector, unit_obs.action_mask
                )

            actions[uid] = (action, log_prob, ACTION_NAMES[action])

        return actions

    def compute_joint_reward(
        self,
        step_result: Dict,
        actions: Dict[str, Tuple[int, float, str]],
        force_before: int,
        force_after: int
    ) -> Dict[str, float]:
        """
        유닛별 개별 보상 계산
        - 팀 승리/패배 공유 보상
        - 개인 기여도 보상 (적 사상자 기여)
        - 보급 요청 성공 보상
        """
        rewards = {}
        mission_reward = 0.0
        if step_result.get("mission_status") == "blue_win":
            mission_reward = 5.0
        elif step_result.get("mission_status") == "red_win":
            mission_reward = -5.0

        blue_cas = step_result.get("blue_casualties", 0)
        casualty_penalty = blue_cas * 0.05

        for uid, (action, lp, action_name) in actions.items():
            reward = mission_reward - casualty_penalty

            # 보급 요청 보너스
            if action == 6:    # 보급요청
                reward += 0.5

            # 방어 진지 구축 보너스 (불확실성 높을 때)
            if action == 7:    # 방어진지
                reward += 0.3

            rewards[uid] = float(reward)

        return rewards

    def store_transitions(
        self,
        observations: Dict[str, UnitObservation],
        actions: Dict[str, Tuple[int, float, str]],
        rewards: Dict[str, float],
        global_value: float,
        dones: Dict[str, bool]
    ):
        """경험 버퍼에 전환 저장"""
        for uid in actions:
            if uid not in observations:
                continue
            obs = observations[uid]
            action, log_prob, _ = actions[uid]
            reward = rewards.get(uid, 0.0)
            done = dones.get(uid, False)

            if uid not in self.buffers:
                self.buffers[uid] = []
            self.buffers[uid].append(UnitTransition(
                unit_id=uid,
                observation=obs.obs_vector,
                action=action,
                log_prob=log_prob,
                reward=reward,
                global_value=global_value,
                done=done,
                action_mask=obs.action_mask,
            ))

    def get_action_distribution_stats(self, kg) -> Dict[str, Dict]:
        """유닛별 행동 분포 통계"""
        from ontology.combat_schema import ForceAlignment, UnitStatus
        stats = {}
        for uid, unit in kg.units.items():
            if unit.alignment != ForceAlignment.BLUE or unit.status == UnitStatus.DESTROYED:
                continue
            allowed = UNIT_ACTION_MASKS.get(unit.unit_type.value, list(range(N_ACTIONS)))
            stats[uid] = {
                "unit_type": unit.unit_type.value,
                "n_allowed_actions": len(allowed),
                "allowed_actions": [ACTION_NAMES[a] for a in allowed],
            }
        return stats

    def clear_buffers(self):
        self.buffers.clear()


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory, ForceAlignment

    print("=" * 55)
    print("⑥ Multi-Agent PPO (MAPPO) 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=42)
    manager = MAPPOManager(seed=42)

    # 행동 마스크 확인
    print("\n[유닛 유형별 허용 행동]")
    for unit_type, allowed in UNIT_ACTION_MASKS.items():
        action_names = [ACTION_NAMES[a] for a in allowed]
        print(f"  {unit_type:20s}: {action_names}")

    # 전체 유닛 행동 선택
    print("\n[유닛별 행동 선택]")
    actions = manager.select_actions(kg)
    for uid, (action, lp, name) in list(actions.items())[:4]:
        unit = kg.units[uid]
        print(f"  {uid:8s} ({unit.unit_type.value:15s}) → {name:15s} (lp={lp:.3f})")

    # 행동 분포 통계
    dist_stats = manager.get_action_distribution_stats(kg)
    total_action_space = sum(s["n_allowed_actions"] for s in dist_stats.values())
    print(f"\n총 유닛 수: {len(actions)}")
    print(f"평균 허용 행동 수: {total_action_space/max(len(dist_stats),1):.1f}")
    print(f"전체 결합 행동 공간 크기: {N_ACTIONS}^{len(actions)} = 매우 큼 → 분산으로 해결")

    print("\n✅ MAPPO 정상 동작!")
