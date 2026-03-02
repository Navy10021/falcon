"""
self_play_trainer.py
====================
Phase 2: Blue vs Red Self-Play 학습 루프
- 순차적 업데이트 (Nash Equilibrium 수렴 유도)
- Population-based 학습 (과적합 방지)
- Nash Gap 계산 및 수렴 판단
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import os
import copy
from dataclasses import dataclass, field
from tqdm import tqdm
import torch

from ontology.combat_schema import ScenarioFactory, ForceAlignment
from simulator.lanchester_engine import LanchesterEngine
from simulator.maneuver_engine import ManeuverEngine          # N-2: 공간 기동 엔진
from simulator.fog_of_war import FogOfWarFilter, CurriculumScheduler
from rl_agent.blue_agent import BlueAgent, build_state_vector, BlueActionSpace, PPOConfig
from rl_agent.red_agent import RedAgent, RedActionSpace


@dataclass
class SelfPlayConfig:
    """Self-Play 학습 설정"""
    total_episodes: int = 5000
    max_steps_per_episode: int = 50
    update_interval: int = 10       # N 에피소드마다 업데이트
    red_update_interval: int = 10   # Red 업데이트 주기
    population_size: int = 4        # Population 크기
    nash_check_interval: int = 100
    log_interval: int = 50
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    timeout_result_mode: str = "headcount"   # headcount | draw
    map_size: float = 30.0           # N-2: ManeuverEngine 맵 크기
    gnn_checkpoint_path: Optional[str] = None   # N-4: Phase 1 GNN 체크포인트

    # Self-Play 단계
    phase_a_end: float = 0.2   # Phase A: Red 고정
    phase_b_end: float = 0.4   # Phase B: Blue 고정
    phase_c_end: float = 0.8   # Phase C: 교대 업데이트
    # phase_d: Population-based


@dataclass
class EpisodeStats:
    """에피소드 통계"""
    episode: int
    winner: str
    blue_casualties: int
    red_casualties: int
    blue_final_headcount: int
    red_final_headcount: int
    n_steps: int
    blue_force_reduction: float    # 초기 대비 병력 절감율
    nash_gap: float = 0.0
    phase: str = "A"


@dataclass
class PopulationMember:
    """
    League Self-Play PFSP 용 개체군 멤버.
    에이전트 + 최근 매치 결과를 추적해 PFSP 선택 가중치를 계산한다.
    """
    agent: object          # BlueAgent | RedAgent
    wins: int   = 0        # 이 멤버를 상대로 현재 주 에이전트가 이긴 횟수
    losses: int = 0        # 진 횟수
    draws: int  = 0        # 무승부

    @property
    def n_matches(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate_vs_main(self) -> float:
        """주 에이전트의 이 멤버에 대한 승률 (승률이 낮을수록 강적)"""
        n = self.n_matches
        return self.wins / n if n > 0 else 0.5

    def pfsp_weight(self, alpha: float = 2.0) -> float:
        """
        PFSP 선택 가중치 (AlphaStar 방식).
        win_rate ~ 0.5 (아슬아슬한 상대)에 가중치 최대.
        f(p) = p^alpha * (1-p)^alpha  (beta 분포 밀도 비례)
        alpha=2 → 승률 0.5에서 가중치 최대, 극단(0/1)에서 0
        """
        p = float(np.clip(self.win_rate_vs_main, 0.01, 0.99))
        return (p ** alpha) * ((1 - p) ** alpha)


class SelfPlayTrainer:
    """
    Blue vs Red Self-Play 학습기
    
    Phase A (0~20%): Red 고정 (룰 기반), Blue만 학습
    Phase B (20~40%): Blue 고정, Red만 학습
    Phase C (40~80%): 교대 업데이트 (10 epoch 간격)
    Phase D (80~100%): Population-based 학습
    """

    def __init__(self, config: Optional[SelfPlayConfig] = None):
        self.config = config or SelfPlayConfig()
        self.engine = LanchesterEngine(seed=42)
        self.curriculum = CurriculumScheduler()

        # N-2: ManeuverEngine — Phase 1과 동일한 공간 환경 제공
        self.maneuver_engine = ManeuverEngine(
            map_size=int(self.config.map_size), seed=42
        )

        # N-4: Phase 1 GNN 체크포인트 로드 (상태 벡터 품질 향상)
        self.gnn = None
        if (self.config.gnn_checkpoint_path
                and os.path.isfile(self.config.gnn_checkpoint_path)):
            from gnn_model.bayesian_hgt import BayesianHGT
            self.gnn = BayesianHGT(
                node_in_dim=128, hidden_dim=128, n_layers=2, mc_samples=10
            )
            self.gnn.load_state_dict(
                torch.load(self.config.gnn_checkpoint_path, map_location="cpu", weights_only=True)
            )
            self.gnn.eval()

        # 에이전트 초기화
        self.blue_agent = BlueAgent()
        self.red_agent  = RedAgent()

        # PFSP Population (Phase D: League Self-Play)
        self.blue_population: List[PopulationMember] = [
            PopulationMember(agent=BlueAgent()) for _ in range(self.config.population_size)
        ]
        self.red_population: List[PopulationMember] = [
            PopulationMember(agent=RedAgent()) for _ in range(self.config.population_size)
        ]

        # 통계 추적
        self.stats: List[EpisodeStats] = []
        self.nash_gaps: List[float] = []
        self.win_rates: Dict[str, List[float]] = {"blue": [], "red": [], "draw": []}

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    @staticmethod
    def _build_blue_maneuver_targets(kg, blue_action: int, map_size: float = 30.0) -> Dict:
        """
        N-2: Blue 행동 유형 → ManeuverEngine 이동 목표 좌표 딕셔너리 생성.

        train.py._build_blue_maneuver_targets()와 동일한 로직 적용.
        Phase 1과 Phase 2의 공간 환경 일관성(distribution match)을 보장한다.
        """
        from ontology.combat_schema import UnitStatus

        blue_units = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.BLUE
                      and u.status != UnitStatus.DESTROYED]
        red_units  = [u for u in kg.units.values()
                      if u.alignment == ForceAlignment.RED
                      and u.status != UnitStatus.DESTROYED]
        if not blue_units:
            return {}

        if red_units:
            rcx = float(np.mean([u.position.x for u in red_units]))
            rcy = float(np.mean([u.position.y for u in red_units]))
        else:
            rcx, rcy = map_size * 0.6, map_size * 0.5

        bcx = float(np.mean([u.position.x for u in blue_units]))
        bcy = float(np.mean([u.position.y for u in blue_units]))

        targets = {}
        action = int(blue_action)

        for u in blue_units:
            if action == BlueActionSpace.ADVANCE:
                tx, ty = rcx, rcy
            elif action == BlueActionSpace.FLANK:
                if red_units:
                    closest = min(red_units, key=lambda r: u.position.distance_to(r.position))
                    dx = u.position.x - closest.position.x
                    dy = u.position.y - closest.position.y
                    norm = max(math.sqrt(dx**2 + dy**2), 0.1)
                    tx = float(np.clip(closest.position.x + (-dy / norm) * 5.0, 0, map_size - 1))
                    ty = float(np.clip(closest.position.y + ( dx / norm) * 5.0, 0, map_size - 1))
                else:
                    tx, ty = rcx, rcy
            elif action == BlueActionSpace.WITHDRAW:
                dx = bcx - rcx
                dy = bcy - rcy
                norm = max(math.sqrt(dx**2 + dy**2), 0.1)
                tx = float(np.clip(u.position.x + (dx / norm) * 5.0, 0, map_size - 1))
                ty = float(np.clip(u.position.y + (dy / norm) * 5.0, 0, map_size - 1))
            elif action == BlueActionSpace.SUPPORT:
                friendly = [f for f in blue_units if f.unit_id != u.unit_id]
                if friendly:
                    nearest = min(friendly, key=lambda f: u.position.distance_to(f.position))
                    tx, ty = nearest.position.x, nearest.position.y
                else:
                    tx, ty = u.position.x, u.position.y
            else:
                tx = min(u.position.x + 1.0, map_size - 1)
                ty = u.position.y

            targets[u.unit_id] = (tx, ty)

        return targets

    @staticmethod
    def _pfsp_select(population: "List[PopulationMember]") -> "PopulationMember":
        """
        Prioritized Fictitious Self-Play (PFSP) 가중 샘플링.
        win_rate~0.5 인 상대(아슬아슬한 매치)를 더 자주 선택 → 전략 다양성 극대화.
        모든 가중치가 0이면 균등 샘플링으로 폴백.
        """
        weights = np.array([m.pfsp_weight() for m in population], dtype=np.float64)
        total = weights.sum()
        if total < 1e-9:                        # 초기(매치 없음) → 균등 선택
            return np.random.choice(population)  # type: ignore[arg-type]
        probs = weights / total
        idx = np.random.choice(len(population), p=probs)
        return population[idx]

    def _get_current_phase(self, progress: float) -> str:
        cfg = self.config
        if progress < cfg.phase_a_end:   return "A"
        if progress < cfg.phase_b_end:   return "B"
        if progress < cfg.phase_c_end:   return "C"
        return "D"

    def _rule_based_red_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Phase A용 룰 기반 Red 행동 (항상 FORTIFY + AMBUSH 반복)"""
        action = int(np.random.choice([RedActionSpace.AMBUSH, RedActionSpace.FORTIFY]))
        return action, 0.0, 0.0

    def _active_units(self, kg, alignment):
        from ontology.combat_schema import UnitStatus
        return [u for u in kg.units.values() if u.alignment == alignment and u.status != UnitStatus.DESTROYED and u.headcount > 0]

    def _build_action_pairs(self, kg, blue_action: int, red_action: int):
        """간단한 행동-교전 매핑: 선택 행동이 표적 우선순위에 반영되도록 구성"""
        blue_units = self._active_units(kg, ForceAlignment.BLUE)
        red_units = self._active_units(kg, ForceAlignment.RED)
        if not blue_units or not red_units:
            return None

        def pick_targets(units, mode: str):
            if mode == "weakest":
                return sorted(units, key=lambda u: u.headcount)
            if mode == "strongest":
                return sorted(units, key=lambda u: u.combat_power, reverse=True)
            return sorted(units, key=lambda u: u.headcount, reverse=True)

        blue_mode = {
            BlueActionSpace.ADVANCE: "weakest",
            BlueActionSpace.FLANK: "weakest",
            BlueActionSpace.REINFORCE: "strongest",
            BlueActionSpace.SUPPORT: "strongest",
            BlueActionSpace.REALLOCATE: "balanced",
            BlueActionSpace.WITHDRAW: "balanced",
        }.get(int(blue_action), "balanced")

        red_mode = {
            RedActionSpace.AMBUSH: "weakest",
            RedActionSpace.FORTIFY: "strongest",
            RedActionSpace.DECEPTION: "strongest",
        }.get(int(red_action), "balanced")

        blue_targets = pick_targets(red_units, blue_mode)
        red_targets = pick_targets(blue_units, red_mode)

        pairs = []
        blue_attackers = blue_units
        if int(blue_action) == BlueActionSpace.WITHDRAW:
            blue_attackers = blue_units[:max(1, len(blue_units)//2)]

        for i, attacker in enumerate(blue_attackers):
            target = blue_targets[i % len(blue_targets)]
            pairs.append((attacker.unit_id, target.unit_id))

        red_attackers = red_units
        if int(red_action) == RedActionSpace.FORTIFY:
            red_attackers = red_units[:max(1, len(red_units)//2)]

        for i, attacker in enumerate(red_attackers):
            target = red_targets[i % len(red_targets)]
            pairs.append((attacker.unit_id, target.unit_id))

        return pairs

    def run_episode(
        self,
        episode_idx: int,
        progress: float,
        blue_agent: BlueAgent,
        red_agent: RedAgent,
        phase: str
    ) -> EpisodeStats:
        """단일 에피소드 실행"""
        cfg = self.config

        # 시나리오 생성
        fog_level, _ = self.curriculum.update(progress)
        kg = ScenarioFactory.create_standard_scenario(
            n_blue=np.random.randint(6, 12),
            n_red=np.random.randint(4, 10),
            seed=episode_idx
        )
        fog_filter = FogOfWarFilter(fog_level, seed=episode_idx)
        initial_blue_hc = sum(u.headcount for u in kg.units.values()
                              if u.alignment == ForceAlignment.BLUE)
        prev_blue_hc = initial_blue_hc

        blue_total_cas = 0
        red_total_cas  = 0
        done = False
        step_result = None

        for step_t in range(cfg.max_steps_per_episode):
            if done:
                break

            # Blue 관측
            blue_obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)
            avg_uncertainty = float(np.mean(list(uncertainty_map.values()))) if uncertainty_map else 0.0

            # N-4: GNN 불확실성 확장 벡터 — Phase 1과 동일한 상태 공간 유지
            gnn_ext = None
            if self.gnn is not None:
                from gnn_model.bayesian_hgt import prepare_graph_tensors
                x, adj = prepare_graph_tensors(blue_obs_kg)
                with torch.no_grad():
                    gnn_out = self.gnn.predict_with_uncertainty(x, adj)
                _ext = np.array([
                    gnn_out["casualty_mean"].item(), gnn_out["casualty_std"].item(),
                    gnn_out["risk_mean"].item(),      gnn_out["risk_std"].item(),
                    gnn_out["epistemic_uncertainty"].item(),
                    gnn_out["aleatoric_uncertainty"].item(),
                ], dtype=np.float32)
                gnn_ext = np.clip(
                    np.nan_to_num(_ext, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0
                )

            blue_state = build_state_vector(
                blue_obs_kg, gnn_extension=gnn_ext, uncertainty_map=uncertainty_map
            )

            # Red 관측 (반대 방향)
            red_obs_kg, _ = fog_filter.observe(kg, ForceAlignment.RED)
            red_state = build_state_vector(red_obs_kg)

            # 행동 선택
            blue_action, blue_lp, blue_val = blue_agent.select_action(blue_state)
            if phase == "A":
                red_action, red_lp, red_val = self._rule_based_red_action(red_state)
            else:
                red_action, red_lp, red_val = red_agent.select_action(red_state)

            # 기만 전술 적용
            if red_action == RedActionSpace.DECEPTION:
                red_agent.apply_deception(fog_filter)

            # N-2: ManeuverEngine — Blue 행동 유형에 따라 유닛 위치 갱신
            blue_maneuver_targets = self._build_blue_maneuver_targets(
                kg, blue_action, self.config.map_size
            )
            self.maneuver_engine.run_maneuver_step(
                kg, blue_targets=blue_maneuver_targets
            )

            # 시뮬레이션 스텝 (행동 기반 교전 쌍 반영)
            action_pairs = self._build_action_pairs(kg, blue_action, red_action)
            step_result = self.engine.run_step(kg, action_pairs=action_pairs)
            blue_total_cas += step_result.blue_total_casualties
            red_total_cas  += step_result.red_total_casualties

            # N-2: 교전 후 GNN 노드 특성 동기화
            kg.update_node_features()

            done = (step_result.mission_status != "ongoing")

            # Blue 버퍼 추가
            blue_reward = blue_agent.compute_reward(
                step_result, prev_blue_hc,
                step_result.blue_total_headcount,
                avg_uncertainty
            )
            prev_blue_hc = step_result.blue_total_headcount
            blue_agent.buffer.add(
                blue_state, blue_action, blue_reward,
                blue_lp, blue_val, done, avg_uncertainty
            )

            # Red 버퍼 추가 (Phase B, C, D)
            if phase != "A":
                red_reward = red_agent.compute_reward(
                    step_result,
                    step_result.blue_total_casualties,
                    step_result.red_total_casualties,
                    float(step_result.red_total_casualties),
                    red_action
                )
                red_agent.buffer.add(
                    red_state, red_action, red_reward,
                    red_lp, red_val, done
                )

        # 병력 절감율
        final_blue_hc = sum(u.headcount for u in kg.units.values()
                            if u.alignment == ForceAlignment.BLUE)
        force_reduction = (initial_blue_hc - final_blue_hc) / max(initial_blue_hc, 1)

        winner = step_result.mission_status if step_result is not None else "draw"
        if winner == "ongoing" and cfg.timeout_result_mode == "headcount":
            blue_hc = final_blue_hc
            red_hc = sum(u.headcount for u in kg.units.values() if u.alignment == ForceAlignment.RED)
            if blue_hc > red_hc * 1.05:
                winner = "blue_win"
            elif red_hc > blue_hc * 1.05:
                winner = "red_win"
            else:
                winner = "draw"

        return EpisodeStats(
            episode=episode_idx,
            winner=winner,
            blue_casualties=blue_total_cas,
            red_casualties=red_total_cas,
            blue_final_headcount=final_blue_hc,
            red_final_headcount=sum(u.headcount for u in kg.units.values()
                                    if u.alignment == ForceAlignment.RED),
            n_steps=step_t + 1,
            blue_force_reduction=force_reduction,
            phase=phase
        )

    def compute_nash_gap(self, n_eval: int = 20) -> float:
        """
        Nash Gap 계산 (수렴 지표)
        Nash Gap < 0.05면 수렴 판단
        
        두 에이전트의 최선 응답 대비 현재 전략의 이탈 정도
        """
        blue_rewards, red_rewards = [], []

        for i in range(n_eval):
            kg = ScenarioFactory.create_standard_scenario(seed=i + 9999)
            fog_filter = FogOfWarFilter(seed=i)

            blue_total_r, red_total_r = 0.0, 0.0

            for _ in range(20):  # 단축 에피소드
                blue_obs_kg, unc = fog_filter.observe(kg, ForceAlignment.BLUE)
                blue_state = build_state_vector(blue_obs_kg, uncertainty_map=unc)
                red_obs_kg, _ = fog_filter.observe(kg, ForceAlignment.RED)
                red_state = build_state_vector(red_obs_kg)

                blue_action, _, _ = self.blue_agent.select_action(blue_state, deterministic=True)
                red_action,  _, _ = self.red_agent.select_action(red_state,   deterministic=True)

                step = self.engine.run_step(kg)
                blue_total_r += -step.blue_total_casualties * 0.05
                red_total_r  +=  step.blue_total_casualties * 0.08

                if step.mission_status != "ongoing":
                    break

            blue_rewards.append(blue_total_r)
            red_rewards.append(red_total_r)

        # 단순화된 Nash Gap: 두 에이전트 보상의 표준편차 합
        nash_gap = (np.std(blue_rewards) + np.std(red_rewards)) / (n_eval * 2)
        return float(nash_gap)

    def train(self) -> Dict:
        """메인 학습 루프"""
        cfg = self.config
        print(f"\n🚀 Self-Play 학습 시작 (총 {cfg.total_episodes} 에피소드)")

        recent_winners = []
        update_counter = 0

        for ep in tqdm(range(cfg.total_episodes), desc="Self-Play"):
            progress = ep / cfg.total_episodes
            phase = self._get_current_phase(progress)

            # Population Phase: PFSP 가중 에이전트 선택
            if phase == "D":
                blue_member = self._pfsp_select(self.blue_population)
                red_member  = self._pfsp_select(self.red_population)
                blue = blue_member.agent
                red  = red_member.agent
            else:
                blue_member = red_member = None
                blue = self.blue_agent
                red  = self.red_agent

            # 에피소드 실행
            episode_stats = self.run_episode(ep, progress, blue, red, phase)

            # PFSP 결과 업데이트 (Phase D)
            if phase == "D" and blue_member is not None:
                w = episode_stats.winner
                if w == "blue_win":
                    blue_member.wins   += 1
                    red_member.losses  += 1
                elif w == "red_win":
                    blue_member.losses += 1
                    red_member.wins    += 1
                else:
                    blue_member.draws  += 1
                    red_member.draws   += 1
            self.stats.append(episode_stats)
            recent_winners.append(episode_stats.winner)
            if len(recent_winners) > 100:
                recent_winners.pop(0)

            # 업데이트
            update_counter += 1
            if update_counter >= cfg.update_interval:
                if phase in ["A", "C"]:
                    blue.update()
                if phase in ["B", "C"]:
                    red.update()
                if phase == "D":
                    # PFSP: 개체군 전체 업데이트
                    for m in self.blue_population:
                        m.agent.update()
                    for m in self.red_population:
                        m.agent.update()
                update_counter = 0

            # Nash Gap 체크
            if ep > 0 and ep % cfg.nash_check_interval == 0:
                nash_gap = self.compute_nash_gap(n_eval=10)
                self.nash_gaps.append(nash_gap)
                episode_stats.nash_gap = nash_gap

            # 로깅
            if ep % cfg.log_interval == 0 and ep > 0:
                recent = self.stats[-50:] if len(self.stats) >= 50 else self.stats
                blue_wr = sum(1 for s in recent if s.winner == "blue_win") / len(recent)
                red_wr = sum(1 for s in recent if s.winner == "red_win") / len(recent)
                draw_wr = sum(1 for s in recent if s.winner == "draw") / len(recent)
                resolved = [s for s in recent if s.winner in ("blue_win", "red_win")]
                blue_wr_resolved = (sum(1 for s in resolved if s.winner == "blue_win") / max(len(resolved), 1))
                avg_cas = np.mean([s.blue_casualties for s in recent])
                avg_force_reduction = np.mean([s.blue_force_reduction for s in recent])
                nash = self.nash_gaps[-1] if self.nash_gaps else 0.0
                print(f"\n  EP {ep:5d} | Phase={phase} | Blue WR={blue_wr:.1%} (resolved={blue_wr_resolved:.1%}) | "
                      f"Red WR={red_wr:.1%} | Draw={draw_wr:.1%} | "
                      f"Avg Cas={avg_cas:.1f} | Force Reduction={avg_force_reduction:.1%} | "
                      f"Nash Gap={nash:.4f}")

            # 체크포인트 저장
            if ep > 0 and ep % cfg.save_interval == 0:
                self.blue_agent.save(f"{cfg.checkpoint_dir}/blue_ep{ep}.pt")
                self.red_agent.save(f"{cfg.checkpoint_dir}/red_ep{ep}.pt")

        # 최종 통계
        final_stats = self._compute_final_stats()
        print(f"\n✅ 학습 완료!")
        print(f"   최종 Blue 승률: {final_stats['blue_win_rate']:.1%}")
        print(f"   평균 병력 절감율: {final_stats['avg_force_reduction']:.1%}")
        print(f"   최종 Nash Gap: {self.nash_gaps[-1] if self.nash_gaps else 'N/A'}")

        return final_stats

    def _compute_final_stats(self) -> Dict:
        recent = self.stats[-200:] if len(self.stats) >= 200 else self.stats
        resolved = [s for s in recent if s.winner in ("blue_win", "red_win")]

        # PFSP 개체군 다양성: 멤버 간 승률 표준편차 (높을수록 다양한 전략)
        pfsp_blue_wr = [m.win_rate_vs_main for m in self.blue_population if m.n_matches > 0]
        pfsp_red_wr  = [m.win_rate_vs_main for m in self.red_population  if m.n_matches > 0]
        pfsp_diversity = float(np.std(pfsp_blue_wr + pfsp_red_wr)) if pfsp_blue_wr else 0.0

        return {
            "total_episodes": len(self.stats),
            "blue_win_rate": sum(1 for s in recent if s.winner == "blue_win") / max(len(recent), 1),
            "red_win_rate":  sum(1 for s in recent if s.winner == "red_win")  / max(len(recent), 1),
            "draw_rate": sum(1 for s in recent if s.winner == "draw") / max(len(recent), 1),
            "blue_win_rate_resolved": sum(1 for s in resolved if s.winner == "blue_win") / max(len(resolved), 1),
            "resolved_ratio": len(resolved) / max(len(recent), 1),
            "avg_blue_casualties": np.mean([s.blue_casualties for s in recent]),
            "avg_red_casualties":  np.mean([s.red_casualties  for s in recent]),
            "avg_steps": np.mean([s.n_steps for s in recent]),
            "avg_force_reduction": np.mean([s.blue_force_reduction for s in recent]),
            "final_nash_gap": self.nash_gaps[-1] if self.nash_gaps else None,
            "converged": (self.nash_gaps[-1] < 0.05) if self.nash_gaps else False,
            "pfsp_diversity": pfsp_diversity,   # 개체군 전략 다양성 지표
        }


if __name__ == "__main__":
    print("=== Self-Play Trainer Test (단축 실행) ===")
    config = SelfPlayConfig(
        total_episodes=20,
        log_interval=5,
        nash_check_interval=10,
        save_interval=20
    )
    trainer = SelfPlayTrainer(config)
    stats = trainer.train()
    print(f"\n최종 통계: {stats}")
