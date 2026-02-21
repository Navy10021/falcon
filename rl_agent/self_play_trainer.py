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
import numpy as np
import os
import copy
from dataclasses import dataclass, field
from tqdm import tqdm

from ontology.combat_schema import ScenarioFactory, ForceAlignment
from simulator.lanchester_engine import LanchesterEngine
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

        # 에이전트 초기화
        self.blue_agent = BlueAgent()
        self.red_agent  = RedAgent()

        # Population (Phase D용)
        self.blue_population: List[BlueAgent] = [
            BlueAgent() for _ in range(self.config.population_size)
        ]
        self.red_population: List[RedAgent] = [
            RedAgent() for _ in range(self.config.population_size)
        ]

        # 통계 추적
        self.stats: List[EpisodeStats] = []
        self.nash_gaps: List[float] = []
        self.win_rates: Dict[str, List[float]] = {"blue": [], "red": [], "draw": []}

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

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

        blue_total_cas = 0
        red_total_cas  = 0
        done = False

        for step_t in range(cfg.max_steps_per_episode):
            if done:
                break

            # Blue 관측
            blue_obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)
            blue_state = build_state_vector(blue_obs_kg, uncertainty_map=uncertainty_map)
            avg_uncertainty = float(np.mean(list(uncertainty_map.values()))) if uncertainty_map else 0.0

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

            # 시뮬레이션 스텝 (행동 기반 교전 쌍 반영)
            action_pairs = self._build_action_pairs(kg, blue_action, red_action)
            step_result = self.engine.run_step(kg, action_pairs=action_pairs)
            blue_total_cas += step_result.blue_total_casualties
            red_total_cas  += step_result.red_total_casualties

            done = (step_result.mission_status != "ongoing")

            # Blue 버퍼 추가
            blue_reward = blue_agent.compute_reward(
                step_result, initial_blue_hc,
                step_result.blue_total_headcount,
                avg_uncertainty
            )
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

        winner = step_result.mission_status if 'step_result' in dir() else "draw"
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

            # Population Phase: 랜덤 에이전트 선택
            if phase == "D":
                blue = np.random.choice(self.blue_population)
                red  = np.random.choice(self.red_population)
            else:
                blue = self.blue_agent
                red  = self.red_agent

            # 에피소드 실행
            episode_stats = self.run_episode(ep, progress, blue, red, phase)
            self.stats.append(episode_stats)
            recent_winners.append(episode_stats.winner)
            if len(recent_winners) > 100:
                recent_winners.pop(0)

            # 업데이트
            update_counter += 1
            if update_counter >= cfg.update_interval:
                if phase in ["A", "C", "D"]:
                    blue.update()
                if phase in ["B", "C", "D"]:
                    red.update()
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
        return {
            "total_episodes": len(self.stats),
            "blue_win_rate": sum(1 for s in recent if s.winner == "blue_win") / max(len(recent), 1),
            "red_win_rate":  sum(1 for s in recent if s.winner == "red_win")  / max(len(recent), 1),
            "draw_rate": sum(1 for s in recent if s.winner == "draw") / max(len(recent), 1),
            "blue_win_rate_resolved": sum(1 for s in resolved if s.winner == "blue_win") / max(len(resolved), 1),
            "resolved_ratio": len(resolved) / max(len(recent), 1),
            "avg_blue_casualties": np.mean([s.blue_casualties for s in recent]),
            "avg_red_casualties":  np.mean([s.red_casualties  for s in recent]),
            "avg_force_reduction": np.mean([s.blue_force_reduction for s in recent]),
            "final_nash_gap": self.nash_gaps[-1] if self.nash_gaps else None,
            "converged": (self.nash_gaps[-1] < 0.05) if self.nash_gaps else False,
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
