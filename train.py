"""
train.py
========
AI Combat Optimization System — 메인 학습 스크립트
Phase 1 / Phase 2 / Phase 3 선택적 실행
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.reproducibility import set_global_seed
from utils.config_loader import load_config

# 모듈 레벨 logger (main()에서 설정)
logger = logging.getLogger("falcon")


def setup_logger(checkpoint_dir: str, run_id: str) -> None:
    """콘솔 + 파일 핸들러를 가진 루트 logger 설정."""
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 콘솔 핸들러 (INFO 이상)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일 핸들러 (DEBUG 이상)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, f"run_{run_id}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Log file: %s", log_path)


def _build_blue_action_pairs(kg, blue_action, ForceAlignment, UnitStatus, BlueActionSpace):
    """
    Blue 에이전트의 선택 행동을 시뮬레이션 교전 쌍에 반영.
    행동 유형에 따라 공격 대상 우선순위(약한/강한/균형)를 결정한다.
    """
    blue_units = [u for u in kg.units.values()
                  if u.alignment == ForceAlignment.BLUE
                  and u.status != UnitStatus.DESTROYED
                  and u.headcount > 0]
    red_units = [u for u in kg.units.values()
                 if u.alignment == ForceAlignment.RED
                 and u.status != UnitStatus.DESTROYED
                 and u.headcount > 0]
    if not blue_units or not red_units:
        return None

    def pick_targets(units, mode):
        if mode == "weakest":
            return sorted(units, key=lambda u: u.headcount)
        if mode == "strongest":
            return sorted(units, key=lambda u: u.combat_power, reverse=True)
        return sorted(units, key=lambda u: u.headcount, reverse=True)

    mode_map = {
        BlueActionSpace.ADVANCE:    "weakest",
        BlueActionSpace.FLANK:      "weakest",
        BlueActionSpace.REINFORCE:  "strongest",
        BlueActionSpace.SUPPORT:    "strongest",
        BlueActionSpace.REALLOCATE: "balanced",
        BlueActionSpace.WITHDRAW:   "balanced",
    }
    blue_mode = mode_map.get(int(blue_action), "balanced")
    targets = pick_targets(red_units, blue_mode)

    attackers = blue_units
    if int(blue_action) == BlueActionSpace.WITHDRAW:
        attackers = blue_units[:max(1, len(blue_units) // 2)]

    return [(att.unit_id, targets[i % len(targets)].unit_id)
            for i, att in enumerate(attackers)]


def _build_blue_maneuver_targets(kg, blue_action, ForceAlignment, UnitStatus, BlueActionSpace,
                                  map_size: float = 30.0):
    """
    P3-1: Blue 행동 유형 → ManeuverEngine 이동 목표 좌표 딕셔너리 생성.
    각 Blue 유닛에 대해 {unit_id: (tx, ty)} 를 반환한다.

    행동 매핑:
      ADVANCE (1)   → 적 중심 방향으로 전진
      FLANK   (5)   → 가장 가까운 적의 측면 좌표로 기동
      WITHDRAW(2)   → 자군 후방(맵 좌측)으로 후퇴
      REINFORCE(0)  → 현재 위치 유지 (소폭 전진)
      SUPPORT (4)   → 가장 가까운 아군 근처로 이동
      REALLOCATE(3) → 현재 위치 유지
    """
    blue_units = [u for u in kg.units.values()
                  if u.alignment == ForceAlignment.BLUE
                  and u.status != UnitStatus.DESTROYED]
    red_units  = [u for u in kg.units.values()
                  if u.alignment == ForceAlignment.RED
                  and u.status != UnitStatus.DESTROYED]

    if not blue_units:
        return {}

    # 적 중심 계산
    if red_units:
        rcx = float(np.mean([u.position.x for u in red_units]))
        rcy = float(np.mean([u.position.y for u in red_units]))
    else:
        rcx, rcy = map_size * 0.6, map_size * 0.5

    # 아군 중심 계산
    bcx = float(np.mean([u.position.x for u in blue_units]))
    bcy = float(np.mean([u.position.y for u in blue_units]))

    targets = {}
    action = int(blue_action)

    for u in blue_units:
        if action == BlueActionSpace.ADVANCE:
            # 적 중심 방향으로 전진
            tx, ty = rcx, rcy
        elif action == BlueActionSpace.FLANK:
            # 가장 가까운 적의 90도 측면 좌표
            if red_units:
                closest_red = min(red_units, key=lambda r: u.position.distance_to(r.position))
                # 적→유닛 벡터 90도 회전
                dx = u.position.x - closest_red.position.x
                dy = u.position.y - closest_red.position.y
                norm = max(math.sqrt(dx**2 + dy**2), 0.1)
                tx = float(np.clip(closest_red.position.x + (-dy / norm) * 5.0, 0, map_size - 1))
                ty = float(np.clip(closest_red.position.y + ( dx / norm) * 5.0, 0, map_size - 1))
            else:
                tx, ty = rcx, rcy
        elif action == BlueActionSpace.WITHDRAW:
            # 자군 후방(현재 위치에서 적 반대 방향)으로 이동
            dx = bcx - rcx
            dy = bcy - rcy
            norm = max(math.sqrt(dx**2 + dy**2), 0.1)
            tx = float(np.clip(u.position.x + (dx / norm) * 5.0, 0, map_size - 1))
            ty = float(np.clip(u.position.y + (dy / norm) * 5.0, 0, map_size - 1))
        elif action == BlueActionSpace.SUPPORT:
            # 가장 가까운 아군 근처로 이동
            friendly = [f for f in blue_units if f.unit_id != u.unit_id]
            if friendly:
                nearest = min(friendly, key=lambda f: u.position.distance_to(f.position))
                tx, ty = nearest.position.x, nearest.position.y
            else:
                tx, ty = u.position.x, u.position.y
        else:
            # REINFORCE / REALLOCATE: 소폭 전진 유지
            tx = min(u.position.x + 1.0, map_size - 1)
            ty = u.position.y

        targets[u.unit_id] = (tx, ty)

    return targets


def train_phase1(args):
    """Phase 1: Uncertainty-Aware GNN + Blue PPO 훈련"""
    logger.info("[Phase 1] Uncertainty-Aware GNN + PPO 훈련 시작")

    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
    from simulator.lanchester_engine import LanchesterEngine
    from simulator.fog_of_war import FogOfWarFilter, CurriculumScheduler
    from simulator.maneuver_engine import ManeuverEngine  # P3-1
    from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors, CombatGNNLoss
    from rl_agent.blue_agent import BlueAgent, build_state_vector, BlueActionSpace, PPOConfig
    from ontology.doctrine_encoder import DoctrineEncoder  # P3-5

    engine = LanchesterEngine(seed=args.seed)
    maneuver_engine = ManeuverEngine(map_size=30, seed=args.seed)  # P3-1
    curriculum = CurriculumScheduler()
    gnn = BayesianHGT(node_in_dim=32, hidden_dim=128, n_layers=2, mc_samples=args.mc_samples)
    gnn_optim = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    gnn_loss_fn = CombatGNNLoss()

    ppo_config = PPOConfig(lr=args.lr, n_epochs=args.ppo_epochs)
    blue_agent = BlueAgent(config=ppo_config)
    doctrine_encoder = DoctrineEncoder()  # P3-5: 에피소드 간 공유 히스토리 추적

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    rewards_history = []
    win_rate_history = []
    gnn_losses = []

    logger.info("  총 에피소드: %d | Fog 커리큘럼: Enabled | MC Dropout 샘플: %d",
                args.episodes, args.mc_samples)

    for ep in range(args.episodes):
        progress = ep / args.episodes
        fog_level, fog_desc = curriculum.update(progress)
        fog_filter = FogOfWarFilter(fog_level, seed=ep)

        kg = ScenarioFactory.create_standard_scenario(
            n_blue=np.random.randint(5, 12),
            n_red=np.random.randint(4, 9),
            seed=args.seed + ep
        )
        initial_blue_hc = sum(u.headcount for u in kg.units.values()
                              if u.alignment == ForceAlignment.BLUE)

        episode_reward = 0
        done = False
        step_result = None

        # GNN 훈련 데이터 수집
        gnn_batch_x, gnn_batch_adj, gnn_batch_targets = [], [], []

        # P1-1: 스텝별 병력 추적 (직전 스텝 대비 절감 보상)
        prev_blue_hc = initial_blue_hc

        for step_t in range(50):
            if done:
                break

            # Fog 관측
            obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)

            # GNN 불확실성 예측
            x, adj = prepare_graph_tensors(obs_kg)
            with torch.no_grad():
                gnn_out = gnn.predict_with_uncertainty(x, adj)
            gnn_ext = np.array([
                gnn_out["casualty_mean"].item(),
                gnn_out["casualty_std"].item(),
                gnn_out["risk_mean"].item(),
                gnn_out["risk_std"].item(),
                gnn_out["epistemic_uncertainty"].item(),
                gnn_out["aleatoric_uncertainty"].item(),
            ], dtype=np.float32)
            gnn_ext = np.nan_to_num(gnn_ext, nan=0.0, posinf=10.0, neginf=-10.0)
            gnn_ext = np.clip(gnn_ext, -10.0, 10.0).astype(np.float32)

            # PPO 상태
            state = build_state_vector(obs_kg, gnn_extension=gnn_ext,
                                       uncertainty_map=uncertainty_map)
            avg_unc = float(np.mean(list(uncertainty_map.values()))) if uncertainty_map else 0.0

            action, log_prob, value = blue_agent.select_action(state)

            # P3-1: 행동 유형별 이동 목표 계산 후 ManeuverEngine 실행 (유닛 위치 갱신)
            blue_targets = _build_blue_maneuver_targets(
                kg, action, ForceAlignment, UnitStatus, BlueActionSpace
            )
            maneuver_result = maneuver_engine.run_maneuver_step(kg, blue_targets=blue_targets)

            # P1-3: 에이전트 행동을 시뮬레이션에 반영 (타겟 우선순위 결정)
            action_pairs = _build_blue_action_pairs(kg, action, ForceAlignment, UnitStatus, BlueActionSpace)
            step_result = engine.run_step(kg, action_pairs=action_pairs)
            done = (step_result.mission_status != "ongoing")

            # P3-2: GNN 노드 특성 동기화 (변경된 유닛 상태 → 그래프 반영)
            kg.update_node_features()

            # P3-5: 교리 준수도 평가 (기동 정보 포함)
            step_dict = {
                "blue_casualties": step_result.blue_total_casualties,
                "red_casualties":  step_result.red_total_casualties,
                "n_engageable":    maneuver_result.get("n_engageable", 0),
                "flanking_units":  maneuver_result.get("flanking_units", 0),
            }
            compliance = doctrine_encoder.evaluate(kg, step_t, step_dict)

            # P1-1: prev_blue_hc 기반 스텝별 보상 계산 (P2-6: initial_force_size, P3-5: doctrine_score)
            reward = blue_agent.compute_reward(
                step_result, prev_blue_hc,
                step_result.blue_total_headcount, avg_unc,
                initial_force_size=initial_blue_hc,
                doctrine_score=compliance.total_score,
            )
            prev_blue_hc = step_result.blue_total_headcount
            episode_reward += reward

            blue_agent.buffer.add(state, action, reward, log_prob, value, done, avg_unc)

            # GNN 학습 데이터 (P1-2: risk_score 동적 계산)
            gnn_batch_x.append(x)
            gnn_batch_adj.append(adj)
            blue_cas = float(step_result.blue_total_casualties)
            red_cas = float(step_result.red_total_casualties)
            risk_score = blue_cas / (blue_cas + red_cas + 1.0)  # [0, 1]
            gnn_batch_targets.append({
                "blue_casualties": torch.tensor(blue_cas, dtype=torch.float32),
                "risk_score":      torch.tensor(risk_score, dtype=torch.float32),
            })

        # PPO 업데이트
        ppo_metrics = blue_agent.update()

        # GNN 업데이트
        if gnn_batch_x:
            gnn.train()
            total_gnn_loss = 0
            for bx, ba, bt in zip(gnn_batch_x[:5], gnn_batch_adj[:5], gnn_batch_targets[:5]):
                gnn_optim.zero_grad()
                out = gnn(bx, ba)
                loss = gnn_loss_fn(out, bt)
                loss.backward()
                gnn_optim.step()
                total_gnn_loss += loss.item()
            gnn_losses.append(total_gnn_loss / min(5, len(gnn_batch_x)))
            gnn.eval()

        rewards_history.append(episode_reward)
        win_rate_history.append(1 if step_result is not None and 'blue_win' in str(step_result.mission_status) else 0)

        # 로깅
        if ep % args.log_interval == 0 and ep > 0:
            recent_wr = np.mean(win_rate_history[-50:]) if len(win_rate_history) >= 50 else np.mean(win_rate_history)
            recent_r  = np.mean(rewards_history[-50:])  if len(rewards_history) >= 50  else np.mean(rewards_history)
            gnn_l = np.mean(gnn_losses[-20:]) if gnn_losses else 0
            logger.info("EP %5d/%d | Fog=%-8s | WR=%.1%% | R=%.2f | GNN Loss=%.4f",
                        ep, args.episodes, fog_level.name,
                        recent_wr, recent_r, gnn_l)

        # 체크포인트
        if ep > 0 and ep % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"blue_phase1_ep{ep}.pt")
            blue_agent.save(ckpt_path)
            torch.save(gnn.state_dict(), ckpt_path.replace("blue_", "gnn_"))
            logger.info("체크포인트 저장: %s", ckpt_path)

    # 최종 저장
    blue_agent.save(os.path.join(args.checkpoint_dir, "blue_phase1_final.pt"))
    torch.save(gnn.state_dict(), os.path.join(args.checkpoint_dir, "gnn_phase1_final.pt"))
    logger.info("[Phase 1] 훈련 완료 | 최종 Blue 승률: %.1f%%",
                np.mean(win_rate_history[-100:]) * 100)


def train_phase2(args):
    """Phase 2: Self-Play 훈련"""
    logger.info("[Phase 2] Self-Play (Blue vs Red) 훈련 시작")

    from rl_agent.self_play_trainer import SelfPlayTrainer, SelfPlayConfig

    config = SelfPlayConfig(
        total_episodes=args.episodes,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        nash_check_interval=max(50, args.episodes // 20)
    )
    trainer = SelfPlayTrainer(config)
    final_stats = trainer.train()

    logger.info("[Phase 2] 최종 통계:")
    for k, v in final_stats.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        else:
            logger.info("  %s: %s", k, v)


def train_phase3(args):
    """Phase 3: HITL 통합 루프"""
    logger.info("[Phase 3] HITL 통합 훈련 시작")

    from ontology.combat_schema import ScenarioFactory
    from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints
    from hitl.preference_learner import CommanderPreferenceLearner, SelectionRecord

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    generator = ParetoStrategyGenerator(n_candidates=5)
    learner = CommanderPreferenceLearner(commander_id="phase3_commander")
    phase3_stats = {"episodes": args.episodes, "all_violated_episodes": 0}

    for ep in range(args.episodes):
        kg = ScenarioFactory.create_standard_scenario(
            n_blue=np.random.randint(5, 12),
            n_red=np.random.randint(4, 9),
            seed=args.seed + ep,
        )

        constraints = CommanderConstraints(
            min_win_probability=0.55,
            max_time_steps=50,
            prefer_flanking=(ep % 3 == 0),
            prefer_artillery=(ep % 4 == 0),
            avoid_urban=(ep % 5 == 0),
        )

        options = generator.generate(kg, constraints=constraints)
        if generator.last_generation_all_violated:
            phase3_stats["all_violated_episodes"] += 1
        ranked_options = learner.get_personalized_ranking(options)

        ai_recommended = ranked_options[0]
        # 모의 HITL 채택: 기본 70%는 AI 추천 수용, 30%는 차선책 선택
        accept_prob = 0.70
        if len(ranked_options) > 1 and np.random.rand() > accept_prob:
            selected = ranked_options[1]
            feedback_rating = 3
        else:
            selected = ai_recommended
            feedback_rating = 5

        learner.record_selection(
            SelectionRecord(
                scenario_id=ep,
                options_presented=[o.option_id for o in ranked_options],
                selected_option_id=selected.option_id,
                selected_type=selected.strategy_type.value,
                force_size=selected.force_size,
                win_probability=selected.win_probability,
                expected_casualties=selected.expected_casualties,
                feedback_rating=feedback_rating,
                ai_recommended_option_id=ai_recommended.option_id,
            )
        )

        if ep % args.log_interval == 0 and ep > 0:
            logger.info(
                "EP %5d/%d | Adoption=%.1f%% | AI=%-8s | Selected=%-8s | WinP=%.1f%%",
                ep, args.episodes, learner.adoption_rate * 100,
                ai_recommended.label, selected.label,
                selected.win_probability * 100,
            )

    pref_path = os.path.join(args.checkpoint_dir, "hitl_phase3_preferences.json")
    metrics_path = os.path.join(args.checkpoint_dir, "hitl_phase3_metrics.json")
    run_cfg_path = os.path.join(args.checkpoint_dir, "hitl_phase3_run_config.json")

    learner.save(pref_path)

    phase3_stats["adoption_rate"] = learner.adoption_rate
    phase3_stats["all_violated_ratio"] = (
        phase3_stats["all_violated_episodes"] / max(args.episodes, 1)
    )
    with open(metrics_path, "w") as f:
        json.dump(phase3_stats, f, indent=2)

    with open(run_cfg_path, "w") as f:
        json.dump({
            "phase": args.phase,
            "episodes": args.episodes,
            "seed": args.seed,
            "hitl": args.hitl,
            "log_interval": args.log_interval,
            "checkpoint_dir": args.checkpoint_dir
        }, f, indent=2)

    logger.info("[Phase 3] HITL 통합 완료 | AI 추천 채택률: %.1f%%", learner.adoption_rate * 100)
    logger.info("  선호도 저장: %s | 메트릭 저장: %s", pref_path, metrics_path)


def train_phase4(args):
    """
    Phase 4: P3-3 HITL 선호도 → RL 정책 재학습.

    Phase 3에서 학습된 지휘관 선호도(preference_weights)를
    PPO 보상 함수에 반영해 Blue 에이전트를 fine-tuning 한다.

    주요 흐름:
        1. Phase 1/2 체크포인트 로드
        2. HITL 선호도 파일 로드 → PreferenceRewardAdapter 생성
        3. 선호도 스케일이 반영된 보상으로 Phase 1 루프 재실행
    """
    logger.info("[Phase 4] HITL 선호도 반영 재학습 시작")
    logger.info("  preference_model: %s", args.preference_model)
    logger.info("  blue_checkpoint : %s", args.blue_checkpoint or "(없음 — 랜덤 초기화)")

    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
    from simulator.lanchester_engine import LanchesterEngine
    from simulator.fog_of_war import FogOfWarFilter, CurriculumScheduler
    from simulator.maneuver_engine import ManeuverEngine
    from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors, CombatGNNLoss
    from rl_agent.blue_agent import BlueAgent, build_state_vector, BlueActionSpace, PPOConfig
    from ontology.doctrine_encoder import DoctrineEncoder
    from hitl.preference_reward_adapter import PreferenceRewardAdapter
    from rl_agent.inverse_rl import IRLRewardLoader  # P3-4

    # 선호도 어댑터 로드
    if args.preference_model and os.path.isfile(args.preference_model):
        adapter = PreferenceRewardAdapter.from_file(args.preference_model)
    else:
        logger.warning("preference_model 파일 없음 → 균등 선호도(기본값) 사용")
        adapter = PreferenceRewardAdapter.uniform()
    logger.info("  %s", adapter.summary())

    # IRL 보상 로더 (data/irl_demos_summary.json)
    _irl_path = os.path.join(os.path.dirname(__file__), "data", "irl_demos_summary.json")
    try:
        irl_loader = IRLRewardLoader.from_file(_irl_path)
        logger.info("  IRL 보상 가중치 로드 완료: %s", _irl_path)
    except FileNotFoundError:
        irl_loader = None
        logger.warning("  IRL 데이터 파일 없음 (%s) → IRL 보너스 비활성화", _irl_path)

    engine          = LanchesterEngine(seed=args.seed)
    maneuver_engine = ManeuverEngine(map_size=30, seed=args.seed)
    curriculum      = CurriculumScheduler()
    gnn             = BayesianHGT(node_in_dim=32, hidden_dim=128, n_layers=2, mc_samples=args.mc_samples)
    gnn_optim       = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    gnn_loss_fn     = CombatGNNLoss()
    ppo_config      = PPOConfig(lr=args.lr, n_epochs=args.ppo_epochs)
    blue_agent      = BlueAgent(config=ppo_config)
    doctrine_encoder = DoctrineEncoder()

    # 체크포인트 로드
    if args.blue_checkpoint and os.path.isfile(args.blue_checkpoint):
        blue_agent.load(args.blue_checkpoint)
        logger.info("  Blue 에이전트 체크포인트 로드: %s", args.blue_checkpoint)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    rewards_history   = []
    win_rate_history  = []

    logger.info("  총 에피소드: %d | Fog 커리큘럼: Enabled", args.episodes)

    for ep in range(args.episodes):
        progress = ep / args.episodes
        fog_level, _ = curriculum.update(progress)
        fog_filter   = FogOfWarFilter(fog_level, seed=ep)

        kg = ScenarioFactory.create_standard_scenario(
            n_blue=np.random.randint(5, 12),
            n_red=np.random.randint(4, 9),
            seed=args.seed + ep,
        )
        initial_blue_hc = sum(u.headcount for u in kg.units.values()
                              if u.alignment == ForceAlignment.BLUE)
        prev_blue_hc    = initial_blue_hc
        episode_reward  = 0.0
        done            = False
        step_result     = None

        for step_t in range(50):
            if done:
                break

            obs_kg, uncertainty_map = fog_filter.observe(kg, ForceAlignment.BLUE)
            x, adj = prepare_graph_tensors(obs_kg)
            with torch.no_grad():
                gnn_out = gnn.predict_with_uncertainty(x, adj)
            gnn_ext = np.array([
                gnn_out["casualty_mean"].item(), gnn_out["casualty_std"].item(),
                gnn_out["risk_mean"].item(),      gnn_out["risk_std"].item(),
                gnn_out["epistemic_uncertainty"].item(),
                gnn_out["aleatoric_uncertainty"].item(),
            ], dtype=np.float32)
            gnn_ext = np.clip(np.nan_to_num(gnn_ext, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

            state = build_state_vector(obs_kg, gnn_extension=gnn_ext,
                                       uncertainty_map=uncertainty_map)
            avg_unc = float(np.mean(list(uncertainty_map.values()))) if uncertainty_map else 0.0
            action, log_prob, value = blue_agent.select_action(state)

            # ManeuverEngine: 위치 갱신
            blue_targets    = _build_blue_maneuver_targets(
                kg, action, ForceAlignment, UnitStatus, BlueActionSpace)
            maneuver_result = maneuver_engine.run_maneuver_step(kg, blue_targets=blue_targets)

            # LanchesterEngine: 교전
            action_pairs = _build_blue_action_pairs(
                kg, action, ForceAlignment, UnitStatus, BlueActionSpace)
            step_result = engine.run_step(kg, action_pairs=action_pairs)
            done = (step_result.mission_status != "ongoing")

            kg.update_node_features()

            # 교리 평가
            step_dict = {
                "blue_casualties": step_result.blue_total_casualties,
                "red_casualties":  step_result.red_total_casualties,
                "n_engageable":    maneuver_result.get("n_engageable", 0),
                "flanking_units":  maneuver_result.get("flanking_units", 0),
            }
            compliance = doctrine_encoder.evaluate(kg, step_t, step_dict)

            # 기본 보상 항목 계산
            terminal         = step_result.mission_status != "ongoing"
            force_reduction  = prev_blue_hc - step_result.blue_total_headcount
            win_reward_raw   = (blue_agent._W_WIN if step_result.mission_status == "blue_win"
                                else -blue_agent._W_WIN if step_result.mission_status == "red_win"
                                else 0.0)
            casualty_penalty_raw = step_result.blue_total_casualties * blue_agent._W_CASUALTY
            force_reward_raw     = blue_agent._W_FORCE_SAVE * force_reduction if force_reduction > 0 else 0.0
            survival_bonus_raw   = (blue_agent._W_FORCE_RATIO
                                    * step_result.blue_total_headcount / max(initial_blue_hc, 1)
                                    if terminal and initial_blue_hc > 0 else 0.0)
            enemy_dmg_raw        = step_result.red_total_casualties * blue_agent._W_ENEMY_DMG
            doctrine_bonus_raw   = blue_agent._W_DOCTRINE * (compliance.total_score - 0.5)
            unc_pen_raw          = (blue_agent.config.uncertainty_penalty_coef
                                    * avg_unc * force_reduction * 0.05
                                    if avg_unc > 0.5 and force_reduction > 0 else 0.0)

            # P3-4: IRL 보너스 (교리 기반 행동 강화)
            irl_bonus = irl_loader.compute_irl_bonus(kg, step_result, maneuver_result) \
                        if irl_loader is not None else 0.0

            # 선호도 스케일 적용 (IRL 보너스는 enemy_damage에 합산)
            reward = adapter.compute_scaled_reward(
                base_reward=0.0,
                win_reward=win_reward_raw,
                force_reward=force_reward_raw,
                casualty_penalty=casualty_penalty_raw,
                survival_bonus=survival_bonus_raw,
                enemy_damage=enemy_dmg_raw + irl_bonus,
                doctrine_bonus=doctrine_bonus_raw,
                uncertainty_penalty=unc_pen_raw,
            )
            prev_blue_hc    = step_result.blue_total_headcount
            episode_reward += reward
            blue_agent.buffer.add(state, action, reward, log_prob, value, done, avg_unc)

        blue_agent.update()
        rewards_history.append(episode_reward)
        win_rate_history.append(
            1 if step_result is not None and "blue_win" in str(step_result.mission_status) else 0)

        if ep % args.log_interval == 0 and ep > 0:
            recent_wr = np.mean(win_rate_history[-50:]) if len(win_rate_history) >= 50 else np.mean(win_rate_history)
            recent_r  = np.mean(rewards_history[-50:])  if len(rewards_history) >= 50  else np.mean(rewards_history)
            logger.info("EP %5d/%d | WR=%.1f%% | R=%.2f", ep, args.episodes, recent_wr * 100, recent_r)

        if ep > 0 and ep % args.save_interval == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"blue_phase4_ep{ep}.pt")
            blue_agent.save(ckpt)
            logger.info("체크포인트 저장: %s", ckpt)

    blue_agent.save(os.path.join(args.checkpoint_dir, "blue_phase4_final.pt"))
    logger.info("[Phase 4] 재학습 완료 | 최종 Blue 승률: %.1f%%",
                np.mean(win_rate_history[-100:]) * 100)


def main():
    parser = argparse.ArgumentParser(description="AI Combat Optimization System Training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="학습 Phase (1: GNN+PPO, 2: Self-Play, 3: HITL, 4: 선호도 반영 재학습)")
    parser.add_argument("--preference-model", type=str, default=None,
                        help="Phase 4: HITL 선호도 JSON 파일 경로")
    parser.add_argument("--blue-checkpoint", type=str, default=None,
                        help="Phase 4: Blue 에이전트 초기 체크포인트")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 설정 파일 경로 (예: configs/phase1.yaml)")
    parser.add_argument("--episodes", type=int, default=None, help="에피소드 수")
    parser.add_argument("--lr", type=float, default=None, help="학습률")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    parser.add_argument("--mc-samples", type=int, default=None, help="MC Dropout 샘플 수")
    parser.add_argument("--ppo-epochs", type=int, default=None, help="PPO 업데이트 에포크")
    parser.add_argument("--log-interval", type=int, default=None, help="로깅 간격")
    parser.add_argument("--save-interval", type=int, default=None, help="저장 간격")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="체크포인트 저장 경로")
    parser.add_argument("--hitl", action="store_true", help="Phase 3 HITL 통합 루프 활성화")

    args = parser.parse_args()

    # YAML config 로드 후 CLI 미지정 값에 기본값 적용
    cfg = load_config(args.config, args, section="training")
    # argparse default를 None → config 값으로 채움
    defaults = {
        "episodes": 500, "lr": 3e-4, "seed": 42, "mc_samples": 10,
        "ppo_epochs": 4, "log_interval": 50, "save_interval": 200,
        "checkpoint_dir": "checkpoints",
    }
    for key, fallback in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, cfg.get(key, fallback))

    # 고유 run_id 생성 (타임스탬프 기반)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id

    setup_logger(args.checkpoint_dir, run_id)

    logger.info("AI Combat Optimization System v2.0")
    logger.info("run_id=%s | Phase=%d | Episodes=%d | Seed=%d",
                run_id, args.phase, args.episodes, args.seed)
    if args.config:
        logger.info("Config file: %s", args.config)

    set_global_seed(args.seed)

    if args.phase == 1:
        train_phase1(args)
    elif args.phase == 2:
        train_phase2(args)
    elif args.phase == 3:
        if not args.hitl:
            logger.warning("Phase 3는 --hitl 옵션과 함께 실행하는 것을 권장합니다.")
        train_phase3(args)
    elif args.phase == 4:
        if not args.preference_model:
            logger.warning("--preference-model 미지정 → 균등 선호도로 재학습합니다.")
        train_phase4(args)


if __name__ == "__main__":
    main()
