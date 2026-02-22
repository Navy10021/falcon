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


def train_phase1(args):
    """Phase 1: Uncertainty-Aware GNN + Blue PPO 훈련"""
    logger.info("[Phase 1] Uncertainty-Aware GNN + PPO 훈련 시작")

    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
    from simulator.lanchester_engine import LanchesterEngine
    from simulator.fog_of_war import FogOfWarFilter, CurriculumScheduler
    from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors, CombatGNNLoss
    from rl_agent.blue_agent import BlueAgent, build_state_vector, BlueActionSpace, PPOConfig

    engine = LanchesterEngine(seed=args.seed)
    curriculum = CurriculumScheduler()
    gnn = BayesianHGT(node_in_dim=32, hidden_dim=128, n_layers=2, mc_samples=args.mc_samples)
    gnn_optim = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    gnn_loss_fn = CombatGNNLoss()

    ppo_config = PPOConfig(lr=args.lr, n_epochs=args.ppo_epochs)
    blue_agent = BlueAgent(config=ppo_config)

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

            # P1-3: 에이전트 행동을 시뮬레이션에 반영 (타겟 우선순위 결정)
            action_pairs = _build_blue_action_pairs(kg, action, ForceAlignment, UnitStatus, BlueActionSpace)
            step_result = engine.run_step(kg, action_pairs=action_pairs)
            done = (step_result.mission_status != "ongoing")

            # P1-1: prev_blue_hc 기반 스텝별 보상 계산 (P2-6: initial_force_size 전달)
            reward = blue_agent.compute_reward(
                step_result, prev_blue_hc,
                step_result.blue_total_headcount, avg_unc,
                initial_force_size=initial_blue_hc,
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


def main():
    parser = argparse.ArgumentParser(description="AI Combat Optimization System Training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="학습 Phase (1: GNN+PPO, 2: Self-Play, 3: HITL preference learning loop)")
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


if __name__ == "__main__":
    main()
