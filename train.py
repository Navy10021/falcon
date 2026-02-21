"""
train.py
========
AI Combat Optimization System — 메인 학습 스크립트
Phase 1 / Phase 2 / Phase 3 선택적 실행
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils.reproducibility import set_global_seed


def train_phase1(args):
    """Phase 1: Uncertainty-Aware GNN + Blue PPO 훈련"""
    print("🔵 Phase 1: Uncertainty-Aware GNN + PPO 훈련 시작")

    from ontology.combat_schema import ScenarioFactory, ForceAlignment
    from simulator.lanchester_engine import LanchesterEngine
    from simulator.fog_of_war import FogOfWarFilter, CurriculumScheduler
    from gnn_model.bayesian_hgt import BayesianHGT, prepare_graph_tensors, CombatGNNLoss
    from rl_agent.blue_agent import BlueAgent, build_state_vector, PPOConfig

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

    print(f"   총 에피소드: {args.episodes}")
    print(f"   Fog 커리큘럼: Enabled")
    print(f"   MC Dropout 샘플: {args.mc_samples}")

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

        # GNN 훈련 데이터 수집
        gnn_batch_x, gnn_batch_adj, gnn_batch_target = [], [], []

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

            # 환경 스텝
            step_result = engine.run_step(kg)
            done = (step_result.mission_status != "ongoing")

            reward = blue_agent.compute_reward(
                step_result, initial_blue_hc,
                step_result.blue_total_headcount, avg_unc
            )
            episode_reward += reward

            blue_agent.buffer.add(state, action, reward, log_prob, value, done, avg_unc)

            # GNN 학습 데이터
            gnn_batch_x.append(x)
            gnn_batch_adj.append(adj)
            target_cas = torch.tensor(float(step_result.blue_total_casualties), dtype=torch.float32)
            gnn_batch_target.append(target_cas)

        # PPO 업데이트
        ppo_metrics = blue_agent.update()

        # GNN 업데이트
        if gnn_batch_x:
            gnn.train()
            total_gnn_loss = 0
            for bx, ba, bt in zip(gnn_batch_x[:5], gnn_batch_adj[:5], gnn_batch_target[:5]):
                gnn_optim.zero_grad()
                out = gnn(bx, ba)
                targets = {"blue_casualties": bt, "risk_score": torch.tensor(0.5)}
                loss = gnn_loss_fn(out, targets)
                loss.backward()
                gnn_optim.step()
                total_gnn_loss += loss.item()
            gnn_losses.append(total_gnn_loss / min(5, len(gnn_batch_x)))

        rewards_history.append(episode_reward)
        win_rate_history.append(1 if 'blue_win' in str(step_result.mission_status) else 0)

        # 로깅
        if ep % args.log_interval == 0 and ep > 0:
            recent_wr = np.mean(win_rate_history[-50:]) if len(win_rate_history) >= 50 else np.mean(win_rate_history)
            recent_r  = np.mean(rewards_history[-50:])  if len(rewards_history) >= 50  else np.mean(rewards_history)
            gnn_l = np.mean(gnn_losses[-20:]) if gnn_losses else 0
            print(f"  EP {ep:5d}/{args.episodes} | Fog={fog_level.name:8s} | "
                  f"WR={recent_wr:.1%} | R={recent_r:.2f} | GNN Loss={gnn_l:.4f}")

        # 체크포인트
        if ep > 0 and ep % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"blue_phase1_ep{ep}.pt")
            blue_agent.save(ckpt_path)
            torch.save(gnn.state_dict(), ckpt_path.replace("blue_", "gnn_"))
            print(f"  💾 체크포인트 저장: {ckpt_path}")

    # 최종 저장
    blue_agent.save(os.path.join(args.checkpoint_dir, "blue_phase1_final.pt"))
    torch.save(gnn.state_dict(), os.path.join(args.checkpoint_dir, "gnn_phase1_final.pt"))
    print(f"\n✅ Phase 1 훈련 완료!")
    print(f"   최종 Blue 승률: {np.mean(win_rate_history[-100:]):.1%}")


def train_phase2(args):
    """Phase 2: Self-Play 훈련"""
    print("🔴 Phase 2: Self-Play (Blue vs Red) 훈련 시작")

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

    print(f"\n📊 Phase 2 최종 통계:")
    for k, v in final_stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")


def train_phase3(args):
    """Phase 3: HITL 통합 루프"""
    print("🟣 Phase 3: HITL 통합 훈련 시작")

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
            print(
                f"  EP {ep:5d}/{args.episodes} | "
                f"Adoption={learner.adoption_rate:.1%} | "
                f"AI={ai_recommended.label:8s} | "
                f"Selected={selected.label:8s} | "
                f"WinP={selected.win_probability:.1%}"
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

    print("\n✅ Phase 3 HITL 통합 완료!")
    print(f"   AI 추천 채택률: {learner.adoption_rate:.1%}")
    print(f"   선호도 저장: {pref_path}")
    print(f"   메트릭 저장: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Combat Optimization System Training")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="학습 Phase (1: GNN+PPO, 2: Self-Play, 3: HITL preference learning loop)")
    parser.add_argument("--episodes", type=int, default=500, help="에피소드 수")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--mc-samples", type=int, default=10, help="MC Dropout 샘플 수")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO 업데이트 에포크")
    parser.add_argument("--log-interval", type=int, default=50, help="로깅 간격")
    parser.add_argument("--save-interval", type=int, default=200, help="저장 간격")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="체크포인트 저장 경로")
    parser.add_argument("--hitl", action="store_true", help="Phase 3 HITL 통합 루프 활성화")

    args = parser.parse_args()

    print(f"\n🧠 AI Combat Optimization System v2.0")
    print(f"   Phase {args.phase} | Episodes: {args.episodes} | Seed: {args.seed}")
    print("─" * 50)

    set_global_seed(args.seed)

    if args.phase == 1:
        train_phase1(args)
    elif args.phase == 2:
        train_phase2(args)
    elif args.phase == 3:
        if not args.hitl:
            print("⚠️  Phase 3는 --hitl 옵션과 함께 실행하는 것을 권장합니다.")
        train_phase3(args)


if __name__ == "__main__":
    main()
