"""Self-play trainer metrics sanity checks."""
from rl_agent.self_play_trainer import SelfPlayTrainer, SelfPlayConfig


def main():
    cfg = SelfPlayConfig(total_episodes=20, log_interval=10, nash_check_interval=10, save_interval=99999)
    trainer = SelfPlayTrainer(cfg)
    stats = trainer.train()

    assert "draw_rate" in stats
    assert "blue_win_rate_resolved" in stats
    assert "resolved_ratio" in stats
    assert 0.0 <= stats["draw_rate"] <= 1.0
    assert 0.0 <= stats["blue_win_rate_resolved"] <= 1.0
    assert 0.0 <= stats["resolved_ratio"] <= 1.0
    print("✅ phase2 selfplay metrics test passed")


if __name__ == "__main__":
    main()
