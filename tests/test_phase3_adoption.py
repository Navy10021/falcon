"""Validate new adoption-rate semantics with explicit AI recommendation IDs."""
from hitl.preference_learner import CommanderPreferenceLearner, SelectionRecord
from hitl.pareto_generator import StrategyType


def main():
    learner = CommanderPreferenceLearner("tester")

    # 3 adopted, 1 rejected => 75%
    records = [
        ("option_1", "option_1"),
        ("option_2", "option_2"),
        ("option_3", "option_4"),
        ("option_4", "option_4"),
    ]

    for i, (ai_opt, selected) in enumerate(records):
        learner.record_selection(SelectionRecord(
            scenario_id=i,
            options_presented=["option_1", "option_2", "option_3", "option_4"],
            selected_option_id=selected,
            selected_type=StrategyType.BALANCED.value,
            force_size=800,
            win_probability=0.8,
            expected_casualties=10,
            ai_recommended_option_id=ai_opt,
        ))

    assert abs(learner.adoption_rate - 0.75) < 1e-6, learner.adoption_rate
    print("✅ phase3 adoption semantics test passed")


if __name__ == "__main__":
    main()
