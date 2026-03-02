"""
tests/test_phase3_r13_r14.py
=============================
R13: Counterfactual reward shaping integration tests
R14: HITL web interface Pareto candidate selection tests
"""
import copy
import pytest
import numpy as np


# ──────────────────────────────────────────────
# R13: Counterfactual Feedback Loop
# ──────────────────────────────────────────────

class TestR13CounterfactualFeedback:
    """R13 Counterfactual Reward Shaping 통합 테스트"""

    def test_cf_feedback_loop_instantiate(self):
        """CounterfactualFeedbackLoop 인스턴스 생성"""
        from explainability.counterfactual_feedback import (
            CounterfactualFeedbackLoop, CounterfactualFeedbackConfig,
        )
        config = CounterfactualFeedbackConfig(
            n_actions=6, warmup_episodes=5, cf_interval=1,
        )
        loop = CounterfactualFeedbackLoop(config=config)
        assert loop.config.n_actions == 6
        assert loop.config.warmup_episodes == 5

    def test_cf_feedback_loop_warmup_passthrough(self):
        """warmup 기간 동안 보상 변경 없이 원본 반환"""
        from explainability.counterfactual_feedback import (
            CounterfactualFeedbackLoop, CounterfactualFeedbackConfig,
        )
        config = CounterfactualFeedbackConfig(warmup_episodes=10, cf_interval=1)
        loop = CounterfactualFeedbackLoop(config=config)

        original_rewards = [1.0, 2.0, -0.5, 3.0]

        # warmup 중 → 원본 그대로
        loop.begin_episode()
        for i, r in enumerate(original_rewards):
            loop.record_step(i, action=0, reward=r, kg_snapshot=None)
        result = loop.end_episode(
            original_rewards, engine=None,
            ForceAlignment=None, UnitStatus=None, BlueActionSpace=None,
        )
        assert result == original_rewards

    def test_cf_shaper_compute_and_apply(self):
        """CounterfactualRewardShaper의 compute_shaping + apply_shaping 검증"""
        from explainability.reward_shaper import (
            CounterfactualRewardShaper, RewardShapingConfig,
        )
        config = RewardShapingConfig(
            shaping_scale=0.5, regret_threshold=0.01, max_bonus=2.0,
        )
        shaper = CounterfactualRewardShaper(config=config)

        actions = [0, 1, 2]
        rewards = [1.0, -0.5, 2.0]
        # Step 0: action=0 → actual=1.0, best=action 2 with 3.0 → regret=2.0
        # Step 1: action=1 → actual=-0.5, best=action 0 with 0.5 → regret=1.0
        # Step 2: action=2 → actual=2.0, best=action 2 → regret=0 (no shaping)
        cf_results = [
            {0: 1.0, 1: 0.5, 2: 3.0},
            {0: 0.5, 1: -0.5, 2: 0.0},
            {0: 1.0, 1: 1.5, 2: 2.0},
        ]

        signals = shaper.compute_shaping(actions, rewards, cf_results)
        assert len(signals) >= 1  # at least one step with regret > threshold

        # Verify shaping bonuses are positive
        for sig in signals:
            assert sig.shaping_bonus > 0
            assert sig.regret > 0

        # Apply shaping
        shaped = shaper.apply_shaping(rewards, signals)
        assert len(shaped) == len(rewards)
        # Shaped rewards should be >= original (only additive bonuses)
        for s, r in zip(shaped, rewards):
            assert s >= r

    def test_cf_feedback_full_episode(self):
        """CF 피드백 루프 전체 에피소드 시뮬레이션 (post-warmup)"""
        from explainability.counterfactual_feedback import (
            CounterfactualFeedbackLoop, CounterfactualFeedbackConfig,
        )
        from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
        from simulator.lanchester_engine import LanchesterEngine
        from rl_agent.blue_agent import BlueActionSpace

        config = CounterfactualFeedbackConfig(
            n_actions=BlueActionSpace.N_ACTIONS,
            warmup_episodes=0,  # 즉시 활성화
            cf_interval=1,
        )
        loop = CounterfactualFeedbackLoop(config=config)

        engine = LanchesterEngine(seed=42)
        kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=42)

        # 간단한 3-step 에피소드
        loop.begin_episode()
        rewards = []
        for step_t in range(3):
            kg_snap = copy.deepcopy(kg)
            action = step_t % BlueActionSpace.N_ACTIONS
            reward = float(np.random.uniform(-1, 1))
            rewards.append(reward)
            loop.record_step(step_t, action, reward, kg_snap)

        # 실제 build_action_pairs 함수 없이도 fallback으로 동작
        shaped = loop.end_episode(
            rewards, engine, ForceAlignment, UnitStatus, BlueActionSpace,
        )
        assert len(shaped) == len(rewards)
        # shaped는 float 리스트
        for v in shaped:
            assert isinstance(v, (int, float))

    def test_cf_feedback_summary(self):
        """CF 피드백 요약 통계 검증"""
        from explainability.counterfactual_feedback import (
            CounterfactualFeedbackLoop, CounterfactualFeedbackConfig,
        )
        config = CounterfactualFeedbackConfig(warmup_episodes=100)
        loop = CounterfactualFeedbackLoop(config=config)

        # 3 에피소드 실행 (warmup 중이므로 shaping 없음)
        for _ in range(3):
            loop.begin_episode()
            loop.record_step(0, 0, 1.0, None)
            loop.end_episode([1.0], None, None, None, None)

        summary = loop.get_summary()
        assert summary["cf_episodes_total"] == 3
        assert summary["cf_episodes_shaped"] == 0  # warmup 중이므로

    def test_cf_replay_priorities(self):
        """경험 재생 우선순위 계산 검증"""
        from explainability.reward_shaper import (
            CounterfactualRewardShaper, RewardShapingConfig, ShapingSignal,
        )
        shaper = CounterfactualRewardShaper(
            config=RewardShapingConfig(priority_boost=3.0)
        )

        signals = [
            ShapingSignal(step=1, action_taken=0, best_counterfactual_action=2,
                         regret=0.5, shaping_bonus=0.1, explanation="test"),
            ShapingSignal(step=3, action_taken=1, best_counterfactual_action=0,
                         regret=1.0, shaping_bonus=0.2, explanation="test"),
        ]

        priorities = shaper.compute_replay_priorities(signals, n_steps=5)
        assert len(priorities) == 5
        assert abs(priorities.sum() - 1.0) < 1e-5  # 정규화됨
        # step 1과 3의 우선순위가 높아야 함
        assert priorities[1] > priorities[0]
        assert priorities[3] > priorities[0]

    def test_cf_shaper_top_regrets(self):
        """전체 학습의 top regret 조회"""
        from explainability.reward_shaper import (
            CounterfactualRewardShaper, RewardShapingConfig,
        )
        shaper = CounterfactualRewardShaper(
            config=RewardShapingConfig(regret_threshold=0.01)
        )

        # 2개 에피소드 처리
        for ep in range(2):
            cf_results = [
                {0: 1.0, 1: 3.0, 2: 0.5},  # action 0 → regret = 2.0
                {0: 0.0, 1: 0.0, 2: 0.0},  # no regret
            ]
            shaper.compute_shaping([0, 1], [1.0, 0.0], cf_results)

        top = shaper.get_top_regrets(n=3)
        assert len(top) >= 1
        assert top[0].regret >= top[-1].regret  # 내림차순


# ──────────────────────────────────────────────
# R14: HITL Web Interface Pareto Enhancement
# ──────────────────────────────────────────────

class TestR14HITLWebInterface:
    """R14 HITL 웹 인터페이스 Pareto 후보 선택 테스트"""

    def test_session_initialize(self):
        """HITLSession 초기화 및 Pareto 후보 생성"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_r14", seed=42)
        state = session.initialize(n_blue=6, n_red=4)

        assert state["initialized"] is True
        assert len(state["strategy_options"]) > 0
        assert state["battle_status"]["blue_headcount"] > 0

    def test_session_pareto_data(self):
        """Pareto 시각화 데이터 구조 검증"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_pareto", seed=42)
        session.initialize(n_blue=6, n_red=4)

        pareto = session.get_pareto_data()
        assert len(pareto) > 0
        for p in pareto:
            assert "x" in p  # win_probability
            assert "y" in p  # casualties
            assert "size" in p  # force
            assert "risk" in p
            assert 0 <= p["x"] <= 1

    def test_session_update_constraints(self):
        """R14: 제약 조건 업데이트 → Pareto 재생성"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_constraints", seed=42)
        session.initialize(n_blue=8, n_red=6)

        # 엄격한 제약 적용
        result = session.update_constraints({
            "min_win_probability": 0.7,
            "max_casualties": 50,
            "max_time_steps": 40,
            "prefer_artillery": True,
        })

        assert "strategy_options" in result
        assert result["n_options"] > 0
        assert result["constraints_applied"]["min_win_probability"] == 0.7

        # 히스토리에 constraint_update 기록 확인
        assert any(h["type"] == "constraint_update" for h in session.history)

    def test_session_select_strategy(self):
        """전략 선택 및 피드백"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_select", seed=42)
        session.initialize(n_blue=6, n_red=4)

        # 첫 번째 전략 선택
        opts = session.strategy_options
        assert len(opts) > 0
        result = session.select_strategy(opts[0].strategy_id, feedback="Good option")
        assert "selected" in result
        assert result["selected"]["id"] == opts[0].strategy_id

    def test_session_invalid_strategy_select(self):
        """존재하지 않는 전략 선택 시 에러"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_invalid", seed=42)
        session.initialize()
        result = session.select_strategy("nonexistent_strategy")
        assert "error" in result

    def test_pareto_opt_to_web_conversion(self):
        """ParetoGenerator StrategyOption → Web StrategyOption 변환"""
        from hitl.web_interface import _pareto_opt_to_web
        from hitl.pareto_generator import (
            ParetoStrategyGenerator, CommanderConstraints,
        )
        from ontology.combat_schema import ScenarioFactory

        kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=42)
        gen = ParetoStrategyGenerator(n_candidates=3, mc_eval_runs=2)
        options = gen.generate(kg, CommanderConstraints())

        assert len(options) > 0
        web_opt = _pareto_opt_to_web(options[0], 0)

        assert web_opt.strategy_id == "strategy_0"
        assert web_opt.is_recommended is True
        assert web_opt.force_required > 0
        assert 0 <= web_opt.win_probability <= 1

        d = web_opt.to_dict()
        assert "id" in d
        assert "win_probability" in d
        assert "force_required" in d

    def test_state_includes_pareto_and_constraints(self):
        """get_state()에 pareto_data와 active_constraints 포함 확인"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_state", seed=42)
        session.initialize(n_blue=6, n_red=4)

        state = session.get_state()
        assert "pareto_data" in state
        assert "active_constraints" in state
        assert isinstance(state["pareto_data"], list)

    def test_constraint_update_without_init_returns_error(self):
        """초기화 전 제약 업데이트 시 에러 메시지"""
        from hitl.web_interface import HITLSession
        session = HITLSession("test_no_init", seed=42)
        result = session.update_constraints({"min_win_probability": 0.6})
        assert "error" in result
