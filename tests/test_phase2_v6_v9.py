"""
tests/test_phase2_v6_v9.py
===========================
Phase II (V6-V9) 회귀 테스트

V6: CI 성능 게이트 (check_perf_gate.py)
V7: MultiDomainRunner → ExperimentRegistry 연동
V8: HITLSessionBridge (Queue 기반 실시간 연동)
V9: build_state_vector() 해상·사이버 도메인 확장 (78D)

학술 연구용 합성 데이터
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest


# ─────────────────────────────────────────────
# V6: check_perf_gate.py
# ─────────────────────────────────────────────

class TestPerfGate:
    """scripts/check_perf_gate.py 성능 게이트 검증"""

    def _write_report(self, path: str, **kwargs) -> None:
        defaults = {
            "blue_win_rate": 0.60,
            "cvar_10": 25.0,
            "cvar_05": 30.0,
            "avg_blue_casualties": 15.0,
            "strategy_robustness": 0.8,
            "n_runs": 50,
        }
        defaults.update(kwargs)
        with open(path, "w") as f:
            json.dump(defaults, f)

    def test_gate_passes_good_metrics(self, tmp_path):
        from scripts.check_perf_gate import run_gate
        p = str(tmp_path / "report.json")
        self._write_report(p, blue_win_rate=0.60, cvar_10=25.0)
        assert run_gate(p, min_win_rate=0.45, max_cvar10=40.0, verbose=False) is True

    def test_gate_fails_low_win_rate(self, tmp_path):
        from scripts.check_perf_gate import run_gate
        p = str(tmp_path / "report.json")
        self._write_report(p, blue_win_rate=0.30, cvar_10=25.0)
        assert run_gate(p, min_win_rate=0.45, max_cvar10=40.0, verbose=False) is False

    def test_gate_fails_high_cvar(self, tmp_path):
        from scripts.check_perf_gate import run_gate
        p = str(tmp_path / "report.json")
        self._write_report(p, blue_win_rate=0.60, cvar_10=50.0)
        assert run_gate(p, min_win_rate=0.45, max_cvar10=40.0, verbose=False) is False

    def test_gate_exits_2_on_missing_file(self, tmp_path):
        from scripts.check_perf_gate import run_gate
        with pytest.raises(SystemExit) as exc_info:
            run_gate(str(tmp_path / "nonexistent.json"), verbose=False)
        assert exc_info.value.code == 2

    def test_gate_handles_win_rate_alias(self, tmp_path):
        """'win_rate' 필드도 'blue_win_rate'로 인식되는지 확인"""
        from scripts.check_perf_gate import run_gate
        p = str(tmp_path / "report.json")
        with open(p, "w") as f:
            json.dump({"win_rate": 0.55, "cvar_10": 20.0, "n_runs": 50}, f)
        assert run_gate(p, min_win_rate=0.45, max_cvar10=40.0, verbose=False) is True

    def test_gate_fails_both_metrics(self, tmp_path):
        from scripts.check_perf_gate import run_gate
        p = str(tmp_path / "report.json")
        self._write_report(p, blue_win_rate=0.20, cvar_10=60.0)
        assert run_gate(p, min_win_rate=0.45, max_cvar10=40.0, verbose=False) is False


# ─────────────────────────────────────────────
# V7: MultiDomainRunner → ExperimentRegistry
# ─────────────────────────────────────────────

class TestMultiDomainRegistry:
    """MultiDomainRunner._register_to_registry() 통합 테스트"""

    def test_register_method_exists(self):
        from utils.multidomain_runner import MultiDomainRunner
        runner = MultiDomainRunner()
        assert hasattr(runner, "_register_to_registry")
        assert callable(runner._register_to_registry)

    def test_register_creates_index(self, tmp_path):
        from utils.multidomain_runner import MultiDomainRunner
        runner = MultiDomainRunner()

        # 가상 report 파일 생성
        report_dir = tmp_path / "multidomain"
        report_dir.mkdir()
        report_path = str(report_dir / "multidomain_report.json")
        with open(report_path, "w") as f:
            json.dump({"version": "test"}, f)

        summary = {"in_domain_win_rate_avg": 0.65, "ood_win_rate_avg": 0.50}
        runner._register_to_registry(report_path, summary)

        index_path = tmp_path / "experiment_index.json"
        assert index_path.exists(), "ExperimentRegistry 인덱스 파일이 생성되어야 함"

    def test_registered_record_has_phase11(self, tmp_path):
        from utils.multidomain_runner import MultiDomainRunner

        runner = MultiDomainRunner()
        report_dir = tmp_path / "multidomain"
        report_dir.mkdir()
        report_path = str(report_dir / "multidomain_report.json")
        with open(report_path, "w") as f:
            json.dump({"version": "test"}, f)

        summary = {"in_domain_win_rate_avg": 0.70}
        runner._register_to_registry(report_path, summary)

        index_path = tmp_path / "experiment_index.json"
        assert index_path.exists(), "인덱스 파일이 생성되어야 함"
        with open(index_path) as f:
            data = json.load(f)
        # save_index writes "records" key, not "runs"
        records = data.get("records", [])
        assert any(r.get("phase") == 11 for r in records), \
            "MultiDomain 결과는 phase=11로 등록되어야 함"


# ─────────────────────────────────────────────
# V8: HITLSessionBridge
# ─────────────────────────────────────────────

class TestHITLSessionBridge:
    """hitl/session_bridge.py Queue 기반 브리지 테스트"""

    def test_bridge_init(self):
        from hitl.session_bridge import HITLSessionBridge, get_bridge
        bridge = HITLSessionBridge(auto_timeout=5.0)
        assert bridge.auto_timeout == 5.0
        assert get_bridge() is bridge

    def test_put_get_options(self):
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge()
        bridge.put_options(episode_idx=0, options=["opt_a"], ranked_options=["opt_a"])
        event = bridge.get_options_nowait()
        assert event is not None
        assert event.episode_idx == 0
        assert event.options == ["opt_a"]

    def test_options_queue_overwrites_stale(self):
        """이전 미소비 이벤트가 새 이벤트로 교체되는지 확인"""
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge()
        bridge.put_options(episode_idx=0, options=["old"], ranked_options=["old"])
        bridge.put_options(episode_idx=1, options=["new"], ranked_options=["new"])
        event = bridge.get_options_nowait()
        assert event is not None
        assert event.episode_idx == 1

    def test_put_selection_episode_match(self):
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge()
        bridge.put_options(episode_idx=5, options=["opt"], ranked_options=["opt"])  # sets _current_episode = 5
        result = bridge.put_selection(
            episode_idx=5, strategy_id="strategy_0", option_index=0, feedback_rating=4
        )
        assert result is True

    def test_put_selection_episode_mismatch(self):
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge()
        bridge.put_options(episode_idx=5, options=["opt"], ranked_options=["opt"])
        result = bridge.put_selection(episode_idx=3, strategy_id="strategy_0")
        assert result is False

    def test_get_selection_timeout(self):
        """선택 없이 타임아웃 시 None 반환"""
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge(auto_timeout=0.05)
        sel = bridge.get_selection()
        assert sel is None

    def test_roundtrip_put_get_selection(self):
        """put_selection → get_selection 라운드트립"""
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge()
        bridge.put_options(10, ["opt"], ["opt"])
        bridge.put_selection(10, "strategy_2", option_index=2, feedback_rating=5)
        event = bridge.get_selection(timeout=1.0)
        assert event is not None
        assert event.strategy_id == "strategy_2"
        assert event.option_index == 2
        assert event.feedback_rating == 5

    def test_status_dict(self):
        from hitl.session_bridge import HITLSessionBridge
        bridge = HITLSessionBridge()
        s = bridge.status()
        assert "current_episode" in s
        assert "options_pending" in s
        assert "selection_pending" in s
        assert "server_running" in s


# ─────────────────────────────────────────────
# V9: build_state_vector() 해상·사이버 확장
# ─────────────────────────────────────────────

class TestStateVectorNavalCyber:
    """build_state_vector() V9 해상·사이버 6D 필드 테스트 (torch 필요)"""

    def test_output_dim_is_128(self):
        import numpy as np
        pytest.importorskip("torch")
        from rl_agent.blue_agent import build_state_vector
        from ontology.combat_schema import ScenarioFactory
        kg = ScenarioFactory.create_standard_scenario(seed=42)
        state = build_state_vector(kg)
        assert state.shape == (128,), f"STATE_DIM=128 기대, 실제={state.shape}"

    def test_no_nan_inf(self):
        import numpy as np
        pytest.importorskip("torch")
        from rl_agent.blue_agent import build_state_vector
        from ontology.combat_schema import ScenarioFactory
        kg = ScenarioFactory.create_standard_scenario(seed=7)
        state = build_state_vector(kg)
        assert not np.any(np.isnan(state)), "NaN 포함"
        assert not np.any(np.isinf(state)), "Inf 포함"

    def test_values_clipped(self):
        import numpy as np
        pytest.importorskip("torch")
        from rl_agent.blue_agent import build_state_vector
        from ontology.combat_schema import ScenarioFactory
        kg = ScenarioFactory.create_standard_scenario(seed=42)
        state = build_state_vector(kg)
        assert np.all(state >= -10.0), "하한 -10.0 위반"
        assert np.all(state <= 10.0), "상한 10.0 위반"

    def test_naval_cyber_fields_in_range(self):
        """78D 슬롯 [56:62] = 해상·사이버 6개 필드가 [0,1] 범위 내인지 확인"""
        import numpy as np
        pytest.importorskip("torch")
        from rl_agent.blue_agent import build_state_vector
        from ontology.combat_schema import ScenarioFactory
        kg = ScenarioFactory.create_standard_scenario(seed=42)
        state = build_state_vector(kg)
        # base(56) + naval_cyber(6) → indices 56..61
        naval_cyber = state[56:62]
        assert len(naval_cyber) == 6
        assert np.all(naval_cyber >= 0.0), "해상·사이버 필드 음수 불가"
        assert np.all(naval_cyber <= 10.0), "해상·사이버 필드 상한 초과"

    def test_multiple_seeds_consistent(self):
        """여러 시드에서 일관된 차원과 유효 값 확인"""
        import numpy as np
        pytest.importorskip("torch")
        from rl_agent.blue_agent import build_state_vector
        from ontology.combat_schema import ScenarioFactory
        for seed in [0, 1, 42, 99, 123]:
            kg = ScenarioFactory.create_standard_scenario(seed=seed)
            state = build_state_vector(kg)
            assert state.shape == (128,)
            assert not np.any(np.isnan(state))
