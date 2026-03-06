"""
tests/test_hp_search.py
=======================
CVaR-aware 하이퍼파라미터 탐색 (R15) 단위 테스트

검증 항목:
  1. HPSearchConfig 기본값 유효성
  2. CVaRAwareHPSearch 인스턴스화
  3. 격자 조합 수 검증
  4. Pareto 프런트 계산 정확성 (도미난스 로직)
  5. 게이트 통과 후보가 없을 때 폴백 동작
  6. 경량 탐색 실행 (1 조합 × 1 seed × 5 MC runs)
  7. summary_table 출력 형식
"""
import json
import os
import types
import pytest
import numpy as np


# ──────────────────────────────────────────────
# 픽스처
# ──────────────────────────────────────────────

@pytest.fixture
def minimal_config(tmp_path):
    from utils.hp_search import HPSearchConfig
    return HPSearchConfig(
        param_grid={
            "lr": [3e-4],
            "clip_range": [0.2],
            "entropy_coef": [0.01],
        },
        n_mc_runs=5,
        seeds=[42],
        output_dir=str(tmp_path / "hp_out"),
        save_registry=False,
    )


@pytest.fixture
def base_args():
    """train.py args 최소 mock"""
    ns = types.SimpleNamespace(
        lr=3e-4,
        clip_range=0.2,
        entropy_coef=0.01,
        episodes=10,
        seed=42,
        checkpoint_dir="/tmp/falcon_hp_test",
    )
    return ns


# ──────────────────────────────────────────────
# 1. HPSearchConfig 기본값
# ──────────────────────────────────────────────

def test_hp_search_config_defaults():
    from utils.hp_search import HPSearchConfig
    cfg = HPSearchConfig()
    assert "lr" in cfg.param_grid
    assert "clip_range" in cfg.param_grid
    assert "entropy_coef" in cfg.param_grid
    assert cfg.n_mc_runs > 0
    assert len(cfg.seeds) > 0
    assert 0.0 < cfg.min_win_rate < 1.0
    assert cfg.max_cvar10_casualties > 0


# ──────────────────────────────────────────────
# 2. 인스턴스화
# ──────────────────────────────────────────────

def test_hp_search_instantiate(minimal_config):
    from utils.hp_search import CVaRAwareHPSearch
    search = CVaRAwareHPSearch(minimal_config)
    assert search is not None
    assert search.config is minimal_config


# ──────────────────────────────────────────────
# 3. 격자 조합 수 검증
# ──────────────────────────────────────────────

def test_param_grid_combinations():
    import itertools
    from utils.hp_search import HPSearchConfig
    cfg = HPSearchConfig(
        param_grid={"lr": [1e-4, 3e-4, 1e-3], "clip_range": [0.1, 0.2]},
        seeds=[0],
    )
    combos = list(itertools.product(*cfg.param_grid.values()))
    assert len(combos) == 6  # 3 × 2


# ──────────────────────────────────────────────
# 4. Pareto 프런트 계산 정확성
# ──────────────────────────────────────────────

def test_pareto_front_logic():
    from utils.hp_search import CVaRAwareHPSearch, HPCandidate, HPSearchConfig

    search = CVaRAwareHPSearch(HPSearchConfig())

    # A: wr=0.7, cvar=20 (우월)
    # B: wr=0.6, cvar=25 (A에 지배됨)
    # C: wr=0.8, cvar=30 (A와 비지배 — wr 높지만 cvar 높음)
    def make(wr, cvar, gate=True):
        return HPCandidate(
            params={}, mean_win_rate=wr, std_win_rate=0.0,
            mean_cvar10=cvar, mean_casualties=cvar, strategy_robustness=0.5,
            n_seeds=1, passes_gate=gate,
        )

    search._results = [make(0.7, 20), make(0.6, 25), make(0.8, 30)]
    pareto = search._compute_pareto_front()

    pareto_wrs = {c.mean_win_rate for c in pareto}
    assert 0.6 not in pareto_wrs, "B는 A에 지배되어야 함"
    assert 0.7 in pareto_wrs
    assert 0.8 in pareto_wrs


# ──────────────────────────────────────────────
# 5. 게이트 통과 후보 없을 때 폴백
# ──────────────────────────────────────────────

def test_pareto_fallback_when_no_gate_passed():
    from utils.hp_search import CVaRAwareHPSearch, HPCandidate, HPSearchConfig

    search = CVaRAwareHPSearch(HPSearchConfig())

    def make(wr, cvar):
        return HPCandidate(
            params={}, mean_win_rate=wr, std_win_rate=0.0,
            mean_cvar10=cvar, mean_casualties=cvar, strategy_robustness=0.5,
            n_seeds=1, passes_gate=False,  # 모두 게이트 미통과
        )

    search._results = [make(0.4, 50), make(0.35, 55), make(0.45, 60)]
    fallback = search._compute_pareto_front()

    # 게이트 미통과 시 win_rate 내림차순 전체 반환
    assert len(fallback) == 3
    assert fallback[0].mean_win_rate >= fallback[1].mean_win_rate


# ──────────────────────────────────────────────
# 6. 경량 end-to-end 탐색 실행
# ──────────────────────────────────────────────

def test_hp_search_run_minimal(minimal_config, base_args, tmp_path):
    """1조합 × 1seed × 5 MC runs — smoke test"""
    from utils.hp_search import CVaRAwareHPSearch

    search = CVaRAwareHPSearch(minimal_config)
    pareto = search.run(base_args)

    # 반환값 유효성
    assert isinstance(pareto, list)
    assert len(pareto) >= 1  # 최소 1개 후보

    # 결과 파일 생성 확인
    out_file = os.path.join(minimal_config.output_dir, "hp_search_results.json")
    assert os.path.isfile(out_file), "결과 JSON이 저장되어야 함"

    with open(out_file) as f:
        data = json.load(f)
    assert "results" in data
    assert len(data["results"]) == 1  # 조합 수 = 1
    assert "mean_win_rate" in data["results"][0]
    assert "mean_cvar10" in data["results"][0]
    assert "passes_gate" in data["results"][0]


# ──────────────────────────────────────────────
# 7. summary_table 형식
# ──────────────────────────────────────────────

def test_summary_table_format(minimal_config, base_args):
    from utils.hp_search import CVaRAwareHPSearch

    search = CVaRAwareHPSearch(minimal_config)
    search.run(base_args)

    table = search.summary_table()
    assert isinstance(table, str)
    assert len(table) > 0
    # 결과가 1개 이상이면 수치 포함
    assert "WR=" in table or "결과 없음" in table
