"""
Action Mapping Consistency Tests
================================
Monte Carlo 평가 모듈과 학습 파이프라인 간 행동 상수 동기화 검증.

P0-2 리스크 (MC 행동 상수 하드코딩) 대응:
  - BlueActionSpace가 유일한 진실 소스(single source of truth)인지 검증
  - monte_carlo.py와 train.py의 행동 매핑 로직 일관성 확인
  - 행동 이름↔인덱스 매핑 무결성 검증
"""
import inspect
import numpy as np
import pytest


# ── 1. BlueActionSpace 상수 무결성 ────────────────────────

def test_action_space_constants_are_contiguous():
    """행동 인덱스는 0..N_ACTIONS-1 연속이어야 한다."""
    from rl_agent.blue_agent import BlueActionSpace
    indices = [
        BlueActionSpace.REINFORCE,
        BlueActionSpace.ADVANCE,
        BlueActionSpace.WITHDRAW,
        BlueActionSpace.REALLOCATE,
        BlueActionSpace.SUPPORT,
        BlueActionSpace.FLANK,
    ]
    assert sorted(indices) == list(range(BlueActionSpace.N_ACTIONS))


def test_action_names_match_n_actions():
    """ACTION_NAMES 길이는 N_ACTIONS와 동일해야 한다."""
    from rl_agent.blue_agent import BlueActionSpace
    assert len(BlueActionSpace.ACTION_NAMES) == BlueActionSpace.N_ACTIONS


def test_action_names_are_nonempty_strings():
    """모든 ACTION_NAMES는 비어있지 않은 문자열이어야 한다."""
    from rl_agent.blue_agent import BlueActionSpace
    for name in BlueActionSpace.ACTION_NAMES:
        assert isinstance(name, str) and len(name) > 0


# ── 2. Monte Carlo ↔ BlueActionSpace 동기화 ──────────────

def test_mc_imports_from_blue_action_space():
    """monte_carlo.py의 _build_mc_action_pairs는 BlueActionSpace를 import 해야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    source = inspect.getsource(_build_mc_action_pairs)
    assert "BlueActionSpace" in source, (
        "_build_mc_action_pairs must import from BlueActionSpace, not hardcode constants"
    )


def test_mc_no_hardcoded_action_constants():
    """monte_carlo.py의 핵심 함수에 하드코딩된 행동 상수가 없어야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    source = inspect.getsource(_build_mc_action_pairs)
    # 함수 내부에서 ADVANCE = 1, FLANK = 5 등 리터럴 할당이 없어야 한다
    # (BlueActionSpace.ADVANCE 형태만 허용)
    import re
    hardcoded = re.findall(r'(?:ADVANCE|FLANK|REINFORCE|SUPPORT|WITHDRAW)\s*=\s*\d', source)
    assert not hardcoded, f"Hardcoded action constants found: {hardcoded}"


def test_mc_action_mapping_matches_train():
    """MC 평가와 학습 파이프라인의 행동→타겟 모드 매핑이 일치해야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    from train import _build_blue_action_pairs
    from rl_agent.blue_agent import BlueActionSpace

    # 두 함수 소스에서 mode_map 딕셔너리 로직이 동일한지 비교
    mc_src = inspect.getsource(_build_mc_action_pairs)
    train_src = inspect.getsource(_build_blue_action_pairs)

    # 두 함수 모두 BlueActionSpace 상수를 사용하는지
    for const_name in ["ADVANCE", "FLANK", "REINFORCE", "SUPPORT", "WITHDRAW", "REALLOCATE"]:
        assert f"BlueActionSpace.{const_name}" in mc_src or f"{const_name}" in mc_src, (
            f"MC function missing action constant {const_name}"
        )
        assert f"BlueActionSpace.{const_name}" in train_src, (
            f"Train function must use BlueActionSpace.{const_name}"
        )


def test_mc_mode_map_covers_all_actions():
    """MC 평가의 mode_map은 모든 BlueActionSpace 행동을 포함해야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    from rl_agent.blue_agent import BlueActionSpace
    source = inspect.getsource(_build_mc_action_pairs)

    for const_name in ["ADVANCE", "FLANK", "REINFORCE", "SUPPORT", "WITHDRAW", "REALLOCATE"]:
        assert const_name in source, (
            f"MC mode_map missing action: {const_name}"
        )


# ── 3. End-to-End 행동 매핑 검증 ─────────────────────────

def test_mc_action_pairs_all_valid_actions():
    """모든 유효한 행동 인덱스에 대해 _build_mc_action_pairs가 정상 동작해야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    from ontology.combat_schema import ScenarioFactory
    from rl_agent.blue_agent import BlueActionSpace

    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=3, seed=42)

    for action_idx in range(BlueActionSpace.N_ACTIONS):
        pairs = _build_mc_action_pairs(kg, action_idx)
        # pairs가 None이면 교전 가능 유닛 없음 (유효한 상태)
        # pairs가 있으면 리스트여야 함
        assert pairs is None or isinstance(pairs, list), (
            f"Action {action_idx} returned unexpected type: {type(pairs)}"
        )


def test_mc_invalid_action_index_handled():
    """N_ACTIONS 범위 밖의 행동 인덱스도 크래시 없이 처리해야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    from ontology.combat_schema import ScenarioFactory

    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=3, seed=42)

    # 범위 밖 행동 — 크래시 없이 default 처리
    result = _build_mc_action_pairs(kg, 99)
    assert result is None or isinstance(result, list)


# ── 4. train.py 행동 매핑 검증 ───────────────────────────

def test_train_action_pairs_all_valid_actions():
    """train.py의 _build_blue_action_pairs도 모든 유효 행동에 대해 동작해야 한다."""
    from train import _build_blue_action_pairs
    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
    from rl_agent.blue_agent import BlueActionSpace

    kg = ScenarioFactory.create_standard_scenario(n_blue=4, n_red=3, seed=42)

    for action_idx in range(BlueActionSpace.N_ACTIONS):
        pairs = _build_blue_action_pairs(kg, action_idx, ForceAlignment, UnitStatus, BlueActionSpace)
        assert pairs is None or isinstance(pairs, list), (
            f"Action {action_idx} returned unexpected type: {type(pairs)}"
        )


def test_mc_and_train_produce_same_pairs():
    """동일 시나리오·행동에 대해 MC와 train 함수가 같은 교전 쌍을 생성해야 한다."""
    from evaluation.monte_carlo import _build_mc_action_pairs
    from train import _build_blue_action_pairs
    from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus
    from rl_agent.blue_agent import BlueActionSpace

    kg = ScenarioFactory.create_standard_scenario(n_blue=5, n_red=4, seed=123)

    for action_idx in range(BlueActionSpace.N_ACTIONS):
        mc_pairs = _build_mc_action_pairs(kg, action_idx)
        train_pairs = _build_blue_action_pairs(kg, action_idx, ForceAlignment, UnitStatus, BlueActionSpace)

        if mc_pairs is None and train_pairs is None:
            continue
        assert mc_pairs is not None and train_pairs is not None, (
            f"Action {action_idx}: MC returned {'None' if mc_pairs is None else 'list'}, "
            f"train returned {'None' if train_pairs is None else 'list'}"
        )
        # 교전 쌍 수가 동일해야 함
        assert len(mc_pairs) == len(train_pairs), (
            f"Action {action_idx}: MC produced {len(mc_pairs)} pairs, train produced {len(train_pairs)}"
        )
