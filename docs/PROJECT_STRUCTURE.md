# Project Structure Conventions

This document defines a cleaner folder policy for FALCON so future additions stay organized.

## Current top-level domains

- `ontology/`: schema and military-domain knowledge graph logic
- `simulator/`: battle simulation engines and environment dynamics
- `gnn_model/`: Bayesian graph model and uncertainty components
- `rl_agent/`: PPO / self-play / RL agents and trainers
- `hitl/`: human-in-the-loop constraints, ranking, and preference learning
- `evaluation/`: benchmark and Monte Carlo evaluation
- `explainability/`: report and interpretation utilities
- `visualization/`: dashboard and visualization code
- `tests/`: executable test scripts
- `data/`: sample/generated data artifacts
- `utils/`: cross-cutting utilities (seed/reproducibility, helpers)
- `docs/`: documentation and reports

## Documentation layout (new)

- `docs/reports/`: analysis and planning docs
  - `CODE_REVIEW_IMPROVEMENT_PLAN.md`
- `docs/PROJECT_STRUCTURE.md`: repository structure policy (this file)

## Rules for future cleanliness

1. **No new root-level docs** except `README.md`, `README_KOR.md`, `LICENSE`, and packaging files.
2. Put design/review/planning files under `docs/reports/`.
3. Put reusable helper code under `utils/` (not inside `tests/`).
4. Keep test files prefixed as `tests/test_*.py`.
5. If a module grows large, split by responsibility under its own subfolder (e.g., `rl_agent/self_play/`).

## Recommended next cleanup (non-breaking)

- Split `tests/` into:
  - `tests/smoke/`
  - `tests/integration/`
  - `tests/regression/`
- Move notebook assets into `docs/notebooks/` or `notebooks/` with naming convention.
- Add `configs/` for phase-specific YAML configs to reduce CLI-only configuration drift.

## Phase Contract (P1)

학습/평가 단계 간 상태·보상 의미 드리프트를 방지하기 위한 최소 계약:

| Phase | 상태 입력 계약 | 보상 계약 | 종료/평가 계약 |
|---|---|---|---|
| Phase 1 (PPO) | `build_state_vector()` 128D, 불확실성/시계열 확장 허용 | `force_size_before`는 **직전 step의 blue headcount**를 사용한 차분 보상 | `mission_status`가 `ongoing`이 아닐 때 종료 |
| Phase 2 (Self-Play PPO) | Phase 1과 동일한 상태 벡터 계약 + Maneuver 반영 | Blue 보상은 Phase 1과 동일하게 `prev_blue_hc` 기반 차분 | timeout 시 `headcount` 정책으로 승패 판정 가능 |
| Phase 2 (MAPPO) | 유닛별 로컬 관측 + action mask, CTDE critic | 팀 보상 + 유닛별 행동 보너스(보급/방어) | mixed engine step 결과의 `mission_status` 사용 |
| Phase 3 (HITL) | Pareto 후보는 휴리스틱 생성 후 top-k MC-lite 보정 | Hard constraints 위반 후보 제외(또는 최소 위반 fallback) | 선택/채택률 로그와 함께 옵션 이력 저장 |
| Evaluation (MC) | 학습과 동일한 action→engagement pair 매핑 | 단일 보상 대신 성능 분포(win/casualty/force/CVaR) 추적 | 고정 schema JSON(`falcon-mc-report-v1`) 산출 |

운영 규칙:
- 새로운 보상 항목을 추가할 때는 위 표와 테스트를 동시에 갱신한다.
- `force_size_before`는 "에피소드 시작값"이 아니라 "직전 step 값"으로 통일한다.
- 평가 리포트는 run metadata(`run_id`, `git_sha`, seed 계열)를 반드시 포함한다.
