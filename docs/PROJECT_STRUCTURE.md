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
