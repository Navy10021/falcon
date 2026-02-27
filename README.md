<div align="center">

# 🦅 FALCON: Force-Adaptive Learning for Combat Optimization Network

### Ontology-driven decision support with Bayesian GNN, Adversarial RL, and HITL command constraints

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

**FALCON is an experimentation framework for uncertainty-aware, robust battlefield decision recommendation.**

</div>

---

## Overview

FALCON integrates four core ideas into a single experimentation pipeline:

1. **Ontology-first state representation** for units, missions, command relations, and constraints.
2. **Bayesian graph reasoning** for confidence-aware situation understanding under partial observability.
3. **Adversarial reinforcement learning** for robust policy search against deceptive or worst-case opponents.
4. **Human-in-the-loop (HITL)** reranking under commander preferences and ROE-like constraints.

This repository is designed for research and prototyping workflows such as:
- robust policy comparison in simulated combat settings,
- uncertainty/risk-aware action recommendation,
- reproducible evaluation with Monte Carlo rollouts and benchmark suites.

---

## Key Capabilities

- 🧠 **Knowledge-structured combat modeling** via ontology modules.
- 🌫️ **Uncertainty-aware inference** via Bayesian GNN components.
- ⚔️ **Adversarial RL suite** including self-play and robust optimization variants.
- 👨‍✈️ **Commander-centered decision support** through HITL preference/constraint modules.
- 📊 **Evaluation stack** for scenario suites, Monte Carlo robustness, and metrics reporting.
- 🧾 **Reproducibility-focused demo pipeline** that generates artifacts and reports.

---

## Technical Stack (구현 기술 스택)

FALCON은 “아이디어 제안” 단계가 아닌, 실제 실험/검증 루프를 빠르게 돌릴 수 있도록 아래와 같은 기술 조합으로 구성됩니다.

### Core Runtime & ML

- **Python 3.10+**: 실험 코드, 시뮬레이터, 평가 파이프라인 전체를 통합.
- **PyTorch 2.x**: Bayesian GNN, 정책 네트워크, 학습 루프 구현.
- **NumPy / SciPy**: 수치 계산, 분포/샘플링, 통계 기반 평가.
- **scikit-learn**: 보정(calibration), 분석/메타 모델 보조.

### RL & Decision Intelligence

- **Gymnasium**: 실험 가능한 전장 환경 인터페이스 표준화.
- **Stable-Baselines3**: PPO 계열 기반의 빠른 베이스라인 확보.
- **Custom Adversarial RL Modules**: RARL, self-play, NFSP, MAPPO, league/PSRO 확장.

### Knowledge / Graph / Ontology Layer

- **Domain Ontology Modules (`ontology/`)**: 병과/임무/지휘관계/교전규칙 구조화.
- **Graph-centric Reasoning (`gnn_model/`)**: 부분 관측 상황에서 관계 기반 추론.
- **Constraint Encoding (ROE/HITL)**: 인간 지휘 의도와 제약을 정책 선택에 직접 반영.

### Evaluation, Ops, and Reproducibility

- **YAML Config System (`configs/`)**: 시나리오/평가/학습 설정의 선언적 관리.
- **Monte Carlo & Benchmark Suite (`evaluation/`, `demo/evaluation/`)**: 강건성 및 일반화 성능 검증.
- **PyTest-based Regression Gates (`tests/`)**: 단계별 계약 테스트 및 수치 안정성 점검.
- **TensorBoard / Rich / Plotly / Matplotlib / Seaborn**: 학습 추적, 리포팅, 시각화.

---

## Why FALCON: Differentiation & Feasibility

### 1) 차별성 (Differentiation)

1. **Ontology + Bayesian GNN + Adversarial RL + HITL의 단일 파이프라인 통합**  
   기존 접근이 규칙기반, 순수 RL, 혹은 단일 상황판단에 머무는 반면 FALCON은 지식표현·불확실성·강건학습·지휘자 제약을 연동합니다.

2. **정확도 중심이 아닌 “신뢰 가능한 추천” 중심 설계**  
   확률적 신뢰도(uncertainty), 정책 강건성, ROE 제약 준수 여부를 함께 평가해 실전형 의사결정 보조에 맞춥니다.

3. **실험 가능성과 설명 가능성을 동시에 고려한 구조**  
   재현 가능한 시나리오/시드 기반 평가와 함께 AAR/해석 모듈을 둬서 결과를 “왜 그런 추천이 나왔는지” 추적할 수 있습니다.

### 2) 실현 가능성 (Feasibility)

1. **이미 분리된 모듈 아키텍처**  
   `ontology`, `gnn_model`, `rl_agent`, `simulator`, `hitl`, `evaluation`로 분해되어 병렬 개발 및 단계적 고도화가 용이합니다.

2. **점진적 적용 전략이 가능**  
   Rule baseline → RL baseline → adversarial/self-play → HITL 재정렬 순으로 성능과 안정성을 단계적으로 검증할 수 있습니다.

3. **검증 친화적 실험 체계 내장**  
   CLI 실행 경로, 테스트 스위트, Monte Carlo 평가가 준비되어 있어 “모델 성능 주장”을 반복 가능하게 만들 수 있습니다.

4. **CPU-only 환경에서도 프로토타이핑 가능**  
   의존성/설계가 경량 실험을 지원하여, 초기 도입 단계에서 인프라 부담을 낮출 수 있습니다.

---

## Repository Structure

```text
falcon/
├── demo/              # Packaged demo/evaluation/reporting entrypoints
├── ontology/          # Domain schema, constraints, and ontology logic
├── gnn_model/         # Bayesian / uncertainty-aware GNN components
├── rl_agent/          # RL agents, adversarial/self-play training modules
├── simulator/         # Combat dynamics and simulation environments
├── hitl/              # Human-in-the-loop preference and constraint handling
├── evaluation/        # Benchmarks and Monte Carlo evaluation utilities
├── tests/             # Regression, contract, smoke, and phase tests
├── demo.py            # Top-level demo runner
└── evaluate.py        # Top-level evaluation runner
```

---

## Quick Start

### 1) Environment setup

```bash
git clone https://github.com/Navy10021/falcon
cd falcon

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run demo pipeline

```bash
python demo.py
```

### 3) Run packaged demo entrypoint

```bash
python -m demo.demo --scenario urban_defense --seed 42 --policy rule --out runs/demo_urban
```

### 4) Run evaluation

```bash
python evaluate.py --fast
python -m demo.evaluate --suite small --mc 20 --seed 42 --output-dir runs/eval
```

---

## Typical Workflow

1. Select or define a scenario and baseline policy.
2. Execute demo rollouts to generate trajectories/artifacts.
3. Run evaluation suites (fast or full) with Monte Carlo sampling.
4. Compare results with robustness and mission-effectiveness metrics.
5. Iterate on ontology, policy, uncertainty modeling, and HITL constraints.

---

## Development & Testing

```bash
# optional developer dependencies
pip install -r requirements-dev.txt

# run tests
pytest -q
```

If you are contributing algorithmic changes, include:
- reproducible run commands,
- seed/config information,
- before/after metric evidence.

---

## Documentation

- Korean extended documentation: [`README_KOR.md`](README_KOR.md)
- Demo package details: [`demo/DEMO_README.md`](demo/DEMO_README.md)

---

## Contributing

Contributions are welcome for:
- model quality and stability improvements,
- robustness/evaluation extensions,
- explainability and HITL UX improvements,
- test coverage and reproducibility tooling.

Please open an issue or PR with clear context and validation steps.

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
