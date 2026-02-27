# FALCON (Force-Adaptive Learning for Combat Optimization Network)

FALCON은 **전장 의사결정 지원**을 위해 설계된 연구용 프레임워크입니다. 이 저장소는 단일 모델만이 아니라, 아래를 묶은 **통합 실험 스택**을 제공합니다.

- 온톨로지 기반 시나리오/전력 모델링
- 불확실성 인지(Bayesian) GNN 계열 표현
- 적대적 강화학습(자기대전/리그/강건화)
- ROE(교전수칙)·HITL·사후분석(AAR) 산출물

---

## TL;DR

- **5분 이해 포인트**: FALCON은 `시나리오 생성 → 전투 시뮬레이션 → 정책 평가 → 4종 아티팩트 출력`의 파이프라인으로 동작합니다.
- **5분 실행 포인트**: 아래 Quickstart 명령 1개로 데모를 실행하면 결과 파일 4종이 즉시 생성됩니다.
- **핵심 산출물 위치**: `runs/proposal_demo/summary.json`, `metrics.csv`, `fig_episode.png`, `aar.html`.

---

## 1) 아키텍처 개요 (Reviewer용)

FALCON은 다음 6개 계층으로 이해하면 빠릅니다.

1. **Knowledge/Ontology Layer**
   - `ontology/combat_schema.py`: 부대/전력/지휘 구조 스키마
   - `ontology/scenario_presets.py`: 표준 시나리오 프리셋
2. **Simulation Layer**
   - `simulator/*`: 교전, 소모, 기동, 안개(fog-of-war) 모델
3. **Representation & Uncertainty Layer**
   - `gnn_model/*`: 그래프 기반 상태 표현, 불확실성 처리/보정
4. **Decision/RL Layer**
   - `rl_agent/*`: PPO·자기대전·리그·강건학습 계열
5. **Human & Policy Layer**
   - `hitl/*`: 제약 기반 의사결정 보조 및 인간 최종 판단 흐름
   - `ontology/roe_ethics.py`: ROE/윤리 규칙 점검
6. **Evaluation & Reporting Layer**
   - `evaluation/*`: 성능/강건성 평가
   - `falcon/report.py`, `falcon/io/artifacts.py`: 보고서/아티팩트 저장

> 즉, FALCON은 “정책 그 자체”보다, **정책을 신뢰 가능하게 검증·보고하기 위한 시스템 아키텍처**에 가깝습니다.

---

## 2) Technical Spec (핵심만)

### 입력
- 시나리오 이름 (`--scenario`)
- 랜덤 시드 (`--seed`)
- 정책 모드 (`--policy`)

### 실행 단위
- 기본 데모는 5개 에피소드의 교전 결과를 누적 계산

### 출력(고정)
- `summary.json`: 성공률/손실률/ROE 위반률/런타임 요약
- `metrics.csv`: 에피소드별 상세 메트릭
- `fig_episode.png`: 에피소드 결과 시각화
- `aar.html`: 자동 사후분석(AAR) 리포트

### 기본 데모 파라미터
- Scenario 기본값: `urban_defense`
- Seed 기본값: `42`
- Policy 기본값: `rule`

---

## 3) Quickstart (3분 데모, 명령 1개)

아래 **한 줄**만 실행하세요.

```bash
python -m falcon.demo --scenario urban_defense --seed 42 --policy rule --out runs/proposal_demo
```

실행 완료 후:
- 콘솔에 아티팩트 경로가 출력되고,
- `runs/proposal_demo/`에 결과 4종이 생성됩니다.

---

## 4) 공모 제출용 산출물 4종 (파일명/경로 고정)

아래 4개는 심사 제출 시 기준 산출물로 고정합니다.

1. `runs/proposal_demo/summary.json`
2. `runs/proposal_demo/metrics.csv`
3. `runs/proposal_demo/fig_episode.png`
4. `runs/proposal_demo/aar.html`

---

## 5) 공모 KPI ↔ Metric ↔ Artifact 매핑

| 공모 KPI | 측정 Metric | 기준 Artifact | 해석 포인트 |
|---|---|---|---|
| 임무 달성도 | `success_rate` | `summary.json` | 에피소드 성공 비율 |
| 아군 피해 최소화 | `friendly_loss` | `summary.json`, `metrics.csv` | 평균 피해율 + 회차 분포 |
| 교전 규범 준수 | `roe_violation_rate`, `roe_violations` | `summary.json`, `metrics.csv` | ROE 위반 빈도 추적 |
| 전투 효율/운용성 | `runtime_sec`, `duration_sec` | `summary.json`, `metrics.csv` | 처리시간 및 에피소드 지연 |
| 설명가능성/사후검토 | 정성 점검(리포트 내용) | `aar.html`, `fig_episode.png` | 의사결정 결과의 설명 가능성 |

---

## 6) Reproducibility

### Seed 고정
- 권장: `--seed 42` (문서/발표/심사용 공통)
- 비교 실험: `42, 43, 44` 다중 시드로 평균/분산 보고

### Suite (권장 검증 루틴)
1. 데모 실행 (아티팩트 생성 확인)
2. 단위 테스트 실행
3. 필요 시 평가 스크립트로 시나리오별 비교

예시:

```bash
python -m falcon.demo --scenario urban_defense --seed 42 --policy rule --out runs/repro_seed42
pytest -q
```

### CPU-only 실행
- 데모는 CPU 환경에서 동작하도록 설계됨
- GPU 없이도 아티팩트 생성/검토 가능
- CI/심사 환경에서는 우선 CPU-only 경로를 기준으로 검증 권장

---

## 7) Proposal Assets 디렉터리 가이드

기획서 삽입용 리소스는 아래 경로를 사용합니다.

- `docs/proposal_assets/architecture.png` (placeholder)
- `docs/proposal_assets/results_table.md` (placeholder)
- `docs/proposal_assets/roe_case_study.md` (template)
- `docs/proposal_assets/calibration.png` (placeholder)
- `docs/figures/` (추가 도판 저장용)

---

