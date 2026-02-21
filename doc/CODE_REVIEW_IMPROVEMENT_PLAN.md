# FALCON 전체 코드 리뷰: 수정 / 보완 / 발전 항목 정리

작성일: 2026-02-21

## 1) 검토 범위 / 방식

이번 점검은 저장소 전반(ontology/simulator/gnn_model/rl_agent/hitl/evaluation/demo/train/evaluate/readme)을 대상으로,
- CLI 진입점 일관성
- 학습/평가 파이프라인 신뢰성
- 수치 안정성/지표 해석성
- 문서-코드 정합성
관점으로 수행했다.

---

## 2) 우선순위별 개선 과제

## P0 (즉시 보완 권장)

### P0-1. Demo의 "학습 전 정책"과 "학습 후 정책" 모드를 분리
- **현재 관찰**: `demo.py`는 `BlueAgent()/RedAgent()`를 무작위 초기화로 사용해 Monte Carlo에서 낮은 승률이 나올 수 있다.
- **문제**: 신규 사용자가 시스템 품질을 낮게 오해할 가능성이 높다.
- **개선 제안**:
  1. `--checkpoint` (blue/red 각각) 옵션 추가
  2. 체크포인트 미지정 시 `baseline` 모드 명시 출력
  3. 출력 리포트에 `policy_source=random_init|checkpoint` 필드 포함
- **기대 효과**: 데모 해석 오류 감소, 신뢰도 향상.

### P0-2. GNN 사상자 예측값 음수 방지
- **현재 관찰**: 데모 출력에서 `casualty_mean`이 음수로 관찰됨(예: -0.1).
- **문제**: 물리적으로 해석 불가능한 값이며 설명 가능성을 해친다.
- **개선 제안**:
  1. 출력 헤드에 `softplus` 또는 `clamp(min=0)` 적용
  2. CI 하한/상한 계산 시도 동일 스케일로 정합성 유지
  3. 평가 리포트에서 음수 예측 자동 경고
- **기대 효과**: 지표/설명 품질 개선, 사용자 신뢰 상승.

### P0-3. HITL 제약 위반 후보 반환 정책 개선
- **현재 관찰**: hard constraint를 모두 위반하면 현재는 위반 후보를 그대로 반환.
- **문제**: 지휘관 관점에서 “제약을 준수하지 않는 추천”으로 보일 수 있음.
- **개선 제안**:
  1. fallback 시 `feasibility_relaxation` 전략(최소 위반 해 탐색) 적용
  2. “모든 후보 위반” 시 명시 플래그 + 자동 완화안 제시
  3. 제약 위반 패널티를 Pareto 정렬 스코어에 반영
- **기대 효과**: HITL 실사용성 향상.

---

## P1 (단기 개선 권장)

### P1-1. `train.py` Phase 3 루프의 학습 목적/산출물 명확화
- **현재 관찰**: Phase 3는 preference JSON 저장 중심이며 RL 파라미터 업데이트는 없음.
- **문제**: 사용자 입장에서 "Phase 3 training"의 기대와 실제가 다를 수 있음.
- **개선 제안**:
  1. `--phase 3` 도움말에 "preference learning loop" 명시
  2. 산출물 표준화(`metrics.json`, `preferences.json`, `run_config.yaml`)
  3. 이후 RL 정책 반영 단계(offline fine-tuning) 연계 계획 문서화
- **기대 효과**: 기능 기대치 정렬, 운영 혼선 감소.

### P1-2. CLI 계약 테스트 추가
- **현재 관찰**: `evaluate.py --benchmark historical`, `train.py --phase 3 --hitl`는 최근 보완됨.
- **문제**: 추후 리팩터링 시 재파손 위험.
- **개선 제안**:
  1. `tests/test_cli_contract.py` 신설
  2. argparse 파싱 성공/실패 케이스를 최소 스모크로 고정
- **기대 효과**: 문서-CLI 불일치 재발 방지.

### P1-3. 재현성 강화(Seed 완전 고정)
- **현재 관찰**: 일부 모듈은 seed를 받지만 전체 파이프라인 deterministic 보장은 약함.
- **개선 제안**:
  1. 공통 `set_global_seed()` 유틸 작성
  2. numpy/torch/cudnn deterministic 옵션 통합
  3. 실행 로그에 seed 및 라이브러리 버전 출력
- **기대 효과**: 실험 재현성 및 디버깅 생산성 향상.

### P1-4. 평가 속도/비용 제어 옵션 확장
- **현재 관찰**: Monte Carlo runs 증가 시 체감 시간이 크게 증가.
- **개선 제안**:
  1. `--num-workers` (병렬 시뮬레이션)
  2. 진행률 출력 on/off
  3. `--fast` 프리셋(낮은 runs, 빠른 smoke)
- **기대 효과**: CI/로컬 점검 시간 단축.

---

## P2 (중기 발전 과제)

### P2-1. 설정 체계 통합 (YAML 기반)
- **문제**: train/evaluate/demo의 인자/기본값이 분산되어 drift 가능성 존재.
- **개선 제안**:
  1. `configs/*.yaml` 도입
  2. CLI는 override만 담당
  3. 실행 시 최종 config snapshot 저장

### P2-2. 로깅 체계 표준화
- **문제**: print 중심 로그는 장기 실험 추적에 한계.
- **개선 제안**:
  1. structlog 또는 표준 logging 포맷화
  2. run_id / episode / seed / phase 메타 일원화
  3. csv/jsonl 출력 지원

### P2-3. 지표 해석 레이어 강화
- **문제**: 승률 단일 지표 의존 시 전략 특성 파악이 어려움.
- **개선 제안**:
  1. force-efficiency frontier(승률-병력절감) 리포트
  2. 실패 시나리오 자동 군집화(terrain/action)
  3. calibration(ECE/Brier) 정식 리포팅

### P2-4. 문서 이중화(영문/국문) 유지 전략
- **문제**: `README.md` / `README_KOR.md` 동기화 drift 위험.
- **개선 제안**:
  1. 단일 소스 문서 + 번역 파생 방식
  2. 릴리즈 체크리스트에 문서 동기화 포함

---

## 3) 추천 실행 순서 (실행 가능 Backlog)

1. **P0-1, P0-2, P0-3** 먼저 처리 (사용자 체감 품질 핵심)  
2. CLI 계약 테스트(P1-2) 추가해 재발 방지  
3. 재현성(P1-3) + 속도 제어(P1-4) 반영  
4. 설정/로깅/지표 체계(P2)로 구조적 고도화

---

## 4) 완료 정의(Definition of Done) 제안

- Demo:
  - baseline/checkpoint 모드가 명확히 구분되어 출력된다.
- GNN:
  - casualty 예측값 음수가 발생하지 않는다.
- HITL:
  - hard constraint 불충족 시 완화/재탐색 로직이 설명 가능하게 동작한다.
- CLI:
  - 핵심 명령(phase/benchmark) 계약 테스트가 CI에서 통과한다.
- Docs:
  - README(EN/KR) 명령어 및 옵션이 코드와 1:1 일치한다.

