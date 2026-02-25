# FALCON 프로젝트 종합 심층 분석 보고서 v3

**작성일**: 2026-02-25  
**분석 기준**: 현재 `main` 코드베이스 (로컬 워킹트리)  
**참고 문서**: `FALCON_COMPREHENSIVE_ANALYSIS.md`, `FALCON_COMPREHENSIVE_ANALYSIS_v2.md`  
**분석 목표**: 현재 구조를 유지한 상태에서 품질/성능/연구 확장성을 함께 높이는 실행 가능한 개선안 제시

---

## 0. Executive Summary

FALCON은 이미 **온톨로지-시뮬레이션-GNN-RL-HITL-평가**로 이어지는 end-to-end 파이프라인을 확보했고, v2에서 제기된 핵심 연결성 결함(행동 반영, Maneuver 통합, IRL `time_pressure`, Phase4 재학습 루프 등)을 다수 해결했다. 현재 단계의 핵심 과제는 “새 모듈 추가”보다 **학습·평가·운영의 일관성 강화**다.

본 v3 분석의 결론은 다음 3가지다.

1. **기초 아키텍처는 유지해야 한다.** (계층 분리는 현재 강점)
2. **성과를 제한하는 병목은 연결 불일치/운영 안정성 이슈다.**
   - Phase 1/4와 Phase 2의 보상 정의 차이
   - 로깅 핸들러 중복 가능성
   - Pareto 후보 생성의 휴리스틱 의존도
3. **향후 12주 로드맵은 “신뢰성→일관성→성능고도화” 순서가 최적**이다.

---

## 1. 분석 관점 및 방법

### 1-1. 분석 관점

- **구조 보존 원칙**: 기존 6계층(ontology/simulator/gnn/rl/hitl/evaluation) 유지
- **연결 일관성 원칙**: 학습 루프와 평가 루프가 동일한 행동/상태 의미를 공유
- **실험 재현성 원칙**: 설정/시드/로그/평가지표를 동일 실험 단위로 추적 가능해야 함
- **운영 전개성 원칙**: 연구 코드에서 운영형 코드로 전환 가능한 최소 규율 확보

### 1-2. 분석 방법

- v1/v2 보고서의 제안 사항이 실제 코드에 반영되었는지 재검증
- `train.py`, `self_play_trainer.py`, `evaluation/monte_carlo.py`, `inverse_rl.py`, `pareto_generator.py` 중심으로 파이프라인 연결 점검
- 현재 잔존 리스크를 **즉시(P0) / 단기(P1) / 중기(P2) / 장기(P3)**로 재분류

---

## 2. 현재 아키텍처 성숙도 진단 (v3)

## 2-1. 아키텍처 총평

현재 FALCON은 “모듈이 많지만 파편화된 상태”를 넘어, **핵심 경로가 실제로 연결되는 통합형 연구 플랫폼**으로 진입했다. 특히 아래 연결이 의미 있다.

- Phase 1/4에서 행동→기동→교전→교리평가→보상 업데이트까지 폐루프 구현
- Phase 2(Self-Play)에 Maneuver + GNN 확장 입력 반영
- Monte Carlo 평가에 행동 반영 로직 추가
- HITL 선호도와 IRL 보너스가 Phase 4 보상에 동시 반영

이는 “아이디어 집합”에서 “실험 가능한 체계”로 전환되었다는 신호다.

## 2-2. 계층별 성숙도 매트릭스

| 계층 | 성숙도 | 현재 상태 | 핵심 보완 포인트 |
|---|---|---|---|
| Ontology | 상 | 시나리오/교리/멀티도메인 스키마 풍부 | 멀티도메인 학습 루프 실사용화 |
| Simulator | 상 | Lanchester + Maneuver + Dynamics 결합 진전 | 계산량/성능 최적화, 상호작용 fidelity 개선 |
| GNN | 상 | Bayesian 불확실성 + 학습 루프 연결 | Calibration/Attention pooling 고도화 |
| RL | 중상 | PPO/Self-Play/Phase4 재학습 확보 | Phase 간 보상 정의 일관화 |
| HITL | 중상 | 제약+Pareto+선호도 학습 루프 확보 | 후보 성능 추정의 실측 기반 강화 |
| Evaluation | 중상 | MC 병렬화+행동 반영+CVaR 도입 | 지표 해석 가능성/대시보드 통합 |

---

## 3. v2 대비 주요 진전 사항 (확인 결과)

## 3-1. 학습-평가 일관성 개선

- Monte Carlo에서 에이전트 행동이 실제 `run_step`에 반영되도록 수정되어, “학습된 정책을 평가할 때 정책이 무시되는 문제”가 해소됨.
- Self-Play에 ManeuverEngine이 연동되어 Phase 1과의 환경 괴리가 완화됨.

## 3-2. 선호도/교리/IRL 통합 강화

- Phase 4에서 HITL 선호도 스케일과 IRL 보너스를 결합해 재학습하는 루프가 안정화됨.
- `time_pressure`가 상수값이 아닌 에피소드 진행도 기반으로 계산되도록 개선됨.

## 3-3. 실험 인프라 개선

- YAML config 기반 실행 및 CI 워크플로가 정착되어 재현성/회귀 검출 기반이 강화됨.

---

## 4. 심층 리스크 분석 (현재 구조 유지 전제)

> 아래 항목은 “아키텍처 변경 없이” 효과적으로 개선 가능한 실질 리스크다.

## 4-1. P0 (즉시) — 신뢰성/정합성

### P0-1. Self-Play 보상의 `force_size_before` 의미 불일치

- Phase 1/4는 `prev_blue_hc`를 사용해 스텝 차분 보상을 계산하는데,
- Self-Play(`run_episode`)는 초기 병력을 계속 전달한다.

**영향**: Phase 1/4와 Phase 2의 보상 의미가 달라 정책 전이 시 편차 발생 가능.  
**개선**: Self-Play도 `prev_blue_hc` 추적 방식으로 정렬.

### P0-2. logger 핸들러 누적 가능성

`setup_logger()`가 호출될 때 기존 핸들러 제거 없이 추가한다. 단일 실행에서는 문제 없지만 반복 실행/테스트에서 중복 로그 위험이 있다.

**영향**: 로그 중복, 파일 핸들러 중복 오픈, 디버깅 혼선.  
**개선**: `if logger.handlers: clear` 패턴 적용.

## 4-2. P1 (단기) — 실험 일관성/품질

### P1-1. Pareto 후보 생성은 여전히 휴리스틱 중심

`ParetoStrategyGenerator`는 동적 `win_base`를 사용하지만, 후보별 기대 성능(`win_probability`, `casualties`, `time`)이 실제 rollout 기반이 아니라 파라미터화된 근사값 중심이다.

**영향**: 지휘관 UI에서 제시되는 전략 간 간격이 실제 환경과 어긋날 수 있음.  
**개선**: 후보 top-k에 한해 `mc_eval_runs`를 실제 사용해 보정(경량 N-run).

### P1-2. 학습 단계별 상태/보상 규약 문서화 부족

현재는 코드로는 연결됐지만, “Phase별 입력/보상의 계약(contract)”이 문서화되어 있지 않아 기능 추가 시 드리프트 위험이 높다.

**개선**: `docs/reports` 또는 `docs/PROJECT_STRUCTURE.md`에 phase contract 표준 추가.

## 4-3. P2 (중기) — 확장성과 운영성

### P2-1. MAPPO의 실제 운영 경로 부재

`mappo.py`는 torch 기반 학습 가능 컴포넌트를 포함하지만, 기본 학습 엔트리(`train.py`)에서 정식 실험 플래그/루틴으로 통합되지는 않았다.

**개선**: `--algorithm {ppo,mappo}` 수준의 최소 경로를 제공해 실험 가능 상태로 승격.

### P2-2. Evaluation 결과의 운영형 표준 부족

Monte Carlo report는 풍부하지만, 실험 단위(run_id/config/git_sha)와 1:1 매핑되는 저장 포맷이 약하다.

**개선**: JSON schema 고정 + 결과 메타데이터 표준화.

## 4-4. P3 (장기) — 연구 고도화

### P3-1. 다도메인 스키마의 학습 루프 완전 통합

온톨로지 레벨의 멀티도메인 자산은 풍부하나, 학습/평가 루프에서 도메인 조합 실험 매트릭스로 완전 전개되지는 않았다.

### P3-2. 불확실성-의사결정 결합 고도화

현재 불확실성은 상태 확장/패널티로 반영되지만, 행동 선택 단계에서 위험 민감 정책(CVaR-aware policy)로 직접 결합되면 전장 의사결정 해석력이 증가한다.

---

## 5. 개선/발전 제안 (구조 유지형)

## 5-1. 설계 원칙

- 모듈 추가보다 **표준 인터페이스/계약 강화** 우선
- “더 많은 알고리즘”보다 **학습-평가 정합성 + 재현성** 우선
- 기능 개발 전 **측정 지표를 먼저 정의**

## 5-2. 핵심 개선안

1. **Phase 보상 계약 통일**
   - `force_size_before`, terminal 보너스, uncertainty penalty의 공통 사양 정립
2. **Pareto 후보 실측 보정 레이어 추가**
   - 후보 생성(heuristic) + 후보 검증(MC-lite) 2단 구조
3. **실험 아티팩트 표준화**
   - `run_manifest.json`(config/hash/seed/phase/checkpoint) 저장
4. **MAPPO 최소 실험 경로 개통**
   - 기존 코드 활용해 실험 가능 상태만 먼저 확보
5. **리스크 민감형 정책 평가 지표 추가**
   - 평균 성능 + 꼬리위험(CVaR) 동시 관리

---

## 6. 우선순위별 실행 로드맵 (v3)

## 6-1. 0~2주 (Priority-1: 신뢰성 회복)

- [R1] Self-Play 보상 계산을 `prev_blue_hc` 기반으로 정렬
- [R2] logger 핸들러 중복 방지
- [R3] 실험 결과 저장 시 `run_id/config/git_sha/phase` 자동 기록

**완료 기준 (DoD)**
- 동일 seed에서 Phase1↔2 전이 실험 분산(variance) 유의미 감소
- 로그 중복 라인 0건
- 결과 파일만으로 실험 재실행 가능

## 6-2. 2~6주 (Priority-2: 품질/운영성 강화)

- [R4] Pareto 후보 top-k에 MC-lite 보정 연결 (`mc_eval_runs` 실제 사용)
- [R5] Phase contract 문서화 + 테스트(계약 테스트) 추가
- [R6] MC 리포트 JSON schema 고정 및 시각화 대시보드 연동

**완료 기준 (DoD)**
- Pareto 추천과 실제 rollout 결과의 rank correlation 개선
- Phase contract 위반 PR을 CI에서 자동 차단
- 분석가가 단일 JSON으로 실험 비교 가능

## 6-3. 6~12주 (Priority-3: 성능 고도화)

- [R7] MAPPO 엔트리 포인트 연결 (`--algorithm mappo`)
- [R8] 멀티도메인 학습 시나리오 배치 실험(ground/air/naval/multi)
- [R9] 위험 민감 평가(CVaR 중심) 기반 하이퍼파라미터 탐색

### 실행 상세안

#### [R7] MAPPO 엔트리 포인트 연결 고도화

- **목표**: `train.py --phase 2 --algorithm mappo`를 표준 실험 파이프라인(로그/체크포인트/리포트)과 완전 정렬
- **핵심 작업**
  1. CLI 계약 고정: `phase=2 + algorithm in {ppo,mappo}` 조합 외 입력을 명시적으로 차단
  2. 공통 산출물 통일: PPO/MAPPO 모두 `run_manifest.json`, `metrics.json`, `evaluation_summary.json` 생성
  3. 비교 리포트 자동화: 동일 seed/시나리오에서 PPO vs MAPPO 성능 차이를 테이블로 출력
- **정량 지표**
  - `win_rate`, `mission_time`, `blue_casualties`, `force_reduction`, `stability(std over seeds)`
  - 학습 효율: 동일 wall-clock 대비 수렴 에피소드 수

#### [R8] 멀티도메인 학습 시나리오 배치 실험

- **목표**: 단일 도메인 최적화가 아닌 도메인 전이/일반화 성능을 계량화
- **실험 매트릭스**
  - Train Domain: `ground`, `air`, `naval`, `multi`
  - Eval Domain: `ground`, `air`, `naval`, `multi`
  - Seed: 최소 5개(권장 10개)
- **분석 산출물**
  1. 4x4 일반화 성능 매트릭스(평균 ± 표준편차)
  2. 학습 곡선(에피소드 vs 성능) + 도메인별 수렴 속도
  3. OOD(Out-of-Domain) 성능 저하율: `in-domain 대비 성능 감소 %`

#### [R9] 위험 민감 평가(CVaR 중심) 기반 하이퍼파라미터 탐색

- **목표**: 평균 성능 최대화와 꼬리위험 최소화 간 균형점 탐색
- **탐색 파라미터(우선순위)**
  - `lr`, `clip_range`, `entropy_coef`, `gae_lambda`, `value_coef`
  - 위험 민감 계수(`risk_alpha`, `cvar_quantile`)를 별도 축으로 관리
- **평가 방식**
  1. 각 설정당 다중 seed 롤아웃 수행
  2. 평균 성능 + `CVaR@10` + 최악 10% 구간 평균을 동시 기록
  3. Pareto 프런트(Mean vs CVaR)에서 후보군 선별
- **의사결정 규칙**
  - 운영 배치 후보는 `평균 성능 하위 5% 이내`이면서 `CVaR 개선폭 상위 30%` 조건을 만족해야 함

**완료 기준 (DoD)**
- PPO 대비 MAPPO 이점/한계가 정량 리포트로 도출
- 멀티도메인 일반화 성능 곡선 확보
- 평균 성능과 최악 10% 성능 간 트레이드오프 맵 확보

### DoD 측정 프로토콜(권장)

1. **비교 공정성 확보**: PPO/MAPPO 동일 seed 세트, 동일 시나리오 분포, 동일 평가 budget 사용
2. **통계 신뢰성 확보**: 평균값뿐 아니라 95% CI와 효과크기(Cohen's d) 병기
3. **리스크 관점 반영**: 평균 점수 우위만으로 채택하지 않고 `CVaR@10` 개선 여부를 필수 게이트로 적용
4. **재현성 보장**: 리포트에 `config hash`, `git sha`, `seed list`, `checkpoint path`를 포함

---

## 7. KPI 프레임워크 (권장)

다음 4개 축으로 KPI를 고정하면 향후 개선의 효과를 명확히 확인할 수 있다.

1. **임무 성과**: win_rate, mission_time
2. **비용 효율**: blue_casualties, force_reduction
3. **강건성**: CVaR(5/10), worst_case_casualties
4. **지휘 적합성**: doctrine_compliance, HITL adoption_rate

권장 방식은 각 KPI를 단일 점수로 합치기보다, Pareto 관점으로 병렬 추적하는 것이다.

---

## 8. 최종 결론

FALCON은 현재 **구조적으로는 충분히 성숙했고, 기능적으로도 핵심 폐루프가 작동하는 상태**다. 이제 가장 중요한 것은 아키텍처 재설계가 아니라,

- 보상/평가 의미의 완전 정렬,
- 실험 아티팩트 표준화,
- Pareto/HITL 제안의 실측 기반 강화,

를 통해 **연구 신뢰도와 운영 전개성을 동시에 끌어올리는 것**이다.

즉, v3 단계의 전략은 “크게 바꾸는 것”이 아니라 “지금 있는 강한 구조를 신뢰 가능한 체계로 완성하는 것”이다.

---

## Appendix A. v3 우선 작업 체크리스트

- [ ] Self-Play `prev_blue_hc` 정렬 패치
- [ ] `setup_logger` 핸들러 중복 방지
- [ ] `run_manifest.json` 표준 저장
- [ ] Pareto MC-lite 보정 연결
- [ ] Phase contract 문서 + CI 계약 테스트
- [ ] MAPPO 실험 엔트리 1차 개통
