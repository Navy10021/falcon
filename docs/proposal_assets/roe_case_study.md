# ROE 차단 + HITL 선호 변경 사례 연구 (PR-4)

## 1) 시나리오 개요
- **Scenario**: `urban_defense`
- **Seed**: `42`
- **정책**: `rule`
- **목표**: 동일 조건에서 ROE on/off 및 HITL 선호(`loss_min` vs `time_min`) 변경이 산출물에 어떻게 반영되는지 검증

## 2) 재현 명령어
```bash
python -m demo.demo --scenario urban_defense --seed 42 --out runs/pr4_roe_on_loss --roe on  --preference loss_min
python -m demo.demo --scenario urban_defense --seed 42 --out runs/pr4_roe_off_time --roe off --preference time_min
```

## 3) 전/후 비교 (Before/After)

### A. ROE 차단 이벤트 가시화
- **Before (ROE off)**: `metrics.csv` 이벤트가 `ROE_DISABLED`, 차단 이벤트 없음
- **After (ROE on)**: Episode 3~5에서 `ROE_BLOCKED` 발생, `selected_action`이 `assault -> hold`로 대체

`runs/pr4_roe_on_loss/metrics.csv` 발췌:
```csv
3,failure,...,ROE_BLOCKED,민간 피해 위험도(0.45)가 임계치(0.45) 이상이어서 assault를 차단했습니다.,hold,...
4,success,...,ROE_BLOCKED,민간 피해 위험도(0.51)가 임계치(0.45) 이상이어서 assault를 차단했습니다.,hold,...
5,success,...,ROE_BLOCKED,민간 피해 위험도(0.57)가 임계치(0.45) 이상이어서 assault를 차단했습니다.,hold,...
```

### B. HITL preference 전환에 따른 tradeoff 변화
- **ROE on + loss_min (`runs/pr4_roe_on_loss/summary.json`)**
  - `loss_tradeoff`: **0.3144**
  - `time_tradeoff`: **0.5375**
- **ROE off + time_min (`runs/pr4_roe_off_time/summary.json`)**
  - `loss_tradeoff`: **0.3099**
  - `time_tradeoff`: **0.5285**

> 동일 seed에서도 preference 가중치가 달라지면서 summary의 loss/time 지표가 함께 변함.

## 4) 산출물 근거 위치 (고정 경로)
- `runs/pr4_roe_on_loss/summary.json`
- `runs/pr4_roe_on_loss/metrics.csv`
- `runs/pr4_roe_on_loss/fig_episode.png`
- `runs/pr4_roe_on_loss/aar.html`
- `runs/pr4_roe_off_time/summary.json`
- `runs/pr4_roe_off_time/metrics.csv`
- `runs/pr4_roe_off_time/fig_episode.png`
- `runs/pr4_roe_off_time/aar.html`

## 5) 해석 요약
- ROE가 켜지면 위험 임계치 이상의 assault가 차단되고 `ROE_BLOCKED` 이벤트/근거 문구가 `metrics.csv`와 `aar.html`에 동시에 기록된다.
- 지휘관 선호를 `loss_min` ↔ `time_min`으로 바꾸면 summary의 `loss_tradeoff`/`time_tradeoff`가 달라져 의사결정 성향 변화가 정량적으로 확인된다.
