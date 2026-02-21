"""
realtime_dashboard.py
=====================
G. 실시간 인터랙티브 전투 대시보드
- 전투 상황을 HTML 파일로 실시간 렌더링
- 유닛 위치 맵, 전력 비교, 자원 현황, AI 추천 전략
- 불확실성 히트맵, 재계획 경보
- Plotly.js 기반 (외부 서버 불필요, 단일 HTML)

학술 연구용 합성 데이터
"""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────
# 대시보드 데이터 모델
# ──────────────────────────────────────────────

@dataclass
class UnitMarker:
    unit_id: str
    x: float
    y: float
    alignment: str      # "blue" | "red"
    unit_type: str
    headcount: int
    status: str         # "active" | "degraded" | "critical" | "destroyed"
    ammo: float
    fuel: float
    combat_power: float

    def color(self) -> str:
        if self.alignment == "blue":
            return {"active": "#1565C0", "degraded": "#1976D2",
                    "critical": "#42A5F5", "destroyed": "#90CAF9"}.get(self.status, "#1565C0")
        return {"active": "#B71C1C", "degraded": "#C62828",
                "critical": "#EF5350", "destroyed": "#EF9A9A"}.get(self.status, "#B71C1C")

    def symbol(self) -> str:
        return {"infantry": "●", "armor": "◆", "artillery": "▲",
                "aviation": "★", "engineer": "■", "signal": "◎",
                "logistics": "⬡", "electronic_warfare": "⊕"}.get(self.unit_type, "●")


@dataclass
class DashboardSnapshot:
    step: int
    max_steps: int
    blue_units: List[UnitMarker]
    red_units: List[UnitMarker]
    blue_win_rate: float
    uncertainty: float
    alert_level: str        # "normal" | "caution" | "warning" | "critical"
    alert_reasons: List[str]
    pareto_options: List[Dict]
    recommended_strategy: str
    combat_log: List[str]   # 최근 전투 이벤트


@dataclass
class BattleHistory:
    """대시보드 전체 히스토리 (애니메이션용)"""
    snapshots: List[DashboardSnapshot] = field(default_factory=list)
    blue_headcount_history: List[int] = field(default_factory=list)
    red_headcount_history: List[int] = field(default_factory=list)
    win_rate_history: List[float] = field(default_factory=list)
    uncertainty_history: List[float] = field(default_factory=list)

    def add(self, snap: DashboardSnapshot):
        self.snapshots.append(snap)
        total_blue = sum(u.headcount for u in snap.blue_units if u.status != "destroyed")
        total_red  = sum(u.headcount for u in snap.red_units  if u.status != "destroyed")
        self.blue_headcount_history.append(total_blue)
        self.red_headcount_history.append(total_red)
        self.win_rate_history.append(snap.blue_win_rate)
        self.uncertainty_history.append(snap.uncertainty)


# ──────────────────────────────────────────────
# KG → DashboardSnapshot 변환
# ──────────────────────────────────────────────

def kg_to_snapshot(
    kg,
    step: int,
    max_steps: int,
    win_rate: float = 0.5,
    uncertainty: float = 0.3,
    alert_level: str = "normal",
    alert_reasons: Optional[List[str]] = None,
    pareto_options: Optional[List[Dict]] = None,
    recommended_strategy: str = "balanced",
    combat_log: Optional[List[str]] = None,
) -> DashboardSnapshot:
    from ontology.combat_schema import ForceAlignment

    def make_marker(unit) -> UnitMarker:
        return UnitMarker(
            unit_id=unit.unit_id,
            x=float(unit.position.x),
            y=float(unit.position.y),
            alignment="blue" if unit.alignment == ForceAlignment.BLUE else "red",
            unit_type=unit.unit_type.value,
            headcount=unit.headcount,
            status=unit.status.value,
            ammo=float(unit.capability.ammo_level),
            fuel=float(unit.capability.fuel_level),
            combat_power=float(unit.combat_power),
        )

    blue_units = [make_marker(u) for u in kg.units.values()
                  if u.alignment == ForceAlignment.BLUE]
    red_units  = [make_marker(u) for u in kg.units.values()
                  if u.alignment != ForceAlignment.BLUE]

    default_options = [
        {"label": "균형", "win_prob": win_rate * 0.95, "force": 900, "casualties": 45},
        {"label": "최소병력", "win_prob": win_rate * 0.80, "force": 650, "casualties": 35},
        {"label": "최대승률", "win_prob": min(win_rate * 1.15, 0.99), "force": 1100, "casualties": 30},
    ]

    return DashboardSnapshot(
        step=step,
        max_steps=max_steps,
        blue_units=blue_units,
        red_units=red_units,
        blue_win_rate=float(win_rate),
        uncertainty=float(uncertainty),
        alert_level=alert_level,
        alert_reasons=alert_reasons or [],
        pareto_options=pareto_options or default_options,
        recommended_strategy=recommended_strategy,
        combat_log=combat_log or [f"스텝 {step}: 교전 진행 중"],
    )


# ──────────────────────────────────────────────
# HTML 대시보드 생성기
# ──────────────────────────────────────────────

class DashboardRenderer:
    """
    Plotly.js 기반 단일 HTML 대시보드 생성
    - 인터랙티브 전술 지도 (유닛 위치)
    - 전력/사상자 추이 그래프
    - 자원 게이지 (탄약/연료)
    - Pareto 전략 비교 레이더 차트
    - 불확실성 히트맵
    - 실시간 갱신 시뮬레이션 (JS 애니메이션)
    """

    ALERT_COLORS = {
        "normal":   ("#4CAF50", "정상"),
        "caution":  ("#FF9800", "주의"),
        "warning":  ("#FF5722", "경고"),
        "critical": ("#F44336", "위기"),
    }

    def render(self, history: BattleHistory, output_path: str = "dashboard.html") -> str:
        """전체 히스토리를 HTML 대시보드로 렌더링"""
        snapshots_json = self._serialize_history(history)
        html = self._build_html(snapshots_json, history)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def render_static(self, snap: DashboardSnapshot, output_path: str = "dashboard_static.html") -> str:
        """단일 스냅샷 정적 HTML"""
        history = BattleHistory()
        history.add(snap)
        return self.render(history, output_path)

    def _serialize_history(self, history: BattleHistory) -> str:
        data = []
        for snap in history.snapshots:
            data.append({
                "step": snap.step,
                "max_steps": snap.max_steps,
                "blue_win_rate": snap.blue_win_rate,
                "uncertainty": snap.uncertainty,
                "alert_level": snap.alert_level,
                "alert_reasons": snap.alert_reasons,
                "recommended_strategy": snap.recommended_strategy,
                "combat_log": snap.combat_log[-5:],  # 최근 5개
                "blue_units": [
                    {"id": u.unit_id, "x": u.x, "y": u.y,
                     "type": u.unit_type, "hc": u.headcount,
                     "status": u.status, "ammo": u.ammo,
                     "fuel": u.fuel, "cp": u.combat_power,
                     "color": u.color(), "symbol": u.symbol()}
                    for u in snap.blue_units
                ],
                "red_units": [
                    {"id": u.unit_id, "x": u.x, "y": u.y,
                     "type": u.unit_type, "hc": u.headcount,
                     "status": u.status, "ammo": u.ammo,
                     "fuel": u.fuel, "cp": u.combat_power,
                     "color": u.color(), "symbol": u.symbol()}
                    for u in snap.red_units
                ],
                "pareto_options": snap.pareto_options,
                "blue_total_hc": sum(u.headcount for u in snap.blue_units
                                      if u.status != "destroyed"),
                "red_total_hc":  sum(u.headcount for u in snap.red_units
                                      if u.status != "destroyed"),
                "blue_avg_ammo": float(np.mean([u.ammo for u in snap.blue_units]) if snap.blue_units else 0),
                "blue_avg_fuel": float(np.mean([u.fuel for u in snap.blue_units]) if snap.blue_units else 0),
            })
        return json.dumps(data, ensure_ascii=False)

    def _build_html(self, snapshots_json: str, history: BattleHistory) -> str:
        hc_blue = history.blue_headcount_history
        hc_red  = history.red_headcount_history
        wr_hist = history.win_rate_history
        unc_hist = history.uncertainty_history
        steps = list(range(len(hc_blue)))

        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎯 AI Combat Optimization — 실시간 전투 대시보드</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0e1a; color: #e0e6f0; font-family: 'Segoe UI', sans-serif; }}

  .header {{
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2a4a 100%);
    padding: 16px 24px;
    border-bottom: 2px solid #1565C0;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .header h1 {{ font-size: 1.4rem; color: #64B5F6; letter-spacing: 1px; }}
  .header .subtitle {{ font-size: 0.85rem; color: #78909C; }}

  .step-control {{
    background: #0d1b2a; padding: 12px 24px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid #1a2a4a;
  }}
  .step-control label {{ color: #90A4AE; font-size: 0.85rem; }}
  #stepSlider {{ flex: 1; max-width: 400px; accent-color: #1565C0; }}
  #stepDisplay {{ color: #64B5F6; font-weight: bold; min-width: 80px; }}

  .alert-bar {{
    padding: 8px 24px; font-size: 0.9rem; font-weight: 600;
    display: flex; align-items: center; gap: 12px;
    transition: background 0.3s;
  }}
  .alert-normal   {{ background: #1B5E20; color: #A5D6A7; }}
  .alert-caution  {{ background: #E65100; color: #FFCC02; }}
  .alert-warning  {{ background: #BF360C; color: #FFAB40; }}
  .alert-critical {{ background: #B71C1C; color: #FF8A80; animation: blink 1s infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0.6; }} }}

  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 12px; padding: 12px;
  }}
  .card {{
    background: #0d1b2a; border: 1px solid #1a2a4a;
    border-radius: 8px; padding: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }}
  .card-title {{
    font-size: 0.78rem; color: #78909C; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #1a2a4a;
    padding-bottom: 6px;
  }}
  .card-wide  {{ grid-column: span 2; }}
  .card-full  {{ grid-column: span 3; }}

  .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
  .stat-label {{ color: #90A4AE; font-size: 0.85rem; }}
  .stat-val {{ font-weight: bold; }}
  .blue-val {{ color: #64B5F6; }}
  .red-val  {{ color: #EF5350; }}

  .gauge-bar {{ background: #1a2a4a; border-radius: 4px; height: 8px; margin: 4px 0; }}
  .gauge-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}

  .strategy-option {{
    background: #0a1628; border: 1px solid #1a2a4a; border-radius: 6px;
    padding: 10px; margin-bottom: 8px; cursor: pointer;
    transition: border-color 0.2s;
  }}
  .strategy-option:hover {{ border-color: #1565C0; }}
  .strategy-option.recommended {{ border-color: #1565C0; background: #0d2040; }}
  .strategy-label {{ font-weight: bold; color: #64B5F6; }}
  .strategy-stats {{ font-size: 0.78rem; color: #78909C; margin-top: 4px; }}
  .strategy-bar {{ background: #1a2a4a; border-radius: 2px; height: 4px; margin-top: 6px; }}
  .strategy-bar-fill {{ background: #1565C0; height: 100%; border-radius: 2px; }}

  .log-entry {{
    font-size: 0.78rem; color: #90A4AE; padding: 3px 0;
    border-bottom: 1px solid #0a1628;
  }}
  .log-entry:last-child {{ border: none; color: #B0BEC5; }}

  .plotly-chart {{ height: 220px; }}
  #tacticalMap {{ height: 340px; }}

  .win-rate-display {{
    font-size: 2.2rem; font-weight: bold;
    text-align: center; padding: 8px 0;
  }}
  .uncertainty-display {{
    text-align: center; font-size: 0.85rem; color: #78909C;
    margin-top: 4px;
  }}

  .btn {{
    background: #1565C0; color: white; border: none;
    padding: 6px 16px; border-radius: 4px; cursor: pointer;
    font-size: 0.85rem; transition: background 0.2s;
  }}
  .btn:hover {{ background: #1976D2; }}
  .btn.playing {{ background: #2E7D32; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🎯 AI Combat Optimization System — 실시간 전투 대시보드</h1>
    <div class="subtitle">합성 전투 데이터 기반 연구용 시각화 | 실제 작전 정보 미포함</div>
  </div>
  <div style="text-align:right; color:#546E7A; font-size:0.8rem;">
    Phase 1~3 통합 시스템<br>Bayesian GNN + MAPPO + HITL
  </div>
</div>

<div class="step-control">
  <label>시뮬레이션 스텝:</label>
  <input type="range" id="stepSlider" min="0" max="{len(history.snapshots)-1}" value="0">
  <span id="stepDisplay">스텝 1 / {len(history.snapshots)}</span>
  <button class="btn" id="playBtn" onclick="togglePlay()">▶ 재생</button>
  <button class="btn" onclick="resetView()">↺ 초기화</button>
</div>

<div class="alert-bar alert-normal" id="alertBar">
  ⬜ 정상 — 시스템 이상 없음
</div>

<div class="grid">

  <!-- 전술 지도 -->
  <div class="card card-wide" style="grid-row: span 2;">
    <div class="card-title">🗺️ 전술 지도 — 유닛 배치 현황</div>
    <div id="tacticalMap"></div>
  </div>

  <!-- 승률 & 불확실성 -->
  <div class="card">
    <div class="card-title">📊 AI 승률 예측</div>
    <div class="win-rate-display blue-val" id="winRateDisplay">--.--%</div>
    <div class="uncertainty-display">불확실성: <span id="uncertaintyDisplay">--</span></div>
    <div style="margin-top: 12px;">
      <div class="gauge-bar">
        <div class="gauge-fill" id="winRateBar" style="width:50%;background:linear-gradient(90deg,#1565C0,#42A5F5);"></div>
      </div>
    </div>
  </div>

  <!-- AI 추천 전략 -->
  <div class="card">
    <div class="card-title">🤖 AI 추천 전략 (Pareto)</div>
    <div id="strategyOptions"></div>
  </div>

  <!-- 전력 추이 -->
  <div class="card card-full">
    <div class="card-title">📈 전력 & 승률 추이</div>
    <div id="trendChart" class="plotly-chart"></div>
  </div>

  <!-- 자원 현황 -->
  <div class="card">
    <div class="card-title">⚡ Blue Force 자원 현황</div>
    <div id="resourcePanel"></div>
  </div>

  <!-- 전투 로그 -->
  <div class="card">
    <div class="card-title">📋 전투 이벤트 로그</div>
    <div id="combatLog"></div>
  </div>

  <!-- 전력 비교 -->
  <div class="card">
    <div class="card-title">⚔️ 전력 비교</div>
    <div id="forceComparison"></div>
  </div>

</div>

<script>
const HISTORY = {snapshots_json};
const STEPS = {steps};
const HC_BLUE = {hc_blue};
const HC_RED = {hc_red};
const WR_HIST = {wr_hist};
const UNC_HIST = {unc_hist};

let currentStep = 0;
let playing = false;
let playInterval = null;

const ALERT_CLASSES = {{
  normal: 'alert-normal',
  caution: 'alert-caution',
  warning: 'alert-warning',
  critical: 'alert-critical'
}};
const ALERT_ICONS = {{
  normal: '⬜ 정상',
  caution: '🟡 주의',
  warning: '🟠 경고',
  critical: '🔴 위기'
}};

function initTrendChart() {{
  const traces = [
    {{
      x: STEPS, y: HC_BLUE,
      name: 'Blue 병력', line: {{color:'#1565C0', width:2}},
      type: 'scatter', yaxis: 'y'
    }},
    {{
      x: STEPS, y: HC_RED,
      name: 'Red 병력', line: {{color:'#B71C1C', width:2}},
      type: 'scatter', yaxis: 'y'
    }},
    {{
      x: STEPS, y: WR_HIST.map(v => v*100),
      name: '승률(%)', line: {{color:'#4CAF50', width:2, dash:'dot'}},
      type: 'scatter', yaxis: 'y2'
    }},
    {{
      x: STEPS, y: UNC_HIST.map(v => v*100),
      name: '불확실성(%)', line: {{color:'#FF9800', width:1.5, dash:'dash'}},
      type: 'scatter', yaxis: 'y2'
    }},
  ];

  const stepLine = {{
    type: 'scatter', x: [currentStep, currentStep],
    y: [0, Math.max(...HC_BLUE, ...HC_RED) * 1.1],
    name: '현재 스텝', line: {{color:'#FFD700', width:1, dash:'dot'}},
    yaxis: 'y', showlegend: false
  }};

  Plotly.newPlot('trendChart', [...traces, stepLine], {{
    paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a1628',
    font: {{color: '#90A4AE', size: 11}},
    margin: {{t:10, b:30, l:50, r:50}},
    legend: {{orientation:'h', y:-0.25, font:{{size:10}}}},
    xaxis: {{gridcolor:'#1a2a4a', title:'스텝'}},
    yaxis: {{gridcolor:'#1a2a4a', title:'병력', side:'left'}},
    yaxis2: {{overlaying:'y', side:'right', title:'%', range:[0,110],
               gridcolor:'#1a2a4a'}},
  }}, {{responsive:true, displayModeBar:false}});
}}

function updateTrendStepLine() {{
  Plotly.restyle('trendChart', {{
    x: [[currentStep, currentStep]],
    y: [[0, Math.max(...HC_BLUE, ...HC_RED) * 1.1]]
  }}, [4]);
}}

function updateTacticalMap(snap) {{
  const blueX = snap.blue_units.map(u => u.x);
  const blueY = snap.blue_units.map(u => u.y);
  const blueText = snap.blue_units.map(u =>
    `${{u.id}}<br>${{u.type}}<br>병력: ${{u.hc}}명<br>탄약: ${{(u.ammo*100).toFixed(0)}}%<br>연료: ${{(u.fuel*100).toFixed(0)}}%`
  );

  const redX = snap.red_units.map(u => u.x);
  const redY = snap.red_units.map(u => u.y);
  const redText = snap.red_units.map(u =>
    `${{u.id}}<br>${{u.type}}<br>병력: ${{u.hc}}명`
  );

  const traces = [
    {{
      x: blueX, y: blueY, text: blueText,
      mode: 'markers+text',
      marker: {{
        color: snap.blue_units.map(u => u.color),
        size: snap.blue_units.map(u => Math.max(10, u.hc/15)),
        symbol: 'circle', line: {{color:'white', width:1}}
      }},
      textfont: {{color:'#90CAF9', size:9}},
      hoverinfo: 'text',
      name: 'Blue Force', type: 'scatter'
    }},
    {{
      x: redX, y: redY, text: redText,
      mode: 'markers',
      marker: {{
        color: snap.red_units.map(u => u.color),
        size: snap.red_units.map(u => Math.max(10, u.hc/15)),
        symbol: 'square', line: {{color:'white', width:1}}
      }},
      hoverinfo: 'text',
      name: 'Red Force', type: 'scatter'
    }},
  ];

  const layout = {{
    paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a1628',
    font: {{color: '#90A4AE'}},
    margin: {{t:10, b:30, l:40, r:10}},
    xaxis: {{range:[-2, 32], gridcolor:'#1a2a4a', title:'X (km)'}},
    yaxis: {{range:[-2, 32], gridcolor:'#1a2a4a', title:'Y (km)'}},
    legend: {{x:0, y:1, bgcolor:'rgba(0,0,0,0.5)', font:{{size:11}}}},
    shapes: [
      // 전선 (대략적)
      {{type:'line', x0:5, y0:0, x1:20, y1:30,
        line:{{color:'#FFD700', width:1, dash:'dot'}}}}
    ],
    annotations: [{{
      x:13, y:15, text:'전선', font:{{color:'#FFD700', size:10}},
      showarrow:false
    }}]
  }};

  if (document.getElementById('tacticalMap')._hasPlotted) {{
    Plotly.react('tacticalMap', traces, layout, {{responsive:true, displayModeBar:false}});
  }} else {{
    Plotly.newPlot('tacticalMap', traces, layout, {{responsive:true, displayModeBar:false}});
    document.getElementById('tacticalMap')._hasPlotted = true;
  }}
}}

function updateUI(stepIdx) {{
  const snap = HISTORY[stepIdx];

  // 스텝 표시
  document.getElementById('stepDisplay').textContent =
    `스텝 ${{snap.step + 1}} / ${{snap.max_steps}}`;
  document.getElementById('stepSlider').value = stepIdx;

  // 경보
  const alertBar = document.getElementById('alertBar');
  alertBar.className = 'alert-bar ' + (ALERT_CLASSES[snap.alert_level] || 'alert-normal');
  const reasons = snap.alert_reasons.length > 0 ? ' — ' + snap.alert_reasons.join(' | ') : '';
  alertBar.textContent = (ALERT_ICONS[snap.alert_level] || '⬜ 정상') + reasons;

  // 승률
  const wrPct = (snap.blue_win_rate * 100).toFixed(1);
  const wrColor = snap.blue_win_rate > 0.6 ? '#64B5F6' :
                  snap.blue_win_rate > 0.4 ? '#FF9800' : '#EF5350';
  document.getElementById('winRateDisplay').style.color = wrColor;
  document.getElementById('winRateDisplay').textContent = wrPct + '%';
  document.getElementById('uncertaintyDisplay').textContent =
    (snap.uncertainty * 100).toFixed(1) + '% 불확실';
  document.getElementById('winRateBar').style.width = wrPct + '%';

  // 전략 옵션
  const optDiv = document.getElementById('strategyOptions');
  optDiv.innerHTML = '';
  (snap.pareto_options || []).forEach((opt, i) => {{
    const isRec = opt.label === snap.recommended_strategy || i === 0;
    const wp = (opt.win_prob * 100).toFixed(1);
    optDiv.innerHTML += `
      <div class="strategy-option ${{isRec ? 'recommended' : ''}}">
        <div class="strategy-label">${{isRec ? '⭐ ' : ''}}${{opt.label}}</div>
        <div class="strategy-stats">
          승률 ${{wp}}% | 병력 ${{opt.force}}명 | 예상사상자 ${{opt.casualties}}명
        </div>
        <div class="strategy-bar">
          <div class="strategy-bar-fill" style="width:${{wp}}%;"></div>
        </div>
      </div>`;
  }});

  // 자원
  const resDiv = document.getElementById('resourcePanel');
  const ammoColor = snap.blue_avg_ammo > 0.4 ? '#4CAF50' :
                    snap.blue_avg_ammo > 0.2 ? '#FF9800' : '#F44336';
  const fuelColor = snap.blue_avg_fuel > 0.4 ? '#4CAF50' :
                    snap.blue_avg_fuel > 0.2 ? '#FF9800' : '#F44336';
  resDiv.innerHTML = `
    <div class="stat-row">
      <span class="stat-label">탄약 평균</span>
      <span style="color:${{ammoColor}}">${{(snap.blue_avg_ammo*100).toFixed(0)}}%</span>
    </div>
    <div class="gauge-bar"><div class="gauge-fill" style="width:${{snap.blue_avg_ammo*100}}%;background:${{ammoColor}};"></div></div>
    <div class="stat-row" style="margin-top:8px;">
      <span class="stat-label">연료 평균</span>
      <span style="color:${{fuelColor}}">${{(snap.blue_avg_fuel*100).toFixed(0)}}%</span>
    </div>
    <div class="gauge-bar"><div class="gauge-fill" style="width:${{snap.blue_avg_fuel*100}}%;background:${{fuelColor}};"></div></div>
    <div class="stat-row" style="margin-top:10px;">
      <span class="stat-label">활성 유닛</span>
      <span class="blue-val">${{snap.blue_units.filter(u=>u.status!='destroyed').length}}개</span>
    </div>`;

  // 전력 비교
  const fcDiv = document.getElementById('forceComparison');
  const totalBlue = snap.blue_total_hc;
  const totalRed  = snap.red_total_hc;
  const total = totalBlue + totalRed;
  const bluePct = total > 0 ? (totalBlue/total*100).toFixed(1) : 50;
  const redPct  = total > 0 ? (totalRed /total*100).toFixed(1) : 50;
  fcDiv.innerHTML = `
    <div class="stat-row">
      <span class="stat-label blue-val">🔵 Blue</span>
      <span class="blue-val" style="font-weight:bold;">${{totalBlue}}명</span>
    </div>
    <div style="background:#1a2a4a;border-radius:4px;height:12px;margin:6px 0;overflow:hidden;display:flex;">
      <div style="width:${{bluePct}}%;background:linear-gradient(90deg,#1565C0,#42A5F5);"></div>
      <div style="width:${{redPct}}%;background:linear-gradient(90deg,#EF5350,#B71C1C);"></div>
    </div>
    <div class="stat-row">
      <span class="stat-label red-val">🔴 Red</span>
      <span class="red-val" style="font-weight:bold;">${{totalRed}}명</span>
    </div>
    <div class="stat-row" style="margin-top:12px;">
      <span class="stat-label">전력비</span>
      <span style="color:${{totalBlue>totalRed?'#64B5F6':'#EF5350'}}">
        ${{totalBlue>totalRed?'Blue 우세':'Red 우세'}} (${{(totalBlue/Math.max(totalRed,1)).toFixed(2)}}:1)
      </span>
    </div>`;

  // 전투 로그
  const logDiv = document.getElementById('combatLog');
  logDiv.innerHTML = (snap.combat_log || []).slice().reverse().map(
    log => `<div class="log-entry">${{log}}</div>`
  ).join('');

  // 지도 + 추이
  updateTacticalMap(snap);
  updateTrendStepLine();
}}

function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('playBtn');
  if (playing) {{
    btn.textContent = '⏸ 일시정지';
    btn.classList.add('playing');
    playInterval = setInterval(() => {{
      if (currentStep < HISTORY.length - 1) {{
        currentStep++;
        updateUI(currentStep);
      }} else {{
        playing = false;
        clearInterval(playInterval);
        btn.textContent = '▶ 재생';
        btn.classList.remove('playing');
      }}
    }}, 800);
  }} else {{
    clearInterval(playInterval);
    btn.textContent = '▶ 재생';
    btn.classList.remove('playing');
  }}
}}

function resetView() {{
  playing = false;
  clearInterval(playInterval);
  document.getElementById('playBtn').textContent = '▶ 재생';
  document.getElementById('playBtn').classList.remove('playing');
  currentStep = 0;
  updateUI(0);
}}

document.getElementById('stepSlider').addEventListener('input', function() {{
  currentStep = parseInt(this.value);
  updateUI(currentStep);
}});

// 초기화
initTrendChart();
updateUI(0);
</script>
</body>
</html>"""


# ──────────────────────────────────────────────
# 시뮬레이션 실행 + 대시보드 생성 헬퍼
# ──────────────────────────────────────────────

def run_and_render(
    n_blue: int = 8,
    n_red: int = 6,
    max_steps: int = 20,
    output_path: str = "dashboard.html",
    seed: int = 42
) -> str:
    """시뮬레이션 실행 후 대시보드 생성"""
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine
    from simulator.resource_manager import ResourceManager

    kg = ScenarioFactory.create_standard_scenario(n_blue=n_blue, n_red=n_red, seed=seed)
    engine = MixedLanchesterEngine(seed=seed)
    rm = ResourceManager(seed=seed)
    renderer = DashboardRenderer()
    history = BattleHistory()
    rng = np.random.RandomState(seed)

    combat_log = []
    win_rate = 0.65
    uncertainty = 0.45

    alert_colors = {0: "normal", 1: "caution", 2: "warning", 3: "critical"}

    for step in range(max_steps):
        # 승률 / 불확실성 업데이트
        result = engine.run_step_mixed(kg, resource_manager=rm)
        blue_hc = result["blue_headcount"]
        red_hc  = result["red_headcount"]
        total = blue_hc + red_hc + 1
        win_rate = float(np.clip(
            (blue_hc / total) * 1.2 + rng.normal(0, 0.03), 0.05, 0.98
        ))
        uncertainty = float(np.clip(
            0.45 - step * 0.015 + rng.normal(0, 0.02), 0.1, 0.8
        ))

        # 경보 수준
        if win_rate < 0.3:
            alert_lvl = "critical"
            reasons = [f"승률 {win_rate:.1%} 위기"]
        elif win_rate < 0.45:
            alert_lvl = "warning"
            reasons = [f"승률 {win_rate:.1%} 경고"]
        elif win_rate < 0.55:
            alert_lvl = "caution"
            reasons = ["승률 주의 구간"]
        else:
            alert_lvl = "normal"
            reasons = []

        # 로그
        b_cas = result["blue_casualties"]
        r_cas = result["red_casualties"]
        if b_cas > 0 or r_cas > 0:
            combat_log.append(
                f"스텝 {step+1}: Blue -{b_cas}명  Red -{r_cas}명  ({result['mission_status']})"
            )

        snap = kg_to_snapshot(
            kg, step=step, max_steps=max_steps,
            win_rate=win_rate,
            uncertainty=uncertainty,
            alert_level=alert_lvl,
            alert_reasons=reasons,
            recommended_strategy="균형",
            combat_log=list(combat_log[-5:]),
        )
        history.add(snap)

        if result["mission_status"] != "ongoing":
            break

    return renderer.render(history, output_path)


if __name__ == "__main__":
    import os
    print("=" * 55)
    print("G. 실시간 대시보드 생성 테스트")
    print("=" * 55)

    out = run_and_render(
        n_blue=8, n_red=6, max_steps=20,
        output_path="/tmp/combat_dashboard.html"
    )
    size = os.path.getsize(out) / 1024
    print(f"\n✅ 대시보드 생성 완료: {out}")
    print(f"   파일 크기: {size:.1f} KB")

    # 단일 스냅샷 테스트
    from ontology.combat_schema import ScenarioFactory
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=4, seed=0)
    snap = kg_to_snapshot(kg, step=0, max_steps=10, win_rate=0.65)
    renderer = DashboardRenderer()
    static_out = renderer.render_static(snap, "/tmp/dashboard_static.html")
    size2 = os.path.getsize(static_out) / 1024
    print(f"✅ 정적 대시보드: {static_out} ({size2:.1f} KB)")
    print("\n✅ 실시간 대시보드 정상 동작!")
