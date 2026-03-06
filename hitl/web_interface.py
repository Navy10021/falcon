"""
web_interface.py
================
HITL 웹 인터페이스 프로토타입 — REST API + HTML 대시보드

지휘관이 웹 브라우저에서 직접:
  1. Pareto 전략 후보를 시각적으로 비교
  2. 자연어로 제약 조건을 입력
  3. 전략을 선택하고 피드백을 제공
  4. 실시간 전투 상황을 모니터링
  5. [R14] 제약 조건을 변경하여 Pareto 후보를 재생성

할 수 있는 웹 인터페이스 프로토타입.

실행::

    python -m hitl.web_interface --port 8050

또는 프로그래밍 방식::

    app = create_falcon_app()
    serve(app, host="0.0.0.0", port=8050)

학술 연구용 합성 데이터
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────
# API Data Models
# ──────────────────────────────────────────────

@dataclass
class StrategyOption:
    """웹 UI에 표시할 전략 옵션"""
    strategy_id: str
    strategy_type: str
    win_probability: float
    expected_casualties: float
    expected_time: float
    force_required: int
    risk_level: str         # "low" | "medium" | "high"
    description: str
    is_recommended: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.strategy_id,
            "type": self.strategy_type,
            "win_probability": round(self.win_probability, 3),
            "expected_casualties": round(self.expected_casualties, 1),
            "expected_time": round(self.expected_time, 1),
            "force_required": self.force_required,
            "risk_level": self.risk_level,
            "description": self.description,
            "is_recommended": self.is_recommended,
        }


@dataclass
class CommanderInput:
    """지휘관 입력 데이터"""
    command_text: str = ""
    selected_strategy_id: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[str] = None
    satisfaction_score: Optional[float] = None  # 0-1


@dataclass
class BattleStatus:
    """현재 전투 상황"""
    step: int = 0
    max_steps: int = 50
    win_rate: float = 0.5
    blue_headcount: int = 0
    red_headcount: int = 0
    blue_casualties: int = 0
    red_casualties: int = 0
    alert_level: str = "normal"  # "normal" | "caution" | "critical"
    mission_status: str = "ongoing"

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "win_rate": round(self.win_rate, 3),
            "blue_headcount": self.blue_headcount,
            "red_headcount": self.red_headcount,
            "blue_casualties": self.blue_casualties,
            "red_casualties": self.red_casualties,
            "alert_level": self.alert_level,
            "mission_status": self.mission_status,
            "progress_pct": round(self.step / max(self.max_steps, 1) * 100, 1),
        }


# ──────────────────────────────────────────────
# Helper: ParetoGenerator StrategyOption → Web StrategyOption 변환
# ──────────────────────────────────────────────

def _pareto_opt_to_web(opt, index: int) -> StrategyOption:
    """hitl.pareto_generator.StrategyOption → web StrategyOption 변환"""
    cas = getattr(opt, "expected_casualties", 0)
    risk = "low" if cas < 30 else "medium" if cas < 60 else "high"
    strategy_type = opt.strategy_type
    if hasattr(strategy_type, "value"):
        strategy_type = strategy_type.value
    return StrategyOption(
        strategy_id=f"strategy_{index}",
        strategy_type=str(strategy_type),
        win_probability=getattr(opt, "win_probability", 0.5),
        expected_casualties=float(cas),
        expected_time=float(getattr(opt, "expected_time_steps", 30)),
        force_required=int(getattr(opt, "force_size", 0)),
        risk_level=getattr(opt, "risk_level", risk),
        description=getattr(opt, "trade_offs", getattr(opt, "label", "")),
        is_recommended=(index == 0),
    )


# ──────────────────────────────────────────────
# HITL Session Manager
# ──────────────────────────────────────────────

class HITLSession:
    """
    단일 HITL 세션 관리.

    지휘관 한 명의 인터랙션 상태를 추적한다.
    """

    def __init__(self, session_id: str, seed: int = 42):
        self.session_id = session_id
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.battle_status = BattleStatus()
        self.strategy_options: List[StrategyOption] = []
        self.commander_inputs: List[CommanderInput] = []
        self.history: List[Dict] = []
        self.active_constraints: Dict[str, Any] = {}

        self._initialized = False
        self.kg = None

    def initialize(self, n_blue: int = 8, n_red: int = 6):
        """시나리오 초기화 및 Pareto 후보 생성"""
        from ontology.combat_schema import ScenarioFactory, ForceAlignment
        from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints

        self.kg = ScenarioFactory.create_standard_scenario(
            n_blue=n_blue, n_red=n_red, seed=self.seed
        )

        blue_hc = sum(
            u.headcount for u in self.kg.units.values()
            if u.alignment == ForceAlignment.BLUE
        )
        red_hc = sum(
            u.headcount for u in self.kg.units.values()
            if u.alignment == ForceAlignment.RED
        )

        self.battle_status.blue_headcount = blue_hc
        self.battle_status.red_headcount = red_hc

        # Pareto 후보 생성
        gen = ParetoStrategyGenerator(n_candidates=5, mc_eval_runs=3)
        constraints = CommanderConstraints()
        options = gen.generate(self.kg, constraints)

        self.strategy_options = [
            _pareto_opt_to_web(opt, i) for i, opt in enumerate(options)
        ]

        self._initialized = True
        return self.get_state()

    def update_constraints(self, constraints_dict: Dict[str, Any]) -> Dict:
        """
        R14: 지휘관 제약 조건 업데이트 → Pareto 후보 재생성.

        Parameters
        ----------
        constraints_dict : dict
            {
                "max_force_size": int or None,
                "min_win_probability": float (0-1),
                "max_casualties": int or None,
                "max_time_steps": int,
                "prefer_flanking": bool,
                "prefer_artillery": bool,
                "avoid_urban": bool,
            }
        """
        from hitl.pareto_generator import ParetoStrategyGenerator, CommanderConstraints

        if self.kg is None:
            return {"error": "Session not initialized. Call /api/session/init first."}

        self.active_constraints = constraints_dict

        # CommanderConstraints 구성
        cc = CommanderConstraints(
            max_force_size=constraints_dict.get("max_force_size"),
            min_win_probability=constraints_dict.get("min_win_probability", 0.5),
            max_casualties=constraints_dict.get("max_casualties"),
            max_time_steps=constraints_dict.get("max_time_steps", 50),
            prefer_flanking=constraints_dict.get("prefer_flanking", False),
            prefer_artillery=constraints_dict.get("prefer_artillery", False),
            avoid_urban=constraints_dict.get("avoid_urban", False),
        )

        gen = ParetoStrategyGenerator(n_candidates=5, mc_eval_runs=3)
        options = gen.generate(self.kg, cc)

        self.strategy_options = [
            _pareto_opt_to_web(opt, i) for i, opt in enumerate(options)
        ]

        self.history.append({
            "type": "constraint_update",
            "constraints": constraints_dict,
            "n_options": len(self.strategy_options),
        })

        cmd_input = CommanderInput(
            command_text="[constraint_update]",
            constraints=constraints_dict,
        )
        self.commander_inputs.append(cmd_input)

        return {
            "constraints_applied": constraints_dict,
            "strategy_options": [s.to_dict() for s in self.strategy_options],
            "n_options": len(self.strategy_options),
            "all_violated": gen.last_generation_all_violated,
            "note": gen.last_generation_note,
        }

    def process_command(self, command_text: str) -> Dict:
        """자연어 명령 처리"""
        from hitl.natural_language_interface import CommandInterface

        interface = CommandInterface()
        parsed = interface.process(command_text)
        cc = parsed.to_commander_constraints()

        cmd_input = CommanderInput(
            command_text=command_text,
            constraints={
                "max_force_size": cc.max_force_size if cc else None,
                "max_casualties": cc.max_casualties if cc else None,
                "min_win_probability": cc.min_win_probability if cc else None,
            },
        )
        self.commander_inputs.append(cmd_input)

        return {
            "intent": parsed.intent.value,
            "constraints": cmd_input.constraints,
            "confidence": parsed.overall_confidence,
            "needs_clarification": parsed.needs_clarification,
            "clarification_questions": [
                c.clarification_question for c in parsed.constraints
                if c.needs_clarification
            ],
            "summary": parsed.summary(),
        }

    def select_strategy(self, strategy_id: str, feedback: str = "") -> Dict:
        """전략 선택 및 피드백 기록"""
        selected = next(
            (s for s in self.strategy_options if s.strategy_id == strategy_id),
            None,
        )
        if selected is None:
            return {"error": f"Strategy '{strategy_id}' not found"}

        cmd_input = CommanderInput(
            selected_strategy_id=strategy_id,
            feedback=feedback,
        )
        self.commander_inputs.append(cmd_input)

        self.history.append({
            "type": "strategy_selection",
            "strategy_id": strategy_id,
            "strategy_type": selected.strategy_type,
            "feedback": feedback,
        })

        return {
            "selected": selected.to_dict(),
            "message": f"'{selected.strategy_type}' 전략이 선택되었습니다.",
        }

    def get_pareto_data(self) -> List[Dict]:
        """R14: Pareto 시각화용 데이터 (scatter plot)"""
        return [
            {
                "id": s.strategy_id,
                "type": s.strategy_type,
                "x": s.win_probability,  # x축: 승률
                "y": s.expected_casualties,  # y축: 사상자
                "size": s.force_required,  # 버블 크기: 병력
                "time": s.expected_time,
                "risk": s.risk_level,
                "recommended": s.is_recommended,
            }
            for s in self.strategy_options
        ]

    def get_state(self) -> Dict:
        """현재 세션 상태 반환"""
        return {
            "session_id": self.session_id,
            "initialized": self._initialized,
            "battle_status": self.battle_status.to_dict(),
            "strategy_options": [s.to_dict() for s in self.strategy_options],
            "pareto_data": self.get_pareto_data(),
            "active_constraints": self.active_constraints,
            "n_commander_inputs": len(self.commander_inputs),
            "history_length": len(self.history),
        }


# ──────────────────────────────────────────────
# Web App Factory
# ──────────────────────────────────────────────

def create_falcon_app(bridge=None):
    """
    FALCON HITL 웹 앱 생성.

    Flask 기반 REST API + 간단한 HTML 대시보드.
    Flask가 없으면 None 반환.

    Parameters
    ----------
    bridge : HITLSessionBridge, optional
        V8: Phase 3 학습 루프와의 실시간 연동 브리지.
        None이면 독립 실행 모드 (브리지 없이 동작).
    """
    try:
        from flask import Flask, jsonify, request, render_template_string
    except ImportError:
        return None

    app = Flask(__name__)
    sessions: Dict[str, HITLSession] = {}

    def get_or_create_session(session_id: str = "default") -> HITLSession:
        if session_id not in sessions:
            sessions[session_id] = HITLSession(session_id)
        return sessions[session_id]

    @app.route("/")
    def index():
        return render_template_string(_DASHBOARD_HTML)

    @app.route("/api/session/init", methods=["POST"])
    def init_session():
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        n_blue = data.get("n_blue", 8)
        n_red = data.get("n_red", 6)
        session = get_or_create_session(session_id)
        result = session.initialize(n_blue=n_blue, n_red=n_red)
        return jsonify(result)

    @app.route("/api/session/state")
    def session_state():
        session_id = request.args.get("session_id", "default")
        session = get_or_create_session(session_id)
        return jsonify(session.get_state())

    @app.route("/api/command", methods=["POST"])
    def process_command():
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        command = data.get("command", "")
        session = get_or_create_session(session_id)
        result = session.process_command(command)
        return jsonify(result)

    @app.route("/api/strategy/select", methods=["POST"])
    def select_strategy():
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        strategy_id = data.get("strategy_id", "")
        feedback = data.get("feedback", "")
        feedback_rating = int(data.get("feedback_rating", 5))
        session = get_or_create_session(session_id)
        result = session.select_strategy(strategy_id, feedback)

        # V8: 브리지가 연결된 경우 Phase 3 루프에 선택 이벤트 전달
        if bridge is not None:
            ep_idx = data.get("episode_idx", bridge._current_episode)
            opt_index = data.get("option_index", 0)
            bridge.put_selection(
                episode_idx=ep_idx,
                strategy_id=strategy_id,
                option_index=opt_index,
                feedback_rating=feedback_rating,
                feedback_text=feedback,
            )
            result["bridge_notified"] = True

        return jsonify(result)

    @app.route("/api/constraints/update", methods=["POST"])
    def update_constraints():
        """R14: 제약 조건 업데이트 → Pareto 후보 재생성"""
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")
        constraints = data.get("constraints", {})
        session = get_or_create_session(session_id)
        result = session.update_constraints(constraints)
        return jsonify(result)

    @app.route("/api/pareto/data")
    def pareto_data():
        """R14: Pareto 시각화 데이터"""
        session_id = request.args.get("session_id", "default")
        session = get_or_create_session(session_id)
        return jsonify({"pareto": session.get_pareto_data()})

    # ── V8: 브리지 전용 엔드포인트 ──────────────────────────────

    @app.route("/api/bridge/status")
    def bridge_status():
        """V8: Phase 3 학습 루프와의 브리지 상태 조회"""
        if bridge is None:
            return jsonify({"bridge": None, "mode": "standalone"})
        return jsonify({"bridge": bridge.status(), "mode": "live"})

    @app.route("/api/bridge/options")
    def bridge_options():
        """V8: Phase 3에서 전달된 최신 Pareto 후보 조회 (폴링용)"""
        if bridge is None:
            return jsonify({"options": None, "episode_idx": -1})
        event = bridge.get_options_nowait()
        if event is None:
            return jsonify({"options": None, "episode_idx": bridge._current_episode})
        return jsonify({
            "episode_idx": event.episode_idx,
            "options": [_pareto_opt_to_web(o, i).to_dict()
                        for i, o in enumerate(event.ranked_options)],
        })

    return app


# ──────────────────────────────────────────────
# Dashboard HTML Template (R14 Enhanced)
# ──────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>FALCON HITL Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0e17; color: #e0e0e0; }
        .header { background: #111827; padding: 16px 24px; border-bottom: 2px solid #1e40af;
                  display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 22px; color: #60a5fa; }
        .header .badge { background: #1e40af; padding: 4px 10px; border-radius: 12px;
                        font-size: 11px; color: #93c5fd; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
        .full-width { grid-column: 1 / -1; }
        .panel { background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; padding: 16px; }
        .panel h2 { color: #93c5fd; font-size: 16px; margin-bottom: 12px; }
        .strategy-card { background: #1e293b; border: 1px solid #334155; border-radius: 6px;
                        padding: 12px; margin-bottom: 8px; cursor: pointer; transition: all 0.2s; }
        .strategy-card:hover { border-color: #60a5fa; transform: translateX(4px); }
        .strategy-card.recommended { border-color: #22c55e; border-width: 2px; }
        .strategy-card.selected { border-color: #f59e0b; background: #1e293b; }
        .metric { display: inline-block; margin-right: 16px; }
        .metric-value { font-size: 20px; font-weight: bold; color: #60a5fa; }
        .metric-label { font-size: 11px; color: #9ca3af; }
        input, textarea, select { background: #1e293b; border: 1px solid #334155; color: #e0e0e0;
                         padding: 8px 12px; border-radius: 4px; width: 100%; margin-bottom: 8px; }
        input[type="number"] { width: 120px; }
        input[type="checkbox"] { width: auto; margin-right: 6px; }
        button { background: #1e40af; color: white; border: none; padding: 8px 16px;
                border-radius: 4px; cursor: pointer; margin-top: 4px; margin-right: 8px; }
        button:hover { background: #2563eb; }
        button.secondary { background: #374151; }
        button.secondary:hover { background: #4b5563; }
        .alert-normal { color: #22c55e; }
        .alert-caution { color: #f59e0b; }
        .alert-critical { color: #ef4444; }
        #response { background: #0f172a; padding: 12px; border-radius: 4px;
                   margin-top: 8px; min-height: 60px; font-size: 13px; white-space: pre-wrap; }
        .constraint-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
        .constraint-row label { min-width: 140px; font-size: 13px; color: #9ca3af; }
        .constraint-row input { margin-bottom: 0; }
        #pareto-canvas { width: 100%; height: 280px; background: #0f172a; border-radius: 6px;
                        border: 1px solid #1e3a5f; position: relative; }
        .pareto-dot { position: absolute; border-radius: 50%; cursor: pointer;
                     transition: transform 0.2s; opacity: 0.85; }
        .pareto-dot:hover { transform: scale(1.4); opacity: 1; }
        .pareto-dot.recommended { border: 2px solid #22c55e; }
        .axis-label { position: absolute; font-size: 10px; color: #6b7280; }
        .pareto-tooltip { position: absolute; background: #1e293b; border: 1px solid #334155;
                         padding: 8px 10px; border-radius: 4px; font-size: 11px;
                         pointer-events: none; display: none; z-index: 10; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FALCON HITL Command Interface</h1>
        <span class="badge">R14 Pareto Selection</span>
    </div>
    <div class="container">
        <!-- Left: Battle Status + Strategies -->
        <div class="panel">
            <h2>Battle Status</h2>
            <div id="status">
                <div class="metric"><span class="metric-value" id="win-rate">--</span>
                    <br><span class="metric-label">Win Rate</span></div>
                <div class="metric"><span class="metric-value" id="blue-hc">--</span>
                    <br><span class="metric-label">Blue Force</span></div>
                <div class="metric"><span class="metric-value" id="red-hc">--</span>
                    <br><span class="metric-label">Red Force</span></div>
                <div class="metric"><span class="metric-value" id="alert-level">--</span>
                    <br><span class="metric-label">Alert</span></div>
            </div>
            <h2 style="margin-top:16px">Strategy Options</h2>
            <div id="strategies"></div>
        </div>

        <!-- Right: Constraints + Command -->
        <div class="panel">
            <h2>Commander Constraints</h2>
            <div class="constraint-row">
                <label>Max Force Size:</label>
                <input type="number" id="c-max-force" placeholder="(no limit)" min="10" step="10">
            </div>
            <div class="constraint-row">
                <label>Min Win Probability:</label>
                <input type="number" id="c-min-win" value="0.50" min="0" max="1" step="0.05">
            </div>
            <div class="constraint-row">
                <label>Max Casualties:</label>
                <input type="number" id="c-max-cas" placeholder="(no limit)" min="0" step="5">
            </div>
            <div class="constraint-row">
                <label>Max Time Steps:</label>
                <input type="number" id="c-max-time" value="50" min="10" step="5">
            </div>
            <div class="constraint-row">
                <label>
                    <input type="checkbox" id="c-flanking"> Prefer Flanking
                </label>
                <label>
                    <input type="checkbox" id="c-artillery"> Prefer Artillery
                </label>
                <label>
                    <input type="checkbox" id="c-no-urban"> Avoid Urban
                </label>
            </div>
            <button onclick="applyConstraints()">Apply Constraints</button>
            <button class="secondary" onclick="resetConstraints()">Reset</button>

            <h2 style="margin-top:16px">Natural Language Command</h2>
            <textarea id="command-input" rows="2"
                placeholder="Enter command (e.g., 'Minimize casualties, win rate above 60%')"></textarea>
            <button onclick="sendCommand()">Send Command</button>
            <button class="secondary" onclick="initSession()">New Scenario</button>
            <div id="response"></div>
        </div>

        <!-- Full-width: Pareto Scatter -->
        <div class="panel full-width">
            <h2>Pareto Front Visualization</h2>
            <div id="pareto-canvas">
                <span class="axis-label" style="bottom:4px;left:50%;">Win Probability &rarr;</span>
                <span class="axis-label" style="top:50%;left:4px;transform:rotate(-90deg) translateX(-50%);
                      transform-origin:left center;">Casualties &darr;</span>
                <div class="pareto-tooltip" id="pareto-tooltip"></div>
            </div>
        </div>
    </div>
    <script>
        let currentData = null;

        async function initSession() {
            const res = await fetch('/api/session/init', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: 'default', n_blue: 8, n_red: 6})
            });
            const data = await res.json();
            currentData = data;
            updateUI(data);
        }
        async function sendCommand() {
            const cmd = document.getElementById('command-input').value;
            const res = await fetch('/api/command', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            });
            const data = await res.json();
            document.getElementById('response').innerText = JSON.stringify(data, null, 2);
        }
        async function applyConstraints() {
            const constraints = {};
            const maxForce = document.getElementById('c-max-force').value;
            if (maxForce) constraints.max_force_size = parseInt(maxForce);
            const minWin = document.getElementById('c-min-win').value;
            if (minWin) constraints.min_win_probability = parseFloat(minWin);
            const maxCas = document.getElementById('c-max-cas').value;
            if (maxCas) constraints.max_casualties = parseInt(maxCas);
            const maxTime = document.getElementById('c-max-time').value;
            if (maxTime) constraints.max_time_steps = parseInt(maxTime);
            constraints.prefer_flanking = document.getElementById('c-flanking').checked;
            constraints.prefer_artillery = document.getElementById('c-artillery').checked;
            constraints.avoid_urban = document.getElementById('c-no-urban').checked;

            const res = await fetch('/api/constraints/update', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({constraints: constraints})
            });
            const data = await res.json();
            document.getElementById('response').innerText =
                (data.note ? data.note + '\\n\\n' : '') + JSON.stringify(data, null, 2);
            // Refresh state
            const stateRes = await fetch('/api/session/state');
            const stateData = await stateRes.json();
            currentData = stateData;
            updateUI(stateData);
        }
        function resetConstraints() {
            document.getElementById('c-max-force').value = '';
            document.getElementById('c-min-win').value = '0.50';
            document.getElementById('c-max-cas').value = '';
            document.getElementById('c-max-time').value = '50';
            document.getElementById('c-flanking').checked = false;
            document.getElementById('c-artillery').checked = false;
            document.getElementById('c-no-urban').checked = false;
            applyConstraints();
        }
        function updateUI(data) {
            if (data.battle_status) {
                const bs = data.battle_status;
                document.getElementById('win-rate').innerText = (bs.win_rate * 100).toFixed(1) + '%';
                document.getElementById('blue-hc').innerText = bs.blue_headcount;
                document.getElementById('red-hc').innerText = bs.red_headcount;
                const al = document.getElementById('alert-level');
                al.innerText = bs.alert_level.toUpperCase();
                al.className = 'metric-value alert-' + bs.alert_level;
            }
            if (data.strategy_options) {
                const container = document.getElementById('strategies');
                container.innerHTML = data.strategy_options.map(s =>
                    `<div class="strategy-card ${s.is_recommended ? 'recommended' : ''}"
                          onclick="selectStrategy('${s.id}')">
                        <strong>${s.type}</strong> |
                        Win: ${(s.win_probability*100).toFixed(1)}% |
                        Cas: ${s.expected_casualties.toFixed(0)} |
                        Force: ${s.force_required} |
                        Risk: ${s.risk_level}
                        ${s.is_recommended ? ' [Recommended]' : ''}
                        <div style="font-size:11px;color:#9ca3af;margin-top:4px">${s.description}</div>
                    </div>`
                ).join('');
            }
            if (data.pareto_data) {
                renderPareto(data.pareto_data);
            }
        }
        function renderPareto(points) {
            const canvas = document.getElementById('pareto-canvas');
            // Clear old dots
            canvas.querySelectorAll('.pareto-dot').forEach(d => d.remove());
            if (!points || points.length === 0) return;

            const W = canvas.offsetWidth - 60;
            const H = canvas.offsetHeight - 40;
            const pad = {left: 40, top: 10, right: 20, bottom: 30};

            const xMin = Math.min(...points.map(p => p.x)) - 0.05;
            const xMax = Math.max(...points.map(p => p.x)) + 0.05;
            const yMin = 0;
            const yMax = Math.max(...points.map(p => p.y)) * 1.2 + 1;

            const colors = {low: '#22c55e', medium: '#f59e0b', high: '#ef4444'};

            points.forEach(p => {
                const px = pad.left + ((p.x - xMin) / (xMax - xMin)) * W;
                const py = pad.top + (1 - (p.y - yMin) / (yMax - yMin)) * H;
                const sz = Math.max(14, Math.min(36, p.size / 20));

                const dot = document.createElement('div');
                dot.className = 'pareto-dot' + (p.recommended ? ' recommended' : '');
                dot.style.left = px + 'px';
                dot.style.top = py + 'px';
                dot.style.width = sz + 'px';
                dot.style.height = sz + 'px';
                dot.style.background = colors[p.risk] || '#60a5fa';
                dot.style.marginLeft = (-sz/2) + 'px';
                dot.style.marginTop = (-sz/2) + 'px';

                dot.onmouseenter = (e) => {
                    const tt = document.getElementById('pareto-tooltip');
                    tt.style.display = 'block';
                    tt.style.left = (px + 20) + 'px';
                    tt.style.top = (py - 20) + 'px';
                    tt.innerHTML = `<strong>${p.type}</strong><br>
                        Win: ${(p.x*100).toFixed(1)}% | Cas: ${p.y.toFixed(0)}<br>
                        Force: ${p.size} | Time: ${p.time} steps`;
                };
                dot.onmouseleave = () => {
                    document.getElementById('pareto-tooltip').style.display = 'none';
                };
                dot.onclick = () => selectStrategy(p.id);

                canvas.appendChild(dot);
            });
        }
        async function selectStrategy(id) {
            const res = await fetch('/api/strategy/select', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({strategy_id: id})
            });
            const data = await res.json();
            document.getElementById('response').innerText = JSON.stringify(data, null, 2);
            // Highlight selected card
            document.querySelectorAll('.strategy-card').forEach(c => c.classList.remove('selected'));
        }
        initSession();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FALCON HITL Web Interface")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    app = create_falcon_app()
    if app is None:
        print("Flask not installed. Install with: pip install flask")
        print("Running in headless mode (API data model test only)...")
        session = HITLSession("test", seed=42)
        state = session.initialize()
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        print(f"Starting FALCON HITL at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=True)
