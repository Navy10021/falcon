"""
web_interface.py
================
HITL 웹 인터페이스 프로토타입 — REST API + HTML 대시보드

지휘관이 웹 브라우저에서 직접:
  1. Pareto 전략 후보를 시각적으로 비교
  2. 자연어로 제약 조건을 입력
  3. 전략을 선택하고 피드백을 제공
  4. 실시간 전투 상황을 모니터링

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

        self._initialized = False

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

        self.strategy_options = []
        for i, opt in enumerate(options):
            risk = "low" if opt.expected_casualties < 30 else "medium" if opt.expected_casualties < 60 else "high"
            self.strategy_options.append(StrategyOption(
                strategy_id=f"strategy_{i}",
                strategy_type=opt.strategy_type,
                win_probability=opt.win_probability,
                expected_casualties=opt.expected_casualties,
                expected_time=opt.expected_time,
                force_required=opt.force_required,
                risk_level=risk,
                description=opt.description,
                is_recommended=(i == 0),
            ))

        self._initialized = True
        return self.get_state()

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

    def get_state(self) -> Dict:
        """현재 세션 상태 반환"""
        return {
            "session_id": self.session_id,
            "initialized": self._initialized,
            "battle_status": self.battle_status.to_dict(),
            "strategy_options": [s.to_dict() for s in self.strategy_options],
            "n_commander_inputs": len(self.commander_inputs),
            "history_length": len(self.history),
        }


# ──────────────────────────────────────────────
# Web App Factory
# ──────────────────────────────────────────────

def create_falcon_app():
    """
    FALCON HITL 웹 앱 생성.

    Flask 기반 REST API + 간단한 HTML 대시보드.
    Flask가 없으면 None 반환.
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
        session = get_or_create_session(session_id)
        result = session.select_strategy(strategy_id, feedback)
        return jsonify(result)

    return app


# ──────────────────────────────────────────────
# Dashboard HTML Template
# ──────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>FALCON HITL Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0e17; color: #e0e0e0; }
        .header { background: #111827; padding: 16px 24px; border-bottom: 2px solid #1e40af; }
        .header h1 { font-size: 22px; color: #60a5fa; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
        .panel { background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; padding: 16px; }
        .panel h2 { color: #93c5fd; font-size: 16px; margin-bottom: 12px; }
        .strategy-card { background: #1e293b; border: 1px solid #334155; border-radius: 6px;
                        padding: 12px; margin-bottom: 8px; cursor: pointer; }
        .strategy-card:hover { border-color: #60a5fa; }
        .strategy-card.recommended { border-color: #22c55e; }
        .metric { display: inline-block; margin-right: 16px; }
        .metric-value { font-size: 20px; font-weight: bold; color: #60a5fa; }
        .metric-label { font-size: 11px; color: #9ca3af; }
        input, textarea { background: #1e293b; border: 1px solid #334155; color: #e0e0e0;
                         padding: 8px 12px; border-radius: 4px; width: 100%; }
        button { background: #1e40af; color: white; border: none; padding: 8px 16px;
                border-radius: 4px; cursor: pointer; margin-top: 8px; }
        button:hover { background: #2563eb; }
        .alert-normal { color: #22c55e; }
        .alert-caution { color: #f59e0b; }
        .alert-critical { color: #ef4444; }
        #response { background: #0f172a; padding: 12px; border-radius: 4px;
                   margin-top: 8px; min-height: 60px; font-size: 13px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FALCON HITL Command Interface</h1>
    </div>
    <div class="container">
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
        <div class="panel">
            <h2>Commander Input</h2>
            <textarea id="command-input" rows="3"
                placeholder="Enter command (e.g., 'Minimize casualties, win rate above 60%')"></textarea>
            <button onclick="sendCommand()">Send Command</button>
            <button onclick="initSession()">New Scenario</button>
            <div id="response"></div>
        </div>
    </div>
    <script>
        async function initSession() {
            const res = await fetch('/api/session/init', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: 'default', n_blue: 8, n_red: 6})
            });
            const data = await res.json();
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
                        <strong>${s.type}</strong> | Win: ${(s.win_probability*100).toFixed(1)}%
                        | Cas: ${s.expected_casualties.toFixed(0)}
                        | Risk: ${s.risk_level}
                        ${s.is_recommended ? ' [Recommended]' : ''}
                    </div>`
                ).join('');
            }
        }
        async function selectStrategy(id) {
            const res = await fetch('/api/strategy/select', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({strategy_id: id})
            });
            const data = await res.json();
            document.getElementById('response').innerText = JSON.stringify(data, null, 2);
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
