"""
generate_data.py
================
온톨로지 + 시뮬레이션 데이터 생성 스크립트

Step 1.5: demo.py 실행 이후, train.py 이전 반드시 실행
  - 다양한 전술 시나리오 합성 생성 (Knowledge Graph)
  - 시뮬레이션 롤아웃 에피소드 생성 (RL 학습용)
  - IRL 전문가 데모 생성 (Inverse RL 학습용)
  - 데이터 통계 리포트 및 시각화 HTML 출력

사용법:
  python generate_data.py                   # 기본 (100 시나리오)
  python generate_data.py --scenarios 500   # 대규모
  python generate_data.py --quick           # 빠른 검증 (10개)
  python generate_data.py --visualize       # 통계 시각화 포함

학술 연구용 합성 데이터
"""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np


# ── 데이터 생성 설정 ─────────────────────────

@dataclass
class GenerationConfig:
    n_scenarios: int = 100          # 시나리오 수
    n_blue_range: tuple = (4, 12)   # Blue 유닛 범위
    n_red_range:  tuple = (3, 10)   # Red 유닛 범위
    episode_steps: int = 20         # 에피소드 최대 스텝
    n_irl_demos:   int = 50         # IRL 데모 수 per 전술
    seed: int = 42
    output_dir: str = "data"
    visualize: bool = True


# ── 시나리오 레지스트리 ──────────────────────

SCENARIO_TEMPLATES = [
    {"name": "assault",         "blue": (6, 12), "red": (4, 8),  "label": "정면 공격"},
    {"name": "defense",         "blue": (4, 8),  "red": (6, 12), "label": "방어 작전"},
    {"name": "flanking",        "blue": (5, 10), "red": (5, 9),  "label": "측면 기동"},
    {"name": "encirclement",    "blue": (8, 12), "red": (4, 7),  "label": "포위 섬멸"},
    {"name": "delay",           "blue": (3, 6),  "red": (8, 12), "label": "지연 방어"},
    {"name": "coastal_landing", "blue": (6, 10), "red": (5, 9),  "label": "상륙 작전"},
    {"name": "urban_combat",    "blue": (5, 9),  "red": (5, 8),  "label": "시가지 전투"},
    {"name": "mountain_ops",    "blue": (4, 8),  "red": (4, 7),  "label": "산악 작전"},
]


# ── 데이터 생성기 ────────────────────────────

class DataGenerator:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

    # ── 시나리오 생성 ─────────────────────────

    def generate_scenarios(self) -> List[Dict]:
        from ontology.combat_schema import ScenarioFactory, ForceAlignment, UnitStatus

        print(f"\n[1/4] 전장 시나리오 생성 ({self.cfg.n_scenarios}개)...")
        scenarios = []
        templates = SCENARIO_TEMPLATES

        for i in range(self.cfg.n_scenarios):
            tmpl = templates[i % len(templates)]
            n_blue = int(self.rng.randint(*tmpl["blue"]))
            n_red  = int(self.rng.randint(*tmpl["red"]))
            seed   = int(self.rng.randint(0, 99999))

            kg = ScenarioFactory.create_standard_scenario(
                n_blue=n_blue, n_red=n_red, seed=seed
            )
            stats = kg.get_stats()

            # 유닛 상세 데이터 추출
            units_data = []
            for uid, u in kg.units.items():
                units_data.append({
                    "id": uid,
                    "alignment": u.alignment.value,
                    "type": u.unit_type.value,
                    "headcount": u.headcount,
                    "combat_power": round(u.combat_power, 4),
                    "status": u.status.value,
                    "position": {"x": round(u.position.x, 2), "y": round(u.position.y, 2)},
                    "capability": {
                        "firepower":    round(u.capability.firepower, 3),
                        "mobility":     round(u.capability.mobility, 3),
                        "protection":   round(u.capability.protection, 3),
                        "range_km":     round(u.capability.range_km, 2),
                        "ammo_level":   round(u.capability.ammo_level, 3),
                        "fuel_level":   round(u.capability.fuel_level, 3),
                        "comms_quality":round(u.capability.comms_quality, 3),
                        "ew_resistance":round(u.capability.ew_resistance, 3),
                    }
                })

            scenario = {
                "id": f"scenario_{i:04d}",
                "template": tmpl["name"],
                "label": tmpl["label"],
                "seed": seed,
                "n_blue": n_blue,
                "n_red": n_red,
                "stats": stats,
                "units": units_data,
                "graph": {
                    "n_nodes": stats["total_nodes"],
                    "n_edges": stats["total_edges"],
                    "blue_combat_power": round(stats["blue_combat_power"], 4),
                    "red_combat_power":  round(stats["red_combat_power"], 4),
                    "power_ratio": round(
                        stats["blue_combat_power"] / max(stats["red_combat_power"], 1e-8), 3
                    ),
                }
            }
            scenarios.append(scenario)

            if (i + 1) % 20 == 0:
                print(f"  생성 중: {i+1}/{self.cfg.n_scenarios}")

        # 저장
        path = os.path.join(self.cfg.output_dir, "scenarios.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 시나리오 {len(scenarios)}개 → {path}")
        return scenarios

    # ── 에피소드 롤아웃 생성 ──────────────────

    def generate_episodes(self, scenarios: List[Dict]) -> List[Dict]:
        from ontology.combat_schema import ScenarioFactory
        from simulator.mixed_lanchester import MixedLanchesterEngine
        from simulator.resource_manager import ResourceManager

        print(f"\n[2/4] 시뮬레이션 에피소드 생성 ({len(scenarios)}개)...")
        episodes = []
        n_win = n_loss = n_draw = 0

        for i, sc in enumerate(scenarios):
            seed = sc["seed"]
            kg = ScenarioFactory.create_standard_scenario(
                n_blue=sc["n_blue"], n_red=sc["n_red"], seed=seed
            )
            engine = MixedLanchesterEngine(seed=seed)
            rm = ResourceManager(seed=seed)

            steps_data = []
            blue_hc0 = sc["stats"]["blue_headcount"]
            red_hc0  = sc["stats"]["red_headcount"]

            for step in range(self.cfg.episode_steps):
                result = engine.run_step_mixed(kg, resource_manager=rm)
                # 단순 행동 (학습용 레이블)
                action = int(self.rng.choice([0,1,2,3,4,5,6,7]))
                steps_data.append({
                    "step": step,
                    "action": action,
                    "blue_hc": result["blue_headcount"],
                    "red_hc":  result["red_headcount"],
                    "blue_cas": result["blue_casualties"],
                    "red_cas":  result["red_casualties"],
                    "status":  result["mission_status"],
                })
                if result["mission_status"] != "ongoing":
                    break

            final = steps_data[-1]
            # 결과 판정
            if final["blue_hc"] > final["red_hc"] * 1.2:
                outcome = "blue_win"; n_win += 1
            elif final["red_hc"] > final["blue_hc"] * 1.2:
                outcome = "red_win"; n_loss += 1
            else:
                outcome = "draw"; n_draw += 1

            total_blue_cas = sum(s["blue_cas"] for s in steps_data)
            force_red_ratio = 1 - final["blue_hc"] / max(blue_hc0, 1)

            episodes.append({
                "episode_id": sc["id"],
                "template":   sc["template"],
                "outcome":    outcome,
                "n_steps":    len(steps_data),
                "blue_hc_initial": blue_hc0,
                "red_hc_initial":  red_hc0,
                "blue_hc_final":   final["blue_hc"],
                "red_hc_final":    final["red_hc"],
                "total_blue_casualties": total_blue_cas,
                "force_reduction_ratio": round(force_red_ratio, 3),
                "steps": steps_data,
            })

            if (i + 1) % 20 == 0:
                print(f"  롤아웃: {i+1}/{len(scenarios)}  "
                      f"[Win:{n_win} Loss:{n_loss} Draw:{n_draw}]")

        path = os.path.join(self.cfg.output_dir, "episodes.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(episodes, f, ensure_ascii=False, indent=2)

        win_rate = n_win / len(episodes) if episodes else 0
        print(f"  ✅ 에피소드 {len(episodes)}개 → {path}")
        print(f"     Blue 승률: {win_rate:.1%}  "
              f"(Win:{n_win} Loss:{n_loss} Draw:{n_draw})")
        return episodes

    # ── IRL 전문가 데모 생성 ──────────────────

    def generate_irl_demos(self) -> Dict:
        from rl_agent.inverse_rl import DemonstrationGenerator, MaxEntropyIRL, STATE_DIM

        print(f"\n[3/4] IRL 전문가 데모 생성...")
        gen = DemonstrationGenerator(seed=self.cfg.seed)
        demos = gen.generate(n_per_tactic=self.cfg.n_irl_demos, episode_len=15)

        # IRL 학습
        irl = MaxEntropyIRL(feature_dim=STATE_DIM, lr=0.02, n_iter=150, seed=self.cfg.seed)
        learned_reward = irl.train(demos, verbose=False)

        demo_summary = {
            "n_demos": len(demos),
            "tactics": list(set(d.doctrine_label for d in demos)),
            "episode_len": 15,
            "learned_reward_weights": {
                name: round(float(w), 4)
                for name, w in zip(
                    ["blue_force_ratio","casualty_efficiency","force_size_penalty",
                     "mass_score","offensive_score","security_score","maneuver_score",
                     "ammo_level","fuel_level","time_pressure","flanking_bonus","envelopment_bonus"],
                    learned_reward.weights
                )
            },
            "final_training_loss": round(learned_reward.training_loss[-1], 6),
            "top_features": [
                {"name": n, "weight": round(float(w), 4)}
                for n, w in learned_reward.top_features(5)
            ],
        }

        path = os.path.join(self.cfg.output_dir, "irl_demos_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(demo_summary, f, ensure_ascii=False, indent=2)

        print(f"  ✅ IRL 데모 {len(demos)}개 (전술 {len(demo_summary['tactics'])}종)")
        print(f"     학습 손실: {demo_summary['final_training_loss']:.6f}")
        return demo_summary

    # ── 통계 리포트 생성 ──────────────────────

    def generate_statistics(
        self,
        scenarios: List[Dict],
        episodes: List[Dict],
        irl_summary: Dict
    ) -> Dict:
        print(f"\n[4/4] 데이터 통계 계산...")

        # 시나리오 통계
        power_ratios = [s["graph"]["power_ratio"] for s in scenarios]
        blue_cp_list = [s["graph"]["blue_combat_power"] for s in scenarios]
        red_cp_list  = [s["graph"]["red_combat_power"]  for s in scenarios]
        n_nodes_list = [s["graph"]["n_nodes"] for s in scenarios]
        n_edges_list = [s["graph"]["n_edges"] for s in scenarios]

        # 유닛 타입 분포
        type_counts = {}
        for sc in scenarios:
            for u in sc["units"]:
                key = f"{u['alignment']}_{u['type']}"
                type_counts[key] = type_counts.get(key, 0) + 1

        # 에피소드 통계
        outcomes = [e["outcome"] for e in episodes]
        n_win  = outcomes.count("blue_win")
        n_loss = outcomes.count("red_win")
        n_draw = outcomes.count("draw")
        step_counts = [e["n_steps"] for e in episodes]
        fr_ratios = [e["force_reduction_ratio"] for e in episodes]
        cas_totals = [e["total_blue_casualties"] for e in episodes]

        # 템플릿별 통계
        template_stats = {}
        for e in episodes:
            tmpl = e["template"]
            if tmpl not in template_stats:
                template_stats[tmpl] = {"total": 0, "win": 0}
            template_stats[tmpl]["total"] += 1
            if e["outcome"] == "blue_win":
                template_stats[tmpl]["win"] += 1

        stats = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.cfg),
            "scenario_stats": {
                "total": len(scenarios),
                "n_templates": len(SCENARIO_TEMPLATES),
                "avg_nodes": round(float(np.mean(n_nodes_list)), 1),
                "avg_edges": round(float(np.mean(n_edges_list)), 1),
                "avg_power_ratio": round(float(np.mean(power_ratios)), 3),
                "power_ratio_std": round(float(np.std(power_ratios)), 3),
                "avg_blue_cp": round(float(np.mean(blue_cp_list)), 3),
                "avg_red_cp":  round(float(np.mean(red_cp_list)), 3),
                "unit_type_distribution": type_counts,
            },
            "episode_stats": {
                "total": len(episodes),
                "blue_win": n_win, "red_win": n_loss, "draw": n_draw,
                "blue_win_rate": round(n_win / max(len(episodes), 1), 3),
                "avg_steps": round(float(np.mean(step_counts)), 1),
                "avg_force_reduction": round(float(np.mean(fr_ratios)), 3),
                "avg_blue_casualties": round(float(np.mean(cas_totals)), 1),
                "template_win_rates": {
                    t: round(v["win"] / max(v["total"], 1), 2)
                    for t, v in template_stats.items()
                },
            },
            "irl_stats": {
                "n_demos": irl_summary["n_demos"],
                "n_tactics": len(irl_summary["tactics"]),
                "tactics": irl_summary["tactics"],
                "top_reward_features": irl_summary["top_features"],
            },
        }

        path = os.path.join(self.cfg.output_dir, "data_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 통계 저장 → {path}")
        return stats


# ── HTML 시각화 생성기 ───────────────────────

class OntologyVisualizer:
    """온톨로지 데이터 통계 시각화 (Plotly.js 단일 HTML)"""

    def render(
        self,
        scenarios: List[Dict],
        episodes: List[Dict],
        stats: Dict,
        output_path: str = "ontology_stats.html"
    ) -> str:
        html = self._build_html(scenarios, episodes, stats)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def _build_html(self, scenarios, episodes, stats) -> str:
        sc_stats = stats["scenario_stats"]
        ep_stats = stats["episode_stats"]
        irl_stats = stats["irl_stats"]

        # ── 차트 데이터 준비 ──────────────────

        # 1. 유닛 타입 분포 (Blue vs Red)
        type_dist = sc_stats["unit_type_distribution"]
        unit_types = ["infantry","armor","artillery","aviation","engineer",
                      "signal","logistics","electronic_warfare"]
        blue_counts = [type_dist.get(f"blue_{t}", 0) for t in unit_types]
        red_counts  = [type_dist.get(f"red_{t}",  0) for t in unit_types]
        type_labels_kr = ["보병","기갑","포병","항공","공병","통신","군수","전자전"]

        # 2. 전술 유형별 승률
        tmpl_wr = ep_stats["template_win_rates"]
        tmpl_names = list(tmpl_wr.keys())
        tmpl_rates = [tmpl_wr[t]*100 for t in tmpl_names]
        tmpl_labels = {
            "assault":"정면공격","defense":"방어작전","flanking":"측면기동",
            "encirclement":"포위섬멸","delay":"지연방어","coastal_landing":"상륙작전",
            "urban_combat":"시가지전투","mountain_ops":"산악작전",
        }
        tmpl_kr = [tmpl_labels.get(t, t) for t in tmpl_names]

        # 3. 전력 비율 분포 (히스토그램용)
        power_ratios = [s["graph"]["power_ratio"] for s in scenarios]
        pr_bins = [round(0.4+i*0.1, 1) for i in range(20)]
        pr_counts = [0]*len(pr_bins)
        for pr in power_ratios:
            idx = min(int((pr-0.4)/0.1), len(pr_bins)-1)
            idx = max(0, idx)
            pr_counts[idx] += 1

        # 4. 에피소드 결과 파이
        outcomes_labels = ["Blue 승리","Red 승리","무승부"]
        outcomes_values = [ep_stats["blue_win"], ep_stats["red_win"], ep_stats["draw"]]

        # 5. 병력 감소율 분포
        fr_ratios = [e["force_reduction_ratio"] for e in episodes]
        fr_bins = [round(i*0.05, 2) for i in range(21)]
        fr_counts = [0]*len(fr_bins)
        for fr in fr_ratios:
            idx = min(int(fr/0.05), len(fr_bins)-1)
            idx = max(0, idx)
            fr_counts[idx] += 1

        # 6. IRL 보상 가중치
        irl_feats = [f["name"] for f in irl_stats["top_reward_features"]]
        irl_weights = [f["weight"] for f in irl_stats["top_reward_features"]]
        feat_labels_kr = {
            "blue_force_ratio":"Blue 전투력비","casualty_efficiency":"교전효율",
            "force_size_penalty":"병력절감","mass_score":"집중원칙",
            "offensive_score":"공세원칙","security_score":"방호원칙",
            "maneuver_score":"기동원칙","flanking_bonus":"측면기동",
            "envelopment_bonus":"포위달성","ammo_level":"탄약수준",
            "fuel_level":"연료수준","time_pressure":"시간압박",
        }
        irl_labels_kr = [feat_labels_kr.get(f, f) for f in irl_feats]
        irl_colors = ["#22C55E" if w > 0 else "#EF4444" for w in irl_weights]

        # 7. 스텝 수 분포
        step_counts = [e["n_steps"] for e in episodes]
        max_steps = max(step_counts) if step_counts else 20
        step_bins = list(range(1, max_steps+2))
        step_hist = [0]*(max_steps+1)
        for s in step_counts:
            step_hist[min(s, max_steps)] += 1

        return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Combat Optimization — Ontology Data Statistics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2234;
    --border: #1e3a5f;
    --accent: #0ea5e9;
    --accent2: #22c55e;
    --accent3: #f59e0b;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --blue-unit: #3b82f6;
    --red-unit: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }}
  /* Header */
  .header {{
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1128 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 600px 200px at 80% 50%, rgba(14,165,233,0.08) 0%, transparent 70%);
    pointer-events: none;
  }}
  .header-grid {{ display: flex; align-items: center; gap: 20px; }}
  .header-icon {{
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; box-shadow: 0 0 20px rgba(14,165,233,0.3);
    flex-shrink: 0;
  }}
  .header-title {{ font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }}
  .header-sub {{ font-size: 13px; color: var(--muted); margin-top: 3px; }}
  .header-badge {{
    margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;
  }}
  .badge {{
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
    border: 1px solid; white-space: nowrap;
  }}
  .badge-blue  {{ color: #60a5fa; border-color: #1d4ed8; background: rgba(29,78,216,0.15); }}
  .badge-green {{ color: #4ade80; border-color: #15803d; background: rgba(21,128,61,0.15); }}
  .badge-amber {{ color: #fbbf24; border-color: #b45309; background: rgba(180,83,9,0.15); }}

  /* Stats bar */
  .statsbar {{
    display: flex; gap: 1px; background: var(--border);
    border-bottom: 1px solid var(--border);
  }}
  .stat-item {{
    flex: 1; background: var(--surface); padding: 14px 20px;
    display: flex; flex-direction: column; gap: 2px;
  }}
  .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); }}
  .stat-value {{ font-size: 22px; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; }}
  .stat-sub {{ font-size: 11px; color: var(--muted); }}

  /* Main layout */
  .main {{ padding: 28px 32px; }}
  .section-title {{
    font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px;
    color: var(--muted); margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; overflow: hidden;
  }}
  .card-title {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 14px; }}
  .chart-area {{ height: 260px; }}
  .chart-wide  {{ height: 280px; }}
  .chart-short {{ height: 200px; }}

  /* IRL insight panel */
  .insight-panel {{
    background: linear-gradient(135deg, rgba(14,165,233,0.06), rgba(34,197,94,0.04));
    border: 1px solid rgba(14,165,233,0.2); border-radius: 12px;
    padding: 20px; margin-bottom: 20px;
  }}
  .insight-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }}
  .insight-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 6px var(--accent);
  }}
  .insight-title {{ font-size: 14px; font-weight: 700; color: #fff; }}
  .insight-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }}
  .insight-card {{
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 12px;
  }}
  .ik {{ font-size: 11px; color: var(--muted); }}
  .iv {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}
  .iv.pos {{ color: var(--accent2); }}
  .iv.neg {{ color: var(--red); }}

  /* Footer */
  .footer {{
    text-align: center; padding: 20px; color: var(--muted); font-size: 12px;
    border-top: 1px solid var(--border);
  }}

  @media (max-width: 900px) {{
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
    .insight-grid {{ grid-template-columns: 1fr 1fr; }}
    .header-badge {{ display: none; }}
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-grid">
    <div class="header-icon">🧠</div>
    <div>
      <div class="header-title">AI Combat Optimization — 온톨로지 데이터 통계</div>
      <div class="header-sub">Ontology Statistics Dashboard &nbsp;·&nbsp; 생성: {stats["generated_at"]}</div>
    </div>
    <div class="header-badge">
      <span class="badge badge-blue">📋 시나리오 {sc_stats["total"]}개</span>
      <span class="badge badge-green">🎬 에피소드 {ep_stats["total"]}개</span>
      <span class="badge badge-amber">🤖 IRL 데모 {irl_stats["n_demos"]}개</span>
    </div>
  </div>
</div>

<!-- Stats Bar -->
<div class="statsbar">
  <div class="stat-item">
    <span class="stat-label">총 시나리오</span>
    <span class="stat-value">{sc_stats["total"]}</span>
    <span class="stat-sub">{sc_stats["n_templates"]}개 전술 유형</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">평균 그래프 크기</span>
    <span class="stat-value">{sc_stats["avg_nodes"]:.0f}</span>
    <span class="stat-sub">노드 · 엣지 {sc_stats["avg_edges"]:.0f}개</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">Blue 승률</span>
    <span class="stat-value">{ep_stats["blue_win_rate"]:.0%}</span>
    <span class="stat-sub">Win {ep_stats["blue_win"]} / Loss {ep_stats["red_win"]} / Draw {ep_stats["draw"]}</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">평균 병력 감소율</span>
    <span class="stat-value">{ep_stats["avg_force_reduction"]:.0%}</span>
    <span class="stat-sub">작전 중 아군 손실</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">평균 교전 스텝</span>
    <span class="stat-value">{ep_stats["avg_steps"]:.1f}</span>
    <span class="stat-sub">스텝 / 에피소드</span>
  </div>
  <div class="stat-item">
    <span class="stat-label">평균 전력 비율</span>
    <span class="stat-value">{sc_stats["avg_power_ratio"]:.2f}</span>
    <span class="stat-sub">Blue/Red 전투력비</span>
  </div>
</div>

<div class="main">

<!-- IRL Insight Panel -->
<div class="insight-panel">
  <div class="insight-header">
    <div class="insight-dot"></div>
    <span class="insight-title">🔬 IRL (역강화학습) 핵심 인사이트 — AI가 교범에서 학습한 가치</span>
  </div>
  <div class="insight-grid">
    {"".join(f'''<div class="insight-card">
      <div class="ik">{feat_labels_kr.get(f["name"], f["name"])}</div>
      <div class="iv {'pos' if f['weight'] > 0 else 'neg'}">{f['weight']:+.3f}</div>
    </div>''' for f in irl_stats["top_reward_features"])}
  </div>
</div>

<!-- Row 1 -->
<p class="section-title">📊 시나리오 분석 (Scenario Analysis)</p>
<div class="grid-3">
  <div class="card">
    <div class="card-title">유닛 타입 분포 (Blue vs Red)</div>
    <div class="chart-area" id="ch-unit-type"></div>
  </div>
  <div class="card">
    <div class="card-title">전력 비율 분포 (Blue/Red Combat Power Ratio)</div>
    <div class="chart-area" id="ch-power-ratio"></div>
  </div>
  <div class="card">
    <div class="card-title">에피소드 결과 분포</div>
    <div class="chart-area" id="ch-outcome-pie"></div>
  </div>
</div>

<!-- Row 2 -->
<p class="section-title">🎬 에피소드 분석 (Episode Analysis)</p>
<div class="grid-2">
  <div class="card">
    <div class="card-title">전술 유형별 Blue 승률</div>
    <div class="chart-wide" id="ch-template-wr"></div>
  </div>
  <div class="card">
    <div class="card-title">병력 감소율 분포 (Force Reduction Ratio)</div>
    <div class="chart-wide" id="ch-fr-dist"></div>
  </div>
</div>

<!-- Row 3 -->
<p class="section-title">🤖 IRL 학습 분석 (Inverse RL Analysis)</p>
<div class="grid-2">
  <div class="card">
    <div class="card-title">학습된 보상 가중치 (Top Features)</div>
    <div class="chart-wide" id="ch-irl-weights"></div>
  </div>
  <div class="card">
    <div class="card-title">에피소드 스텝 수 분포</div>
    <div class="chart-wide" id="ch-steps-dist"></div>
  </div>
</div>

</div><!-- /main -->
<div class="footer">
  🧠 AI Combat Optimization System v2.0 &nbsp;·&nbsp;
  학술 연구용 합성 데이터 &nbsp;·&nbsp; 2026 국방 AI 경진대회
</div>

<script>
const PLOTLY_DEFAULTS = {{
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: {{ family: 'Noto Sans KR, Malgun Gothic, sans-serif', color: '#94a3b8', size: 11 }},
  margin: {{ t: 10, r: 10, b: 40, l: 50 }},
  showlegend: true,
  legend: {{ bgcolor: 'transparent', font: {{ color: '#94a3b8', size: 10 }} }},
}};
const AXIS_STYLE = {{
  gridcolor: '#1e3a5f', zerolinecolor: '#1e3a5f',
  tickfont: {{ color: '#64748b', size: 10 }},
}};

// 1. 유닛 타입 분포
Plotly.newPlot('ch-unit-type', [
  {{ type: 'bar', name: 'Blue 아군', x: {type_labels_kr},
     y: {blue_counts}, marker: {{ color: 'rgba(59,130,246,0.85)', line: {{ color:'#1d4ed8', width:1 }} }} }},
  {{ type: 'bar', name: 'Red 적군',  x: {type_labels_kr},
     y: {red_counts},  marker: {{ color: 'rgba(239,68,68,0.75)',  line: {{ color:'#b91c1c', width:1 }} }} }},
], {{
  ...PLOTLY_DEFAULTS,
  barmode: 'group',
  xaxis: {{ ...AXIS_STYLE, tickangle: -30 }},
  yaxis: {{ ...AXIS_STYLE, title: '유닛 수' }},
  margin: {{ t:10, r:10, b:60, l:50 }},
}}, {{responsive:true, displayModeBar:false}});

// 2. 전력 비율 분포
Plotly.newPlot('ch-power-ratio', [{{
  type: 'bar', name: '시나리오 수',
  x: {pr_bins}, y: {pr_counts},
  marker: {{
    color: {pr_counts}.map(v => v > {max(pr_counts)//2} ? '#0ea5e9' : 'rgba(14,165,233,0.5)'),
    line: {{ color:'#0284c7', width:0.5 }}
  }},
}}], {{
  ...PLOTLY_DEFAULTS, showlegend: false,
  xaxis: {{ ...AXIS_STYLE, title: 'Blue/Red 전투력 비율' }},
  yaxis: {{ ...AXIS_STYLE, title: '시나리오 수' }},
  shapes: [{{ type:'line', x0:1.0, x1:1.0, y0:0, y1:1, yref:'paper',
    line:{{ color:'#f59e0b', width:1.5, dash:'dot' }} }}],
  annotations: [{{ x:1.0, y:0.95, yref:'paper', text:'균형선', showarrow:false,
    font:{{ color:'#f59e0b', size:10 }}, xanchor:'left', xshift:4 }}],
}}, {{responsive:true, displayModeBar:false}});

// 3. 에피소드 결과 파이
Plotly.newPlot('ch-outcome-pie', [{{
  type: 'pie',
  labels: {outcomes_labels},
  values: {outcomes_values},
  hole: 0.52,
  marker: {{ colors: ['#3b82f6','#ef4444','#94a3b8'],
    line: {{ color:'#0a0e1a', width:2 }} }},
  textfont: {{ color: '#fff', size: 11 }},
  hovertemplate: '%{{label}}: %{{value}}개 (%{{percent}})<extra></extra>',
}}], {{
  ...PLOTLY_DEFAULTS,
  margin: {{ t:10, r:10, b:10, l:10 }},
  legend: {{ bgcolor:'transparent', font:{{ color:'#94a3b8', size:10 }}, orientation:'h', y:-0.05 }},
  annotations: [{{ text:`<b>{ep_stats["blue_win_rate"]:.0%}</b><br>Blue 승률`, x:0.5, y:0.5,
    font:{{ color:'#fff', size:13 }}, showarrow:false }}],
}}, {{responsive:true, displayModeBar:false}});

// 4. 전술별 승률
Plotly.newPlot('ch-template-wr', [{{
  type: 'bar', orientation: 'h',
  x: {tmpl_rates},
  y: {tmpl_kr},
  marker: {{
    color: {tmpl_rates}.map(v => v >= 60 ? '#22c55e' : v >= 40 ? '#f59e0b' : '#ef4444'),
    line: {{ color:'rgba(255,255,255,0.1)', width:0.5 }}
  }},
  text: {tmpl_rates}.map(v => v.toFixed(0)+'%'), textposition:'outside',
  textfont: {{ color:'#94a3b8', size:10 }},
}}], {{
  ...PLOTLY_DEFAULTS, showlegend:false,
  xaxis: {{ ...AXIS_STYLE, title:'Blue 승률 (%)', range:[0, 100] }},
  yaxis: {{ ...AXIS_STYLE, automargin:true }},
  margin: {{ t:10, r:60, b:40, l:90 }},
  shapes: [{{ type:'line', x0:50, x1:50, y0:-0.5, y1:{len(tmpl_names)}-0.5,
    line:{{ color:'#64748b', width:1, dash:'dot' }} }}],
}}, {{responsive:true, displayModeBar:false}});

// 5. 병력 감소율 분포
Plotly.newPlot('ch-fr-dist', [{{
  type: 'bar',
  x: {[round(b*100) for b in fr_bins]},
  y: {fr_counts},
  marker: {{
    color: {[round(b*100) for b in fr_bins]}.map(v =>
      v < 15 ? '#22c55e' : v < 30 ? '#f59e0b' : '#ef4444'),
    line: {{ color:'rgba(255,255,255,0.05)', width:0.3 }}
  }},
}}], {{
  ...PLOTLY_DEFAULTS, showlegend:false,
  xaxis: {{ ...AXIS_STYLE, title:'아군 병력 감소율 (%)' }},
  yaxis: {{ ...AXIS_STYLE, title:'에피소드 수' }},
  shapes: [
    {{ type:'rect', x0:0, x1:15, y0:0, y1:1, yref:'paper',
      fillcolor:'rgba(34,197,94,0.06)', line:{{ width:0 }} }},
    {{ type:'rect', x0:15, x1:25, y0:0, y1:1, yref:'paper',
      fillcolor:'rgba(245,158,11,0.05)', line:{{ width:0 }} }},
    {{ type:'line', x0:{round(ep_stats["avg_force_reduction"]*100)},
      x1:{round(ep_stats["avg_force_reduction"]*100)}, y0:0, y1:1, yref:'paper',
      line:{{ color:'#0ea5e9', width:2, dash:'dash' }} }},
  ],
  annotations: [{{ x:{round(ep_stats["avg_force_reduction"]*100)}, y:0.9, yref:'paper',
    text:'평균', showarrow:false, font:{{ color:'#0ea5e9', size:10 }}, xanchor:'left', xshift:4 }}],
}}, {{responsive:true, displayModeBar:false}});

// 6. IRL 보상 가중치
Plotly.newPlot('ch-irl-weights', [{{
  type: 'bar', orientation: 'h',
  x: {irl_weights},
  y: {irl_labels_kr},
  marker: {{
    color: {irl_colors},
    line: {{ color:'rgba(255,255,255,0.1)', width:0.5 }}
  }},
  text: {irl_weights}.map(v => v.toFixed(3)), textposition:'outside',
  textfont: {{ color:'#94a3b8', size:10 }},
}}], {{
  ...PLOTLY_DEFAULTS, showlegend:false,
  xaxis: {{ ...AXIS_STYLE, title:'가중치', zeroline:true, zerolinewidth:1.5, zerolinecolor:'#334155' }},
  yaxis: {{ ...AXIS_STYLE, automargin:true }},
  margin: {{ t:10, r:60, b:40, l:100 }},
}}, {{responsive:true, displayModeBar:false}});

// 7. 스텝 수 분포
Plotly.newPlot('ch-steps-dist', [{{
  type: 'bar',
  x: Array.from({{length:{max_steps+1}}}, (_,i)=>i+1),
  y: {step_hist},
  marker: {{
    color: 'rgba(14,165,233,0.7)',
    line: {{ color:'#0284c7', width:0.5 }}
  }},
}}], {{
  ...PLOTLY_DEFAULTS, showlegend:false,
  xaxis: {{ ...AXIS_STYLE, title:'에피소드 스텝 수' }},
  yaxis: {{ ...AXIS_STYLE, title:'에피소드 수' }},
  shapes: [{{ type:'line', x0:{ep_stats["avg_steps"]}, x1:{ep_stats["avg_steps"]}, y0:0, y1:1, yref:'paper',
    line:{{ color:'#f59e0b', width:1.5, dash:'dash' }} }}],
  annotations: [{{ x:{ep_stats["avg_steps"]}, y:0.9, yref:'paper',
    text:`평균 {ep_stats["avg_steps"]:.1f}스텝`, showarrow:false,
    font:{{ color:'#f59e0b', size:10 }}, xanchor:'left', xshift:4 }}],
}}, {{responsive:true, displayModeBar:false}});
</script>
</body>
</html>"""


# ── 메인 ─────────────────────────────────────

def print_final_report(stats: Dict):
    sc = stats["scenario_stats"]
    ep = stats["episode_stats"]
    irl = stats["irl_stats"]
    print("""
╔══════════════════════════════════════════════╗
║     데이터 생성 완료 리포트                  ║
╠══════════════════════════════════════════════╣""")
    print(f"║  📋 시나리오  : {sc['total']:4d}개 ({sc['n_templates']}개 전술 유형)")
    print(f"║  📊 평균 노드 : {sc['avg_nodes']:.0f}개 노드 / {sc['avg_edges']:.0f}개 엣지")
    print(f"║  ⚖️  평균 전력비: {sc['avg_power_ratio']:.2f} ± {sc['power_ratio_std']:.2f}")
    print(f"║  🎬 에피소드  : {ep['total']:4d}개")
    print(f"║  🏆 Blue 승률 : {ep['blue_win_rate']:.0%}  (Win:{ep['blue_win']} Loss:{ep['red_win']} Draw:{ep['draw']})")
    print(f"║  📉 병력감소율: {ep['avg_force_reduction']:.0%} 평균")
    print(f"║  ⏱️  평균 스텝 : {ep['avg_steps']:.1f}스텝")
    print(f"║  🤖 IRL 데모  : {irl['n_demos']:4d}개 ({irl['n_tactics']}개 전술)")
    print("""╚══════════════════════════════════════════════╝""")


def main():
    parser = argparse.ArgumentParser(
        description="온톨로지 + 시뮬레이션 데이터 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python generate_data.py                   기본 (100 시나리오)
  python generate_data.py --scenarios 500   대규모
  python generate_data.py --quick           빠른 검증 (10개)
  python generate_data.py --no-viz          시각화 생략
        """
    )
    parser.add_argument("--scenarios",  type=int,  default=100,   help="생성할 시나리오 수")
    parser.add_argument("--steps",      type=int,  default=20,    help="에피소드 최대 스텝")
    parser.add_argument("--irl-demos",  type=int,  default=50,    help="IRL 데모 수 (전술당)")
    parser.add_argument("--seed",       type=int,  default=42,    help="랜덤 시드")
    parser.add_argument("--output",     type=str,  default="data",help="출력 디렉토리")
    parser.add_argument("--quick",      action="store_true",       help="빠른 검증 (10개)")
    parser.add_argument("--no-viz",     action="store_true",       help="시각화 생략")
    args = parser.parse_args()

    if args.quick:
        args.scenarios = 10
        args.irl_demos = 5

    print("""
╔══════════════════════════════════════════════╗
║  🧠 AI Combat Optimization — 데이터 생성기  ║
║  병력 절감을 위한 AI 기술 전장 활용 방안    ║
╚══════════════════════════════════════════════╝""")
    print(f"  시나리오: {args.scenarios}개 | 스텝: {args.steps} | IRL 데모: {args.irl_demos}/전술 | 시드: {args.seed}")
    print(f"  출력: ./{args.output}/\n")

    t0 = time.time()
    cfg = GenerationConfig(
        n_scenarios=args.scenarios,
        episode_steps=args.steps,
        n_irl_demos=args.irl_demos,
        seed=args.seed,
        output_dir=args.output,
        visualize=not args.no_viz,
    )

    gen = DataGenerator(cfg)
    scenarios  = gen.generate_scenarios()
    episodes   = gen.generate_episodes(scenarios)
    irl_summary = gen.generate_irl_demos()
    stats      = gen.generate_statistics(scenarios, episodes, irl_summary)

    # 시각화
    if not args.no_viz:
        print("\n[+] 온톨로지 통계 시각화 생성...")
        viz = OntologyVisualizer()
        viz_path = os.path.join(args.output, "ontology_stats.html")
        out = viz.render(scenarios, episodes, stats, viz_path)
        print(f"  ✅ 시각화 → {out}")
        print(f"     브라우저에서 열기: file://{os.path.abspath(out)}")

    elapsed = time.time() - t0
    print(f"\n  ⏱️  총 소요 시간: {elapsed:.1f}초")
    print_final_report(stats)
    print(f"\n  📁 생성된 파일:")
    for fname in ["scenarios.json","episodes.json","irl_demos_summary.json","data_stats.json","ontology_stats.html"]:
        fpath = os.path.join(args.output, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"    • {fpath}  ({size//1024}KB)" if size > 1024 else f"    • {fpath}  ({size}B)")
    print()


if __name__ == "__main__":
    main()
