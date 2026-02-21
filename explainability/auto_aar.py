"""
auto_aar.py — I. 자동 전투 후 분석 (Auto After-Action Review)
에피소드 로그 → 결정적 순간 / 최선 선택 / 놓친 기회 / 개선 권고
학술 연구용 합성 데이터
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ── 에피소드 로그 구조 ────────────────────────

@dataclass
class StepRecord:
    """단일 스텝 기록"""
    step: int
    action: int
    action_name: str
    blue_headcount: int
    red_headcount: int
    blue_casualties: int
    red_casualties: int
    win_rate: float
    uncertainty: float
    ammo_level: float
    fuel_level: float
    mission_status: str
    doctrine_score: float = 0.5
    flanking: bool = False
    enveloping: bool = False
    reward: float = 0.0

    @property
    def blue_red_ratio(self) -> float:
        return self.blue_headcount / max(self.red_headcount, 1)

    @property
    def casualty_efficiency(self) -> float:
        return self.red_casualties / max(self.blue_casualties, 1)


@dataclass
class EpisodeLog:
    """전체 에피소드 로그"""
    episode_id: str
    initial_blue_hc: int
    initial_red_hc: int
    outcome: str               # "win" | "loss" | "draw"
    steps: List[StepRecord] = field(default_factory=list)
    doctrine_compliance_avg: float = 0.5

    def add_step(self, record: StepRecord):
        self.steps.append(record)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def total_blue_casualties(self) -> int:
        return sum(s.blue_casualties for s in self.steps)

    @property
    def total_red_casualties(self) -> int:
        return sum(s.red_casualties for s in self.steps)

    @property
    def force_reduction_ratio(self) -> float:
        final_blue = self.steps[-1].blue_headcount if self.steps else self.initial_blue_hc
        return 1 - final_blue / max(self.initial_blue_hc, 1)


# ── AAR 분석 결과 ─────────────────────────────

@dataclass
class TurningPoint:
    """결정적 전환점"""
    step: int
    event_type: str    # "win_rate_shift" | "heavy_loss" | "opportunity_taken" | "resource_crisis"
    description: str
    impact: float      # 영향도 [-1, 1]  양수=유리, 음수=불리
    evidence: Dict

    def to_str(self) -> str:
        icon = "📈" if self.impact > 0 else "📉"
        return f"  {icon} 스텝 {self.step:2d} [{self.event_type}]: {self.description}"


@dataclass
class AARReport:
    """자동 전투 후 분석 리포트"""
    episode_id: str
    outcome: str
    executive_summary: str
    turning_points: List[TurningPoint]
    best_decisions: List[Dict]
    missed_opportunities: List[Dict]
    resource_analysis: Dict
    doctrine_compliance: Dict
    recommendations: List[str]
    lessons_learned: List[str]
    overall_score: float       # [0,1] 전투 수행 종합 점수

    def to_report(self) -> str:
        outcome_icon = {"win":"🏆","loss":"💀","draw":"🤝"}.get(self.outcome,"❓")
        lines = [
            "="*55,
            f"📋 자동 전투 후 분석 (AAR) — {outcome_icon} {self.outcome.upper()}",
            f"   에피소드: {self.episode_id}",
            "="*55,
            f"\n{self.executive_summary}",
        ]
        if self.turning_points:
            lines += ["\n[⚡ 결정적 전환점]"]
            for tp in sorted(self.turning_points, key=lambda t: abs(t.impact), reverse=True)[:4]:
                lines.append(tp.to_str())

        if self.best_decisions:
            lines += ["\n[✅ 최선의 결정]"]
            for d in self.best_decisions[:3]:
                lines.append(f"  스텝 {d['step']:2d} [{d['action']}]: {d['reason']}")

        if self.missed_opportunities:
            lines += ["\n[⚠️ 놓친 기회]"]
            for m in self.missed_opportunities[:3]:
                lines.append(f"  스텝 {m['step']:2d}: {m['description']}")

        lines += ["\n[📊 자원 분석]"]
        for k, v in self.resource_analysis.items():
            lines.append(f"  {k}: {v}")

        lines += ["\n[💡 개선 권고]"]
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")

        lines += ["\n[📚 교훈]"]
        for lesson in self.lessons_learned:
            lines.append(f"  • {lesson}")

        lines.append(f"\n[종합 수행 점수: {self.overall_score:.0%}]")
        lines.append("="*55)
        return "\n".join(lines)


# ── AAR 생성기 ───────────────────────────────

class AutoAARGenerator:
    """
    에피소드 로그 → 자동 전투 후 분석
    결정적 순간 탐지, 최선/놓친 결정 식별, 권고 생성
    """

    ACTION_NAMES = ["대기","전진","후퇴","교전","지원","측면기동","보급요청","방어진지구축"]

    def generate(self, log: EpisodeLog) -> AARReport:
        if not log.steps:
            return self._empty_report(log)

        turning_points  = self._find_turning_points(log)
        best_decisions  = self._find_best_decisions(log)
        missed_opps     = self._find_missed_opportunities(log)
        resource_anal   = self._analyze_resources(log)
        doctrine_anal   = self._analyze_doctrine(log)
        recommendations = self._generate_recommendations(log, turning_points, missed_opps)
        lessons         = self._extract_lessons(log, turning_points)
        overall_score   = self._compute_overall_score(log)
        summary         = self._write_executive_summary(log, overall_score, turning_points)

        return AARReport(
            episode_id=log.episode_id,
            outcome=log.outcome,
            executive_summary=summary,
            turning_points=turning_points,
            best_decisions=best_decisions,
            missed_opportunities=missed_opps,
            resource_analysis=resource_anal,
            doctrine_compliance=doctrine_anal,
            recommendations=recommendations,
            lessons_learned=lessons,
            overall_score=overall_score,
        )

    def _find_turning_points(self, log: EpisodeLog) -> List[TurningPoint]:
        tps = []
        steps = log.steps
        n = len(steps)
        if n < 2:
            return tps

        for i in range(1, n):
            s_prev, s_curr = steps[i-1], steps[i]

            # 승률 급변
            wr_delta = s_curr.win_rate - s_prev.win_rate
            if abs(wr_delta) > 0.10:
                tps.append(TurningPoint(
                    step=s_curr.step,
                    event_type="win_rate_shift",
                    description=f"승률 {'상승' if wr_delta>0 else '급락'} "
                                f"({s_prev.win_rate:.0%}→{s_curr.win_rate:.0%})",
                    impact=wr_delta,
                    evidence={"wr_before": s_prev.win_rate, "wr_after": s_curr.win_rate},
                ))

            # 대규모 손실
            if s_curr.blue_casualties > 30:
                tps.append(TurningPoint(
                    step=s_curr.step,
                    event_type="heavy_loss",
                    description=f"아군 대규모 손실 ({s_curr.blue_casualties}명)",
                    impact=-0.3 - s_curr.blue_casualties/200,
                    evidence={"casualties": s_curr.blue_casualties},
                ))

            # 측면 기동 성공
            if s_curr.flanking and s_curr.red_casualties > s_curr.blue_casualties:
                tps.append(TurningPoint(
                    step=s_curr.step,
                    event_type="opportunity_taken",
                    description=f"측면 기동 성공 → 적 {s_curr.red_casualties}명 제압",
                    impact=0.25,
                    evidence={"red_cas": s_curr.red_casualties},
                ))

            # 자원 위기
            if s_curr.ammo_level < 0.15 or s_curr.fuel_level < 0.15:
                res = "탄약" if s_curr.ammo_level < 0.15 else "연료"
                tps.append(TurningPoint(
                    step=s_curr.step,
                    event_type="resource_crisis",
                    description=f"{res} 위기 수준 도달",
                    impact=-0.2,
                    evidence={"ammo": s_curr.ammo_level, "fuel": s_curr.fuel_level},
                ))

        # 최종 결과
        last = steps[-1]
        if log.outcome == "win":
            tps.append(TurningPoint(
                step=last.step,
                event_type="mission_complete",
                description=f"임무 완료 — 최종 전력비 {last.blue_red_ratio:.2f}:1",
                impact=1.0,
                evidence={"outcome": "win"},
            ))
        elif log.outcome == "loss":
            tps.append(TurningPoint(
                step=last.step,
                event_type="mission_failed",
                description="임무 실패",
                impact=-1.0,
                evidence={"outcome": "loss"},
            ))

        return sorted(tps, key=lambda t: t.step)

    def _find_best_decisions(self, log: EpisodeLog) -> List[Dict]:
        best = []
        for i, s in enumerate(log.steps):
            # 높은 교전 효율
            if s.casualty_efficiency > 3.0 and s.red_casualties > 0:
                best.append({
                    "step": s.step,
                    "action": self.ACTION_NAMES[s.action] if s.action < 8 else str(s.action),
                    "reason": f"교전 효율 {s.casualty_efficiency:.1f}:1 달성 "
                              f"(Red -{s.red_casualties}명 / Blue -{s.blue_casualties}명)",
                    "score": s.casualty_efficiency,
                })
            # 적시 보급
            if s.action == 6 and i > 0:
                prev_ammo = log.steps[i-1].ammo_level
                if prev_ammo < 0.30:
                    best.append({
                        "step": s.step,
                        "action": "보급요청",
                        "reason": f"탄약 {prev_ammo:.0%}에서 적시 보급 요청 — 전투 지속력 확보",
                        "score": 0.8,
                    })
            # 측면 기동
            if s.flanking:
                best.append({
                    "step": s.step,
                    "action": "측면기동",
                    "reason": "SURPRISE·MANEUVER 원칙 적용 — 적 취약점 공략",
                    "score": 0.75,
                })
        return sorted(best, key=lambda d: d["score"], reverse=True)[:5]

    def _find_missed_opportunities(self, log: EpisodeLog) -> List[Dict]:
        missed = []
        for i, s in enumerate(log.steps):
            # 낮은 승률에도 후퇴하지 않음
            if s.win_rate < 0.30 and s.action not in (2, 7, 6):
                missed.append({
                    "step": s.step,
                    "description": f"승률 {s.win_rate:.0%}에서 방어/보급 미실시 "
                                   f"(선택: {self.ACTION_NAMES[s.action] if s.action<8 else s.action})",
                    "potential": "철수 또는 방어진지 구축으로 손실 최소화 가능",
                })
            # 대기 남발
            if s.action == 0 and s.win_rate > 0.70:
                missed.append({
                    "step": s.step,
                    "description": f"승률 {s.win_rate:.0%} 상황에서 불필요한 대기",
                    "potential": "공세 전환으로 전투 조기 종결 가능",
                })
            # 자원 고갈 전 미보급
            if s.ammo_level < 0.20 and s.action != 6:
                missed.append({
                    "step": s.step,
                    "description": f"탄약 {s.ammo_level:.0%}에서 보급 미실시",
                    "potential": "선제 보급으로 전투 지속성 확보 가능",
                })
        return missed[:5]

    def _analyze_resources(self, log: EpisodeLog) -> Dict:
        steps = log.steps
        if not steps:
            return {}
        ammo_vals = [s.ammo_level for s in steps]
        fuel_vals = [s.fuel_level for s in steps]
        resupply_count = sum(1 for s in steps if s.action == 6)

        return {
            "최저 탄약":   f"{min(ammo_vals):.0%} (스텝 {ammo_vals.index(min(ammo_vals))+1})",
            "최저 연료":   f"{min(fuel_vals):.0%} (스텝 {fuel_vals.index(min(fuel_vals))+1})",
            "평균 탄약":   f"{np.mean(ammo_vals):.0%}",
            "보급 요청":   f"{resupply_count}회",
            "자원 위기":   f"{'없음' if min(ammo_vals) > 0.15 and min(fuel_vals) > 0.15 else '발생'}",
        }

    def _analyze_doctrine(self, log: EpisodeLog) -> Dict:
        if not log.steps:
            return {}
        scores = [s.doctrine_score for s in log.steps]
        action_counts = {}
        for s in log.steps:
            a = self.ACTION_NAMES[s.action] if s.action < 8 else str(s.action)
            action_counts[a] = action_counts.get(a, 0) + 1
        most_used = max(action_counts, key=action_counts.get)
        return {
            "평균 교리 준수도": f"{np.mean(scores):.0%}",
            "최저 교리 준수도": f"{min(scores):.0%}",
            "가장 많이 쓴 행동": f"{most_used} ({action_counts[most_used]}회)",
            "측면 기동 횟수":   str(sum(1 for s in log.steps if s.flanking)),
        }

    def _generate_recommendations(
        self, log: EpisodeLog,
        turning_points: List[TurningPoint],
        missed: List[Dict]
    ) -> List[str]:
        recs = []

        # 손실 기반 권고
        loss_ratio = log.force_reduction_ratio
        if loss_ratio > 0.4:
            recs.append(f"아군 손실 {loss_ratio:.0%} — 다음에는 초기 단계에서 방어진지 선점 권장")
        if loss_ratio < 0.15 and log.outcome == "win":
            recs.append("병력 절감 성공 — 현재 전술을 표준화하여 유사 작전에 적용 권장")

        # 자원 기반 권고
        if any("보급 미실시" in m["description"] for m in missed):
            recs.append("탄약 30% 도달 전 선제 보급 요청 루틴 수립 권장")

        # 기동 기반 권고
        flanking_count = sum(1 for s in log.steps if s.flanking)
        if flanking_count == 0:
            recs.append("측면 기동 미활용 — MANEUVER·SURPRISE 원칙 강화 훈련 필요")
        elif flanking_count >= 3:
            recs.append("측면 기동 효과적 활용 — 포위 기동으로 발전 고려")

        # 시간 기반
        if log.n_steps > 20:
            recs.append("장시간 작전으로 피로도 누적 위험 — 조기 결전 전술 연습 권장")

        if not recs:
            recs.append("전반적으로 우수한 전술 판단 — 현재 수준 유지")

        return recs[:4]

    def _extract_lessons(
        self, log: EpisodeLog,
        turning_points: List[TurningPoint]
    ) -> List[str]:
        lessons = []

        # 결과별 교훈
        if log.outcome == "win" and log.force_reduction_ratio < 0.2:
            lessons.append("최소 병력으로 임무 달성 — 병력 효율 극대화 전술 확인됨")
        if log.outcome == "loss":
            worst_tp = min(turning_points, key=lambda t: t.impact, default=None)
            if worst_tp:
                lessons.append(f"패인: {worst_tp.description} — 유사 상황 대응 SOP 수립 필요")

        # 교리 교훈
        avg_doc = np.mean([s.doctrine_score for s in log.steps]) if log.steps else 0.5
        if avg_doc > 0.7:
            lessons.append("교리 원칙 준수가 전투 성과와 상관관계 확인됨")
        else:
            lessons.append(f"교리 준수도 {avg_doc:.0%} — 9대 원칙 재숙달 훈련 필요")

        # 자원 교훈
        if any(s.ammo_level < 0.1 for s in log.steps):
            lessons.append("탄약 고갈 경험 — '탄약 20% 선 = 보급 트리거' 원칙 수립")

        if not lessons:
            lessons.append("주요 교훈 없음 — 전반적으로 계획 대로 진행")

        return lessons[:4]

    def _compute_overall_score(self, log: EpisodeLog) -> float:
        """종합 수행 점수 [0,1]"""
        if not log.steps:
            return 0.0
        outcome_score = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(log.outcome, 0.3)
        efficiency    = 1 - log.force_reduction_ratio
        avg_wr        = float(np.mean([s.win_rate for s in log.steps]))
        avg_doc       = float(np.mean([s.doctrine_score for s in log.steps]))
        return float(np.clip(
            outcome_score * 0.4 + efficiency * 0.25 + avg_wr * 0.2 + avg_doc * 0.15,
            0, 1
        ))

    def _write_executive_summary(
        self,
        log: EpisodeLog,
        score: float,
        turning_points: List[TurningPoint]
    ) -> str:
        outcome_map = {"win":"승리","loss":"패배","draw":"무승부"}
        outcome_kr = outcome_map.get(log.outcome, log.outcome)
        lines = [
            f"총 {log.n_steps}스텝 작전, 최종 결과: {outcome_kr}",
            f"아군 손실: {log.total_blue_casualties}명 ({log.force_reduction_ratio:.0%})"
            f"  적군 손실: {log.total_red_casualties}명",
            f"전투 수행 종합 점수: {score:.0%}",
        ]
        if turning_points:
            most_critical = max(turning_points, key=lambda t: abs(t.impact))
            lines.append(f"핵심 전환점: 스텝 {most_critical.step} — {most_critical.description}")
        return "\n".join(lines)

    def _empty_report(self, log: EpisodeLog) -> AARReport:
        return AARReport(
            episode_id=log.episode_id, outcome=log.outcome,
            executive_summary="데이터 없음", turning_points=[],
            best_decisions=[], missed_opportunities=[],
            resource_analysis={}, doctrine_compliance={},
            recommendations=["데이터 수집 후 재분석 필요"],
            lessons_learned=[], overall_score=0.0,
        )


# ── 합성 에피소드 생성 헬퍼 ──────────────────

def generate_synthetic_episode(outcome: str = "win", n_steps: int = 15, seed: int = 42) -> EpisodeLog:
    rng = np.random.RandomState(seed)
    log = EpisodeLog(
        episode_id=f"ep_{seed}_{outcome}",
        initial_blue_hc=1000, initial_red_hc=800, outcome=outcome
    )
    blue_hc, red_hc = 1000, 800
    ammo, fuel = 0.9, 0.9
    win_rate = 0.55 if outcome == "win" else 0.40

    action_names = ["대기","전진","후퇴","교전","지원","측면기동","보급요청","방어진지구축"]
    for step in range(n_steps):
        action = int(rng.choice([1,3,5,6], p=[0.3,0.3,0.2,0.2]))
        blue_cas = int(rng.randint(0, 25))
        red_cas  = int(rng.randint(5, 40)) if outcome == "win" else int(rng.randint(0, 15))
        blue_hc  = max(0, blue_hc - blue_cas)
        red_hc   = max(0, red_hc  - red_cas)
        ammo     = max(0.0, ammo - rng.uniform(0.02, 0.07))
        fuel     = max(0.0, fuel - rng.uniform(0.01, 0.04))
        if action == 6: ammo = min(1.0, ammo + 0.3)
        wr_delta = 0.02 if outcome == "win" else -0.01
        win_rate = float(np.clip(win_rate + wr_delta + rng.normal(0, 0.03), 0.05, 0.99))
        doc_score = float(np.clip(0.5 + rng.normal(0, 0.15), 0, 1))
        log.add_step(StepRecord(
            step=step, action=action,
            action_name=action_names[action],
            blue_headcount=blue_hc, red_headcount=red_hc,
            blue_casualties=blue_cas, red_casualties=red_cas,
            win_rate=win_rate, uncertainty=float(rng.uniform(0.2, 0.6)),
            ammo_level=ammo, fuel_level=fuel,
            mission_status="ongoing" if step < n_steps-1 else outcome,
            doctrine_score=doc_score,
            flanking=(action == 5),
            reward=red_cas * 0.1 - blue_cas * 0.15,
        ))
    return log


if __name__ == "__main__":
    print("="*55); print("I. 자동 전투 후 분석 (AAR) 테스트"); print("="*55)
    generator = AutoAARGenerator()
    for outcome in ["win", "loss"]:
        log = generate_synthetic_episode(outcome=outcome, n_steps=15, seed=42)
        report = generator.generate(log)
        print(report.to_report())
    print("✅ AAR 정상 동작!")
