"""
doctrine_encoder.py — C. 교리 인코딩 레이어
ADP(Army Doctrine Publication) 9대 원칙을 하드/소프트 제약으로 인코딩
위반 탐지 + 보상 패널티 + 교리 언어 설명 생성
학술 연구용 합성 데이터
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class DoctrinePrinciple(Enum):
    MASS        = "mass"        # 전투력 집중
    OBJECTIVE   = "objective"   # 목표 명확성
    OFFENSIVE   = "offensive"   # 공세 주도권
    SECURITY    = "security"    # 방호 / 기습 방지
    SURPRISE    = "surprise"    # 기습
    ECONOMY     = "economy"     # 전력 절약
    MANEUVER    = "maneuver"    # 기동
    UNITY       = "unity"       # 통일된 지휘
    SIMPLICITY  = "simplicity"  # 단순성


PRINCIPLE_DESCRIPTIONS = {
    DoctrinePrinciple.MASS:       "결정적 지점에 전투력을 집중한다",
    DoctrinePrinciple.OBJECTIVE:  "명확하고 달성 가능한 임무 목표를 유지한다",
    DoctrinePrinciple.OFFENSIVE:  "공세 주도권을 확보하고 유지한다",
    DoctrinePrinciple.SECURITY:   "적의 기습으로부터 아군을 보호한다",
    DoctrinePrinciple.SURPRISE:   "적이 예상치 못한 시간·장소·방법으로 타격한다",
    DoctrinePrinciple.ECONOMY:    "비결정적 지점에서 전력을 절약한다",
    DoctrinePrinciple.MANEUVER:   "적 취약점으로 기동하여 우세를 달성한다",
    DoctrinePrinciple.UNITY:      "단일 지휘 하에 통합된 노력을 보장한다",
    DoctrinePrinciple.SIMPLICITY: "명확하고 단순한 계획으로 혼선을 방지한다",
}


@dataclass
class DoctrineViolation:
    principle: DoctrinePrinciple
    severity: str           # "minor" | "major" | "critical"
    description: str
    penalty: float          # 보상 패널티 값
    step: int
    evidence: Dict          # 위반 근거 수치

    def to_report(self) -> str:
        sev_icon = {"minor": "⚠️", "major": "🟠", "critical": "🔴"}.get(self.severity, "⚠️")
        return (f"  {sev_icon} [{self.principle.value.upper()}] {self.description}"
                f" (패널티=-{self.penalty:.2f})")


@dataclass
class DoctrineCompliance:
    step: int
    total_score: float          # [0,1] 높을수록 교리 준수
    principle_scores: Dict[str, float]
    violations: List[DoctrineViolation]
    positive_actions: List[str]

    def summary(self) -> str:
        lines = [f"📋 교리 준수도: {self.total_score:.0%} (스텝 {self.step})"]
        for p, s in sorted(self.principle_scores.items(), key=lambda x: x[1]):
            bar = "█" * int(s * 10)
            lines.append(f"  {p:12s}: {s:.2f} [{bar:<10s}]")
        if self.violations:
            lines.append("  위반 사항:")
            for v in self.violations:
                lines.append(v.to_report())
        return "\n".join(lines)


class DoctrineEncoder:
    """
    ADP 9대 원칙 기반 교리 준수 평가기

    기능:
    1. 매 스텝 전투 상태 → 교리 원칙별 준수도 측정
    2. 위반 탐지 + 보상 패널티 계산
    3. 교리 언어 설명 생성 ("왜 이 결정인가")
    4. 누적 준수율 추적
    """

    VIOLATION_THRESHOLDS = {
        DoctrinePrinciple.MASS:       0.35,   # 전투력 집중도 하한
        DoctrinePrinciple.OFFENSIVE:  0.40,   # 공세 유지 임계치
        DoctrinePrinciple.SECURITY:   0.25,   # 방호 하한
        DoctrinePrinciple.ECONOMY:    0.70,   # 비결정 지점 낭비 상한
    }

    PENALTY_WEIGHTS = {
        "minor":    0.05,
        "major":    0.20,
        "critical": 0.50,
    }

    def __init__(self):
        self.history: List[DoctrineCompliance] = []
        self.cumulative_penalty: float = 0.0

    def evaluate(self, kg, step: int, step_result: Dict) -> DoctrineCompliance:
        """
        전투 상태 + 스텝 결과 → 교리 준수도 평가
        """
        from ontology.combat_schema import ForceAlignment, UnitStatus

        blue = [u for u in kg.units.values()
                if u.alignment == ForceAlignment.BLUE and u.status != UnitStatus.DESTROYED]
        red  = [u for u in kg.units.values()
                if u.alignment == ForceAlignment.RED  and u.status != UnitStatus.DESTROYED]

        blue_cp  = sum(u.combat_power for u in blue)
        red_cp   = sum(u.combat_power for u in red) + 1e-8
        total_cp = blue_cp + red_cp

        violations: List[DoctrineViolation] = []
        scores: Dict[str, float] = {}
        positives: List[str] = []

        # ① MASS — 전투력 집중
        # 활성 유닛 중 고전투력 유닛 비율
        if blue:
            high_cp_units = [u for u in blue if u.combat_power > 2.0]
            mass_ratio = len(high_cp_units) / len(blue)
            scores["mass"] = mass_ratio
            if mass_ratio < self.VIOLATION_THRESHOLDS[DoctrinePrinciple.MASS]:
                sev = "critical" if mass_ratio < 0.15 else "major"
                violations.append(DoctrineViolation(
                    DoctrinePrinciple.MASS, sev,
                    f"전투력 집중 부족 (고전투력 유닛 {mass_ratio:.0%})",
                    self.PENALTY_WEIGHTS[sev], step, {"mass_ratio": mass_ratio}
                ))
            elif mass_ratio > 0.6:
                positives.append("전투력 효과적으로 집중됨 (MASS 원칙 충족)")
        else:
            scores["mass"] = 0.0

        # ② OBJECTIVE — 목표 달성 진행도
        red_loss_ratio = 1 - (red_cp / (blue_cp + red_cp + 1e-8) * 2)
        obj_score = float(np.clip(red_loss_ratio, 0, 1))
        scores["objective"] = obj_score

        # ③ OFFENSIVE — 공세 주도권 (Blue가 Red보다 강한가)
        offensive_score = float(np.clip(blue_cp / (red_cp + 1e-8) / 2, 0, 1))
        scores["offensive"] = offensive_score
        if offensive_score < self.VIOLATION_THRESHOLDS[DoctrinePrinciple.OFFENSIVE]:
            violations.append(DoctrineViolation(
                DoctrinePrinciple.OFFENSIVE, "major",
                f"공세 주도권 상실 (Blue/Red 전투력비 {blue_cp/red_cp:.2f})",
                self.PENALTY_WEIGHTS["major"], step,
                {"blue_cp": blue_cp, "red_cp": red_cp}
            ))

        # ④ SECURITY — 방호 (탄약/연료 수준)
        if blue:
            avg_ammo = float(np.mean([u.capability.ammo_level for u in blue]))
            avg_fuel = float(np.mean([u.capability.fuel_level  for u in blue]))
            sec_score = (avg_ammo + avg_fuel) / 2
            scores["security"] = sec_score
            if sec_score < self.VIOLATION_THRESHOLDS[DoctrinePrinciple.SECURITY]:
                violations.append(DoctrineViolation(
                    DoctrinePrinciple.SECURITY, "critical",
                    f"자원 고갈로 방호 능력 상실 (탄약 {avg_ammo:.0%}, 연료 {avg_fuel:.0%})",
                    self.PENALTY_WEIGHTS["critical"], step,
                    {"ammo": avg_ammo, "fuel": avg_fuel}
                ))
        else:
            scores["security"] = 0.0

        # ⑤ SURPRISE — 기습 (측면/야간 공격 시도 여부 — 근사치)
        # 가상: 결과에 측면 기동 기록이 있으면 기습 요소 있음
        surprise_score = 0.5  # 기본값 (실제 구현: 기동 엔진 연동)
        if step_result.get("flanking_units", 0) > 0:
            surprise_score = 0.8
            positives.append("측면 기동으로 기습 요소 달성 (SURPRISE 원칙)")
        scores["surprise"] = surprise_score

        # ⑥ ECONOMY — 전력 절약 (손실 대비 효과)
        blue_cas = step_result.get("blue_casualties", 0)
        red_cas  = step_result.get("red_casualties", 0)
        total_cas = blue_cas + red_cas + 1
        economy_score = float(np.clip(1 - blue_cas / total_cas, 0, 1))
        scores["economy"] = economy_score
        if blue_cas > red_cas * 2:
            violations.append(DoctrineViolation(
                DoctrinePrinciple.ECONOMY, "major",
                f"아군 손실 과다 (Blue {blue_cas}명 vs Red {red_cas}명)",
                self.PENALTY_WEIGHTS["major"], step,
                {"blue_cas": blue_cas, "red_cas": red_cas}
            ))

        # ⑦ MANEUVER — 기동 (엔진 연동, 근사치)
        maneuver_score = 0.5
        if step_result.get("n_engageable", 0) > 0:
            maneuver_score = min(0.9, 0.5 + step_result["n_engageable"] * 0.1)
        scores["maneuver"] = maneuver_score

        # ⑧ UNITY — 통일된 지휘 (활성 유닛 수 / 초기 유닛 수)
        n_blue_active = len(blue)
        unity_score = float(np.clip(n_blue_active / max(8, n_blue_active + 2), 0, 1))
        scores["unity"] = unity_score

        # ⑨ SIMPLICITY — 단순성 (위반 수 적을수록 단순)
        simplicity_score = max(0.0, 1.0 - len(violations) * 0.15)
        scores["simplicity"] = simplicity_score

        # 전체 준수도
        total = float(np.mean(list(scores.values())))
        penalty = sum(v.penalty for v in violations)
        self.cumulative_penalty += penalty

        compliance = DoctrineCompliance(
            step=step, total_score=total,
            principle_scores=scores,
            violations=violations,
            positive_actions=positives,
        )
        self.history.append(compliance)
        return compliance

    def compute_doctrine_penalty(self, compliance: DoctrineCompliance) -> float:
        """PPO 보상 함수용 교리 패널티"""
        return -sum(v.penalty for v in compliance.violations)

    def generate_explanation(
        self,
        action: int,
        compliance: DoctrineCompliance,
        action_names: Optional[List[str]] = None
    ) -> str:
        """
        행동 결정의 교리적 설명 생성
        "왜 이 행동인가" → 군사 교리 언어
        """
        if action_names is None:
            action_names = ["대기","전진","후퇴","교전","지원","측면기동","보급요청","방어진지구축"]

        action_name = action_names[action] if action < len(action_names) else str(action)

        # 교리 원칙 → 행동 연관성 매핑
        action_doctrine_map = {
            0: [DoctrinePrinciple.SECURITY, DoctrinePrinciple.ECONOMY],
            1: [DoctrinePrinciple.OFFENSIVE, DoctrinePrinciple.MANEUVER],
            2: [DoctrinePrinciple.SECURITY, DoctrinePrinciple.ECONOMY],
            3: [DoctrinePrinciple.MASS, DoctrinePrinciple.OFFENSIVE],
            4: [DoctrinePrinciple.UNITY, DoctrinePrinciple.MASS],
            5: [DoctrinePrinciple.SURPRISE, DoctrinePrinciple.MANEUVER],
            6: [DoctrinePrinciple.SECURITY, DoctrinePrinciple.ECONOMY],
            7: [DoctrinePrinciple.SECURITY, DoctrinePrinciple.SIMPLICITY],
        }

        related_principles = action_doctrine_map.get(action, [])
        lines = [f"🎯 [{action_name}] 선택 근거:"]

        for p in related_principles:
            score = compliance.principle_scores.get(p.value, 0.5)
            desc = PRINCIPLE_DESCRIPTIONS[p]
            if score > 0.6:
                lines.append(f"  ✅ {p.value.upper()}: {desc} (준수도 {score:.0%})")
            else:
                lines.append(f"  ⚠️ {p.value.upper()}: {desc} — 보완 필요 ({score:.0%})")

        if compliance.violations:
            worst = max(compliance.violations, key=lambda v: v.penalty)
            lines.append(f"  ⚡ 주요 위반: {worst.description}")

        return "\n".join(lines)

    def generate_doctrine_report(self) -> str:
        """전체 에피소드 교리 준수 리포트"""
        if not self.history:
            return "데이터 없음"

        avg_score = float(np.mean([c.total_score for c in self.history]))
        total_violations = sum(len(c.violations) for c in self.history)

        # 원칙별 평균 점수
        principle_avgs = {}
        for p in DoctrinePrinciple:
            vals = [c.principle_scores.get(p.value, 0) for c in self.history]
            principle_avgs[p.value] = float(np.mean(vals))

        worst_principle = min(principle_avgs, key=principle_avgs.get)
        best_principle  = max(principle_avgs, key=principle_avgs.get)

        lines = [
            "=" * 50,
            "📚 교리 준수 최종 리포트",
            "=" * 50,
            f"전체 준수도: {avg_score:.0%}",
            f"총 위반 건수: {total_violations}건",
            f"누적 패널티: -{self.cumulative_penalty:.2f}",
            f"최고 원칙: {best_principle} ({principle_avgs[best_principle]:.0%})",
            f"최저 원칙: {worst_principle} ({principle_avgs[worst_principle]:.0%})",
            "",
            "[원칙별 평균 준수도]",
        ]
        for p, s in sorted(principle_avgs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(s * 15)
            lines.append(f"  {p:12s}: {s:.0%} [{bar:<15s}]")

        lines.append("=" * 50)
        return "\n".join(lines)

    def get_compliance_vector(self) -> np.ndarray:
        """최신 준수도를 9D 벡터로 반환 (PPO 상태 통합용)"""
        if not self.history:
            return np.full(9, 0.5, dtype=np.float32)
        latest = self.history[-1]
        return np.array([
            latest.principle_scores.get(p.value, 0.5)
            for p in DoctrinePrinciple
        ], dtype=np.float32)


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine
    from simulator.resource_manager import ResourceManager

    print("=" * 55)
    print("C. 교리 인코딩 레이어 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    engine = MixedLanchesterEngine(seed=42)
    rm = ResourceManager(seed=42)
    doctrine = DoctrineEncoder()

    print("\n[10 스텝 교리 준수도 추적]")
    for step in range(10):
        result = engine.run_step_mixed(kg, resource_manager=rm)
        compliance = doctrine.evaluate(kg, step, result)
        penalty = doctrine.compute_doctrine_penalty(compliance)
        print(f"  스텝 {step+1:2d}: 준수도={compliance.total_score:.0%}  "
              f"위반={len(compliance.violations)}건  패널티={penalty:.2f}")
        if step == 3:
            exp = doctrine.generate_explanation(5, compliance)   # 측면기동
            print(f"\n  [행동 설명] (스텝 4, 측면기동)\n{exp}\n")

    print(doctrine.generate_doctrine_report())
    vec = doctrine.get_compliance_vector()
    print(f"\n[PPO 상태 벡터 9D]: {vec.round(2)}")
    print("✅ 교리 인코딩 정상 동작!")
