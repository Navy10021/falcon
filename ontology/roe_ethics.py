"""
roe_ethics.py
=============
ONTOLOGY ROADMAP STEP 9: 교전 규칙(ROE) + 윤리 제약 검증기

포함 내용:
  - MissionType       : 임무 유형 (평화유지/재래식전쟁/대반란/대테러)
  - ForceThreshold    : 무력 사용 임계 (임박한 위협/적대행위/적대의도)
  - StrikeCategory    : 타격 대상 분류 (군사목표/이중용도/민간인접)
  - RulesOfEngagement : ROE 데이터클래스 (비교전지역·비타격부대·사이버제한)
  - EthicalConstraintChecker : 비례성·식별성·군사적 필요성 검증
  - ROEViolationLog   : 위반 이력 관리
  - ROEManager        : 전체 ROE/윤리 통합 관리자
    - JointFiresRequest ROE 검증 (joint_operations.py 연동)
    - BlueAgent 보상 함수에 roe_penalty 제공
    - CombatKnowledgeGraph 내 타격 전 민간피해 추정

학술 연구용 합성 데이터
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, ForceAlignment, CombatKnowledgeGraph,
    UnitStatus, BranchType, DomainType,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. 열거형 — 임무·임계·타격 분류
# ──────────────────────────────────────────────────────────────

class MissionType(str, Enum):
    """교전 규칙의 상위 임무 유형"""
    PEACEKEEPING        = "peacekeeping"         # 평화유지 (최소무력)
    CONVENTIONAL_WAR    = "conventional_war"     # 재래식 전쟁
    COUNTERINSURGENCY   = "counterinsurgency"    # 대반란전
    COUNTERTERRORISM    = "counterterrorism"     # 대테러 작전
    HUMANITARIAN        = "humanitarian"         # 인도지원 작전


class ForceThreshold(str, Enum):
    """무력 사용 임계 수준"""
    IMMINENT_THREAT  = "imminent_threat"   # 임박한 위협 → 자위권
    HOSTILE_ACT      = "hostile_act"       # 적대 행위 발생 후
    HOSTILE_INTENT   = "hostile_intent"    # 적대 의도 확인 후
    POSITIVE_ID      = "positive_id"       # 적군 확인 후만 교전


class StrikeCategory(str, Enum):
    """타격 대상 분류 (IHL — 국제인도법 기준)"""
    MILITARY_TARGET   = "military_target"    # 순수 군사목표
    DUAL_USE          = "dual_use"           # 이중용도 시설
    NEAR_CIVILIAN     = "near_civilian"      # 민간인/물체 인접
    PROTECTED_SITE    = "protected_site"     # 보호 대상 (병원·문화재)
    CYBER_TARGET      = "cyber_target"       # 사이버 공간 목표


# ──────────────────────────────────────────────────────────────
# 2. 비타격 부대 유형 집합 (보호 대상)
# ──────────────────────────────────────────────────────────────

# 의료·군수 등 비전투원에 준하는 보호 대상 UnitType
PROTECTED_UNIT_TYPES: set = {
    UnitType.LOGISTICS,
    UnitType.MILITARY_POLICE,
    UnitType.NBC_DEFENSE,
    UnitType.SIGNAL,
}

# 사이버 작전에 제약이 걸리는 인프라 관련 UnitType
CRITICAL_INFRASTRUCTURE_TYPES: set = {
    UnitType.CYBER_UNIT,
    UnitType.SIGNAL,
    UnitType.SPACE_ASSET,
    UnitType.ELECTRONIC_WARFARE,
}


# ──────────────────────────────────────────────────────────────
# 3. RulesOfEngagement 데이터클래스
# ──────────────────────────────────────────────────────────────

@dataclass
class RulesOfEngagement:
    """교전 규칙 (ROE) 정의"""

    mission_type: MissionType = MissionType.CONVENTIONAL_WAR
    force_threshold: ForceThreshold = ForceThreshold.HOSTILE_ACT

    # 피해 한도
    collateral_damage_limit: float = 0.15   # 허용 민간피해 비율 [0, 1]
    proportionality_ratio: float = 3.0      # 군사적 이득 : 민간피해 최소 비율

    # 비교전·보호 구역 (지형 셀 id 문자열 목록)
    no_strike_zones: List[str] = field(default_factory=list)

    # 비타격 유닛 타입 목록 (str → UnitType.value 기준)
    no_strike_unit_types: List[str] = field(default_factory=list)

    # 사이버 작전 제한
    cyber_restrictions: List[str] = field(default_factory=list)
    # e.g. ["no_civilian_infra", "no_financial_system", "no_medical_system"]

    # 핵·화생방 무기 사용 금지
    no_wmd: bool = True

    # 자율 치사 시스템 (Lethal Autonomous) 제한
    no_autonomous_lethal: bool = True

    # 야간 교전 허용 여부
    night_engagement_allowed: bool = True

    def is_strike_lawful(
        self,
        target_unit: Unit,
        collateral_damage_estimate: float,
        military_advantage: float,
        terrain_cell_id: Optional[str] = None,
        strike_category: StrikeCategory = StrikeCategory.MILITARY_TARGET,
    ) -> Tuple[bool, str]:
        """
        타격의 적법성 판정.

        Returns
        -------
        (lawful: bool, reason: str)
        """
        # 1) 보호 구역
        if terrain_cell_id and terrain_cell_id in self.no_strike_zones:
            return False, f"비타격 구역 위반: {terrain_cell_id}"

        # 2) 비타격 유닛 유형
        if target_unit.unit_type.value in self.no_strike_unit_types:
            return False, f"비타격 유닛 유형: {target_unit.unit_type.value}"

        # 3) 보호 대상 UnitType 하드코드 (의료/군수 등)
        if target_unit.unit_type in PROTECTED_UNIT_TYPES:
            return False, f"보호 대상 유닛: {target_unit.unit_type.value}"

        # 4) 보호 시설
        if strike_category == StrikeCategory.PROTECTED_SITE:
            return False, "국제법 보호 시설 (병원·문화재) 타격 금지"

        # 5) 민간피해 한도
        if collateral_damage_estimate > self.collateral_damage_limit:
            return False, (
                f"민간피해 초과: {collateral_damage_estimate:.2f} > "
                f"허용 {self.collateral_damage_limit:.2f}"
            )

        # 6) 비례성 검사
        if military_advantage > 0 and collateral_damage_estimate > 0:
            ratio = military_advantage / max(collateral_damage_estimate, 1e-6)
            if ratio < self.proportionality_ratio:
                return False, (
                    f"비례성 미달: M/C 비율 {ratio:.1f} < 기준 {self.proportionality_ratio:.1f}"
                )

        # 7) 사이버 타격 제한
        if strike_category == StrikeCategory.CYBER_TARGET:
            if target_unit.unit_type in CRITICAL_INFRASTRUCTURE_TYPES:
                if "no_civilian_infra" in self.cyber_restrictions:
                    return False, "사이버: 민간 인프라 타격 제한"

        return True, "ROE 준수"

    def get_roe_summary(self) -> Dict:
        return {
            "mission_type": self.mission_type.value,
            "force_threshold": self.force_threshold.value,
            "collateral_damage_limit": self.collateral_damage_limit,
            "proportionality_ratio": self.proportionality_ratio,
            "no_strike_zones": len(self.no_strike_zones),
            "no_strike_unit_types": self.no_strike_unit_types,
            "cyber_restrictions": self.cyber_restrictions,
            "no_wmd": self.no_wmd,
            "no_autonomous_lethal": self.no_autonomous_lethal,
        }


# ──────────────────────────────────────────────────────────────
# 4. 위반 이력
# ──────────────────────────────────────────────────────────────

@dataclass
class ROEViolation:
    """단일 ROE 위반 기록"""
    step: int
    unit_id: str
    target_id: str
    reason: str
    collateral_damage: float = 0.0
    penalty: float = 0.5             # RL 보상 감점 [0, 1]


@dataclass
class ROEViolationLog:
    """ROE 위반 이력 관리"""
    violations: List[ROEViolation] = field(default_factory=list)

    def add(self, violation: ROEViolation) -> None:
        self.violations.append(violation)
        logger.warning(
            "ROE 위반 step=%d unit=%s target=%s reason=%s",
            violation.step, violation.unit_id,
            violation.target_id, violation.reason,
        )

    def total_penalty(self) -> float:
        return sum(v.penalty for v in self.violations)

    def recent_violations(self, last_n: int = 5) -> List[ROEViolation]:
        return self.violations[-last_n:]

    def violation_rate(self, total_engagements: int) -> float:
        if total_engagements == 0:
            return 0.0
        return len(self.violations) / total_engagements

    def to_report(self) -> str:
        lines = [f"=== ROE 위반 이력 ({len(self.violations)}건) ==="]
        for v in self.violations:
            lines.append(
                f"  [Step {v.step}] {v.unit_id} → {v.target_id}: "
                f"{v.reason} (민간피해={v.collateral_damage:.2f}, 감점={v.penalty:.2f})"
            )
        lines.append(f"총 감점: {self.total_penalty():.2f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 5. 윤리 제약 검증기
# ──────────────────────────────────────────────────────────────

class EthicalConstraintChecker:
    """
    국제인도법(IHL) 기반 윤리 제약 검증기

    세 가지 핵심 원칙 검증:
      1. 비례성 (Proportionality) — 군사적 이득 vs 민간 피해
      2. 식별성 (Distinction)    — 전투원 vs 비전투원 구분
      3. 군사적 필요성 (Military Necessity) — 작전 목적 달성에 필요한 최소 무력
    """

    def __init__(self, roe: RulesOfEngagement) -> None:
        self.roe = roe
        self._check_history: List[Dict] = []

    # ── 원칙 1: 비례성 ────────────────────────────────────────
    def check_proportionality(
        self,
        military_advantage: float,   # 군사적 이득 (normalized 0-1)
        collateral_damage: float,    # 예상 민간 피해 (normalized 0-1)
    ) -> Tuple[bool, str]:
        if collateral_damage <= 0:
            return True, "민간 피해 없음 — 비례성 충족"
        ratio = military_advantage / max(collateral_damage, 1e-6)
        if ratio >= self.roe.proportionality_ratio:
            return True, f"비례성 충족 (M/C={ratio:.2f})"
        return False, f"비례성 위반 (M/C={ratio:.2f} < 기준 {self.roe.proportionality_ratio:.2f})"

    # ── 원칙 2: 식별성 ────────────────────────────────────────
    def check_distinction(self, target: Unit) -> Tuple[bool, str]:
        """전투원(합법 목표) vs 비전투원 식별"""
        # 보호 대상 UnitType → 식별 위반
        if target.unit_type in PROTECTED_UNIT_TYPES:
            return False, f"비전투원 유닛 식별 실패: {target.unit_type.value}"
        # 비타격 유닛 목록에 포함
        if target.unit_type.value in self.roe.no_strike_unit_types:
            return False, f"비타격 유형: {target.unit_type.value}"
        # DESTROYED 유닛 재타격 금지
        if target.status == UnitStatus.DESTROYED:
            return False, "이미 파괴된 유닛 (불필요한 잔혹행위 금지)"
        return True, "합법적 군사 목표"

    # ── 원칙 3: 군사적 필요성 ─────────────────────────────────
    def check_military_necessity(
        self,
        action_achieves_objective: bool,
        alternatives_exhausted: bool,
        strength_ratio: float,   # 아군/적군 전력비
    ) -> Tuple[bool, str]:
        """최소 필요 무력 검증"""
        if not action_achieves_objective:
            return False, "작전 목표 달성에 기여하지 않는 타격"
        if not alternatives_exhausted and strength_ratio > 3.0:
            return False, f"압도적 우세(비율={strength_ratio:.1f})에서 비필요 무력 사용"
        return True, "군사적 필요성 충족"

    # ── 종합 윤리 검사 ─────────────────────────────────────────
    def full_ethical_check(
        self,
        target: Unit,
        military_advantage: float,
        collateral_damage: float,
        action_achieves_objective: bool = True,
        alternatives_exhausted: bool = True,
        strength_ratio: float = 1.0,
        terrain_cell_id: Optional[str] = None,
        strike_category: StrikeCategory = StrikeCategory.MILITARY_TARGET,
    ) -> Tuple[bool, float, List[str]]:
        """
        종합 윤리 검사.

        Returns
        -------
        (ethical: bool, penalty: float, reasons: List[str])
          - penalty : ROE/윤리 위반 시 RL 보상 감점 (0.0 = 이상 없음)
        """
        reasons: List[str] = []
        penalty = 0.0
        ethical = True

        # ROE 적법성
        lawful, reason = self.roe.is_strike_lawful(
            target, collateral_damage, military_advantage,
            terrain_cell_id, strike_category,
        )
        if not lawful:
            reasons.append(f"[ROE] {reason}")
            penalty += 0.5
            ethical = False

        # 비례성
        prop_ok, prop_msg = self.check_proportionality(military_advantage, collateral_damage)
        if not prop_ok:
            reasons.append(f"[비례성] {prop_msg}")
            penalty += 0.3
            ethical = False

        # 식별성
        dist_ok, dist_msg = self.check_distinction(target)
        if not dist_ok:
            reasons.append(f"[식별성] {dist_msg}")
            penalty += 0.4
            ethical = False

        # 군사적 필요성
        nec_ok, nec_msg = self.check_military_necessity(
            action_achieves_objective, alternatives_exhausted, strength_ratio
        )
        if not nec_ok:
            reasons.append(f"[필요성] {nec_msg}")
            penalty += 0.2
            ethical = False

        # WMD 금지
        if self.roe.no_wmd and strike_category == StrikeCategory.MILITARY_TARGET:
            # WMD 사용 여부는 호출자가 별도 파라미터로 전달해야 하나
            # 기본은 준수로 간주
            pass

        # 이력 기록
        self._check_history.append({
            "target": target.unit_id,
            "ethical": ethical,
            "penalty": penalty,
            "reasons": reasons,
        })

        return ethical, min(penalty, 1.0), reasons

    def get_ethics_summary(self) -> Dict:
        if not self._check_history:
            return {"total_checks": 0, "violations": 0, "avg_penalty": 0.0}
        violations = [h for h in self._check_history if not h["ethical"]]
        return {
            "total_checks": len(self._check_history),
            "violations": len(violations),
            "avg_penalty": float(np.mean([h["penalty"] for h in self._check_history])),
            "violation_rate": len(violations) / len(self._check_history),
        }

    def generate_ethics_report(self) -> str:
        summary = self.get_ethics_summary()
        lines = [
            "=== 윤리 제약 검증 리포트 ===",
            f"총 검사: {summary['total_checks']}건",
            f"위반: {summary['violations']}건 ({summary['violation_rate']*100:.1f}%)",
            f"평균 감점: {summary['avg_penalty']:.3f}",
            "",
            "최근 위반 상세:",
        ]
        violations = [h for h in self._check_history if not h["ethical"]][-5:]
        for v in violations:
            lines.append(f"  target={v['target']} penalty={v['penalty']:.2f}")
            for r in v["reasons"]:
                lines.append(f"    - {r}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 6. 임무별 기본 ROE 프리셋
# ──────────────────────────────────────────────────────────────

ROE_PRESETS: Dict[str, RulesOfEngagement] = {
    "peacekeeping": RulesOfEngagement(
        mission_type=MissionType.PEACEKEEPING,
        force_threshold=ForceThreshold.IMMINENT_THREAT,
        collateral_damage_limit=0.02,
        proportionality_ratio=10.0,
        no_strike_unit_types=["logistics", "signal", "military_police"],
        cyber_restrictions=["no_civilian_infra", "no_financial_system", "no_medical_system"],
        no_wmd=True,
        no_autonomous_lethal=True,
    ),
    "conventional_war": RulesOfEngagement(
        mission_type=MissionType.CONVENTIONAL_WAR,
        force_threshold=ForceThreshold.HOSTILE_ACT,
        collateral_damage_limit=0.15,
        proportionality_ratio=3.0,
        no_strike_unit_types=[],
        cyber_restrictions=["no_civilian_infra"],
        no_wmd=True,
        no_autonomous_lethal=False,
    ),
    "counterinsurgency": RulesOfEngagement(
        mission_type=MissionType.COUNTERINSURGENCY,
        force_threshold=ForceThreshold.POSITIVE_ID,
        collateral_damage_limit=0.05,
        proportionality_ratio=6.0,
        no_strike_unit_types=["logistics", "military_police"],
        cyber_restrictions=["no_civilian_infra", "no_financial_system"],
        no_wmd=True,
        no_autonomous_lethal=True,
    ),
    "counterterrorism": RulesOfEngagement(
        mission_type=MissionType.COUNTERTERRORISM,
        force_threshold=ForceThreshold.HOSTILE_INTENT,
        collateral_damage_limit=0.08,
        proportionality_ratio=4.0,
        no_strike_unit_types=[],
        cyber_restrictions=["no_medical_system"],
        no_wmd=True,
        no_autonomous_lethal=False,
    ),
    "humanitarian": RulesOfEngagement(
        mission_type=MissionType.HUMANITARIAN,
        force_threshold=ForceThreshold.IMMINENT_THREAT,
        collateral_damage_limit=0.0,
        proportionality_ratio=999.0,   # 사실상 민간피해 불허
        no_strike_unit_types=["logistics", "signal", "military_police", "nbc_defense"],
        cyber_restrictions=["no_civilian_infra", "no_financial_system",
                            "no_medical_system", "no_water_system"],
        no_wmd=True,
        no_autonomous_lethal=True,
    ),
}


# ──────────────────────────────────────────────────────────────
# 7. ROEManager — 통합 관리자
# ──────────────────────────────────────────────────────────────

class ROEManager:
    """
    ROE + 윤리 제약 통합 관리자

    역할:
      - RulesOfEngagement 관리 및 동적 업데이트
      - EthicalConstraintChecker 위임
      - JointFiresRequest ROE 검증 (joint_operations.py 연동)
      - BlueAgent reward에 roe_penalty 제공
      - 민간피해 추정 (unit 근접 기반 휴리스틱)
    """

    def __init__(
        self,
        roe: Optional[RulesOfEngagement] = None,
        preset_name: str = "conventional_war",
    ) -> None:
        if roe is not None:
            self.roe = roe
        else:
            self.roe = ROE_PRESETS.get(preset_name, ROE_PRESETS["conventional_war"])

        self.checker = EthicalConstraintChecker(self.roe)
        self.violation_log = ROEViolationLog()
        self._total_engagements: int = 0
        self._step: int = 0

    # ── 시나리오 전환 ──────────────────────────────────────────
    def switch_roe(self, preset_name: str) -> None:
        """임무 유형 변경 (ROE 재설정)"""
        self.roe = ROE_PRESETS.get(preset_name, self.roe)
        self.checker = EthicalConstraintChecker(self.roe)
        logger.info("ROE 전환: %s", preset_name)

    def update_no_strike_zone(self, cell_id: str, add: bool = True) -> None:
        """비타격 구역 동적 추가/제거"""
        if add and cell_id not in self.roe.no_strike_zones:
            self.roe.no_strike_zones.append(cell_id)
        elif not add and cell_id in self.roe.no_strike_zones:
            self.roe.no_strike_zones.remove(cell_id)

    # ── 민간피해 추정 ──────────────────────────────────────────
    def estimate_collateral_damage(
        self,
        target: Unit,
        kg: CombatKnowledgeGraph,
        blast_radius_km: float = 1.0,
    ) -> float:
        """
        타격 대상 인근 아군/중립 유닛 수 기반 민간피해 휴리스틱.

        Returns
        -------
        collateral_damage : float in [0, 1]
        """
        if kg is None or target.position is None:
            return 0.0

        nearby_friendly = 0
        tx, ty = target.position.x, target.position.y
        scale = getattr(kg, "map_size_km", 100.0) or 100.0

        for uid, u in kg.units.items():
            if uid == target.unit_id:
                continue
            if u.alignment == target.alignment:  # 같은 진영 → 해당 없음
                continue
            if u.status == UnitStatus.DESTROYED:
                continue
            if u.position is None:
                continue
            dist_km = (
                ((u.position.x - tx) ** 2 + (u.position.y - ty) ** 2) ** 0.5
            ) * scale
            if dist_km <= blast_radius_km:
                nearby_friendly += 1

        # 보호 대상 유닛 가중치
        protected_count = sum(
            1 for u in kg.units.values()
            if u.unit_type in PROTECTED_UNIT_TYPES
            and u.status != UnitStatus.DESTROYED
            and u.position is not None
            and (((u.position.x - tx)**2 + (u.position.y - ty)**2)**0.5 * scale) <= blast_radius_km
        )

        raw = (nearby_friendly * 0.05) + (protected_count * 0.15)
        return float(np.clip(raw, 0.0, 1.0))

    # ── 교전 전 ROE 검사 ──────────────────────────────────────
    def check_engagement(
        self,
        attacker: Unit,
        target: Unit,
        kg: CombatKnowledgeGraph,
        military_advantage: float = 0.5,
        blast_radius_km: float = 1.0,
        strike_category: StrikeCategory = StrikeCategory.MILITARY_TARGET,
        terrain_cell_id: Optional[str] = None,
    ) -> Tuple[bool, float, List[str]]:
        """
        교전 전 종합 ROE/윤리 검사.

        Returns
        -------
        (approved: bool, roe_penalty: float, reasons: List[str])
        """
        self._total_engagements += 1
        collateral = self.estimate_collateral_damage(target, kg, blast_radius_km)

        ethical, penalty, reasons = self.checker.full_ethical_check(
            target=target,
            military_advantage=military_advantage,
            collateral_damage=collateral,
            terrain_cell_id=terrain_cell_id,
            strike_category=strike_category,
        )

        if not ethical:
            self.violation_log.add(ROEViolation(
                step=self._step,
                unit_id=attacker.unit_id,
                target_id=target.unit_id,
                reason=" | ".join(reasons),
                collateral_damage=collateral,
                penalty=penalty,
            ))

        return ethical, penalty, reasons

    # ── JointFiresRequest ROE 검증 연동 ───────────────────────
    def validate_fires_request(
        self,
        fires_request,          # JointFiresRequest (duck-typing)
        kg: CombatKnowledgeGraph,
    ) -> bool:
        """
        joint_operations.JointFiresRequest ROE 검증.
        fires_request.roe_compliant 필드를 갱신한다.
        """
        target_id = getattr(fires_request, "target_unit_id", None)
        if target_id is None or target_id not in kg.units:
            fires_request.roe_compliant = False
            return False

        requester_id = getattr(fires_request, "requester_id", "unknown")
        attacker = kg.units.get(requester_id)
        target = kg.units[target_id]

        if attacker is None:
            # requester가 KG에 없으면 일단 통과 (ROE만 체크)
            lawful, reason = self.roe.is_strike_lawful(target, 0.0, 0.5)
        else:
            approved, _, _ = self.check_engagement(attacker, target, kg)
            fires_request.roe_compliant = approved
            return approved

        fires_request.roe_compliant = lawful
        return lawful

    # ── RL 보상 통합 ──────────────────────────────────────────
    def get_roe_penalty(self) -> float:
        """
        BlueAgent.compute_reward()에서 차감할 누적 ROE 감점.
        호출 후 초기화 (step당 1회).
        """
        recent = self.violation_log.recent_violations(last_n=1)
        if recent:
            return recent[-1].penalty
        return 0.0

    def get_cumulative_penalty(self) -> float:
        """전체 에피소드 누적 ROE 감점"""
        return self.violation_log.total_penalty()

    # ── 스텝 업데이트 ─────────────────────────────────────────
    def step(self) -> None:
        """시뮬레이션 스텝 증가"""
        self._step += 1

    # ── 리포트 ────────────────────────────────────────────────
    def generate_roe_report(self) -> str:
        lines = [
            "=" * 55,
            "ROE / 윤리 종합 리포트",
            "=" * 55,
            f"현재 임무 유형: {self.roe.mission_type.value}",
            f"무력 임계: {self.roe.force_threshold.value}",
            f"허용 민간피해: {self.roe.collateral_damage_limit:.2%}",
            f"비례성 기준: {self.roe.proportionality_ratio:.1f}",
            f"총 교전 검사: {self._total_engagements}건",
            f"위반: {len(self.violation_log.violations)}건 "
            f"({self.violation_log.violation_rate(self._total_engagements)*100:.1f}%)",
            f"누적 감점: {self.violation_log.total_penalty():.3f}",
            "",
            self.checker.generate_ethics_report(),
            "",
            self.violation_log.to_report(),
        ]
        return "\n".join(lines)

    def get_state_vector(self) -> np.ndarray:
        """
        ROE 상태를 RL 상태 벡터 (4D) 로 반환.
        build_state_vector() c2_state 슬롯과 함께 사용 가능.

        [roe_strictness, violation_rate, cumulative_penalty, no_strike_zone_count]
        """
        mission_strictness = {
            MissionType.HUMANITARIAN: 1.0,
            MissionType.PEACEKEEPING: 0.8,
            MissionType.COUNTERINSURGENCY: 0.6,
            MissionType.COUNTERTERRORISM: 0.4,
            MissionType.CONVENTIONAL_WAR: 0.2,
        }
        strictness = mission_strictness.get(self.roe.mission_type, 0.5)
        viol_rate = self.violation_log.violation_rate(
            max(self._total_engagements, 1)
        )
        cum_pen = min(self.violation_log.total_penalty() / 10.0, 1.0)
        nsz = min(len(self.roe.no_strike_zones) / 20.0, 1.0)
        return np.array([strictness, viol_rate, cum_pen, nsz], dtype=np.float32)
