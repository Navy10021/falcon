"""
joint_operations.py
===================
ONTOLOGY ROADMAP STEP 6: 지휘통제(C2) + 합동작전 온톨로지

포함 내용:
  - CommandStructure  : 지휘 계층 구조 (HQ → 예하 부대)
  - JointFiresRequest : 합동 화력 지원 요청 (CAS·해군포·로켓·사이버타격)
  - C2Link            : 통신 링크 품질 관리
  - JointOperationsManager : C2·합동화력 통합 관리자
    - RL 에이전트 상태 벡터에 c2_quality / joint_fires_available 값 제공
    - 전투 결과에 합동화력 보너스 적용
    - 사이버·EW 공격에 의한 C2 약화 처리

학술 연구용 합성 데이터
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, ForceAlignment, CombatKnowledgeGraph, Position,
    UnitStatus, BranchType, DomainType,
)


# ──────────────────────────────────────────────────────────────
# 1. 통신 링크
# ──────────────────────────────────────────────────────────────

@dataclass
class C2Link:
    """두 부대 간 통신 링크"""
    src_id: str
    dst_id: str
    quality: float = 1.0         # 통신 품질 [0, 1]
    jamming_level: float = 0.0   # 재밍 수준 [0, 1]
    encrypted: bool = True

    @property
    def effective_quality(self) -> float:
        """실효 통신 품질 (재밍 적용)"""
        return float(np.clip(self.quality * (1.0 - 0.7 * self.jamming_level), 0.0, 1.0))


# ──────────────────────────────────────────────────────────────
# 2. 지휘 계층 구조
# ──────────────────────────────────────────────────────────────

@dataclass
class CommandStructure:
    """
    지휘 계층 구조 (단일 사령부 기준)

    headquarters_id : 사령부 / 지휘소 유닛 ID
    subordinate_ids : 직속 예하 부대 IDs
    supported_ids   : 지원 관계 부대 IDs (배속 아님)
    """
    headquarters_id: str
    subordinate_ids: List[str] = field(default_factory=list)
    supported_ids:   List[str] = field(default_factory=list)
    communication_quality: float = 1.0
    c2_degradation: float = 0.0        # 사이버·EW 공격 등으로 인한 C2 약화

    def effective_command_quality(self) -> float:
        """실효 지휘통제 품질"""
        return float(np.clip(
            self.communication_quality * (1.0 - self.c2_degradation), 0.0, 1.0
        ))

    def all_subordinate_ids(self) -> List[str]:
        return list(set(self.subordinate_ids + self.supported_ids))


# ──────────────────────────────────────────────────────────────
# 3. 합동 화력 지원
# ──────────────────────────────────────────────────────────────

SUPPORT_TYPE_EFFECTIVENESS: Dict[str, float] = {
    "cas":           0.70,   # 근접항공지원 (CAS)
    "naval_gunfire": 0.65,   # 함포 사격
    "rocket":        0.75,   # 다련장로켓
    "missile_strike":0.85,   # 정밀미사일
    "cyber_strike":  0.60,   # 사이버 공격
    "space_isr":     0.50,   # 우주 ISR 지원 (정보제공)
    "ew_jam":        0.55,   # 전자전 재밍
}


@dataclass
class JointFiresRequest:
    """합동 화력 지원 요청·결과"""
    requester_id:     str
    supporter_id:     str
    support_type:     str           # SUPPORT_TYPE_EFFECTIVENESS 키
    target_id:        Optional[str] = None
    target_location:  Optional[Position] = None
    effectiveness:    float = 0.0   # 실제 효과 [0, 1] (처리 후 기록)
    response_delay_steps: int = 1   # 응답 지연 스텝
    roe_compliant:    bool = True
    processed:        bool = False

    def base_effectiveness(self) -> float:
        return SUPPORT_TYPE_EFFECTIVENESS.get(self.support_type, 0.5)


# ──────────────────────────────────────────────────────────────
# 4. 합동작전 관리자
# ──────────────────────────────────────────────────────────────

class JointOperationsManager:
    """
    C2·합동화력·EW 통합 관리자

    파이프라인 연동:
      1. build_state_vector() → c2_quality / joint_fires_available 인자로 전달
      2. 시뮬레이터 스텝 이후 apply_joint_fires_results() 호출
      3. 사이버·EW 결과를 받아 apply_c2_degradation() 호출
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.command_structures: Dict[str, CommandStructure] = {}
        self.c2_links: Dict[Tuple[str, str], C2Link] = {}
        self.pending_requests: List[JointFiresRequest] = []
        self.processed_requests: List[JointFiresRequest] = []
        self._step: int = 0

    # ── C2 구조 관리 ───────────────────────────────────────────

    def register_command_structure(self, cs: CommandStructure) -> None:
        self.command_structures[cs.headquarters_id] = cs
        # HQ → 예하 링크 자동 생성
        for sub_id in cs.all_subordinate_ids():
            link = C2Link(
                src_id=cs.headquarters_id,
                dst_id=sub_id,
                quality=cs.communication_quality,
            )
            self.c2_links[(cs.headquarters_id, sub_id)] = link

    def auto_build_from_kg(self, kg: CombatKnowledgeGraph,
                            alignment: ForceAlignment = ForceAlignment.BLUE) -> None:
        """KG의 유닛 구조에서 C2 계층을 자동 구성"""
        units = [u for u in kg.units.values() if u.alignment == alignment]
        if not units:
            return

        # 가장 높은 전투력 유닛을 HQ로 지정
        hq = max(units, key=lambda u: u.combat_power)
        subs = [u.unit_id for u in units if u.unit_id != hq.unit_id]
        cs = CommandStructure(
            headquarters_id=hq.unit_id,
            subordinate_ids=subs,
            communication_quality=hq.capability.comms_quality,
        )
        self.register_command_structure(cs)

    def apply_c2_degradation(self, hq_id: str, degradation: float) -> None:
        """사이버·EW 공격에 의한 C2 약화 적용"""
        if hq_id in self.command_structures:
            cs = self.command_structures[hq_id]
            cs.c2_degradation = float(np.clip(cs.c2_degradation + degradation, 0.0, 1.0))
        # C2 링크 재밍
        for link in self.c2_links.values():
            if link.src_id == hq_id:
                link.jamming_level = float(np.clip(link.jamming_level + degradation * 0.5, 0.0, 1.0))

    def recover_c2(self, hq_id: str, recovery: float = 0.1) -> None:
        """통신 복구"""
        if hq_id in self.command_structures:
            cs = self.command_structures[hq_id]
            cs.c2_degradation = float(np.clip(cs.c2_degradation - recovery, 0.0, 1.0))

    # ── 합동 화력 지원 ─────────────────────────────────────────

    def request_fires_support(
        self,
        requester_id: str,
        supporter_id: str,
        support_type: str,
        target_id: Optional[str] = None,
        target_location: Optional[Position] = None,
        roe_compliant: bool = True,
    ) -> JointFiresRequest:
        """합동 화력 지원 요청 등록"""
        req = JointFiresRequest(
            requester_id=requester_id,
            supporter_id=supporter_id,
            support_type=support_type,
            target_id=target_id,
            target_location=target_location,
            roe_compliant=roe_compliant,
            response_delay_steps=self._get_response_delay(support_type),
        )
        self.pending_requests.append(req)
        return req

    def _get_response_delay(self, support_type: str) -> int:
        delays = {
            "cas": 2, "naval_gunfire": 3, "rocket": 1,
            "missile_strike": 2, "cyber_strike": 1,
            "space_isr": 1, "ew_jam": 1,
        }
        return delays.get(support_type, 2)

    def process_pending_fires(self, kg: CombatKnowledgeGraph) -> List[Dict]:
        """
        대기 중인 화력 요청 처리.
        - ROE 비준수 요청 거부
        - C2 품질에 비례한 효과 적용
        - 대상 유닛 전투력/headcount 감소
        """
        self._step += 1
        results = []

        still_pending = []
        for req in self.pending_requests:
            req.response_delay_steps -= 1
            if req.response_delay_steps > 0:
                still_pending.append(req)
                continue

            if not req.roe_compliant:
                results.append({"request": req, "status": "rejected_roe"})
                continue

            # C2 품질 조회
            c2_q = self.get_c2_quality_for(req.requester_id)

            # 효과 계산 (C2 품질 + 노이즈 반영)
            base_eff = req.base_effectiveness()
            noise = self.rng.normal(0, 0.1)
            eff = float(np.clip(base_eff * c2_q + noise, 0.0, 1.0))
            req.effectiveness = eff
            req.processed = True

            # 대상 유닛에 적용
            damage = 0
            if req.target_id and req.target_id in kg.units:
                target = kg.units[req.target_id]
                if target.status != UnitStatus.DESTROYED:
                    losses = int(target.headcount * eff * 0.3)
                    target.headcount = max(0, target.headcount - losses)
                    damage = losses
                    # 상태 업데이트
                    ratio = target.headcount / max(target.headcount + losses, 1)
                    if ratio < 0.05:   target.status = UnitStatus.DESTROYED
                    elif ratio < 0.30: target.status = UnitStatus.CRITICAL
                    elif ratio < 0.70: target.status = UnitStatus.DEGRADED

            results.append({
                "request": req,
                "status": "executed",
                "effectiveness": eff,
                "damage": damage,
                "c2_quality": c2_q,
            })
            self.processed_requests.append(req)

        self.pending_requests = still_pending
        return results

    # ── 상태 조회 (RL 에이전트 연동) ───────────────────────────

    def get_c2_quality_for(self, unit_id: str) -> float:
        """특정 유닛의 C2 품질 반환"""
        for hq_id, cs in self.command_structures.items():
            if unit_id in cs.all_subordinate_ids() or unit_id == hq_id:
                return cs.effective_command_quality()
        return 1.0  # 등록 안 된 유닛: 완전 자율

    def get_overall_c2_quality(self, alignment: ForceAlignment,
                                kg: CombatKnowledgeGraph) -> float:
        """해당 진영 전체 평균 C2 품질 (build_state_vector 용)"""
        units = [u for u in kg.units.values() if u.alignment == alignment]
        if not units:
            return 1.0
        qualities = [self.get_c2_quality_for(u.unit_id) for u in units]
        return float(np.mean(qualities))

    def get_joint_fires_available(self, alignment: ForceAlignment,
                                   kg: CombatKnowledgeGraph) -> float:
        """
        합동 화력 가용성 지수 [0, 1]
        — 항공·해군·미사일 지원 가능 유닛 비율 기반
        """
        support_types = {
            UnitType.FIGHTER, UnitType.STRIKE_AIRCRAFT, UnitType.BOMBER,
            UnitType.NAVAL_AVIATION, UnitType.DESTROYER, UnitType.FRIGATE,
            UnitType.BALLISTIC_MISSILE, UnitType.CRUISE_MISSILE, UnitType.MLRS,
            UnitType.UAV_STRIKE, UnitType.LOITERING_MUNITION,
        }
        units = [u for u in kg.units.values()
                 if u.alignment == alignment and u.status != UnitStatus.DESTROYED]
        if not units:
            return 0.0
        support_count = sum(1 for u in units if u.unit_type in support_types)
        return float(np.clip(support_count / len(units), 0.0, 1.0))

    def get_combat_bonus(self, unit_id: str) -> float:
        """
        합동 화력 및 C2 보너스 배수 반환 (전투력 곱셈 인자)
        — 시뮬레이터의 alpha/beta 보정에 활용 가능
        """
        c2_q = self.get_c2_quality_for(unit_id)
        # C2 품질: 0.7 ~ 1.2 범위의 전투력 배수
        return float(0.7 + 0.5 * c2_q)

    # ── 리포트 ─────────────────────────────────────────────────

    def generate_c2_report(self, kg: CombatKnowledgeGraph) -> str:
        lines = ["=== C2 상태 보고 ==="]
        for hq_id, cs in self.command_structures.items():
            hq_name = kg.units[hq_id].designation if hq_id in kg.units else hq_id
            eff = cs.effective_command_quality()
            lines.append(
                f"  [{hq_name}] C2품질={eff:.2f}  예하={len(cs.subordinate_ids)}개  "
                f"약화={cs.c2_degradation:.2f}"
            )
        lines.append(f"  처리된 화력요청: {len(self.processed_requests)}건")
        lines.append(f"  대기 중 화력요청: {len(self.pending_requests)}건")
        return "\n".join(lines)


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory

    print("=== STEP 6: C2 + 합동작전 테스트 ===")
    kg = ScenarioFactory.create_joint_scenario(seed=42)

    mgr = JointOperationsManager(seed=42)
    mgr.auto_build_from_kg(kg, ForceAlignment.BLUE)
    mgr.auto_build_from_kg(kg, ForceAlignment.RED)

    print(f"Blue C2 품질: {mgr.get_overall_c2_quality(ForceAlignment.BLUE, kg):.3f}")
    print(f"합동화력 가용성: {mgr.get_joint_fires_available(ForceAlignment.BLUE, kg):.3f}")

    # 화력 지원 요청
    blue_units = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
    red_units  = [u for u in kg.units.values() if u.alignment == ForceAlignment.RED]
    if blue_units and red_units:
        mgr.request_fires_support(
            requester_id=blue_units[0].unit_id,
            supporter_id=blue_units[1].unit_id if len(blue_units) > 1 else blue_units[0].unit_id,
            support_type="cas",
            target_id=red_units[0].unit_id,
        )
        results = mgr.process_pending_fires(kg)
        results = mgr.process_pending_fires(kg)  # delay=2 이므로 2번 호출
        for r in results:
            print(f"  화력지원 결과: {r.get('status')}  효과={r.get('effectiveness', 0):.2f}  피해={r.get('damage', 0)}")

    # C2 약화 테스트
    hq_id = list(mgr.command_structures.keys())[0]
    mgr.apply_c2_degradation(hq_id, 0.3)
    print(f"C2 약화 후 품질: {mgr.get_overall_c2_quality(ForceAlignment.BLUE, kg):.3f}")

    print(mgr.generate_c2_report(kg))
    print("✅ STEP 6 C2 + 합동작전 테스트 완료!")
