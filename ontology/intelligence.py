"""
intelligence.py
===============
창의적 확장 6-1: 정보 온톨로지 (Intelligence Ontology)

포함 내용:
  - IntelSourceType   : HUMINT/SIGINT/IMINT/MASINT/OSINT/TECHINT 수집 출처
  - IntelReliabilityGrade / IntelConfidenceLevel : NATO STANAG 2511 신뢰도 체계
  - INTELReport       : 정보 보고서 (불확실성 감소 효과 포함)
  - FogOfWarState     : 유닛별 안개전쟁 불확실도
  - ISRAssetModel     : ISR 자산 수집 능력 (탐지거리·확률·EMS 영향)
  - IntelligenceFusionEngine : Dempster-Shafer 다중출처 융합
  - IntelligenceManager : 통합 관리자

파이프라인 연동:
  ISR 수집 → INTELReport 생성 → 융합 → 안개전쟁 업데이트
  → isr_quality      → CombatDynamicsManager.step_update(isr_quality=...)
  → uncertainty_map  → build_state_vector(uncertainty_map=...)
  → gnn_extension    → build_state_vector(gnn_extension=...)   [8D]
  → intel_state_vec  → ROEManager 감점 조건 완화 (ISR 확인 기반 교전)

학술 연구용 합성 데이터
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, ForceAlignment, UnitStatus,
    CombatKnowledgeGraph, Position,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. 열거형 — 수집 출처 / NATO 신뢰도 체계
# ──────────────────────────────────────────────────────────────

class IntelSourceType(str, Enum):
    """정보 수집 출처 (INT disciplines)"""
    HUMINT  = "humint"    # 인간정보 (포로심문, 첩보원, 현지주민)
    SIGINT  = "sigint"    # 신호정보 (통신감청, ELINT, COMINT)
    IMINT   = "imint"     # 영상정보 (위성/UAV 광학·SAR 영상)
    MASINT  = "masint"    # 측정신호정보 (음향·진동·화학·핵)
    OSINT   = "osint"     # 공개출처정보 (언론·SNS·공개자료)
    TECHINT = "techint"   # 기술정보 (노획장비·기술분석)
    CYBER   = "cyber"     # 사이버 정보 (네트워크 침투·감청)
    SPACE   = "space"     # 우주정보 (정찰위성·조기경보위성)


class IntelReliabilityGrade(str, Enum):
    """
    출처 신뢰도 (Source Reliability) — NATO STANAG 2511 A~F

    A: 완전히 신뢰 (Completely Reliable)
    B: 일반적으로 신뢰 (Usually Reliable)
    C: 다소 신뢰 (Fairly Reliable)
    D: 일반적으로 신뢰 불가 (Not Usually Reliable)
    E: 신뢰 불가 (Unreliable)
    F: 신뢰도 미판정 (Reliability Cannot Be Judged)
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class IntelConfidenceLevel(int, Enum):
    """
    정보 확실도 (Information Credibility) — 1~6

    1: 확인됨 (Confirmed by other sources)
    2: 개연성 높음 (Probably True)
    3: 가능성 있음 (Possibly True)
    4: 의심됨 (Doubtful)
    5: 비개연적 (Improbable)
    6: 진실 불명 (Truth Cannot Be Judged)
    """
    CONFIRMED    = 1
    PROBABLE     = 2
    POSSIBLE     = 3
    DOUBTFUL     = 4
    IMPROBABLE   = 5
    UNASSESSED   = 6


# NATO 신뢰도 등급 → float 변환 테이블
_RELIABILITY_WEIGHT: Dict[str, float] = {
    "A": 0.95, "B": 0.80, "C": 0.60,
    "D": 0.40, "E": 0.20, "F": 0.50,
}

# 확실도 → float 변환 테이블
_CONFIDENCE_WEIGHT: Dict[int, float] = {
    1: 0.95, 2: 0.80, 3: 0.60,
    4: 0.35, 5: 0.15, 6: 0.50,
}

# 수집 출처별 기본 불확실성 감소량 (per-report)
_SOURCE_UNC_REDUCTION: Dict[str, float] = {
    "humint":  0.25,   # 인간정보 — 정밀하나 범위 좁음
    "sigint":  0.20,   # 신호정보 — 위치 추정 가능
    "imint":   0.35,   # 영상정보 — 위치·장비 식별 우수
    "masint":  0.15,   # 측정신호 — 간접 증거
    "osint":   0.10,   # 공개정보 — 낮은 정밀도
    "techint": 0.18,   # 기술정보 — 장비 종류 식별
    "cyber":   0.22,   # 사이버  — C2 구조 파악 우수
    "space":   0.40,   # 우주정보 — 광역 커버리지
}


# ──────────────────────────────────────────────────────────────
# 2. INTELReport — 정보 보고서
# ──────────────────────────────────────────────────────────────

@dataclass
class INTELReport:
    """
    정보 보고서 (Individual Intelligence Report)

    하나의 보고서는 하나의 대상 부대에 대한 수집 정보를 담는다.
    여러 보고서가 IntelligenceFusionEngine에 의해 융합된다.
    """
    report_id: str
    source_type: IntelSourceType
    collector_unit_id: str              # 수집 부대 ID
    collected_at_step: int              # 수집 타임스텝

    # 대상 정보 (적 부대 추정)
    target_unit_id: Optional[str] = None          # 확인된 유닛 ID
    target_position_est: Optional[Position] = None # 추정 위치
    target_unit_type_est: Optional[str] = None     # 추정 UnitType (str)
    strength_estimate: Optional[float] = None      # 추정 전투력 [0, 1]

    # NATO STANAG 2511 신뢰도
    reliability_grade: IntelReliabilityGrade = IntelReliabilityGrade.C
    confidence_level: IntelConfidenceLevel   = IntelConfidenceLevel.POSSIBLE

    # 시의성
    perishability_steps: int = 10   # 정보 유효 기간 (스텝)

    # 위치 정확도
    position_accuracy_km: float = 5.0

    @property
    def combined_confidence(self) -> float:
        """
        출처 신뢰도 × 정보 확실도 → 종합 신뢰도 [0, 1]
        """
        r = _RELIABILITY_WEIGHT.get(self.reliability_grade.value, 0.5)
        c = _CONFIDENCE_WEIGHT.get(self.confidence_level.value, 0.5)
        return float(r * c)

    @property
    def uncertainty_reduction(self) -> float:
        """
        이 보고서가 줄이는 불확실도 (출처 기준 × 종합 신뢰도)
        """
        base = _SOURCE_UNC_REDUCTION.get(self.source_type.value, 0.10)
        return float(base * self.combined_confidence)

    def is_valid(self, current_step: int) -> bool:
        """정보 유효 기간 내 여부"""
        return (current_step - self.collected_at_step) <= self.perishability_steps

    def to_dict(self) -> Dict:
        return {
            "report_id":       self.report_id,
            "source":          self.source_type.value,
            "collector":       self.collector_unit_id,
            "step":            self.collected_at_step,
            "target":          self.target_unit_id,
            "confidence":      round(self.combined_confidence, 3),
            "unc_reduction":   round(self.uncertainty_reduction, 3),
            "pos_accuracy_km": self.position_accuracy_km,
        }


# ──────────────────────────────────────────────────────────────
# 3. FogOfWarState — 유닛별 안개전쟁 상태
# ──────────────────────────────────────────────────────────────

@dataclass
class FogOfWarState:
    """
    단일 부대에 대한 안개전쟁 불확실도 상태

    adversary(적군)에 대한 정보 불확도를 추적.
    INTELReport가 누적될수록 불확실도 감소.
    """
    unit_id: str
    is_detected: bool = False
    position_uncertainty_km: float = 15.0   # 위치 불확도 (km)
    strength_uncertainty: float = 0.8        # 전투력 불확도 [0, 1]
    type_confirmed: bool = False             # 부대 유형 확인 여부
    last_confirmed_step: int = -1

    # 누적된 confidence mass (Dempster-Shafer)
    _confidence_mass: float = field(default=0.0, repr=False)

    def apply_intel_report(self, report: INTELReport, current_step: int) -> None:
        """보고서 한 건을 적용하여 불확실도 업데이트"""
        if not report.is_valid(current_step):
            return

        reduction = report.uncertainty_reduction
        # 위치 불확도 감소 (position_accuracy_km이 이미 km 단위)
        self.position_uncertainty_km = max(
            0.5,
            self.position_uncertainty_km * (1.0 - reduction * 0.5)
            - report.position_accuracy_km * reduction * 0.05
        )
        # 전투력 불확도 감소
        self.strength_uncertainty = max(
            0.05,
            self.strength_uncertainty * (1.0 - reduction)
        )
        # 탐지 여부
        if report.combined_confidence >= 0.4:
            self.is_detected = True
            self.last_confirmed_step = current_step

        # 유형 확인
        if report.target_unit_type_est and report.combined_confidence >= 0.6:
            self.type_confirmed = True

        # 누적 신뢰 질량 업데이트 (DS 이론 — 단순 합산 upper-bound)
        self._confidence_mass = min(1.0, self._confidence_mass + reduction * 0.5)

    def decay(self, steps_elapsed: int = 1) -> None:
        """시간 경과에 따른 정보 노후화 (불확실도 자연 증가)"""
        decay_rate = 0.05 * steps_elapsed
        self.position_uncertainty_km = min(
            20.0, self.position_uncertainty_km * (1.0 + decay_rate * 0.3)
        )
        self.strength_uncertainty = min(
            1.0, self.strength_uncertainty + decay_rate * 0.1
        )
        self._confidence_mass = max(0.0, self._confidence_mass - decay_rate * 0.5)

    @property
    def overall_uncertainty(self) -> float:
        """종합 불확실도 [0, 1] — GNN uncertainty_map 값으로 사용"""
        pos_norm = min(self.position_uncertainty_km / 20.0, 1.0)
        return float(np.clip(
            (pos_norm * 0.5 + self.strength_uncertainty * 0.5)
            * (1.0 - self._confidence_mass * 0.3),
            0.0, 1.0
        ))


# ──────────────────────────────────────────────────────────────
# 4. ISRAssetModel — ISR 자산 수집 능력
# ──────────────────────────────────────────────────────────────

class ISRAssetModel:
    """
    ISR 자산별 수집 능력 모델

    각 UnitType에 따라:
      - 탐지 반경 (km)
      - 기본 탐지 확률
      - EMS 취약성 계수
      - 제공 가능 정보 출처 유형
    """

    # UnitType → (탐지반경 km, 기본탐지확률, ems_취약성, 출처유형, 위치정확도 km)
    _ASSET_TABLE: Dict[str, Tuple[float, float, float, str, float]] = {
        # (range_km, detect_prob, ems_vuln, source, pos_acc_km)
        "reconnaissance":   (30.0,  0.75, 0.30, "humint",  2.0),
        "special_forces":   (20.0,  0.80, 0.20, "humint",  1.0),
        "uav_recon":        (80.0,  0.85, 0.50, "imint",   0.5),
        "isr_aircraft":     (200.0, 0.90, 0.40, "imint",   0.3),
        "space_asset":      (999.0, 0.70, 0.10, "space",   1.5),
        "electronic_warfare":(50.0, 0.85, 0.15, "sigint",  3.0),
        "cyber_unit":       (999.0, 0.75, 0.05, "cyber",   5.0),  # 사이버 — 위치 무관
        "signal":           (40.0,  0.70, 0.35, "sigint",  4.0),
        "submarine":        (50.0,  0.65, 0.25, "masint",  5.0),  # 음향 탐지
        "destroyer":        (30.0,  0.60, 0.40, "masint",  3.0),
        "loitering_munition":(30.0, 0.70, 0.55, "imint",   1.0),
        "psyops":           (0.0,   0.50, 0.05, "humint",  10.0),
        # 기본값 (명시되지 않은 유형)
        "_default":         (10.0,  0.30, 0.50, "humint",  8.0),
    }

    # 출처별 NATO 신뢰도 등급 기본값
    _SOURCE_RELIABILITY: Dict[str, str] = {
        "humint": "C",    # 인간정보 — 비교적 낮음
        "sigint": "B",    # 신호정보 — 객관적
        "imint":  "B",    # 영상정보 — 시각적 확인
        "masint": "C",    # 측정정보 — 간접
        "osint":  "D",    # 공개정보 — 낮음
        "techint":"B",
        "cyber":  "C",    # 사이버 — 변조 가능성
        "space":  "A",    # 위성 — 최고
    }

    @classmethod
    def _get_asset_params(cls, unit_type_value: str) -> Tuple[float, float, float, str, float]:
        return cls._ASSET_TABLE.get(unit_type_value, cls._ASSET_TABLE["_default"])

    @classmethod
    def can_collect(
        cls,
        collector: Unit,
        target: Unit,
        ems_jamming: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        수집 가능 여부 및 탐지 확률 반환.

        Parameters
        ----------
        collector    : ISR 수집 부대
        target       : 탐지 대상 부대
        ems_jamming  : 전자전 재밍 강도 [0, 1]

        Returns
        -------
        (can_attempt: bool, detect_prob: float)
        """
        params = cls._get_asset_params(collector.unit_type.value)
        range_km, base_prob, ems_vuln, _, _ = params

        # 사이버/우주 정보는 거리 무관
        if collector.unit_type.value in ("cyber_unit", "space_asset"):
            can_attempt = True
        else:
            # 위치 기반 거리 계산
            # Position 좌표는 시나리오에서 [0, map_size_km] 범위 사용 → scale=1.0
            if collector.position is None or target.position is None:
                return False, 0.0
            dx = collector.position.x - target.position.x
            dy = collector.position.y - target.position.y
            dist_km = math.sqrt(dx * dx + dy * dy)   # 좌표가 이미 km 단위
            can_attempt = dist_km <= range_km

        # EMS 재밍 효과 적용
        jam_effect = ems_jamming * ems_vuln
        final_prob = float(np.clip(base_prob * (1.0 - jam_effect), 0.01, 1.0))

        # 수집 부대 상태 보정
        if collector.status == UnitStatus.DESTROYED:
            return False, 0.0
        if collector.status.value == "critical":
            final_prob *= 0.3
        elif collector.status.value == "degraded":
            final_prob *= 0.6

        # ISR 자산 자체 능력치 반영
        if hasattr(collector.capability, "isr_capability"):
            final_prob *= collector.capability.isr_capability

        return can_attempt, final_prob

    @classmethod
    def generate_report(
        cls,
        collector: Unit,
        target: Unit,
        step: int,
        report_id: str,
        ems_jamming: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> Optional[INTELReport]:
        """
        ISR 수집 시도 → 성공 시 INTELReport 반환, 실패 시 None.
        """
        if rng is None:
            rng = np.random.RandomState()

        can_attempt, prob = cls.can_collect(collector, target, ems_jamming)
        if not can_attempt:
            return None

        if rng.random() > prob:
            return None  # 탐지 실패

        params = cls._get_asset_params(collector.unit_type.value)
        range_km, _, _, source_str, pos_acc_km = params

        # EMS 영향 → 위치 정확도 저하
        pos_acc_km *= (1.0 + ems_jamming * 2.0)

        # 신뢰도 등급 결정
        reliability_key = cls._SOURCE_RELIABILITY.get(source_str, "C")
        reliability = IntelReliabilityGrade(reliability_key)

        # 확실도 결정 (확률 기반)
        if prob >= 0.80:
            confidence = IntelConfidenceLevel.PROBABLE
        elif prob >= 0.60:
            confidence = IntelConfidenceLevel.POSSIBLE
        elif prob >= 0.40:
            confidence = IntelConfidenceLevel.DOUBTFUL
        else:
            confidence = IntelConfidenceLevel.UNASSESSED

        # 전투력 추정 (노이즈 포함)
        strength_noise = rng.normal(0, 0.15)
        strength_est = float(np.clip(target.combat_power + strength_noise, 0.0, 1.0))

        # 위치 추정 (노이즈 포함) — 좌표가 이미 km 단위이므로 직접 더함
        pos_est = None
        if target.position is not None:
            pos_est = Position(
                x=float(np.clip(target.position.x + rng.normal(0, pos_acc_km), 0, 9999)),
                y=float(np.clip(target.position.y + rng.normal(0, pos_acc_km), 0, 9999)),
            )

        return INTELReport(
            report_id=report_id,
            source_type=IntelSourceType(source_str),
            collector_unit_id=collector.unit_id,
            collected_at_step=step,
            target_unit_id=target.unit_id,
            target_position_est=pos_est,
            target_unit_type_est=target.unit_type.value if prob >= 0.60 else None,
            strength_estimate=strength_est if prob >= 0.50 else None,
            reliability_grade=reliability,
            confidence_level=confidence,
            perishability_steps=max(5, int(15 * (1.0 - ems_jamming))),
            position_accuracy_km=pos_acc_km,
        )


# ──────────────────────────────────────────────────────────────
# 5. IntelligenceFusionEngine — 다중출처 정보 융합
# ──────────────────────────────────────────────────────────────

class IntelligenceFusionEngine:
    """
    다중 출처 정보 융합 (Multi-Source Intelligence Fusion)

    Dempster-Shafer 이론을 단순화한 베이지안 업데이트 방식:
      - 동일 대상에 대한 N개 보고서를 합성
      - 출처가 다양할수록 신뢰도 상승
      - 출처 간 충돌 시 신뢰도 감쇄
    """

    @staticmethod
    def fuse(reports: List[INTELReport], current_step: int) -> float:
        """
        유효 보고서 목록을 융합하여 종합 신뢰도 [0, 1] 반환.

        Returns
        -------
        fused_confidence : float in [0, 1]
        """
        valid = [r for r in reports if r.is_valid(current_step)]
        if not valid:
            return 0.0

        # 다른 출처 유형 수 (다양성 보너스)
        source_types = {r.source_type for r in valid}
        diversity_bonus = min(len(source_types) * 0.05, 0.20)

        # Murphy 결합 규칙 (단순 가중 기하 평균으로 근사)
        weights = [r.combined_confidence for r in valid]
        total_w = sum(weights)
        if total_w == 0:
            return 0.0

        fused = sum(r.combined_confidence * r.uncertainty_reduction
                    for r in valid) / max(total_w, 1e-6)

        # 출처 다양성 보너스 적용
        fused = min(1.0, fused + diversity_bonus)

        # 보고서 수 보너스 (최대 +0.15)
        count_bonus = min(len(valid) * 0.03, 0.15)
        return float(min(1.0, fused + count_bonus))

    @staticmethod
    def fuse_uncertainty_reduction(reports: List[INTELReport], current_step: int) -> float:
        """
        보고서들의 누적 불확실도 감소량 계산.
        중복 감소 방지를 위해 complementary product 방식 사용.

        누적 감소: 1 - ∏(1 - r_i)   [각 보고서의 독립 기여]
        """
        valid = [r for r in reports if r.is_valid(current_step)]
        if not valid:
            return 0.0

        product = 1.0
        for r in valid:
            product *= (1.0 - r.uncertainty_reduction)

        return float(min(0.95, 1.0 - product))


# ──────────────────────────────────────────────────────────────
# 6. IntelligenceManager — 통합 관리자
# ──────────────────────────────────────────────────────────────

class IntelligenceManager:
    """
    정보 온톨로지 통합 관리자

    역할:
      1. collect_phase()  — ISR 유닛이 적 부대 탐지 → 보고서 생성
      2. update_fog_of_war() — FogOfWarState 업데이트 (불확실도 감소)
      3. decay_intel()    — 정보 노후화 (스텝별)
      4. 파이프라인 연동값 제공:
         - get_isr_quality(alignment, kg) → float
             CombatDynamicsManager.step_update(isr_quality=...)
         - get_uncertainty_map(alignment)  → Dict[str, float]
             build_state_vector(uncertainty_map=...)
         - get_gnn_state_features(alignment, kg) → np.ndarray (8D)
             build_state_vector(gnn_extension=...)
         - get_intel_summary(alignment)   → str (리포트 텍스트)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.RandomState(seed)
        self._step: int = 0
        self._report_counter: int = 0

        # 전체 보고서 이력 (누적)
        self._reports: List[INTELReport] = []

        # 대상 유닛별 보고서 인덱스: {unit_id → [report, ...]}
        self._reports_by_target: Dict[str, List[INTELReport]] = {}

        # 적군 유닛별 안개전쟁 상태: {unit_id → FogOfWarState}
        self._fog: Dict[str, FogOfWarState] = {}

        # ISR 융합 엔진
        self._fusion = IntelligenceFusionEngine()

    # ── 초기화 ────────────────────────────────────────────────
    def initialize_from_kg(self, kg: CombatKnowledgeGraph) -> None:
        """KG에서 적군 유닛 목록을 읽어 FogOfWarState 초기화"""
        for unit in kg.units.values():
            if unit.unit_id not in self._fog:
                self._fog[unit.unit_id] = FogOfWarState(
                    unit_id=unit.unit_id,
                    is_detected=False,
                    position_uncertainty_km=15.0,
                    strength_uncertainty=0.8,
                )

    # ── 수집 국면 ─────────────────────────────────────────────
    def collect_phase(
        self,
        kg: CombatKnowledgeGraph,
        collector_alignment: ForceAlignment,
        ems_jamming: float = 0.0,
    ) -> List[INTELReport]:
        """
        collector_alignment 진영의 ISR 부대가 적 부대를 탐지 시도.

        Parameters
        ----------
        kg                   : 현재 지식 그래프
        collector_alignment  : 수집하는 진영 (BLUE/RED)
        ems_jamming          : 전자전 재밍 강도 [0, 1]

        Returns
        -------
        new_reports : 이번 스텝에서 새로 생성된 보고서 목록
        """
        collectors = [
            u for u in kg.units.values()
            if u.alignment == collector_alignment
            and u.status != UnitStatus.DESTROYED
        ]
        targets = [
            u for u in kg.units.values()
            if u.alignment != collector_alignment
            and u.status != UnitStatus.DESTROYED
        ]

        new_reports: List[INTELReport] = []

        for collector in collectors:
            for target in targets:
                self._report_counter += 1
                rid = f"RPT-{self._step:04d}-{self._report_counter:06d}"
                report = ISRAssetModel.generate_report(
                    collector=collector,
                    target=target,
                    step=self._step,
                    report_id=rid,
                    ems_jamming=ems_jamming,
                    rng=self.rng,
                )
                if report is not None:
                    new_reports.append(report)
                    self._reports.append(report)
                    self._reports_by_target.setdefault(
                        target.unit_id, []
                    ).append(report)

        return new_reports

    # ── 안개전쟁 업데이트 ─────────────────────────────────────
    def update_fog_of_war(
        self,
        new_reports: List[INTELReport],
        kg: Optional[CombatKnowledgeGraph] = None,
    ) -> None:
        """
        새 보고서를 FogOfWarState에 반영하여 불확실도 갱신.
        KG가 제공되면 적군 신규 유닛도 등록.
        """
        if kg is not None:
            for uid in kg.units:
                if uid not in self._fog:
                    self._fog[uid] = FogOfWarState(unit_id=uid)

        for report in new_reports:
            if report.target_unit_id is None:
                continue
            fog = self._fog.setdefault(
                report.target_unit_id,
                FogOfWarState(unit_id=report.target_unit_id),
            )
            fog.apply_intel_report(report, self._step)

    # ── 정보 노후화 ───────────────────────────────────────────
    def decay_intel(self, steps: int = 1) -> None:
        """모든 FogOfWarState의 불확실도를 시간 경과만큼 증가"""
        for fog in self._fog.values():
            fog.decay(steps)

    # ── 스텝 업데이트 ─────────────────────────────────────────
    def step(
        self,
        kg: CombatKnowledgeGraph,
        collector_alignment: ForceAlignment,
        ems_jamming: float = 0.0,
        decay_steps: int = 1,
    ) -> List[INTELReport]:
        """
        1개 타임스텝 처리:
        1) 정보 노후화 → 2) ISR 수집 → 3) FogOfWar 업데이트 → 4) step 증가

        Returns
        -------
        new_reports : 이번 스텝 새 보고서
        """
        self.decay_intel(decay_steps)
        new_reports = self.collect_phase(kg, collector_alignment, ems_jamming)
        self.update_fog_of_war(new_reports, kg)
        self._step += 1
        return new_reports

    # ── 파이프라인 연동값 ─────────────────────────────────────

    def get_isr_quality(
        self,
        collector_alignment: ForceAlignment,
        kg: CombatKnowledgeGraph,
    ) -> float:
        """
        해당 진영의 현재 ISR 품질 [0, 1].
        → CombatDynamicsManager.step_update(isr_quality=...)에 전달.

        ISR 자산 수, 상태, 최근 보고서 신뢰도 기반 계산.
        """
        collectors = [
            u for u in kg.units.values()
            if u.alignment == collector_alignment
            and u.status != UnitStatus.DESTROYED
        ]

        # ISR 자산 유닛 수 기여
        isr_types = {
            "uav_recon", "isr_aircraft", "space_asset",
            "reconnaissance", "electronic_warfare", "cyber_unit",
            "special_forces", "submarine",
        }
        isr_count = sum(
            1 for u in collectors if u.unit_type.value in isr_types
        )
        total = max(len(collectors), 1)
        isr_ratio = isr_count / total

        # 최근 유효 보고서의 평균 신뢰도
        recent = [r for r in self._reports if r.is_valid(self._step)]
        if recent:
            avg_conf = float(np.mean([r.combined_confidence for r in recent[-20:]]))
        else:
            avg_conf = 0.3  # 정보 없음 기본값

        # 탐지된 적 부대 비율
        target_units = [
            u for u in kg.units.values()
            if u.alignment != collector_alignment
        ]
        detected = sum(
            1 for u in target_units
            if self._fog.get(u.unit_id, FogOfWarState(u.unit_id)).is_detected
        )
        detection_rate = detected / max(len(target_units), 1)

        isr_quality = float(np.clip(
            isr_ratio * 0.30 + avg_conf * 0.40 + detection_rate * 0.30,
            0.0, 1.0
        ))
        return isr_quality

    def get_uncertainty_map(
        self,
        collector_alignment: ForceAlignment,
        kg: Optional[CombatKnowledgeGraph] = None,
    ) -> Dict[str, float]:
        """
        적군 유닛별 종합 불확실도 맵.
        → build_state_vector(uncertainty_map=...) 에 전달.

        Returns
        -------
        {unit_id → overall_uncertainty [0, 1]}
        """
        result: Dict[str, float] = {}

        # 적군 유닛만
        if kg is not None:
            for uid, unit in kg.units.items():
                if unit.alignment == collector_alignment:
                    continue
                fog = self._fog.get(uid)
                if fog is not None:
                    result[uid] = fog.overall_uncertainty
                else:
                    result[uid] = 0.9  # 정보 없음 → 최대 불확실
        else:
            for uid, fog in self._fog.items():
                result[uid] = fog.overall_uncertainty

        return result

    def get_gnn_state_features(
        self,
        collector_alignment: ForceAlignment,
        kg: CombatKnowledgeGraph,
    ) -> np.ndarray:
        """
        GNN 확장 슬롯용 8D 정보 상태 특성.
        → build_state_vector(gnn_extension=...) 에 전달.

        구성:
          [0] ISR 커버리지 비율 (탐지된 적 / 전체 적)
          [1] 평균 정보 신선도 (최근 보고서 비율)
          [2] 다중출처 융합 신뢰도
          [3] 평균 FogOfWar 불확실도 (적 유닛)
          [4] SIGINT 강도 (통신 감청 비율)
          [5] IMINT 커버리지 (영상 정보 비율)
          [6] HUMINT 비율
          [7] 정보 우위 점수 (아군 ISR 자산 비율)
        """
        target_units = [
            u for u in kg.units.values()
            if u.alignment != collector_alignment
        ]
        n_target = max(len(target_units), 1)

        # [0] 탐지율
        detected = sum(
            1 for u in target_units
            if self._fog.get(u.unit_id, FogOfWarState(u.unit_id)).is_detected
        )
        coverage = detected / n_target

        # [1] 정보 신선도
        recent = [r for r in self._reports if r.is_valid(self._step)]
        total_hist = len(self._reports)
        freshness = len(recent) / max(total_hist, 1)

        # [2] 다중출처 융합 신뢰도 (평균)
        fused_conf = 0.0
        if target_units:
            confs = []
            for u in target_units:
                rpts = self._reports_by_target.get(u.unit_id, [])
                if rpts:
                    confs.append(
                        self._fusion.fuse(rpts, self._step)
                    )
            fused_conf = float(np.mean(confs)) if confs else 0.0

        # [3] 평균 FogOfWar 불확실도
        fow_values = [
            self._fog.get(u.unit_id, FogOfWarState(u.unit_id)).overall_uncertainty
            for u in target_units
        ]
        avg_fow = float(np.mean(fow_values)) if fow_values else 0.9

        # [4] SIGINT 강도
        sigint_rpts = [r for r in recent if r.source_type == IntelSourceType.SIGINT]
        sigint_ratio = len(sigint_rpts) / max(len(recent), 1)

        # [5] IMINT 커버리지
        imint_rpts = [r for r in recent if r.source_type == IntelSourceType.IMINT]
        imint_ratio = len(imint_rpts) / max(len(recent), 1)

        # [6] HUMINT 비율
        humint_rpts = [r for r in recent if r.source_type == IntelSourceType.HUMINT]
        humint_ratio = len(humint_rpts) / max(len(recent), 1)

        # [7] 정보 우위 점수
        isr_score = self.get_isr_quality(collector_alignment, kg)

        return np.array([
            coverage, freshness, fused_conf, avg_fow,
            sigint_ratio, imint_ratio, humint_ratio, isr_score,
        ], dtype=np.float32)

    def get_node_uncertainty_weights(
        self,
        collector_alignment: ForceAlignment,
        kg: CombatKnowledgeGraph,
    ) -> Dict[str, float]:
        """
        GNN 노드별 에피스테믹 불확실성 가중치.
        BayesianHGT의 uncertainty 출력에 곱하여 안개전쟁 반영.

        Returns
        -------
        {unit_id → uncertainty_multiplier [1.0, 2.0]}
        """
        weights: Dict[str, float] = {}
        for uid, unit in kg.units.items():
            if unit.alignment == collector_alignment:
                weights[uid] = 1.0  # 아군 — 불확실성 없음
                continue
            fog = self._fog.get(uid)
            if fog is None:
                weights[uid] = 2.0
            else:
                # overall_uncertainty 0 → 배수 1.0, 1 → 배수 2.0
                weights[uid] = 1.0 + fog.overall_uncertainty
        return weights

    # ── 리포트 및 통계 ────────────────────────────────────────
    def get_intel_summary(
        self,
        collector_alignment: ForceAlignment,
        kg: CombatKnowledgeGraph,
    ) -> str:
        """정보 상황 요약 리포트 (텍스트)"""
        target_units = [
            u for u in kg.units.values()
            if u.alignment != collector_alignment
        ]
        detected = [
            u for u in target_units
            if self._fog.get(u.unit_id, FogOfWarState(u.unit_id)).is_detected
        ]
        recent_rpts = [r for r in self._reports if r.is_valid(self._step)]
        isr_q = self.get_isr_quality(collector_alignment, kg)

        lines = [
            "=" * 52,
            f"  정보 상황 요약 (Step {self._step})",
            "=" * 52,
            f"  탐지된 적 부대: {len(detected)}/{len(target_units)}",
            f"  유효 정보 보고서: {len(recent_rpts)}건 "
            f"(총 {len(self._reports)}건)",
            f"  ISR 품질: {isr_q:.3f}",
            "",
            "  출처별 보고서 분포:",
        ]
        for src in IntelSourceType:
            cnt = sum(
                1 for r in recent_rpts if r.source_type == src
            )
            if cnt > 0:
                lines.append(f"    {src.value.upper():10s}: {cnt}건")

        lines.append("")
        lines.append("  적 부대별 안개전쟁 상태:")
        for u in target_units[:8]:  # 최대 8개 표시
            fog = self._fog.get(u.unit_id, FogOfWarState(u.unit_id))
            lines.append(
                f"    {u.unit_id:15s}: "
                f"탐지={'O' if fog.is_detected else 'X'}  "
                f"위치불확도={fog.position_uncertainty_km:.1f}km  "
                f"전력불확도={fog.strength_uncertainty:.2f}"
            )
        if len(target_units) > 8:
            lines.append(f"    ... 외 {len(target_units)-8}개 부대")

        return "\n".join(lines)

    def get_step_summary(self) -> Dict:
        """현재 스텝 정보 상태 딕셔너리 (로깅용)"""
        recent = [r for r in self._reports if r.is_valid(self._step)]
        detected = sum(1 for f in self._fog.values() if f.is_detected)
        avg_unc = float(np.mean(
            [f.overall_uncertainty for f in self._fog.values()]
        )) if self._fog else 1.0

        return {
            "step": self._step,
            "total_reports": len(self._reports),
            "valid_reports": len(recent),
            "detected_units": detected,
            "total_tracked": len(self._fog),
            "avg_uncertainty": round(avg_unc, 4),
        }
