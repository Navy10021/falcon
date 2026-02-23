"""
cyber_effects.py
================
사이버 공격 효과 전파 모델 (Cyber Effects Propagation)

지원 효과:
  - C2 교란 (C2 Disruption)      : 지휘 통제망 마비
  - 통신 재밍 (Comms Denial)     : 통신 두절 → 지연 및 분리
  - GPS 거부 (GPS Denial)        : 위치 정확도 저하
  - 레이더 기만 (Sensor Spoofing) : 허위 정보 삽입
  - 전력망 공격 (Power Grid)     : 기지·시설 운용 불능
  - 물류·보급망 교란 (Logistics) : 재보급 지연

핵심 모델:
  CyberAttack         : 단일 공격 명세
  CyberDefense        : 방어 능력 (패치·암호화·리던던시)
  CyberEffectEngine   : 공격 발생 → 전파 → 유닛 영향 계산
  EWEffectModel       : 전자전(EW) 효과 (GPS/통신 재밍)

참고:
  - MITRE ATT&CK for ICS: https://attack.mitre.org/matrices/ics/
  - 사이버 효과 지속 시간: 공격 강도 × 방어 취약성 / 복구 속도
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ontology.combat_schema import (
    ForceAlignment, Unit, UnitType, UnitStatus, CombatKnowledgeGraph,
)


# ──────────────────────────────────────────────────────────────
# 열거형
# ──────────────────────────────────────────────────────────────

class CyberTargetSystem(Enum):
    """사이버 공격 대상 시스템"""
    C2_NETWORK      = "c2_network"       # 지휘통제망
    COMMS_BACKBONE  = "comms_backbone"   # 통신 간선망
    GPS_INFRA       = "gps_infrastructure"  # GPS 인프라
    RADAR_NETWORK   = "radar_network"    # 레이더망
    LOGISTICS_DB    = "logistics_database"  # 군수 DB
    POWER_GRID      = "power_grid"       # 전력망 (기지·지휘소)
    EARLY_WARNING   = "early_warning"    # 조기경보체계
    SENSOR_NETWORK  = "sensor_network"   # 센서 네트워크


class CyberEffectType(Enum):
    """사이버 효과 유형"""
    DISRUPTION  = "disruption"   # 기능 저하 (부분 마비)
    DENIAL      = "denial"       # 완전 거부 (전면 마비)
    DECEPTION   = "deception"    # 기만 (허위 정보)
    DESTRUCTION = "destruction"  # 파괴 (물리적 피해 유발)


class CyberAttackPhase(Enum):
    """공격 단계 (MITRE ATT&CK ICS 참조)"""
    RECONNAISSANCE = "recon"      # 정찰
    INTRUSION      = "intrusion"  # 침투
    LATERAL_MOVE   = "lateral"    # 횡이동
    EXECUTION      = "execution"  # 실행 (효과 발생)
    PERSISTENCE    = "persist"    # 지속 (재감염)


# ──────────────────────────────────────────────────────────────
# 데이터클래스
# ──────────────────────────────────────────────────────────────

@dataclass
class CyberAttack:
    """사이버 공격 명세"""
    attacker_id: str                    # 공격 부대 unit_id
    target_system: CyberTargetSystem    # 공격 대상 시스템
    effect_type: CyberEffectType        # 효과 유형
    intensity: float                    # 공격 강도 [0, 1]
    duration_steps: int                 # 효과 지속 스텝 수
    affected_alignment: ForceAlignment  # 피해 측 진영
    # 선택
    phase: CyberAttackPhase = CyberAttackPhase.EXECUTION
    roe_compliant: bool = True          # 교전규칙 준수


@dataclass
class CyberDefenseCapability:
    """사이버 방어 능력"""
    unit_id: str
    encryption_level: float     # 암호화 수준 [0, 1]
    redundancy: float           # 리던던시 [0, 1] (백업 시스템)
    patch_rate: float           # 취약점 패치 속도 [0, 1]
    detection_capability: float # 침입 탐지 능력 [0, 1]

    def overall_defense(self) -> float:
        """종합 방어력 (가중 평균)"""
        return float(np.dot(
            [self.encryption_level, self.redundancy,
             self.patch_rate, self.detection_capability],
            [0.30, 0.25, 0.25, 0.20]
        ))


@dataclass
class CyberEffectResult:
    """사이버 효과 적용 결과"""
    target_system: CyberTargetSystem
    effect_type: CyberEffectType
    attack_success: bool                # 공격 성공 여부
    severity: float                     # 효과 크기 [0, 1]
    duration_remaining: int             # 잔여 지속 스텝
    affected_unit_ids: List[str]        # 영향받은 유닛 목록
    # 구체적 효과
    c2_degradation: float = 0.0        # C2 능력 저하 [0, 1]
    comms_degradation: float = 0.0     # 통신 저하 [0, 1]
    gps_degradation: float = 0.0       # GPS 정확도 저하 [0, 1]
    sensor_degradation: float = 0.0    # 센서 저하 [0, 1]
    logistics_delay_factor: float = 1.0  # 보급 지연 배수 (>1=지연)


@dataclass
class CyberBattleState:
    """사이버 전장 상태 (매 스텝 추적)"""
    step: int
    active_effects: List[CyberEffectResult] = field(default_factory=list)
    # 진영별 종합 저하 지표
    blue_c2_level: float    = 1.0   # 1=정상, 0=완전 마비
    blue_comms_level: float = 1.0
    blue_gps_level: float   = 1.0
    red_c2_level: float     = 1.0
    red_comms_level: float  = 1.0
    red_gps_level: float    = 1.0


# ──────────────────────────────────────────────────────────────
# 사이버 효과 대상 맵 (시스템 → 영향 받는 유닛 유형)
# ──────────────────────────────────────────────────────────────

TARGET_SYSTEM_AFFECTED_TYPES: Dict[CyberTargetSystem, Set[UnitType]] = {
    CyberTargetSystem.C2_NETWORK: {
        UnitType.SIGNAL, UnitType.INFANTRY, UnitType.ARMOR, UnitType.ARTILLERY,
        UnitType.DESTROYER, UnitType.FIGHTER,
    },
    CyberTargetSystem.COMMS_BACKBONE: {
        UnitType.SIGNAL, UnitType.ELECTRONIC_WARFARE,
    },
    CyberTargetSystem.GPS_INFRA: {
        UnitType.INFANTRY, UnitType.ARMOR, UnitType.ARTILLERY,
        UnitType.FIGHTER, UnitType.CRUISE_MISSILE,
    },
    CyberTargetSystem.RADAR_NETWORK: {
        UnitType.SAM_BATTERY, UnitType.EARLY_WARNING, UnitType.DESTROYER,
    },
    CyberTargetSystem.LOGISTICS_DB: {
        UnitType.LOGISTICS,
    },
    CyberTargetSystem.POWER_GRID: {
        UnitType.SIGNAL, UnitType.SAM_BATTERY, UnitType.CYBER_UNIT,
    },
    CyberTargetSystem.EARLY_WARNING: {
        UnitType.EARLY_WARNING, UnitType.ISR_AIRCRAFT,
    },
    CyberTargetSystem.SENSOR_NETWORK: {
        UnitType.ISR_AIRCRAFT, UnitType.UAV_RECON, UnitType.EARLY_WARNING,
    },
}

# 시스템별 효과 프로필 (성공 시 저하 크기)
SYSTEM_EFFECT_PROFILE: Dict[CyberTargetSystem, Dict[str, float]] = {
    CyberTargetSystem.C2_NETWORK:    {"c2": 0.80, "comms": 0.30},
    CyberTargetSystem.COMMS_BACKBONE:{"comms": 0.90, "c2": 0.20},
    CyberTargetSystem.GPS_INFRA:     {"gps": 0.85},
    CyberTargetSystem.RADAR_NETWORK: {"sensor": 0.75},
    CyberTargetSystem.LOGISTICS_DB:  {"logistics_delay": 2.5},
    CyberTargetSystem.POWER_GRID:    {"c2": 0.60, "sensor": 0.40},
    CyberTargetSystem.EARLY_WARNING: {"sensor": 0.90},
    CyberTargetSystem.SENSOR_NETWORK:{"sensor": 0.70, "gps": 0.20},
}


# ──────────────────────────────────────────────────────────────
# CyberEffectEngine
# ──────────────────────────────────────────────────────────────

class CyberEffectEngine:
    """
    사이버 공격 효과 전파 엔진.

    워크플로우:
      1. 공격자 유닛의 cyber_offense 능력 + 공격 강도 → 공격 성공 판정
      2. 목표 시스템의 방어력 → 효과 크기 결정
      3. 영향받는 유닛 목록 추출 → CyberEffectResult 생성
      4. 매 스텝 `step_decay()` 호출로 지속 스텝 감소 및 효과 소멸
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self._active_effects: List[CyberEffectResult] = []
        self._defense_cache: Dict[str, CyberDefenseCapability] = {}

    # ── 방어 능력 추론 ────────────────────────────────────────

    def _infer_defense(self, unit: Unit) -> CyberDefenseCapability:
        """유닛 Capability로부터 방어 능력 추론"""
        cap = unit.capability
        # comms_quality → 암호화·패치, ew_resistance → 탐지
        return CyberDefenseCapability(
            unit_id               = unit.unit_id,
            encryption_level      = float(getattr(cap, "comms_quality", 0.5)),
            redundancy            = float(getattr(cap, "ew_resistance",  0.4)),
            patch_rate            = float(getattr(cap, "comms_quality", 0.5)) * 0.8,
            detection_capability  = float(getattr(cap, "ew_resistance",  0.4)) * 0.9,
        )

    def _attacker_power(self, attacker: Unit, intensity: float) -> float:
        """공격자 사이버 공격력"""
        cyber_off = float(getattr(attacker.capability, "cyber_offense", 0.5))
        return float(np.clip(cyber_off * intensity, 0.0, 1.0))

    # ── 공격 시뮬레이션 ──────────────────────────────────────

    def launch_attack(
        self,
        attack: CyberAttack,
        kg: CombatKnowledgeGraph,
        defender_units: Optional[List[Unit]] = None,
    ) -> CyberEffectResult:
        """
        사이버 공격 실행 → 효과 계산 및 내부 상태에 등록.

        attacker_id가 kg에 없어도 intensity만으로 작동(독립 공격자 가정).
        """
        # ── 공격력
        attacker = kg.units.get(attack.attacker_id)
        atk_power = (
            self._attacker_power(attacker, attack.intensity)
            if attacker is not None
            else float(attack.intensity)
        )

        # ── 피해 측 유닛 탐색
        affected_types = TARGET_SYSTEM_AFFECTED_TYPES.get(
            attack.target_system, set()
        )
        if defender_units is None:
            defender_units = [
                u for u in kg.units.values()
                if u.alignment == attack.affected_alignment
                and u.status != UnitStatus.DESTROYED
                and u.unit_type in affected_types
            ]

        # ── 방어력 평균
        if defender_units:
            avg_def = float(np.mean([
                self._infer_defense(u).overall_defense()
                for u in defender_units
            ]))
        else:
            avg_def = 0.3  # 방어 유닛 없으면 취약

        # ── 성공 판정: 공격력 > 방어력 (확률적)
        success_prob = float(np.clip(
            atk_power / (atk_power + avg_def + 1e-6), 0, 1
        ))
        noise = self.rng.uniform(-0.10, 0.10)
        attack_success = self.rng.random() < (success_prob + noise)

        # ── 효과 크기
        profile = SYSTEM_EFFECT_PROFILE.get(attack.target_system, {})
        severity = float(np.clip(
            atk_power * (1.0 - avg_def * 0.5), 0.0, 1.0
        )) if attack_success else 0.0

        # 효과 유형별 추가 보정
        if attack.effect_type == CyberEffectType.DENIAL:
            severity = min(1.0, severity * 1.3)
        elif attack.effect_type == CyberEffectType.DECEPTION:
            severity = severity * 0.8   # 기만은 탐지 어렵지만 효과 낮음

        # ── 구체적 효과값 계산
        c2_deg  = profile.get("c2",      0.0) * severity
        com_deg = profile.get("comms",   0.0) * severity
        gps_deg = profile.get("gps",     0.0) * severity
        sen_deg = profile.get("sensor",  0.0) * severity
        log_dly = 1.0 + (profile.get("logistics_delay", 0.0) - 1.0) * severity

        affected_ids = [u.unit_id for u in defender_units]

        result = CyberEffectResult(
            target_system        = attack.target_system,
            effect_type          = attack.effect_type,
            attack_success       = attack_success,
            severity             = float(severity),
            duration_remaining   = attack.duration_steps if attack_success else 0,
            affected_unit_ids    = affected_ids,
            c2_degradation       = float(c2_deg),
            comms_degradation    = float(com_deg),
            gps_degradation      = float(gps_deg),
            sensor_degradation   = float(sen_deg),
            logistics_delay_factor = float(log_dly),
        )

        if attack_success:
            self._active_effects.append(result)

        return result

    def step_decay(self) -> None:
        """매 시뮬레이션 스텝: 활성 효과 지속 스텝 감소 및 소멸 처리"""
        remaining = []
        for eff in self._active_effects:
            eff.duration_remaining -= 1
            if eff.duration_remaining > 0:
                remaining.append(eff)
        self._active_effects = remaining

    def get_battle_state(self, step: int, kg: CombatKnowledgeGraph) -> CyberBattleState:
        """현재 스텝의 사이버 전장 상태 계산"""
        state = CyberBattleState(step=step, active_effects=list(self._active_effects))

        for eff in self._active_effects:
            # 영향 유닛의 진영 결정
            alignment = None
            for uid in eff.affected_unit_ids:
                if uid in kg.units:
                    alignment = kg.units[uid].alignment
                    break

            if alignment == ForceAlignment.BLUE:
                state.blue_c2_level    = max(0.0, state.blue_c2_level    - eff.c2_degradation)
                state.blue_comms_level = max(0.0, state.blue_comms_level - eff.comms_degradation)
                state.blue_gps_level   = max(0.0, state.blue_gps_level   - eff.gps_degradation)
            elif alignment == ForceAlignment.RED:
                state.red_c2_level     = max(0.0, state.red_c2_level     - eff.c2_degradation)
                state.red_comms_level  = max(0.0, state.red_comms_level  - eff.comms_degradation)
                state.red_gps_level    = max(0.0, state.red_gps_level    - eff.gps_degradation)

        return state

    @property
    def active_effects(self) -> List[CyberEffectResult]:
        return list(self._active_effects)


# ──────────────────────────────────────────────────────────────
# EW (전자전) 효과 모델
# ──────────────────────────────────────────────────────────────

@dataclass
class EWEffectResult:
    """전자전 효과 결과"""
    jammer_unit_id: str
    jam_type: str               # "gps" | "comms" | "radar" | "combined"
    jam_radius_km: float        # 재밍 유효 반경
    gps_deg: float              # GPS 저하 [0, 1]
    comms_deg: float            # 통신 저하 [0, 1]
    radar_deg: float            # 레이더 저하 [0, 1]
    affected_units: List[str]   # 영향 유닛


class EWEffectModel:
    """
    전자전(Electronic Warfare) 효과 모델.
    재밍 유닛 위치 + 반경 → 영향권 내 유닛에 저하 적용.
    """

    JAM_PROFILES: Dict[str, Dict[str, float]] = {
        "gps":      {"gps_deg": 0.80, "comms_deg": 0.10, "radar_deg": 0.05, "radius_km": 80.0},
        "comms":    {"gps_deg": 0.05, "comms_deg": 0.85, "radar_deg": 0.10, "radius_km": 50.0},
        "radar":    {"gps_deg": 0.05, "comms_deg": 0.10, "radar_deg": 0.80, "radius_km": 60.0},
        "combined": {"gps_deg": 0.55, "comms_deg": 0.55, "radar_deg": 0.50, "radius_km": 40.0},
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def apply_jamming(
        self,
        jammer: Unit,
        kg: CombatKnowledgeGraph,
        jam_type: str = "combined",
        target_alignment: Optional[ForceAlignment] = None,
    ) -> EWEffectResult:
        """
        재머 유닛 기준 반경 내 적군 유닛에 EW 효과 적용.
        """
        profile = self.JAM_PROFILES.get(jam_type, self.JAM_PROFILES["combined"])
        radius  = profile["radius_km"]
        # EW 저항력 보정
        ew_pow = float(getattr(jammer.capability, "ew_offense", 0.5))

        affected: List[str] = []
        for uid, unit in kg.units.items():
            if target_alignment and unit.alignment != target_alignment:
                continue
            if unit.alignment == jammer.alignment:
                continue
            if unit.status == UnitStatus.DESTROYED:
                continue
            dist = jammer.position.distance_to(unit.position)
            if dist > radius:
                continue
            # 거리 감쇄: 1 - (dist/radius)^0.5
            fade = 1.0 - math.sqrt(dist / max(radius, 0.1))
            # 목표 EW 저항력
            resistance = float(getattr(unit.capability, "ew_resistance", 0.3))
            effective = ew_pow * fade * (1.0 - resistance * 0.5)
            if effective > 0.05:
                affected.append(uid)

        noise = float(self.rng.uniform(0.85, 1.15))
        return EWEffectResult(
            jammer_unit_id = jammer.unit_id,
            jam_type       = jam_type,
            jam_radius_km  = radius,
            gps_deg        = float(np.clip(profile["gps_deg"]   * ew_pow * noise, 0, 1)),
            comms_deg      = float(np.clip(profile["comms_deg"] * ew_pow * noise, 0, 1)),
            radar_deg      = float(np.clip(profile["radar_deg"] * ew_pow * noise, 0, 1)),
            affected_units = affected,
        )


# ──────────────────────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory

    print("=== Cyber Effects Engine Test ===")
    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=5, seed=42)
    engine = CyberEffectEngine(seed=42)

    # 사이버 공격 시뮬레이션
    attacks = [
        CyberAttack(
            attacker_id       = "attacker_001",
            target_system     = CyberTargetSystem.C2_NETWORK,
            effect_type       = CyberEffectType.DISRUPTION,
            intensity         = 0.75,
            duration_steps    = 5,
            affected_alignment = ForceAlignment.BLUE,
        ),
        CyberAttack(
            attacker_id       = "attacker_002",
            target_system     = CyberTargetSystem.GPS_INFRA,
            effect_type       = CyberEffectType.DENIAL,
            intensity         = 0.60,
            duration_steps    = 3,
            affected_alignment = ForceAlignment.BLUE,
        ),
    ]

    for atk in attacks:
        result = engine.launch_attack(atk, kg)
        print(f"  {atk.target_system.value:20s} → "
              f"{'성공' if result.attack_success else '실패'} | "
              f"severity={result.severity:.2f} | "
              f"C2저하={result.c2_degradation:.2f} | "
              f"GPS저하={result.gps_degradation:.2f} | "
              f"지속={result.duration_remaining}스텝")

    state = engine.get_battle_state(0, kg)
    print(f"\n  Blue C2={state.blue_c2_level:.2f} | "
          f"Comms={state.blue_comms_level:.2f} | "
          f"GPS={state.blue_gps_level:.2f}")

    print("\n=== EW Effect Test ===")
    ew_model = EWEffectModel(seed=42)
    blue_units = [u for u in kg.units.values()
                  if u.alignment == ForceAlignment.BLUE]
    if blue_units:
        jammer = blue_units[0]
        ew_result = ew_model.apply_jamming(
            jammer, kg, jam_type="gps", target_alignment=ForceAlignment.RED
        )
        print(f"  재머: {jammer.unit_id} | 유형: {ew_result.jam_type} | "
              f"영향 유닛: {len(ew_result.affected_units)}개 | "
              f"GPS저하: {ew_result.gps_deg:.2f}")
