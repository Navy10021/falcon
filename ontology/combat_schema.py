"""
combat_schema.py
================
전장 온톨로지 스키마 정의
Unit, Terrain, Capability, Relationship 클래스 및 Knowledge Graph 빌더

확장 내역 (ONTOLOGY ROADMAP STEP 1-3):
- BranchType: 군종 분류 (육·해·공군·해병대·전략·우주·사이버)
- EchelonType: 지휘 계층 (분대~야전군, 단함~함대, 편대~비행단)
- DomainType: 작전 도메인 (지상·해상·공중·수중·사이버·우주·전략)
- UnitType: 42종 (원래 8종 + 신규 34종 — 육해공 해병대 미사일 드론 사이버 우주)
- Capability: 19개 필드 (기본 8 + 확장 11 — 대기갑/대공/대잠/사이버 등)
- Unit: branch/echelon/domain/designation 필드 추가

학술 연구용 - 합성 데이터 기반
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class UnitType(Enum):
    # ── 기존 8종 (하위 호환) ──────────────────────────
    INFANTRY           = "infantry"
    ARMOR              = "armor"
    ARTILLERY          = "artillery"
    AVIATION           = "aviation"
    ENGINEER           = "engineer"
    SIGNAL             = "signal"
    LOGISTICS          = "logistics"
    ELECTRONIC_WARFARE = "electronic_warfare"

    # ── 육군 / 지상 신규 ─────────────────────────────
    MECHANIZED_INFANTRY    = "mechanized_infantry"      # 기계화보병
    AIRBORNE               = "airborne"                 # 공수
    SPECIAL_FORCES         = "special_forces"           # 특수전
    ANTI_ARMOR             = "anti_armor"               # 대전차
    AIR_DEFENSE_ARTILLERY  = "air_defense_artillery"    # 방공포병
    RECONNAISSANCE         = "reconnaissance"           # 정찰
    NBC_DEFENSE            = "nbc_defense"              # 화생방방호
    MILITARY_POLICE        = "military_police"          # 헌병

    # ── 해군 ─────────────────────────────────────────
    DESTROYER        = "destroyer"        # 구축함
    FRIGATE          = "frigate"          # 호위함
    SUBMARINE        = "submarine"        # 잠수함
    MINE_WARFARE     = "mine_warfare"     # 기뢰전함
    AMPHIBIOUS_SHIP  = "amphibious_ship"  # 상륙함
    NAVAL_AVIATION   = "naval_aviation"   # 함재항공
    COASTAL_DEFENSE  = "coastal_defense"  # 해안방어

    # ── 공군 ─────────────────────────────────────────
    FIGHTER           = "fighter"           # 전투기
    STRIKE_AIRCRAFT   = "strike_aircraft"   # 공격기
    BOMBER            = "bomber"            # 폭격기
    TRANSPORT_AIRCRAFT= "transport_aircraft"# 수송기
    ISR_AIRCRAFT      = "isr_aircraft"      # ISR기
    EARLY_WARNING     = "early_warning"     # 조기경보기
    AIR_REFUELING     = "air_refueling"     # 공중급유기
    SAM_BATTERY       = "sam_battery"       # 지대공미사일포대

    # ── 해병대 ───────────────────────────────────────
    MARINE_INFANTRY = "marine_infantry"  # 해병보병
    MARINE_ARMOR    = "marine_armor"     # 해병기갑
    MARINE_AVIATION = "marine_aviation"  # 해병항공

    # ── 미사일/로켓 ──────────────────────────────────
    BALLISTIC_MISSILE  = "ballistic_missile"   # 탄도미사일
    CRUISE_MISSILE     = "cruise_missile"      # 순항미사일
    MLRS               = "mlrs"                # 다련장로켓

    # ── 드론/무인체계 ────────────────────────────────
    UAV_RECON          = "uav_recon"           # 정찰 UAV
    UAV_STRIKE         = "uav_strike"          # 공격 UAV
    LOITERING_MUNITION = "loitering_munition"  # 배회형탄약

    # ── 사이버 / 우주 ────────────────────────────────
    CYBER_UNIT  = "cyber_unit"   # 사이버부대
    SPACE_ASSET = "space_asset"  # 우주자산 (위성 등)


class BranchType(Enum):
    """군종"""
    ARMY         = "army"          # 육군
    NAVY         = "navy"          # 해군
    AIR_FORCE    = "air_force"     # 공군
    MARINE_CORPS = "marine_corps"  # 해병대
    STRATEGIC    = "strategic"     # 전략사령부 (탄도탄 등)
    SPACE        = "space"         # 우주군
    CYBER        = "cyber"         # 사이버사령부


class EchelonType(Enum):
    """지휘 계층 제대"""
    # 지상
    SQUAD      = "squad"        # 분대 (8-12명)
    PLATOON    = "platoon"      # 소대 (30-50명)
    COMPANY    = "company"      # 중대 (100-200명)
    BATTALION  = "battalion"    # 대대 (400-1000명)
    REGIMENT   = "regiment"     # 연대 (2000-3500명)
    BRIGADE    = "brigade"      # 여단 (3000-5000명)
    DIVISION   = "division"     # 사단 (10000-20000명)
    CORPS      = "corps"        # 군단 (30000-80000명)
    FIELD_ARMY = "field_army"   # 야전군 (100000명+)
    # 해군
    SHIP       = "ship"         # 단함
    TASK_GROUP = "task_group"   # 기동전단
    FLEET      = "fleet"        # 함대
    # 공군
    FLIGHT     = "flight"       # 편대 (2-4기)
    SQUADRON   = "squadron"     # 비행대대 (12-24기)
    WING       = "wing"         # 비행단 (50-100기)
    # 사이버/특수
    TEAM       = "team"         # 팀


class DomainType(Enum):
    """작전 도메인"""
    LAND       = "land"        # 지상
    SEA        = "sea"         # 해상
    AIR        = "air"         # 공중
    SUBSURFACE = "subsurface"  # 수중
    CYBER      = "cyber"       # 사이버
    SPACE      = "space"       # 우주
    STRATEGIC  = "strategic"   # 전략 (탄도미사일 등)


class TerrainType(Enum):
    OPEN     = "open"
    URBAN    = "urban"
    FOREST   = "forest"
    MOUNTAIN = "mountain"
    RIVER    = "river"
    COASTAL  = "coastal"


class ForceAlignment(Enum):
    BLUE    = "blue"     # 아군
    RED     = "red"      # 적군
    NEUTRAL = "neutral"


class UnitStatus(Enum):
    ACTIVE    = "active"
    DEGRADED  = "degraded"    # 30~70% 전투력
    CRITICAL  = "critical"    # < 30% 전투력
    DESTROYED = "destroyed"


class RelationType(Enum):
    SUPPORTS    = "supports"
    THREATENS   = "threatens"
    ADJACENT_TO = "adjacent_to"
    CONTROLS    = "controls"
    SUPPLIES    = "supplies"
    BLOCKS      = "blocks"


# ──────────────────────────────────────────────
# Branch / Echelon / Domain 매핑 테이블
# ──────────────────────────────────────────────

# UnitType → 기본 군종
_UNIT_BRANCH: Dict[str, BranchType] = {
    # 육군 기존
    "infantry":           BranchType.ARMY,
    "armor":              BranchType.ARMY,
    "artillery":          BranchType.ARMY,
    "aviation":           BranchType.ARMY,
    "engineer":           BranchType.ARMY,
    "signal":             BranchType.ARMY,
    "logistics":          BranchType.ARMY,
    "electronic_warfare": BranchType.CYBER,
    # 육군 신규
    "mechanized_infantry":   BranchType.ARMY,
    "airborne":              BranchType.ARMY,
    "special_forces":        BranchType.ARMY,
    "anti_armor":            BranchType.ARMY,
    "air_defense_artillery": BranchType.ARMY,
    "reconnaissance":        BranchType.ARMY,
    "nbc_defense":           BranchType.ARMY,
    "military_police":       BranchType.ARMY,
    # 해군
    "destroyer":       BranchType.NAVY,
    "frigate":         BranchType.NAVY,
    "submarine":       BranchType.NAVY,
    "mine_warfare":    BranchType.NAVY,
    "amphibious_ship": BranchType.NAVY,
    "naval_aviation":  BranchType.NAVY,
    "coastal_defense": BranchType.NAVY,
    # 공군
    "fighter":            BranchType.AIR_FORCE,
    "strike_aircraft":    BranchType.AIR_FORCE,
    "bomber":             BranchType.AIR_FORCE,
    "transport_aircraft": BranchType.AIR_FORCE,
    "isr_aircraft":       BranchType.AIR_FORCE,
    "early_warning":      BranchType.AIR_FORCE,
    "air_refueling":      BranchType.AIR_FORCE,
    "sam_battery":        BranchType.AIR_FORCE,
    # 해병대
    "marine_infantry": BranchType.MARINE_CORPS,
    "marine_armor":    BranchType.MARINE_CORPS,
    "marine_aviation": BranchType.MARINE_CORPS,
    # 전략 / 미사일
    "ballistic_missile": BranchType.STRATEGIC,
    "cruise_missile":    BranchType.STRATEGIC,
    "mlrs":              BranchType.ARMY,
    # 드론
    "uav_recon":          BranchType.AIR_FORCE,
    "uav_strike":         BranchType.AIR_FORCE,
    "loitering_munition": BranchType.ARMY,
    # 사이버/우주
    "cyber_unit":  BranchType.CYBER,
    "space_asset": BranchType.SPACE,
}

# UnitType → 기본 작전 도메인
_UNIT_DOMAIN: Dict[str, DomainType] = {
    "infantry": DomainType.LAND, "armor": DomainType.LAND,
    "artillery": DomainType.LAND, "aviation": DomainType.AIR,
    "engineer": DomainType.LAND, "signal": DomainType.LAND,
    "logistics": DomainType.LAND, "electronic_warfare": DomainType.CYBER,
    "mechanized_infantry": DomainType.LAND, "airborne": DomainType.AIR,
    "special_forces": DomainType.LAND, "anti_armor": DomainType.LAND,
    "air_defense_artillery": DomainType.LAND, "reconnaissance": DomainType.LAND,
    "nbc_defense": DomainType.LAND, "military_police": DomainType.LAND,
    "destroyer": DomainType.SEA, "frigate": DomainType.SEA,
    "submarine": DomainType.SUBSURFACE, "mine_warfare": DomainType.SEA,
    "amphibious_ship": DomainType.SEA, "naval_aviation": DomainType.AIR,
    "coastal_defense": DomainType.LAND,
    "fighter": DomainType.AIR, "strike_aircraft": DomainType.AIR,
    "bomber": DomainType.AIR, "transport_aircraft": DomainType.AIR,
    "isr_aircraft": DomainType.AIR, "early_warning": DomainType.AIR,
    "air_refueling": DomainType.AIR, "sam_battery": DomainType.LAND,
    "marine_infantry": DomainType.LAND, "marine_armor": DomainType.LAND,
    "marine_aviation": DomainType.AIR,
    "ballistic_missile": DomainType.STRATEGIC,
    "cruise_missile": DomainType.STRATEGIC, "mlrs": DomainType.LAND,
    "uav_recon": DomainType.AIR, "uav_strike": DomainType.AIR,
    "loitering_munition": DomainType.AIR,
    "cyber_unit": DomainType.CYBER, "space_asset": DomainType.SPACE,
}


# ──────────────────────────────────────────────
# Core Data Classes
# ──────────────────────────────────────────────

@dataclass
class Capability:
    """
    유닛 능력치 스키마 (확장판)

    기본 8개 필드 + 확장 11개 필드 (도메인 특화)
    to_vector()      : 기본 8-dim (GNN 입력 호환)
    to_extended_vector(): 전체 19-dim (Unit.to_feature_vector() 에서 사용)
    """
    # ── 기본 8개 필드 ────────────────────────────────
    firepower:     float    # 화력 지수 [0, 1]
    mobility:      float    # 기동력 [0, 1]
    protection:    float    # 방호력 [0, 1]
    range_km:      float    # 유효 사거리 (km)
    ammo_level:    float    # 탄약 수준 [0, 1]
    fuel_level:    float    # 연료 수준 [0, 1]
    comms_quality: float    # 통신 품질 [0, 1]
    ew_resistance: float    # 전자전 저항력 [0, 1]

    # ── 확장 11개 필드 (기본값 0.0 — 하위 호환) ─────
    anti_armor:        float = 0.0  # 대기갑 능력 [0, 1]
    anti_air:          float = 0.0  # 대공 능력 [0, 1]
    anti_submarine:    float = 0.0  # 대잠 능력 [0, 1]
    air_to_air:        float = 0.0  # 공대공 능력 [0, 1]
    stealth_factor:    float = 0.0  # 스텔스 지수 [0, 1]
    cyber_offense:     float = 0.0  # 사이버 공격 [0, 1]
    cyber_defense:     float = 0.0  # 사이버 방어 [0, 1]
    ew_offense:        float = 0.0  # 전자전 공격 [0, 1]
    isr_capability:    float = 0.0  # ISR 능력 [0, 1]
    amphibious_capable:float = 0.0  # 상륙 작전 능력 [0, 1]
    air_assault_capable:float= 0.0  # 공중강습 능력 [0, 1]

    def to_vector(self) -> np.ndarray:
        """기본 8-dim 능력치 벡터 (GNN 하위 호환)"""
        return np.array([
            self.firepower, self.mobility, self.protection,
            self.range_km / 100.0,  # normalize
            self.ammo_level, self.fuel_level,
            self.comms_quality, self.ew_resistance
        ], dtype=np.float32)

    def to_extended_vector(self) -> np.ndarray:
        """전체 19-dim 능력치 벡터 (Unit.to_feature_vector 사용)"""
        base = self.to_vector()  # 8-dim
        ext = np.array([
            self.anti_armor, self.anti_air, self.anti_submarine,
            self.air_to_air, self.stealth_factor,
            self.cyber_offense, self.cyber_defense, self.ew_offense,
            self.isr_capability, self.amphibious_capable, self.air_assault_capable,
        ], dtype=np.float32)
        return np.concatenate([base, ext])  # 19-dim

    @classmethod
    def from_unit_type(cls, unit_type: UnitType, noise_std: float = 0.05) -> "Capability":
        """유닛 유형별 기본 능력치 생성 (약간의 노이즈 포함)"""
        # (firepower, mobility, protection, range_km,
        #  ammo_level, fuel_level, comms_quality, ew_resistance)
        base_caps: Dict[UnitType, Tuple] = {
            # ── 기존 8종 ──────────────────────────────
            UnitType.INFANTRY:           (0.50, 0.40, 0.40,    2.0, 0.80, 0.90, 0.70, 0.50),
            UnitType.ARMOR:              (0.90, 0.60, 0.90,    3.0, 0.70, 0.60, 0.60, 0.40),
            UnitType.ARTILLERY:          (0.95, 0.30, 0.30,   30.0, 0.70, 0.60, 0.70, 0.50),
            UnitType.AVIATION:           (0.80, 0.95, 0.50,   50.0, 0.50, 0.40, 0.80, 0.60),
            UnitType.ENGINEER:           (0.30, 0.50, 0.50,    1.0, 0.80, 0.70, 0.70, 0.50),
            UnitType.SIGNAL:             (0.10, 0.50, 0.30,    5.0, 0.90, 0.80, 0.99, 0.80),
            UnitType.LOGISTICS:          (0.20, 0.60, 0.30,    1.0, 0.90, 0.90, 0.70, 0.40),
            UnitType.ELECTRONIC_WARFARE: (0.30, 0.40, 0.30,   20.0, 0.80, 0.70, 0.90, 0.90),
            # ── 육군 신규 ─────────────────────────────
            UnitType.MECHANIZED_INFANTRY:(0.65, 0.65, 0.65,    3.0, 0.80, 0.70, 0.70, 0.50),
            UnitType.AIRBORNE:           (0.60, 0.80, 0.30,    2.5, 0.70, 0.80, 0.75, 0.50),
            UnitType.SPECIAL_FORCES:     (0.70, 0.90, 0.40,    3.0, 0.80, 0.85, 0.85, 0.70),
            UnitType.ANTI_ARMOR:         (0.85, 0.50, 0.40,    5.0, 0.75, 0.70, 0.70, 0.50),
            UnitType.AIR_DEFENSE_ARTILLERY:(0.70,0.30, 0.40,  50.0, 0.70, 0.60, 0.80, 0.70),
            UnitType.RECONNAISSANCE:     (0.30, 0.90, 0.30,   15.0, 0.70, 0.85, 0.90, 0.70),
            UnitType.NBC_DEFENSE:        (0.20, 0.40, 0.70,    2.0, 0.80, 0.70, 0.70, 0.50),
            UnitType.MILITARY_POLICE:    (0.40, 0.60, 0.50,    2.0, 0.80, 0.80, 0.70, 0.40),
            # ── 해군 ──────────────────────────────────
            UnitType.DESTROYER:          (0.85, 0.70, 0.60,  150.0, 0.70, 0.60, 0.85, 0.70),
            UnitType.FRIGATE:            (0.70, 0.75, 0.55,  100.0, 0.70, 0.65, 0.80, 0.60),
            UnitType.SUBMARINE:          (0.90, 0.60, 0.80,  200.0, 0.75, 0.70, 0.70, 0.80),
            UnitType.MINE_WARFARE:       (0.50, 0.40, 0.50,   20.0, 0.80, 0.60, 0.60, 0.40),
            UnitType.AMPHIBIOUS_SHIP:    (0.40, 0.40, 0.60,   20.0, 0.70, 0.50, 0.75, 0.50),
            UnitType.NAVAL_AVIATION:     (0.85, 0.90, 0.50,  500.0, 0.50, 0.40, 0.85, 0.70),
            UnitType.COASTAL_DEFENSE:    (0.75, 0.20, 0.50,  200.0, 0.70, 0.60, 0.75, 0.60),
            # ── 공군 ──────────────────────────────────
            UnitType.FIGHTER:            (0.85, 0.98, 0.50,  200.0, 0.50, 0.30, 0.90, 0.70),
            UnitType.STRIKE_AIRCRAFT:    (0.90, 0.90, 0.40,  400.0, 0.50, 0.30, 0.85, 0.65),
            UnitType.BOMBER:             (0.95, 0.70, 0.40, 1000.0, 0.60, 0.30, 0.85, 0.65),
            UnitType.TRANSPORT_AIRCRAFT: (0.10, 0.85, 0.30, 2000.0, 0.90, 0.40, 0.85, 0.50),
            UnitType.ISR_AIRCRAFT:       (0.10, 0.80, 0.30,  500.0, 0.80, 0.40, 0.95, 0.60),
            UnitType.EARLY_WARNING:      (0.10, 0.75, 0.30,  500.0, 0.70, 0.40, 0.99, 0.70),
            UnitType.AIR_REFUELING:      (0.10, 0.70, 0.20, 2000.0, 0.90, 0.50, 0.85, 0.50),
            UnitType.SAM_BATTERY:        (0.80, 0.20, 0.50,  100.0, 0.70, 0.60, 0.85, 0.60),
            # ── 해병대 ────────────────────────────────
            UnitType.MARINE_INFANTRY:    (0.60, 0.70, 0.40,    3.0, 0.80, 0.85, 0.75, 0.55),
            UnitType.MARINE_ARMOR:       (0.80, 0.65, 0.80,    4.0, 0.70, 0.60, 0.70, 0.50),
            UnitType.MARINE_AVIATION:    (0.80, 0.95, 0.50,  200.0, 0.50, 0.40, 0.85, 0.65),
            # ── 미사일/로켓 ───────────────────────────
            UnitType.BALLISTIC_MISSILE:  (0.99, 0.10, 0.50, 1000.0, 0.60, 0.70, 0.50, 0.60),
            UnitType.CRUISE_MISSILE:     (0.95, 0.70, 0.30,  500.0, 0.80, 0.80, 0.50, 0.70),
            UnitType.MLRS:               (0.90, 0.40, 0.30,   80.0, 0.70, 0.60, 0.70, 0.50),
            # ── 드론 ──────────────────────────────────
            UnitType.UAV_RECON:          (0.10, 0.90, 0.10,  200.0, 0.90, 0.70, 0.95, 0.50),
            UnitType.UAV_STRIKE:         (0.80, 0.85, 0.20,  200.0, 0.60, 0.70, 0.90, 0.60),
            UnitType.LOITERING_MUNITION: (0.85, 0.80, 0.10,  100.0, 0.70, 0.70, 0.85, 0.50),
            # ── 사이버/우주 ───────────────────────────
            UnitType.CYBER_UNIT:         (0.10, 0.10, 0.10,10000.0, 0.90, 0.90, 0.99, 0.99),
            UnitType.SPACE_ASSET:        (0.50, 0.10, 0.30,10000.0, 0.80, 0.90, 0.99, 0.80),
        }

        # 확장 능력치 (anti_armor, anti_air, anti_sub, air_to_air,
        #              stealth, cyber_off, cyber_def, ew_off, isr, amphibious, air_assault)
        ext_caps: Dict[UnitType, Tuple] = {
            UnitType.INFANTRY:           (0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.10, 0.00, 0.00),
            UnitType.ARMOR:              (0.70, 0.10, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.10, 0.00, 0.00),
            UnitType.ARTILLERY:          (0.40, 0.30, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.20, 0.00, 0.00),
            UnitType.AVIATION:           (0.30, 0.40, 0.00, 0.50, 0.00, 0.00, 0.10, 0.00, 0.40, 0.00, 0.70),
            UnitType.ENGINEER:           (0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.05, 0.30, 0.00),
            UnitType.SIGNAL:             (0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.60, 0.20, 0.50, 0.00, 0.00),
            UnitType.LOGISTICS:          (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00),
            UnitType.ELECTRONIC_WARFARE: (0.00, 0.10, 0.00, 0.00, 0.00, 0.50, 0.60, 0.80, 0.60, 0.00, 0.00),
            # 육군 신규
            UnitType.MECHANIZED_INFANTRY:(0.40, 0.20, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.15, 0.00, 0.00),
            UnitType.AIRBORNE:           (0.30, 0.10, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.20, 0.00, 0.90),
            UnitType.SPECIAL_FORCES:     (0.40, 0.10, 0.00, 0.00, 0.20, 0.20, 0.30, 0.10, 0.70, 0.30, 0.50),
            UnitType.ANTI_ARMOR:         (0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.10, 0.00, 0.00),
            UnitType.AIR_DEFENSE_ARTILLERY:(0.00,0.90,0.00, 0.00, 0.00, 0.00, 0.10, 0.20, 0.30, 0.00, 0.00),
            UnitType.RECONNAISSANCE:     (0.05, 0.10, 0.00, 0.00, 0.10, 0.00, 0.20, 0.30, 0.90, 0.00, 0.20),
            UnitType.NBC_DEFENSE:        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.20, 0.00, 0.00),
            UnitType.MILITARY_POLICE:    (0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.10, 0.00, 0.00),
            # 해군
            UnitType.DESTROYER:          (0.30, 0.70, 0.60, 0.10, 0.00, 0.10, 0.50, 0.30, 0.50, 0.00, 0.00),
            UnitType.FRIGATE:            (0.20, 0.50, 0.70, 0.00, 0.00, 0.05, 0.40, 0.20, 0.40, 0.00, 0.00),
            UnitType.SUBMARINE:          (0.20, 0.10, 0.50, 0.00, 0.90, 0.05, 0.30, 0.10, 0.30, 0.00, 0.00),
            UnitType.MINE_WARFARE:       (0.10, 0.20, 0.40, 0.00, 0.20, 0.00, 0.10, 0.00, 0.20, 0.00, 0.00),
            UnitType.AMPHIBIOUS_SHIP:    (0.10, 0.00, 0.10, 0.00, 0.00, 0.00, 0.20, 0.00, 0.20, 0.90, 0.00),
            UnitType.NAVAL_AVIATION:     (0.40, 0.60, 0.50, 0.60, 0.20, 0.00, 0.30, 0.00, 0.50, 0.30, 0.60),
            UnitType.COASTAL_DEFENSE:    (0.20, 0.80, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.30, 0.00, 0.00),
            # 공군
            UnitType.FIGHTER:            (0.40, 0.05, 0.00, 0.95, 0.10, 0.00, 0.20, 0.00, 0.30, 0.00, 0.00),
            UnitType.STRIKE_AIRCRAFT:    (0.80, 0.00, 0.00, 0.30, 0.20, 0.00, 0.20, 0.00, 0.40, 0.00, 0.00),
            UnitType.BOMBER:             (0.70, 0.00, 0.00, 0.10, 0.30, 0.00, 0.20, 0.00, 0.30, 0.00, 0.00),
            UnitType.TRANSPORT_AIRCRAFT: (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.10, 0.50, 0.60),
            UnitType.ISR_AIRCRAFT:       (0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.20, 0.50, 0.99, 0.00, 0.00),
            UnitType.EARLY_WARNING:      (0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.30, 0.40, 0.95, 0.00, 0.00),
            UnitType.AIR_REFUELING:      (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.05, 0.00, 0.00),
            UnitType.SAM_BATTERY:        (0.00, 0.95, 0.00, 0.00, 0.00, 0.00, 0.20, 0.30, 0.50, 0.00, 0.00),
            # 해병대
            UnitType.MARINE_INFANTRY:    (0.20, 0.05, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.20, 0.80, 0.00),
            UnitType.MARINE_ARMOR:       (0.50, 0.10, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.20, 0.70, 0.00),
            UnitType.MARINE_AVIATION:    (0.30, 0.50, 0.00, 0.50, 0.00, 0.00, 0.20, 0.00, 0.40, 0.60, 0.40),
            # 미사일
            UnitType.BALLISTIC_MISSILE:  (0.60, 0.00, 0.00, 0.00, 0.40, 0.00, 0.10, 0.00, 0.10, 0.00, 0.00),
            UnitType.CRUISE_MISSILE:     (0.70, 0.00, 0.00, 0.00, 0.50, 0.00, 0.10, 0.00, 0.10, 0.00, 0.00),
            UnitType.MLRS:               (0.60, 0.20, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.10, 0.00, 0.00),
            # 드론
            UnitType.UAV_RECON:          (0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.10, 0.00, 0.95, 0.00, 0.00),
            UnitType.UAV_STRIKE:         (0.70, 0.00, 0.00, 0.10, 0.50, 0.00, 0.10, 0.00, 0.50, 0.00, 0.00),
            UnitType.LOITERING_MUNITION: (0.85, 0.00, 0.00, 0.00, 0.40, 0.00, 0.10, 0.00, 0.30, 0.00, 0.00),
            # 사이버/우주
            UnitType.CYBER_UNIT:         (0.00, 0.00, 0.00, 0.00, 0.00, 0.95, 0.85, 0.60, 0.50, 0.00, 0.00),
            UnitType.SPACE_ASSET:        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.99, 0.00, 0.00),
        }

        bv = base_caps.get(unit_type, base_caps[UnitType.INFANTRY])
        ev = ext_caps.get(unit_type, (0.0,) * 11)

        def _n(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return float(np.clip(v + np.random.normal(0, noise_std), lo, hi))

        return cls(
            firepower=_n(bv[0]),      mobility=_n(bv[1]),
            protection=_n(bv[2]),     range_km=max(0.5, bv[3] + np.random.normal(0, 1.0)),
            ammo_level=_n(bv[4]),     fuel_level=_n(bv[5]),
            comms_quality=_n(bv[6]),  ew_resistance=bv[7],      # ew_resistance: no noise
            # 확장 필드 (small noise)
            anti_armor=_n(ev[0]),       anti_air=_n(ev[1]),
            anti_submarine=_n(ev[2]),   air_to_air=_n(ev[3]),
            stealth_factor=_n(ev[4]),
            cyber_offense=_n(ev[5]),    cyber_defense=_n(ev[6]),
            ew_offense=_n(ev[7]),       isr_capability=_n(ev[8]),
            amphibious_capable=_n(ev[9]), air_assault_capable=_n(ev[10]),
        )


@dataclass
class Position:
    """전장 좌표"""
    x: float    # 동서 (km)
    y: float    # 남북 (km)
    z: float = 0.0  # 고도 (km)

    def distance_to(self, other: "Position") -> float:
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2))

    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class Unit:
    """
    전술 유닛 스키마 (확장판)

    기본 필드 외에 군종(branch), 제대(echelon), 작전도메인(domain),
    상하위 부대 연결, 부대 명칭, 교리 레이블이 추가됨.
    모두 기본값이 있으므로 기존 코드와 하위 호환.
    """
    # ── 필수 필드 (기존) ─────────────────────────────
    unit_id:    str
    unit_type:  UnitType
    alignment:  ForceAlignment
    position:   Position
    capability: Capability
    headcount:  int              # 병력 수

    # ── 선택 필드 (기존, 기본값 있음) ───────────────
    status:      UnitStatus = UnitStatus.ACTIVE
    morale:      float      = 0.8     # 사기 [0, 1]
    experience:  float      = 0.5     # 경험치 [0, 1]
    objective:   Optional[str] = None
    attached_to: Optional[str] = None  # 상위 부대 ID (레거시)

    # ── 신규 확장 필드 ───────────────────────────────
    branch:          BranchType  = field(default=None)    # type: ignore[assignment]
    echelon:         EchelonType = EchelonType.BATTALION
    domain:          DomainType  = field(default=None)    # type: ignore[assignment]
    parent_unit_id:  Optional[str] = None
    child_unit_ids:  List[str]   = field(default_factory=list)
    designation:     str         = ""   # 부대 명칭 (예: "1사단 11연대 2대대")
    doctrine_label:  Optional[str] = None

    def __post_init__(self):
        """branch / domain 미지정 시 UnitType 기반 자동 추론"""
        if self.branch is None:
            self.branch = _UNIT_BRANCH.get(
                self.unit_type.value, BranchType.ARMY)
        if self.domain is None:
            self.domain = _UNIT_DOMAIN.get(
                self.unit_type.value, DomainType.LAND)

    @property
    def combat_power(self) -> float:
        """종합 전투력 지수 계산"""
        cap_vec = self.capability.to_vector()
        weights = np.array([0.25, 0.15, 0.20, 0.05, 0.10, 0.05, 0.10, 0.10])
        base = float(np.dot(cap_vec[:8], weights))
        morale_factor = 0.7 + 0.3 * self.morale
        exp_factor    = 0.8 + 0.2 * self.experience
        status_factor = {
            UnitStatus.ACTIVE:    1.0,
            UnitStatus.DEGRADED:  0.55,
            UnitStatus.CRITICAL:  0.25,
            UnitStatus.DESTROYED: 0.0
        }[self.status]
        return base * morale_factor * exp_factor * status_factor

    def to_feature_vector(self) -> np.ndarray:
        """
        GNN 노드 특성 벡터 (105-dim → 128-dim pad in get_node_features)

        구성:
          unit_type_onehot (42) + alignment_onehot (3) + status_onehot (4)
          + branch_onehot (7) + echelon_onehot (16) + domain_onehot (7)
          + position (3) + capability_extended (19)
          + [headcount/1000, morale, experience, combat_power] (4)
          = 105-dim
        """
        unit_types    = list(UnitType)
        alignments    = list(ForceAlignment)
        statuses      = list(UnitStatus)
        branches      = list(BranchType)
        echelons      = list(EchelonType)
        domains       = list(DomainType)

        ut_onehot = np.zeros(len(unit_types),  dtype=np.float32)
        al_onehot = np.zeros(len(alignments),  dtype=np.float32)
        st_onehot = np.zeros(len(statuses),    dtype=np.float32)
        br_onehot = np.zeros(len(branches),    dtype=np.float32)
        ec_onehot = np.zeros(len(echelons),    dtype=np.float32)
        dm_onehot = np.zeros(len(domains),     dtype=np.float32)

        ut_onehot[unit_types.index(self.unit_type)]    = 1.0
        al_onehot[alignments.index(self.alignment)]    = 1.0
        st_onehot[statuses.index(self.status)]         = 1.0
        br_onehot[branches.index(self.branch)]         = 1.0
        ec_onehot[echelons.index(self.echelon)]        = 1.0
        dm_onehot[domains.index(self.domain)]          = 1.0

        return np.concatenate([
            ut_onehot,                          # 42
            al_onehot,                          # 3
            st_onehot,                          # 4
            br_onehot,                          # 7
            ec_onehot,                          # 16
            dm_onehot,                          # 7
            self.position.to_vector(),          # 3
            self.capability.to_extended_vector(),# 19
            [self.headcount / 1000.0,           # 1
             self.morale,                       # 1
             self.experience,                   # 1
             self.combat_power],                # 1
        ]).astype(np.float32)                   # total: 105


@dataclass
class TerrainCell:
    """지형 셀 스키마"""
    cell_id:      str
    position:     Position
    terrain_type: TerrainType
    elevation:    float = 0.0    # 고도 (m)
    cover_factor: float = 0.0   # 엄폐 계수 [0, 1]
    concealment:  float = 0.0   # 은폐 계수 [0, 1]
    movement_cost:float = 1.0   # 기동 비용 배수

    @classmethod
    def from_type(cls, cell_id: str, pos: Position, terrain: TerrainType) -> "TerrainCell":
        terrain_params = {
            TerrainType.OPEN:     (0.0,  0.0, 0.05, 1.0),
            TerrainType.URBAN:    (0.0,  0.7, 0.60, 2.5),
            TerrainType.FOREST:   (50.0, 0.4, 0.80, 1.8),
            TerrainType.MOUNTAIN: (500., 0.5, 0.60, 3.5),
            TerrainType.RIVER:    (0.0,  0.1, 0.10, 4.0),
            TerrainType.COASTAL:  (0.0,  0.2, 0.20, 1.5),
        }
        elev, cover, conceal, move = terrain_params[terrain]
        return cls(cell_id, pos, terrain, elev, cover, conceal, move)

    def to_feature_vector(self) -> np.ndarray:
        terrain_onehot = np.zeros(len(TerrainType))
        terrain_onehot[list(TerrainType).index(self.terrain_type)] = 1.0
        return np.concatenate([
            terrain_onehot,
            [self.elevation / 1000.0, self.cover_factor,
             self.concealment, self.movement_cost / 4.0]
        ]).astype(np.float32)


# ──────────────────────────────────────────────
# Knowledge Graph Builder
# ──────────────────────────────────────────────

# GNN 입력 노드 특성 차원 (Unit: 105, TerrainCell: 10, 패딩 목표: 128)
NODE_FEATURE_DIM = 128


class CombatKnowledgeGraph:
    """
    전장 지식 그래프 (이종 그래프)
    노드 유형: Unit, TerrainCell
    엣지 유형: RelationType
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.units: Dict[str, Unit] = {}
        self.terrain_cells: Dict[str, TerrainCell] = {}
        self._edge_counter = 0

    def add_unit(self, unit: Unit) -> None:
        self.units[unit.unit_id] = unit
        self.graph.add_node(
            unit.unit_id,
            node_type="unit",
            features=unit.to_feature_vector(),
            alignment=unit.alignment.value
        )

    def add_terrain(self, cell: TerrainCell) -> None:
        self.terrain_cells[cell.cell_id] = cell
        self.graph.add_node(
            cell.cell_id,
            node_type="terrain",
            features=cell.to_feature_vector()
        )

    def add_relation(
        self,
        src_id: str,
        dst_id: str,
        relation: RelationType,
        weight: float = 1.0,
        **attrs
    ) -> None:
        self.graph.add_edge(
            src_id, dst_id,
            key=self._edge_counter,
            relation=relation.value,
            weight=weight,
            **attrs
        )
        self._edge_counter += 1

    def build_spatial_relations(self, support_radius_km: float = 5.0,
                                 threat_radius_km: float = 10.0) -> None:
        """공간 관계 자동 생성"""
        unit_list = list(self.units.values())
        for i, u1 in enumerate(unit_list):
            for j, u2 in enumerate(unit_list):
                if i == j:
                    continue
                dist = u1.position.distance_to(u2.position)
                # 인접(아군 지원) 관계
                if dist <= support_radius_km:
                    if u1.alignment == u2.alignment:
                        self.add_relation(u1.unit_id, u2.unit_id,
                                          RelationType.SUPPORTS,
                                          weight=1 - dist / support_radius_km)
                # 위협(교전 가능) 관계
                if dist <= u1.capability.range_km and u1.alignment != u2.alignment:
                    self.add_relation(u1.unit_id, u2.unit_id,
                                      RelationType.THREATENS,
                                      weight=1 - dist / threat_radius_km)

        # 지형-유닛 점령 관계
        for unit in self.units.values():
            closest_cell = min(
                self.terrain_cells.values(),
                key=lambda c: unit.position.distance_to(c.position),
                default=None
            )
            if closest_cell:
                self.add_relation(unit.unit_id, closest_cell.cell_id,
                                  RelationType.CONTROLS, weight=unit.combat_power)

    def get_adjacency_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """엣지 인덱스 및 속성 반환 (PyG 형식)"""
        node_list = list(self.graph.nodes())
        node_idx  = {n: i for i, n in enumerate(node_list)}

        edge_index   = []
        edge_weights = []
        for src, dst, data in self.graph.edges(data=True):
            edge_index.append([node_idx[src], node_idx[dst]])
            edge_weights.append(data.get("weight", 1.0))

        if edge_index:
            return (np.array(edge_index, dtype=np.int64).T,
                    np.array(edge_weights, dtype=np.float32))
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    def get_node_features(self) -> np.ndarray:
        """전체 노드 특성 행렬 반환 (NODE_FEATURE_DIM=128 으로 패딩)"""
        features = []
        for node_id in self.graph.nodes():
            feat = self.graph.nodes[node_id]["features"]
            padded = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
            padded[:len(feat)] = feat
            features.append(padded)
        return np.stack(features) if features else np.zeros((0, NODE_FEATURE_DIM))

    def get_stats(self) -> Dict:
        """그래프 통계"""
        blue_units = [u for u in self.units.values() if u.alignment == ForceAlignment.BLUE]
        red_units  = [u for u in self.units.values() if u.alignment == ForceAlignment.RED]
        return {
            "total_nodes":       self.graph.number_of_nodes(),
            "total_edges":       self.graph.number_of_edges(),
            "blue_units":        len(blue_units),
            "red_units":         len(red_units),
            "blue_headcount":    sum(u.headcount for u in blue_units),
            "red_headcount":     sum(u.headcount for u in red_units),
            "blue_combat_power": sum(u.combat_power for u in blue_units),
            "red_combat_power":  sum(u.combat_power for u in red_units),
        }

    def update_node_features(self) -> None:
        """
        P3-2: 시뮬레이션 스텝 이후 변경된 유닛 상태를 그래프 노드 features에 동기화.

        headcount, status, morale, experience, combat_power 등 동적으로 변하는
        속성이 GNN 입력에 실시간 반영되도록 한다.
        (TerrainCell은 정적이므로 유닛 노드만 갱신)
        """
        for unit_id, unit in self.units.items():
            if unit_id in self.graph.nodes:
                self.graph.nodes[unit_id]["features"] = unit.to_feature_vector()


# ──────────────────────────────────────────────
# Scenario Factory
# ──────────────────────────────────────────────

class ScenarioFactory:
    """표준 시나리오 생성기"""

    @staticmethod
    def create_standard_scenario(
        n_blue: int = 8,
        n_red: int = 6,
        map_size_km: float = 30.0,
        seed: Optional[int] = None
    ) -> CombatKnowledgeGraph:
        """표준 교전 시나리오 생성 (기존 호환 — 육군 위주)"""
        rng = np.random.RandomState(seed)

        kg = CombatKnowledgeGraph()

        # 지형 셀 생성 (격자)
        terrain_types = list(TerrainType)
        grid_size = 5
        cell_size = map_size_km / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                terrain = terrain_types[rng.randint(len(terrain_types))]
                pos = Position(i * cell_size + cell_size / 2,
                               j * cell_size + cell_size / 2)
                cell = TerrainCell.from_type(f"cell_{i}_{j}", pos, terrain)
                kg.add_terrain(cell)

        # Blue 유닛 생성 (좌측)
        blue_types = [
            UnitType.INFANTRY, UnitType.ARMOR, UnitType.ARTILLERY,
            UnitType.INFANTRY, UnitType.ENGINEER, UnitType.SIGNAL,
            UnitType.LOGISTICS, UnitType.INFANTRY,
        ]
        for i in range(n_blue):
            unit_type = blue_types[i % len(blue_types)]
            pos = Position(
                rng.uniform(0, map_size_km * 0.4),
                rng.uniform(0, map_size_km)
            )
            unit = Unit(
                unit_id=f"blue_{i}",
                unit_type=unit_type,
                alignment=ForceAlignment.BLUE,
                position=pos,
                capability=Capability.from_unit_type(unit_type),
                headcount=rng.randint(80, 200),
                morale=rng.uniform(0.6, 1.0),
                experience=rng.uniform(0.4, 0.9)
            )
            kg.add_unit(unit)

        # Red 유닛 생성 (우측)
        red_types = [
            UnitType.INFANTRY, UnitType.ARMOR, UnitType.ARTILLERY,
            UnitType.INFANTRY, UnitType.ELECTRONIC_WARFARE, UnitType.INFANTRY,
        ]
        for i in range(n_red):
            unit_type = red_types[i % len(red_types)]
            pos = Position(
                rng.uniform(map_size_km * 0.6, map_size_km),
                rng.uniform(0, map_size_km)
            )
            unit = Unit(
                unit_id=f"red_{i}",
                unit_type=unit_type,
                alignment=ForceAlignment.RED,
                position=pos,
                capability=Capability.from_unit_type(unit_type),
                headcount=rng.randint(60, 180),
                morale=rng.uniform(0.5, 0.9),
                experience=rng.uniform(0.3, 0.8)
            )
            kg.add_unit(unit)

        # 공간 관계 생성
        kg.build_spatial_relations()

        return kg

    @staticmethod
    def create_joint_scenario(
        map_size_km: float = 50.0,
        seed: Optional[int] = None
    ) -> CombatKnowledgeGraph:
        """
        합동 작전 시나리오 (육·해·공군·해병대·드론 혼합).

        Blue: 육군 대대 + 해군 구축함 + 공군 전투기 + 해병보병 + 공격 UAV
        Red:  육군 기갑 + 미사일 + 방공포병 + 잠수함 + 전자전
        """
        rng = np.random.RandomState(seed)
        kg = CombatKnowledgeGraph()

        # 지형
        terrain_types = list(TerrainType)
        grid_size = 6
        cell_size = map_size_km / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                terrain = terrain_types[rng.randint(len(terrain_types))]
                pos = Position(i * cell_size + cell_size / 2,
                               j * cell_size + cell_size / 2)
                kg.add_terrain(TerrainCell.from_type(f"cell_{i}_{j}", pos, terrain))

        # Blue 합동 전력
        blue_force = [
            (UnitType.MECHANIZED_INFANTRY, EchelonType.BATTALION, "1기보대대", 850),
            (UnitType.ARMOR,               EchelonType.BATTALION, "전차대대",  600),
            (UnitType.ARTILLERY,           EchelonType.BATTALION, "포병대대",  400),
            (UnitType.FIGHTER,             EchelonType.SQUADRON,  "11비행대대",  0),
            (UnitType.DESTROYER,           EchelonType.SHIP,      "DDH-971",     300),
            (UnitType.MARINE_INFANTRY,     EchelonType.BATTALION, "해병 1대대",  800),
            (UnitType.UAV_STRIKE,          EchelonType.FLIGHT,    "UAV편대-1",   0),
            (UnitType.SAM_BATTERY,         EchelonType.COMPANY,   "패트리어트 1포대", 120),
            (UnitType.RECONNAISSANCE,      EchelonType.COMPANY,   "정찰중대",   150),
        ]
        for i, (utype, echelon, desg, hc) in enumerate(blue_force):
            pos = Position(
                rng.uniform(0, map_size_km * 0.35),
                rng.uniform(0, map_size_km)
            )
            headcount = hc if hc > 0 else rng.randint(10, 50)
            unit = Unit(
                unit_id=f"blue_{i}",
                unit_type=utype,
                alignment=ForceAlignment.BLUE,
                position=pos,
                capability=Capability.from_unit_type(utype),
                headcount=headcount,
                echelon=echelon,
                designation=desg,
                morale=rng.uniform(0.70, 1.0),
                experience=rng.uniform(0.50, 0.95),
            )
            kg.add_unit(unit)

        # Red 합동 전력
        red_force = [
            (UnitType.ARMOR,              EchelonType.BRIGADE,  "적기갑여단",    3000),
            (UnitType.BALLISTIC_MISSILE,  EchelonType.REGIMENT, "미사일여단",       0),
            (UnitType.AIR_DEFENSE_ARTILLERY, EchelonType.BATTALION, "방공대대",    400),
            (UnitType.SUBMARINE,          EchelonType.SHIP,     "잠수함-R1",       80),
            (UnitType.ELECTRONIC_WARFARE, EchelonType.BATTALION, "전자전대대",    300),
            (UnitType.INFANTRY,           EchelonType.DIVISION, "적보병사단", 12000),
            (UnitType.MLRS,               EchelonType.BATTALION, "다련장대대",   350),
        ]
        for i, (utype, echelon, desg, hc) in enumerate(red_force):
            pos = Position(
                rng.uniform(map_size_km * 0.65, map_size_km),
                rng.uniform(0, map_size_km)
            )
            headcount = hc if hc > 0 else rng.randint(10, 50)
            unit = Unit(
                unit_id=f"red_{i}",
                unit_type=utype,
                alignment=ForceAlignment.RED,
                position=pos,
                capability=Capability.from_unit_type(utype),
                headcount=headcount,
                echelon=echelon,
                designation=desg,
                morale=rng.uniform(0.55, 0.90),
                experience=rng.uniform(0.35, 0.80),
            )
            kg.add_unit(unit)

        kg.build_spatial_relations()
        return kg


if __name__ == "__main__":
    print("=== Combat Ontology (Extended) Test ===")

    # 표준 시나리오
    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    stats = kg.get_stats()
    print(f"[표준 시나리오] Knowledge Graph 생성 완료")
    for k, v in stats.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

    node_feats = kg.get_node_features()
    edge_idx, edge_weights = kg.get_adjacency_info()
    print(f"   노드 특성 행렬: {node_feats.shape}  (dim={NODE_FEATURE_DIM})")
    print(f"   엣지 인덱스: {edge_idx.shape}, 엣지 가중치: {edge_weights.shape}")

    # 합동 시나리오
    kg2 = ScenarioFactory.create_joint_scenario(seed=42)
    stats2 = kg2.get_stats()
    print(f"\n[합동 시나리오] Knowledge Graph 생성 완료")
    for k, v in stats2.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

    # Unit 확장 필드 확인
    sample_unit = list(kg2.units.values())[0]
    print(f"\n[샘플 유닛] {sample_unit.designation} ({sample_unit.unit_type.value})")
    print(f"   군종: {sample_unit.branch.value}")
    print(f"   제대: {sample_unit.echelon.value}")
    print(f"   도메인: {sample_unit.domain.value}")
    print(f"   특성벡터: {sample_unit.to_feature_vector().shape}")

    print(f"\nUnitType 총 {len(list(UnitType))}종 정의됨")
    print("✅ 확장 온톨로지 테스트 완료!")
