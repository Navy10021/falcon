"""
military_units.py
=================
ONTOLOGY ROADMAP STEP 4: 현실 전력 규모 상수 / 팩토리

한국군(ROK) 및 북한군(KPA) 표준 제대별 병력·장비 규모 테이블.
실제 공개된 국방백서·IISS Military Balance 기준 근사값이며,
학술 시뮬레이션 목적의 합성 데이터입니다.

사용 예:
    from ontology.military_units import ROKForceTable, build_rok_corps
    kg = build_rok_corps(corps_id="III", seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, BranchType, EchelonType, DomainType,
    Capability, Position, ForceAlignment,
    CombatKnowledgeGraph, TerrainCell, TerrainType, ScenarioFactory,
)


# ──────────────────────────────────────────────────────────────
# 1. 제대별 표준 병력 규모 (한국군 기준)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EchelonSize:
    """제대별 표준 병력 규모"""
    name_kr:   str
    headcount: int           # 병력 수 (대략)
    echelon:   EchelonType


# 지상군 제대 표
ARMY_ECHELON_TABLE: Dict[str, EchelonSize] = {
    "squad":       EchelonSize("분대",    10,    EchelonType.SQUAD),
    "platoon":     EchelonSize("소대",    40,    EchelonType.PLATOON),
    "company":     EchelonSize("중대",   160,    EchelonType.COMPANY),
    "battalion":   EchelonSize("대대",   700,    EchelonType.BATTALION),
    "regiment":    EchelonSize("연대",  2500,    EchelonType.REGIMENT),
    "brigade":     EchelonSize("여단",  4000,    EchelonType.BRIGADE),
    "division":    EchelonSize("사단", 14000,    EchelonType.DIVISION),
    "corps":       EchelonSize("군단", 50000,    EchelonType.CORPS),
    "field_army":  EchelonSize("야전군", 150000, EchelonType.FIELD_ARMY),
}

# 해군 함정 표
NAVY_SHIP_TABLE: Dict[str, EchelonSize] = {
    "destroyer":       EchelonSize("구축함 (DDH/DDG)", 300, EchelonType.SHIP),
    "frigate":         EchelonSize("호위함 (FF/FFG)",  180, EchelonType.SHIP),
    "submarine_209":   EchelonSize("잠수함 209급",      35, EchelonType.SHIP),
    "submarine_214":   EchelonSize("잠수함 214급 (KSS-III)", 50, EchelonType.SHIP),
    "lph":             EchelonSize("독도함 (LPH)",     330, EchelonType.SHIP),
    "lsl":             EchelonSize("상륙함 (LST)",     150, EchelonType.SHIP),
    "pcg":             EchelonSize("초계함 (PKG)",      40, EchelonType.SHIP),
    "mine_layer":      EchelonSize("기뢰부설함",        80, EchelonType.SHIP),
}

# 공군 비행단 표
AIR_FORCE_WING_TABLE: Dict[str, EchelonSize] = {
    "fighter_wing":     EchelonSize("전투비행단",    1200, EchelonType.WING),
    "transport_wing":   EchelonSize("수송비행단",     800, EchelonType.WING),
    "fighter_squadron": EchelonSize("비행대대 (전투)", 200, EchelonType.SQUADRON),
    "isr_squadron":     EchelonSize("비행대대 (ISR)",  150, EchelonType.SQUADRON),
    "sam_group":        EchelonSize("방공포병여단",   2000, EchelonType.BRIGADE),
}

# 해병대 제대 표
MARINE_ECHELON_TABLE: Dict[str, EchelonSize] = {
    "marine_division": EchelonSize("해병 사단",   9000, EchelonType.DIVISION),
    "marine_brigade":  EchelonSize("해병 여단",   3500, EchelonType.BRIGADE),
    "marine_battalion":EchelonSize("해병 대대",    800, EchelonType.BATTALION),
}


# ──────────────────────────────────────────────────────────────
# 2. 주요 무기 체계 식별자 (한국군 현역)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WeaponSystem:
    """무기 체계 메타데이터"""
    designation:   str          # 공식 명칭
    unit_type:     UnitType
    branch:        BranchType
    range_km:      float        # 유효 사거리 (km)
    headcount_ref: int          # 운용 인원 기준값
    notes:         str = ""


ROK_WEAPON_SYSTEMS: List[WeaponSystem] = [
    # 지상군
    WeaponSystem("K1A2 전차대대",      UnitType.ARMOR,    BranchType.ARMY,    3.5,   500, "105mm → 120mm 활강포"),
    WeaponSystem("K2 흑표 전차대대",   UnitType.ARMOR,    BranchType.ARMY,    4.0,   500, "120mm 활강포, 복합장갑"),
    WeaponSystem("K9 자주포대대",      UnitType.ARTILLERY,BranchType.ARMY,   40.0,   400, "155mm, 사거리 40km"),
    WeaponSystem("천무 다련장대대",    UnitType.MLRS,     BranchType.ARMY,   80.0,   350, "131mm/230mm 복합 MLRS"),
    WeaponSystem("현무-2C 미사일여단", UnitType.BALLISTIC_MISSILE, BranchType.STRATEGIC, 500.0, 200, "탄도미사일"),
    WeaponSystem("현무-3C 순항미사일", UnitType.CRUISE_MISSILE,    BranchType.STRATEGIC,1500.0, 100, "순항미사일"),
    WeaponSystem("패트리어트 PAC-3",   UnitType.SAM_BATTERY,       BranchType.AIR_FORCE, 70.0, 120, "종말단계 요격"),
    WeaponSystem("천궁-II 포대",       UnitType.SAM_BATTERY,       BranchType.AIR_FORCE, 40.0, 100, "탄도탄 요격 가능"),
    WeaponSystem("비궁 대전차대대",    UnitType.ANTI_ARMOR,BranchType.ARMY,    5.0,   300, "ATGM"),
    # 해군
    WeaponSystem("세종대왕급 DDG",     UnitType.DESTROYER,  BranchType.NAVY, 150.0, 300, "SPY-1D 이지스"),
    WeaponSystem("충무공이순신급 DDH", UnitType.DESTROYER,  BranchType.NAVY, 120.0, 280, "함대함/함대공"),
    WeaponSystem("인천급 FFG",         UnitType.FRIGATE,    BranchType.NAVY, 100.0, 180, "스텔스 설계"),
    WeaponSystem("장보고급 SS",        UnitType.SUBMARINE,  BranchType.NAVY, 200.0,  35, "209형 잠수함"),
    WeaponSystem("도산안창호급 SSN",   UnitType.SUBMARINE,  BranchType.NAVY, 500.0,  50, "3000t급 KSS-III"),
    WeaponSystem("독도함 LPH",         UnitType.AMPHIBIOUS_SHIP, BranchType.NAVY, 20.0, 330, "상륙강습함"),
    # 공군
    WeaponSystem("F-35A 전투비행단",   UnitType.FIGHTER,   BranchType.AIR_FORCE, 200.0, 200, "스텔스 전투기"),
    WeaponSystem("F-15K 전투비행단",   UnitType.STRIKE_AIRCRAFT, BranchType.AIR_FORCE, 400.0, 200, "슬램이글"),
    WeaponSystem("KF-21 보라매",       UnitType.FIGHTER,   BranchType.AIR_FORCE, 200.0, 180, "한국형 전투기"),
    WeaponSystem("E-737 피스아이",     UnitType.EARLY_WARNING, BranchType.AIR_FORCE, 350.0, 50, "공중조기경보"),
    WeaponSystem("백두 ELINT",         UnitType.ISR_AIRCRAFT,  BranchType.AIR_FORCE, 300.0, 30, "전자정보수집"),
    WeaponSystem("중고도 정찰 UAV",    UnitType.UAV_RECON, BranchType.AIR_FORCE, 200.0,   5, "RQ-4 블록30"),
    WeaponSystem("무인 공격기 (개발)",  UnitType.UAV_STRIKE,BranchType.AIR_FORCE, 200.0,   3, "개발 중"),
    # 해병대
    WeaponSystem("해병 1사단",         UnitType.MARINE_INFANTRY, BranchType.MARINE_CORPS,  3.0, 9000, "포항 주둔"),
    WeaponSystem("해병 2사단",         UnitType.MARINE_INFANTRY, BranchType.MARINE_CORPS,  3.0, 9000, "김포 주둔"),
    WeaponSystem("상륙돌격장갑차대대", UnitType.MARINE_ARMOR,    BranchType.MARINE_CORPS,  4.0,  500, "KAAV"),
]


# ──────────────────────────────────────────────────────────────
# 3. 북한군(KPA) 대응 전력 (교전 시뮬레이션 적군 기준)
# ──────────────────────────────────────────────────────────────

KPA_WEAPON_SYSTEMS: List[WeaponSystem] = [
    WeaponSystem("폭풍호 전차대대",    UnitType.ARMOR,   BranchType.ARMY, 3.0, 500, "T-62 계열"),
    WeaponSystem("170mm 자주포대대",   UnitType.ARTILLERY,BranchType.ARMY,50.0, 350, "장사정포"),
    WeaponSystem("방사포여단 (300mm)", UnitType.MLRS,    BranchType.ARMY,200.0,300, "장거리 방사포"),
    WeaponSystem("화성-15형 ICBM",     UnitType.BALLISTIC_MISSILE, BranchType.STRATEGIC,13000.0,50, "ICBM"),
    WeaponSystem("화성-12형 IRBM",     UnitType.BALLISTIC_MISSILE, BranchType.STRATEGIC,4500.0,50,  "IRBM"),
    WeaponSystem("KN-23 단거리탄도",   UnitType.BALLISTIC_MISSILE, BranchType.STRATEGIC, 600.0, 30, "SRBM"),
    WeaponSystem("특수작전군 여단",    UnitType.SPECIAL_FORCES, BranchType.ARMY, 5.0, 3000, "비대칭전"),
    WeaponSystem("사이버부대",         UnitType.CYBER_UNIT,      BranchType.CYBER, 10000.0, 200, "121국"),
    WeaponSystem("전파교란부대",       UnitType.ELECTRONIC_WARFARE, BranchType.CYBER,  30.0, 150, "GPS 교란"),
]


# ──────────────────────────────────────────────────────────────
# 4. Knowledge Graph 생성 헬퍼
# ──────────────────────────────────────────────────────────────

def _make_unit(
    unit_id: str,
    ws: WeaponSystem,
    alignment: ForceAlignment,
    pos: Position,
    echelon: EchelonType,
    headcount_override: Optional[int] = None,
    morale: float = 0.80,
    experience: float = 0.65,
    rng: Optional[np.random.RandomState] = None,
) -> Unit:
    """WeaponSystem → Unit 변환 헬퍼"""
    if rng is None:
        rng = np.random.RandomState()
    hc = headcount_override if headcount_override is not None else ws.headcount_ref
    cap = Capability.from_unit_type(ws.unit_type)
    # range_km는 WeaponSystem 테이블값 사용
    cap = Capability(
        firepower=cap.firepower, mobility=cap.mobility, protection=cap.protection,
        range_km=ws.range_km, ammo_level=cap.ammo_level, fuel_level=cap.fuel_level,
        comms_quality=cap.comms_quality, ew_resistance=cap.ew_resistance,
        anti_armor=cap.anti_armor, anti_air=cap.anti_air, anti_submarine=cap.anti_submarine,
        air_to_air=cap.air_to_air, stealth_factor=cap.stealth_factor,
        cyber_offense=cap.cyber_offense, cyber_defense=cap.cyber_defense,
        ew_offense=cap.ew_offense, isr_capability=cap.isr_capability,
        amphibious_capable=cap.amphibious_capable, air_assault_capable=cap.air_assault_capable,
    )
    return Unit(
        unit_id=unit_id,
        unit_type=ws.unit_type,
        alignment=alignment,
        position=pos,
        capability=cap,
        headcount=hc,
        branch=ws.branch,
        echelon=echelon,
        designation=ws.designation,
        morale=float(np.clip(morale + rng.normal(0, 0.05), 0.3, 1.0)),
        experience=float(np.clip(experience + rng.normal(0, 0.05), 0.2, 1.0)),
    )


def build_rok_corps(
    corps_id: str = "I",
    map_size_km: float = 80.0,
    seed: Optional[int] = None,
) -> CombatKnowledgeGraph:
    """
    한국군 1개 군단 규모 KG 생성.

    Blue: 2개 보병사단 + 기갑여단 + 포병여단 + 공군 지원 + 해군 함정
    Red:  KPA 2개 군단(압축) + 미사일 + 특수전

    학술 시뮬레이션용 합성 데이터.
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()

    # 지형 격자
    grid = 8
    cell_size = map_size_km / grid
    terrain_types = list(TerrainType)
    for i in range(grid):
        for j in range(grid):
            terrain = terrain_types[rng.randint(len(terrain_types))]
            pos = Position(i * cell_size + cell_size / 2,
                           j * cell_size + cell_size / 2)
            kg.add_terrain(TerrainCell.from_type(f"cell_{i}_{j}", pos, terrain))

    # ── Blue (ROK) ──────────────────────────────────────
    blue_units_spec: List[Tuple[WeaponSystem, EchelonType, int]] = [
        # (WeaponSystem, echelon, headcount)
        (WeaponSystem("보병사단-1", UnitType.INFANTRY, BranchType.ARMY, 2.0, 14000),
         EchelonType.DIVISION, 14000),
        (WeaponSystem("보병사단-2", UnitType.INFANTRY, BranchType.ARMY, 2.0, 14000),
         EchelonType.DIVISION, 14000),
        (WeaponSystem("기갑여단", UnitType.ARMOR, BranchType.ARMY, 3.5, 4000),
         EchelonType.BRIGADE, 4000),
        (WeaponSystem("포병여단", UnitType.ARTILLERY, BranchType.ARMY, 40.0, 3000),
         EchelonType.BRIGADE, 3000),
        (WeaponSystem("천무 다련장대대", UnitType.MLRS, BranchType.ARMY, 80.0, 350),
         EchelonType.BATTALION, 350),
        (WeaponSystem("정찰대대", UnitType.RECONNAISSANCE, BranchType.ARMY, 15.0, 600),
         EchelonType.BATTALION, 600),
        (WeaponSystem("방공포병여단", UnitType.AIR_DEFENSE_ARTILLERY, BranchType.AIR_FORCE, 50.0, 2000),
         EchelonType.BRIGADE, 2000),
        (WeaponSystem("F-35A 비행대대", UnitType.FIGHTER, BranchType.AIR_FORCE, 200.0, 200),
         EchelonType.SQUADRON, 200),
        (WeaponSystem("세종대왕급 DDG", UnitType.DESTROYER, BranchType.NAVY, 150.0, 300),
         EchelonType.SHIP, 300),
        (WeaponSystem("공격 UAV 편대", UnitType.UAV_STRIKE, BranchType.AIR_FORCE, 200.0, 10),
         EchelonType.FLIGHT, 10),
        (WeaponSystem("공병대대", UnitType.ENGINEER, BranchType.ARMY, 1.0, 700),
         EchelonType.BATTALION, 700),
        (WeaponSystem("군수지원대대", UnitType.LOGISTICS, BranchType.ARMY, 1.0, 600),
         EchelonType.BATTALION, 600),
    ]

    for i, (ws, echelon, hc) in enumerate(blue_units_spec):
        pos = Position(
            rng.uniform(0, map_size_km * 0.40),
            rng.uniform(0, map_size_km),
        )
        unit = _make_unit(
            f"blue_{corps_id}_{i}",
            ws=ws, alignment=ForceAlignment.BLUE,
            pos=pos, echelon=echelon, headcount_override=hc,
            morale=0.82, experience=0.68, rng=rng,
        )
        kg.add_unit(unit)

    # ── Red (KPA) ────────────────────────────────────────
    red_units_spec: List[Tuple[WeaponSystem, EchelonType, int]] = [
        (WeaponSystem("KPA 보병사단", UnitType.INFANTRY, BranchType.ARMY, 2.0, 12000),
         EchelonType.DIVISION, 12000),
        (WeaponSystem("KPA 기갑여단", UnitType.ARMOR, BranchType.ARMY, 3.0, 3500),
         EchelonType.BRIGADE, 3500),
        (WeaponSystem("170mm 자주포대대", UnitType.ARTILLERY, BranchType.ARMY, 50.0, 350),
         EchelonType.BATTALION, 350),
        (WeaponSystem("300mm 방사포여단", UnitType.MLRS, BranchType.ARMY, 200.0, 500),
         EchelonType.BRIGADE, 500),
        (WeaponSystem("KN-23 미사일여단", UnitType.BALLISTIC_MISSILE, BranchType.STRATEGIC, 600.0, 200),
         EchelonType.REGIMENT, 200),
        (WeaponSystem("특수작전여단", UnitType.SPECIAL_FORCES, BranchType.ARMY, 5.0, 3000),
         EchelonType.BRIGADE, 3000),
        (WeaponSystem("전파교란대대", UnitType.ELECTRONIC_WARFARE, BranchType.CYBER, 30.0, 300),
         EchelonType.BATTALION, 300),
        (WeaponSystem("방공미사일여단", UnitType.AIR_DEFENSE_ARTILLERY, BranchType.ARMY, 40.0, 1500),
         EchelonType.BRIGADE, 1500),
        (WeaponSystem("정찰 UAV 편대", UnitType.UAV_RECON, BranchType.AIR_FORCE, 200.0, 20),
         EchelonType.FLIGHT, 20),
    ]

    for i, (ws, echelon, hc) in enumerate(red_units_spec):
        pos = Position(
            rng.uniform(map_size_km * 0.60, map_size_km),
            rng.uniform(0, map_size_km),
        )
        unit = _make_unit(
            f"red_{i}",
            ws=ws, alignment=ForceAlignment.RED,
            pos=pos, echelon=echelon, headcount_override=hc,
            morale=0.72, experience=0.58, rng=rng,
        )
        kg.add_unit(unit)

    kg.build_spatial_relations()
    return kg


def build_amphibious_scenario(
    map_size_km: float = 60.0,
    seed: Optional[int] = None,
) -> CombatKnowledgeGraph:
    """
    해병대 상륙 작전 시나리오 KG.

    Blue: 해병 1사단 + 상륙함 + 구축함 + F-35B(해병항공) + UAV 정찰
    Red:  해안방어여단 + 방공포 + 기뢰전 + 대전차
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()

    grid = 6
    cell_size = map_size_km / grid
    terrain_types = [TerrainType.COASTAL, TerrainType.URBAN, TerrainType.OPEN,
                     TerrainType.FOREST, TerrainType.MOUNTAIN, TerrainType.RIVER]
    for i in range(grid):
        for j in range(grid):
            terrain = terrain_types[(i + j) % len(terrain_types)]
            pos = Position(i * cell_size + cell_size / 2,
                           j * cell_size + cell_size / 2)
            kg.add_terrain(TerrainCell.from_type(f"cell_{i}_{j}", pos, terrain))

    # Blue 상륙부대
    blue_force: List[Tuple] = [
        ("blue_lph",  UnitType.AMPHIBIOUS_SHIP,  BranchType.NAVY,         EchelonType.SHIP,     "독도함 LPH",           330),
        ("blue_ddg",  UnitType.DESTROYER,        BranchType.NAVY,         EchelonType.SHIP,     "세종대왕함 DDG",        300),
        ("blue_ff1",  UnitType.FRIGATE,           BranchType.NAVY,         EchelonType.SHIP,     "인천함 FFG",            180),
        ("blue_mi1",  UnitType.MARINE_INFANTRY,   BranchType.MARINE_CORPS, EchelonType.DIVISION, "해병 1사단",           9000),
        ("blue_ma1",  UnitType.MARINE_ARMOR,      BranchType.MARINE_CORPS, EchelonType.BATTALION,"KAAV 대대",             500),
        ("blue_mav",  UnitType.MARINE_AVIATION,   BranchType.MARINE_CORPS, EchelonType.SQUADRON, "F-35B 비행대대",        200),
        ("blue_uav",  UnitType.UAV_RECON,         BranchType.AIR_FORCE,    EchelonType.FLIGHT,   "정찰 UAV 편대",          10),
        ("blue_sam",  UnitType.SAM_BATTERY,       BranchType.AIR_FORCE,    EchelonType.COMPANY,  "함상 SAM",              120),
    ]

    for uid, utype, branch, echelon, desg, hc in blue_force:
        pos = Position(rng.uniform(0, map_size_km * 0.35),
                       rng.uniform(0, map_size_km))
        cap = Capability.from_unit_type(utype)
        unit = Unit(
            unit_id=uid, unit_type=utype, alignment=ForceAlignment.BLUE,
            position=pos, capability=cap, headcount=hc,
            branch=branch, echelon=echelon, designation=desg,
            morale=rng.uniform(0.75, 0.95),
            experience=rng.uniform(0.60, 0.90),
        )
        kg.add_unit(unit)

    # Red 해안방어전력
    red_force: List[Tuple] = [
        ("red_cd1", UnitType.COASTAL_DEFENSE,     BranchType.NAVY,    EchelonType.REGIMENT, "해안방어연대",  2000),
        ("red_aa1", UnitType.AIR_DEFENSE_ARTILLERY,BranchType.ARMY,   EchelonType.BRIGADE,  "방공포병여단",  2000),
        ("red_mw1", UnitType.MINE_WARFARE,         BranchType.NAVY,   EchelonType.SHIP,     "기뢰부설함",     80),
        ("red_at1", UnitType.ANTI_ARMOR,           BranchType.ARMY,   EchelonType.BATTALION,"대전차대대",    400),
        ("red_inf", UnitType.INFANTRY,             BranchType.ARMY,   EchelonType.DIVISION, "수비사단",    10000),
        ("red_art", UnitType.ARTILLERY,            BranchType.ARMY,   EchelonType.BATTALION,"해안포대대",    350),
        ("red_ew1", UnitType.ELECTRONIC_WARFARE,   BranchType.CYBER,  EchelonType.BATTALION,"전자전대대",    250),
    ]

    for uid, utype, branch, echelon, desg, hc in red_force:
        pos = Position(rng.uniform(map_size_km * 0.55, map_size_km),
                       rng.uniform(0, map_size_km))
        cap = Capability.from_unit_type(utype)
        unit = Unit(
            unit_id=uid, unit_type=utype, alignment=ForceAlignment.RED,
            position=pos, capability=cap, headcount=hc,
            branch=branch, echelon=echelon, designation=desg,
            morale=rng.uniform(0.60, 0.85),
            experience=rng.uniform(0.45, 0.75),
        )
        kg.add_unit(unit)

    kg.build_spatial_relations()
    return kg


# ──────────────────────────────────────────────────────────────
# 5. ROK Force Table (조회용 클래스)
# ──────────────────────────────────────────────────────────────

class ROKForceTable:
    """한국군 전력 테이블 조회 유틸리티"""

    @staticmethod
    def get_echelon_size(key: str) -> Optional[EchelonSize]:
        for table in [ARMY_ECHELON_TABLE, MARINE_ECHELON_TABLE,
                      AIR_FORCE_WING_TABLE, NAVY_SHIP_TABLE]:
            if key in table:
                return table[key]
        return None

    @staticmethod
    def list_weapon_systems(branch: Optional[BranchType] = None) -> List[WeaponSystem]:
        if branch is None:
            return ROK_WEAPON_SYSTEMS
        return [ws for ws in ROK_WEAPON_SYSTEMS if ws.branch == branch]

    @staticmethod
    def summary() -> str:
        lines = [
            "=== ROK Force Table 요약 ===",
            f"무기체계 등록: {len(ROK_WEAPON_SYSTEMS)}종",
            f"KPA 대응전력: {len(KPA_WEAPON_SYSTEMS)}종",
        ]
        for branch in BranchType:
            cnt = sum(1 for ws in ROK_WEAPON_SYSTEMS if ws.branch == branch)
            if cnt:
                lines.append(f"  [{branch.value}] {cnt}종")
        return "\n".join(lines)


if __name__ == "__main__":
    print(ROKForceTable.summary())

    print("\n[군단급 시나리오 생성]")
    kg = build_rok_corps(corps_id="I", seed=42)
    stats = kg.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n[상륙 작전 시나리오 생성]")
    kg2 = build_amphibious_scenario(seed=42)
    stats2 = kg2.get_stats()
    for k, v in stats2.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n✅ military_units.py 테스트 완료!")
