"""
scenario_presets.py
===================
ONTOLOGY ROADMAP STEP 7: 현실 시나리오 프리셋

시나리오:
  1. 한반도 방어작전 (Korea Defense)        — 사단~군단급 정규전
  2. 공중우세 쟁취 (Air Superiority Battle) — 공군 위주
  3. 시가전 (Urban Warfare)                 — 소대~중대급 근접전
  4. 다영역 경쟁 (Multi-Domain Contest)     — 5개 도메인 동시 운용
  5. 사이버·전자전 (Cyber-EW)               — 비운동성 전장

각 시나리오는:
  - CombatKnowledgeGraph 반환
  - 현실적 병력 규모 (군사 공개 데이터 기반 합성)
  - JointOperationsManager C2 구조 포함 (선택)

학술 연구용 합성 데이터
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ontology.combat_schema import (
    Unit, UnitType, BranchType, EchelonType, DomainType,
    Capability, Position, ForceAlignment, UnitStatus,
    CombatKnowledgeGraph, TerrainCell, TerrainType, ScenarioFactory,
)
from ontology.joint_operations import JointOperationsManager


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _unit(
    uid: str,
    utype: UnitType,
    alignment: ForceAlignment,
    x: float, y: float,
    hc: int,
    echelon: EchelonType = EchelonType.BATTALION,
    designation: str = "",
    morale: float = 0.80,
    experience: float = 0.65,
) -> Unit:
    cap = Capability.from_unit_type(utype)
    return Unit(
        unit_id=uid, unit_type=utype, alignment=alignment,
        position=Position(x, y), capability=cap, headcount=hc,
        echelon=echelon, designation=designation,
        morale=morale, experience=experience,
    )


def _grid_terrain(kg: CombatKnowledgeGraph, map_km: float,
                  grid: int, pattern: List[TerrainType],
                  rng: np.random.RandomState) -> None:
    cell_size = map_km / grid
    for i in range(grid):
        for j in range(grid):
            t = pattern[(i + j * 2) % len(pattern)] if pattern else \
                list(TerrainType)[rng.randint(len(list(TerrainType)))]
            pos = Position(i * cell_size + cell_size / 2, j * cell_size + cell_size / 2)
            kg.add_terrain(TerrainCell.from_type(f"cell_{i}_{j}", pos, t))


# ──────────────────────────────────────────────────────────────
# 시나리오 1: 한반도 방어작전
# ──────────────────────────────────────────────────────────────

def scenario_korea_defense(
    seed: Optional[int] = None,
    with_c2: bool = True,
) -> Tuple[CombatKnowledgeGraph, Optional[JointOperationsManager]]:
    """
    한반도 방어작전 시나리오 (군단급)

    Blue (ROK + 미군 지원):
      육군 2개 사단 + 기갑여단 + 포병여단 + 공군 비행단 + 해군 전단
    Red (KPA):
      전방군단 (기계화사단 + 보병사단 + 탄도미사일 + 특수전 + EW)

    지형: 산악·개활지·강 혼재 (한반도 북부 접경 지역)
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()
    MAP = 100.0  # 100×100 km

    terrain_pattern = [
        TerrainType.MOUNTAIN, TerrainType.OPEN, TerrainType.FOREST,
        TerrainType.RIVER,    TerrainType.MOUNTAIN, TerrainType.OPEN,
    ]
    _grid_terrain(kg, MAP, 8, terrain_pattern, rng)

    # ── Blue (ROK/US) ────────────────────────────────────────
    blue_force = [
        # (uid, UnitType, x_range, y, headcount, echelon, designation)
        ("b_div1",  UnitType.INFANTRY,        rng.uniform(5, 30), rng.uniform(10, 90), 14000, EchelonType.DIVISION, "수도기계화사단"),
        ("b_div2",  UnitType.MECHANIZED_INFANTRY, rng.uniform(5, 30), rng.uniform(10, 90), 13000, EchelonType.DIVISION, "제7기동사단"),
        ("b_armor", UnitType.ARMOR,            rng.uniform(5, 25), rng.uniform(10, 90), 4000,  EchelonType.BRIGADE,  "제1기갑여단"),
        ("b_arty",  UnitType.ARTILLERY,        rng.uniform(5, 20), rng.uniform(10, 90), 3000,  EchelonType.BRIGADE,  "포병여단"),
        ("b_mlrs",  UnitType.MLRS,             rng.uniform(5, 20), rng.uniform(10, 90), 350,   EchelonType.BATTALION,"천무대대"),
        ("b_sam",   UnitType.AIR_DEFENSE_ARTILLERY, rng.uniform(5, 25), rng.uniform(10, 90), 2000, EchelonType.BRIGADE, "방공포병여단"),
        ("b_f35",   UnitType.FIGHTER,          rng.uniform(2, 15), rng.uniform(10, 90), 200,   EchelonType.SQUADRON, "F-35A 11비행대대"),
        ("b_f15",   UnitType.STRIKE_AIRCRAFT,  rng.uniform(2, 15), rng.uniform(10, 90), 200,   EchelonType.SQUADRON, "F-15K 비행대대"),
        ("b_aew",   UnitType.EARLY_WARNING,    rng.uniform(2, 15), rng.uniform(10, 90), 50,    EchelonType.FLIGHT,   "E-737 피스아이"),
        ("b_ddg",   UnitType.DESTROYER,        rng.uniform(2, 10), rng.uniform(10, 90), 300,   EchelonType.SHIP,     "세종대왕급 DDG"),
        ("b_sub",   UnitType.SUBMARINE,        rng.uniform(2, 10), rng.uniform(10, 90), 50,    EchelonType.SHIP,     "214급 잠수함"),
        ("b_spec",  UnitType.SPECIAL_FORCES,   rng.uniform(5, 35), rng.uniform(10, 90), 800,   EchelonType.BRIGADE,  "특수전사령부"),
        ("b_recon", UnitType.RECONNAISSANCE,   rng.uniform(5, 40), rng.uniform(10, 90), 600,   EchelonType.BATTALION,"정찰대대"),
        ("b_ew",    UnitType.ELECTRONIC_WARFARE,rng.uniform(5, 25), rng.uniform(10, 90), 300,   EchelonType.BATTALION,"전자전대대"),
        ("b_uav",   UnitType.UAV_RECON,        rng.uniform(5, 30), rng.uniform(10, 90), 20,    EchelonType.FLIGHT,   "정찰UAV편대"),
        ("b_log",   UnitType.LOGISTICS,        rng.uniform(5, 20), rng.uniform(10, 90), 2000,  EchelonType.BRIGADE,  "군수지원여단"),
    ]

    for uid, utype, x, y, hc, echelon, desg in blue_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.BLUE, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.75, 0.95), experience=rng.uniform(0.60, 0.90)))

    # ── Red (KPA) ────────────────────────────────────────────
    red_force = [
        ("r_mdiv",  UnitType.MECHANIZED_INFANTRY, rng.uniform(65, 95), rng.uniform(10, 90), 12000, EchelonType.DIVISION, "KPA 기계화사단"),
        ("r_inf1",  UnitType.INFANTRY,        rng.uniform(65, 95), rng.uniform(10, 90), 11000, EchelonType.DIVISION, "KPA 보병사단"),
        ("r_armor", UnitType.ARMOR,            rng.uniform(65, 95), rng.uniform(10, 90), 3500,  EchelonType.BRIGADE,  "KPA 기갑여단"),
        ("r_arty",  UnitType.ARTILLERY,        rng.uniform(70, 95), rng.uniform(10, 90), 350,   EchelonType.BATTALION,"170mm 자주포대대"),
        ("r_mlrs",  UnitType.MLRS,             rng.uniform(70, 95), rng.uniform(10, 90), 500,   EchelonType.BRIGADE,  "300mm 방사포여단"),
        ("r_bm",    UnitType.BALLISTIC_MISSILE,rng.uniform(75, 95), rng.uniform(10, 90), 200,   EchelonType.REGIMENT, "KN-23 탄도미사일여단"),
        ("r_spec",  UnitType.SPECIAL_FORCES,   rng.uniform(60, 90), rng.uniform(10, 90), 3000,  EchelonType.BRIGADE,  "KPA 특수작전여단"),
        ("r_ew",    UnitType.ELECTRONIC_WARFARE,rng.uniform(70, 95), rng.uniform(10, 90), 300,   EchelonType.BATTALION,"GPS교란대대"),
        ("r_cyber", UnitType.CYBER_UNIT,       rng.uniform(75, 95), rng.uniform(10, 90), 200,   EchelonType.TEAM,     "121국"),
        ("r_sam",   UnitType.AIR_DEFENSE_ARTILLERY, rng.uniform(65, 95), rng.uniform(10, 90), 1500, EchelonType.BRIGADE, "KPA 방공여단"),
    ]

    for uid, utype, x, y, hc, echelon, desg in red_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.RED, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.55, 0.80), experience=rng.uniform(0.40, 0.70)))

    kg.build_spatial_relations(support_radius_km=15.0, threat_radius_km=30.0)

    mgr = None
    if with_c2:
        mgr = JointOperationsManager(seed=seed or 0)
        mgr.auto_build_from_kg(kg, ForceAlignment.BLUE)
        mgr.auto_build_from_kg(kg, ForceAlignment.RED)

    return kg, mgr


# ──────────────────────────────────────────────────────────────
# 시나리오 2: 공중우세 쟁취
# ──────────────────────────────────────────────────────────────

def scenario_air_superiority(
    seed: Optional[int] = None,
    with_c2: bool = True,
) -> Tuple[CombatKnowledgeGraph, Optional[JointOperationsManager]]:
    """
    공중우세 쟁취 시나리오

    Blue: 전투비행단(F-35·F-15K) + 조기경보 + 방공망(SAM·PAC-3)
    Red:  적 전투기 + 탄도미사일 + 방공체계

    지형: 주로 개활지 + 일부 산악 (공중전 위주)
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()
    MAP = 200.0   # 공역 200×200 km

    _grid_terrain(kg, MAP, 6,
                  [TerrainType.OPEN, TerrainType.MOUNTAIN, TerrainType.OPEN,
                   TerrainType.COASTAL, TerrainType.OPEN, TerrainType.FOREST], rng)

    blue_force = [
        ("b_f35a",  UnitType.FIGHTER,          rng.uniform(5, 50), rng.uniform(10, 190), 200, EchelonType.SQUADRON, "F-35A 비행대대"),
        ("b_f15k",  UnitType.STRIKE_AIRCRAFT,  rng.uniform(5, 50), rng.uniform(10, 190), 200, EchelonType.SQUADRON, "F-15K 비행대대"),
        ("b_kf21",  UnitType.FIGHTER,          rng.uniform(5, 50), rng.uniform(10, 190), 180, EchelonType.SQUADRON, "KF-21 비행대대"),
        ("b_aew",   UnitType.EARLY_WARNING,    rng.uniform(2, 30), rng.uniform(10, 190), 50,  EchelonType.FLIGHT,   "E-737 피스아이"),
        ("b_kc",    UnitType.AIR_REFUELING,    rng.uniform(2, 30), rng.uniform(10, 190), 30,  EchelonType.FLIGHT,   "KC-330 공중급유기"),
        ("b_pac3",  UnitType.SAM_BATTERY,      rng.uniform(5, 40), rng.uniform(10, 190), 120, EchelonType.COMPANY,  "패트리어트 PAC-3"),
        ("b_cheon", UnitType.AIR_DEFENSE_ARTILLERY, rng.uniform(5, 40), rng.uniform(10, 190), 100, EchelonType.COMPANY, "천궁-II 포대"),
        ("b_isr",   UnitType.ISR_AIRCRAFT,     rng.uniform(2, 30), rng.uniform(10, 190), 30,  EchelonType.FLIGHT,   "백두 ELINT"),
    ]

    for uid, utype, x, y, hc, echelon, desg in blue_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.BLUE, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.80, 0.95), experience=rng.uniform(0.70, 0.95)))

    red_force = [
        ("r_mig",   UnitType.FIGHTER,          rng.uniform(150, 195), rng.uniform(10, 190), 150, EchelonType.SQUADRON, "KPA MiG-29 비행대"),
        ("r_su",    UnitType.STRIKE_AIRCRAFT,  rng.uniform(150, 195), rng.uniform(10, 190), 100, EchelonType.SQUADRON, "KPA Su-25 공격기"),
        ("r_bm",    UnitType.BALLISTIC_MISSILE,rng.uniform(160, 195), rng.uniform(10, 190), 100, EchelonType.REGIMENT, "KN-23 미사일여단"),
        ("r_cm",    UnitType.CRUISE_MISSILE,   rng.uniform(160, 195), rng.uniform(10, 190), 50,  EchelonType.REGIMENT, "순항미사일여단"),
        ("r_sam",   UnitType.AIR_DEFENSE_ARTILLERY, rng.uniform(150, 195), rng.uniform(10, 190), 200, EchelonType.BRIGADE, "S-300 방공여단"),
        ("r_ew",    UnitType.ELECTRONIC_WARFARE,rng.uniform(155, 195), rng.uniform(10, 190), 150, EchelonType.BATTALION,"전파교란대대"),
        ("r_uav",   UnitType.UAV_RECON,        rng.uniform(150, 195), rng.uniform(10, 190), 20,  EchelonType.FLIGHT,  "정찰UAV편대"),
    ]

    for uid, utype, x, y, hc, echelon, desg in red_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.RED, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.55, 0.80), experience=rng.uniform(0.40, 0.70)))

    kg.build_spatial_relations(support_radius_km=30.0, threat_radius_km=80.0)

    mgr = None
    if with_c2:
        mgr = JointOperationsManager(seed=seed or 0)
        mgr.auto_build_from_kg(kg, ForceAlignment.BLUE)
        mgr.auto_build_from_kg(kg, ForceAlignment.RED)

    return kg, mgr


# ──────────────────────────────────────────────────────────────
# 시나리오 3: 시가전
# ──────────────────────────────────────────────────────────────

def scenario_urban_warfare(
    seed: Optional[int] = None,
    with_c2: bool = True,
) -> Tuple[CombatKnowledgeGraph, Optional[JointOperationsManager]]:
    """
    시가전 시나리오 (소대~대대급)

    Blue: 특전사 + 보병대대 + 공병 + 드론 정찰
    Red:  도시 방어 보병 + 저격팀 + 비정규전 세력

    지형: 도시 위주 + 일부 폐허
    특수: 부수피해 최소화 ROE 적용 권장
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()
    MAP = 10.0   # 10×10 km (시가지 단위)

    _grid_terrain(kg, MAP, 5,
                  [TerrainType.URBAN, TerrainType.URBAN, TerrainType.OPEN,
                   TerrainType.URBAN, TerrainType.FOREST], rng)

    blue_force = [
        ("b_spec",  UnitType.SPECIAL_FORCES,   rng.uniform(0.5, 3), rng.uniform(0.5, 9.5), 200, EchelonType.COMPANY,  "특전사 1중대"),
        ("b_inf1",  UnitType.INFANTRY,         rng.uniform(0.5, 3), rng.uniform(0.5, 9.5), 700, EchelonType.BATTALION,"1보병대대"),
        ("b_inf2",  UnitType.INFANTRY,         rng.uniform(0.5, 3), rng.uniform(0.5, 9.5), 700, EchelonType.BATTALION,"2보병대대"),
        ("b_eng",   UnitType.ENGINEER,         rng.uniform(0.5, 3), rng.uniform(0.5, 9.5), 400, EchelonType.BATTALION,"공병대대"),
        ("b_uav",   UnitType.UAV_RECON,        rng.uniform(0.5, 3), rng.uniform(0.5, 9.5), 10,  EchelonType.FLIGHT,   "드론정찰편대"),
        ("b_lm",    UnitType.LOITERING_MUNITION,rng.uniform(0.5, 3),rng.uniform(0.5, 9.5), 10,  EchelonType.FLIGHT,   "배회형탄약편대"),
        ("b_mp",    UnitType.MILITARY_POLICE,  rng.uniform(0.5, 3), rng.uniform(0.5, 9.5), 160, EchelonType.COMPANY,  "헌병중대"),
    ]

    for uid, utype, x, y, hc, echelon, desg in blue_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.BLUE, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.80, 0.95), experience=rng.uniform(0.70, 0.95)))

    red_force = [
        ("r_inf1",  UnitType.INFANTRY,         rng.uniform(7, 9.5), rng.uniform(0.5, 9.5), 800, EchelonType.BATTALION,"방어보병대대"),
        ("r_inf2",  UnitType.INFANTRY,         rng.uniform(7, 9.5), rng.uniform(0.5, 9.5), 600, EchelonType.BATTALION,"방어보병대대2"),
        ("r_spec",  UnitType.SPECIAL_FORCES,   rng.uniform(5, 9.5), rng.uniform(0.5, 9.5), 200, EchelonType.COMPANY,  "저격/비정규 세력"),
        ("r_at",    UnitType.ANTI_ARMOR,       rng.uniform(7, 9.5), rng.uniform(0.5, 9.5), 150, EchelonType.COMPANY,  "대전차 진지"),
        ("r_ew",    UnitType.ELECTRONIC_WARFARE,rng.uniform(7, 9.5), rng.uniform(0.5, 9.5),100, EchelonType.TEAM,     "GPS교란팀"),
        ("r_uav",   UnitType.UAV_RECON,        rng.uniform(6, 9.5), rng.uniform(0.5, 9.5), 5,   EchelonType.FLIGHT,   "상업드론 정찰"),
    ]

    for uid, utype, x, y, hc, echelon, desg in red_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.RED, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.65, 0.85), experience=rng.uniform(0.50, 0.80)))

    kg.build_spatial_relations(support_radius_km=1.5, threat_radius_km=3.0)

    mgr = None
    if with_c2:
        mgr = JointOperationsManager(seed=seed or 0)
        mgr.auto_build_from_kg(kg, ForceAlignment.BLUE)
        mgr.auto_build_from_kg(kg, ForceAlignment.RED)

    return kg, mgr


# ──────────────────────────────────────────────────────────────
# 시나리오 4: 다영역 경쟁
# ──────────────────────────────────────────────────────────────

def scenario_multidomain_contest(
    seed: Optional[int] = None,
    with_c2: bool = True,
) -> Tuple[CombatKnowledgeGraph, Optional[JointOperationsManager]]:
    """
    다영역 경쟁 시나리오 (지상+해상+공중+사이버+우주)

    Blue: 합동전력 (육해공+해병+사이버+우주자산)
    Red:  다영역 경쟁 전력 (미사일·사이버·EW 위주)

    특수: 5개 도메인 동시 운용, ISR/사이버 효과가 전투 결과에 영향
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()
    MAP = 150.0

    _grid_terrain(kg, MAP, 7,
                  [TerrainType.COASTAL, TerrainType.OPEN, TerrainType.MOUNTAIN,
                   TerrainType.FOREST, TerrainType.RIVER, TerrainType.URBAN, TerrainType.OPEN], rng)

    blue_force = [
        # 지상
        ("b_div1",  UnitType.MECHANIZED_INFANTRY, rng.uniform(5, 40), rng.uniform(5, 145), 13000, EchelonType.DIVISION, "기동사단"),
        ("b_arty",  UnitType.ARTILLERY,           rng.uniform(5, 35), rng.uniform(5, 145), 3000,  EchelonType.BRIGADE,  "포병여단"),
        ("b_sam",   UnitType.AIR_DEFENSE_ARTILLERY,rng.uniform(5, 35), rng.uniform(5, 145), 2000, EchelonType.BRIGADE,  "방공여단"),
        # 해군
        ("b_ddg",   UnitType.DESTROYER,           rng.uniform(2, 20), rng.uniform(5, 145), 300,   EchelonType.SHIP,     "이지스 DDG"),
        ("b_ff",    UnitType.FRIGATE,             rng.uniform(2, 20), rng.uniform(5, 145), 180,   EchelonType.SHIP,     "FFG 호위함"),
        ("b_sub",   UnitType.SUBMARINE,           rng.uniform(2, 20), rng.uniform(5, 145), 50,    EchelonType.SHIP,     "214급 잠수함"),
        ("b_lph",   UnitType.AMPHIBIOUS_SHIP,     rng.uniform(2, 20), rng.uniform(5, 145), 330,   EchelonType.SHIP,     "독도함"),
        # 공군
        ("b_f35",   UnitType.FIGHTER,             rng.uniform(2, 30), rng.uniform(5, 145), 200,   EchelonType.SQUADRON, "F-35A 비행대"),
        ("b_isr",   UnitType.ISR_AIRCRAFT,        rng.uniform(2, 25), rng.uniform(5, 145), 30,    EchelonType.FLIGHT,   "ISR 비행대"),
        ("b_aew",   UnitType.EARLY_WARNING,       rng.uniform(2, 25), rng.uniform(5, 145), 50,    EchelonType.FLIGHT,   "E-737"),
        # 해병대
        ("b_mi",    UnitType.MARINE_INFANTRY,     rng.uniform(5, 30), rng.uniform(5, 145), 9000,  EchelonType.DIVISION, "해병 1사단"),
        # 드론
        ("b_uavs",  UnitType.UAV_STRIKE,          rng.uniform(5, 40), rng.uniform(5, 145), 20,    EchelonType.FLIGHT,   "공격UAV편대"),
        ("b_lm",    UnitType.LOITERING_MUNITION,  rng.uniform(5, 40), rng.uniform(5, 145), 30,    EchelonType.FLIGHT,   "배회탄약편대"),
        # 사이버/우주
        ("b_cyber", UnitType.CYBER_UNIT,          rng.uniform(5, 35), rng.uniform(5, 145), 300,   EchelonType.TEAM,     "사이버사령부"),
        ("b_space", UnitType.SPACE_ASSET,         rng.uniform(5, 30), rng.uniform(5, 145), 50,    EchelonType.TEAM,     "우주자산"),
    ]

    for uid, utype, x, y, hc, echelon, desg in blue_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.BLUE, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.75, 0.95), experience=rng.uniform(0.60, 0.90)))

    red_force = [
        ("r_div",   UnitType.INFANTRY,            rng.uniform(110, 145), rng.uniform(5, 145), 12000, EchelonType.DIVISION,  "적 보병사단"),
        ("r_arm",   UnitType.ARMOR,               rng.uniform(110, 145), rng.uniform(5, 145), 3500,  EchelonType.BRIGADE,   "적 기갑여단"),
        ("r_bm1",   UnitType.BALLISTIC_MISSILE,   rng.uniform(115, 145), rng.uniform(5, 145), 200,   EchelonType.REGIMENT,  "탄도미사일여단"),
        ("r_cm",    UnitType.CRUISE_MISSILE,       rng.uniform(115, 145), rng.uniform(5, 145), 100,   EchelonType.REGIMENT,  "순항미사일여단"),
        ("r_sub",   UnitType.SUBMARINE,            rng.uniform(115, 145), rng.uniform(5, 145), 35,    EchelonType.SHIP,      "적 잠수함"),
        ("r_fighter",UnitType.FIGHTER,             rng.uniform(115, 145), rng.uniform(5, 145), 120,   EchelonType.SQUADRON,  "적 전투기 비행대"),
        ("r_ew",    UnitType.ELECTRONIC_WARFARE,   rng.uniform(110, 145), rng.uniform(5, 145), 300,   EchelonType.BATTALION, "EW여단"),
        ("r_cyber", UnitType.CYBER_UNIT,           rng.uniform(115, 145), rng.uniform(5, 145), 200,   EchelonType.TEAM,      "사이버부대"),
        ("r_uav",   UnitType.UAV_STRIKE,           rng.uniform(110, 145), rng.uniform(5, 145), 30,    EchelonType.FLIGHT,    "공격UAV"),
        ("r_sam",   UnitType.AIR_DEFENSE_ARTILLERY,rng.uniform(110, 145), rng.uniform(5, 145), 1500,  EchelonType.BRIGADE,   "방공여단"),
    ]

    for uid, utype, x, y, hc, echelon, desg in red_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.RED, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.55, 0.80), experience=rng.uniform(0.45, 0.75)))

    kg.build_spatial_relations(support_radius_km=20.0, threat_radius_km=50.0)

    mgr = None
    if with_c2:
        mgr = JointOperationsManager(seed=seed or 0)
        mgr.auto_build_from_kg(kg, ForceAlignment.BLUE)
        mgr.auto_build_from_kg(kg, ForceAlignment.RED)

    return kg, mgr


# ──────────────────────────────────────────────────────────────
# 시나리오 5: 사이버·전자전 중심
# ──────────────────────────────────────────────────────────────

def scenario_cyber_ew(
    seed: Optional[int] = None,
    with_c2: bool = True,
) -> Tuple[CombatKnowledgeGraph, Optional[JointOperationsManager]]:
    """
    사이버·전자전 중심 시나리오

    특수: C2 시스템 공격이 승리의 핵심
    Blue: 사이버사령부 + EW + ISR + 일반 전력
    Red:  사이버부대 + GPS교란 + 탄도미사일
    """
    rng = np.random.RandomState(seed)
    kg = CombatKnowledgeGraph()
    MAP = 80.0

    _grid_terrain(kg, MAP, 6,
                  [TerrainType.OPEN, TerrainType.URBAN, TerrainType.MOUNTAIN,
                   TerrainType.FOREST, TerrainType.COASTAL, TerrainType.OPEN], rng)

    blue_force = [
        ("b_cyber", UnitType.CYBER_UNIT,          rng.uniform(5, 25), rng.uniform(5, 75), 500,  EchelonType.TEAM,     "사이버사령부 주력"),
        ("b_ew",    UnitType.ELECTRONIC_WARFARE,   rng.uniform(5, 30), rng.uniform(5, 75), 400,  EchelonType.BATTALION,"전자전대대"),
        ("b_isr",   UnitType.ISR_AIRCRAFT,        rng.uniform(2, 20), rng.uniform(5, 75), 50,   EchelonType.FLIGHT,   "ISR편대"),
        ("b_space", UnitType.SPACE_ASSET,         rng.uniform(5, 25), rng.uniform(5, 75), 80,   EchelonType.TEAM,     "통신위성"),
        ("b_sig",   UnitType.SIGNAL,              rng.uniform(5, 30), rng.uniform(5, 75), 400,  EchelonType.BATTALION,"통신대대"),
        ("b_div",   UnitType.MECHANIZED_INFANTRY,  rng.uniform(5, 35), rng.uniform(5, 75), 13000,EchelonType.DIVISION, "기동사단"),
        ("b_sam",   UnitType.SAM_BATTERY,         rng.uniform(5, 30), rng.uniform(5, 75), 200,  EchelonType.COMPANY,  "방공포대"),
    ]

    for uid, utype, x, y, hc, echelon, desg in blue_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.BLUE, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.80, 0.95), experience=rng.uniform(0.70, 0.95)))

    red_force = [
        ("r_cyber", UnitType.CYBER_UNIT,           rng.uniform(55, 75), rng.uniform(5, 75), 400, EchelonType.TEAM,    "사이버전부대"),
        ("r_ew",    UnitType.ELECTRONIC_WARFARE,    rng.uniform(50, 75), rng.uniform(5, 75), 350, EchelonType.BATTALION,"EW/GPS교란대대"),
        ("r_bm",    UnitType.BALLISTIC_MISSILE,     rng.uniform(55, 75), rng.uniform(5, 75), 200, EchelonType.REGIMENT,"탄도미사일여단"),
        ("r_inf",   UnitType.INFANTRY,              rng.uniform(55, 75), rng.uniform(5, 75),10000, EchelonType.DIVISION,"방어사단"),
        ("r_sam",   UnitType.AIR_DEFENSE_ARTILLERY, rng.uniform(55, 75), rng.uniform(5, 75),1500, EchelonType.BRIGADE, "방공여단"),
        ("r_uav",   UnitType.UAV_RECON,             rng.uniform(50, 75), rng.uniform(5, 75), 30,  EchelonType.FLIGHT,  "정찰UAV"),
    ]

    for uid, utype, x, y, hc, echelon, desg in red_force:
        kg.add_unit(_unit(uid, utype, ForceAlignment.RED, x, y, hc, echelon, desg,
                          morale=rng.uniform(0.60, 0.80), experience=rng.uniform(0.50, 0.75)))

    kg.build_spatial_relations(support_radius_km=10.0, threat_radius_km=25.0)

    mgr = None
    if with_c2:
        mgr = JointOperationsManager(seed=seed or 0)
        mgr.auto_build_from_kg(kg, ForceAlignment.BLUE)
        mgr.auto_build_from_kg(kg, ForceAlignment.RED)

    return kg, mgr


# ──────────────────────────────────────────────────────────────
# 편의 함수
# ──────────────────────────────────────────────────────────────

SCENARIO_REGISTRY: Dict[str, callable] = {
    "korea_defense":      scenario_korea_defense,
    "air_superiority":    scenario_air_superiority,
    "urban_warfare":      scenario_urban_warfare,
    "multidomain_contest":scenario_multidomain_contest,
    "cyber_ew":           scenario_cyber_ew,
}


def load_scenario(
    name: str,
    seed: Optional[int] = None,
    with_c2: bool = True,
) -> Tuple[CombatKnowledgeGraph, Optional[JointOperationsManager]]:
    """이름으로 시나리오 로드"""
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"알 수 없는 시나리오: {name}. 가능: {list(SCENARIO_REGISTRY.keys())}")
    return SCENARIO_REGISTRY[name](seed=seed, with_c2=with_c2)


if __name__ == "__main__":
    print("=== STEP 7: 시나리오 프리셋 테스트 ===\n")

    for name, fn in SCENARIO_REGISTRY.items():
        kg, mgr = fn(seed=42)
        stats = kg.get_stats()
        c2_q = mgr.get_overall_c2_quality(ForceAlignment.BLUE, kg) if mgr else 1.0
        jf   = mgr.get_joint_fires_available(ForceAlignment.BLUE, kg) if mgr else 0.0
        nf   = kg.get_node_features()
        print(f"[{name}]")
        print(f"  Blue: {stats['blue_units']}유닛/{stats['blue_headcount']:,}명  "
              f"Red: {stats['red_units']}유닛/{stats['red_headcount']:,}명")
        print(f"  GNN 노드특성: {nf.shape}  C2품질: {c2_q:.2f}  합동화력: {jf:.2f}")
    print("\n✅ STEP 7 시나리오 프리셋 테스트 완료!")
