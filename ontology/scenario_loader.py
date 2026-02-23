"""
scenario_loader.py
==================
YAML 시나리오 설정 파일 로더 (Scenario Configuration Loader)

configs/scenarios/*.yaml 파일을 읽어 CombatKnowledgeGraph를 생성하거나,
학습 설정 딕셔너리를 반환한다.

사용 예시:
    loader = ScenarioLoader()
    kg, cfg = loader.load("configs/scenarios/korea_defense.yaml")

    # 모든 시나리오 로드
    scenarios = loader.load_all("configs/scenarios/")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from ontology.combat_schema import (
    CombatKnowledgeGraph, Unit, UnitType, UnitStatus,
    ForceAlignment, Position, Capability, BranchType, EchelonType,
    ScenarioFactory,
)


# ──────────────────────────────────────────────────────────────
# UnitType / EchelonType / BranchType 매핑 테이블
# ──────────────────────────────────────────────────────────────

_UNIT_TYPE_MAP: Dict[str, UnitType] = {
    u.value: u for u in UnitType
}
# 별칭 추가
_UNIT_TYPE_MAP.update({
    "anti_ship_missile": UnitType.CRUISE_MISSILE,  # YAML alias
    "surface_combatant": UnitType.DESTROYER,
})

_ECHELON_MAP: Dict[str, EchelonType] = {e.value: e for e in EchelonType}
_ECHELON_MAP.update({
    "wing":    EchelonType.BRIGADE,   # 공군 비행단 ≈ 여단급
    "ship":    EchelonType.COMPANY,   # 함정 ≈ 중대급
    "flight":  EchelonType.PLATOON,
    "element": EchelonType.PLATOON,
    "squadron": EchelonType.BATTALION,
    "corps":   EchelonType.REGIMENT,  # 군단급 (근사)
})

_BRANCH_FROM_TYPE: Dict[UnitType, BranchType] = {
    UnitType.DESTROYER:        BranchType.NAVY,
    UnitType.FRIGATE:          BranchType.NAVY,
    UnitType.SUBMARINE:        BranchType.NAVY,
    UnitType.MINE_WARFARE:     BranchType.NAVY,
    UnitType.AMPHIBIOUS_SHIP:  BranchType.NAVY,
    UnitType.COASTAL_DEFENSE:  BranchType.NAVY,
    UnitType.NAVAL_AVIATION:   BranchType.NAVY,
    UnitType.FIGHTER:          BranchType.AIR_FORCE,
    UnitType.STRIKE_AIRCRAFT:  BranchType.AIR_FORCE,
    UnitType.BOMBER:           BranchType.AIR_FORCE,
    UnitType.ISR_AIRCRAFT:     BranchType.AIR_FORCE,
    UnitType.EARLY_WARNING:    BranchType.AIR_FORCE,
    UnitType.AIR_REFUELING:    BranchType.AIR_FORCE,
    UnitType.SAM_BATTERY:      BranchType.AIR_FORCE,
    UnitType.TRANSPORT_AIRCRAFT: BranchType.AIR_FORCE,
    UnitType.MARINE_INFANTRY:  BranchType.MARINE_CORPS,
    UnitType.MARINE_ARMOR:     BranchType.MARINE_CORPS,
    UnitType.MARINE_AVIATION:  BranchType.MARINE_CORPS,
    UnitType.CYBER_UNIT:       BranchType.CYBER,
    UnitType.SPACE_ASSET:      BranchType.SPACE,
    UnitType.BALLISTIC_MISSILE: BranchType.STRATEGIC,
    UnitType.CRUISE_MISSILE:   BranchType.STRATEGIC,
}


# ──────────────────────────────────────────────────────────────
# ScenarioLoader
# ──────────────────────────────────────────────────────────────

class ScenarioLoader:
    """
    YAML 시나리오 설정 파일 → CombatKnowledgeGraph 변환 로더.

    YAML 구조:
      scenario.name / preset_fn
      force.blue.units[] / force.red.units[]
      terrain / environment / training
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        base_dir: YAML 파일의 기본 검색 디렉토리.
                  None이면 현재 작업 디렉토리 기준.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    # ── 공개 API ──────────────────────────────────────────────

    def load(
        self,
        yaml_path: str,
        seed: Optional[int] = None,
    ) -> Tuple[CombatKnowledgeGraph, Dict[str, Any]]:
        """
        YAML 파일 로드 → (CombatKnowledgeGraph, 학습 설정) 반환.

        yaml_path가 절대 경로가 아니면 base_dir 기준으로 해석.
        YAML 없거나 파싱 불가 시 → 표준 시나리오 폴백.
        """
        if not _YAML_AVAILABLE:
            return self._fallback_scenario(seed), {}

        path = Path(yaml_path)
        if not path.is_absolute():
            path = self.base_dir / yaml_path
        if not path.exists():
            return self._fallback_scenario(seed), {}

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            return self._fallback_scenario(seed), {}

        kg = self._build_kg(cfg, seed)
        training_cfg = cfg.get("training", {})
        return kg, training_cfg

    def load_all(
        self,
        scenario_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Tuple[CombatKnowledgeGraph, Dict]]:
        """
        디렉토리 내 모든 *.yaml 파일 로드.
        반환: {파일명(stem): (kg, training_cfg)}
        """
        if not _YAML_AVAILABLE:
            return {}

        d = Path(scenario_dir) if scenario_dir else self.base_dir / "configs" / "scenarios"
        if not d.exists():
            return {}

        result = {}
        for yaml_file in sorted(d.glob("*.yaml")):
            try:
                kg, cfg = self.load(str(yaml_file), seed=seed)
                result[yaml_file.stem] = (kg, cfg)
            except Exception:
                pass  # 개별 실패 무시

        return result

    def get_scenario_names(self, scenario_dir: Optional[str] = None) -> List[str]:
        """사용 가능한 시나리오 이름 목록"""
        if not _YAML_AVAILABLE:
            return []
        d = Path(scenario_dir) if scenario_dir else self.base_dir / "configs" / "scenarios"
        if not d.exists():
            return []
        return [p.stem for p in sorted(d.glob("*.yaml"))]

    # ── 내부: KG 생성 ────────────────────────────────────────

    def _build_kg(self, cfg: Dict, seed: Optional[int]) -> CombatKnowledgeGraph:
        """YAML cfg → CombatKnowledgeGraph"""
        kg = CombatKnowledgeGraph()

        rng = np.random.RandomState(seed if seed is not None else 42)
        force_cfg = cfg.get("force", {})

        for alignment_str, align_enum in [
            ("blue", ForceAlignment.BLUE),
            ("red",  ForceAlignment.RED),
        ]:
            side_cfg = force_cfg.get(alignment_str, {})
            units_cfg = side_cfg.get("units", [])

            for unit_spec in units_cfg:
                count = int(unit_spec.get("count", 1))
                for i in range(count):
                    unit = self._build_unit(
                        unit_spec, align_enum, rng,
                        suffix=f"_{i}" if count > 1 else ""
                    )
                    kg.add_unit(unit)

        # 지형 생성 (terrain 섹션 있으면 타입 비율 반영)
        terrain_cfg = cfg.get("terrain", {})
        map_size = float(terrain_cfg.get("map_size_km", 30.0))
        self._add_terrain(kg, terrain_cfg, map_size, rng)

        kg.build_spatial_relations()
        return kg

    def _build_unit(
        self,
        spec: Dict,
        alignment: ForceAlignment,
        rng: np.random.RandomState,
        suffix: str = "",
    ) -> Unit:
        """단일 유닛 YAML 명세 → Unit 객체"""
        from ontology.combat_schema import DomainType

        unit_type_str = spec.get("type", "infantry")
        unit_type = _UNIT_TYPE_MAP.get(unit_type_str, UnitType.INFANTRY)

        echelon_str = spec.get("echelon", "battalion")
        echelon = _ECHELON_MAP.get(echelon_str, EchelonType.BATTALION)

        branch = _BRANCH_FROM_TYPE.get(unit_type, BranchType.ARMY)

        headcount = int(spec.get("headcount", 500))
        morale    = float(spec.get("morale",    0.80))
        experience = float(spec.get("experience", 0.65))
        designation = spec.get("designation", "")
        if suffix:
            designation = f"{designation}{suffix}"

        # 유닛 ID 생성
        align_prefix = "blue" if alignment == ForceAlignment.BLUE else "red"
        unit_id = f"{align_prefix}_{unit_type_str}{suffix}_{rng.randint(1000, 9999)}"

        # 랜덤 위치 생성
        map_size = 30.0
        pos = Position(
            x=float(rng.uniform(0, map_size)),
            y=float(rng.uniform(0, map_size)),
        )

        # Capability 생성 (유닛 유형별 기본값)
        cap = self._default_capability(unit_type)

        return Unit(
            unit_id     = unit_id,
            unit_type   = unit_type,
            alignment   = alignment,
            position    = pos,
            capability  = cap,
            headcount   = headcount,
            status      = UnitStatus.ACTIVE,
            morale      = morale,
            experience  = experience,
            branch      = branch,
            echelon     = echelon,
            designation = designation,
        )

    def _default_capability(self, unit_type: UnitType) -> Capability:
        """유닛 유형별 기본 Capability"""
        defaults: Dict[str, Dict[str, float]] = {
            "infantry":          {"firepower":0.5, "mobility":0.4, "protection":0.3},
            "armor":             {"firepower":0.8, "mobility":0.7, "protection":0.8},
            "artillery":         {"firepower":0.9, "mobility":0.3, "protection":0.3, "range_km":30.0},
            "destroyer":         {"firepower":0.8, "mobility":0.8, "protection":0.6, "range_km":200.0},
            "submarine":         {"firepower":0.7, "mobility":0.6, "protection":0.5, "range_km":80.0},
            "fighter":           {"firepower":0.9, "mobility":0.95,"protection":0.4, "range_km":300.0},
            "marine_infantry":   {"firepower":0.6, "mobility":0.5, "protection":0.4},
            "cyber_unit":        {"firepower":0.0, "mobility":0.0, "protection":0.3,
                                  "cyber_offense":0.8, "cyber_defense":0.7},
            "ballistic_missile": {"firepower":1.0, "mobility":0.1, "protection":0.1, "range_km":800.0},
        }
        key = unit_type.value
        vals = defaults.get(key, {"firepower":0.5, "mobility":0.5, "protection":0.4})
        return Capability(
            firepower   = vals.get("firepower",   0.5),
            mobility    = vals.get("mobility",    0.5),
            protection  = vals.get("protection",  0.4),
            range_km    = vals.get("range_km",    15.0),
            ammo_level  = vals.get("ammo_level",  0.9),
            fuel_level  = vals.get("fuel_level",  0.9),
            comms_quality = vals.get("comms_quality", 0.8),
            ew_resistance = vals.get("ew_resistance", 0.5),
        )

    def _add_terrain(
        self,
        kg: CombatKnowledgeGraph,
        terrain_cfg: Dict,
        map_size: float,
        rng: np.random.RandomState,
    ) -> None:
        """지형 섹션 기반 테레인 셀 추가"""
        from ontology.combat_schema import TerrainCell, TerrainType

        terrain_types_cfg = terrain_cfg.get("types", [
            {"terrain": "open", "ratio": 1.0}
        ])
        # 비율 → 테레인 풀
        pool = []
        for entry in terrain_types_cfg:
            t_str = entry.get("terrain", "open")
            ratio = float(entry.get("ratio", 0.1))
            pool.extend([t_str] * max(1, int(ratio * 10)))

        n_cells = min(20, max(6, len(pool)))
        for i in range(n_cells):
            t_str = rng.choice(pool)
            try:
                t_type = TerrainType(t_str)
            except ValueError:
                t_type = TerrainType.OPEN

            cell = TerrainCell(
                cell_id      = f"cell_{i}",
                terrain_type = t_type,
                x_min        = float(i % 5 * (map_size / 5)),
                x_max        = float((i % 5 + 1) * (map_size / 5)),
                y_min        = float(i // 5 * (map_size / 4)),
                y_max        = float((i // 5 + 1) * (map_size / 4)),
            )
            kg.add_terrain(cell)

    def _fallback_scenario(self, seed: Optional[int]) -> CombatKnowledgeGraph:
        """YAML 사용 불가 시 표준 시나리오 반환"""
        return ScenarioFactory.create_standard_scenario(seed=seed or 42)


# ──────────────────────────────────────────────────────────────
# 편의 함수
# ──────────────────────────────────────────────────────────────

def load_scenario(
    yaml_path: str,
    seed: Optional[int] = None,
    base_dir: Optional[str] = None,
) -> Tuple[CombatKnowledgeGraph, Dict]:
    """단일 YAML 시나리오 로드 편의 함수"""
    loader = ScenarioLoader(base_dir=base_dir)
    return loader.load(yaml_path, seed=seed)


# ──────────────────────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Scenario Loader Test ===")
    loader = ScenarioLoader()
    scenarios = loader.load_all("configs/scenarios/", seed=42)
    print(f"로드된 시나리오: {len(scenarios)}개")

    for name, (kg, cfg) in scenarios.items():
        stats = kg.get_stats()
        episodes = cfg.get("episodes", "?")
        print(f"  [{name:25s}] "
              f"Blue={stats['blue_units']}유닛/{stats['blue_headcount']:,}명 | "
              f"Red={stats['red_units']}유닛/{stats['red_headcount']:,}명 | "
              f"에피소드={episodes}")

    print("\n개별 시나리오 로드:")
    kg2, cfg2 = load_scenario("configs/scenarios/korea_defense.yaml", seed=7)
    print(f"  한반도 방어작전: {len(kg2.units)}개 유닛, "
          f"학습 에피소드={cfg2.get('episodes', '?')}")
