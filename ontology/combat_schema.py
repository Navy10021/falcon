"""
combat_schema.py
================
전장 온톨로지 스키마 정의
Unit, Terrain, Capability, Relationship 클래스 및 Knowledge Graph 빌더

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
    INFANTRY = "infantry"
    ARMOR = "armor"
    ARTILLERY = "artillery"
    AVIATION = "aviation"
    ENGINEER = "engineer"
    SIGNAL = "signal"
    LOGISTICS = "logistics"
    ELECTRONIC_WARFARE = "electronic_warfare"


class TerrainType(Enum):
    OPEN = "open"
    URBAN = "urban"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    RIVER = "river"
    COASTAL = "coastal"


class ForceAlignment(Enum):
    BLUE = "blue"   # 아군
    RED = "red"     # 적군
    NEUTRAL = "neutral"


class UnitStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"      # 30~70% 전투력
    CRITICAL = "critical"      # < 30% 전투력
    DESTROYED = "destroyed"


class RelationType(Enum):
    SUPPORTS = "supports"
    THREATENS = "threatens"
    ADJACENT_TO = "adjacent_to"
    CONTROLS = "controls"
    SUPPLIES = "supplies"
    BLOCKS = "blocks"


# ──────────────────────────────────────────────
# Core Data Classes
# ──────────────────────────────────────────────

@dataclass
class Capability:
    """유닛 능력치 스키마"""
    firepower: float          # 화력 지수 [0, 1]
    mobility: float           # 기동력 [0, 1]
    protection: float         # 방호력 [0, 1]
    range_km: float           # 유효 사거리 (km)
    ammo_level: float         # 탄약 수준 [0, 1]
    fuel_level: float         # 연료 수준 [0, 1]
    comms_quality: float      # 통신 품질 [0, 1]
    ew_resistance: float      # 전자전 저항력 [0, 1]

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.firepower, self.mobility, self.protection,
            self.range_km / 100.0,  # normalize
            self.ammo_level, self.fuel_level,
            self.comms_quality, self.ew_resistance
        ], dtype=np.float32)

    @classmethod
    def from_unit_type(cls, unit_type: UnitType, noise_std: float = 0.05) -> "Capability":
        """유닛 유형별 기본 능력치 생성 (약간의 노이즈 포함)"""
        base_caps = {
            UnitType.INFANTRY: (0.5, 0.4, 0.4, 2.0, 0.8, 0.9, 0.7, 0.5),
            UnitType.ARMOR:    (0.9, 0.6, 0.9, 3.0, 0.7, 0.6, 0.6, 0.4),
            UnitType.ARTILLERY:(0.95, 0.3, 0.3, 30.0, 0.7, 0.6, 0.7, 0.5),
            UnitType.AVIATION: (0.8, 0.95, 0.5, 50.0, 0.5, 0.4, 0.8, 0.6),
            UnitType.ENGINEER: (0.3, 0.5, 0.5, 1.0, 0.8, 0.7, 0.7, 0.5),
            UnitType.SIGNAL:   (0.1, 0.5, 0.3, 5.0, 0.9, 0.8, 0.99, 0.8),
            UnitType.LOGISTICS:(0.2, 0.6, 0.3, 1.0, 0.9, 0.9, 0.7, 0.4),
            UnitType.ELECTRONIC_WARFARE: (0.3, 0.4, 0.3, 20.0, 0.8, 0.7, 0.9, 0.9),
        }
        vals = base_caps[unit_type]
        noisy = [np.clip(v + np.random.normal(0, noise_std), 0, 1) for v in vals[:-1]]
        noisy.append(vals[-1])  # ew_resistance no noise
        return cls(
            firepower=noisy[0], mobility=noisy[1], protection=noisy[2],
            range_km=max(0.5, vals[3] + np.random.normal(0, 1.0)),
            ammo_level=noisy[4], fuel_level=noisy[5],
            comms_quality=noisy[6], ew_resistance=noisy[7]
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
    """전술 유닛 스키마"""
    unit_id: str
    unit_type: UnitType
    alignment: ForceAlignment
    position: Position
    capability: Capability
    headcount: int              # 병력 수
    status: UnitStatus = UnitStatus.ACTIVE
    morale: float = 0.8         # 사기 [0, 1]
    experience: float = 0.5     # 경험치 [0, 1]
    objective: Optional[str] = None
    attached_to: Optional[str] = None  # 상위 부대 ID

    @property
    def combat_power(self) -> float:
        """종합 전투력 지수 계산"""
        cap_vec = self.capability.to_vector()
        weights = np.array([0.25, 0.15, 0.20, 0.05, 0.10, 0.05, 0.10, 0.10])
        base = float(np.dot(cap_vec[:8], weights))
        morale_factor = 0.7 + 0.3 * self.morale
        exp_factor = 0.8 + 0.2 * self.experience
        status_factor = {
            UnitStatus.ACTIVE: 1.0,
            UnitStatus.DEGRADED: 0.55,
            UnitStatus.CRITICAL: 0.25,
            UnitStatus.DESTROYED: 0.0
        }[self.status]
        return base * morale_factor * exp_factor * status_factor

    def to_feature_vector(self) -> np.ndarray:
        """GNN 노드 특성 벡터 (28차원)"""
        unit_type_onehot = np.zeros(len(UnitType))
        unit_type_onehot[list(UnitType).index(self.unit_type)] = 1.0

        alignment_onehot = np.zeros(3)
        alignment_onehot[list(ForceAlignment).index(self.alignment)] = 1.0

        status_onehot = np.zeros(len(UnitStatus))
        status_onehot[list(UnitStatus).index(self.status)] = 1.0

        return np.concatenate([
            unit_type_onehot,           # 8
            alignment_onehot,           # 3
            status_onehot,              # 4
            self.position.to_vector(),  # 3
            self.capability.to_vector(),# 8
            [self.headcount / 1000.0,   # 1 (normalized)
             self.morale,               # 1
             self.experience,           # 1
             self.combat_power],        # 1
        ]).astype(np.float32)           # total: 30


@dataclass
class TerrainCell:
    """지형 셀 스키마"""
    cell_id: str
    position: Position
    terrain_type: TerrainType
    elevation: float = 0.0      # 고도 (m)
    cover_factor: float = 0.0   # 엄폐 계수 [0, 1]
    concealment: float = 0.0    # 은폐 계수 [0, 1]
    movement_cost: float = 1.0  # 기동 비용 배수

    @classmethod
    def from_type(cls, cell_id: str, pos: Position, terrain: TerrainType) -> "TerrainCell":
        terrain_params = {
            TerrainType.OPEN:     (0.0, 0.0, 0.05, 1.0),
            TerrainType.URBAN:    (0.0, 0.7, 0.6,  2.5),
            TerrainType.FOREST:   (50.0,0.4, 0.8,  1.8),
            TerrainType.MOUNTAIN: (500.,0.5, 0.6,  3.5),
            TerrainType.RIVER:    (0.0, 0.1, 0.1,  4.0),
            TerrainType.COASTAL:  (0.0, 0.2, 0.2,  1.5),
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
                # 인접 관계
                if dist <= support_radius_km:
                    if u1.alignment == u2.alignment:
                        self.add_relation(u1.unit_id, u2.unit_id,
                                          RelationType.SUPPORTS, weight=1 - dist/support_radius_km)
                # 위협 관계
                if dist <= u1.capability.range_km and u1.alignment != u2.alignment:
                    self.add_relation(u1.unit_id, u2.unit_id,
                                      RelationType.THREATENS, weight=1 - dist/threat_radius_km)

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
        node_idx = {n: i for i, n in enumerate(node_list)}

        edge_index = []
        edge_weights = []
        for src, dst, data in self.graph.edges(data=True):
            edge_index.append([node_idx[src], node_idx[dst]])
            edge_weights.append(data.get("weight", 1.0))

        if edge_index:
            return (np.array(edge_index, dtype=np.int64).T,
                    np.array(edge_weights, dtype=np.float32))
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    def get_node_features(self) -> np.ndarray:
        """전체 노드 특성 행렬 반환"""
        features = []
        for node_id in self.graph.nodes():
            feat = self.graph.nodes[node_id]["features"]
            # 패딩: 최대 차원에 맞춤 (Unit: 30, Terrain: 10)
            padded = np.zeros(32, dtype=np.float32)
            padded[:len(feat)] = feat
            features.append(padded)
        return np.stack(features) if features else np.zeros((0, 32))

    def get_stats(self) -> Dict:
        """그래프 통계"""
        blue_units = [u for u in self.units.values() if u.alignment == ForceAlignment.BLUE]
        red_units  = [u for u in self.units.values() if u.alignment == ForceAlignment.RED]
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "blue_units": len(blue_units),
            "red_units": len(red_units),
            "blue_headcount": sum(u.headcount for u in blue_units),
            "red_headcount": sum(u.headcount for u in red_units),
            "blue_combat_power": sum(u.combat_power for u in blue_units),
            "red_combat_power": sum(u.combat_power for u in red_units),
        }


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
        """표준 교전 시나리오 생성"""
        rng = np.random.RandomState(seed)

        kg = CombatKnowledgeGraph()

        # 지형 셀 생성 (격자)
        terrain_types = list(TerrainType)
        grid_size = 5
        cell_size = map_size_km / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                terrain = terrain_types[rng.randint(len(terrain_types))]
                pos = Position(i * cell_size + cell_size/2, j * cell_size + cell_size/2)
                cell = TerrainCell.from_type(f"cell_{i}_{j}", pos, terrain)
                kg.add_terrain(cell)

        # Blue 유닛 생성 (좌측)
        blue_types = [UnitType.INFANTRY, UnitType.ARMOR, UnitType.ARTILLERY,
                      UnitType.INFANTRY, UnitType.ENGINEER, UnitType.SIGNAL,
                      UnitType.LOGISTICS, UnitType.INFANTRY]
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
        red_types = [UnitType.INFANTRY, UnitType.ARMOR, UnitType.ARTILLERY,
                     UnitType.INFANTRY, UnitType.ELECTRONIC_WARFARE, UnitType.INFANTRY]
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


if __name__ == "__main__":
    print("=== Combat Ontology Test ===")
    kg = ScenarioFactory.create_standard_scenario(n_blue=8, n_red=6, seed=42)
    stats = kg.get_stats()
    print(f"✅ Knowledge Graph 생성 완료")
    for k, v in stats.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

    node_feats = kg.get_node_features()
    edge_idx, edge_weights = kg.get_adjacency_info()
    print(f"\n✅ 노드 특성 행렬: {node_feats.shape}")
    print(f"✅ 엣지 인덱스: {edge_idx.shape}, 엣지 가중치: {edge_weights.shape}")
