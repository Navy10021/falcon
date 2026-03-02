"""
composer.py
===========
Simulator Composer — 설정 기반 엔진 조합기

학습 루프에서 활성 엔진을 선택하면 자동으로 해당 엔진들을 조합하여
통합 `step()` 인터페이스를 제공한다.

사용 예시::

    config = ComposerConfig(
        combat_engine="lanchester",
        effects=["fog_of_war", "weather"],
        resource_engine="resource_manager",
    )
    composer = SimulatorComposer(config, seed=42)
    step_result = composer.step(kg, action_pairs=pairs)

학술 연구용 합성 데이터
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class ComposerConfig:
    """시뮬레이터 조합 설정"""
    # 전투 엔진 선택
    combat_engine: str = "lanchester"  # lanchester | mixed_lanchester | combat_dynamics
    # 이동 엔진 선택
    movement_engine: Optional[str] = None  # maneuver | None
    # 효과 엔진 (복수 선택 가능)
    effects: List[str] = field(default_factory=list)  # fog_of_war, weather, cyber
    # 자원 관리 엔진
    resource_engine: Optional[str] = None  # resource_manager | None
    # Fog of War 설정
    fog_level: str = "MODERATE"
    # 효과 적용 순서 (중요: weather → fog → cyber 순서 권장)
    effect_order: List[str] = field(default_factory=lambda: ["weather", "fog_of_war", "cyber"])


# ──────────────────────────────────────────────
# Engine Registry
# ──────────────────────────────────────────────

_COMBAT_ENGINES = {
    "lanchester": ("simulator.lanchester_engine", "LanchesterEngine"),
    "mixed_lanchester": ("simulator.mixed_lanchester", "MixedLanchesterEngine"),
    "combat_dynamics": ("simulator.combat_dynamics", "CombatDynamicsEngine"),
}

_MOVEMENT_ENGINES = {
    "maneuver": ("simulator.maneuver_engine", "ManeuverEngine"),
}

_EFFECT_ENGINES = {
    "fog_of_war": ("simulator.fog_of_war", "FogOfWarFilter"),
    "weather": ("simulator.weather_model", "WeatherModel"),
    "cyber": ("simulator.cyber_effects", "CyberEffectsEngine"),
}

_RESOURCE_ENGINES = {
    "resource_manager": ("simulator.resource_manager", "ResourceManager"),
}


def _lazy_import(module_path: str, class_name: str):
    """지연 로딩으로 엔진 클래스를 import"""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ──────────────────────────────────────────────
# Composed Step Result
# ──────────────────────────────────────────────

@dataclass
class ComposedStepResult:
    """통합 스텝 결과"""
    # 전투 결과 (원래 엔진의 결과 객체)
    combat_result: Any = None
    # 전투 결과에서 추출된 표준 필드
    mission_status: str = "ongoing"
    blue_total_casualties: int = 0
    red_total_casualties: int = 0
    # 효과 결과
    weather_effects: Optional[Dict] = None
    cyber_effects: Optional[Dict] = None
    fog_uncertainty: Optional[Dict[str, float]] = None
    # 자원 이벤트
    supply_events: Optional[List] = None
    # 메타데이터
    active_engines: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Simulator Composer
# ──────────────────────────────────────────────

class SimulatorComposer:
    """
    설정 기반 시뮬레이터 조합기.

    여러 엔진을 조합하여 단일 step() 인터페이스 제공.
    """

    def __init__(self, config: Optional[ComposerConfig] = None, seed: int = 0):
        self.config = config or ComposerConfig()
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 엔진 인스턴스 (지연 생성)
        self._combat = None
        self._movement = None
        self._effects: Dict[str, Any] = {}
        self._resource = None

        self._init_engines()

    def _init_engines(self):
        """설정에 따라 엔진 인스턴스 생성"""
        cfg = self.config

        # 전투 엔진
        if cfg.combat_engine in _COMBAT_ENGINES:
            mod_path, cls_name = _COMBAT_ENGINES[cfg.combat_engine]
            cls = _lazy_import(mod_path, cls_name)
            self._combat = cls(seed=self.seed)

        # 이동 엔진
        if cfg.movement_engine and cfg.movement_engine in _MOVEMENT_ENGINES:
            mod_path, cls_name = _MOVEMENT_ENGINES[cfg.movement_engine]
            cls = _lazy_import(mod_path, cls_name)
            self._movement = cls(seed=self.seed)

        # 효과 엔진
        for eff_name in cfg.effects:
            if eff_name in _EFFECT_ENGINES:
                mod_path, cls_name = _EFFECT_ENGINES[eff_name]
                cls = _lazy_import(mod_path, cls_name)
                if eff_name == "fog_of_war":
                    from simulator.fog_of_war import FogLevel
                    fog_level = getattr(FogLevel, cfg.fog_level.upper(), FogLevel.MODERATE)
                    self._effects[eff_name] = cls(fog_level, seed=self.seed)
                else:
                    self._effects[eff_name] = cls(seed=self.seed)

        # 자원 관리 엔진
        if cfg.resource_engine and cfg.resource_engine in _RESOURCE_ENGINES:
            mod_path, cls_name = _RESOURCE_ENGINES[cfg.resource_engine]
            cls = _lazy_import(mod_path, cls_name)
            self._resource = cls(seed=self.seed)

    @property
    def combat_engine(self):
        """전투 엔진 직접 접근 (하위 호환)"""
        return self._combat

    @property
    def active_engine_names(self) -> List[str]:
        """활성 엔진 이름 목록"""
        names = []
        if self._combat:
            names.append(self.config.combat_engine)
        if self._movement:
            names.append(self.config.movement_engine)
        names.extend(self._effects.keys())
        if self._resource:
            names.append(self.config.resource_engine)
        return names

    def observe(self, kg, alignment) -> tuple:
        """
        Fog of War 관측 필터 적용.

        Returns (observed_kg, uncertainty_map).
        Fog 필터가 없으면 (kg, None) 반환.
        """
        if "fog_of_war" in self._effects:
            return self._effects["fog_of_war"].observe(kg, alignment)
        return kg, None

    def step(
        self,
        kg,
        action_pairs=None,
        **kwargs,
    ) -> ComposedStepResult:
        """
        통합 시뮬레이션 스텝.

        1. 날씨 효과 적용 (있으면)
        2. 사이버 효과 적용 (있으면)
        3. 전투 엔진 실행
        4. 이동 엔진 실행 (있으면)
        5. 자원 관리 (있으면)
        """
        result = ComposedStepResult(active_engines=self.active_engine_names)

        # 1. 날씨 효과
        if "weather" in self._effects:
            try:
                weather = self._effects["weather"]
                weather_result = weather.apply_weather_effects(kg)
                result.weather_effects = weather_result
            except Exception:
                pass

        # 2. 사이버 효과
        if "cyber" in self._effects:
            try:
                cyber = self._effects["cyber"]
                cyber_result = cyber.apply_effects(kg)
                result.cyber_effects = cyber_result
            except Exception:
                pass

        # 3. 전투 엔진 실행
        if self._combat is not None:
            combat_kwargs = {}
            if action_pairs is not None:
                combat_kwargs["action_pairs"] = action_pairs
            # resource_manager 전달 (mixed_lanchester 호환)
            if self._resource is not None:
                combat_kwargs["resource_manager"] = self._resource

            combat_result = self._combat.run_step(kg, **combat_kwargs)
            result.combat_result = combat_result

            # 표준 필드 추출
            if hasattr(combat_result, "mission_status"):
                result.mission_status = combat_result.mission_status
            elif isinstance(combat_result, dict):
                result.mission_status = combat_result.get("mission_status", "ongoing")

            if hasattr(combat_result, "blue_total_casualties"):
                result.blue_total_casualties = combat_result.blue_total_casualties
            if hasattr(combat_result, "red_total_casualties"):
                result.red_total_casualties = combat_result.red_total_casualties

        # 4. 이동 엔진
        if self._movement is not None:
            try:
                self._movement.step(kg)
            except Exception:
                pass

        # 5. 자원 관리 (auto supply)
        if self._resource is not None:
            try:
                result.supply_events = self._resource.auto_supply_step(kg)
            except Exception:
                pass

        return result

    def get_summary(self) -> Dict:
        """현재 조합 상태 요약"""
        return {
            "combat_engine": self.config.combat_engine,
            "movement_engine": self.config.movement_engine,
            "effects": list(self._effects.keys()),
            "resource_engine": self.config.resource_engine,
            "active_engines": self.active_engine_names,
            "seed": self.seed,
        }

    @classmethod
    def from_yaml(cls, yaml_path: str, seed: int = 0) -> "SimulatorComposer":
        """YAML 설정 파일에서 Composer 생성"""
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        engines = raw.get("engines", {})
        config = ComposerConfig(
            combat_engine=engines.get("combat", "lanchester"),
            movement_engine=engines.get("movement"),
            effects=engines.get("effects", []),
            resource_engine=engines.get("resources"),
            fog_level=engines.get("fog_level", "MODERATE"),
        )
        return cls(config, seed=seed)

    @classmethod
    def default(cls, seed: int = 0) -> "SimulatorComposer":
        """기본 설정 (Lanchester only)"""
        return cls(ComposerConfig(), seed=seed)

    @classmethod
    def full(cls, seed: int = 0) -> "SimulatorComposer":
        """전체 엔진 활성화"""
        config = ComposerConfig(
            combat_engine="lanchester",
            movement_engine="maneuver",
            effects=["fog_of_war", "weather", "cyber"],
            resource_engine="resource_manager",
        )
        return cls(config, seed=seed)
