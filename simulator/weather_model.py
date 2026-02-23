"""
weather_model.py
================
기후·환경 효과 모델 (Weather & Environment Effects)

전투 수행 능력에 미치는 기후·환경 수정자 계산:
  - 이동성 (mobility)        : 강설·폭우·진창 → 감소
  - 화력 (firepower)         : 시계 제한, 유도무기 성능 저하
  - 항공 운용 (air_ops)      : 기상 제한 → 출격 불가
  - 통신 (comms)             : 기상 노이즈 → 통신 저하
  - 센서 (sensor)            : 안개·강수 → 레이더·광학 저하
  - 사기 (morale)            : 극한 기상 → 피로도 증가

계절·시간대·기상 조건이 종합적으로 작용.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ontology.combat_schema import Unit, UnitType, ForceAlignment, CombatKnowledgeGraph


# ──────────────────────────────────────────────────────────────
# 열거형
# ──────────────────────────────────────────────────────────────

class WeatherCondition(Enum):
    """기상 조건"""
    CLEAR      = "clear"       # 맑음
    OVERCAST   = "overcast"    # 흐림
    RAIN       = "rain"        # 강우
    HEAVY_RAIN = "heavy_rain"  # 폭우
    SNOW       = "snow"        # 강설
    BLIZZARD   = "blizzard"    # 눈보라
    FOG        = "fog"         # 짙은 안개
    SANDSTORM  = "sandstorm"   # 모래폭풍
    TYPHOON    = "typhoon"     # 태풍 (극단적)


class Season(Enum):
    """계절"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class TimeOfDay(Enum):
    """시간대"""
    DAWN     = "dawn"      # 새벽 (04:00~06:00)
    DAYTIME  = "daytime"   # 주간 (06:00~18:00)
    DUSK     = "dusk"      # 황혼 (18:00~20:00)
    NIGHTTIME = "nighttime" # 야간 (20:00~04:00)


# ──────────────────────────────────────────────────────────────
# 데이터클래스
# ──────────────────────────────────────────────────────────────

@dataclass
class WeatherModifiers:
    """기상 효과 수정자 (1.0=정상, <1.0=저하, >1.0=향상)"""
    mobility:    float = 1.0    # 이동성 수정자
    firepower:   float = 1.0    # 화력 수정자
    air_ops:     float = 1.0    # 항공 운용 가능성 [0,1]
    comms:       float = 1.0    # 통신 품질 수정자
    sensor:      float = 1.0    # 센서 효율 수정자
    morale:      float = 1.0    # 사기 수정자
    visibility_km: float = 15.0 # 유효 가시거리 (km)

    def as_dict(self) -> Dict[str, float]:
        return {
            "mobility": self.mobility,
            "firepower": self.firepower,
            "air_ops": self.air_ops,
            "comms": self.comms,
            "sensor": self.sensor,
            "morale": self.morale,
            "visibility_km": self.visibility_km,
        }

    def to_vector(self) -> np.ndarray:
        """6D 수정자 벡터 (가시거리 제외, 정규화)"""
        return np.array([
            self.mobility,
            self.firepower,
            self.air_ops,
            self.comms,
            self.sensor,
            self.morale,
        ], dtype=np.float32)


@dataclass
class WeatherState:
    """현재 기상 상태"""
    condition:   WeatherCondition  = WeatherCondition.CLEAR
    season:      Season            = Season.SUMMER
    time_of_day: TimeOfDay         = TimeOfDay.DAYTIME
    temperature_c: float           = 15.0    # 온도 (°C)
    wind_speed_kph: float          = 10.0    # 풍속 (km/h)
    precipitation_mm_h: float      = 0.0     # 강수량 (mm/h)
    humidity_pct: float            = 60.0    # 습도 (%)
    visibility_km: float           = 15.0    # 가시거리 (km)
    sea_state: int                 = 2       # 해상 상태 (Beaufort)


# ──────────────────────────────────────────────────────────────
# 기상 조건별 기본 수정자 테이블
# ──────────────────────────────────────────────────────────────

_WEATHER_BASE: Dict[WeatherCondition, WeatherModifiers] = {
    WeatherCondition.CLEAR:      WeatherModifiers(1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 20.0),
    WeatherCondition.OVERCAST:   WeatherModifiers(0.95, 0.95, 0.85, 0.95, 0.90, 0.97, 15.0),
    WeatherCondition.RAIN:       WeatherModifiers(0.80, 0.85, 0.65, 0.85, 0.75, 0.90, 8.0),
    WeatherCondition.HEAVY_RAIN: WeatherModifiers(0.60, 0.70, 0.30, 0.70, 0.55, 0.80, 3.0),
    WeatherCondition.SNOW:       WeatherModifiers(0.55, 0.80, 0.50, 0.85, 0.70, 0.82, 6.0),
    WeatherCondition.BLIZZARD:   WeatherModifiers(0.25, 0.60, 0.05, 0.60, 0.40, 0.65, 0.5),
    WeatherCondition.FOG:        WeatherModifiers(0.75, 0.70, 0.20, 0.90, 0.30, 0.88, 0.5),
    WeatherCondition.SANDSTORM:  WeatherModifiers(0.40, 0.65, 0.10, 0.55, 0.35, 0.72, 0.3),
    WeatherCondition.TYPHOON:    WeatherModifiers(0.10, 0.40, 0.00, 0.40, 0.25, 0.55, 0.2),
}

_SEASON_MODIFIERS: Dict[Season, Dict[str, float]] = {
    Season.SPRING: {"mobility": 0.90, "morale": 1.00},   # 해빙기 진창
    Season.SUMMER: {"mobility": 1.00, "morale": 0.97},   # 고온 피로
    Season.AUTUMN: {"mobility": 0.97, "morale": 1.00},   # 최적
    Season.WINTER: {"mobility": 0.70, "morale": 0.90},   # 동계 이동 제한
}

_TIME_MODIFIERS: Dict[TimeOfDay, Dict[str, float]] = {
    TimeOfDay.DAWN:      {"sensor": 0.70, "morale": 0.92},  # 시계 불량
    TimeOfDay.DAYTIME:   {"sensor": 1.00, "morale": 1.00},
    TimeOfDay.DUSK:      {"sensor": 0.75, "morale": 0.95},
    TimeOfDay.NIGHTTIME: {"sensor": 0.45, "morale": 0.85,   # 야간작전 피로
                          "air_ops": 0.70},                  # 야간 항공 제한
}

# 유닛 유형별 기상 민감도 (1=정상, >1=더 민감)
_UNIT_WEATHER_SENSITIVITY: Dict[str, float] = {
    "infantry":         1.20,  # 도보 이동 → 기상 민감
    "armor":            0.80,  # 장갑 방호 → 기상 덜 민감
    "artillery":        0.90,
    "fighter":          1.50,  # 항공기 → 기상 매우 민감
    "strike_aircraft":  1.50,
    "bomber":           1.40,
    "isr_aircraft":     1.60,  # ISR기 → 기상 최민감
    "destroyer":        0.70,  # 함정 → 상대적 둔감 (해상 고유 기상)
    "submarine":        0.50,  # 잠수함 → 기상 거의 무관
    "marine_infantry":  1.10,
    "cyber_unit":       0.10,  # 사이버 → 기상 무관
    "space_asset":      0.05,  # 우주 → 기상 완전 무관
}


# ──────────────────────────────────────────────────────────────
# WeatherEffectModel
# ──────────────────────────────────────────────────────────────

class WeatherEffectModel:
    """
    기상 효과 계산 엔진.

    WeatherState → WeatherModifiers 계산 (기상 + 계절 + 시간대 복합).
    유닛 유형별 민감도 적용으로 현실적 기상 효과 모델링.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def compute_modifiers(self, state: WeatherState) -> WeatherModifiers:
        """
        WeatherState → WeatherModifiers 계산.
        기상 조건 × 계절 × 시간대 수정자를 곱셈으로 결합.
        """
        base = _WEATHER_BASE[state.condition]

        # 계절 보정
        season_m = _SEASON_MODIFIERS.get(state.season, {})
        # 시간대 보정
        time_m   = _TIME_MODIFIERS.get(state.time_of_day, {})

        mobility  = base.mobility  * season_m.get("mobility",  1.0) * time_m.get("mobility",  1.0)
        firepower = base.firepower
        air_ops   = base.air_ops   * time_m.get("air_ops",   1.0)
        comms     = base.comms
        sensor    = base.sensor    * time_m.get("sensor",    1.0)
        morale    = base.morale    * season_m.get("morale",   1.0) * time_m.get("morale",    1.0)

        # 극한 온도 보정 (-20°C 이하 또는 +40°C 이상)
        if state.temperature_c < -20:
            t_pen = abs(state.temperature_c + 20) / 40.0
            mobility  *= max(0.3, 1.0 - t_pen * 0.5)
            morale    *= max(0.6, 1.0 - t_pen * 0.2)
        elif state.temperature_c > 40:
            t_pen = (state.temperature_c - 40) / 20.0
            morale *= max(0.75, 1.0 - t_pen * 0.15)

        # 강풍 보정 (100 km/h 이상)
        if state.wind_speed_kph > 100:
            wind_pen = (state.wind_speed_kph - 100) / 100.0
            air_ops  *= max(0.0, 1.0 - wind_pen)
            mobility *= max(0.5, 1.0 - wind_pen * 0.3)

        return WeatherModifiers(
            mobility    = float(np.clip(mobility,  0.0, 1.0)),
            firepower   = float(np.clip(firepower, 0.0, 1.0)),
            air_ops     = float(np.clip(air_ops,   0.0, 1.0)),
            comms       = float(np.clip(comms,     0.0, 1.0)),
            sensor      = float(np.clip(sensor,    0.0, 1.0)),
            morale      = float(np.clip(morale,    0.0, 1.0)),
            visibility_km = float(max(0.1, base.visibility_km)),
        )

    def apply_to_unit(self, unit: Unit, modifiers: WeatherModifiers) -> Unit:
        """
        기상 수정자를 유닛에 적용한 복사본 반환.
        유닛 유형별 민감도를 반영.
        """
        import copy
        sensitivity = _UNIT_WEATHER_SENSITIVITY.get(unit.unit_type.value, 1.0)

        def blend(original: float, modifier: float) -> float:
            """원본값 × (1 + sensitivity × (modifier - 1))로 혼합"""
            adjusted = 1.0 + sensitivity * (modifier - 1.0)
            return float(np.clip(original * adjusted, 0.0, 1.0))

        modified = copy.deepcopy(unit)
        cap = modified.capability
        modified.capability = type(cap)(
            firepower    = blend(cap.firepower,  modifiers.firepower),
            mobility     = blend(cap.mobility,   modifiers.mobility),
            protection   = cap.protection,          # 방호력은 기상 무관
            range_km     = cap.range_km * float(np.clip(modifiers.sensor, 0.5, 1.0)),
            ammo_level   = cap.ammo_level,
            fuel_level   = cap.fuel_level,
            comms_quality= blend(cap.comms_quality, modifiers.comms),
            ew_resistance= cap.ew_resistance,
        )
        # 사기 반영
        modified.morale = float(np.clip(
            unit.morale * modifiers.morale, 0.0, 1.0
        ))
        return modified

    def apply_to_kg(
        self,
        kg: CombatKnowledgeGraph,
        state: WeatherState,
        alignment: Optional[ForceAlignment] = None,
    ) -> Tuple[CombatKnowledgeGraph, WeatherModifiers]:
        """
        kg 내 모든 (또는 특정 진영) 유닛에 기상 효과 적용.
        Returns: (수정된 kg 복사본, 수정자)
        """
        import copy
        mods    = self.compute_modifiers(state)
        new_kg  = copy.deepcopy(kg)
        for uid, unit in new_kg.units.items():
            if alignment is None or unit.alignment == alignment:
                new_kg.units[uid] = self.apply_to_unit(unit, mods)
        return new_kg, mods

    def sample_random_weather(
        self,
        season: Season = Season.SUMMER,
        theater: str = "korea",   # "korea" | "desert" | "arctic" | "maritime"
    ) -> WeatherState:
        """
        시즌·작전 지역 기반 랜덤 기상 조건 샘플링.
        학습 다양성을 위해 에피소드마다 호출.
        """
        # 지역별 기상 분포
        _THEATER_WEIGHTS: Dict[str, Dict[WeatherCondition, float]] = {
            "korea": {
                WeatherCondition.CLEAR:      0.40,
                WeatherCondition.OVERCAST:   0.25,
                WeatherCondition.RAIN:       0.15,
                WeatherCondition.HEAVY_RAIN: 0.05,
                WeatherCondition.SNOW:       0.10,
                WeatherCondition.FOG:        0.05,
            },
            "desert": {
                WeatherCondition.CLEAR:      0.55,
                WeatherCondition.OVERCAST:   0.20,
                WeatherCondition.SANDSTORM:  0.20,
                WeatherCondition.FOG:        0.05,
            },
            "arctic": {
                WeatherCondition.SNOW:       0.35,
                WeatherCondition.BLIZZARD:   0.20,
                WeatherCondition.OVERCAST:   0.25,
                WeatherCondition.CLEAR:      0.20,
            },
            "maritime": {
                WeatherCondition.CLEAR:      0.30,
                WeatherCondition.OVERCAST:   0.30,
                WeatherCondition.RAIN:       0.20,
                WeatherCondition.HEAVY_RAIN: 0.10,
                WeatherCondition.TYPHOON:    0.05,
                WeatherCondition.FOG:        0.05,
            },
        }

        weights = _THEATER_WEIGHTS.get(theater, _THEATER_WEIGHTS["korea"])
        conditions = list(weights.keys())
        probs = np.array([weights[c] for c in conditions])
        probs = probs / probs.sum()
        condition = self.rng.choice(conditions, p=probs)  # type: ignore[arg-type]

        # 계절별 온도 범위
        temp_ranges: Dict[Season, Tuple[float, float]] = {
            Season.SPRING: (5.0,  20.0),
            Season.SUMMER: (20.0, 35.0),
            Season.AUTUMN: (5.0,  20.0),
            Season.WINTER: (-20.0, 5.0),
        }
        t_min, t_max = temp_ranges.get(season, (10.0, 25.0))
        temperature = float(self.rng.uniform(t_min, t_max))

        # 가시거리
        visibility = float(_WEATHER_BASE[condition].visibility_km
                           * self.rng.uniform(0.7, 1.3))

        # 시간대 균등 샘플링
        time_of_day = self.rng.choice(list(TimeOfDay))  # type: ignore[arg-type]

        return WeatherState(
            condition       = condition,
            season          = season,
            time_of_day     = time_of_day,
            temperature_c   = temperature,
            wind_speed_kph  = float(self.rng.uniform(0, 60)),
            precipitation_mm_h = float(
                self.rng.uniform(0, 50) if condition in (
                    WeatherCondition.RAIN, WeatherCondition.HEAVY_RAIN,
                    WeatherCondition.SNOW, WeatherCondition.BLIZZARD
                ) else 0.0
            ),
            humidity_pct    = float(self.rng.uniform(40, 95)),
            visibility_km   = float(np.clip(visibility, 0.1, 50.0)),
            sea_state       = int(self.rng.randint(0, 7)),
        )


# ──────────────────────────────────────────────────────────────
# 자가 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory

    print("=== Weather Effect Model Test ===")
    model = WeatherEffectModel(seed=42)

    for cond in WeatherCondition:
        state = WeatherState(condition=cond, season=Season.WINTER, time_of_day=TimeOfDay.NIGHTTIME)
        mods  = model.compute_modifiers(state)
        print(f"  [{cond.name:12s}] "
              f"mob={mods.mobility:.2f} fire={mods.firepower:.2f} "
              f"air={mods.air_ops:.2f} sens={mods.sensor:.2f} "
              f"vis={mods.visibility_km:5.1f}km")

    print("\n=== Random Weather Sampling ===")
    for theater in ["korea", "desert", "arctic", "maritime"]:
        counts: Dict[str, int] = {}
        for _ in range(100):
            ws = model.sample_random_weather(Season.WINTER, theater)
            counts[ws.condition.value] = counts.get(ws.condition.value, 0) + 1
        print(f"  {theater:10s}: {dict(sorted(counts.items(), key=lambda x:-x[1]))}")

    print("\n=== Apply to KG ===")
    kg = ScenarioFactory.create_standard_scenario(seed=42)
    blizzard = WeatherState(condition=WeatherCondition.BLIZZARD, season=Season.WINTER,
                            time_of_day=TimeOfDay.NIGHTTIME, temperature_c=-25.0)
    _, mods = model.apply_to_kg(kg, blizzard)
    vec = mods.to_vector()
    print(f"  BLIZZARD+NIGHT+WINTER 수정자: {np.round(vec, 2)}")
