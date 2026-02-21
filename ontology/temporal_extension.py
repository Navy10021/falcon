"""
temporal_extension.py
=====================
시간 의존 상태 모델링 - 전장 상태의 시계열 추적
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
import numpy as np
from ontology.combat_schema import Unit, UnitStatus, ForceAlignment


@dataclass
class TemporalSnapshot:
    """특정 시점의 전장 스냅샷"""
    timestep: int
    unit_states: Dict[str, Dict]   # unit_id → {status, headcount, position, combat_power}
    total_blue_casualties: int
    total_red_casualties: int
    blue_force_size: int
    red_force_size: int
    events: List[str] = field(default_factory=list)


class TemporalStateTracker:
    """
    전장 상태 시계열 추적기
    - 최근 N개 스냅샷 유지 (슬라이딩 윈도우)
    - 트렌드 분석 (사상자 속도, 전투력 변화율)
    - PPO 상태 벡터에 시계열 정보 포함
    """

    def __init__(self, window_size: int = 10, max_history: int = 100):
        self.window_size = window_size
        self.max_history = max_history
        self.history: Deque[TemporalSnapshot] = deque(maxlen=max_history)
        self.current_timestep = 0

    def record_snapshot(self, units: Dict[str, Unit],
                         events: Optional[List[str]] = None) -> TemporalSnapshot:
        """현재 상태 스냅샷 기록"""
        unit_states = {}
        blue_casualties = 0
        red_casualties = 0
        blue_force = 0
        red_force = 0

        for uid, unit in units.items():
            unit_states[uid] = {
                "status": unit.status.value,
                "headcount": unit.headcount,
                "position": (unit.position.x, unit.position.y),
                "combat_power": unit.combat_power,
                "morale": unit.morale
            }
            if unit.alignment == ForceAlignment.BLUE:
                blue_force += unit.headcount
                if unit.status in [UnitStatus.CRITICAL, UnitStatus.DESTROYED]:
                    blue_casualties += max(0, 200 - unit.headcount)
            else:
                red_force += unit.headcount
                if unit.status in [UnitStatus.CRITICAL, UnitStatus.DESTROYED]:
                    red_casualties += max(0, 180 - unit.headcount)

        snapshot = TemporalSnapshot(
            timestep=self.current_timestep,
            unit_states=unit_states,
            total_blue_casualties=blue_casualties,
            total_red_casualties=red_casualties,
            blue_force_size=blue_force,
            red_force_size=red_force,
            events=events or []
        )
        self.history.append(snapshot)
        self.current_timestep += 1
        return snapshot

    def get_trend_features(self) -> np.ndarray:
        """
        시계열 트렌드 특성 벡터 반환 (PPO 상태에 포함)
        - 최근 window_size 스냅샷 기반
        - 사상자 속도, 전투력 변화율, 세력 비율 변화 등
        """
        if len(self.history) < 2:
            return np.zeros(8, dtype=np.float32)

        window = list(self.history)[-self.window_size:]
        if len(window) < 2:
            return np.zeros(8, dtype=np.float32)

        # 사상자 속도 (단위: 명/스텝)
        blue_cas_rate = (window[-1].total_blue_casualties - window[0].total_blue_casualties) / len(window)
        red_cas_rate  = (window[-1].total_red_casualties  - window[0].total_red_casualties)  / len(window)

        # 병력 변화율
        blue_force_change = (window[-1].blue_force_size - window[0].blue_force_size) / max(window[0].blue_force_size, 1)
        red_force_change  = (window[-1].red_force_size  - window[0].red_force_size)  / max(window[0].red_force_size, 1)

        # 세력 비율
        total_blue = max(window[-1].blue_force_size, 1)
        total_red  = max(window[-1].red_force_size, 1)
        force_ratio = total_blue / (total_blue + total_red)

        # 사상 교환비
        if red_cas_rate > 0:
            exchange_ratio = blue_cas_rate / red_cas_rate
        else:
            exchange_ratio = 0.0

        # 모멘텀 (최근 추세 가속도)
        if len(window) >= 3:
            mid = len(window) // 2
            early_rate = (window[mid].total_blue_casualties - window[0].total_blue_casualties) / max(mid, 1)
            late_rate  = (window[-1].total_blue_casualties - window[mid].total_blue_casualties) / max(len(window) - mid, 1)
            momentum = late_rate - early_rate
        else:
            momentum = 0.0

        return np.array([
            np.clip(blue_cas_rate / 50.0, -1, 1),   # normalized
            np.clip(red_cas_rate  / 50.0, -1, 1),
            np.clip(blue_force_change, -1, 1),
            np.clip(red_force_change,  -1, 1),
            force_ratio,
            np.clip(exchange_ratio / 3.0, 0, 1),
            np.clip(momentum / 20.0, -1, 1),
            self.current_timestep / 100.0  # 경과 시간 (normalized)
        ], dtype=np.float32)

    def get_recent_events_summary(self, n: int = 5) -> List[str]:
        """최근 N개 이벤트 반환"""
        events = []
        for snap in list(self.history)[-n:]:
            events.extend(snap.events)
        return events[-n:]

    def predict_trend(self, steps_ahead: int = 3) -> Dict[str, float]:
        """단순 선형 외삽으로 미래 상태 예측"""
        if len(self.history) < 3:
            return {"blue_casualties": 0, "red_casualties": 0, "blue_force": 0}

        recent = list(self.history)[-5:]
        # 선형 회귀 기울기
        x = np.arange(len(recent))
        blue_cas = np.array([s.total_blue_casualties for s in recent])
        red_cas  = np.array([s.total_red_casualties  for s in recent])
        blue_force = np.array([s.blue_force_size for s in recent])

        def linear_slope(y):
            return np.polyfit(x, y, 1)[0]

        slope_blue_cas  = linear_slope(blue_cas)
        slope_red_cas   = linear_slope(red_cas)
        slope_blue_force = linear_slope(blue_force)

        return {
            "blue_casualties": max(0, blue_cas[-1] + slope_blue_cas * steps_ahead),
            "red_casualties":  max(0, red_cas[-1]  + slope_red_cas  * steps_ahead),
            "blue_force":      max(0, blue_force[-1] + slope_blue_force * steps_ahead),
        }

    def reset(self):
        self.history.clear()
        self.current_timestep = 0


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    print("=== Temporal State Tracker Test ===")
    kg = ScenarioFactory.create_standard_scenario(seed=42)
    tracker = TemporalStateTracker()

    for t in range(5):
        snap = tracker.record_snapshot(kg.units, events=[f"시뮬레이션 스텝 {t}"])
        print(f"  t={t}: Blue={snap.blue_force_size}명, Red={snap.red_force_size}명")

    trend = tracker.get_trend_features()
    print(f"\n✅ 트렌드 특성 벡터: {trend}")
    pred = tracker.predict_trend(steps_ahead=3)
    print(f"✅ 3스텝 후 예측: {pred}")
