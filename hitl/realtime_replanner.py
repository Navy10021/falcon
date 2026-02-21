"""
realtime_replanner.py
=====================
⑪ 실시간 재계획 HITL
- 전투 중 예측이 빗나갔을 때 Pareto 후보 실시간 재생성
- 트리거 조건: 승률 급락, 사상자 초과, 전력 임계치 돌파
- 긴급도에 따른 재계획 빈도 조절
- 지휘관 경보 + 대안 전략 즉시 제시
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class AlertLevel(Enum):
    NORMAL   = "normal"    # 계획대로 진행
    CAUTION  = "caution"   # 주의 (재계획 검토)
    WARNING  = "warning"   # 경고 (재계획 권장)
    CRITICAL = "critical"  # 위기 (즉시 재계획)


@dataclass
class BattleSnapshot:
    """전투 상태 스냅샷 (재계획 트리거 판단용)"""
    step: int
    blue_win_rate_estimate: float
    blue_headcount: int
    blue_headcount_initial: int
    blue_combat_power: float
    blue_combat_power_initial: float
    blue_avg_ammo: float
    blue_avg_fuel: float
    red_headcount: int
    mission_status: str

    @property
    def headcount_loss_ratio(self) -> float:
        return 1 - self.blue_headcount / max(self.blue_headcount_initial, 1)

    @property
    def combat_power_loss_ratio(self) -> float:
        return 1 - self.blue_combat_power / max(self.blue_combat_power_initial, 1e-8)

    @property
    def is_resource_critical(self) -> bool:
        return self.blue_avg_ammo < 0.15 or self.blue_avg_fuel < 0.15


@dataclass
class ReplanEvent:
    """재계획 이벤트 기록"""
    step: int
    alert_level: AlertLevel
    trigger_reason: str
    prev_strategy: str
    new_strategy: str
    win_rate_before: float
    win_rate_after_estimate: float
    accepted: bool = False   # 지휘관 채택 여부


class ReplanTrigger:
    """
    재계획 트리거 조건 판별기
    다중 조건 + 가중합으로 경보 수준 결정
    """

    def __init__(
        self,
        win_rate_drop_threshold: float = 0.15,
        headcount_loss_threshold: float = 0.30,
        power_loss_threshold: float = 0.40,
        cooldown_steps: int = 5
    ):
        self.win_rate_drop_threshold = win_rate_drop_threshold
        self.headcount_loss_threshold = headcount_loss_threshold
        self.power_loss_threshold = power_loss_threshold
        self.cooldown_steps = cooldown_steps
        self.last_replan_step = -999
        self.snapshot_history: List[BattleSnapshot] = []

    def evaluate(self, snap: BattleSnapshot) -> Tuple[AlertLevel, List[str]]:
        """현재 스냅샷 → 경보 수준 + 트리거 사유"""
        self.snapshot_history.append(snap)
        reasons = []
        alert_score = 0.0

        # 쿨다운 중이면 NORMAL
        if snap.step - self.last_replan_step < self.cooldown_steps:
            return AlertLevel.NORMAL, []

        # 승률 추세 하락
        if len(self.snapshot_history) >= 3:
            recent_wrs = [s.blue_win_rate_estimate for s in self.snapshot_history[-3:]]
            wr_drop = recent_wrs[0] - recent_wrs[-1]
            if wr_drop > self.win_rate_drop_threshold:
                reasons.append(f"승률 {wr_drop:.1%}p 급락")
                alert_score += wr_drop * 3

        # 병력 손실
        if snap.headcount_loss_ratio > self.headcount_loss_threshold:
            reasons.append(f"병력 {snap.headcount_loss_ratio:.1%} 손실")
            alert_score += snap.headcount_loss_ratio * 2

        # 전투력 손실
        if snap.combat_power_loss_ratio > self.power_loss_threshold:
            reasons.append(f"전투력 {snap.combat_power_loss_ratio:.1%} 손실")
            alert_score += snap.combat_power_loss_ratio * 1.5

        # 자원 고갈
        if snap.is_resource_critical:
            ammo_msg = f"탄약 {snap.blue_avg_ammo:.0%}" if snap.blue_avg_ammo < 0.15 else ""
            fuel_msg = f"연료 {snap.blue_avg_fuel:.0%}" if snap.blue_avg_fuel < 0.15 else ""
            msg = "/".join(filter(None, [ammo_msg, fuel_msg])) + " 위기"
            reasons.append(msg)
            alert_score += 1.0

        # 승리 가망 없음
        if snap.blue_win_rate_estimate < 0.25:
            reasons.append(f"승률 {snap.blue_win_rate_estimate:.1%} (임계치 미달)")
            alert_score += 2.0

        # 경보 수준 결정
        if alert_score >= 3.0:
            return AlertLevel.CRITICAL, reasons
        elif alert_score >= 1.5:
            return AlertLevel.WARNING, reasons
        elif alert_score >= 0.5:
            return AlertLevel.CAUTION, reasons
        return AlertLevel.NORMAL, []

    def mark_replan(self, step: int):
        self.last_replan_step = step


class RealtimeReplanner:
    """
    실시간 재계획 HITL 시스템
    - 트리거 감지 → 긴급 Pareto 재생성 → 지휘관 경보
    - 경보 수준에 따른 재계획 긴급도 조절
    - 수락/거부 피드백 수집
    """

    ALERT_COLORS = {
        AlertLevel.NORMAL:   "⬜",
        AlertLevel.CAUTION:  "🟡",
        AlertLevel.WARNING:  "🟠",
        AlertLevel.CRITICAL: "🔴",
    }

    REPLAN_STRATEGIES = {
        AlertLevel.CAUTION:  ["balanced", "minimum_force"],
        AlertLevel.WARNING:  ["minimum_casualty", "balanced", "withdraw_preserve"],
        AlertLevel.CRITICAL: ["withdraw_preserve", "minimum_casualty", "hold_position"],
    }

    def __init__(self, cooldown_steps: int = 5, seed: int = 0):
        self.trigger = ReplanTrigger(cooldown_steps=cooldown_steps)
        self.replan_history: List[ReplanEvent] = []
        self.current_strategy: str = "balanced"
        self.rng = np.random.RandomState(seed)
        self.step = 0

    def _generate_emergency_options(
        self,
        alert_level: AlertLevel,
        snap: BattleSnapshot
    ) -> List[Dict]:
        """경보 수준에 따른 긴급 전략 후보 생성"""
        strategy_types = self.REPLAN_STRATEGIES.get(alert_level, ["balanced"])
        options = []
        for st in strategy_types:
            # 상황에 맞는 수치 추정
            if st == "withdraw_preserve":
                win_prob = snap.blue_win_rate_estimate * 0.4 + 0.2
                force_size = int(snap.blue_headcount * 0.7)
                casualties = int(snap.blue_headcount * 0.05)
            elif st == "minimum_casualty":
                win_prob = snap.blue_win_rate_estimate * 0.7
                force_size = snap.blue_headcount
                casualties = int(snap.blue_headcount * 0.03)
            else:
                win_prob = snap.blue_win_rate_estimate * 0.9 + 0.1
                force_size = snap.blue_headcount
                casualties = int(snap.blue_headcount * 0.08)

            options.append({
                "strategy": st,
                "win_probability": float(np.clip(win_prob, 0.05, 0.95)),
                "force_size": force_size,
                "expected_casualties": casualties,
                "urgency": alert_level.value,
            })
        return options

    def process_step(
        self,
        kg,
        step_result: Dict,
        current_win_rate: float
    ) -> Optional[Dict]:
        """
        매 스텝 호출 → 재계획 필요 여부 판단
        Returns: 재계획 정보 (트리거 시) 또는 None
        """
        self.step += 1
        from ontology.combat_schema import ForceAlignment, UnitStatus

        blue_units  = [u for u in kg.units.values() if u.alignment == ForceAlignment.BLUE]
        blue_active = [u for u in blue_units if u.status != UnitStatus.DESTROYED]

        snap = BattleSnapshot(
            step=self.step,
            blue_win_rate_estimate=current_win_rate,
            blue_headcount=sum(u.headcount for u in blue_active),
            blue_headcount_initial=max(sum(u.headcount for u in blue_units), 1),
            blue_combat_power=sum(u.combat_power for u in blue_active),
            blue_combat_power_initial=max(sum(u.combat_power for u in blue_units), 1e-8),
            blue_avg_ammo=float(np.mean([u.capability.ammo_level for u in blue_active])) if blue_active else 0.0,
            blue_avg_fuel=float(np.mean([u.capability.fuel_level  for u in blue_active])) if blue_active else 0.0,
            red_headcount=sum(u.headcount for u in kg.units.values()
                              if u.alignment != ForceAlignment.BLUE
                              and u.status != UnitStatus.DESTROYED),
            mission_status=step_result.get("mission_status", "ongoing"),
        )

        alert_level, reasons = self.trigger.evaluate(snap)

        if alert_level == AlertLevel.NORMAL:
            return None

        # 재계획 트리거
        self.trigger.mark_replan(self.step)
        emergency_options = self._generate_emergency_options(alert_level, snap)
        recommended = emergency_options[0] if emergency_options else None

        event = ReplanEvent(
            step=self.step,
            alert_level=alert_level,
            trigger_reason=" | ".join(reasons),
            prev_strategy=self.current_strategy,
            new_strategy=recommended["strategy"] if recommended else "none",
            win_rate_before=current_win_rate,
            win_rate_after_estimate=recommended["win_probability"] if recommended else current_win_rate,
        )
        self.replan_history.append(event)

        replan_info = {
            "alert_level": alert_level.value,
            "alert_color": self.ALERT_COLORS[alert_level],
            "trigger_reasons": reasons,
            "emergency_options": emergency_options,
            "recommended": recommended,
            "event": event,
        }

        # 자동 채택 (CRITICAL 시)
        if alert_level == AlertLevel.CRITICAL and recommended:
            self.current_strategy = recommended["strategy"]
            event.accepted = True

        return replan_info

    def get_summary(self) -> Dict:
        n_total = len(self.replan_history)
        n_accepted = sum(1 for e in self.replan_history if e.accepted)
        level_counts = {}
        for e in self.replan_history:
            key = e.alert_level.value
            level_counts[key] = level_counts.get(key, 0) + 1
        return {
            "total_replans": n_total,
            "accepted": n_accepted,
            "adoption_rate": n_accepted / max(n_total, 1),
            "level_counts": level_counts,
        }


if __name__ == "__main__":
    from ontology.combat_schema import ScenarioFactory
    from simulator.mixed_lanchester import MixedLanchesterEngine
    from simulator.resource_manager import ResourceManager

    print("=" * 55)
    print("⑪ 실시간 재계획 HITL 테스트")
    print("=" * 55)

    kg = ScenarioFactory.create_standard_scenario(n_blue=6, n_red=6, seed=42)
    engine = MixedLanchesterEngine(seed=42)
    rm = ResourceManager(seed=42)
    replanner = RealtimeReplanner(cooldown_steps=3, seed=42)

    print("\n[20 스텝 실시간 재계획 감시]")
    win_rate = 0.70   # 초기 추정 승률

    for step in range(20):
        result = engine.run_step_mixed(kg, resource_manager=rm)
        # 승률 시뮬레이션 (적 사상자 비율 기반)
        red_hc = result.get("red_headcount", 100)
        blue_hc = result.get("blue_headcount", 100)
        win_rate = float(np.clip(blue_hc / max(blue_hc + red_hc, 1) + 0.1, 0, 1))
        win_rate *= (0.85 + np.random.normal(0, 0.05))   # 하락 추세

        replan_info = replanner.process_step(kg, result, win_rate)

        color = "⬜"
        if replan_info:
            color = replan_info["alert_color"]
            print(f"  스텝 {step+1:2d}: {color} [{replan_info['alert_level'].upper():8s}] "
                  f"승률={win_rate:.1%}  "
                  f"트리거: {' / '.join(replan_info['trigger_reasons'][:1])}")
            if replan_info.get("recommended"):
                print(f"          → 권장 전략: {replan_info['recommended']['strategy']}")
        else:
            print(f"  스텝 {step+1:2d}: {color} 정상  승률={win_rate:.1%}")

        if result["mission_status"] != "ongoing":
            break

    summary = replanner.get_summary()
    print(f"\n[재계획 요약]")
    print(f"  총 재계획: {summary['total_replans']}회")
    print(f"  채택률: {summary['adoption_rate']:.1%}")
    print(f"  경보 분포: {summary['level_counts']}")
    print("\n✅ 실시간 재계획 HITL 정상 동작!")
