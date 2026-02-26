from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ROECheckResult:
    allowed: bool
    event_type: str
    rationale: str


class ROEConstraintEvaluator:
    """Minimal ROE checker used by reproducible demo pipeline."""

    def __init__(self, enabled: bool = True, risk_threshold: float = 0.45):
        self.enabled = enabled
        self.risk_threshold = risk_threshold

    def evaluate(self, action: str, civilian_risk: float, target_type: str = "mixed") -> ROECheckResult:
        if not self.enabled:
            return ROECheckResult(True, "ROE_DISABLED", "ROE 검사를 비활성화하여 행동을 허용했습니다.")

        if action == "assault" and target_type == "mixed" and civilian_risk >= self.risk_threshold:
            return ROECheckResult(
                False,
                "ROE_BLOCKED",
                f"민간 피해 위험도({civilian_risk:.2f})가 임계치({self.risk_threshold:.2f}) 이상이어서 assault를 차단했습니다.",
            )

        return ROECheckResult(True, "ROE_CLEAR", "ROE 위반 조건이 없어 행동을 허용했습니다.")
