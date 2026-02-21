"""
constraint_parser.py
====================
Phase 3: 지휘관 제약 조건 파싱
- 자연어 입력 → 구조화된 제약 조건 변환
- Hard / Soft / Ethical / Resource Constraint 분류
"""

from __future__ import annotations
from typing import Dict, List, Optional
import re
from hitl.pareto_generator import CommanderConstraints


class ConstraintParser:
    """
    지휘관 제약 조건 파서
    딕셔너리 또는 자연어 입력 처리
    """

    def parse_dict(self, constraint_dict: Dict) -> CommanderConstraints:
        """딕셔너리 형태 제약 조건 파싱"""
        return CommanderConstraints(
            max_force_size=constraint_dict.get("max_force"),
            min_win_probability=constraint_dict.get("min_win_rate", 0.5),
            max_casualties=constraint_dict.get("max_casualties"),
            max_time_steps=constraint_dict.get("max_time", 50),
            forbidden_zones=constraint_dict.get("forbidden_zones", []),
            prefer_flanking=constraint_dict.get("prefer_flanking", False),
            prefer_artillery=constraint_dict.get("prefer_artillery", False),
            avoid_urban=constraint_dict.get("avoid_urban", False),
            roe_compliance=constraint_dict.get("roe", True),
            minimize_collateral=constraint_dict.get("minimize_collateral", True),
        )

    def parse_text(self, text: str) -> CommanderConstraints:
        """자연어 텍스트 제약 조건 파싱 (간단한 규칙 기반)"""
        constraints = CommanderConstraints()
        text_lower = text.lower()

        # 병력 제한 패턴
        force_match = re.search(r'(\d+)\s*명\s*이하', text)
        if force_match:
            constraints.max_force_size = int(force_match.group(1))

        # 승률 패턴
        win_match = re.search(r'승률\s*(\d+)\s*%\s*이상', text)
        if win_match:
            constraints.min_win_probability = int(win_match.group(1)) / 100

        # 사상자 패턴
        cas_match = re.search(r'사상자?\s*(\d+)\s*명?\s*이하', text)
        if cas_match:
            constraints.max_casualties = int(cas_match.group(1))

        # 선호도
        if '포병' in text or '포격' in text:
            constraints.prefer_artillery = True
        if '측면' in text or '우회' in text:
            constraints.prefer_flanking = True
        if '도심' in text_lower or 'urban' in text_lower:
            constraints.avoid_urban = True

        return constraints

    def validate(self, constraints: CommanderConstraints) -> List[str]:
        """제약 조건 유효성 검사"""
        warnings = []
        if constraints.min_win_probability > 0.95:
            warnings.append("요구 승률이 너무 높습니다 (>95%). 실현 불가능할 수 있습니다.")
        if constraints.max_force_size and constraints.max_force_size < 50:
            warnings.append("병력 제한이 너무 낮습니다 (<50명).")
        if constraints.max_casualties is not None and constraints.max_casualties == 0:
            warnings.append("사상자 0명 제약은 비현실적입니다.")
        return warnings


if __name__ == "__main__":
    print("=== Constraint Parser Test ===")
    parser = ConstraintParser()

    # 딕셔너리 파싱
    c1 = parser.parse_dict({
        "max_force": 700, "min_win_rate": 0.7,
        "max_casualties": 50, "prefer_artillery": True
    })
    print(f"✅ 딕셔너리 파싱: 최대병력={c1.max_force_size}, 최소승률={c1.min_win_probability}")

    # 텍스트 파싱
    text = "가용 병력 600명 이하로, 승률 70% 이상을 유지하고 포병 지원을 선호합니다."
    c2 = parser.parse_text(text)
    print(f"✅ 텍스트 파싱: 최대병력={c2.max_force_size}, 포병선호={c2.prefer_artillery}")

    warnings = parser.validate(c1)
    print(f"✅ 유효성 검사: {warnings if warnings else '이상 없음'}")
