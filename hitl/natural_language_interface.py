"""
natural_language_interface.py
==============================
H. 자연어 지휘 인터페이스
- 지휘관 자연어 명령 → 구조화된 전술 제약 변환
- 규칙 기반 파서 + 패턴 매칭 (LLM 연동 대비 설계)
- 의도 분류: 목표/지형/시간/병력/교전/보급
- 파싱 결과 지휘관 확인 루프
- 모호한 명령 → 명확화 질문 생성

학술 연구용 합성 데이터
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# ──────────────────────────────────────────────
# 명령 의도 분류
# ──────────────────────────────────────────────

class CommandIntent(Enum):
    SEIZE_OBJECTIVE   = "seize"        # 목표 점령
    HOLD_POSITION     = "hold"         # 진지 고수
    DESTROY_ENEMY     = "destroy"      # 적 격멸
    WITHDRAW          = "withdraw"     # 철수
    DELAY             = "delay"        # 지연
    RECON             = "recon"        # 정찰
    SUPPORT           = "support"      # 지원
    RESUPPLY          = "resupply"     # 보급
    UNKNOWN           = "unknown"      # 불명


class ConstraintType(Enum):
    HARD = "hard"   # 절대 조건 (반드시 준수)
    SOFT = "soft"   # 선호 조건 (가능하면 준수)


@dataclass
class ParsedConstraint:
    """파싱된 단일 제약 조건"""
    constraint_type: ConstraintType
    category: str          # "force", "time", "terrain", "casualty", "objective", "resource"
    key: str               # 제약 키
    value: Any             # 제약 값
    original_text: str     # 원문
    confidence: float      # 파싱 신뢰도 [0,1]
    needs_clarification: bool = False
    clarification_question: str = ""

    def to_dict(self) -> Dict:
        return {
            "type": self.constraint_type.value,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        t = "🔴 필수" if self.constraint_type == ConstraintType.HARD else "🟡 선호"
        return f"  {t} [{self.category}] {self.key}={self.value} (신뢰도 {self.confidence:.0%})"


@dataclass
class ParsedCommand:
    """전체 명령 파싱 결과"""
    original_text: str
    intent: CommandIntent
    constraints: List[ParsedConstraint]
    ambiguous_parts: List[str]
    clarification_questions: List[str]
    overall_confidence: float

    @property
    def hard_constraints(self) -> List[ParsedConstraint]:
        return [c for c in self.constraints if c.constraint_type == ConstraintType.HARD]

    @property
    def soft_constraints(self) -> List[ParsedConstraint]:
        return [c for c in self.constraints if c.constraint_type == ConstraintType.SOFT]

    @property
    def needs_clarification(self) -> bool:
        return len(self.clarification_questions) > 0

    def to_commander_constraints(self):
        """hitl.pareto_generator.CommanderConstraints 변환"""
        from hitl.pareto_generator import CommanderConstraints
        kwargs = {}
        for c in self.constraints:
            if c.key == "max_force_size":
                kwargs["max_force_size"] = int(c.value)
            elif c.key == "min_win_probability":
                kwargs["min_win_probability"] = float(c.value)
            elif c.key == "max_casualties":
                kwargs["max_casualties"] = int(c.value)
            elif c.key == "max_time_steps":
                kwargs["max_time_steps"] = int(c.value)
        return CommanderConstraints(**kwargs)

    def summary(self) -> str:
        lines = [
            f"📋 명령 파싱 결과",
            f"  원문: \"{self.original_text[:60]}{'...' if len(self.original_text)>60 else ''}\"",
            f"  의도: {self.intent.value} (전체 신뢰도: {self.overall_confidence:.0%})",
            f"  제약 조건: 필수 {len(self.hard_constraints)}개 / 선호 {len(self.soft_constraints)}개",
        ]
        for c in self.constraints:
            lines.append(str(c))
        if self.clarification_questions:
            lines.append(f"  ❓ 명확화 필요:")
            for q in self.clarification_questions:
                lines.append(f"    → {q}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 규칙 기반 파서
# ──────────────────────────────────────────────

class NLCommandParser:
    """
    자연어 명령 파서 (규칙 기반 + 패턴 매칭)

    처리 순서:
    1. 전처리 (구두점, 정규화)
    2. 의도 분류 (키워드 기반)
    3. 제약 추출 (패턴 매칭)
    4. 신뢰도 계산
    5. 명확화 질문 생성
    """

    # ── 의도 키워드 ──────────────────────────

    INTENT_KEYWORDS: Dict[CommandIntent, List[str]] = {
        CommandIntent.SEIZE_OBJECTIVE:
            ["점령", "확보", "탈취", "장악", "공격", "진격", "전진", "seize", "capture", "attack"],
        CommandIntent.HOLD_POSITION:
            ["고수", "방어", "유지", "사수", "진지", "방호", "hold", "defend", "maintain"],
        CommandIntent.DESTROY_ENEMY:
            ["격멸", "섬멸", "타격", "제압", "파괴", "destroy", "eliminate", "neutralize"],
        CommandIntent.WITHDRAW:
            ["철수", "후퇴", "퇴각", "이탈", "탈출", "withdraw", "retreat", "pull back"],
        CommandIntent.DELAY:
            ["지연", "저지", "차단", "견제", "delay", "block", "disrupt"],
        CommandIntent.RECON:
            ["정찰", "탐색", "수색", "감시", "recon", "scout", "observe"],
        CommandIntent.SUPPORT:
            ["지원", "증원", "엄호", "cover", "support", "reinforce"],
        CommandIntent.RESUPPLY:
            ["보급", "충전", "탄약", "연료", "resupply", "replenish"],
    }

    # ── 숫자/단위 패턴 ────────────────────────

    NUMBER_PATTERNS = {
        "force":    r"(?:병력|군인|인원|부대)\s*(\d+)\s*(?:명|개|대)",
        "casualty": r"(?:사상자|손실|피해|희생)\s*(?:을|를|은|는)?\s*(\d+)\s*명\s*(?:이내|이하|미만)",
        "time_hr":  r"(\d+)\s*시간\s*(?:이내|안에|내에|내로)",
        "time_min": r"(\d+)\s*분\s*(?:이내|안에|내에)",
        "time_step":r"(\d+)\s*스텝",
        "force_pct":r"병력의?\s*(\d+)\s*%",
        "win_rate": r"(?:승률|성공률)\s*(\d+)\s*%\s*(?:이상|이상으로)",
    }

    # ── 지형 키워드 ──────────────────────────

    TERRAIN_PATTERNS = {
        "avoid_urban":    ["시가지", "도시", "민간", "주거"],
        "prefer_forest":  ["숲", "산림", "수목"],
        "avoid_river":    ["강", "하천", "수계"],
        "prefer_high":    ["고지", "능선", "산"],
    }

    # ── 우선순위 키워드 ──────────────────────

    HARD_KEYWORDS = ["반드시", "절대", "필수", "무조건", "최우선", "must", "critical",
                     "이내", "이하", "미만", "이상", "이상으로", "안에", "내에"]
    SOFT_KEYWORDS = ["가능하면", "되도록", "되면", "선호", "권장", "if possible",
                     "preferably", "ideally", "최대한"]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self._compiled = {
            key: re.compile(pattern, re.UNICODE | re.IGNORECASE)
            for key, pattern in self.NUMBER_PATTERNS.items()
        }

    def parse(self, text: str) -> ParsedCommand:
        """메인 파싱 함수"""
        text_clean = self._preprocess(text)
        intent, intent_conf = self._classify_intent(text_clean)
        constraints = self._extract_constraints(text_clean)
        ambiguous, clarifications = self._find_ambiguities(text_clean, constraints)

        # 전체 신뢰도: 의도 신뢰도 × 평균 제약 신뢰도
        if constraints:
            avg_conf = sum(c.confidence for c in constraints) / len(constraints)
            overall_conf = (intent_conf + avg_conf) / 2
        else:
            overall_conf = intent_conf * 0.7

        return ParsedCommand(
            original_text=text,
            intent=intent,
            constraints=constraints,
            ambiguous_parts=ambiguous,
            clarification_questions=clarifications,
            overall_confidence=float(overall_conf),
        )

    def _preprocess(self, text: str) -> str:
        # 공백 정규화, 소문자화 (영문만)
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def _classify_intent(self, text: str) -> Tuple[CommandIntent, float]:
        """의도 분류 (키워드 점수 기반)"""
        scores: Dict[CommandIntent, float] = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in text)
            if score > 0:
                scores[intent] = score / len(keywords) + score * 0.1

        if not scores:
            return CommandIntent.UNKNOWN, 0.3

        best_intent = max(scores, key=scores.get)
        # 신뢰도: 최고점 / (최고점 + 0.5) → [0,1] 압축
        best_score = scores[best_intent]
        confidence = min(best_score / (best_score + 0.3), 0.95)
        return best_intent, float(confidence)

    def _determine_constraint_type(self, text: str, context: str) -> Tuple[ConstraintType, float]:
        """제약 유형(HARD/SOFT) 결정"""
        context_lower = context.lower()
        hard_score = sum(1 for kw in self.HARD_KEYWORDS if kw in context_lower)
        soft_score = sum(1 for kw in self.SOFT_KEYWORDS if kw in context_lower)

        if hard_score > soft_score:
            return ConstraintType.HARD, min(0.5 + hard_score * 0.15, 0.95)
        elif soft_score > 0:
            return ConstraintType.SOFT, min(0.5 + soft_score * 0.15, 0.90)
        # 기본: 숫자 있으면 HARD, 없으면 SOFT
        if re.search(r'\d+', context):
            return ConstraintType.HARD, 0.70
        return ConstraintType.SOFT, 0.55

    def _extract_constraints(self, text: str) -> List[ParsedConstraint]:
        """제약 조건 추출"""
        constraints = []
        sentences = re.split(r'[,，。.!?\n]', text)

        # ① 병력 수
        m = self._compiled["force"].search(text)
        if m:
            val = int(m.group(1))
            ctx = m.group(0)
            ctype, conf = self._determine_constraint_type(text, ctx + text[max(0,m.start()-20):m.end()+20])
            constraints.append(ParsedConstraint(
                constraint_type=ctype, category="force",
                key="max_force_size", value=val,
                original_text=ctx, confidence=conf * 0.9,
            ))

        # ② 병력 % (전체의 N%)
        m = self._compiled["force_pct"].search(text)
        if m:
            pct = int(m.group(1))
            ctx = m.group(0)
            ctype, conf = self._determine_constraint_type(text, ctx)
            constraints.append(ParsedConstraint(
                constraint_type=ctype, category="force",
                key="max_force_ratio", value=pct / 100,
                original_text=ctx, confidence=conf * 0.85,
            ))

        # ③ 사상자 제한
        m = self._compiled["casualty"].search(text)
        if m:
            val = int(m.group(1))
            ctx = m.group(0)
            ctype, conf = self._determine_constraint_type(text, ctx)
            constraints.append(ParsedConstraint(
                constraint_type=ConstraintType.HARD, category="casualty",
                key="max_casualties", value=val,
                original_text=ctx, confidence=conf,
            ))

        # ④ 시간 제한 (시간 단위)
        m = self._compiled["time_hr"].search(text)
        if m:
            hrs = int(m.group(1))
            steps = hrs * 6  # 1시간 ≈ 6스텝 가정
            ctx = m.group(0)
            ctype, conf = self._determine_constraint_type(text, ctx)
            constraints.append(ParsedConstraint(
                constraint_type=ctype, category="time",
                key="max_time_steps", value=steps,
                original_text=ctx, confidence=conf,
                needs_clarification=(hrs > 24),
                clarification_question=f"{hrs}시간이 매우 깁니다. 확인해주세요." if hrs > 24 else "",
            ))

        # ④-b 시간 제한 (스텝 단위)
        m = self._compiled["time_step"].search(text)
        if m:
            steps = int(m.group(1))
            ctx = m.group(0)
            ctype, conf = self._determine_constraint_type(text, ctx)
            constraints.append(ParsedConstraint(
                constraint_type=ctype, category="time",
                key="max_time_steps", value=steps,
                original_text=ctx, confidence=conf * 0.95,
            ))

        # ⑤ 승률 하한
        m = self._compiled["win_rate"].search(text)
        if m:
            pct = int(m.group(1))
            ctx = m.group(0)
            constraints.append(ParsedConstraint(
                constraint_type=ConstraintType.HARD, category="objective",
                key="min_win_probability", value=pct / 100,
                original_text=ctx, confidence=0.88,
            ))

        # ⑥ 지형 회피/선호
        for key, keywords in self.TERRAIN_PATTERNS.items():
            for sent in sentences:
                if any(kw in sent for kw in keywords):
                    avoid = "피해" in sent or "우회" in sent or "회피" in sent or "금지" in sent
                    prefer = "이용" in sent or "활용" in sent or "통해" in sent or "선호" in sent
                    if avoid:
                        ctype = ConstraintType.HARD if "반드시" in sent or "절대" in sent else ConstraintType.SOFT
                        constraints.append(ParsedConstraint(
                            constraint_type=ctype, category="terrain",
                            key=f"avoid_{key.split('_')[1]}", value=True,
                            original_text=sent.strip(),
                            confidence=0.80,
                        ))
                    elif prefer:
                        constraints.append(ParsedConstraint(
                            constraint_type=ConstraintType.SOFT, category="terrain",
                            key=f"prefer_{key.split('_')[1]}", value=True,
                            original_text=sent.strip(),
                            confidence=0.72,
                        ))

        # ⑦ 특정 키워드 기반 제약
        keyword_rules = [
            (["민간인", "비전투원", "민간 피해"], "casualty", "minimize_civilian", True, ConstraintType.HARD, 0.85),
            (["공중 지원", "항공 지원", "공중 엄호"], "resource", "require_air_support", True, ConstraintType.SOFT, 0.75),
            (["포병 지원", "화력 지원"], "resource", "require_artillery", True, ConstraintType.SOFT, 0.75),
            (["야간", "야간 작전"], "time", "night_operation", True, ConstraintType.SOFT, 0.70),
            (["보급", "탄약 확보"], "resource", "secure_supply_first", True, ConstraintType.SOFT, 0.72),
        ]
        for keywords, category, key, val, ctype, base_conf in keyword_rules:
            for sent in sentences:
                if any(kw in sent for kw in keywords):
                    exists = any(c.key == key for c in constraints)
                    if not exists:
                        constraints.append(ParsedConstraint(
                            constraint_type=ctype, category=category,
                            key=key, value=val,
                            original_text=sent.strip(),
                            confidence=base_conf,
                        ))
                    break

        return constraints

    def _find_ambiguities(
        self, text: str, constraints: List[ParsedConstraint]
    ) -> Tuple[List[str], List[str]]:
        """모호한 표현 탐지 + 명확화 질문 생성"""
        ambiguous = []
        questions = []

        # 저신뢰도 제약
        for c in constraints:
            if c.confidence < 0.65:
                ambiguous.append(c.original_text)
                questions.append(f"\"{c.original_text}\" — {c.category} 제약이 맞습니까?")
            if c.needs_clarification and c.clarification_question:
                questions.append(c.clarification_question)

        # 숫자 없는 모호한 표현
        vague_patterns = [
            (r"최대한\s*(?:많은|빨리|적은)", "구체적인 수치를 지정해 주세요."),
            (r"적당히|어느정도", "어느 정도 수준을 원하시는지 수치로 알려주세요."),
            (r"빠르게|신속히", "구체적인 시간 제한이 있으신가요?"),
            (r"충분한|충분히", "충분한 병력이 몇 명을 의미하는지 알려주세요."),
        ]
        for pattern, question in vague_patterns:
            if re.search(pattern, text):
                questions.append(question)

        # 상충 제약 감지
        force_constraints = [c for c in constraints if c.category == "force"]
        if len(force_constraints) > 1:
            questions.append("병력 관련 제약이 복수입니다. 가장 중요한 제약을 확인해주세요.")

        return ambiguous[:3], questions[:3]


# ──────────────────────────────────────────────
# 대화형 인터페이스 (비대화형 시뮬레이션)
# ──────────────────────────────────────────────

class CommandInterface:
    """
    자연어 지휘 인터페이스 (대화형)
    파싱 결과를 지휘관에게 확인받고 구조화된 제약으로 변환
    """

    def __init__(self):
        self.parser = NLCommandParser()
        self.history: List[Dict] = []

    def process(self, text: str, auto_confirm: bool = True) -> ParsedCommand:
        """
        명령 처리 메인 함수
        auto_confirm=True: 지휘관 확인 없이 자동 채택 (테스트용)
        """
        cmd = self.parser.parse(text)
        self.history.append({"input": text, "parsed": cmd})
        return cmd

    def format_confirmation(self, cmd: ParsedCommand) -> str:
        """지휘관 확인용 포맷"""
        lines = ["=" * 50, "📋 파싱 결과 확인 요청", "=" * 50]
        lines.append(cmd.summary())
        if not cmd.needs_clarification:
            lines.append("\n✅ 위 내용으로 진행하시겠습니까? (예/아니오)")
        else:
            lines.append("\n❓ 명확화 후 재확인이 필요합니다.")
        return "\n".join(lines)

    def batch_process(self, commands: List[str]) -> List[ParsedCommand]:
        """여러 명령 일괄 처리"""
        return [self.process(cmd) for cmd in commands]


# ──────────────────────────────────────────────
# 예제 명령 데이터셋
# ──────────────────────────────────────────────

EXAMPLE_COMMANDS = [
    "3번 고지를 반드시 2시간 이내에 점령하라. 사상자는 50명 이내로 제한한다.",
    "가능하면 시가지는 우회하고, 병력 700명 이내로 임무를 완료하라.",
    "적 기갑 부대를 격멸하되 공중 지원을 최대한 활용하고, 승률 70% 이상 보장하라.",
    "현 진지를 반드시 고수하라. 절대 후퇴하지 마라. 민간인 피해를 최소화하라.",
    "보급을 먼저 확보한 후 전진하라. 가능하면 숲을 이용해 은폐하라.",
    "신속히 철수하라. 병력의 80%를 보존하는 것이 최우선이다.",
    "포병 지원을 요청하고 30스텝 이내에 목표를 달성하라.",
]


if __name__ == "__main__":
    print("=" * 55)
    print("H. 자연어 지휘 인터페이스 테스트")
    print("=" * 55)

    interface = CommandInterface()

    for i, cmd_text in enumerate(EXAMPLE_COMMANDS):
        print(f"\n[명령 {i+1}] \"{cmd_text[:55]}...\"" if len(cmd_text) > 55
              else f"\n[명령 {i+1}] \"{cmd_text}\"")
        cmd = interface.process(cmd_text)
        print(f"  의도: {cmd.intent.value}  신뢰도: {cmd.overall_confidence:.0%}")
        print(f"  필수제약: {len(cmd.hard_constraints)}개  선호제약: {len(cmd.soft_constraints)}개")
        if cmd.constraints:
            best = max(cmd.constraints, key=lambda c: c.confidence)
            print(f"  핵심 제약: {best.key}={best.value}")
        if cmd.clarification_questions:
            print(f"  ❓ {cmd.clarification_questions[0]}")

    # 통합 변환 테스트
    test_cmd = interface.process(EXAMPLE_COMMANDS[0])
    cc = test_cmd.to_commander_constraints()
    print(f"\n[CommanderConstraints 변환]")
    print(f"  max_force_size: {cc.max_force_size}")
    print(f"  max_casualties: {cc.max_casualties}")

    print("\n✅ 자연어 지휘 인터페이스 정상 동작!")
