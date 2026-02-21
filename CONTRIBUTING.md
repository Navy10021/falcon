# 기여 가이드 (Contributing Guide)

## 개발 환경 설정

```bash
git clone https://github.com/your-username/ai-combat-optimization.git
cd ai-combat-optimization
pip install -e ".[full]"
```

## 테스트 실행

```bash
PYTHONPATH=. python tests/test_tier1.py
PYTHONPATH=. python tests/test_tier2.py
PYTHONPATH=. python tests/test_tier3.py
```

## 코드 스타일

- PEP 8 준수
- 한국어 주석 권장 (군사 용어)
- 모든 공개 함수에 docstring 작성

## Pull Request 가이드

1. feature/[기능명] 브랜치에서 작업
2. 기존 테스트 전체 통과 확인
3. 새 기능에 대한 테스트 추가
4. PR 설명에 변경 사항 명시

## 면책 조항

기여하는 코드는 학술 연구 목적임을 확인하며,
실제 작전 정보나 기밀 정보를 포함하지 않아야 합니다.
