"""
scripts/check_perf_gate.py
==========================
CI 성능 회귀 게이트 — evaluate.py --output-json 결과를 읽어 임계값 검사.

사용::

    python scripts/check_perf_gate.py /tmp/perf_gate.json \
        --min-win-rate 0.45 \
        --max-cvar10 40.0

종료 코드:
  0 — 게이트 통과
  1 — 게이트 실패 (임계값 미달)
  2 — 파일/파싱 오류

학술 연구용 합성 데이터
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_report(path: str) -> dict:
    """evaluate.py --output-json 결과 JSON 로드"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[PerfGate] ERROR: 파일 없음 — {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"[PerfGate] ERROR: JSON 파싱 실패 — {exc}", file=sys.stderr)
        sys.exit(2)


def _extract_metrics(report: dict) -> dict:
    """MCEvaluationReport JSON에서 핵심 지표 추출"""
    # evaluate.py → evaluator.save_report_json() 결과 구조에 맞춤
    metrics = {
        "blue_win_rate": report.get("blue_win_rate", report.get("win_rate", None)),
        "cvar_10":       report.get("cvar_10", None),
        "cvar_05":       report.get("cvar_05", None),
        "avg_casualties":report.get("avg_blue_casualties", None),
        "robustness":    report.get("strategy_robustness", None),
        "n_runs":        report.get("n_runs", 0),
    }
    return metrics


def run_gate(
    report_path: str,
    min_win_rate: float = 0.45,
    max_cvar10: float = 40.0,
    verbose: bool = True,
) -> bool:
    """
    성능 게이트 실행.

    Returns
    -------
    bool
        True이면 모든 임계값 통과, False이면 하나 이상 실패.
    """
    report = _load_report(report_path)
    metrics = _extract_metrics(report)

    if verbose:
        print("\n── FALCON 성능 회귀 게이트 ────────────────────")
        print(f"  리포트 파일  : {report_path}")
        print(f"  MC runs     : {metrics['n_runs']}")
        print(f"  Blue 승률   : {metrics['blue_win_rate']:.4f}  (임계값 ≥ {min_win_rate})")
        print(f"  CVaR@10     : {metrics['cvar_10']}  (임계값 ≤ {max_cvar10})")
        print()

    failures: list[str] = []

    # ① 승률 게이트
    wr = metrics["blue_win_rate"]
    if wr is None:
        failures.append(f"blue_win_rate 지표 없음 (리포트 구조 확인 필요)")
    elif wr < min_win_rate:
        failures.append(
            f"blue_win_rate={wr:.4f} < {min_win_rate} "
            f"(미달: {(min_win_rate - wr) * 100:.2f}%p)"
        )

    # ② CVaR@10 게이트
    cvar10 = metrics["cvar_10"]
    if cvar10 is None:
        failures.append("cvar_10 지표 없음 (리포트 구조 확인 필요)")
    elif cvar10 > max_cvar10:
        failures.append(
            f"cvar_10={cvar10:.2f} > {max_cvar10} "
            f"(초과: {cvar10 - max_cvar10:.2f})"
        )

    # 결과 출력
    if failures:
        print("❌ 성능 회귀 게이트 실패:")
        for f in failures:
            print(f"   • {f}")
        return False

    if verbose:
        print("✅ 성능 회귀 게이트 통과")
        if metrics["robustness"] is not None:
            print(f"   strategy_robustness = {metrics['robustness']:.4f}")
        if metrics["avg_casualties"] is not None:
            print(f"   avg_blue_casualties = {metrics['avg_casualties']:.1f}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FALCON CI 성능 회귀 게이트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "report_json",
        help="evaluate.py --output-json으로 생성된 JSON 파일 경로",
    )
    parser.add_argument(
        "--min-win-rate", type=float, default=0.45,
        help="최소 Blue 승률 임계값 (이 값 미만이면 게이트 실패)",
    )
    parser.add_argument(
        "--max-cvar10", type=float, default=40.0,
        help="CVaR@10 최대 허용 사상자 수 (이 값 초과 시 게이트 실패)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="간략 출력 (게이트 결과만)",
    )
    args = parser.parse_args()

    passed = run_gate(
        report_path=args.report_json,
        min_win_rate=args.min_win_rate,
        max_cvar10=args.max_cvar10,
        verbose=not args.quiet,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
