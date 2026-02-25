"""CLI contract smoke tests for train.py / evaluate.py."""
import subprocess
import sys


def _run_cmd(cmd):
    p = subprocess.run(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout


def test_cli_contract():
    """pytest: train.py / evaluate.py CLI 인터페이스 계약 검증."""
    results = []

    def check(name, cond, detail=""):
        results.append((name, cond, detail))
        status = "\u2705 PASS" if cond else "\u274c FAIL"
        print(f"  {status} {name}" + (f" [{detail}]" if detail else ""))

    print("\n" + "=" * 55)
    print("CLI CONTRACT TEST")
    print("=" * 55)

    rc, _ = _run_cmd("python evaluate.py --benchmark historical --benchmark-runs 1")
    check("evaluate.py accepts --benchmark historical", rc == 0, f"rc={rc}")

    rc, _ = _run_cmd("python evaluate.py --benchmark invalid")
    check("evaluate.py rejects invalid benchmark choices", rc != 0, f"rc={rc}")

    rc, _ = _run_cmd("python evaluate.py --fast --monte-carlo 5 --no-progress")
    check("evaluate.py accepts fast/no-progress options", rc == 0, f"rc={rc}")

    rc, _ = _run_cmd("python evaluate.py --fast --monte-carlo 3 --max-steps 3 --output-json /tmp/falcon_eval.json --no-progress")
    check("evaluate.py supports --output-json", rc == 0, f"rc={rc}")

    rc, _ = _run_cmd("python train.py --phase 3 --episodes 1 --hitl")
    check("train.py accepts phase 3 with --hitl", rc == 0, f"rc={rc}")

    rc, _ = _run_cmd("python train.py --phase 2 --episodes 1 --algorithm mappo --log-interval 1")
    check("train.py accepts phase 2 mappo route", rc == 0, f"rc={rc}")

    rc, _ = _run_cmd("python train.py --phase 9")
    check("train.py rejects invalid phase", rc != 0, f"rc={rc}")

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 55)
    print(f"Passed: {len(results) - len(failed)}/{len(results)}")
    if failed:
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")

    assert not failed, (
        "CLI contract failures:\n"
        + "\n".join(f"  FAIL {n}: {d}" for n, _, d in failed)
    )


# ── standalone execution ──────────────────────────────────────
if __name__ == "__main__":
    try:
        test_cli_contract()
        print("\u2705 CLI contract tests passed")
    except AssertionError as exc:
        print(exc)
        sys.exit(1)
