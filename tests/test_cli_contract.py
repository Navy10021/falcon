"""CLI contract smoke tests for train.py / evaluate.py."""
import subprocess
import sys

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"  {PASS if cond else FAIL} {name}" + (f" [{detail}]" if detail else ""))


print("\n" + "=" * 55)
print("CLI CONTRACT TEST")
print("=" * 55)

rc, out = run_cmd("python evaluate.py --benchmark historical --benchmark-runs 1")
check("evaluate.py accepts --benchmark historical", rc == 0, f"rc={rc}")

rc, out = run_cmd("python evaluate.py --benchmark invalid")
check("evaluate.py rejects invalid benchmark choices", rc != 0, f"rc={rc}")

rc, out = run_cmd("python train.py --phase 3 --episodes 1 --hitl")
check("train.py accepts phase 3 with --hitl", rc == 0, f"rc={rc}")

rc, out = run_cmd("python train.py --phase 9")
check("train.py rejects invalid phase", rc != 0, f"rc={rc}")

failed = [r for r in results if not r[1]]
print("\n" + "=" * 55)
print(f"Passed: {len(results)-len(failed)}/{len(results)}")
if failed:
    for name, _, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1)

print("✅ CLI contract tests passed")
