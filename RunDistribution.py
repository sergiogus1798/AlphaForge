"""
RunDistribution.py — Launch the Distribution & Equity analysis module.

Opens two windows:
  1. Distribution Panel  — trade PnL distributions, rolling metrics, MAE/MFE
  2. Equity Curve        — cumulative equity + drawdown

Usage:
    python RunDistribution.py
"""

import sys
import os
import subprocess

SCRIPTS = os.path.join(os.path.dirname(__file__), "scripts")

TOOLS = [
    ("Distribution Panel", "run_distribution.py"),
    ("Equity Curve",       "run_equity.py"),
]

if __name__ == "__main__":
    print("=" * 50)
    print("  AlphaForge — Distribution & Equity")
    print("=" * 50)
    for i, (name, _) in enumerate(TOOLS, 1):
        print(f"  [{i}] {name}")
    print()
    print("  Starting tools…  (close windows individually)")
    print("=" * 50)

    procs = []
    for name, script in TOOLS:
        p = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPTS, script)],
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
        )
        procs.append((name, p))
        print(f"  Started: {name}")

    print("\n  All tools running. Press Ctrl+C here to stop all.\n")

    try:
        for _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n  Stopping…")
        for _, p in procs:
            p.terminate()
