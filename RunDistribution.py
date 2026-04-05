"""
RunDistribution.py — Launch the Distribution & Equity analysis module.

Opens two windows:
  1. Distribution Panel  — trade PnL distributions, rolling metrics, MAE/MFE
  2. Equity Curve        — cumulative equity + drawdown

Usage:
    python RunDistribution.py
    python RunDistribution.py --folder TradingData
    python RunDistribution.py --folder strategies/approved
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
    # Pass --folder through to both sub-scripts if provided
    extra_args = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--folder" and i + 1 < len(args):
            extra_args = ["--folder", args[i + 1]]
            i += 2
        else:
            i += 1

    print("=" * 50)
    print("  AlphaForge — Distribution & Equity")
    print("=" * 50)
    for i, (name, _) in enumerate(TOOLS, 1):
        print(f"  [{i}] {name}")
    if extra_args:
        print(f"  Folder: {extra_args[1]}")
    print()
    print("  Starting tools…  (close windows individually)")
    print("=" * 50)

    procs = []
    for name, script in TOOLS:
        p = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPTS, script)] + extra_args,
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
