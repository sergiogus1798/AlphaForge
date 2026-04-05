"""
RunMonteCarlo.py — Launch the Monte Carlo simulation module.

Opens one window:
  1. Monte Carlo  — return simulation, risk metrics, drawdown distribution

Usage:
    python RunMonteCarlo.py
    python RunMonteCarlo.py --folder TradingData
    python RunMonteCarlo.py --folder strategies/approved
"""

import sys
import os
import subprocess

SCRIPTS = os.path.join(os.path.dirname(__file__), "scripts")

if __name__ == "__main__":
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
    print("  AlphaForge — Monte Carlo")
    print("=" * 50)
    print("  [1] Monte Carlo Simulation")
    if extra_args:
        print(f"  Folder: {extra_args[1]}")
    print()
    print("  Starting…")
    print("=" * 50)

    p = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "run_mc.py")] + extra_args,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )
    print("  Started: Monte Carlo\n")

    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
