"""
RunMonteCarlo.py — Launch the Monte Carlo simulation module.

Opens one window:
  1. Monte Carlo  — return simulation, risk metrics, drawdown distribution

Usage:
    python RunMonteCarlo.py
"""

import sys
import os
import subprocess

SCRIPTS = os.path.join(os.path.dirname(__file__), "scripts")

if __name__ == "__main__":
    print("=" * 50)
    print("  AlphaForge — Monte Carlo")
    print("=" * 50)
    print("  [1] Monte Carlo Simulation")
    print()
    print("  Starting…")
    print("=" * 50)

    p = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPTS, "run_mc.py")],
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
    )
    print("  Started: Monte Carlo\n")

    try:
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
