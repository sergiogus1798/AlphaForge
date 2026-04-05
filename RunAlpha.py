"""
RunAlpha.py — Launch the full Alpha Analysis module.

Opens four windows:
  1. Alpha Detection      — core alpha/beta regression (6 diagnostic plots)
  2. Advanced Alpha Tests — Tests 1-4: autocorrelation, variance ratio, tail, Hurst
  3. Regime Detection     — HMM regimes + drawdown clustering
  4. Rolling Alpha/Beta   — rolling OLS alpha & beta overlay

Usage:
    python RunAlpha.py
    python RunAlpha.py --folder TradingData
    python RunAlpha.py --folder strategies/approved
"""

import sys
import os
import subprocess

SCRIPTS = os.path.join(os.path.dirname(__file__), "scripts")

TOOLS = [
    ("Alpha Detection",      "run_alpha.py"),
    ("Advanced Alpha Tests", "run_advanced.py"),
    ("Regime Detection",     "run_regime.py"),
    ("Rolling Alpha/Beta",   "run_rolling_alpha.py"),
]

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
    print("  AlphaForge — Alpha Analysis")
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
