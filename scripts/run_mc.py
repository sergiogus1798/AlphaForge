"""
run_mc.py — Launch the Monte Carlo panel for AlphaForge.

Usage:
    python scripts/run_mc.py                                    # strategies/approved/
    python scripts/run_mc.py --folder TradingData/
    python scripts/run_mc.py --folder TradingData/ "AUDJPY/Strategy 8.11.130"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.MonteCarlo.plot_mc_panel import plot_mc_panel

DEFAULT_FOLDER  = "strategies/approved"
INITIAL_CAPITAL = 10_000.0


if __name__ == "__main__":
    raw    = sys.argv[1:]
    folder = next((a.split("=", 1)[1] if "=" in a else raw[raw.index(a) + 1]
                   for a in raw if a == "--folder" or a.startswith("--folder=")),
                  DEFAULT_FOLDER)
    _skip = False
    _pos  = []
    for a in raw:
        if _skip:
            _skip = False
            continue
        if a == "--folder":
            _skip = True
            continue
        if a.startswith("--folder="):
            continue
        if not a.startswith("--"):
            _pos.append(a)

    strategies = load_folder(folder)
    if not strategies:
        print("No strategies loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(strategies)} strategies.")

    all_keys = sorted(strategies.keys())
    if _pos:
        query = _pos[0].lower()
        match = next((k for k in all_keys if query in k.lower()), None)
        initial_key = match if match else all_keys[0]
    else:
        initial_key = all_keys[0]

    print(f"Opening: {initial_key}")
    plot_mc_panel(strategies, initial_key, initial_capital=INITIAL_CAPITAL)
