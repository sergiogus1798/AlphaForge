"""
run_mc.py — Launch the Monte Carlo dashboard for AlphaForge.

Usage:
    python run_mc.py                        # defaults to TradingData/
    python run_mc.py <path_to_folder>
"""

import sys
from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.MonteCarlo.dashboard import run_mc_dashboard

DEFAULT_FOLDER = "TradingData"


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    strategies = load_folder(folder)
    if not strategies:
        print("No strategies loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(strategies)} strategy/strategies: {list(strategies.keys())}")
    run_mc_dashboard(strategies)
