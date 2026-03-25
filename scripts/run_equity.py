"""
run_equity.py — Launch the Equity Curve panel for AlphaForge.

Usage:
    python run_equity.py                          # all strategies, first as default
    python run_equity.py "AUDJPY/Strategy 11"    # open on specific strategy
"""

import sys
import pandas as pd
from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.EquityCurve.plot_equity_panel import plot_equity_panel

DEFAULT_FOLDER  = "TradingData"
INITIAL_CAPITAL = 10_000.0
PORTFOLIO_KEY   = "\u25c6 Portfolio (All)"


if __name__ == "__main__":
    args   = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = DEFAULT_FOLDER

    strategies = load_folder(folder)
    if not strategies:
        print("No strategies loaded. Exiting.")
        sys.exit(1)

    # Build portfolio entry (all strategies combined, sorted by close time)
    portfolio_df = (
        pd.concat(strategies.values(), ignore_index=True)
        .sort_values("Close time")
        .reset_index(drop=True)
    )
    strategies = {PORTFOLIO_KEY: portfolio_df, **strategies}

    all_keys = [PORTFOLIO_KEY] + sorted(k for k in strategies if k != PORTFOLIO_KEY)
    print(f"Loaded {len(strategies) - 1} strategies (+1 portfolio).")

    if args:
        query = args[0].lower()
        match = next((k for k in all_keys if query in k.lower()), None)
        if match is None:
            print(f"Strategy '{args[0]}' not found. Available:\n" +
                  "\n".join(f"  {k}" for k in all_keys))
            sys.exit(1)
        initial_key = match
    else:
        initial_key = all_keys[0]

    print(f"Opening: {initial_key}")
    plot_equity_panel(strategies, initial_key, initial_capital=INITIAL_CAPITAL)
