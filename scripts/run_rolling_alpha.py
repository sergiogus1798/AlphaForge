"""
run_rolling_alpha.py — Rolling alpha/beta overlay on asset price.

Usage:
    python scripts/run_rolling_alpha.py --folder TradingData/
    python scripts/run_rolling_alpha.py --folder TradingData/ "AUDJPY/Strategy 8.11.130"
    python scripts/run_rolling_alpha.py --folder TradingData/ "AUDJPY/Strategy 8.11.130" --lookback 126 --step 22
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.AlphaDetection.data_cleaner import load_price_data
from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector
from alphaforge.IndividualAnalysis.RollingAlpha import plot_rolling_alpha

DEFAULT_FOLDER   = "strategies/approved"
DEFAULT_LOOKBACK = 252
DEFAULT_STEP     = 22


def _parse_args():
    raw      = sys.argv[1:]
    folder   = DEFAULT_FOLDER
    key      = None
    lookback = DEFAULT_LOOKBACK
    step     = DEFAULT_STEP

    i = 0
    while i < len(raw):
        a = raw[i]
        if a == "--folder" and i + 1 < len(raw):
            folder = raw[i + 1]; i += 2
        elif a.startswith("--folder="):
            folder = a.split("=", 1)[1]; i += 1
        elif a == "--lookback" and i + 1 < len(raw):
            lookback = int(raw[i + 1]); i += 2
        elif a == "--step" and i + 1 < len(raw):
            step = int(raw[i + 1]); i += 2
        elif not a.startswith("--"):
            key = a; i += 1
        else:
            i += 1

    return folder, key, lookback, step


def main():
    folder, key, lookback, step = _parse_args()

    strategies = load_folder(folder)
    if not strategies:
        print(f"No strategies found in {folder}. Exiting.")
        sys.exit(1)

    all_keys = sorted(strategies.keys())

    if key is None:
        key = all_keys[0]
    elif key not in strategies:
        match = next((k for k in all_keys if key.lower() in k.lower()), None)
        if match:
            key = match
        else:
            print(f"Strategy '{key}' not found. Available:")
            for k in all_keys:
                print(f"  {k}")
            sys.exit(1)

    print(f"  Strategy : {key}")
    print(f"  Lookback : {lookback}d  |  Step : {step}d")
    print()

    def load_strategy(k: str):
        pair     = k.split("/")[0]
        price_df = load_price_data(pair, data_folder="AssetsData")
        detector = AlphaDetector(strategies[k], price_df,
                                 strategy_name=k, pair=pair)
        aligned  = detector.build_returns()
        return aligned, price_df, pair

    plot_rolling_alpha(
        all_strategies = strategies,
        load_strategy  = load_strategy,
        initial_key    = key,
        lookback       = lookback,
        step           = step,
    )


if __name__ == "__main__":
    main()
