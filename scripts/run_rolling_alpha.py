"""
run_rolling_alpha.py — Rolling alpha/beta overlay on asset price.

Usage:
    python run_rolling_alpha.py                              # first strategy
    python run_rolling_alpha.py "AUDJPY/Strategy 11.12.166"
    python run_rolling_alpha.py "AUDJPY/Strategy 11.12.166" --lookback 126 --step 22
"""

import sys

from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.AlphaDetection.data_cleaner import load_price_data
from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector
from alphaforge.IndividualAnalysis.RollingAlpha import plot_rolling_alpha

DEFAULT_FOLDER   = "strategies/approved"
DEFAULT_LOOKBACK = 252
DEFAULT_STEP     = 22


def _parse_args():
    key      = None
    lookback = DEFAULT_LOOKBACK
    step     = DEFAULT_STEP

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--lookback" and i + 1 < len(args):
            lookback = int(args[i + 1]); i += 2
        elif a == "--step" and i + 1 < len(args):
            step = int(args[i + 1]); i += 2
        else:
            key = a; i += 1

    return key, lookback, step


def main():
    key, lookback, step = _parse_args()

    strategies = load_folder(DEFAULT_FOLDER)

    if key is None:
        key = next(iter(strategies))

    if key not in strategies:
        print(f"Strategy '{key}' not found. Available:")
        for k in strategies:
            print(f"  {k}")
        sys.exit(1)

    print(f"  Lookback : {lookback}d  |  Step : {step}d")
    print()

    def load_strategy(k: str):
        """Given a strategy key, return (aligned, price_df, pair)."""
        pair     = k.split("/")[0]
        price_df = load_price_data(pair)
        detector = AlphaDetector(strategies[k], price_df,
                                 strategy_name=k, pair=pair)
        aligned  = detector.build_returns()
        return aligned, price_df, pair

    plot_rolling_alpha(
        all_strategies  = strategies,
        load_strategy   = load_strategy,
        initial_key     = key,
        lookback        = lookback,
        step            = step,
    )


if __name__ == "__main__":
    main()
