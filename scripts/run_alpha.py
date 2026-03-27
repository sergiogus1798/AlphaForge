"""
run_alpha.py — Launch the core Alpha Detection tool.

Loads strategy trades from strategies/approved/, loads the matching D1 price file
from AssetsData/, then opens the alpha detection figure (6-subplot dark theme).

Usage:
    python run_alpha.py                              # auto-picks first strategy
    python run_alpha.py "AUDJPY/Strategy 7.11.56"   # specific strategy key
    python run_alpha.py "AUDJPY/Strategy 7.11.56" 10000  # custom initial capital
"""

import sys
import warnings

from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.AlphaDetection.data_cleaner import load_price_data
from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector, plot_alpha_report

DEFAULT_FOLDER  = "strategies/approved"
DEFAULT_CAPITAL = 10_000.0


def _pick_strategy(strategies: dict, key: str | None) -> tuple[str, object]:
    """Return (strategy_name, trades_df) — either the requested key or the first."""
    if key and key in strategies:
        return key, strategies[key]
    if key:
        print(f"  Warning: '{key}' not found. Available strategies:")
        for k in list(strategies.keys())[:10]:
            print(f"    {k}")
        print("  Falling back to first strategy.\n")
    name = next(iter(strategies))
    return name, strategies[name]


def _extract_pair(strategy_name: str) -> str:
    """
    Infer the instrument pair from the strategy name.
    Expects names like "AUDJPY/Strategy 7.11.56" or "USDJPY/Strategy 1".
    """
    return strategy_name.split("/")[0].strip()


if __name__ == "__main__":
    key     = sys.argv[1] if len(sys.argv) > 1 else None
    capital = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAPITAL

    # ── Load strategies ────────────────────────────────────────────────────────
    strategies = load_folder(DEFAULT_FOLDER)
    if not strategies:
        print("No strategies found in strategies/approved/. Exiting.")
        sys.exit(1)

    strategy_name, trades_df = _pick_strategy(strategies, key)
    pair = _extract_pair(strategy_name)

    print(f"  Strategy  :  {strategy_name}")
    print(f"  Pair      :  {pair}")
    print(f"  Capital   :  {capital:,.0f}")
    print()

    # ── Load price data ────────────────────────────────────────────────────────
    try:
        price_df = load_price_data(pair, data_folder="AssetsData")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    # ── Run alpha detection ────────────────────────────────────────────────────
    detector = AlphaDetector(
        trades_df       = trades_df,
        price_df        = price_df,
        strategy_name   = strategy_name,
        pair            = pair,
        initial_capital = capital,
    )

    def _reload(chosen_key: str) -> AlphaDetector:
        pr  = _extract_pair(chosen_key)
        pf  = load_price_data(pr, data_folder="AssetsData")
        return AlphaDetector(
            trades_df       = strategies[chosen_key],
            price_df        = pf,
            strategy_name   = chosen_key,
            pair            = pr,
            initial_capital = capital,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_alpha_report(
            detector,
            all_strategies = strategies,
            reload_fn      = _reload,
        )
