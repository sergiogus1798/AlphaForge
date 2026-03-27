"""
run_advanced.py — Launch the Advanced Alpha Tests.

Always opens TWO figures:
  1. Static figure (2×2) — Tests 1-4 (no placebo)
  2. Rolling figure (interactive 3×2) — type a window size and click Run

Usage:
    python run_advanced.py                               # first strategy
    python run_advanced.py "AUDJPY/Strategy 7.11.56"    # specific strategy
    python run_advanced.py "AUDJPY/Strategy 7.11.56" --capital 50000
"""

import sys
import warnings

from alphaforge.loader import load_folder
from alphaforge.IndividualAnalysis.AlphaDetection.data_cleaner import load_price_data
from alphaforge.IndividualAnalysis.AlphaDetection import AlphaDetector
from alphaforge.IndividualAnalysis.AlphaDetection.advanced_tests import AdvancedTests
from alphaforge.IndividualAnalysis.AlphaDetection.plot_advanced import plot_advanced_report

DEFAULT_FOLDER  = "strategies/approved"
DEFAULT_CAPITAL = 10_000.0


def _parse_args():
    key            = None
    n_permutations = 1000
    capital        = DEFAULT_CAPITAL
    init_window    = 756

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--perms" and i + 1 < len(args):
            n_permutations = int(args[i + 1]); i += 2
        elif a == "--capital" and i + 1 < len(args):
            capital = float(args[i + 1]); i += 2
        elif a == "--window" and i + 1 < len(args):
            init_window = int(args[i + 1]); i += 2
        elif not a.startswith("--"):
            key = a; i += 1
        else:
            i += 1

    return key, n_permutations, capital, init_window


def _pick_strategy(strategies: dict, key: str | None):
    if key and key in strategies:
        return key, strategies[key]
    if key:
        print(f"  Warning: '{key}' not found. Available:")
        for k in list(strategies.keys())[:10]:
            print(f"    {k}")
        print("  Falling back to first strategy.\n")
    name = next(iter(strategies))
    return name, strategies[name]


def _extract_pair(strategy_name: str) -> str:
    return strategy_name.split("/")[0].strip()


if __name__ == "__main__":
    key, n_permutations, capital, init_window = _parse_args()

    strategies = load_folder(DEFAULT_FOLDER)
    if not strategies:
        print("No strategies found in strategies/approved/. Exiting.")
        sys.exit(1)

    strategy_name, trades_df = _pick_strategy(strategies, key)
    pair = _extract_pair(strategy_name)

    print(f"  Strategy    :  {strategy_name}")
    print(f"  Pair        :  {pair}")
    print(f"  Capital     :  {capital:,.0f}")
    print(f"  Permutations:  {n_permutations:,}")
    print()

    try:
        price_df = load_price_data(pair, data_folder="AssetsData")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    detector = AlphaDetector(
        trades_df=trades_df, price_df=price_df,
        strategy_name=strategy_name, pair=pair,
        initial_capital=capital,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aligned = detector.build_returns()
        core    = detector.run_core_regression(aligned)

    print(f"  Core: a={core.alpha*100:+.4f}%/day  b={core.beta:+.3f}  "
          f"t(a)={core.alpha_tstat:+.2f}  R2={core.r_squared:.3f}  [{core.verdict}]")
    print()

    tester = AdvancedTests(aligned, core)

    def _reload(chosen_key: str):
        """Load a different strategy and return (AdvancedTests, name, pair)."""
        name = chosen_key
        df   = strategies[chosen_key]
        pr   = _extract_pair(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf  = load_price_data(pr, data_folder="AssetsData")
            det = AlphaDetector(trades_df=df, price_df=pf,
                                strategy_name=name, pair=pr,
                                initial_capital=capital)
            aln = det.build_returns()
            cr  = det.run_core_regression(aln)
        return AdvancedTests(aln, cr), name, pr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_advanced_report(
            tester,
            strategy_name          = strategy_name,
            pair                   = pair,
            initial_rolling_window = init_window,
            n_permutations         = n_permutations,
            all_strategies         = strategies,
            reload_fn              = _reload,
        )
