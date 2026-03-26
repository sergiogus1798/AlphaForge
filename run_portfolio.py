"""
run_portfolio.py — Portfolio Generation Pipeline.

Loads all strategies, builds the universe (pre-computed correlation matrices),
runs the generation pipeline (Steps 1-5), and launches the results dashboard.

Usage:
    python run_portfolio.py
    python run_portfolio.py --capital 10000 --daily-limit 4 --total-limit 9
    python run_portfolio.py --min 3 --max 8 --portfolios 50000
"""

import sys
import warnings

import matplotlib.pyplot as plt

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.universe import build_universe
from alphaforge.portfolio.generator.pipeline import run_pipeline
from alphaforge.portfolio.stress.mae_stress import stress_test_all
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.portfolio.dashboard_corr import run_correlation_dashboard

DEFAULT_FOLDER = "TradingData"


def _parse_args() -> PortfolioConfig:
    args = sys.argv[1:]
    kw   = {}

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--capital"     and i + 1 < len(args): kw["account_balance"]         = float(args[i+1]); i += 2
        elif a == "--daily-limit"  and i + 1 < len(args): kw["daily_loss_limit_pct"]    = float(args[i+1]) / 100; i += 2
        elif a == "--total-limit"  and i + 1 < len(args): kw["total_drawdown_limit_pct"] = float(args[i+1]) / 100; i += 2
        elif a == "--min"          and i + 1 < len(args): kw["min_strategies"]           = int(args[i+1]); i += 2
        elif a == "--max"          and i + 1 < len(args): kw["max_strategies"]           = int(args[i+1]); i += 2
        elif a == "--portfolios"   and i + 1 < len(args): kw["n_portfolios"]             = int(args[i+1]); i += 2
        elif a == "--seed"         and i + 1 < len(args): kw["random_seed"]              = int(args[i+1]); i += 2
        else: i += 1

    return PortfolioConfig(**kw)


if __name__ == "__main__":
    config = _parse_args()

    print()
    print("=" * 56)
    print("  AlphaForge — Portfolio Generation")
    print("=" * 56)
    print(f"  Account        : ${config.account_balance:,.0f}")
    print(f"  Daily limit    : {config.daily_loss_limit_pct*100:.1f}%  "
          f"(${config.daily_loss_limit_usd:,.0f})")
    print(f"  Total DD limit : {config.total_drawdown_limit_pct*100:.1f}%  "
          f"(${config.total_drawdown_limit_usd:,.0f})")
    print(f"  Portfolio size : {config.min_strategies}–{config.max_strategies} strategies")
    print(f"  Combinations   : {config.n_portfolios:,}")
    print(f"  Top selected   : {config.top_combinations}")
    print("=" * 56)
    print()

    # Load strategies
    strategies = load_folder(DEFAULT_FOLDER)
    if not strategies:
        print("  No strategies found in TradingData/. Exiting.")
        sys.exit(1)
    print(f"  {len(strategies)} strategies loaded.\n")

    # Run full pipeline (universe pre-computation happens inside)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(strategies, config, verbose=True)

    if not result.combinations:
        result.warn_if_empty()
        sys.exit(0)

    # MAE stress test (informational — no portfolios removed)
    print("  Running MAE stress test...")
    stress_test_all(result.combinations, config, verbose=True)

    # Launch dashboard
    print(f"  Launching dashboard ({result.count} combinations, "
          f"{result.total_portfolios} portfolios)...")
    # Build all windows first (non-blocking), then show them all together
    run_portfolio_dashboard(result.combinations, config, strategies=strategies, _show=False)
    print("  Launching deep correlation analysis...")
    run_correlation_dashboard(result.combinations, config, strategies, _show=False)
    plt.show()   # blocks here; all windows open simultaneously
