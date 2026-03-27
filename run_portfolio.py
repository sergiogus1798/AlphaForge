"""
run_portfolio.py — Portfolio Generation Pipeline.

Loads all strategies, builds the universe (pre-computed correlation matrices),
runs the generation pipeline, and launches the results dashboard.

Usage:
    python run_portfolio.py                          # random generation
    python run_portfolio.py --ga                     # genetic algorithm
    python run_portfolio.py --capital 10000 --daily-limit 4 --total-limit 9
    python run_portfolio.py --min 3 --max 8 --portfolios 50000
    python run_portfolio.py --ga --generations 200 --population 80
"""

import sys
import warnings

import matplotlib.pyplot as plt

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.universe import build_universe
from alphaforge.portfolio.generator.pipeline import run_pipeline
from alphaforge.portfolio.generator.genetic import run_genetic
from alphaforge.portfolio.stress.mae_stress import stress_test_all
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.portfolio.dashboard_corr import run_correlation_dashboard
from alphaforge.portfolio.dashboard_mc import run_mc_dashboard
from alphaforge.portfolio.exporter import export_all
from alphaforge.paths import STRATEGIES_APPROVED as DEFAULT_FOLDER, clean_portfolios_output


def _parse_args() -> tuple[PortfolioConfig, bool]:
    args    = sys.argv[1:]
    kw      = {}
    use_ga  = False

    i = 0
    while i < len(args):
        a = args[i]
        if   a == "--ga":                                                          use_ga = True; i += 1
        elif a == "--capital"      and i + 1 < len(args): kw["account_balance"]         = float(args[i+1]); i += 2
        elif a == "--daily-limit"  and i + 1 < len(args): kw["daily_loss_limit_pct"]    = float(args[i+1]) / 100; i += 2
        elif a == "--total-limit"  and i + 1 < len(args): kw["total_drawdown_limit_pct"] = float(args[i+1]) / 100; i += 2
        elif a == "--min"          and i + 1 < len(args): kw["min_strategies"]           = int(args[i+1]); i += 2
        elif a == "--max"          and i + 1 < len(args): kw["max_strategies"]           = int(args[i+1]); i += 2
        elif a == "--portfolios"   and i + 1 < len(args): kw["n_portfolios"]             = int(args[i+1]); i += 2
        elif a == "--seed"         and i + 1 < len(args): kw["random_seed"]              = int(args[i+1]); i += 2
        elif a == "--population"   and i + 1 < len(args): kw["ga_population_size"]       = int(args[i+1]); i += 2
        elif a == "--generations"  and i + 1 < len(args): kw["ga_generations"]           = int(args[i+1]); i += 2
        else: i += 1

    return PortfolioConfig(**kw), use_ga


if __name__ == "__main__":
    config, use_ga = _parse_args()

    print()
    print("=" * 56)
    print("  AlphaForge — Portfolio Generation")
    print("=" * 56)
    print(f"  Method         : {'Genetic Algorithm' if use_ga else 'Random sampling'}")
    print(f"  Account        : ${config.account_balance:,.0f}")
    print(f"  Daily limit    : {config.daily_loss_limit_pct*100:.1f}%  "
          f"(${config.daily_loss_limit_usd:,.0f})")
    print(f"  Total DD limit : {config.total_drawdown_limit_pct*100:.1f}%  "
          f"(${config.total_drawdown_limit_usd:,.0f})")
    print(f"  Portfolio size : {config.min_strategies}–{config.max_strategies} strategies")
    if use_ga:
        print(f"  Population     : {config.ga_population_size}")
        print(f"  Generations    : {config.ga_generations}")
    else:
        print(f"  Combinations   : {config.n_portfolios:,}")
    print(f"  Top selected   : {config.top_combinations}")
    print("=" * 56)
    print()

    # Wipe portfolios/ output folder before new run
    clean_portfolios_output(verbose=True)

    # Load strategies
    strategies = load_folder(DEFAULT_FOLDER)
    if not strategies:
        print("  No strategies found in strategies/approved/. Exiting.")
        sys.exit(1)
    print(f"  {len(strategies)} strategies loaded.\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if use_ga:
            universe   = build_universe(strategies, same_asset_same_day=config.same_asset_same_day)
            combinations = run_genetic(strategies, config, universe, verbose=True)
        else:
            result       = run_pipeline(strategies, config, verbose=True)
            combinations = result.combinations
            if not combinations:
                result.warn_if_empty()
                sys.exit(0)

    if not combinations:
        print("  No valid combinations found. Exiting.")
        sys.exit(0)

    # MAE stress test (informational — no portfolios removed)
    print("  Running MAE stress test...")
    stress_test_all(combinations, config, verbose=True)

    # Export results (Excel + PDF) for each combination
    print("  Exporting results...")
    export_all(combinations, config, strategies, verbose=True)

    # Launch dashboards
    n_combos    = len(combinations)
    n_portfolios = sum(len([vp for vp in cr.portfolios.values() if vp is not None])
                       for cr in combinations)
    print(f"  Launching dashboard ({n_combos} combinations, {n_portfolios} portfolios)...")
    run_portfolio_dashboard(combinations, config, strategies=strategies, _show=False)
    print("  Launching deep correlation analysis...")
    run_correlation_dashboard(combinations, config, strategies, _show=False)
    print("  Launching Monte Carlo dashboard...")
    run_mc_dashboard(combinations, config, _show=False)
    plt.show()
