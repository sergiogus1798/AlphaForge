"""
run_analysis.py — Open dashboards for portfolios saved in portfolios/.

For each Combination*/ folder that contains a strategies/ subfolder, loads
the CSVs, rebuilds all 4 weighting methods, runs the stress test, and opens
all 3 dashboards — exactly the same view as after run_portfolio.py.

To analyse a combination, just drop the strategy CSVs into:
    portfolios/CombinationXXX/strategies/

The asset symbol is read directly from the Symbol column in each CSV —
no folder structure or naming convention required.

Usage:
    python run_analysis.py                   # reads portfolios/
    python run_analysis.py my_portfolios/    # custom folder
"""

from __future__ import annotations

import os
import re
import sys
import warnings

import matplotlib.pyplot as plt

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.generator.weighting import (
    compute_all_weights,
    build_weighted_portfolio,
    equal_weight,
)
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate
from alphaforge.portfolio.generator.fitness import score_combinations
from alphaforge.portfolio.generator.combo_result import CombinationResult
from alphaforge.portfolio.stress.mae_stress import stress_test_all
from alphaforge.portfolio.generator.wf import compute_all_wf_equities
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.portfolio.dashboard_corr import run_correlation_dashboard
from alphaforge.portfolio.dashboard_mc import run_mc_dashboard
from alphaforge.paths import PORTFOLIOS_OUTPUT


def _scan_combination_dirs(root: str) -> list[tuple[int, str]]:
    """Return sorted (rank, path) for every Combination*/ folder that has a strategies/ subfolder."""
    if not os.path.isdir(root):
        return []
    results = []
    for entry in sorted(os.listdir(root)):
        m = re.fullmatch(r"Combination(\d+)", entry)
        if not m:
            continue
        combo_dir   = os.path.join(root, entry)
        strat_dir   = os.path.join(combo_dir, "strategies")
        if os.path.isdir(strat_dir) and any(
            f.lower().endswith(".csv")
            for _, _, files in os.walk(strat_dir)
            for f in files
        ):
            results.append((int(m.group(1)), combo_dir))
    return results


def _rebuild_combination(
    rank: int,
    strategies: dict,
    config: PortfolioConfig,
) -> CombinationResult | None:
    combo = tuple(sorted(strategies.keys()))
    if len(combo) < 2:
        return None

    portfolios: dict = {}
    weights_all = compute_all_weights(combo, strategies)

    eq_w         = equal_weight(list(combo))
    eq_portfolio = build_weighted_portfolio(combo, strategies, eq_w)
    eq_scaled    = rescale(combo, "equal", eq_w, eq_portfolio, config)
    portfolios["equal"] = validate(eq_scaled, config)

    for method in ("min_variance", "risk_parity", "hrp"):
        w         = weights_all[method]
        portfolio = build_weighted_portfolio(combo, strategies, w)
        scaled    = rescale(combo, method, w, portfolio, config)
        portfolios[method] = validate(scaled, config)

    return CombinationResult(
        combination   = combo,
        rank          = rank,
        raw_return_dd = portfolios["equal"].metrics.get("return_dd_ratio", 0.0),
        raw_metrics   = portfolios["equal"].metrics,
        portfolios    = portfolios,
        fitness_score = 0.0,
    )


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else PORTFOLIOS_OUTPUT

    print()
    print("=" * 56)
    print("  AlphaForge — Portfolio Analysis")
    print("=" * 56)
    print(f"  Source : {root}/")

    combo_dirs = _scan_combination_dirs(root)
    if not combo_dirs:
        print(
            f"\n  No Combination*/strategies/ folders with CSVs found in {root}/\n"
            f"  Drop your strategy CSVs into:\n"
            f"    {root}/Combination001/strategies/\n"
            f"    {root}/Combination002/strategies/\n"
            f"  ... then re-run.\n"
        )
        sys.exit(1)

    print(f"  Found {len(combo_dirs)} combination(s) with strategies.\n")
    config = PortfolioConfig()

    combinations : list[CombinationResult]  = []
    all_strategies: dict                    = {}

    for rank, combo_dir in combo_dirs:
        label     = os.path.basename(combo_dir)
        strat_dir = os.path.join(combo_dir, "strategies")

        print(f"  Loading {label}...", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strats = load_folder(strat_dir)

        if not strats:
            print("SKIP (no strategies loaded)")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cr = _rebuild_combination(rank, strats, config)

        if cr is None:
            print("SKIP (need at least 2 strategies)")
            continue

        combinations.append(cr)
        all_strategies.update(strats)
        print(f"OK  ({len(strats)} strategies)")

    if not combinations:
        print("\n  No valid combinations. Exiting.")
        sys.exit(1)

    print(f"\n  {len(combinations)} combination(s) loaded.\n")

    # Re-score all combinations together (normalised fitness)
    print("  Scoring...")
    candidates = [(cr.combination, cr.portfolios["equal"]) for cr in combinations]
    scored     = score_combinations(candidates, config)
    score_map  = {combo: s for combo, _, s in scored}
    for cr in combinations:
        cr.fitness_score = score_map.get(cr.combination, 0.0)
    combinations.sort(key=lambda c: c.fitness_score, reverse=True)
    for i, cr in enumerate(combinations, 1):
        cr.rank = i

    # Stress test
    print("  Running MAE stress test...")
    stress_test_all(combinations, config, verbose=True)

    # Walk-forward equity
    print("  Computing walk-forward equity...")
    compute_all_wf_equities(combinations, all_strategies, config, verbose=True)

    # Dashboards
    print(f"  Opening dashboards ({len(combinations)} combination(s))...")
    run_portfolio_dashboard(combinations, config, strategies=all_strategies, _show=False)
    run_correlation_dashboard(combinations, config, all_strategies, _show=False)
    run_mc_dashboard(combinations, config, _show=False)
    plt.show()


if __name__ == "__main__":
    main()
