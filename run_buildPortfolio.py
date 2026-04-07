"""
run_buildPortfolio.py — Incremental portfolio builder.

Two-phase approach:
  Phase 1 — Find a valid base combination of `base_size` strategies that
             pass all correlation filters against each other.
  Phase 2 — Test every remaining strategy one by one. If it passes all
             filters against every strategy already in the portfolio, add it.

No weighting is applied during the search — only correlation/filter checks.
The final combination is then weighted with all 4 methods and shown in
the portfolio dashboard.

Usage:
    python run_buildPortfolio.py
    python run_buildPortfolio.py --folder TradingData
    python run_buildPortfolio.py --base 5 --max 15
    python run_buildPortfolio.py --seed 42
    python run_buildPortfolio.py --runs 5          # find 5 different base combos, pick best
"""

from __future__ import annotations

import sys
import os
import random
import warnings

import matplotlib.pyplot as plt
from tqdm import tqdm

from alphaforge.loader.loader import load_folder
from alphaforge.portfolio.config import PortfolioConfig
from alphaforge.portfolio.universe import build_universe
from alphaforge.portfolio.generator.filters import (
    FilterStats,
    _apply_static_filters,
    _monthly_pnl,
)
from alphaforge.portfolio.generator.genetic import (
    _RollingCache,
    _build_adjacency,
)
from alphaforge.portfolio.generator.weighting import (
    compute_all_weights,
    build_weighted_portfolio,
    equal_weight,
)
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate
from alphaforge.portfolio.generator.combo_result import CombinationResult
from alphaforge.portfolio.generator.wf import compute_all_wf_equities
from alphaforge.portfolio.stress.mae_stress import stress_test_all
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.portfolio.dashboard_corr import run_correlation_dashboard
from alphaforge.portfolio.dashboard_mc import run_mc_dashboard
from alphaforge.paths import STRATEGIES_APPROVED


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> dict:
    args = sys.argv[1:]
    out  = {
        "folder":    STRATEGIES_APPROVED,
        "base_size": 5,
        "max_size":  15,
        "seed":      None,
        "runs":      1,
    }
    i = 0
    while i < len(args):
        a = args[i]
        if   a == "--folder" and i + 1 < len(args): out["folder"]    = args[i+1];       i += 2
        elif a == "--base"   and i + 1 < len(args): out["base_size"] = int(args[i+1]);  i += 2
        elif a == "--max"    and i + 1 < len(args): out["max_size"]  = int(args[i+1]);  i += 2
        elif a == "--seed"   and i + 1 < len(args): out["seed"]      = int(args[i+1]);  i += 2
        elif a == "--runs"   and i + 1 < len(args): out["runs"]      = int(args[i+1]);  i += 2
        else: i += 1
    return out


# ── Phase 1: find a valid base combo ─────────────────────────────────────────

def _find_base(
    names      : list[str],
    adj        : dict[str, set[str]],
    base_size  : int,
    rng        : random.Random,
    max_tries  : int = 10_000,
) -> tuple[str, ...] | None:
    """
    Greedy search for a valid base combo of exactly base_size strategies.
    Shuffles the name list and walks through adding each strategy if it is
    compatible with all already-chosen ones. Retries with different shuffles.
    """
    for _ in range(max_tries):
        shuffled  = names[:]
        rng.shuffle(shuffled)
        combo = []
        for candidate in shuffled:
            if len(combo) >= base_size:
                break
            if all(candidate in adj[existing] for existing in combo):
                combo.append(candidate)
        if len(combo) == base_size:
            return tuple(sorted(combo))
    return None


# ── Phase 2: expand by testing each remaining strategy ────────────────────────

def _expand(
    base       : tuple[str, ...],
    remaining  : list[str],
    adj        : dict[str, set[str]],
    max_size   : int,
    rng        : random.Random,
) -> tuple[tuple[str, ...], list[str], dict[str, list[str]]]:
    """
    Test each remaining strategy against the current portfolio.
    Returns (final_combo, added_names, rejected_map).
    rejected_map: {candidate: [conflicting strategies]} or {"__cap__": []} if max_size reached.
    """
    portfolio = list(base)
    added     = []
    rejected  : dict[str, list[str]] = {}

    # Shuffle so different runs explore a different order
    candidates = remaining[:]
    rng.shuffle(candidates)

    for candidate in candidates:
        if len(portfolio) >= max_size:
            rejected[candidate] = ["__cap__"]
            continue
        # Check against every strategy already in the portfolio (O(1) per pair)
        conflicts = [existing for existing in portfolio if candidate not in adj[existing]]
        if not conflicts:
            portfolio.append(candidate)
            added.append(candidate)
        else:
            rejected[candidate] = conflicts

    return tuple(sorted(portfolio)), added, rejected


# ── Score a combo (equal-weight return/DD) ────────────────────────────────────

def _score(combo: tuple, strategies: dict, config: PortfolioConfig) -> float:
    weights   = equal_weight(list(combo))
    portfolio = build_weighted_portfolio(combo, strategies, weights)
    scaled    = rescale(combo, "equal", weights, portfolio, config)
    vp        = validate(scaled, config)
    return vp.metrics.get("return_dd_ratio", 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = _parse_args()

    print()
    print("=" * 60)
    print("  AlphaForge — Incremental Portfolio Builder")
    print("=" * 60)
    print(f"  Folder    : {params['folder']}")
    print(f"  Base size : {params['base_size']} strategies")
    print(f"  Max size  : {params['max_size']} strategies")
    print(f"  Runs      : {params['runs']}")
    print(f"  Seed      : {params['seed']}")
    print("=" * 60)
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    strategies = load_folder(params["folder"])
    if not strategies:
        print(f"  No strategies found in {params['folder']}. Exiting.")
        sys.exit(1)
    print(f"  {len(strategies)} strategies loaded.\n")

    config = PortfolioConfig(
        min_strategies=params["base_size"],
        max_strategies=params["max_size"],
    )
    rng = random.Random(params["seed"])

    # ── Precompute ────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("  Building universe matrices...")
        universe = build_universe(
            strategies,
            same_asset_window_hours=config.same_asset_window_hours,
            verbose=False,
        )

        monthly_cache = {name: _monthly_pnl(df) for name, df in strategies.items()}

        print("  Building rolling correlation cache...")
        rolling_cache = _RollingCache()
        rolling_cache.build(list(strategies.keys()), monthly_cache, config, verbose=True)

        print("  Building compatibility graph...")
        adj = _build_adjacency(
            list(strategies.keys()), strategies, config,
            monthly_cache, rolling_cache, universe,
        )

    names      = list(strategies.keys())
    n_compat   = sum(len(v) for v in adj.values()) // 2
    n_pairs    = len(names) * (len(names) - 1) // 2
    print(f"  Compatible pairs: {n_compat}/{n_pairs} "
          f"({n_compat / max(1, n_pairs) * 100:.1f}%)\n")

    # ── Run phases ────────────────────────────────────────────────────────────
    best_combo    = None
    best_score    = -1e9
    best_added    = []
    best_rejected : dict[str, list[str]] = {}

    for run in range(params["runs"]):
        print(f"  ── Run {run + 1}/{params['runs']} " + "─" * 40)

        # Phase 1
        base = _find_base(names, adj, params["base_size"], rng)
        if base is None:
            print(f"  Could not find a valid base of {params['base_size']} strategies. "
                  f"Try relaxing filters or reducing --base.\n")
            continue

        print(f"  Phase 1 — Base ({len(base)} strategies):")
        for s in base:
            print(f"    + {s}")

        # Phase 2
        remaining = [n for n in names if n not in set(base)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combo, added, rejected = _expand(base, remaining, adj, params["max_size"], rng)

        n_rejected = len(rejected)
        n_capped   = sum(1 for v in rejected.values() if v == ["__cap__"])
        n_filtered = n_rejected - n_capped
        print(f"\n  Phase 2 — Expansion ({len(added)} added, {n_filtered} filtered, {n_capped} capped):")
        for s in added:
            print(f"    ✓  {s}")
        for s, conflicts in rejected.items():
            if conflicts == ["__cap__"]:
                print(f"    —  {s}  (portfolio full)")
            else:
                print(f"    ✗  {s}  (conflicts with: {', '.join(conflicts)})")

        # Score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = _score(combo, strategies, config)

        print(f"\n  Final combo: {len(combo)} strategies  |  R/DD = {score:.3f}\n")

        if score > best_score:
            best_score    = score
            best_combo    = combo
            best_added    = added
            best_rejected = rejected

    if best_combo is None:
        print("  No valid portfolio found. Try relaxing filters or reducing --base.\n")
        sys.exit(1)

    # ── Build CombinationResult for best combo ────────────────────────────────
    print(f"  Best combo ({len(best_combo)} strategies, R/DD = {best_score:.3f}):")
    for s in best_combo:
        print(f"    • {s}")
    print()

    print("  Applying weighting methods...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        weights_all = compute_all_weights(best_combo, strategies)
        portfolios  = {}
        eq_weights  = equal_weight(list(best_combo))
        eq_portfolio = build_weighted_portfolio(best_combo, strategies, eq_weights)
        portfolios["equal"] = validate(rescale(best_combo, "equal", eq_weights, eq_portfolio, config), config)

        for method in ("min_variance", "risk_parity", "hrp"):
            w = weights_all[method]
            p = build_weighted_portfolio(best_combo, strategies, w)
            portfolios[method] = validate(rescale(best_combo, method, w, p, config), config)

    result = CombinationResult(
        combination   = best_combo,
        rank          = 1,
        raw_return_dd = best_score,
        raw_metrics   = portfolios["equal"].metrics,
        portfolios    = portfolios,
        fitness_score = best_score,
    )
    combinations = [result]

    print("  Running stress test...")
    stress_test_all(combinations, config, verbose=False)

    print("  Computing walk-forward equity...")
    compute_all_wf_equities(combinations, strategies, config, verbose=False)

    print("  Launching dashboards...\n")
    run_portfolio_dashboard(combinations, config, strategies=strategies, _show=False)
    run_correlation_dashboard(combinations, config, strategies, _show=False)
    run_mc_dashboard(combinations, config, _show=False)
    plt.show()
