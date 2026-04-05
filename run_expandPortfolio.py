"""
run_expandPortfolio.py — Expand an existing portfolio by finding the best strategy additions.

Loads a base portfolio from a Combination folder, then searches the full
TradingData/ universe for the best additional strategies to add — keeping
every base strategy fixed and only adding new ones that pass all filters
with the existing portfolio and with each other.

The search is exhaustive: all valid subsets of size 1..max_additions are
scored and ranked. Since the compatible candidate pool is typically small
after filtering against the base, exhaustive search is always fast.

Usage:
    python run_expandPortfolio.py                              # defaults
    python run_expandPortfolio.py --base portfolios/Combo002   # custom base folder
    python run_expandPortfolio.py --max 12                     # allow up to 12 strategies total
    python run_expandPortfolio.py --no-rolling                 # disable rolling correlation filter
    python run_expandPortfolio.py --no-co-loss                 # disable co-loss filter
    python run_expandPortfolio.py --no-tail                    # disable tail correlation filter
    python run_expandPortfolio.py --no-same-asset              # disable same-asset window filter
    python run_expandPortfolio.py --pearson 0.45               # relax Pearson threshold
    python run_expandPortfolio.py --spearman 0.55              # relax Spearman threshold
    python run_expandPortfolio.py --rolling 0.50               # relax rolling correlation threshold
    python run_expandPortfolio.py --no-rolling --pearson 0.45  # combine flags freely
"""

from __future__ import annotations

import os
import sys
import warnings
from itertools import combinations as iter_combinations

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
from alphaforge.portfolio.generator.weighting import (
    compute_all_weights,
    build_weighted_portfolio,
    equal_weight,
)
from alphaforge.portfolio.generator.scaler import rescale
from alphaforge.portfolio.generator.validator import validate
from alphaforge.portfolio.generator.combo_result import CombinationResult
from alphaforge.portfolio.generator.fitness import score_combinations
from alphaforge.portfolio.generator.genetic import _RollingCache
from alphaforge.portfolio.generator.wf import compute_all_wf_equities
from alphaforge.portfolio.stress.mae_stress import stress_test_all
from alphaforge.portfolio.dashboard import run_portfolio_dashboard
from alphaforge.portfolio.dashboard_corr import run_correlation_dashboard
from alphaforge.portfolio.dashboard_mc import run_mc_dashboard
from alphaforge.paths import PORTFOLIOS_OUTPUT, STRATEGIES_APPROVED

TRADING_DATA = STRATEGIES_APPROVED


def _basename(key: str) -> str:
    """Extract the strategy filename part from a (possibly asset-prefixed) key."""
    return key.split("/")[-1]


def _parse_args() -> tuple[str, dict]:
    args     = sys.argv[1:]
    base_dir = os.path.join(PORTFOLIOS_OUTPUT, "Combination001")
    kw: dict = {}

    i = 0
    while i < len(args):
        a = args[i]
        if   a == "--base"         and i + 1 < len(args): base_dir                          = args[i+1]; i += 2
        elif a == "--max"          and i + 1 < len(args): kw["max_strategies"]               = int(args[i+1]); i += 2
        elif a == "--pearson"      and i + 1 < len(args): kw["max_pearson_corr"]             = float(args[i+1]); i += 2
        elif a == "--spearman"     and i + 1 < len(args): kw["max_spearman_corr"]            = float(args[i+1]); i += 2
        elif a == "--rolling"      and i + 1 < len(args): kw["max_rolling_corr"]             = float(args[i+1]); \
                                                           kw["max_rolling_corr_recent"]      = float(args[i+1]); i += 2
        elif a == "--no-rolling":                          kw["enable_rolling_filter"]        = False; i += 1
        elif a == "--no-co-loss":                          kw["enable_co_loss_filter"]        = False; i += 1
        elif a == "--no-tail":                             kw["enable_tail_corr_filter"]      = False; i += 1
        elif a == "--no-same-asset":                       kw["enable_same_asset_filter"]     = False; i += 1
        else: i += 1

    return base_dir, kw


def main() -> None:
    # ── CLI args ──────────────────────────────────────────────────────────────
    base_dir, config_kw = _parse_args()
    max_total = config_kw.get("max_strategies", 10)

    print()
    print("=" * 60)
    print("  AlphaForge — Portfolio Expander")
    print("=" * 60)
    print(f"  Base portfolio : {base_dir}/strategies/")
    print(f"  Candidate pool : {TRADING_DATA}/")
    print(f"  Max total size : {max_total} strategies")
    if config_kw:
        overrides = {k: v for k, v in config_kw.items() if k != "max_strategies"}
        if overrides:
            print(f"  Filter overrides: {overrides}")
    print()

    # ── Load base portfolio ───────────────────────────────────────────────────
    strat_dir = os.path.join(base_dir, "strategies")
    if not os.path.isdir(strat_dir):
        print(f"  ERROR: {strat_dir}/ not found.\n"
              f"  Drop your strategy CSVs into {strat_dir}/\n")
        sys.exit(1)

    print("  Loading base portfolio...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_strategies = load_folder(strat_dir)

    if len(base_strategies) < 2:
        print(f"  ERROR: need at least 2 strategies in the base portfolio.\n")
        sys.exit(1)

    base_names     = list(base_strategies.keys())
    base_basenames = {_basename(k) for k in base_names}
    print(f"\n  Base portfolio: {len(base_names)} strategies.\n")
    for name in base_names:
        print(f"    • {name}")

    # ── Load TradingData universe, remove duplicates ──────────────────────────
    print(f"\n  Loading candidate pool from {TRADING_DATA}/...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe_all = load_folder(TRADING_DATA)

    candidates = {
        k: v for k, v in universe_all.items()
        if _basename(k) not in base_basenames
    }

    skipped = len(universe_all) - len(candidates)
    print(f"\n  Universe: {len(universe_all)} strategies loaded.")
    if skipped:
        print(f"  Skipped {skipped} already in base portfolio.")
    print(f"  Candidates to evaluate: {len(candidates)}\n")

    if not candidates:
        print("  No new candidates found. Exiting.\n")
        sys.exit(1)

    # ── Merge all strategies for filter computation ───────────────────────────
    all_strategies = {**base_strategies, **candidates}
    all_names      = list(all_strategies.keys())
    cand_names     = list(candidates.keys())

    # ── Config ───────────────────────────────────────────────────────────────
    config = PortfolioConfig(
        min_strategies=len(base_names),
        max_strategies=max_total,
        **{k: v for k, v in config_kw.items() if k != "max_strategies"},
    )

    max_additions = config.max_strategies - len(base_names)
    if max_additions < 1:
        print(f"  ERROR: base portfolio already has {len(base_names)} strategies "
              f"which meets or exceeds the max total of {config.max_strategies}.\n"
              f"  Use --max N with N > {len(base_names)}.\n")
        sys.exit(1)

    # ── Build universe matrices ───────────────────────────────────────────────
    print("  Building universe matrices...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        univ = build_universe(
            all_strategies,
            same_asset_window_hours=config.same_asset_window_hours,
            verbose=True,
        )

    # ── Build monthly P&L cache (for rolling filter) ──────────────────────────
    monthly_cache: dict = {
        name: _monthly_pnl(df) for name, df in all_strategies.items()
    }

    # ── Build rolling correlation cache ──────────────────────────────────────
    rolling_cache = _RollingCache()
    rolling_cache.build(all_names, monthly_cache, config, verbose=True)

    # ── Filter candidates against base portfolio ──────────────────────────────
    print("  Filtering candidates against base portfolio...")
    dummy = FilterStats()
    compatible: list[str] = []

    for cand in tqdm(cand_names, desc="  Checking", unit="cand"):
        ok = True
        for base in base_names:
            pair = (base, cand)
            if not _apply_static_filters(
                pair, all_strategies, config, monthly_cache, dummy, universe=univ
            ):
                ok = False
                break
            if config.enable_rolling_filter and not rolling_cache.passes(base, cand):
                ok = False
                break
        if ok:
            compatible.append(cand)

    print(f"\n  {len(compatible)}/{len(cand_names)} candidates compatible with the base portfolio.")

    if not compatible:
        print("  No candidates passed all filters against the base portfolio.\n"
              "  Try relaxing thresholds (e.g. --no-rolling).\n")
        sys.exit(1)

    # ── Build adjacency among compatible candidates ───────────────────────────
    adj: dict[str, set[str]] = {c: set() for c in compatible}
    for i, a in enumerate(compatible):
        for b in compatible[i + 1:]:
            pair = (a, b)
            if not _apply_static_filters(
                pair, all_strategies, config, monthly_cache, dummy, universe=univ
            ):
                continue
            if config.enable_rolling_filter and not rolling_cache.passes(a, b):
                continue
            adj[a].add(b)
            adj[b].add(a)

    # ── Exhaustive search over all valid addition subsets ─────────────────────
    base_combo  = tuple(sorted(base_names))
    valid_combos: list[tuple[tuple, object]] = []

    total_subsets = sum(
        len(list(iter_combinations(compatible, size)))
        for size in range(1, max_additions + 1)
    )
    print(f"\n  Searching {total_subsets:,} addition subsets "
          f"(size 1..{max_additions}) from {len(compatible)} candidates...")

    with tqdm(total=total_subsets, desc="  Evaluating", unit="subset") as bar:
        for size in range(1, max_additions + 1):
            for additions in iter_combinations(compatible, size):
                bar.update(1)

                # All pairs within the addition set must be mutually compatible
                additions_list = list(additions)
                if not all(
                    additions_list[j] in adj[additions_list[i]]
                    for i in range(len(additions_list))
                    for j in range(i + 1, len(additions_list))
                ):
                    continue

                full_combo = tuple(sorted(base_combo + additions))
                weights_eq = equal_weight(list(full_combo))
                portfolio  = build_weighted_portfolio(full_combo, all_strategies, weights_eq)
                scaled     = rescale(full_combo, "equal", weights_eq, portfolio, config)
                vp         = validate(scaled, config)
                valid_combos.append((full_combo, vp))

    print(f"\n  {len(valid_combos)} valid expanded portfolios found.")

    if not valid_combos:
        print("  No valid expansions found.\n")
        sys.exit(1)

    # ── Score all valid combos ────────────────────────────────────────────────
    print("  Scoring...")
    scored = score_combinations(valid_combos, config)

    # ── Print top results ─────────────────────────────────────────────────────
    top_n = min(config.top_combinations, len(scored))
    print(f"\n  Top {top_n} expanded portfolios:\n")
    print(f"  {'Rank':>4}  {'Score':>6}  {'R/DD':>6}  {'Ret%':>6}  {'Win%':>5}  {'#':>2}  Additions")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*2}  ---------")

    for rank, (combo, vp, score) in enumerate(scored[:top_n], 1):
        additions_in_result = [s for s in combo if s not in base_combo]
        m   = vp.metrics
        rdd = m.get("return_dd_ratio", 0.0)
        ret = m.get("yearly_avg_pct_return", 0.0)
        win = m.get("winning_months_pct", 0.0)
        add_str = ", ".join(additions_in_result) if additions_in_result else "(none)"
        print(f"  {rank:>4}  {score:>6.4f}  {rdd:>6.2f}  {ret:>5.1f}%  {win:>4.0%}  "
              f"{len(combo):>2}  {add_str}")

    # ── Build CombinationResults for top-N ───────────────────────────────────
    print(f"\n  Building all weighting methods for top {top_n}...")
    results: list[CombinationResult] = []

    for rank, (combo, eq_vp, score) in enumerate(
        tqdm(scored[:top_n], desc="  Weighting", unit="combo"), 1
    ):
        portfolios = {"equal": eq_vp}
        weights_all = compute_all_weights(combo, all_strategies)

        for method in ("min_variance", "risk_parity", "hrp"):
            w  = weights_all[method]
            p  = build_weighted_portfolio(combo, all_strategies, w)
            s  = rescale(combo, method, w, p, config)
            portfolios[method] = validate(s, config)

        results.append(CombinationResult(
            combination   = combo,
            rank          = rank,
            raw_return_dd = eq_vp.metrics.get("return_dd_ratio", 0.0),
            raw_metrics   = eq_vp.metrics,
            portfolios    = portfolios,
            fitness_score = score,
        ))

    # ── Stress test ───────────────────────────────────────────────────────────
    print("\n  Running MAE stress test...")
    stress_test_all(results, config, verbose=True)

    # ── Walk-forward equity ───────────────────────────────────────────────────
    print("  Computing walk-forward equity...")
    compute_all_wf_equities(results, all_strategies, config, verbose=True)

    # ── Open dashboards ───────────────────────────────────────────────────────
    print(f"  Opening dashboards ({len(results)} combination(s))...")
    run_portfolio_dashboard(results, config, strategies=all_strategies, _show=False)
    run_correlation_dashboard(results, config, all_strategies, _show=False)
    run_mc_dashboard(results, config, _show=False)
    plt.show()


if __name__ == "__main__":
    main()
